"""Independent integrity, split, schema, geometry and loader checks."""

import argparse
import json
from pathlib import Path
import numpy as np
import h5py
from .store import Store, json_write, sha256
from .episodes import read_episode, StateDataset
from .robot import in_runtime, dispatch


def run(store, sources=None, limit=0):
    checks = []

    def check(name, condition, details=None):
        checks.append({"check": name, "pass": bool(condition), "details": details})

    seen = {}
    verified = 0
    failures = []
    # Read every acquired byte once per inode. Both mirror hardlinks and blobs
    # point to the same inode; no synthetic full-source checksum is asserted for
    # files obtained by selective extraction.
    for row in store.rows("files", "status='downloaded'"):
        if sources and row["source_id"] not in sources:
            continue
        p = store.root / "data/raw" / row["source_id"] / row["path"]
        if not p.exists():
            failures.append(
                {"source": row["source_id"], "path": row["path"], "error": "missing"}
            )
            continue
        key = (p.stat().st_dev, p.stat().st_ino)
        if key not in seen:
            seen[key] = sha256(p)
        if p.stat().st_size != row["bytes"] or seen[key] != row["sha256"]:
            failures.append(
                {
                    "source": row["source_id"],
                    "path": row["path"],
                    "error": "size/hash mismatch",
                }
            )
        verified += 1
    check(
        "all_downloaded_payloads_size_and_sha256",
        not failures,
        {"files_checked": verified, "distinct_inodes": len(seen), "failures": failures},
    )
    groups = {}
    content_groups = {}
    episodes = store.rows("episodes")
    nobjects = 0
    valid_episodes = 0
    for row in episodes:
        if sources and row["source_id"] not in sources:
            continue
        p = store.root / row["path"]
        a, m = read_episode(p)
        problems = []
        t = a["time_s"]
        if not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
            problems.append("non-monotonic/nonfinite timestamps")
        for key, value in a.items():
            if key.endswith(("pose", "_pose")) and value.shape[-1:] == (7,):
                finite = np.isfinite(value).all(axis=-1)
                if not finite.all():
                    mask = a.get(key.rsplit("/", 1)[0] + "/valid")
                    if mask is None or not np.array_equal(mask, finite):
                        problems.append(key + ": missing validity mask")
                if (
                    finite.any()
                    and np.max(np.abs(np.linalg.norm(value[finite, 3:], axis=-1) - 1))
                    > 1e-5
                ):
                    problems.append(key + ": quaternion not unit wxyz")
        for obj in m.get("objects", {}).values():
            assets = obj.get("meshes", []) + ([obj["mesh"]] if obj.get("mesh") else [])
            for asset in assets:
                if not (store.root / asset).is_file():
                    problems.append("missing linked mesh " + asset)
        j = json.loads(p.with_suffix(".json").read_text())
        if sha256(p) != j.get("hdf5_sha256"):
            problems.append("normalized HDF5 hash mismatch")
        groups.setdefault(m["source_group"], set()).add(row["split"])
        content_groups.setdefault(m["content_fingerprint"], set()).add(row["split"])
        check("episode:" + row["id"], not problems, problems)
        valid_episodes += not problems
        nobjects += len(m.get("objects", {}))
    check(
        "source_group_split_leakage",
        all(len(v) == 1 for v in groups.values()),
        {"source_groups": len(groups)},
    )
    check(
        "exact_normalized_duplicate_split_leakage",
        all(len(v) == 1 for v in content_groups.values()),
        {"distinct_content_fingerprints": len(content_groups)},
    )
    if episodes:
        paths = [
            store.root / x["path"]
            for x in episodes
            if x["source_id"] == "hf__robomimic__robomimic_datasets"
        ]
        loader = StateDataset(
            paths,
            [
                "hand/right_pose",
                "objects/Can/pose",
                "source/action",
                "contact/observed",
            ],
            window=16,
        )
        batches = [loader[i] for i in range(min(2, len(loader)))]
        check(
            "manipulation_batch_loading",
            len(batches) == 2
            and all(
                x["data"]["objects/Can/pose"].shape == (16, 7)
                and not x["available"]["contact/observed"]
                for x in batches
            ),
            {"batch_size": len(batches), "unobserved_contact_mask": False},
        )
    json_write(
        store.root / "runs/validation.json",
        {
            "checks": checks,
            "all_checks_pass": all(c["pass"] for c in checks),
            "scope": "acquired-file integrity, representative canonical episodes and source splits; retarget rejects remain rejects",
        },
    )
    # Existing project's independent kinematic audit runs in its pinned runtime.
    dispatch("reachy_retarget.validate", ["--root", str(store.root), "--tracking-only"])
    physics_records(store)
    return checks


def tracking(store):
    from rl_tracking.audit_data import audit

    reports = []
    for path in sorted((store.root / "data/retargeted").rglob("tracking.npz")):
        try:
            independent = audit([path])
            reports.append(
                {
                    "path": str(path.relative_to(store.root)),
                    "pass": True,
                    "existing_reachy_control_audit": independent,
                }
            )
        except Exception as e:
            reports.append(
                {
                    "path": str(path.relative_to(store.root)),
                    "pass": False,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
    json_write(
        store.root / "runs/tracking-validation.json",
        {
            "all_checks_pass": bool(reports) and all(x["pass"] for x in reports),
            "files": reports,
        },
    )
    print(
        "Existing rl_tracking audit:",
        sum(r["pass"] for r in reports),
        "/",
        len(reports),
        flush=True,
    )
    return reports


def physics_records(store):
    """Independently inspect persisted object motion/contact, not hand targets."""
    from scipy.spatial.transform import Rotation

    reports = []
    for result in sorted(
        (store.root / "runs/contact/can_fixed10_final").glob("demo_*/result.json")
    ):
        meta = json.loads(result.read_text())
        with h5py.File(result.with_name("replay.h5"), "r") as f:
            pose = f["simulation/Can_pose"][()]
            time = f["time_s"][()]
            flags = f["simulation/contacts_hand_force_open"][()]
            grip = f["command/gripper_position"][()]
            actual = f["simulation/joint_position"][()]
            if len(time) < 2:
                continue
            dt = np.diff(time)
            linear = np.linalg.norm(np.diff(pose[:, :3], axis=0) / dt[:, None], axis=1)
            rotations = Rotation.from_quat(pose[:, [4, 5, 6, 3]])
            angular = (rotations[1:] * rotations[:-1].inv()).magnitude() / dt
            placement = np.asarray(meta["conversion"]["world_placement"])
            local = (pose[:, :3] - placement[:3, 3]) @ placement[:3, :3]
            inside = (
                (abs(local[:, 0] - 0.1) < 0.175)
                & (abs(local[:, 1] - 0.28) < 0.225)
                & (local[:, 2] > 0.84)
                & (local[:, 2] < 0.96)
            )
            stable = (
                inside[1:]
                & flags[1:, 2].astype(bool)
                & ~flags[1:, 0].astype(bool)
                & (linear < 0.02)
                & (angular < 0.2)
            )
            duration = 0.0
            for ok, step in zip(stable[::-1], dt[::-1]):
                if not ok:
                    break
                duration += step
            lift = float(pose[:, 2].max() - pose[0, 2])
            grasp = bool(np.any(flags[:, 0].astype(bool) & (grip < 0)))
            task = bool(lift >= 0.08 and duration >= 1.0 and grasp)
            independent = {
                "lift_m_from_first_record": lift,
                "final_stable_s_finite_difference": duration,
                "grasp_contact": grasp,
                "task_success": task,
                "finite_state": bool(
                    np.isfinite(pose).all() and np.isfinite(actual).all()
                ),
                "command_actual_target_separated": all(
                    k in f
                    for k in (
                        "command/joint_position",
                        "target/hand_pose_world",
                        "simulation/joint_position",
                        "goal/Can_final_source_pose_world",
                    )
                ),
            }
        # A reported pass must be corroborated from persisted physical state.
        passes = (
            (not meta["success"] or task)
            and independent["finite_state"]
            and independent["command_actual_target_separated"]
        )
        reports.append(
            {
                "sequence": meta["source_sequence"],
                "record_validation_pass": passes,
                "reported_physical_pass": meta["success"],
                "independent": independent,
            }
        )
    json_write(
        store.root / "runs/physics-record-validation.json",
        {
            "all_checks_pass": len(reports) == 10
            and all(x["record_validation_pass"] for x in reports),
            "trials": reports,
        },
    )
    return reports


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--tracking-only", action="store_true")
    a = p.parse_args()
    if a.tracking_only:
        tracking(Store(a.root))
    else:
        run(Store(a.root))
