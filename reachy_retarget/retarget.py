"""Kinematic retargeting with failure records and tracking-contract exports."""

import argparse
import json
from pathlib import Path
import numpy as np
import h5py
from scipy.spatial.transform import Rotation
from .episodes import read_episode, matrices_to_pose, pose_to_matrices, interpolate_pose
from .store import Store, json_write, sha256
from .robot import Robot, ARMS, NECK, in_runtime, dispatch


def neck_targets(a, placement, r, q):
    """Retain head rotations as a separate derived, bounded neck channel."""
    head = None
    if "human/joint_names" in a:
        names = list(a["human/joint_names"])
        if "mixamorig:Head" in names:
            head = pose_to_matrices(
                a["human/joint_pose"][:, names.index("mixamorig:Head")]
            )[:, :3, :3]
    elif "human/body_orientation" in a:
        head = (
            pose_to_matrices(a["human/root_pose"])[:, :3, :3]
            @ a["human/body_orientation"][:, 6]
        )
    if head is None:
        return q, {
            "available": False,
            "mapping": "held at zero unless native neck joint observations are present",
        }
    initial = r.r.q0.copy()
    r.fk(initial)
    home = r.r.data.oMf[r.r.head].rotation.copy()
    target = placement[:3, :3] @ head @ head[0].T @ placement[:3, :3].T @ home
    last = np.zeros(3)
    errors = []
    for i, R in enumerate(target):
        q[i, r.neck_ids] = last
        for _ in range(15):
            r.fk(q[i])
            current = r.r.data.oMf[r.r.head].rotation
            e = r.pin.log3(R @ current.T)
            if np.linalg.norm(e) < 0.005:
                break
            J = r.pin.computeFrameJacobian(
                r.r.model,
                r.r.data,
                q[i],
                r.r.head,
                r.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[3:, r.neck_v]
            delta = np.linalg.solve(J.T @ J + np.eye(3) * 1e-4, J.T @ e)
            q[i, r.neck_ids] = np.clip(
                q[i, r.neck_ids] + np.clip(delta, -0.1, 0.1),
                r.r.model.lowerPositionLimit[r.neck_ids] + 0.031,
                r.r.model.upperPositionLimit[r.neck_ids] - 0.031,
            )
        last = q[i, r.neck_ids].copy()
        errors.append(float(np.linalg.norm(e)))
    return q, {
        "available": True,
        "mapping": "initial head orientation aligned to robot head; subsequent source world rotations preserved within neck limits",
        "max_orientation_residual_rad": max(errors),
        "saturated_frames": sum(x > 0.1 for x in errors),
    }


def derived_gripper(a):
    """Geometric thumb/index aperture heuristic, never a contact label."""
    widths = {}
    if "human/joint_names" in a:
        names = list(a["human/joint_names"])
        p = a["human/joint_pose"][..., :3]
        for side, label in [("left", "Left"), ("right", "Right")]:
            thumb = "mixamorig:" + label + "HandThumb4"
            index = "mixamorig:" + label + "HandIndex4"
            if thumb in names and index in names:
                widths[side] = np.linalg.norm(
                    p[:, names.index(thumb)] - p[:, names.index(index)], axis=1
                )
    elif "human/joint_position" in a:
        p = a["human/joint_position"]
        for side, offset in [("left", 23), ("right", 48)]:
            widths[side] = np.linalg.norm(p[:, offset + 24] - p[:, offset + 21], axis=1)
    return {
        s: np.clip(
            (0.554 - np.arcsin(np.clip((0.07584 - w) / 0.099994, -1, 1))) / 0.4689,
            -0.06,
            2.2,
        )
        for s, w in widths.items()
    }


def gripper_trajectories(arrays, metadata, source_times, target_times):
    """Derive finger targets without treating source commands as observed motion."""
    values = {
        side: np.interp(target_times, source_times, aperture)
        for side, aperture in derived_gripper(arrays).items()
    }
    mapping = {side: {"method": "thumb-index aperture heuristic", "resampling": "linear interpolation",
                      "observed_reachy_joint": False} for side in values}
    env = metadata.get("env_args", {})
    if metadata.get("source_format") == "robomimic-hdf5":
        from .simstate import ACTIVE_OBJECTS
        config = env.get("env_kwargs", {}).get("controller_configs", {})
        controller = config.get("body_parts", {}).get("right", {}) if env.get("env_version") == "1.5.1" else config
        action = np.asarray(arrays.get("source/action"))
        arms = 2 if env.get("env_name") == "TwoArmTransport" else 1
        if (metadata.get("robot_type") != "Panda" or env.get("env_version") not in ("1.5.1", "1.4.1")
                or env.get("env_name") not in ACTIVE_OBJECTS
                or controller.get("type") != "OSC_POSE"
                or (env.get("env_version") == "1.5.1" and controller.get("gripper", {}).get("type") != "GRIP")
                or action.shape != (len(source_times), 7*arms)
                or not np.isin(action[:, 6::7], [-1., 1.]).all()):
            raise ValueError("Unverified robomimic gripper action schema")
        # Use the dilated source clock and zero-order hold: interpolation would
        # start closing before the recorded grasp command.
        indices = np.clip(np.searchsorted(source_times, target_times, side="right") - 1, 0, len(source_times)-1)
        for arm, side in enumerate(("right", "left")[:arms]):
            column = arm*7+6
            values[side] = np.where(action[indices, column] > 0, -0.06, 2.0)
            mapping[side] = {
            "method": "source binary OSC gripper command mapped to Reachy finger targets",
            "source_channel": f"source/action[:, {column}]", "source_open": -1., "source_close": 1.,
            "open_position_rad": 2., "closed_position_rad": -0.06,
            "resampling": "zero-order hold on globally dilated source timestamps",
            "reference": "reachy_retarget/contact.py:convert_grasp",
            "observed_reachy_joint": False, "contact_validated": False,
            }
    return values, mapping


def aligned_goals(arrays, metadata, robot):
    n = len(arrays["time_s"])
    home = robot.fk(robot.q)
    goals = np.tile(home, (n, 1, 1, 1))
    present = [
        h for h, s in enumerate(("left", "right")) if "hand/" + s + "_pose" in arrays
    ]
    if not present:
        raise ValueError("Source has no documented hand poses")
    src = {
        h: pose_to_matrices(arrays["hand/" + ("left", "right")[h] + "_pose"])
        for h in present
    }
    # One shared rigid placement for the human, both hands, and every object.
    # Anatomical wrist-to-tool orientation is an explicit constant attachment.
    T = np.eye(4)
    if metadata.get("robot_type") == "Panda":
        T[:3, :3] = Rotation.from_euler("z", -90, degrees=True).as_matrix()
        T[:3, 3] = [0.60, -0.1, 0]
        if metadata.get("source_format") == "ManiSkill-HDF5":
            # PickCube's source tabletop is at world z=0. Move the entire scene,
            # including objects, into a Reachy-height workspace without scaling.
            T[2, 3] = .8
    else:
        root = pose_to_matrices(arrays["human/root_pose"])[0]
        # Face the robot's +X using the human's initial shoulder axis when known.
        if "human/joint_names" in arrays:
            names = list(arrays["human/joint_names"])
            poses = pose_to_matrices(arrays["human/joint_pose"])
            l = poses[0, names.index("mixamorig:LeftArm"), :3, 3]
            r = poses[0, names.index("mixamorig:RightArm"), :3, 3]
            left = l - r
            angle = np.arctan2(left[1], left[0])
            T[:3, :3] = Rotation.from_euler("z", np.pi / 2 - angle).as_matrix()
        elif "human/joint_position" in arrays:
            # ParaHome uses world camera coordinates: construct a gravity/body
            # frame from pelvis->head and shoulder directions, not a guessed axis.
            joints = arrays["human/joint_position"][0]
            z = joints[6] - joints[0]
            z /= np.linalg.norm(z)
            y = joints[11] - joints[7]
            y -= z * np.dot(y, z)
            y /= np.linalg.norm(y)
            x = np.cross(y, z)
            T[:3, :3] = np.stack([x, y, z])
        center = np.mean([src[h][0, :3, 3] for h in present], axis=0)
        T[:3, 3] = np.array([0.4, 0, 1.0]) - T[:3, :3] @ center
    attachments = {}
    for h in present:
        transformed = T @ src[h]
        desired = Rotation.from_euler("y", -np.pi / 2).as_matrix()
        attachment = transformed[0, :3, :3].T @ desired
        # Robot source gets the same constant tool alignment; positions remain
        # the original observed EEF path until contact-specific offset calibration.
        transformed[:, :3, :3] = transformed[:, :3, :3] @ attachment
        goals[:, h] = transformed
        attachments[("left", "right")[h]] = attachment.tolist()
    return goals, T, present, attachments


def resample_object_track(times, poses, target_times, placement, source_valid=None):
    """Transform/resample measured poses without bridging invalid observations."""
    poses = np.asarray(poses)
    available = np.isfinite(poses).all(axis=1) & (np.abs(np.linalg.norm(poses[:, 3:], axis=1)-1.) < 1e-3)
    if source_valid is not None:
        if np.shape(source_valid) != (len(times),):
            raise ValueError("Object validity mask does not match source timestamps")
        available &= np.asarray(source_valid, dtype=bool)
    output = np.full((len(target_times), 7), np.nan)
    valid = np.zeros(len(target_times), dtype=bool)
    if not available.any():
        return output, valid
    transformed = matrices_to_pose(placement @ pose_to_matrices(poses[available]))
    left = np.clip(np.searchsorted(times, target_times, side="right")-1, 0, len(times)-2)
    valid = available[left] & available[left+1] & (target_times >= times[0]) & (target_times <= times[-1])
    if available.sum() >= 2:
        output[valid] = interpolate_pose(times[available], transformed, target_times[valid])
    # An exact valid sample remains valid even if its neighbor is missing.
    nearest = np.clip(np.searchsorted(times, target_times), 0, len(times)-1)
    exact = np.isclose(target_times, times[nearest], rtol=0, atol=1e-10) & available[nearest]
    if exact.any():
        source_to_available = np.cumsum(available)-1
        output[exact] = transformed[source_to_available[nearest[exact]]]
        valid[exact] = True
    return output, valid


def one(store, row):
    a, m = read_episode(store.root / row["path"])
    from .scope import require_objects
    require_objects(a)
    r = Robot(store.root)
    t = a["time_s"]
    native = "robot/joint_position" in a
    source_t = t.copy()
    placement = np.eye(4)
    attachments = {}
    present = [0, 1]
    native_mapping = None
    if native:
        names = list(a["robot/joint_names"])
        q = np.tile(r.q, (len(t), 1))
        for i, name in enumerate(names):
            if name not in r.r.q_indices:
                raise ValueError("Unknown Reachy joint " + name)
            q[:, r.r.q_indices[name]] = a["robot/joint_position"][:, i]
        goals = np.array([r.fk(x) for x in q])
        ik_errors = []
        ids = r.r._joint_q_indices
        inside = (q[:, ids] >= r.r.model.lowerPositionLimit[ids] + 0.03) & (
            q[:, ids] <= r.r.model.upperPositionLimit[ids] - 0.03
        )
        native_mapping = {
            "method": "observed joint replay",
            "source_frames_outside_controller_margin": int(np.sum(~inside.all(axis=1))),
        }
        if not inside.all():
            # The pinned controller has narrower wrist limits than some public
            # Reachy recordings. Preserve their FK goals and solve a bounded
            # redundant arm/base configuration; never widen the controller limits.
            last = q[0].copy()
            last[ids] = np.clip(
                last[ids],
                r.r.model.lowerPositionLimit[ids] + 0.031,
                r.r.model.upperPositionLimit[ids] - 0.031,
            )
            solved = []
            for i, g in enumerate(goals):
                last, pe, re = r.ik(g, last, iterations=300 if i == 0 else 80)
                solved.append(last.copy())
                ik_errors.append([pe, re])
            q = np.asarray(solved)
            native_mapping["method"] = (
                "bounded IK of observed FK goals under pinned controller limits"
            )
    else:
        goals, placement, present, attachments = aligned_goals(a, m, r)
        # Solve at source timestamps. Preserve all source frames; do not pretend
        # interpolated 100 Hz samples are independent measurements.
        qs = []
        ik_errors = []
        last = r.q.copy()
        for i, g in enumerate(goals):
            last, pe, re = r.ik(g, last, present, iterations=150 if i == 0 else 40)
            qs.append(last.copy())
            ik_errors.append([pe, re])
        q = np.asarray(qs)
        # Undemonstrated hands follow the solved robot rather than becoming
        # fabricated stationary world-frame labels that overconstrain the base.
        absent = set((0, 1)) - set(present)
        if absent:
            solved_fk = np.array([r.fk(x) for x in q])
            for h in absent:
                goals[:, h] = solved_fk[:, h]
    q, neck_mapping = (
        neck_targets(a, placement, r, q)
        if not native
        else (
            q,
            {
                "available": any(n in list(a["robot/joint_names"]) for n in NECK),
                "mapping": "native observed radians or zero hold",
            },
        )
    )
    # Synchronized global time dilation keeps the source geometry intact.
    dq = np.array([r.pin.difference(r.r.model, x, y) for x, y in zip(q[:-1], q[1:])])
    ratios = np.abs(dq / np.diff(t)[:, None]) / r.r.caps
    factor = max(1.0, float(np.max(ratios)) * 1.05)
    # A huge factor is a discontinuity diagnostic, not a useful training motion.
    required_dilation = factor
    discontinuous = factor > 20
    if discontinuous:
        factor = 1.0  # Keep a diagnostic conversion; never export it as training data.
    t = t * factor
    target_t = np.arange(0, t[-1] + 1e-9, 0.01)
    arms = np.stack([np.interp(target_t, t, q[:, j]) for j in r.arm_ids], axis=1)
    neck = np.stack([np.interp(target_t, t, q[:, j]) for j in r.neck_ids], axis=1)
    bases = np.array([r.base(x) for x in q])
    bases[:, 2] = np.unwrap(bases[:, 2])
    bases = np.stack([np.interp(target_t, t, bases[:, j]) for j in range(3)], axis=1)
    full = np.array([r.pack(arm, base) for arm, base in zip(arms, bases)])
    actual = np.array([r.fk(x) for x in full])
    target_pose = np.stack(
        [
            interpolate_pose(t, matrices_to_pose(goals[:, h]), target_t)
            for h in range(2)
        ],
        axis=1,
    )
    gaps = []
    margins = []
    for x in full[::2]:
        gap, margin, _ = r.r.geometry(x)
        gaps.append(gap)
        margins.append(margin)
    errors = np.linalg.norm(actual[:, :, :3, 3] - target_pose[:, :, :3], axis=-1)
    target_matrix = pose_to_matrices(target_pose)
    orientation_error = Rotation.from_matrix(
        (target_matrix[:, :, :3, :3] @ actual[:, :, :3, :3].swapaxes(-1, -2)).reshape(
            -1, 3, 3
        )
    ).magnitude()
    valid = (
        not discontinuous
        and min(gaps) >= 0.01 - 1e-6
        and min(margins) >= 0.03 - 1e-6
        and float(errors.max()) <= 0.02
        and orientation_error.max() <= 0.1
    )
    out = store.root / "data/retargeted" / row["source_id"] / row["id"]
    out.mkdir(parents=True, exist_ok=True)
    grippers, gripper_mapping = gripper_trajectories(a, m, t, target_t)
    with h5py.File(out / "motion.h5", "w") as f:
        f["time_s"] = target_t
        f["robot/joint_position"] = arms
        f["robot/base_pose_xyyaw"] = bases
        f["robot/neck_position"] = neck
        f["robot/joint_names"] = np.asarray(ARMS, dtype=h5py.string_dtype())
        f["robot/neck_names"] = np.asarray(NECK, dtype=h5py.string_dtype())
        f.attrs["units"] = "m, rad, s; poses xyz+wxyz; right-handed world"
        for side, gripper in grippers.items():
            f["derived/gripper/" + side + "_position"] = gripper
        f.attrs["derived_gripper_semantics"] = json.dumps(gripper_mapping)
        f["target/hand_pose_world"] = target_pose
        f["robot/hand_pose_world"] = matrices_to_pose(actual)
        object_names = {key.split("/")[1] for key in a if key.startswith("objects/") and key.endswith("/pose")}
        for oid in sorted(object_names):
            key = f"objects/{oid}/pose"
            values, valid_object = resample_object_track(t, a[key], target_t, placement, a.get(f"objects/{oid}/valid"))
            f[key] = values
            f[f"objects/{oid}/valid"] = valid_object
        f.attrs["metadata_json"] = json.dumps(
            {
                "source_episode": row["id"],
                "split": row["split"],
                "physics_validated": False,
                "controller_backend": r.backend,
            }
        )
    report = {
        "source_episode": row["id"],
        "source_id": row["source_id"],
        "source_sequence": row["source_sequence"],
        "split": row["split"],
        "source_duration_s": float(source_t[-1]),
        "retarget_duration_s": float(target_t[-1]),
        "time_dilation": factor,
        "world_placement": placement.tolist(),
        "wrist_to_tool_rotation": attachments,
        "required_velocity_dilation": required_dilation,
        "discontinuity_rejected": discontinuous,
        "demonstrated_hands": present,
        "native_mapping": native_mapping,
        "environment_collision_scope": "Only self-clearance is checked for kinematic exports; explicit environment contact is validated separately in the Can benchmark",
        "neck_mapping": neck_mapping,
        "tracking_neck_policy": "existing tracking environment locks neck at zero; neck channel retained separately in HDF5",
        "neck_orientation_pass": not neck_mapping.get("available", False)
        or neck_mapping.get("max_orientation_residual_rad", 0) <= 0.1,
        "min_self_clearance_m": min(gaps),
        "min_joint_margin_rad": min(margins),
        "max_position_error_m": float(errors.max()),
        "max_ik_orientation_error_rad": max((x[1] for x in ik_errors), default=0),
        "max_orientation_error_rad": float(orientation_error.max()),
        "status": "kinematic_pass" if valid else "kinematic_fail",
        "physics_validated": False,
        "controller_identity": r.identity,
        "controller_backend": r.backend,
        "gripper_mapping": gripper_mapping,
        "source_hdf5_sha256": sha256(store.root / row["path"]),
        "missing_gripper_policy": "No invented command; source gripper channels retained in normalized episode",
    }
    json_write(out / "validation.json", report)
    # Export only audited examples; no rejection filtering hidden in summary.
    if valid:
        export_tracking(
            store, row, target_t, matrices_to_pose(actual), arms, bases, out
        )
    else:
        for old in (out / "tracking.npz", out / "tracking.json"):
            if old.exists():
                old.unlink()
    return report


def export_tracking(store, row, t, poses, arms, bases, out):
    # Independent 6 s windows; all windows retain the parent's source split.
    starts = list(range(0, max(1, len(t) - 1), 600))
    records = []
    initial = []
    jw = []
    bw = []
    sources = []
    for start in starts:
        idx = np.minimum(np.arange(start, start + 601), len(t) - 1)
        # The existing RL loader resets the base to the origin for each window.
        origin = np.eye(4)
        origin[:3, :3] = Rotation.from_euler("z", bases[start, 2]).as_matrix()
        origin[:2, 3] = bases[start, :2]
        inverse = np.linalg.inv(origin)
        local_pose = matrices_to_pose(inverse @ pose_to_matrices(poses[idx]))
        local_base = bases[idx].copy()
        local_base[:, :2] = (bases[idx, :2] - bases[start, :2]) @ origin[:2, :2]
        local_base[:, 2] -= bases[start, 2]
        records.append(local_pose)
        initial.append(arms[start])
        jw.append(arms[idx])
        bw.append(local_base)
        sources.append(row["id"] + f":{start}")
    path = out / "tracking.npz"
    parent = json.loads((store.root / row["path"]).with_suffix(".json").read_text())[
        "source_group"
    ]
    np.savez_compressed(
        path,
        poses=np.array(records, dtype=np.float32),
        initial=np.array(initial, dtype=np.float32),
        family=np.full(len(records), 6, dtype=np.int64),
        source=np.array(sources),
        time_s=np.arange(601) * 0.01,
        parent_source=np.array([parent] * len(records)),
        witness_joint=np.array(jw, dtype=np.float32),
        witness_base=np.array(bw, dtype=np.float32),
        motion_duration_s=np.array([min(6.0, t[-1] - t[s]) for s in starts]),
        initial_pose_mode=np.array(["retargeted"] * len(records)),
    )
    audit = (
        json.loads((out / "validation.json").read_text())
        if (out / "validation.json").exists()
        else {}
    )
    json_write(
        path.with_suffix(".json"),
        {
            "schema": "reachy-nominal-v2",
            "dt": 0.01,
            "duration_s": 6.0,
            "split": row["split"],
            "debug": False,
            "minimum_sampled_witness_clearance_m": audit.get("min_self_clearance_m"),
            "families": [
                "static",
                "left_arm",
                "right_arm",
                "both_arms",
                "neck",
                "base",
                "cooperation",
            ],
            "count": len(records),
            "independent_sources": [parent],
            "sha256": sha256(path),
            "parent_source": parent,
            "witness_role": "offline kinematic audit only; not actor/critic input",
            "family_policy": "cooperation unless independently verified stationary",
            "resampling": "100 Hz interpolation; not new measurements",
            "tail_padding": "hold final state",
        },
    )


def run(store, sources=None, limit=5):
    if not in_runtime():
        args = ["--root", str(store.root), "--limit", str(limit)]
        for source in sources or []:
            args += ["--source", source]
        return dispatch("reachy_retarget.retarget", args)
    rows = store.rows("episodes")
    counts = {}
    reports = []
    for row in rows:
        if sources and row["source_id"] not in sources:
            continue
        if limit and counts.get(row["source_id"], 0) >= limit:
            continue
        counts[row["source_id"]] = counts.get(row["source_id"], 0) + 1
        try:
            report = one(store, row)
        except Exception as e:
            report = {
                "source_episode": row["id"],
                "source_id": row["source_id"],
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
            }
        reports.append(report)
        print(
            {
                k: v
                for k, v in report.items()
                if k
                not in (
                    "controller_identity",
                    "world_placement",
                    "wrist_to_tool_rotation",
                )
            },
            flush=True,
        )
        json_write(store.root / "runs/retargeting.json", reports)
    return reports


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--source", action="append")
    p.add_argument("--limit", type=int, default=5)
    a = p.parse_args()
    run(Store(a.root), a.source, a.limit)
