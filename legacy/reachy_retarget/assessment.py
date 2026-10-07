"""Offline coverage audit of artifacts that exist in explicitly selected roots.

No Store is created: opening this module or auditing an old catalog never creates
or repairs a ledger. Historical counts and source accessibility claims are kept
apart from current local evidence. This is an artifact audit, not a simulator.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import h5py
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def rows(root, table):
    """Read only; a missing ledger represents no ledger evidence."""
    if table not in {"sources", "files", "episodes"}:
        raise ValueError(table)
    path = Path(root) / "catalog/ledger.sqlite"
    if not path.exists():
        return []
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM " + table)]


def object_channels(h5):
    """Require measured numeric pose channels, never an object/task name alone."""
    result = []
    if "objects" not in h5:
        return result
    n = len(h5["time_s"]) if "time_s" in h5 else None
    for name, group in h5["objects"].items():
        row = {"object_id": name, "valid_pose_frames": 0, "usable": False}
        result.append(row)
        if not isinstance(group, h5py.Group) or "pose" not in group:
            row["error"] = "missing_pose_channel"
            continue
        pose = np.asarray(group["pose"])
        if pose.ndim != 2 or pose.shape != (n, 7) or pose.dtype.kind not in "fiu":
            row["error"] = "expected_time_aligned_numeric_xyz_wxyz_pose"
            continue
        valid = np.isfinite(pose).all(axis=1)
        quaternion_norm = np.linalg.norm(pose[:, 3:], axis=1)
        valid &= np.isclose(quaternion_norm, 1.0, atol=1e-3)
        row["has_explicit_validity_mask"] = "valid" in group
        if "valid" in group:
            mask = np.asarray(group["valid"], dtype=bool)
            if mask.shape != (n,):
                row["error"] = "invalid_object_validity_mask_shape"
                continue
            valid &= mask
        row.update(valid_pose_frames=int(valid.sum()), total_frames=n,
                   complete=bool(valid.all()), usable=bool(valid.any()))
        if not valid.any():
            row["error"] = "no_valid_object_pose_frames"
    return result


def scenario(metadata):
    env_name = metadata.get("env_args", {}).get("env_name")
    if env_name:
        return {"name": env_name, "evidence": "source env_args.env_name"}
    for key in ("scenario", "task"):
        if isinstance(metadata.get(key), str):
            return {"name": metadata[key], "evidence": "source metadata." + key}
    return {"name": "unlabeled", "evidence": "No verified scenario label; sequence names are retained separately."}


def audit_episode(root, path):
    root, path = Path(root), Path(path)
    with h5py.File(path, "r") as f:
        metadata = json.loads(f.attrs.get("metadata_json", "{}"))
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            metadata.update(load_json(sidecar))
        objects = object_channels(f)
        hands = [side for side in ("left", "right") if "hand/" + side + "_pose" in f]
        timestamps = np.asarray(f["time_s"]) if "time_s" in f else np.array([])
    expected_hash = metadata.get("hdf5_sha256")
    hash_valid = digest(path) == expected_hash if expected_hash else None
    valid_time = (len(timestamps) >= 2 and np.isfinite(timestamps).all()
                  and bool((np.diff(timestamps) > 0).all()))
    object_metadata = metadata.get("objects", {})
    for obj in objects:
        declared = object_metadata.get(obj["object_id"], {})
        obj["object_type"] = declared.get("category", declared.get("type", "unlabeled"))
        obj["parent_object"] = declared.get("parent_object")
        obj["asset_reference"] = declared.get("mesh", declared.get("meshes", declared.get("asset")))
    eid = metadata.get("episode_id", path.stem)
    sid = metadata.get("source_id", path.parent.name)
    usable = any(obj["usable"] for obj in objects)
    row = {
        "root": str(root), "path": str(path.relative_to(root)), "episode_id": eid,
        "source_id": sid, "source_sequence": metadata.get("source_sequence"),
        "source_group": metadata.get("source_group", sid + "/" + str(metadata.get("source_sequence", eid))),
        "content_fingerprint": metadata.get("content_fingerprint"),
        "source_format": metadata.get("source_format"), "scenario": scenario(metadata),
        "objects": objects, "declared_objects_without_pose": sorted(set(object_metadata) - {o["object_id"] for o in objects}),
        "hands": hands, "frames": len(timestamps), "valid_time": bool(valid_time),
        "hdf5_checksum_valid": hash_valid, "missing_source_fields": metadata.get("missing", []),
        "source_provenance": metadata.get("provenance", []),
        "object_aware": usable, "normalized": bool(valid_time and hash_valid is not False),
        "kinematic_pass": False, "retargeted": False,
        "contact_attempts": [], "contact_validated": False,
    }
    out = root / "data/retargeted" / sid / eid
    motion, validation = out / "motion.h5", out / "validation.json"
    if validation.exists():
        report = load_json(validation)
        row["retarget_report"] = {
            "path": str(validation.relative_to(root)), "status": report.get("status"),
            "controller_backend": report.get("controller_backend"),
            "error": report.get("error"), "physics_validated_claim": report.get("physics_validated"),
            "failure_reasons": report.get("failure_reasons"),
            "max_position_error_m": report.get("max_position_error_m"),
            "max_orientation_error_rad": report.get("max_orientation_error_rad"),
            "min_self_clearance_m": report.get("min_self_clearance_m"),
            "min_joint_margin_rad": report.get("min_joint_margin_rad"),
            "discontinuity_rejected": report.get("discontinuity_rejected"),
            "time_dilation": report.get("time_dilation"),
            "neck_orientation_pass": report.get("neck_orientation_pass"),
            "neck_max_orientation_residual_rad": report.get("neck_mapping", {}).get("max_orientation_residual_rad"),
        }
        if motion.exists():
            with h5py.File(motion, "r") as f:
                preserved = object_channels(f)
            row["retargeted"] = True
            row["retargeted_objects"] = preserved
            expected_objects = {o["object_id"] for o in objects if o["usable"]}
            observed_objects = {o["object_id"] for o in preserved if o["usable"]}
            row["kinematic_pass"] = bool(
                row["normalized"] and usable and expected_objects <= observed_objects
                and report.get("status") == "kinematic_pass"
                and report.get("source_episode") == eid
                and (not report.get("source_hdf5_sha256") or report["source_hdf5_sha256"] == digest(path))
            )
    if not usable:
        row["object_pipeline_status"] = "unsupported_no_valid_object_pose"
    elif not hands:
        row["object_pipeline_status"] = "unsupported_no_hand_pose_adapter"
    else:
        row["object_pipeline_status"] = "kinematic_pass" if row["kinematic_pass"] else "normalized_not_kinematically_validated"
    return row


def audit_contact(root, path, episodes):
    report = load_json(path)
    matching = [e for e in episodes if e["episode_id"] == report.get("source_episode")
                and (not report.get("source_id") or e["source_id"] == report["source_id"])]
    episode = matching[0] if len(matching) == 1 else None
    replay, scene = path.with_name("replay.h5"), path.with_name("scene.xml")
    scene_valid = bool(scene.exists() and report.get("scene_sha256") == digest(scene))
    actual_state = False
    if replay.exists():
        with h5py.File(replay, "r") as f:
            actual_state = all(key in f for key in ("time_s", "simulation/qpos"))
            if episode:
                actual_state &= all("simulation/" + o["object_id"] + "_pose" in f
                                    for o in episode["objects"] if o["usable"])
    passed = report.get("status") == "physical_pass" and report.get("success") is True
    verified = bool(passed and episode and episode["object_aware"] and scene_valid and actual_state
                    and report.get("object_state_assignment") == "reset only; dynamic free joint for every physics step")
    row = {
        "root": str(root), "path": str(path.relative_to(root)),
        "source_id": report.get("source_id", episode["source_id"] if episode else None),
        "source_episode": report.get("source_episode"), "source_sequence": report.get("source_sequence"),
        "source_group": episode["source_group"] if episode else None,
        "reported_status": report.get("status"), "reported_pass": passed,
        "artifact_verified_pass": verified, "scene_checksum_valid": scene_valid,
        "actual_state_replay_available": bool(actual_state),
        "failure_reasons": report.get("failure_reasons", []),
        "grasp_stability_pass": report.get("grasp_stability_pass"),
        "controller_backend": report.get("controller_backend"),
        "time_dilation": report.get("conversion", {}).get("time_dilation"),
        "criteria": report.get("criteria", {}),
        "metrics": {key: report.get(key) for key in (
            "max_lift_m", "final_stable_placement_s", "max_hand_can_penetration_m",
            "bilateral_grasp_contact_observed", "actual_velocity_pass", "grasp_drift",
            "robot_self_penetration_substeps", "robot_environment_penetration_substeps")},
    }
    if episode:
        episode["contact_attempts"].append(row["path"])
        episode["contact_validated"] |= verified
    return row


def audit_root(root):
    root = Path(root).resolve()
    historical_path = root / "catalog/datasets.json"
    historical = load_json(historical_path) if historical_path.exists() else []
    sources = {row["id"]: {"source_id": row["id"], "url": row.get("url"),
               "catalog_status": row.get("status"), "catalog_scope": "historical_export",
               "historical_downloaded_files": row.get("downloaded_files", 0)} for row in historical}
    for entry in rows(root, "sources"):
        sources.setdefault(entry["id"], {"source_id": entry["id"]}).update(
            url=entry.get("url"), catalog_status=entry.get("status"), catalog_scope="local_ledger")
    files = rows(root, "files")
    for entry in files:
        sid = entry["source_id"]
        source = sources.setdefault(sid, {"source_id": sid})
        source.setdefault("file_evidence", []).append({
            "path": entry["path"], "url": entry.get("url"),
            "ledger_status": entry.get("status"), "ledger_sha256": entry.get("sha256"),
            "present": (root / "data/raw" / sid / entry["path"]).is_file(),
        })
    errors, episodes, contacts = [], [], []
    for path in sorted((root / "data/normalized").glob("*/*.h5")):
        try:
            episodes.append(audit_episode(root, path))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"path": str(path.relative_to(root)), "error": str(exc)})
    for path in sorted((root / "runs/contact").glob("*/*/result.json")):
        try:
            contacts.append(audit_contact(root, path, episodes))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"path": str(path.relative_to(root)), "error": str(exc)})
    for ep in episodes:
        sources.setdefault(ep["source_id"], {"source_id": ep["source_id"]})
        contact_supported = (ep["source_id"] == "hf__robomimic__robomimic_datasets"
                             and str(ep["source_sequence"]).startswith("v1.5/can/ph/demo_"))
        ep["contact_backend_supported"] = contact_supported
        ep["contact_status"] = ("saved_physical_pass" if ep["contact_validated"] else
                                "attempted_without_verified_pass" if ep["contact_attempts"] else
                                "not_run" if contact_supported else "unsupported_source_scene")
    for sid, source in sources.items():
        matches = [e for e in episodes if e["source_id"] == sid]
        evidence = source.get("file_evidence", [])
        source.update(
            catalogued=True, accessibility="not_rechecked_offline",
            locally_present_files=sum(f["present"] for f in evidence),
            downloaded_files=sum(f["present"] and f["ledger_status"] == "downloaded" for f in evidence),
            stale_downloaded_entries=sum(not f["present"] and f["ledger_status"] == "downloaded" for f in evidence),
            normalized_episodes=sum(e["normalized"] for e in matches),
            object_aware_episodes=sum(e["normalized"] and e["object_aware"] for e in matches),
            robot_only_or_missing_object_episodes=sum(not e["object_aware"] for e in matches),
            kinematic_pass_episodes=sum(e["kinematic_pass"] for e in matches),
            contact_validated_episodes=sum(e["contact_validated"] for e in matches),
        )
    stale_episodes = [r for r in rows(root, "episodes") if not (root / r["path"]).is_file()]
    return {"root": str(root), "ledger_present": (root / "catalog/ledger.sqlite").exists(),
            "sources": list(sources.values()), "episodes": episodes, "contact_attempts": contacts,
            "stale_episode_entries": [{k: r[k] for k in ("id", "source_id", "path")} for r in stale_episodes],
            "errors": errors}


def assess(roots):
    audits = [audit_root(root) for root in roots]
    episodes = [e for audit in audits for e in audit["episodes"]]
    # Union only explicit source groups or exact normalized fingerprints. This
    # collapses repeated runs and copies but does not claim to solve crop overlap.
    parent = list(range(len(episodes)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, ep in enumerate(episodes):
        keys = [("source_group", ep["source_group"])]
        if ep["content_fingerprint"]:
            keys.append(("content_fingerprint", ep["content_fingerprint"]))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, ep in enumerate(episodes):
        groups[find(i)].append(ep)
    summary = {
        "roots": len(audits), "catalogued_source_ids": len({s["source_id"] for a in audits for s in a["sources"]}),
        "locally_downloaded_source_ids": len({s["source_id"] for a in audits for s in a["sources"] if s["downloaded_files"]}),
        "normalized_artifact_copies": len(episodes), "distinct_recorded_source_groups": len(groups),
        "object_aware_source_groups": sum(any(e["normalized"] and e["object_aware"] for e in g) for g in groups.values()),
        "kinematic_pass_source_groups": sum(any(e["kinematic_pass"] for e in g) for g in groups.values()),
        "contact_validated_source_groups": sum(any(e["contact_validated"] for e in g) for g in groups.values()),
        "contact_attempts": sum(len(a["contact_attempts"]) for a in audits),
        "contact_reported_passes": sum(c["reported_pass"] for a in audits for c in a["contact_attempts"]),
        "contact_artifact_verified_passes": sum(c["artifact_verified_pass"] for a in audits for c in a["contact_attempts"]),
        "object_aware_dataset_ids": sorted({e["source_id"] for e in episodes if e["normalized"] and e["object_aware"]}),
        "kinematic_pass_dataset_ids": sorted({e["source_id"] for e in episodes if e["kinematic_pass"]}),
        "contact_validated_dataset_ids": sorted({e["source_id"] for e in episodes if e["contact_validated"]}),
        "source_scoped_object_ids": sorted({e["source_id"] + "/" + o["object_id"] for e in episodes for o in e["objects"] if o["usable"]}),
        "labeled_scenarios": sorted({e["scenario"]["name"] for e in episodes if e["object_aware"] and e["scenario"]["name"] != "unlabeled"}),
    }
    return {"schema": "reachy-object-coverage-v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Offline artifact inspection; no network, robot access, controller execution, or physics rerun.",
            "counting_policy": "Repeated roots and attempts collapse by source group or exact normalized fingerprint. Unknown re-timed/cropped/reformatted overlap remains unresolved; source groups are not claimed to be independent demonstrations.",
            "validation_policy": "Object-aware requires a numeric time-aligned xyz+wxyz object pose with at least one valid frame. Kinematic pass requires matching saved output and object channels. Contact pass requires the saved result, scene checksum, complete-state replay, linked object episode and declared free dynamic object policy; it is not an independent reproduction.",
            "summary": summary, "roots": audits}


def markdown(report):
    lines = ["# Local object-aware coverage audit", "", "Generated: " + report["generated_utc"], "", report["scope"], "",
             "Historical catalog exports are descriptions of a previous acquisition run. They do not establish that payloads exist on this machine. Public accessibility is not rechecked by this offline command.", "",
             "| Root | Current downloaded source IDs | Normalized artifacts | Object-aware | Kinematic pass | Contact-validated episodes |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for audit in report["roots"]:
        eps = audit["episodes"]
        lines.append("| " + audit["root"] + " | " + " | ".join(str(n) for n in (
            sum(s["downloaded_files"] > 0 for s in audit["sources"]), len(eps),
            sum(e["object_aware"] for e in eps), sum(e["kinematic_pass"] for e in eps),
            sum(e["contact_validated"] for e in eps))) + " |")
    lines += ["", "Counts of downloaded source IDs can include asset bundles; they are not counts of demonstration datasets.", "", report["counting_policy"], "",
              "## Episode evidence", "", "| Source / sequence | Object IDs with usable poses | Scenario evidence | Kinematic | Contact |", "| --- | --- | --- | --- | --- |"]
    for audit in report["roots"]:
        for ep in audit["episodes"]:
            obj = ", ".join(o["object_id"] for o in ep["objects"] if o["usable"]) or "none"
            lines.append(f"| {ep['source_id']} / {ep['source_sequence']} | {obj} | {ep['scenario']['name']} | {ep['kinematic_pass']} | {ep['contact_validated']} |")
    lines += ["", "Unlabeled object types/scenarios remain unlabeled. A task name containing an object does not qualify an episode for this object-aware pipeline.", "",
              "## Scenario coverage by run root", "",
              "| Root | Dataset | Source scenario | Cases | Kinematic passes | Contact backend |", "| --- | --- | --- | ---: | ---: | --- |"]
    for audit in report["roots"]:
        scenarios = defaultdict(list)
        for ep in audit["episodes"]:
            label = ep["scenario"]["name"]
            if label == "unlabeled":
                label = str(ep["source_sequence"]) + " (source sequence label)"
            scenarios[(ep["source_id"], label)].append(ep)
        for (sid, label), cases in sorted(scenarios.items()):
            support = "Can validator available" if all(e["contact_backend_supported"] for e in cases) else "unsupported scene"
            lines.append(f"| {Path(audit['root']).name} | {sid} | {label} | {len(cases)} | {sum(e['kinematic_pass'] for e in cases)} | {support} |")
    lines += ["", "Only the robomimic v1.5 proficient-human Can scene has a dynamic contact validator in the current implementation. All other object/scenario rows are unsupported for physical validation, including kinematically passing cube, nut, stacking, threading, tool-hanging and HUMOTO cases. They require scene/collision assets, contact and gripper adapters, and task-specific success criteria. No contact success is inferred from object-pose playback or source success labels.", "",
              "## Kinematic failures", "",
              "| Root / source sequence | Position error (mm) | Orientation error (degrees) | Minimum self-clearance (mm) | Time dilation | Neck pass |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    def metric(value, scale=1):
        return "unknown" if value is None else f"{float(value) * scale:.3f}"
    for audit in report["roots"]:
        for ep in audit["episodes"]:
            r = ep.get("retarget_report", {})
            if r.get("status") != "kinematic_fail":
                continue
            lines.append(f"| {Path(audit['root']).name} / {ep['source_sequence']} | {metric(r.get('max_position_error_m'), 1000)} | {metric(r.get('max_orientation_error_rad'), 180 / np.pi)} | {metric(r.get('min_self_clearance_m'), 1000)} | {metric(r.get('time_dilation'))} | {r.get('neck_orientation_pass')} |")
    lines += ["", "Negative self-clearance indicates overlapping collision proxies. These failures remain failed despite retiming; the underlying files and measured errors are retained.", "",
              "## Contact attempts and protocol limits", "",
              "| Root / attempt | Source sequence | Saved result | Lift (mm) | Final stable time (s) | Maximum hand/object penetration (mm) | Failure reasons |", "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    for audit in report["roots"]:
        for attempt in audit["contact_attempts"]:
            metrics = attempt["metrics"]
            path = str(Path(attempt["path"]).parent.relative_to("runs/contact"))
            lines.append(f"| {Path(audit['root']).name} / {path} | {attempt['source_sequence']} | {attempt['reported_status']} | {metric(metrics['max_lift_m'], 1000)} | {metric(metrics['final_stable_placement_s'])} | {metric(metrics['max_hand_can_penetration_m'], 1000)} | {', '.join(attempt['failure_reasons']) or 'none recorded'} |")
    lines += ["", "Each attempt retains its own criteria in the JSON. Some earlier Can development passes predate the newer grasp-drift gate and are not interchangeable with the fixed-grasp three-case matrix. Repeated Can tuning attempts and copies across preview roots are one source demonstration when their recorded source group matches. The current matrix is reported separately by root above.", "",
              "## Aggregated evidence", "", "```json", json.dumps(report["summary"], indent=2), "```", "", report["validation_policy"], "",
              "## Reproduce", "", "```bash", ".venv/bin/python -m reachy_retarget.assessment --root . --root runs/local-preview --root runs/local-preview-v2 --root runs/object-matrix --output runs/coverage/assessment.json --markdown docs/coverage-audit.md", "```", "",
              "The JSON retains per-source local file evidence, missing files, episode checksums, validity coverage, exact source groups, failed contact attempts and per-attempt criteria. No acquisition ledger or source payload is changed.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    result = assess(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(result))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
