"""Auditable scope and outcome reports; no advertised totals counted as acquired."""

import collections
import gzip
import json
from pathlib import Path
import shutil
from .store import json_write, now
from .discovery import SEEDS

PAYLOAD = (
    ".parquet",
    ".hdf5",
    ".h5",
    ".pkl",
    ".npy",
    ".npz",
    ".bvh",
    ".amc",
    ".asf",
    ".glb",
    ".fbx",
    ".pt",
    ".tar",
    ".zip",
    ".gz",
)


def read(path, fallback):
    return json.loads(path.read_text()) if path.exists() else fallback


def family(sid):
    low = sid.lower()
    if "apple_storage_2" in low:
        return "reachy_apple_storage_2"
    if any(
        w in low
        for w in ("simheo__reachy2_pick_place", "glannuzel__reachy2_pick_place")
    ):
        return "reachy2_pick_place_collection"
    for name, patterns in [
        ("droid", ["droid"]),
        ("bridge", ["bridge"]),
        ("libero", ["libero"]),
        ("metaworld", ["metaworld"]),
        ("robocasa", ["robocasa"]),
        ("fastumi", ["fastumi"]),
        ("furniturebench", ["furniture_bench"]),
        ("hoi4d", ["hoi4d"]),
        ("hot3d", ["hot3d"]),
        ("dexycb", ["dexycb"]),
        ("ho3d", ["ho3d"]),
        ("oakink2", ["oakink-v2"]),
        ("oakink", ["oakink-v1"]),
        ("h2o", ["h2o-annotations"]),
        ("mimicgen", ["mimicgen_datasets"]),
        ("maniskill", ["maniskill_demonstrations"]),
        ("robomimic", ["robomimic_datasets"]),
        ("hoi_retarget", ["hoi-retarget"]),
        ("umi", ["umi_cup"]),
    ]:
        if any(p in low for p in patterns):
            return name
    return sid


def run(store, sources=None, limit=0):
    source_rows = store.rows("sources")
    byid = {r["id"]: r for r in source_rows}

    def corpus(sid):
        known = family(sid)
        if known != sid:
            return known
        parent = json.loads(byid.get(sid, {}).get("details", "{}")).get("parent")
        return parent if parent and parent.startswith("oxe__") else sid

    grouped = collections.defaultdict(list)
    for row in store.rows("files"):
        grouped[row["source_id"]].append(row)
    episodes = store.rows("episodes")
    retargets = [
        read(p, {})
        for p in sorted((store.root / "data/retargeted").rglob("validation.json"))
    ]
    physics = read(store.root / "runs/contact/can_fixed10_final/summary.json", [])
    inv = read(store.root / "catalog/inventory_summary.json", {})
    object_inventory = read(store.root / "catalog/object_inventory.json", {})
    sequence_counts = collections.Counter()
    durations = collections.Counter()
    fingerprints = {}
    sequence_file = store.root / "catalog/sequence_inventory.jsonl"
    if sequence_file.exists():
        with sequence_file.open() as f:
            for line in f:
                r = json.loads(line)
                if not r.get("duplicate_of"):
                    sequence_counts[r["source_id"]] += 1
                    durations[r["source_id"]] += r.get("duration_s") or 0
                    fingerprints[r["state_content_sha256"]] = r
    # Select one representation for known overlapping releases. This conservative
    # count does not add alternate encodings, subsets, humanoid retargets or crops.
    representations = collections.defaultdict(list)
    for sid, count in sequence_counts.items():
        representations[corpus(sid)].append(sid)
    primary = {
        fam: max(ids, key=lambda sid: sequence_counts[sid])
        for fam, ids in representations.items()
    }
    unique_lower = sum(sequence_counts[sid] for sid in primary.values())
    hours_lower = sum(durations[sid] for sid in primary.values()) / 3600
    catalog = []
    unique_payload = {}
    linked_objects = set()
    canonical_formats = collections.Counter()
    for ep in episodes:
        m = read((store.root / ep["path"]).with_suffix(".json"), {})
        canonical_formats[m.get("source_format", "unknown")] += 1
        for oid, obj in m.get("objects", {}).items():
            linked_objects.add((ep["source_id"], obj.get("parent_object", oid)))
    for row in source_rows:
        sid = row["id"]
        detail = json.loads(row["details"])
        files = grouped[sid]
        status_counts = collections.Counter(f["status"] for f in files)
        done = [f for f in files if f["status"] == "downloaded"]
        payload = [f for f in done if f["path"].lower().endswith(PAYLOAD)]
        unresolved = [
            f
            for f in files
            if f["status"]
            in ("queued", "failed", "downloading", "retry_wait", "storage_wait")
        ]
        for f in done:
            if f["sha256"]:
                unique_payload[f["sha256"]] = f["bytes"]
        if payload:
            status = (
                "partial_acquired"
                if unresolved
                or any(
                    x in str(detail.get("coverage", "")).lower()
                    for x in ("subset", "public release only", "partial")
                )
                else "acquired_selected_scope"
            )
            reason = detail.get(
                "coverage",
                "Selected 3D state/geometry files acquired; raw corpus completeness is not implied.",
            )
        elif done:
            status = "metadata_only"
            reason = "Metadata/assets acquired; no verified sequence payload counted."
        elif row["status"] in (
            "access_required",
            "excluded",
            "out_of_scope",
            "no_selected_state_files",
        ):
            status = (
                "access_required"
                if row["status"] == "access_required"
                else "unsuitable"
            )
            reason = (
                detail.get("reason")
                or detail.get("restriction")
                or detail.get("decision")
                or "No eligible state payload selected from this release."
            )
        else:
            status = "not_acquired"
            reason = (
                detail.get("error")
                or detail.get("coverage")
                or "Official route inspected; no automatically acquired state payload at this snapshot."
            )
        if unresolved and not payload:
            reason = (
                next((f["error"] for f in unresolved if f.get("error")), None)
                or "Public payloads queued; acquisition not completed."
            )
        schema = []
        infos = [
            f
            for f in done
            if f["path"].endswith(("meta/info.json", "meta_data/info.json"))
        ]
        for info in infos[:8]:
            data = read(store.root / "data/raw" / sid / info["path"], {})
            channels = {
                k: v
                for k, v in data.get("features", {}).items()
                if not any(w in k for w in ("image", "video", "depth"))
            }
            schema.append(
                {
                    "path": info["path"],
                    "version": data.get("codebase_version"),
                    "fps": data.get("fps"),
                    "robot_type": data.get("robot_type"),
                    "features": channels,
                }
            )
        record = {
            "id": sid,
            "category": row["category"],
            "url": row["url"],
            "status": status,
            "reason": reason,
            "revision": detail.get(
                "revision", "unversioned official URL; local payload hashes recorded"
            ),
            "license": detail.get("license")
            or "not verified; inspect upstream license before use",
            "downloaded_files": len(done),
            "downloaded_payload_files": len(payload),
            "downloaded_bytes": sum(f["bytes"] for f in done),
            "file_status_counts": dict(status_counts),
            "observed_sequences": sequence_counts[sid],
            "known_duration_s": durations[sid],
            "canonical_episodes": sum(e["source_id"] == sid for e in episodes),
            "retarget_pass": sum(
                x.get("source_id") == sid and x.get("status") == "kinematic_pass"
                for x in retargets
            ),
            "source_schemas": schema,
            "demonstration_role": "evaluation/test rollouts; not assumed expert demonstrations"
            if any(w in sid.lower() for w in ("eval_", "_test", "__test"))
            else "source demonstrations; success/quality not inferred from availability",
            "objects": "See linked canonical objects and source annotations; absent object poses/contact are not inferred.",
            "conversion_readiness": "representative adapter exercised"
            if any(e["source_id"] == sid for e in episodes)
            else "raw schema retained; source-specific adapter not yet validated",
            "details": detail,
        }
        catalog.append(record)
    indexed = {x["id"]: x for x in catalog}
    alias = {
        "aloha": lambda x: "aloha_static" in x or "aloha_sim" in x,
        "mobile_aloha": lambda x: "aloha_mobile" in x,
        "roboturk": lambda x: "roboturk" in x,
        "umi": lambda x: "umi" in x.lower() and "fastumi" not in x.lower(),
        "oxe": lambda x: x.startswith("oxe__") or x.startswith("hf__lerobot__"),
    }
    seed_report = []
    for sid, category, url in SEEDS:
        related = [
            x
            for x in catalog
            if x["id"] == sid
            or x["details"].get("parent") == sid
            or family(x["id"]) == sid
            or (sid in alias and alias[sid](x["id"]))
        ]
        states = [x for x in related if x["downloaded_payload_files"]]
        base = indexed.get(sid, {})
        status = "partial_acquired" if states else base.get("status", "not_acquired")
        reason = base.get("reason", "Official source inspected.")
        if states:
            reason = (
                "State/geometry acquired from: "
                + ", ".join(x["id"] for x in states)
                + ". Per-release scope and pending files are explicit below."
            )
        seed_report.append(
            {
                "id": sid,
                "url": byid.get(sid, {}).get("url", url),
                "status": status,
                "reason": reason,
                "releases": [x["id"] for x in related],
            }
        )
    tracking = read(store.root / "runs/tracking-validation.json", {})
    validation = read(store.root / "runs/validation.json", {})
    summary = {
        "generated_at": now(),
        "candidate_releases": len(catalog),
        "seed_families": len(SEEDS),
        "releases_with_acquired_payloads": sum(
            bool(x["downloaded_payload_files"]) for x in catalog
        ),
        "corpus_families_with_acquired_payloads": len(
            {corpus(x["id"]) for x in catalog if x["downloaded_payload_files"]}
        ),
        "distinct_state_content_fingerprints": inv.get("unique_state_fingerprints"),
        "conservative_sequences_known_overlap_removed": unique_lower,
        "known_hours_known_overlap_removed": hours_lower,
        "representations_counted": primary,
        "count_caveat": "Conservative observed count: one representation per known overlapping family; unparsed corpora omitted. Unknown semantic overlap may remain, so this is not a proven global unique-sequence total.",
        "canonical_episodes": len(episodes),
        "canonical_object_identities": len(linked_objects),
        "source_scoped_objects_with_verified_pose_mesh_links": object_inventory.get(
            "source_scoped_objects_with_pose_links"
        ),
        "all_raw_inventory_object_geometry_available": object_inventory.get(
            "all_linked_geometry_available"
        ),
        "object_count_scope": "Distinct (source, parent object ID) in representative canonical episodes only; raw geometry archive counts are not equated with objects.",
        "unique_acquired_payload_bytes": sum(unique_payload.values()),
        "free_disk_bytes": shutil.disk_usage(store.root).free,
        "retarget_kinematic_pass": sum(
            x.get("status") == "kinematic_pass" for x in retargets
        ),
        "retarget_attempts": len(retargets),
        "tracking_audit_pass": sum(x["pass"] for x in tracking.get("files", [])),
        "contact_task_pass": sum(x["contact_task_success"] for x in physics),
        "physical_all_checks_pass": sum(x["success"] for x in physics),
        "physical_trials": len(physics),
        "representative_contact_validation_complete": len(physics) == 10
        and any(x["success"] for x in physics),
        "file_status_counts": dict(
            collections.Counter(f["status"] for rows in grouped.values() for f in rows)
        ),
        "integrity_and_schema_checks_pass": validation.get("all_checks_pass"),
    }
    integrity = next(
        (
            x
            for x in validation.get("checks", [])
            if x.get("check") == "all_downloaded_payloads_size_and_sha256"
        ),
        {},
    )
    summary["integrity_verified_files"] = integrity.get("details", {}).get(
        "files_checked", 0
    )
    summary["integrity_snapshot_covers_current_download_count"] = summary[
        "integrity_verified_files"
    ] == summary["file_status_counts"].get("downloaded", 0)
    summary["pending_acquisition_files"] = sum(
        summary["file_status_counts"].get(k, 0)
        for k in ("queued", "downloading", "retry_wait", "storage_wait")
    )
    summary["collection_queue_drained"] = summary["pending_acquisition_files"] == 0
    summary["representative_format_counts"] = dict(canonical_formats)
    summary["representative_format_coverage_pass"] = all(
        canonical_formats[k] >= 5
        for k in ("v2.1", "v3.0", "robomimic-hdf5", "ParaHome-numpy", "glTF-2.0-HUMOTO")
    )
    summary["all_plan_completion_criteria_met"] = (
        summary["collection_queue_drained"]
        and summary["representative_contact_validation_complete"]
        and summary["integrity_and_schema_checks_pass"]
        and summary["representative_format_coverage_pass"]
        and tracking.get("all_checks_pass", False)
        and summary["integrity_snapshot_covers_current_download_count"]
    )
    summary["collector"] = read(
        store.root / "runs/collector-status.json", {"phase": "not_started"}
    )
    json_write(store.root / "catalog/datasets.json", catalog)
    json_write(store.root / "catalog/seed_status.json", seed_report)
    json_write(store.root / "runs/summary.json", summary)
    from .manifest import export_lock

    export_lock(store)
    lines = [
        "# Reachy retarget collection report",
        "",
        f"Snapshot: {summary['generated_at']}",
        "",
        f"**Acquisition is {'finished' if summary['collection_queue_drained'] else 'still incomplete'}:** {summary['pending_acquisition_files']:,} files remain in the durable queue. Provider cooldowns, live workers and individual failures are separate from completed conversions. See `runs/collector-status.json` and `runs/collector.log`.",
        "",
        "This report separates acquired source states, canonical episodes, kinematic conversions, and actual contact simulation. Counts are measured from local payloads, not advertised repository totals.",
        "",
        "| Measure | Observed result |",
        "|---|---:|",
        f"| Releases with payloads (not independent corpora) | {summary['releases_with_acquired_payloads']} |",
        f"| Corpus families with acquired payloads | {summary['corpus_families_with_acquired_payloads']} |",
        f"| Parsed sequences after known family overlap suppression | {unique_lower:,} |",
        f"| Known hours in those sequences | {hours_lower:,.2f} |",
        f"| Unique acquired content bytes | {summary['unique_acquired_payload_bytes'] / 1e9:.3f} GB |",
        f"| Canonical representative episodes | {len(episodes)} |",
        f"| Linked canonical object identities | {len(linked_objects)} |",
        f"| Source-scoped objects with verified pose/mesh links | {summary['source_scoped_objects_with_verified_pose_mesh_links']} |",
        f"| Kinematic passes / representative attempts | {summary['retarget_kinematic_pass']} / {len(retargets)} |",
        f"| Existing tracking audit passes | {summary['tracking_audit_pass']} |",
        f"| Actual contact task successes | {summary['contact_task_pass']} / {len(physics)} |",
        f"| Contact successes passing every physical check | {summary['physical_all_checks_pass']} / {len(physics)} |",
        "",
        "**Counting limits:** Raw annotation archives not yet parsed are excluded from sequence/hour counts. Reformat variants and known subsets are not added as independent demonstrations. Canonical object IDs and verified raw pose/mesh links are separate counts; source-scoped IDs are not asserted to identify distinct physical objects across corpora. See `catalog/object_inventory.json`. The exact sequence inventory, representation choices and exclusions are in `runs/summary.json` and `catalog/sequence_inventory.jsonl`.",
        "",
        "## Representative conversions",
        "",
        "| Source | Canonical | Kinematic pass / attempted |",
        "|---|---:|---:|",
    ]
    for sid in sorted({e["source_id"] for e in episodes}):
        attempts = [r for r in retargets if r.get("source_id") == sid]
        lines.append(
            f"| {sid} | {sum(e['source_id'] == sid for e in episodes)} | {sum(r.get('status') == 'kinematic_pass' for r in attempts)} / {len(attempts)} |"
        )
    lines += [
        "",
        "Rejected motion HDF5 files remain available with numerical reasons; only passing arm/base trajectories receive training NPZ files. Neck channels and their residuals are separate because the existing tracking environment locks the neck. Missing native gripper/object observations remain missing.",
        "",
        "## Fixed Can contact benchmark",
        "",
        "| Source sequence | Task | All checks | Lift (m) | Final stable open/no-contact (s) | Failure reasons |",
        "|---|---|---|---:|---:|---|",
    ]
    for r in physics:
        lines.append(
            f"| {r['source_sequence']} | {'pass' if r['contact_task_success'] else 'fail'} | {'pass' if r['success'] else 'fail'} | {r['max_lift_m']:.4f} | {r['final_stable_placement_s']:.2f} | {', '.join(r['failure_reasons']) or 'none'} |"
        )
    lines += [
        "",
        "The Can stays a dynamic free body under gravity. Source mass, inertia and friction are retained. Replay uses 500 Hz physics and the pinned native controller at 100 Hz, transforming world targets into the measured base frame. A global time scale of 10 and a constant grasp attachment depth of 0.035 m are recorded morphology adjustments. No object weld, runtime pose assignment or target force is used. Videos and actual/command/goal state are in `runs/contact/can_fixed10_final/`.",
        "",
        "**Simulation limits:** ideal planar base servos, robot-only gravity compensation, convex robot collision meshes and assumed robot/environment friction. Calibration used the same ten development demonstrations; these are reproducible feasibility tests, not a held-out generalization estimate. Four trials complete the contact task; stricter joint-speed/collision checks determine the final pass count.",
        "",
        "## Initial candidate coverage",
        "",
        "| Candidate | State | Evidence / scope |",
        "|---|---|---|",
    ]
    for r in seed_report:
        lines.append(
            f"| [{r['id']}]({r['url']}) | {r['status']} | {r['reason'].replace('|', '/')} |"
        )
    lines += [
        "",
        "## Every discovered release",
        "",
        "Full versions, licenses, feature names, file outcomes and reasons are in [datasets.json](catalog/datasets.json). The durable queue is `catalog/ledger.sqlite`.",
        "",
        "| Release | State | Files | GB | Parsed sequences | License |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in catalog:
        lic = (
            ", ".join(r["license"]) if isinstance(r["license"], list) else r["license"]
        )
        lines.append(
            f"| [{r['id']}]({r['url']}) | {r['status']} | {r['downloaded_files']} | {r['downloaded_bytes'] / 1e9:.3f} | {r['observed_sequences']} | {str(lic).replace('|', '/')} |"
        )
    lines += [
        "",
        "## Reproduce",
        "",
        "See [README.md](README.md) for commands, environment pinning, schema, sampling/split policy, loader examples and acquisition restart. `validate` checks the full acquired ledger, representative schemas, checksums and existing tracking compatibility. Source-specific coordinate conventions are recorded only when verified; unverified raw schemas are not silently mapped into Reachy coordinates.",
        "",
    ]
    (store.root / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2), flush=True)
    return summary
