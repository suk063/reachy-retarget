"""Export the source physics scenes (:class:`~reachy_retarget.schema.source.SceneRef`) of built episodes.

Episodes store only the hashes of the scene their tier-P rollout used (``/physics`` attr ``info``
-> ``scene``). This module regenerates that scene from the raw source file and saves it, so a
closed-loop policy rollout can run in the original physics scene (masses, friction, joints)
without the raw data.

    python -m reachy_retarget.scene_export export --out DIR [--name NAME] EPISODE.h5 [EPISODE.h5 ...]
    python -m reachy_retarget.scene_export manifest --build data/pulled/w1-v5 --pvc-build datasets/w1-v5 \\
        --family mimicgen --tier-k --tier-p --publish datasets/scenes-mimicgen-kp \\
        --out runs/manifests/export-scenes-mimicgen-kp.jsonl

``export`` re-reads each episode's source demo through its adapter (``provenance.file``,
``provenance.demo_key``; raw files are only on the PVC), rebuilds the tier-P scene
(:func:`~reachy_retarget.validate.scene.build_scene` with the episode's parked bodies) and checks it
against the episode's record: the source MJCF sha256, every resolved asset (file, sha256) and, when
the MuJoCo version equals the recorded simulator, the compiled scene sha256. A verified scene is
saved under ``DIR/scenes/<source_mjcf_sha256>/`` with only the assets left after removing the source
robot; episodes sharing a source MJCF are exported once. One JSON line per episode goes to
``DIR/records/<NAME>.jsonl`` (also for failures). Nothing here touches the network.

``manifest`` writes a :mod:`cluster.pool` job manifest from a local copy of a build: one episode per
distinct source MJCF, one job per raw source file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

EXPORT_FORMAT = "reachy-retarget-scene-export-v1"


def _episode_info(path) -> tuple[dict, dict | None]:
    import h5py

    with h5py.File(path, "r") as f:
        metadata = json.loads(f.attrs["metadata"])
        info = json.loads(f["physics"].attrs["info"]) if "physics" in f else None
    return metadata, info


def asset_keys(records) -> list[str] | None:
    """Keys of ``SceneRef.assets`` that :func:`build_scene` resolved (its ``info["assets"]``); None
    when an asset came from elsewhere (a resolver or mesh directory), i.e. the SceneRef alone is
    not enough."""
    keys = []
    for rec in records:
        how = rec["found"]
        if how == "assets:exact":
            keys.append(rec["file"])
        elif how.startswith("assets:"):
            keys.append(how.split(":", 1)[1])
        else:
            return None
    return sorted(set(keys))


def scene_checks(built_info: dict, recorded: dict, simulator: str | None) -> dict:
    """Compare a rebuilt scene's ``info`` with the episode's recorded tier-P scene info."""
    import mujoco

    def assets(info):
        return sorted((r["file"], r["sha256"]) for r in info["assets"])

    same_version = simulator == f"mujoco {mujoco.__version__}"
    checks = {"source_mjcf_sha256": built_info["source_mjcf_sha256"] == recorded["source_mjcf_sha256"],
              "assets": assets(built_info) == assets(recorded),
              "parked_bodies": built_info["removed"]["parked_bodies"] == recorded["removed"]["parked_bodies"],
              "scene_sha256": (built_info["scene_sha256"] == recorded["scene_sha256"]) if same_version else None}
    checks["passed"] = all(v is not False for v in checks.values())
    return checks


def _record(path) -> tuple[dict, dict | None]:
    """The export record of one built episode before its scene is read, and its tier-P scene info."""
    metadata, info = _episode_info(path)
    prov = metadata.get("provenance") or {}
    rec = {"format": EXPORT_FORMAT, "episode": str(path), "uid": f"{metadata['dataset']}/{metadata['episode_id']}",
           "family": metadata["family"], "source_file": prov.get("file"), "demo_key": prov.get("demo_key")}
    if info is None or "scene" not in info:
        return {**rec, "status": "no_physics"}, None
    sha = info["scene"]["source_mjcf_sha256"]
    rec.update(source_mjcf_sha256=sha, scene=f"scenes/{sha}", simulator=info.get("simulator"))
    if not prov.get("file") or not prov.get("demo_key"):
        return {**rec, "status": "no_source", "reason": "provenance lacks file or demo_key"}, None
    return rec, info


def export_scene(ref, rec: dict, info: dict, out: Path) -> dict:
    """Check the regenerated SceneRef ``ref`` against the episode's tier-P record and save it."""
    from .validate.scene import build_scene

    t0 = time.time()
    recorded = info["scene"]
    sha = recorded["source_mjcf_sha256"]
    dest = out / "scenes" / sha
    if dest.exists():
        return {**rec, "status": "exists"}
    if ref is None:
        return {**rec, "status": "no_scene", "reason": "the adapter returned no scene"}
    if hashlib.sha256(ref.mjcf.encode()).hexdigest() != sha:
        return {**rec, "status": "mismatch", "reason": "source MJCF sha256 differs from the episode's record"}
    built = build_scene(ref, drop_bodies=recorded["removed"]["parked_bodies"])
    checks = scene_checks(built.info, recorded, info.get("simulator"))
    keys = asset_keys(built.info["assets"])
    if keys is None:
        return {**rec, "status": "unresolved_assets", "checks": checks}
    if not checks["passed"]:
        return {**rec, "status": "mismatch", "checks": checks}
    stage = dest.with_name(f".{sha}.partial")
    doc = ref.save(stage, keys)
    stage.rename(dest)
    return {**rec, "status": "exported", "checks": checks, "assets": len(doc["assets"]),
            "parked_bodies": recorded["removed"]["parked_bodies"], "seconds": round(time.time() - t0, 2)}


def export(paths, out, *, name="export") -> list[dict]:
    """Export the scenes of built episodes (module docstring); each source file is read once."""
    from . import sources

    out = Path(out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    records = []
    log = (out / "records" / f"{name}.jsonl").open("a")

    def done(rec):
        log.write(json.dumps(rec, sort_keys=True) + "\n")
        log.flush()
        records.append(rec)
        print(json.dumps({k: rec.get(k) for k in ("uid", "status", "source_mjcf_sha256")}), flush=True)

    groups = {}  # (family, source file) -> {demo key: (record, info)}
    for path in paths:
        try:
            rec, info = _record(path)
        except Exception as error:  # noqa: BLE001 - failures are recorded, the batch goes on
            done({"format": EXPORT_FORMAT, "episode": str(path), "status": "error", "error": repr(error)})
            continue
        if info is None:
            done(rec)
        else:
            groups.setdefault((rec["family"], rec["source_file"]), {})[rec["demo_key"]] = (rec, info)
    with log:
        for (family, source), wanted in groups.items():
            pending = dict(wanted)
            try:
                for src in sources.iter_episodes(family, source, demos=list(wanted)):
                    key = (src.provenance or {}).get("demo_key") if src is not None else None
                    if key not in pending:
                        continue
                    rec, info = pending.pop(key)
                    try:
                        done(export_scene(src.scene, rec, info, out))
                    except Exception as error:  # noqa: BLE001
                        done({**rec, "status": "error", "error": repr(error)})
            except Exception as error:  # noqa: BLE001
                for rec, _ in pending.values():
                    done({**rec, "status": "error", "error": repr(error)})
                continue
            for rec, _ in pending.values():
                done({**rec, "status": "no_scene", "reason": "the adapter yielded no episode for this demo"})
    return records


def manifest(build, pvc_build, *, family, tier_k, tier_p, publish, timeout_s=3 * 3600) -> list[dict]:
    """Job lines (one per raw source file) exporting one episode per distinct source MJCF."""
    import pandas as pd

    build = Path(build)
    index = pd.read_parquet(build / "index.parquet")
    rows = index[index.family == family]
    if tier_k:
        rows = rows[rows.tier_k_passed]
    if tier_p:
        rows = rows[rows.tier_p_passed.fillna(False).astype(bool)]
    chosen = {}  # source MJCF sha256 -> (raw file, episode file)
    for rel in sorted(rows.file):
        metadata, info = _episode_info(build / rel)
        if info is None or "scene" not in info:
            continue
        sha = info["scene"]["source_mjcf_sha256"]
        chosen.setdefault(sha, (metadata["provenance"]["file"], rel))
    by_source = {}
    for sha, (source, rel) in sorted(chosen.items()):
        by_source.setdefault(source, []).append(rel)
    jobs = []
    for i, (source, rels) in enumerate(sorted(by_source.items())):
        job_id = f"scenes-{family}-{i:03d}"
        jobs.append({"id": job_id, "timeout_s": timeout_s, "publish": publish, "source_file": source,
                     "argv": ["-m", "reachy_retarget.scene_export", "export", "--out", "{out}", "--name", job_id,
                              *[f"{{pvc}}/{pvc_build.strip('/')}/{rel}" for rel in rels]]})
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export", help="export the source scenes of built episodes")
    p.add_argument("episodes", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="export", help="record file name (unique per job)")
    p = sub.add_parser("manifest", help="write a cluster.pool manifest from a local build copy")
    p.add_argument("--build", required=True, help="local build folder with index.parquet and episodes/")
    p.add_argument("--pvc-build", required=True, help="the same build below the PVC root, e.g. datasets/w1-v5")
    p.add_argument("--family", required=True)
    p.add_argument("--tier-k", action="store_true", help="only episodes passing tier K")
    p.add_argument("--tier-p", action="store_true", help="only episodes passing tier P")
    p.add_argument("--publish", required=True, help="PVC folder the jobs publish to")
    p.add_argument("--out", required=True, help="manifest .jsonl to write (must not exist)")
    args = parser.parse_args(argv)
    if args.command == "export":
        records = export(args.episodes, args.out, name=args.name)
        raise SystemExit(0 if all(r["status"] in ("exported", "exists") for r in records) else 1)
    jobs = manifest(args.build, args.pvc_build, family=args.family, tier_k=args.tier_k, tier_p=args.tier_p,
                    publish=args.publish)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"{out} exists; not overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    print(json.dumps({"jobs": len(jobs), "scenes": sum(len(j["argv"]) - 7 for j in jobs), "out": str(out)}))


if __name__ == "__main__":
    main()
