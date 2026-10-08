"""Catalog generator for MolmoBot-Data scene packages (family ``molmobot``).

Network (HF tree API for shard digests, metadata only) when invoked::

    python -m reachy_retarget.acquire.generators.molmobot --data <root>/raw/molmobot

MolmoBot-Data stores one zstd tar per scene ("package") as a byte range of a shard tar;
packages hold the trajectory HDF5 next to MP4 videos. Objects are only known at t = 0 there,
so the catalog is a bounded, diverse subset: per generation config (both robots, every task
type) packages listed in ``commercial_episodes.parquet`` (commercial-use objects only) are
taken in a fixed pseudo-random order (sha1 of ``part/path``; packages above
``MAX_PACKAGE_BYTES`` compressed are skipped so that one scene cannot fill a budget) until the config's trajectory
budget is met, ``VAL_SHARE`` of it from the validation split. Trajectory counts are exact
where the commercial list names episodes and estimated from the inflated package size (bytes
per trajectory of the packages fetched earlier, per config) where it says ``*``. Each package
is a ``range_package`` row: shard path, shard SHA-256 (HF LFS), offset, compressed size and
``inflated_size`` (publisher package tables; = the sum of the member file sizes, checked at fetch); ``range_sha256`` and member digests are recorded at
fetch time (trust-on-first-use) except for the earlier sample, which keeps its catalog digests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from .common import hf_tree, write_table

FAMILY = "molmobot"
REPO = "allenai/molmobot-data"
REVISION = "159a5aecaf2ba73855940ad325b477b27d85da73"
TABLE = "molmobot.packages.tsv.gz"
TARGET = 20_000
VAL_SHARE = 0.1
MAX_PACKAGE_BYTES = 500_000_000  # a few outlier scenes hold hundreds of episodes; prefer many scenes
CONFIGS = {  # config: (robot, task type)
    "DoorOpeningDataGenConfig": ("rby1m", "door_open"),
    "RBY1OpenDataGenConfig": ("rby1m", "open"),
    "RBY1PickDataGenConfig": ("rby1m", "pick"),
    "RBY1PickAndPlaceDataGenConfig": ("rby1m", "pick_and_place"),
    "FrankaPickOmniCamConfig": ("franka_droid", "pick"),
    "FrankaPickAndPlaceOmniCamConfig": ("franka_droid", "pick_and_place"),
    "FrankaPickAndPlaceColorOmniCamConfig": ("franka_droid", "pick_and_place_color"),
    "FrankaPickAndPlaceNextToOmniCamConfig": ("franka_droid", "pick_and_place_next_to"),
    "FrankaPickAndPlaceOmniCamConfig_ObjectBackfill": ("franka_droid", "pick_and_place (extra object types)"),
}


def _order(row) -> str:
    return hashlib.sha1(f"{row['part']}/{row['path']}".encode()).hexdigest()


def _house(package: str) -> str:
    return "house_" + package.rsplit("_house_", 1)[1].removesuffix(".tar.zst")


def bytes_per_trajectory(sample: list[dict]) -> dict[str, float]:
    """Inflated package bytes per trajectory, per config, from catalogued packages with members."""
    acc = {}
    for f in sample:
        n = sum(m.get("trajectories", 0) for m in f.get("members", []))
        if n:
            a = acc.setdefault(f["config"], [0, 0])
            a[0] += f["inflated_size"]
            a[1] += n
    return {k: v[0] / v[1] for k, v in acc.items()}


def h5_share(sample: list[dict]) -> dict[str, float]:
    acc = {}
    for f in sample:
        a = acc.setdefault(f["config"], [0, 0])
        a[0] += sum(m["size"] for m in f.get("members", []))
        a[1] += f["inflated_size"]
    return {k: v[0] / v[1] for k, v in acc.items() if v[1]}


def select(commercial: list[dict], tables: dict, sample: list[dict], target=TARGET, val_share=VAL_SHARE):
    """Rows of the selected packages (the earlier sample first) and per-config totals."""
    bpt, share = bytes_per_trajectory(sample), h5_share(sample)
    sampled = {(f["config"], f["split"], f["part"], f["package"]) for f in sample}
    budget = target / len(CONFIGS)
    rows, totals = [], {}
    for config in CONFIGS:
        tot = totals.setdefault(config, {"packages": 0, "trajectories_est": 0, "compressed_bytes": 0,
                                         "inflated_bytes": 0, "h5_bytes_est": 0})
        val_got = 0.0
        for split in ("val", "train"):
            want = budget * val_share if split == "val" else budget - val_got
            got = float(sum(m.get("trajectories", 0) for f in sample if (f["config"], f["split"]) == (config, split)
                            for m in f.get("members", [])))
            cands = sorted((c for c in commercial if c["config"] == config and c["split"] == split
                            and c["valid_episodes_string"] != "" and c["size"] <= MAX_PACKAGE_BYTES
                            and (config, split, c["part"], c["path"]) not in sampled),
                           key=_order)
            for c in cands:
                if got >= want:
                    break
                t = tables[(config, split)][(c["part"], c["path"])]
                if t["offset"] != c["offset"] or t["size"] != c["size"] or t["shard_id"] != c["shard_id"]:
                    raise ValueError(f"{c['path']}: commercial list and package table disagree")
                if not t["inflated_size"]:
                    continue  # empty package (no members)
                v = c["valid_episodes_string"]
                est = len(v.split(",")) if v != "*" else max(1, round(t["inflated_size"] / bpt[config]))
                got += est
                rows.append({"path": f"{config}/{split}/part{c['part']}/{_house(c['path'])}",
                             "archive": f"{config}/{split}_shards/{c['shard_id']:05d}.tar", "offset": c["offset"],
                             "stored_size": c["size"], "inflated_size": t["inflated_size"], "config": config,
                             "split": split, "part": c["part"], "package": c["path"], "scene_id": c["scene_id"],
                             "scene_family": c["scene_family"], "commercial_valid_episodes": v,
                             "est_trajectories": est, "dataset": f"molmobot/{config}/{split}"})
                tot["packages"] += 1
                tot["trajectories_est"] += est
                tot["compressed_bytes"] += c["size"]
                tot["inflated_bytes"] += t["inflated_size"]
                tot["h5_bytes_est"] += round(t["inflated_size"] * share.get(config, 0.15))
            if split == "val":
                val_got = got
    return rows, totals


def build(data_dir, out_yaml=None, opener=None, target=TARGET) -> dict:
    import pyarrow.parquet as pq

    from ..catalog import CATALOG_DIR

    data_dir = Path(data_dir)
    out_yaml = Path(out_yaml or CATALOG_DIR / f"{FAMILY}.yaml")
    old = yaml.safe_load(out_yaml.read_text())
    groups = [g for g in old["sources"] if g.get("table") != TABLE]
    inline = next(g for g in groups if g.get("kind") == "range_package")
    sample = inline["files"]
    commercial = pq.read_table(data_dir / "commercial_episodes.parquet").to_pylist()
    tables = {}
    for config in CONFIGS:
        for split in ("train", "val"):
            p = data_dir / config / f"{split}_pkgs-00000-of-00001.parquet"
            if p.exists():
                tables[(config, split)] = {(r["part"], r["path"]): r for r in pq.read_table(p).to_pylist()}
    rows, totals = select(commercial, tables, sample, target)
    shards = {}
    for config in CONFIGS:
        for split in ("train", "val"):
            if any(r["archive"].startswith(f"{config}/{split}_shards/") for r in rows):
                for it in hf_tree(REPO, REVISION, f"{config}/{split}_shards", opener=opener):
                    if it["type"] == "file":
                        shards[it["path"]] = (it["lfs"]["oid"], it["size"])
    for r in rows:
        r["archive_sha256"], r["archive_size"] = shards[r["archive"]]
        if r["offset"] + r["stored_size"] > r["archive_size"]:
            raise ValueError(f"{r['package']}: range beyond its shard")
    cols = ["path", "archive", "archive_sha256", "archive_size", "offset", "stored_size", "inflated_size", "dataset",
            "config", "split", "part", "package", "scene_id", "scene_family", "commercial_valid_episodes",
            "est_trajectories"]
    write_table(out_yaml.parent / TABLE, cols, rows)
    sample_traj = sum(m.get("trajectories", 0) for f in sample for m in f.get("members", []))
    summary = {"packages": len(rows) + len(sample), "trajectories_est": sum(r["est_trajectories"] for r in rows)
               + sample_traj, "compressed_bytes": sum(r["stored_size"] for r in rows)
               + sum(f["stored_size"] for f in sample), "per_config": totals}
    groups.insert(groups.index(inline) + 1, {
        "family": FAMILY, "repo": f"https://huggingface.co/datasets/{REPO}", "revision": REVISION,
        "license": inline["license"], "kind": "range_package", "content": "low_dim", "compression": "zstd",
        "keep": [".h5"], "archive_format": "tar", "url_base": f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/",
        "note": f"Bounded diverse subset (target {target} trajectories, objects known at t0 only): per config "
                f"(2 robots x 9 generation configs) commercial-use packages of <= {MAX_PACKAGE_BYTES // 10**6} MB in "
                "sha1(part/path) order until "
                f"1/9 of the target is met, {int(VAL_SHARE * 100)} % from val. est_trajectories: exact count of "
                "commercial episodes where listed, else inflated size / bytes per trajectory of the earlier sample. "
                "range_sha256 and member sha256 are recorded at fetch time (trust-on-first-use); the member sizes "
                "must sum to the publisher's inflated_size.",
        "table": TABLE, "totals": summary})
    doc = {"sources": groups, **{k: v for k, v in old.items() if k != "sources"}}
    header = "".join(line + "\n" for line in out_yaml.read_text().splitlines() if line.startswith("#"))
    out_yaml.write_text(header + yaml.safe_dump(doc, sort_keys=False, width=120))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="fetched raw/molmobot folder (package tables, commercial list)")
    ap.add_argument("--out", help="catalog yaml to write (default: the packaged catalog)")
    ap.add_argument("--target", type=int, default=TARGET)
    a = ap.parse_args(argv)
    print(json.dumps(build(a.data, a.out, target=a.target)))


if __name__ == "__main__":
    main()
