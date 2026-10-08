"""Catalog generator for MobileManiBench (``arnoldland/MobileManiBench``), G1 robot only.

Network use only when invoked::

    python -m reachy_retarget.acquire.generators.mobilemanibench --cache <dir> [--out <catalog.yaml>]

Every G1 per-object tar listed in ``object_manifest.jsonl`` is indexed by reading its tar
headers with small HTTP ranges (:func:`..common.index_remote_tar`); the member count must
equal the publisher's ``nfiles``. Selected members (``range_member`` rows of
``mobilemanibench.members.tsv.gz``): every episode's ``state_infos.pkl`` and the per-run
configuration (``params/{env,agent}.yaml``, ``git/*.diff``, run ``log.txt``) and per
trajectory batch ``scene_infos.json``/``log.txt``. Never videos (``*.mp4``), policy
checkpoints (``*.pt``/``*.onnx``) or tensorboard events. Member digests are not published
(only whole-tar SHA-256); members first fetched by the earlier subset keep the SHA-256
computed then, the others are verified at fetch time by the tar header that precedes
their bytes (name, size, header checksum) and recorded trust-on-first-use.
"""
from __future__ import annotations

import argparse
import json
import posixpath
import sys
import time
from pathlib import Path

import yaml

from ..transports import KeepAliveOpener, is_image_name
from .common import hf_resolve, index_remote_tar, write_table

FAMILY = "mobilemanibench"
REPO = "arnoldland/MobileManiBench"
REVISION = "88bc86eed0162c3b9a27bc962ce2d4815cbf2e59"
TABLE = "mobilemanibench.members.tsv.gz"
SKIP_SUFFIX = (".mp4", ".pt", ".onnx", ".npz", ".png", ".jpg")


def wanted(name: str) -> str | None:
    """Content class of a member to catalogue, or ``None``."""
    base = posixpath.basename(name)
    if name.endswith(SKIP_SUFFIX) or base.startswith("events.out.tfevents") or is_image_name(name):
        return None
    if base == "state_infos.pkl":
        return "low_dim"
    if base in ("env.yaml", "agent.yaml", "scene_infos.json", "log.txt") or base.endswith(".diff"):
        return "metadata"
    return None


def manifest_rows(path=None) -> list[dict]:
    """G1 rows of the publisher's ``object_manifest.jsonl`` (local copy or the pinned URL)."""
    if path and Path(path).exists():
        text = Path(path).read_text()
    else:
        with KeepAliveOpener()(__import__("urllib.request").request.Request(
                hf_resolve(REPO, REVISION, "MobileManiDataset/object_manifest.jsonl"))) as r:
            text = r.read().decode()
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [r for r in rows if r["robot"] == "G1_Robot"]


def index_tars(cache, manifest=None, workers=48, parallel_tars=6, log=print) -> None:
    """Index every G1 tar into ``<cache>/<tar name>.json`` (skips tars already indexed)."""
    from concurrent.futures import ThreadPoolExecutor

    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    opener = KeepAliveOpener()

    def one(row):
        out = cache / (row["repo_path"].replace("/", "__") + ".json")
        if out.exists():
            return
        url = hf_resolve(REPO, REVISION, row["repo_path"])
        size = row["tar_bytes"]
        t = time.time()
        members = index_remote_tar(url, size, opener=opener, segment=256 << 20, workers=workers, scan=1 << 20)
        files = [m for m in members if m[3] in ("0", "\x00", "7")]
        if len(files) != row["nfiles"]:
            raise IOError(f"{row['repo_path']}: indexed {len(files)} files, manifest nfiles {row['nfiles']}")
        out.write_text(json.dumps({"archive": row["repo_path"], "size": size, "sha256": row["sha256"],
                                   "nfiles": row["nfiles"], "members": files}))
        log(f"{row['repo_path']}: {len(files)} files in {time.time() - t:.0f} s")

    rows = sorted(manifest_rows(manifest), key=lambda r: -r["tar_bytes"])  # largest first
    with ThreadPoolExecutor(parallel_tars) as ex:
        for fut in [ex.submit(one, r) for r in rows]:
            fut.result()


def build(cache, out_yaml=None, manifest=None) -> dict:
    """Write ``mobilemanibench.yaml`` (whole-file groups kept) and the member table."""
    from ..catalog import CATALOG_DIR, load_catalog

    out_yaml = Path(out_yaml or CATALOG_DIR / f"{FAMILY}.yaml")
    old = yaml.safe_load(out_yaml.read_text())
    groups = [g for g in old["sources"] if g.get("kind", "file") == "file"]
    members = next(g for g in old["sources"] if g.get("kind") == "range_member")
    known = {e.path: e.sha256 for e in load_catalog(out_yaml).values() if e.sha256}  # digests of earlier fetches
    zip_members = [f for f in members.get("files", []) if f.get("compression") == "deflate"]
    zip_archives = {f["archive"]: members["archives"][f["archive"]] for f in zip_members}
    rows, archives, counts = [], {}, {"tars": 0, "episodes": 0, "bytes": 0, "files": 0}
    for row in manifest_rows(manifest):
        idx = json.loads((Path(cache) / (row["repo_path"].replace("/", "__") + ".json")).read_text())
        archives[row["repo_path"]] = {"url": hf_resolve(REPO, REVISION, row["repo_path"]), "sha256": row["sha256"],
                                      "size": row["tar_bytes"], "format": "tar", "nfiles": row["nfiles"]}
        counts["tars"] += 1
        group = f"{FAMILY}/{posixpath.dirname(row['repo_path']).split('MobileManiDataset/')[1]}/{row['object']}"
        for off, size, name, _ in idx["members"]:
            content = wanted(name)
            if content is None:
                continue
            rows.append({"path": name, "archive": row["repo_path"], "offset": off, "size": size, "content": content,
                         "dataset": group, "sha256": known.get(name)})
            counts["files"] += 1
            counts["bytes"] += size
            counts["episodes"] += content == "low_dim"
    rows.sort(key=lambda r: r["path"])
    write_table(out_yaml.parent / TABLE, ["path", "archive", "offset", "size", "content", "dataset", "sha256"], rows)
    groups.append({
        "family": FAMILY, "repo": f"https://huggingface.co/datasets/{REPO}", "revision": REVISION, "license": "MIT",
        "kind": "range_member",
        "note": "Single members of the per-object G1 tars (byte ranges; the tars also hold MP4 videos, policy "
                "checkpoints and tensorboard events, which are never requested) and the robot URDF from "
                "Assets/Assets.zip (deflate member, CRC-32 checked). Archive sha256 = publisher "
                "(object_manifest.jsonl = HF LFS oid). Member sha256 is published for none of them: rows with a "
                "sha256 were hashed when the earlier subset was fetched; the others are checked against the tar "
                "header preceding their bytes and recorded trust-on-first-use.",
        "archives": {**archives, **zip_archives},
        "files": zip_members,
        "table": TABLE,
        "totals": counts})
    header = (f"# Pinned MobileManiBench files (family '{FAMILY}'). Generated by\n"
              "# reachy_retarget.acquire.generators.mobilemanibench (tar headers of every G1 tar read over HTTP Range at\n"
              "# the pinned HF revision, member counts checked against object_manifest.jsonl); edit by regenerating.\n"
              "# kind: file = whole files; range_member = one archive member read by byte range (videos never).\n"
              "# The XHand (dexterous hand) tars are excluded. The member rows are in the gzip TSV named by 'table'.\n")
    out_yaml.write_text(header + yaml.safe_dump({"sources": groups}, sort_keys=False, width=120))
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True, help="directory for per-tar header indexes (resumable)")
    ap.add_argument("--manifest", help="local copy of object_manifest.jsonl")
    ap.add_argument("--out", help="catalog yaml to write (default: the packaged catalog)")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--index-only", action="store_true")
    a = ap.parse_args(argv)
    index_tars(a.cache, a.manifest, a.workers, log=lambda s: print(s, flush=True))
    if not a.index_only:
        print(json.dumps(build(a.cache, a.out, a.manifest)))


if __name__ == "__main__":
    sys.exit(main())
