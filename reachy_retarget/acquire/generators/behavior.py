"""Catalog generator for BEHAVIOR-1K 2025 challenge demos (family ``behavior``).

Network (HF tree API, metadata only) when invoked::

    python -m reachy_retarget.acquire.generators.behavior

Catalogues every episode of ``behavior-1k/2025-challenge-demos`` at the pinned revision:
``data/task-*/episode_*.parquet`` (low-dim LeRobot v2.1 frames; HF LFS SHA-256),
``meta/episodes/task-*/episode_*.json`` (episode metadata) and ``annotations/task-*/
episode_*.json`` (skill annotations); both JSON kinds are plain git files pinned by their
git blob SHA-1 (verified at fetch time; their SHA-256 is recorded trust-on-first-use, or
kept where the earlier subset already hashed them). Not catalogued: ``videos/``,
``meta/episodes_stats.jsonl`` and the raw HDF5 (``2025-challenge-rawdata``; the entries of
the earlier 80-episode subset stay, as optional sources of the success flag).

Overlap: ``behavior-1k/2026-challenge-demos`` (LeRobot v3, 100 tasks x 200 episodes) re-exports
these 10,000 episodes as its tasks 0-49 (same task order, episode k of task t = 2025 episode
``t*10000 + 10*(k+1)``; per-task episode lengths are identical); its tasks 50-99 are new.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from .common import hf_digests, hf_tree, write_table

FAMILY = "behavior"
REPO = "behavior-1k/2025-challenge-demos"
REVISION = "33639692425296185f245aaa648530178c7b3d2b"
PREFIX = "2025-challenge-demos/"
TABLE = "behavior.demos.tsv.gz"
EPISODE = re.compile(r"^(data|meta/episodes|annotations)/task-(\d{4})/episode_(\d{8})\.(parquet|json)$")
OVERLAP_2026 = ("behavior-1k/2026-challenge-demos@4f50b44796641a4d526a19d9aeadc8aa51e2f2c2 (LeRobot v3, 100 tasks, "
                "20,000 episodes) re-exports these 10,000 episodes as its tasks 0-49 (same task order; per-task "
                "episode lengths identical for the tasks checked: 0, 25, 49; 2026 episode k of task t = 2025 episode "
                "t*10000 + 10*(k+1)); tasks 50-99 (10,000 episodes) are new. Not catalogued; never count both copies.")


def task_names(path) -> dict[int, str]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {r["task_index"]: r["task_name"] for r in rows}


def rows_from_tree(items, names, known=None) -> list[dict]:
    """Table rows for the per-episode files among HF tree ``items``."""
    known = known or {}
    rows = []
    for it in items:
        m = EPISODE.match(it["path"]) if it["type"] == "file" else None
        if not m:
            continue
        task = int(m[2])
        path = PREFIX + it["path"]
        digests = hf_digests(it)
        rows.append({"src": it["path"], "size": it["size"], "sha256": digests.get("sha256") or known.get(path),
                     "git_blob_sha1": digests.get("git_blob_sha1"),
                     "content": "low_dim" if m[1] == "data" else "metadata",
                     "dataset": f"behavior/2025-challenge/{names[task]}", "task_index": task,
                     "episode_index": int(m[3])})
    rows.sort(key=lambda r: r["src"])
    return rows


def build(out_yaml=None, tasks_jsonl=None, opener=None) -> dict:
    from ..catalog import CATALOG_DIR, load_catalog

    out_yaml = Path(out_yaml or CATALOG_DIR / f"{FAMILY}.yaml")
    old = yaml.safe_load(out_yaml.read_text())
    per_episode = re.compile(r"^2025-challenge-demos/(data|meta/episodes|annotations)/")
    known = {e.path: e.sha256 for e in load_catalog(out_yaml).values()}  # sha256 computed by earlier fetches
    groups = []
    for g in old["sources"]:
        if g.get("table"):
            continue
        files = [f for f in g["files"] if not per_episode.match(f["path"])]
        if files:
            groups.append({**g, "files": files})
    if tasks_jsonl is None:
        from ..transports import KeepAliveOpener, make_request
        with (opener or KeepAliveOpener())(make_request(
                f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/meta/tasks.jsonl")) as r:
            names = {json.loads(x)["task_index"]: json.loads(x)["task_name"]
                     for x in r.read().decode().splitlines() if x.strip()}
    else:
        names = task_names(tasks_jsonl)
    items = [it for sub in ("data", "meta/episodes", "annotations")
             for it in hf_tree(REPO, REVISION, sub, opener=opener)]
    rows = rows_from_tree(items, names, known)
    write_table(out_yaml.parent / TABLE, ["src", "size", "sha256", "git_blob_sha1", "content", "dataset",
                                          "task_index", "episode_index"], rows)
    by = {}
    for r in rows:
        key = r["src"].split("/")[0] if not r["src"].startswith("meta/") else "meta/episodes"
        by.setdefault(key, [0, 0])
        by[key][0] += 1
        by[key][1] += r["size"]
    episodes = {r["episode_index"] for r in rows if r["src"].startswith("data/")}
    totals = {"episodes": len(episodes), "tasks": len({r["task_index"] for r in rows}),
              **{f"{k}_files": v[0] for k, v in by.items()}, **{f"{k}_bytes": v[1] for k, v in by.items()}}
    groups.insert(1, {
        "family": FAMILY, "repo": f"https://huggingface.co/datasets/{REPO}", "revision": REVISION, "license": "MIT",
        "kind": "file", "url_base": f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/",
        "path_prefix": PREFIX,
        "note": "Every episode of the 2025 challenge demos: low-dim parquet (LFS sha256), episode metadata JSON and "
                "skill annotations (plain git files: git blob sha1 pinned, sha256 kept where already computed). No "
                "videos, no episodes_stats.jsonl. " + OVERLAP_2026,
        "table": TABLE, "totals": totals})
    doc = {"sources": groups, **{k: v for k, v in old.items() if k not in ("sources", "subset")},
           "overlap": {"2026-challenge-demos": OVERLAP_2026,
                       "2025-challenge-rawdata": "raw OmniGibson HDF5 of the same episode indices (source of the "
                                                 "parquet); never counted separately"}}
    header = ("# Pinned BEHAVIOR-1K 2025 challenge files (family 'behavior'). Generated by\n"
              "# reachy_retarget.acquire.generators.behavior from the HF tree API at the pinned revisions: parquet / HDF5\n"
              "# digests are HF LFS sha256; plain-git JSON files are pinned by git blob sha1 (and sha256 where an earlier\n"
              "# fetch hashed them). The per-episode rows (all 10,000 episodes) are in the gzip TSV named by 'table'.\n"
              "# kind = how bytes are acquired (file | file_images_embedded | tar_stream | range_member | range_package; all\n"
              "# through reachy_retarget.acquire.fetch); content = what they are (low_dim | metadata | assets | docs ...).\n")
    out_yaml.write_text(header + yaml.safe_dump(doc, sort_keys=False, width=120))
    return totals


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", help="catalog yaml to write (default: the packaged catalog)")
    ap.add_argument("--tasks", help="local copy of meta/tasks.jsonl at the pinned revision")
    a = ap.parse_args(argv)
    print(json.dumps(build(a.out, a.tasks)))


if __name__ == "__main__":
    main()
