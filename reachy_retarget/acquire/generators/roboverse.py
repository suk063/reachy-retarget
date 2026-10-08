"""Catalog generator for RoboVerse (family ``roboverse``): RLBench and CALVIN migrations.

Network (HF tree API, metadata only) when invoked::

    python -m reachy_retarget.acquire.generators.roboverse

Checks that every RLBench task's Franka file (``trajs/rlbench/<task>/v2/franka_v2.pkl.gz``)
at the pinned revision is catalogued (the cross-embodiment ``sawyer``/``ur5e`` variants of
``close_box`` stay excluded as generated variants) and rebuilds the CALVIN rows: every
language-annotated window file ``trajs/calvin/calvin_traj_ann/env_{A,B,C,D,D_val}_out/
task_<N>_v2.pkl`` (all LFS files, publisher SHA-256) in ``roboverse.calvin.tsv.gz``. Window
counts known from earlier fetches stay in the row notes. Other groups are kept as they are.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from .common import hf_digests, hf_tree, write_table

FAMILY = "roboverse"
REPO = "RoboVerseOrg/roboverse_data"
REVISION = "fab63ccaaed54f413901f86edc3fa1ab77a96500"
TABLE = "roboverse.calvin.tsv.gz"
WINDOW = re.compile(r"^trajs/calvin/calvin_traj_ann/env_(A|B|C|D|D_val)_out/task_\d+_v2\.pkl$")


def build(out_yaml=None, opener=None) -> dict:
    from ..catalog import CATALOG_DIR, load_catalog

    out_yaml = Path(out_yaml or CATALOG_DIR / f"{FAMILY}.yaml")
    old = yaml.safe_load(out_yaml.read_text())
    have = {f["path"] for g in old["sources"] for f in g.get("files", [])}
    rl = [i for i in hf_tree(REPO, REVISION, "trajs/rlbench", opener=opener) if i["type"] == "file"]
    franka = [i["path"] for i in rl if i["path"].endswith("/v2/franka_v2.pkl.gz")]
    missing = sorted(set(franka) - have)
    if missing:
        raise ValueError(f"RLBench Franka files not catalogued: {missing}")
    items = [i for i in hf_tree(REPO, REVISION, "trajs/calvin/calvin_traj_ann", opener=opener)
             if i["type"] == "file" and WINDOW.match(i["path"])]
    notes = {e.path: e.note for e in load_catalog(out_yaml).values() if WINDOW.match(e.path)}  # earlier counts
    rows = []
    for i in sorted(items, key=lambda i: i["path"]):
        env = WINDOW.match(i["path"])[1]
        d = hf_digests(i)
        if "sha256" not in d:
            raise ValueError(f"{i['path']}: not an LFS file")
        rows.append({"src": i["path"], "size": i["size"], "sha256": d["sha256"], "content": "low_dim",
                     "dataset": f"roboverse/calvin/env_{env}", "note": notes.get(i["path"])})
    write_table(out_yaml.parent / TABLE, ["src", "size", "sha256", "content", "dataset", "note"], rows)
    groups = []
    for g in old["sources"]:
        if g.get("table") == TABLE:
            continue
        files = [f for f in g.get("files", []) if not WINDOW.match(f["path"])]
        groups.append({**g, "files": files})
    calvin = next(g for g in groups if "CALVIN" in (g.get("note") or ""))
    per_env = {}
    for r in rows:
        per_env.setdefault(r["dataset"], [0, 0])
        per_env[r["dataset"]][0] += 1
        per_env[r["dataset"]][1] += r["size"]
    groups.insert(groups.index(calvin) + 1, {
        **{k: calvin[k] for k in ("family", "repo", "revision", "license")}, "kind": "file",
        "url_base": f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/", "path_prefix": "",
        "note": "CALVIN language-annotated windows (<= 64 states each, crops of the continuous play stream; windows "
                "of one stream may overlap) for scenes A, B, C, D and the D validation split. LFS sha256.",
        "table": TABLE, "totals": {k: {"files": v[0], "bytes": v[1]} for k, v in sorted(per_env.items())}})
    doc = {"sources": groups, **{k: v for k, v in old.items() if k not in ("sources", "calvin_traj_ann_all")}}
    doc["not_catalogued"] = {k: v for k, v in doc.get("not_catalogued", {}).items()
                             if not k.startswith("trajs/calvin/calvin_traj_ann/")}
    header = "".join(line + "\n" for line in out_yaml.read_text().splitlines() if line.startswith("#"))
    out_yaml.write_text(header + yaml.safe_dump(doc, sort_keys=False, width=120))
    return {"rlbench_franka_files": len(franka), "calvin": {k: v for k, v in sorted(per_env.items())}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", help="catalog yaml to write (default: the packaged catalog)")
    print(json.dumps(build(ap.parse_args(argv).out)))


if __name__ == "__main__":
    main()
