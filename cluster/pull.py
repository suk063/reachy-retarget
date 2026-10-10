"""Pull a sample of a published dataset from the PVC to a local folder (explicit, never automatic).

For inspecting what the worker pods produce, e.g. with the tracking viewers. Steps
(:func:`pull_sample`):

1. on a pod, read the record files (``records/**/*.jsonl``) of ``shards`` shards spread evenly
   over the sorted list, group the records by scenario cell (``task`` without a scenario) and draw,
   with a seeded RNG, up to ``per_cell`` tier-K passes and ``failed`` tier-K failures per cell;
   every attempt file of a drawn record is pulled (``<pod path>/episodes/<rest>`` is published at
   ``<dataset>/episodes/<rest>``);
2. local preflight: the destination must not exist and the local disk must keep 50 decimal GB
   free after the transfer (project rule);
3. ``sha256sum`` of the files on the pod, then a ``tar`` stream of them into
   ``<dest>.partial-<nonce>``; every extracted file must match its pod-side hash;
4. the drawn record lines (under the record file's own relative path), ``SHA256SUMS`` and
   ``PULL.json`` (pod, PVC folder, record files read, selection, time, git revision of this repo)
   are written and the staging folder is renamed to ``<dest>``.

The local layout mirrors the PVC folder, so ``python -m reachy_retarget.tracking.viewer --run
<dest>`` and ``python -m reachy_retarget.tracking.mjviewer --run <dest>`` work on it. Nothing on
the PVC is written.

    python -m cluster.pull datasets/tracking-v1 data/pulled/tracking-v1-sample
    python -m cluster.pull datasets/tracking-v1 data/pulled/tracking-v1-more --shards 6 --seed 1

Sources are relative to the PVC root ``/mnt/reachy-retarget``.
"""
from __future__ import annotations

import argparse
import json
import posixpath
import shlex
import shutil
import subprocess
import tarfile
import time
import uuid
from pathlib import Path

from cluster import k8s
from cluster.upload import RESERVE, SUMS, _git_revision, sha256, sums_text

INFO = "PULL.json"

# Runs on the pod with its system python3: argv = dataset root, shards, per_cell, failed, seed.
SELECT = r"""
import json, os, random, sys
root, shards, per_cell, failed, seed = sys.argv[1], *map(int, sys.argv[2:])
paths = sorted(os.path.relpath(os.path.join(d, f), root)
               for d, _, fs in os.walk(os.path.join(root, "records")) for f in fs if f.endswith(".jsonl"))
if not paths:
    sys.exit("no record files under " + root)
step = len(paths) / min(shards, len(paths))
chosen = [paths[int(i * step)] for i in range(min(shards, len(paths)))]
cells = {}
for rel in chosen:
    with open(os.path.join(root, rel)) as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                cell = (rec.get("scenario") or {}).get("cell") or rec.get("task") or "?"
                cells.setdefault(cell, ([], []))[0 if (rec.get("K") or {}).get("passed") else 1].append((rel, line))
rng = random.Random(seed)
records, files = [], []
for cell in sorted(cells):
    for group, k in zip(cells[cell], (per_cell, failed)):
        for rel, line in rng.sample(group, min(k, len(group))):
            rec = json.loads(line)
            names = [a.get("file") for a in rec.get("attempts") or []] + [rec.get("file")]
            names = [n.split("/episodes/", 1)[1] for n in names if n and "/episodes/" in n]
            have = [n for n in dict.fromkeys(names) if os.path.isfile(os.path.join(root, "episodes", n))]
            if have:
                records.append({"record_file": rel, "cell": cell, "line": line.rstrip("\n")})
                files += ["episodes/" + n for n in have]
files = list(dict.fromkeys(files))
print(json.dumps({"record_files": chosen, "cells": {c: [len(g[0]), len(g[1])] for c, g in cells.items()},
                  "records": records, "files": files,
                  "bytes": sum(os.path.getsize(os.path.join(root, f)) for f in files)}))
"""

HASH = """set -eu
cd {root}
tr '\\n' '\\0' | xargs -0 sha256sum --
"""


def _source(source: str) -> str:
    source = posixpath.normpath(source.strip("/"))
    if source.startswith("..") or source in (".", ""):
        raise ValueError(f"{source!r}: source must be a folder below the PVC root")
    return f"{k8s.PVC}/{source}"


def select(source: str, pod: str, *, shards: int, per_cell: int, failed: int, seed: int, run=None) -> dict:
    """Run :data:`SELECT` on ``pod`` for the PVC folder ``source``; returns its JSON result."""
    run = run or (lambda p, s, stdin: k8s.run(p, s, stdin=stdin, timeout=1800))
    args = " ".join(shlex.quote(str(v)) for v in (_source(source), shards, per_cell, failed, seed))
    return json.loads(run(pod, f"python3 - {args}", SELECT.encode()))


def stream_tar(pod: str, root: str, files: list[str], dest: Path) -> None:
    """Extract ``tar -c`` of ``files`` (relative to ``root`` on ``pod``) into ``dest``."""
    proc = subprocess.Popen(["kubectl", "-n", k8s.NAMESPACE, "exec", "-i", pod, "--", "sh", "-c",
                             f"cd {shlex.quote(root)} && tar -cf - -T -"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.stdin.write("".join(f + "\n" for f in files).encode())
    proc.stdin.close()
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
        tar.extractall(dest, filter="data")
    err = proc.stderr.read()
    if proc.wait():
        raise RuntimeError(f"{pod}: tar exit {proc.returncode}: {err.decode(errors='replace')[-2000:]}")


def pull_sample(source: str, dest, pod: str | None = None, *, shards: int = 3, per_cell: int = 10,
                failed: int = 4, seed: int = 0) -> dict:
    """Pull a per-cell sample of the dataset folder ``source`` to ``dest`` (module docstring)."""
    dest = Path(dest).resolve()
    if dest.exists():
        raise FileExistsError(f"{dest} exists; not overwritten")
    pod = pod or k8s.ready_pods()[0]
    root = _source(source)
    sel = select(source, pod, shards=shards, per_cell=per_cell, failed=failed, seed=seed)
    if not sel["files"]:
        raise RuntimeError(f"{root}: no episode files referenced by the drawn records")
    dest.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest.parent).free
    if free - sel["bytes"] < RESERVE:
        raise OSError(f"local free space {free / 1e9:.1f} GB minus {sel['bytes'] / 1e9:.1f} GB would fall below "
                      f"the {RESERVE / 1e9:.0f} GB reserve")
    remote = {}
    for line in k8s.run(pod, HASH.format(root=shlex.quote(root)), stdin="\n".join(sel["files"]).encode(),
                        timeout=3600).splitlines():
        digest, rel = line.split(None, 1)
        remote[rel] = digest
    stage = dest.with_name(f"{dest.name}.partial-{uuid.uuid4().hex[:8]}")
    stage.mkdir()
    t0 = time.time()
    stream_tar(pod, root, sel["files"], stage)
    manifest = {}
    for rel in sel["files"]:
        path = stage / rel
        digest = sha256(path)
        if digest != remote.get(rel):
            raise RuntimeError(f"{rel}: local sha256 {digest} != pod {remote.get(rel)} (staging folder {stage})")
        manifest[rel] = {"sha256": digest, "bytes": path.stat().st_size}
    by_file = {}
    for r in sel["records"]:
        by_file.setdefault(r["record_file"], []).append(r["line"])
    for rel, lines in by_file.items():
        path = stage / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(line + "\n" for line in lines))
        manifest[rel] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (stage / SUMS).write_text(sums_text(manifest))
    info = {"source": root, "pod": pod, "tool": "cluster.pull", "repo_revision": _git_revision(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "selection": {"shards": shards, "per_cell": per_cell, "failed": failed, "seed": seed},
            "record_files_read": sel["record_files"], "cell_counts_read": sel["cells"],
            "records": len(sel["records"]), "episode_files": len(sel["files"]), "bytes": sel["bytes"],
            "note": "record files hold only the drawn lines; episode files are byte-identical to the PVC"}
    (stage / INFO).write_text(json.dumps(info, indent=1))
    stage.rename(dest)
    return {"dest": str(dest), "pod": pod, "records": len(sel["records"]), "episode_files": len(sel["files"]),
            "bytes": sel["bytes"], "seconds": round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", help="dataset folder below the PVC root, e.g. datasets/tracking-v1")
    ap.add_argument("dest", help="new local folder, e.g. data/pulled/tracking-v1-sample")
    ap.add_argument("--pod", help="worker pod (default: first Ready pod)")
    ap.add_argument("--shards", type=int, default=3, help="record files read, spread over the sorted list")
    ap.add_argument("--per-cell", type=int, default=10, help="tier-K passes drawn per cell")
    ap.add_argument("--failed", type=int, default=4, help="tier-K failures drawn per cell")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print(json.dumps(pull_sample(a.source, a.dest, a.pod, shards=a.shards, per_cell=a.per_cell, failed=a.failed,
                                 seed=a.seed)))


if __name__ == "__main__":
    main()
