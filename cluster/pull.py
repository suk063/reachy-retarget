"""Pull a sample of a published dataset from the PVC to a local folder (explicit, never automatic).

For inspecting what the worker pods produce, e.g. with the tracking viewers, and for local training
subsets. Steps (:func:`pull_sample`):

1. on a pod, read the record files (``records/**/*.jsonl``) of ``shards`` shards spread evenly
   over the sorted list, group the records by scenario cell (``task`` without a scenario) and draw,
   with a seeded RNG, up to ``per_cell`` tier-K passes and ``failed`` tier-K failures per cell;
   every attempt file of a drawn record is pulled, or with ``final_only`` only the record's own
   file (its last attempt; earlier ``-try<k>`` attempts are variants of it)
   (``<pod path>/episodes/<rest>`` is published at ``<dataset>/episodes/<rest>``);
2. local preflight: the destination must not exist and the local disk must keep 50 decimal GB
   free after the transfer (project rule);
3. the files are copied by :func:`transfer` into ``<dest>.partial``: tar streams of at most
   ``batch_bytes``, ``workers`` at once over ``pods`` worker pods; per stream the pod-side
   ``sha256sum`` of its files, and every extracted file must match it. A stream cut short is retried
   on another pod; a rerun of the same command resumes from the finished streams and the drawn
   records saved in the staging folder;
4. the drawn record lines (under the record file's own relative path), ``SHA256SUMS`` and
   ``PULL.json`` (pods, PVC folder, record files read, selection, time, git revision of this repo)
   are written and the staging folder is renamed to ``<dest>``.

The local layout mirrors the PVC folder, so ``python -m reachy_retarget.tracking.viewer --run
<dest>`` and ``python -m reachy_retarget.tracking.mjviewer --run <dest>`` work on it. Nothing on
the PVC is written.

    python -m cluster.pull datasets/tracking-v1 data/pulled/tracking-v1-sample
    python -m cluster.pull datasets/tracking-v1 data/pulled/tracking-v1-more --shards 6 --seed 1
    python -m cluster.pull datasets/tracking-v1 data/pulled/tracking-v1-150k --shards 100000 \\
        --per-cell 10000 --failed 0 --final-only --pods 32

``--all`` pulls every file of a small folder (:func:`pull_all`, e.g. exported source scenes) with the
same preflight, hash checks, ``SHA256SUMS`` and ``PULL.json``:

    python -m cluster.pull datasets/scenes-mimicgen-kp data/pulled/scenes-mimicgen-kp --all

Sources are relative to the PVC root ``/mnt/reachy-retarget``.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import posixpath
import shlex
import shutil
import subprocess
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cluster import k8s
from cluster.upload import RESERVE, SUMS, _git_revision, sha256, sums_text

INFO = "PULL.json"
# Staging-folder bookkeeping of a resumable transfer: plan digest, finished streams, the pod-side selection.
PLAN, DONE, SELECTION = ".plan.json", ".streams.jsonl", ".selection.json.gz"

# Runs on the pod with its system python3: argv = dataset root, shards, per_cell, failed, seed[, final_only].
# Only (record file, byte offset) pairs are kept while grouping, so a million records fit a pod's memory;
# the result is gzip-compressed and base64-encoded.
SELECT = r"""
import base64, gzip, json, os, random, sys
root, shards, per_cell, failed, seed = sys.argv[1], *map(int, sys.argv[2:6])
final_only = len(sys.argv) > 6 and sys.argv[6] == "1"
paths = sorted(os.path.relpath(os.path.join(d, f), root)
               for d, _, fs in os.walk(os.path.join(root, "records")) for f in fs if f.endswith(".jsonl"))
if not paths:
    sys.exit("no record files under " + root)
step = len(paths) / min(shards, len(paths))
chosen = [paths[int(i * step)] for i in range(min(shards, len(paths)))]
cells = {}
for rel in chosen:
    with open(os.path.join(root, rel), "rb") as fh:
        offset = 0
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                cell = (rec.get("scenario") or {}).get("cell") or rec.get("task") or "?"
                cells.setdefault(cell, ([], []))[0 if (rec.get("K") or {}).get("passed") else 1].append((rel, offset))
            offset += len(line)
rng = random.Random(seed)
records, files, handles = [], [], {}
for cell in sorted(cells):
    for group, k in zip(cells[cell], (per_cell, failed)):
        for rel, offset in rng.sample(group, min(k, len(group))):
            if rel not in handles:
                handles[rel] = open(os.path.join(root, rel), "rb")
            handles[rel].seek(offset)
            line = handles[rel].readline().decode().rstrip("\n")
            rec = json.loads(line)
            names = [rec.get("file")] if final_only else [a.get("file") for a in rec.get("attempts") or []] + [rec.get("file")]
            names = [n.split("/episodes/", 1)[1] for n in names if n and "/episodes/" in n]
            have = [n for n in dict.fromkeys(names) if os.path.isfile(os.path.join(root, "episodes", n))]
            if have:
                records.append({"record_file": rel, "cell": cell, "line": line})
                files += ["episodes/" + n for n in have]
files = list(dict.fromkeys(files))
sizes = {f: os.path.getsize(os.path.join(root, f)) for f in files}
out = {"record_files": chosen, "cells": {c: [len(g[0]), len(g[1])] for c, g in cells.items()},
       "records": records, "files": files, "sizes": sizes, "bytes": sum(sizes.values())}
sys.stdout.write(base64.b64encode(gzip.compress(json.dumps(out).encode(), 6)).decode())
"""

# 16 files hashed at once: CephFS reads of small files are latency-bound (~0.3 s per file one at a
# time); the hashed files stay in the pod's page cache for the tar stream that follows. With 16 files
# per sha256sum call each call's output is one pipe write, so lines never interleave.
HASH = """set -eu
cd {root}
tr '\\n' '\\0' | xargs -0 -n 16 -P 16 sha256sum --
"""


def _source(source: str) -> str:
    source = posixpath.normpath(source.strip("/"))
    if source.startswith("..") or source in (".", ""):
        raise ValueError(f"{source!r}: source must be a folder below the PVC root")
    return f"{k8s.PVC}/{source}"


def _run(pod, script, stdin=None):
    return k8s.run(pod, script, stdin=stdin, timeout=3600)


def select(source: str, pod: str, *, shards: int, per_cell: int, failed: int, seed: int, final_only: bool = False,
           run=None) -> dict:
    """Run :data:`SELECT` on ``pod`` for the PVC folder ``source``; returns its JSON result."""
    run = run or (lambda p, s, stdin: k8s.run(p, s, stdin=stdin, timeout=3600))
    args = " ".join(shlex.quote(str(v)) for v in (_source(source), shards, per_cell, failed, seed, int(final_only)))
    return json.loads(gzip.decompress(base64.b64decode(run(pod, f"python3 - {args}", SELECT.encode()))))


def stream_tar(pod: str, root: str, files: list[str], dest: Path, timeout: float = 3600) -> None:
    """Extract ``tar -c`` of ``files`` (relative to ``root`` on ``pod``) into ``dest``; a stream
    running longer than ``timeout`` seconds is killed (and fails)."""
    proc = subprocess.Popen(["kubectl", "-n", k8s.NAMESPACE, "exec", "-i", pod, "--", "sh", "-c",
                             f"cd {shlex.quote(root)} && tar -cf - -T -"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timer = threading.Timer(timeout, proc.kill)
    timer.start()
    try:
        proc.stdin.write("".join(f + "\n" for f in files).encode())
        proc.stdin.close()
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            tar.extractall(dest, filter="data")
        err = proc.stderr.read()
    finally:
        timer.cancel()
    if proc.wait():
        raise RuntimeError(f"{pod}: tar exit {proc.returncode}: {err.decode(errors='replace')[-2000:]}")


def batches(files: list[str], sizes: dict, limit: float) -> list[list[str]]:
    """Consecutive groups of files of at most ``limit`` bytes (a larger file is a group of its own)."""
    out, size = [[]], 0
    for f in files:
        if out[-1] and size + sizes[f] > limit:
            out.append([])
            size = 0
        out[-1].append(f)
        size += sizes[f]
    return out


def _stage(dest: Path, plan: dict) -> Path:
    """``<dest>.partial`` for this plan: created, or resumed when it holds the same plan."""
    stage = dest.with_name(f"{dest.name}.partial")
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    if stage.exists():
        held = json.loads((stage / PLAN).read_text()) if (stage / PLAN).exists() else None
        if held != {"sha256": digest}:
            raise FileExistsError(f"{stage} holds another transfer; not resumed")
    else:
        stage.mkdir()
        (stage / PLAN).write_text(json.dumps({"sha256": digest}))
    return stage


def transfer(root: str, files: list[str], sizes: dict, stage: Path, pods: list[str], *, run=None, stream=None,
             batch_bytes: float = 500e6, workers: int | None = None, attempts: int = 4, log=None) -> dict:
    """Copy ``files`` (relative to ``root`` on the pods) into ``stage``; returns their manifest
    ``{rel: {"sha256", "bytes"}}``. Tar streams of at most ``batch_bytes`` run ``workers`` at a time
    (default: one per pod), stream ``i`` on ``pods[i % len(pods)]``. Per stream: the pod-side
    ``sha256sum`` of its files, the stream, and every extracted file checked against it; a failed
    stream is retried up to ``attempts`` times, each on the next pod. Finished streams are appended
    to ``stage/.streams.jsonl`` and skipped when the transfer is resumed. No stream starts if the
    local free space would fall below the reserve."""
    run, stream = run or _run, stream or stream_tar
    stage = Path(stage)
    manifest = {}
    if (stage / DONE).exists():
        for line in (stage / DONE).read_text().splitlines():
            if line.strip():
                manifest.update(json.loads(line))
    groups = [g for g in batches(files, sizes, batch_bytes) if not all(f in manifest for f in g)]
    lock, t0 = threading.Lock(), time.time()
    progress = {"streams": 0, "bytes": 0}

    def one(index, group):
        size = sum(sizes[f] for f in group)
        for attempt in range(attempts):
            pod = pods[(index + attempt) % len(pods)]
            if shutil.disk_usage(stage).free - size < RESERVE:
                raise OSError(f"local free space would fall below the {RESERVE / 1e9:.0f} GB reserve; stopped")
            try:
                remote = {}
                for line in run(pod, HASH.format(root=shlex.quote(root)), "\n".join(group).encode()).splitlines():
                    digest, rel = line.split(None, 1)
                    remote[rel] = digest
                stream(pod, root, group, stage)
                entries = {}
                for rel in group:
                    digest = sha256(stage / rel)
                    if digest != remote.get(rel):
                        raise RuntimeError(f"{rel}: local sha256 {digest} != pod {remote.get(rel)}")
                    entries[rel] = {"sha256": digest, "bytes": (stage / rel).stat().st_size}
                break
            except (tarfile.TarError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
                if attempt == attempts - 1:
                    raise RuntimeError(f"stream {index} failed {attempts} times; last on {pod}: {error}") from error
        with lock:
            with (stage / DONE).open("a") as f:
                f.write(json.dumps(entries) + "\n")
            manifest.update(entries)
            progress["streams"] += 1
            progress["bytes"] += size
            if log:
                rate = progress["bytes"] / max(time.time() - t0, 1e-9)
                log(f"{time.strftime('%H:%M:%S')} streams {progress['streams']}/{len(groups)} "
                    f"{progress['bytes'] / 1e9:.1f} GB {rate / 1e6:.0f} MB/s")

    with ThreadPoolExecutor(workers or len(pods)) as pool:
        futures = [pool.submit(one, i, g) for i, g in enumerate(groups)]
        try:
            for future in futures:
                future.result()
        except BaseException:
            pool.shutdown(cancel_futures=True)  # streams already running finish and are recorded
            raise
    missing = [f for f in files if f not in manifest]
    if missing:
        raise RuntimeError(f"{len(missing)} files were not transferred, e.g. {missing[:3]}")
    return {f: manifest[f] for f in files}


def _finish(stage: Path, dest: Path, manifest: dict, info: dict) -> None:
    (stage / SUMS).write_text(sums_text(manifest))
    (stage / INFO).write_text(json.dumps(info, indent=1))
    for name in (PLAN, DONE, SELECTION):
        (stage / name).unlink(missing_ok=True)
    stage.rename(dest)


def _preflight(dest: Path, total: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = dest.with_name(f"{dest.name}.partial")
    held = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file()) if stage.exists() else 0
    free = shutil.disk_usage(dest.parent).free
    if free - (total - held) < RESERVE:
        raise OSError(f"local free space {free / 1e9:.1f} GB minus {(total - held) / 1e9:.1f} GB would fall below "
                      f"the {RESERVE / 1e9:.0f} GB reserve")


def pull_sample(source: str, dest, pod: str | None = None, *, shards: int = 3, per_cell: int = 10,
                failed: int = 4, seed: int = 0, final_only: bool = False, pods: int = 1, workers: int | None = None,
                batch_bytes: float = 500e6, run=None, stream=None, ready=None, log=None) -> dict:
    """Pull a per-cell sample of the dataset folder ``source`` to ``dest`` (module docstring) over
    ``pods`` worker pods."""
    dest = Path(dest).resolve()
    if dest.exists():
        raise FileExistsError(f"{dest} exists; not overwritten")
    ready = ready or k8s.ready_pods
    names = ready()
    pod = pod or names[0]
    root = _source(source)
    selection = {"shards": shards, "per_cell": per_cell, "failed": failed, "seed": seed, "final_only": final_only}
    saved, sel = dest.with_name(f"{dest.name}.partial") / SELECTION, None
    if saved.exists():  # a resumed transfer keeps its drawn records instead of selecting again
        held = json.loads(gzip.decompress(saved.read_bytes()))
        if held["source"] == root and held["selection"] == selection:
            sel = held["result"]
    if sel is None:
        sel = select(source, pod, **selection, run=(lambda p, s, stdin: run(p, s, stdin)) if run else None)
    if not sel["files"]:
        raise RuntimeError(f"{root}: no episode files referenced by the drawn records")
    _preflight(dest, sel["bytes"])
    stage = _stage(dest, {"source": root, "selection": selection, "files": sel["files"], "sizes": sel["sizes"]})
    if not saved.exists():
        saved.write_bytes(gzip.compress(json.dumps({"source": root, "selection": selection, "result": sel}).encode()))
    used = ([pod] + [p for p in names if p != pod])[:max(1, pods)]
    t0 = time.time()
    manifest = transfer(root, sel["files"], sel["sizes"], stage, used, run=run, stream=stream, workers=workers,
                        batch_bytes=batch_bytes, log=log)
    by_file = {}
    for r in sel["records"]:
        by_file.setdefault(r["record_file"], []).append(r["line"])
    for rel, lines in by_file.items():
        path = stage / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(line + "\n" for line in lines))
        manifest[rel] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    info = {"source": root, "pod": pod, "pods": used, "tool": "cluster.pull", "repo_revision": _git_revision(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "selection": selection,
            "record_files_read": sel["record_files"], "cell_counts_read": sel["cells"],
            "records": len(sel["records"]), "episode_files": len(sel["files"]), "bytes": sel["bytes"],
            "note": "record files hold only the drawn lines; episode files are byte-identical to the PVC"}
    _finish(stage, dest, manifest, info)
    return {"dest": str(dest), "pods": len(used), "records": len(sel["records"]), "episode_files": len(sel["files"]),
            "bytes": sel["bytes"], "seconds": round(time.time() - t0, 1)}


LIST = """set -eu
cd {root}
find . -type f ! -name '.*' -printf '%s\\t%P\\n'
"""


def pull_all(source: str, dest, pod: str | None = None, *, run=None, stream=None, batch_bytes: float = 200e6,
             attempts: int = 3, pods: int = 1, workers: int | None = None, ready=None, log=None) -> dict:
    """Pull every file of the PVC folder ``source`` to ``dest`` (small folders, e.g. exported
    scenes): same preflight, pod-side hashes and staging as :func:`pull_sample`. Files go in tar
    streams of at most ``batch_bytes``; a stream cut short (kubectl exec drops long transfers) is
    retried up to ``attempts`` times."""
    run = run or _run
    dest = Path(dest).resolve()
    if dest.exists():
        raise FileExistsError(f"{dest} exists; not overwritten")
    names = (ready or k8s.ready_pods)() if pods > 1 or pod is None else [pod]
    pod = pod or names[0]
    root = _source(source)
    sizes = {}
    for line in run(pod, LIST.format(root=shlex.quote(root))).splitlines():
        size, rel = line.split("\t", 1)
        sizes[rel] = int(size)
    if not sizes:
        raise RuntimeError(f"{root}: no files")
    files = sorted(sizes)
    total = sum(sizes.values())
    _preflight(dest, total)
    stage = _stage(dest, {"source": root, "all": True, "sizes": sizes})
    used = ([pod] + [p for p in names if p != pod])[:max(1, pods)]
    t0 = time.time()
    manifest = transfer(root, files, sizes, stage, used, run=run, stream=stream, batch_bytes=batch_bytes,
                        workers=workers, attempts=attempts, log=log)
    info = {"source": root, "pod": pod, "pods": used, "tool": "cluster.pull --all", "repo_revision": _git_revision(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": len(files), "bytes": total,
            "note": "every file is byte-identical to the PVC (sha256 checked against a pod-side hash)"}
    _finish(stage, dest, manifest, info)
    return {"dest": str(dest), "pod": pod, "files": len(files), "bytes": total, "seconds": round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", help="dataset folder below the PVC root, e.g. datasets/tracking-v1")
    ap.add_argument("dest", help="new local folder, e.g. data/pulled/tracking-v1-sample")
    ap.add_argument("--all", action="store_true", help="pull every file of the folder instead of a sample")
    ap.add_argument("--pod", help="worker pod that selects (default: first Ready pod)")
    ap.add_argument("--pods", type=int, default=1, help="worker pods streaming in parallel")
    ap.add_argument("--workers", type=int, help="concurrent streams (default: one per pod)")
    ap.add_argument("--batch-mb", type=float, help="tar stream size, MB (default 500; --all 200)")
    ap.add_argument("--shards", type=int, default=3, help="record files read, spread over the sorted list")
    ap.add_argument("--per-cell", type=int, default=10, help="tier-K passes drawn per cell")
    ap.add_argument("--failed", type=int, default=4, help="tier-K failures drawn per cell")
    ap.add_argument("--final-only", action="store_true", help="only each record's final attempt file")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    log = lambda text: print(text, flush=True)
    batch = {"batch_bytes": a.batch_mb * 1e6} if a.batch_mb else {}
    if a.all:
        print(json.dumps(pull_all(a.source, a.dest, a.pod, pods=a.pods, workers=a.workers, log=log, **batch)))
        return
    print(json.dumps(pull_sample(a.source, a.dest, a.pod, shards=a.shards, per_cell=a.per_cell, failed=a.failed,
                                 seed=a.seed, final_only=a.final_only, pods=a.pods, workers=a.workers, log=log,
                                 **batch)))


if __name__ == "__main__":
    main()
