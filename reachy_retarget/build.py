"""Build Reachy episodes from one source file shard (the unit of work of a cluster job).

``python -m reachy_retarget.build --family robomimic --path FILE --shard 3/8 --physics --out DIR``
retargets every ``k``-th episode of the file (``index % n == k``), runs tier K and, when
the source carries a scene, tier P, and writes:

* ``DIR/episodes/<dataset>/<episode_id>.h5`` for every produced trajectory (K or P may fail;
  failed retargets are data too), and
* ``DIR/records/<family>/<path below raw/>.<k>-of-<n>.jsonl`` with one record per source episode,
  including errors.

``--path`` may be repeated: the paths are processed one after the other with the same shard
(cluster jobs pack several small source files into one job; an unreadable file ends only its
own record file). Each record carries ``rss_mb`` (resident memory right after the episode)
  and ``max_rss_mb`` (the process peak so far), so memory growth over a shard is visible.

Memory (cluster pods run two jobs in 4 GiB): only this shard's episodes are read (the shard is
passed to the adapter as ``select``, see :mod:`.sources.registry`, so adapters that support it
never load, compile or replay the other episodes); each episode is released and freed heap
pages are returned to the OS before the next one is read; MuJoCo's compiler asset cache is
off unless ``--mujoco-cache-mb`` is given.

The dataset index is built afterwards over the merged output (``reachy-retarget index``).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

from .evaluate import process_source
from .sources import iter_episodes

_MACOS = sys.platform == "darwin"  # ru_maxrss is bytes on macOS, KiB on Linux


def max_rss_mb() -> float:
    """Peak resident set size of this process (MiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20 if _MACOS else 1 << 10)


def rss_mb() -> float | None:
    """Current resident set size (MiB): ``/proc/self/statm`` on Linux, ``ps`` elsewhere;
    None when unavailable."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / (1 << 20)
    except (OSError, ValueError, IndexError):
        pass
    try:
        import subprocess
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True,
                             timeout=5).stdout
        return int(out.strip()) / 1024  # KiB
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def record_stem(path: str) -> str:
    """A file-unique record name: the path below ``raw/`` (or the full path) with '/' -> '__'."""
    parts = Path(path).parts
    tail = parts[parts.index("raw") + 1:] if "raw" in parts else parts[-4:]
    return "__".join(tail)


def release_memory() -> None:
    """Return freed heap pages to the OS (glibc ``malloc_trim``, macOS
    ``malloc_zone_pressure_relief``; a no-op elsewhere).

    MuJoCo compiles allocate and free hundreds of MB per episode; the allocator keeps freed
    pages, so without this the resident size between episodes stays near the previous peak
    and the next compile adds to it."""
    import ctypes
    try:
        if sys.platform.startswith("linux"):
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        elif _MACOS:
            ctypes.CDLL("libSystem.dylib").malloc_zone_pressure_relief(None, 0)
    except (OSError, AttributeError):
        pass


def limit_mujoco_cache(mb: float) -> None:
    """Cap MuJoCo's process-wide compiler asset cache (decoded meshes and textures; 500 MB by
    default). It only saves decoding time when a later compile uses the same asset files and
    never changes a compiled model, but a full cache stays resident for the whole job."""
    try:
        import mujoco
    except ImportError:
        return
    mujoco.mj_setCacheCapacity(mujoco.mj_getCache(), int(mb * (1 << 20)))


def build(family: str, path, shard: tuple[int, int], out: str, *, physics: bool = True,
          mujoco_cache_mb: float = 0, **source_kw) -> dict:
    """Build one shard of ``path`` (a source path, or a list of them processed in order)."""
    k, n = shard
    if not 0 <= k < n:
        raise ValueError("shard must be k/n with 0 <= k < n")
    limit_mujoco_cache(mujoco_cache_mb)
    records_dir = Path(out) / "records" / family
    records_dir.mkdir(parents=True, exist_ok=True)
    counts = {"episodes": 0, "errors": 0, "K": 0, "P": 0, "P_tested": 0}
    for p in [path] if isinstance(path, (str, os.PathLike)) else list(path):
        _build_path(family, str(p), (k, n), out, records_dir, counts, physics, source_kw)
    counts["max_rss_mb"] = max_rss_mb()
    return counts


def _build_path(family, path, shard, out, records_dir, counts, physics, source_kw) -> None:
    k, n = shard
    stream = iter_episodes(family, path, select=lambda i: i % n == k, **source_kw)
    with (records_dir / f"{record_stem(path)}.{k}-of-{n}.jsonl").open("w") as fh:
        index = 0
        while True:
            t0 = time.perf_counter()
            try:
                src = next(stream)
            except StopIteration:
                break
            except Exception as error:  # an unreadable episode ends this shard; record it
                fh.write(json.dumps({"family": family, "path": path, "index": index, "status": "read_error",
                                     "error": repr(error), "traceback": traceback.format_exc()[-2000:]}) + "\n")
                counts["errors"] += 1
                break
            index += 1
            if src is None:  # another shard's episode, not read
                continue
            try:
                rec = process_source(src, physics=physics, write=str(Path(out) / "episodes"), layout="dataset",
                                     read_seconds=time.perf_counter() - t0)
            except Exception as error:  # keep going; the failure is a record
                rec = {"family": family, "dataset": src.dataset, "episode_id": src.episode_id, "status": "error",
                       "error": repr(error), "traceback": traceback.format_exc()[-2000:]}
                counts["errors"] += 1
            src = None  # release the episode (scene, asset bytes) before the next one is read
            release_memory()
            rec.update(path=path, index=index - 1, rss_mb=rss_mb(), max_rss_mb=max_rss_mb())
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            counts["episodes"] += 1
            counts["K"] += bool((rec.get("K") or {}).get("passed"))
            counts["P_tested"] += rec.get("P") is not None
            counts["P"] += bool((rec.get("P") or {}).get("passed"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", required=True)
    ap.add_argument("--path", required=True, action="append", help="source path (repeatable)")
    ap.add_argument("--shard", default="0/1", help="k/n: take episodes with index %% n == k")
    ap.add_argument("--physics", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mujoco-cache-mb", type=float, default=0,
                    help="MuJoCo compiler asset cache capacity (default 0: off, lowest memory)")
    a = ap.parse_args(argv)
    k, n = (int(x) for x in a.shard.split("/"))
    print(json.dumps(build(a.family, a.path if len(a.path) > 1 else a.path[0], (k, n), a.out, physics=a.physics, mujoco_cache_mb=a.mujoco_cache_mb)))


if __name__ == "__main__":
    main()
