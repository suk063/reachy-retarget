"""Build Reachy episodes from one source file shard (the unit of work of a cluster job).

``python -m reachy_retarget.build --family robomimic --path FILE --shard 3/8 --physics --out DIR``
retargets every ``k``-th episode of the file (``index % n == k``), runs tier K and, when
the source carries a scene, tier P, and writes:

* ``DIR/episodes/<dataset>/<episode_id>.h5`` for every produced trajectory (K or P may fail;
  failed retargets are data too), and
* ``DIR/records/<family>/<path below raw/>.<k>-of-<n>.jsonl`` with one record per source episode,
  including errors.

The dataset index is built afterwards over the merged output (``reachy-retarget index``).
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from .evaluate import process_source
from .sources import iter_episodes


def record_stem(path: str) -> str:
    """A file-unique record name: the path below ``raw/`` (or the full path) with '/' -> '__'."""
    parts = Path(path).parts
    tail = parts[parts.index("raw") + 1:] if "raw" in parts else parts[-4:]
    return "__".join(tail)


def build(family: str, path: str, shard: tuple[int, int], out: str, *, physics: bool = True, **source_kw) -> dict:
    k, n = shard
    if not 0 <= k < n:
        raise ValueError("shard must be k/n with 0 <= k < n")
    records_dir = Path(out) / "records" / family
    records_dir.mkdir(parents=True, exist_ok=True)
    counts = {"episodes": 0, "errors": 0, "K": 0, "P": 0, "P_tested": 0}
    stream = iter_episodes(family, path, **source_kw)
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
            if (index - 1) % n != k:
                continue
            try:
                rec = process_source(src, physics=physics, write=str(Path(out) / "episodes"), layout="dataset",
                                     read_seconds=time.perf_counter() - t0)
            except Exception as error:  # keep going; the failure is a record
                rec = {"family": family, "dataset": src.dataset, "episode_id": src.episode_id, "status": "error",
                       "error": repr(error), "traceback": traceback.format_exc()[-2000:]}
                counts["errors"] += 1
            rec.update(path=path, index=index - 1)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            counts["episodes"] += 1
            counts["K"] += bool((rec.get("K") or {}).get("passed"))
            counts["P_tested"] += rec.get("P") is not None
            counts["P"] += bool((rec.get("P") or {}).get("passed"))
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--shard", default="0/1", help="k/n: take episodes with index % n == k")
    ap.add_argument("--physics", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    k, n = (int(x) for x in a.shard.split("/"))
    print(json.dumps(build(a.family, a.path, (k, n), a.out, physics=a.physics)))


if __name__ == "__main__":
    main()
