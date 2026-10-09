"""Build tracking episodes (docs/tracking.md, "Building").

Synthetic scenarios (indices ``start .. start + count``, shard ``k/n`` takes ``index % n == k``)::

    python -m reachy_retarget.tracking.build synthetic --namespace tracking-v1 --split pilot \\
        --count 300 --jobs 24 --out runs/tracking/pilot

Source hand paths (one source file, shard ``k/n`` over its episodes)::

    python -m reachy_retarget.tracking.build source --family robomimic --path FILE --demos 0-9 \\
        --out runs/tracking/source-pilot

Index (``index.parquet`` from the records' ``index_rows``)::

    python -m reachy_retarget.tracking.build index runs/tracking/pilot

Layout (as :mod:`reachy_retarget.build`): ``DIR/episodes/<dataset>/<episode_id>.h5`` for every
trajectory (K failures included), ``DIR/records/tracking/<stem>.<k>-of-<n>.jsonl`` with one record
per scenario or source episode (errors included), read by :mod:`reachy_retarget.report` and
:mod:`.report`.

Synthetic retry policy: a reference that fails the reachability pre-screen is regenerated with its
motions shrunk (``size`` x 0.75, at most 4 times, no IK). A reference that fails tier K is
regenerated at most twice: slower (``time`` x 1.5) when the speed-boxed IK was at its speed limit
where tracking was first lost and it has not been slowed yet, else smaller (``size`` x 0.75). Every IK attempt is written; earlier ones carry
``variant_of`` = the uid of the last attempt and share its lineage seed.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..build import max_rss_mb, record_stem, release_memory, rss_mb
from ..robot import VELOCITY
from ..schema.io import index_row, write_episode, write_index
from .config import TrackingConfig
from .reference import jsonable

PRESCREEN_TRIES = 4
IK_ATTEMPTS = 3
SHRINK, SLOWER = 0.75, 1.5
KEYS = ("left_max_pos_residual", "left_max_rot_residual", "right_max_pos_residual", "right_max_rot_residual",
        "head_max_rot_residual", "max_speed_ratio", "min_self_clearance", "max_base_deviation_m",
        "max_base_yaw_deviation_rad", "left_unreachable_frames", "right_unreachable_frames",
        "peak_base_speed_body", "base_exceeds_executor_limits")


def _short(metrics):
    return {k: metrics[k] for k in KEYS if k in metrics}


def _write(ep, out_root):
    path = Path(out_root) / "episodes" / ep.dataset / f"{ep.episode_id}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_episode(path, ep)
    return path, jsonable(index_row(ep, str(path.relative_to(Path(out_root) / "episodes"))))


def speed_saturated(ep, frames, scale) -> bool:
    """Whether the IK moved a joint at (almost) its speed box when tracking was first lost: at the
    onset of each failing run (the 5 rows before it). Saturation after a target was lost is the IK
    catching up, not the cause."""
    if not len(frames):
        return False
    dq = np.abs(np.diff(ep.q, axis=0)) / np.diff(ep.time)[:, None] / VELOCITY
    ratio = np.r_[dq.max(axis=1), 0.0]
    onsets = frames[np.r_[True, np.diff(frames) > 1]]
    window = np.unique(np.clip(np.concatenate([onsets + d for d in range(-5, 1)]), 0, len(ratio) - 1))
    return bool(ratio[window].max() > 0.97 * scale)


def failing_frames(ep, cfg: TrackingConfig) -> np.ndarray:
    v = ep.validation
    bad = np.zeros(ep.length, bool)
    for name, tol in (("tcp_pos_residual", cfg.tcp_pos_tol), ("tcp_rot_residual", cfg.tcp_rot_tol)):
        if name in v:
            bad |= np.nan_to_num(np.asarray(v[name]), nan=0.0).max(axis=1) > tol
    if "head_rot_residual" in v:
        bad |= np.asarray(v["head_rot_residual"]) > cfg.head_rot_tol
    if "self_clearance" in v:
        bad |= np.asarray(v["self_clearance"]) < cfg.min_self_clearance
    return np.flatnonzero(bad)


def run_scenario(namespace, split, index, out, cfg: TrackingConfig | None = None) -> dict:
    """Generate, track and write one synthetic scenario (with retries); returns its record."""
    from . import synthetic as S
    from .pipeline import track

    cfg = cfg or TrackingConfig()
    t0 = time.perf_counter()
    sc = S.scenario(namespace, split, index)
    rec = {"family": "tracking", "generator": S.GENERATOR, "dataset": S.DATASET, "episode_id": sc.episode_id,
           "task": sc.cell, "scenario": jsonable(sc.__dict__), "attempts": [], "prescreen": []}
    attempt = S.Attempt()
    results = []
    for k in range(IK_ATTEMPTS):
        ref = None
        for _ in range(PRESCREEN_TRIES + 1):
            ref = S.generate(sc, attempt)
            reasons = S.prescreen(ref)
            if not reasons:
                break
            rec["prescreen"].append({"attempt": jsonable(attempt.__dict__), "reasons": reasons})
            attempt = replace(attempt, size=attempt.size * SHRINK)
        res = track(ref, cfg)
        results.append((attempt, res))
        ep = res.episode
        if res.status != "ok" or ep is None or ep.tier["K"]["passed"] or k == IK_ATTEMPTS - 1:
            break
        frames = failing_frames(ep, cfg)
        slowed = attempt.time > 1.0
        if speed_saturated(ep, frames, cfg.velocity_scale) and not slowed:
            attempt = replace(attempt, time=attempt.time * SLOWER)
        else:
            attempt = replace(attempt, size=attempt.size * SHRINK)
    final_attempt, final = results[-1]
    final_uid = None if final.episode is None else final.episode.uid
    rows = []
    for k, (att, res) in enumerate(results):
        ep = res.episode
        entry = {"attempt": jsonable(att.__dict__), "status": res.status}
        if ep is not None:
            if k < len(results) - 1:
                ep.episode_id = f"{sc.episode_id}-try{k}"
                ep.uid = f"{ep.dataset}/{ep.episode_id}"
                ep.variant_of = final_uid
                ep.lineage = dict(ep.lineage, attempt=k)
            else:
                ep.lineage = dict(ep.lineage, attempt=k)
            path, row = _write(ep, out)
            rows.append(row)
            entry.update(K={"passed": ep.tier["K"]["passed"], "reasons": ep.tier["K"]["reasons"]},
                         file=str(path))
        else:
            entry.update(K={"passed": False, "reasons": res.reasons})
        rec["attempts"].append(entry)
    ep = final.episode
    rec["status"] = final.status
    if ep is None:
        rec.update(K={"passed": False, "reasons": final.reasons}, P=None)
    else:
        rec.update(K={"passed": ep.tier["K"]["passed"], "reasons": ep.tier["K"]["reasons"],
                      "metrics": _short(ep.extra.get("tier_k_metrics", {}))},
                   P=None, regime=ep.regime, body_parts=list(ep.body_parts), output_frames=ep.length,
                   duration=ep.duration, uid=ep.uid, file=rec["attempts"][-1].get("file"))
    rec["index_rows"] = rows
    rec["seconds"] = time.perf_counter() - t0
    return jsonable(rec)


def _scenario_job(args):
    namespace, split, index, out = args
    try:
        rec = run_scenario(namespace, split, index, out)
    except Exception as error:  # the failure is a record; the batch goes on
        rec = {"family": "tracking", "dataset": "tracking/synthetic-v1", "episode_id": f"{namespace}-{split}-{index:06d}",
               "status": "error", "error": repr(error), "traceback": traceback.format_exc()[-3000:],
               "K": {"passed": False, "reasons": [f"error: {error!r}"]}, "P": None}
    release_memory()
    rec.update(index=index, rss_mb=rss_mb(), max_rss_mb=max_rss_mb())
    return rec


def build_synthetic(namespace, split, start, count, out, shard=(0, 1), jobs=1) -> dict:
    k, n = shard
    indices = [i for i in range(start, start + count) if i % n == k]
    records_dir = Path(out) / "records" / "tracking"
    records_dir.mkdir(parents=True, exist_ok=True)
    name = f"synthetic__{namespace}__{split}__{start}-{start + count}.{k}-of-{n}.jsonl"
    counts = {"scenarios": 0, "K": 0, "errors": 0, "episodes_written": 0}
    args = [(namespace, split, i, out) for i in indices]
    with (records_dir / name).open("w") as fh:
        if jobs > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(jobs, initializer=_single_thread) as pool:
                stream = pool.imap_unordered(_scenario_job, args, chunksize=1)
                _drain(stream, fh, counts)
        else:
            _drain(map(_scenario_job, args), fh, counts)
    counts["max_rss_mb"] = max_rss_mb()
    return counts


def _single_thread():
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def _drain(stream, fh, counts):
    t0 = time.perf_counter()
    for rec in stream:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        counts["scenarios"] += 1
        counts["K"] += bool((rec.get("K") or {}).get("passed"))
        counts["errors"] += rec.get("status") == "error"
        counts["episodes_written"] += len(rec.get("index_rows") or [])
        if counts["scenarios"] % 25 == 0:
            print(f"{counts['scenarios']} scenarios, K {counts['K']}, errors {counts['errors']}, "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)


def build_source(family, path, out, shard=(0, 1), demos=None, both_arms=True, **source_kw) -> dict:
    """Track the hand paths of one source file's episodes (tier K only)."""
    from ..sources import iter_episodes
    from . import source as tracking_source
    from .pipeline import track

    k, n = shard
    select = (lambda i: i % n == k and (demos is None or i in demos))
    records_dir = Path(out) / "records" / "tracking"
    records_dir.mkdir(parents=True, exist_ok=True)
    counts = {"episodes": 0, "K": 0, "errors": 0}
    stream = iter_episodes(family, path, select=select, **source_kw)
    with (records_dir / f"source__{family}__{record_stem(str(path))}.{k}-of-{n}.jsonl").open("w") as fh:
        index = 0
        while True:
            t0 = time.perf_counter()
            try:
                src = next(stream)
            except StopIteration:
                break
            except Exception as error:
                fh.write(json.dumps({"family": "tracking", "source_family": family, "path": str(path),
                                     "index": index, "status": "read_error", "error": repr(error),
                                     "traceback": traceback.format_exc()[-2000:]}) + "\n")
                counts["errors"] += 1
                break
            index += 1
            if src is None:
                continue
            for ref in tracking_source.references(src, both_arms=both_arms):
                rec = {"family": "tracking", "source_family": family, "path": str(path), "index": index - 1,
                       "dataset": ref.dataset, "episode_id": ref.episode_id, "task": ref.task,
                       "variant_of": ref.variant_of}
                try:
                    res = track(ref)
                    rec["status"] = res.status
                    ep = res.episode
                    if ep is None:
                        rec.update(K={"passed": False, "reasons": res.reasons}, P=None)
                    else:
                        file, row = _write(ep, out)
                        rec.update(K={"passed": ep.tier["K"]["passed"], "reasons": ep.tier["K"]["reasons"],
                                      "metrics": _short(ep.extra.get("tier_k_metrics", {}))}, P=None,
                                   regime=ep.regime, body_parts=list(ep.body_parts), output_frames=ep.length,
                                   duration=ep.duration, uid=ep.uid, file=str(file), index_rows=[row],
                                   timing=(res.diagnostics or {}).get("timing"))
                except Exception as error:
                    rec.update(status="error", error=repr(error), traceback=traceback.format_exc()[-3000:],
                               K={"passed": False, "reasons": [f"error: {error!r}"]}, P=None)
                    counts["errors"] += 1
                rec.update(seconds=time.perf_counter() - t0, rss_mb=rss_mb(), max_rss_mb=max_rss_mb())
                fh.write(json.dumps(jsonable(rec)) + "\n")
                fh.flush()
                counts["episodes"] += 1
                counts["K"] += bool(rec["K"]["passed"])
            src = None
            release_memory()
    counts["max_rss_mb"] = max_rss_mb()
    return counts


def build_index(out) -> Path:
    """``DIR/episodes/index.parquet`` from the ``index_rows`` of every record under ``DIR``."""
    rows = []
    for p in sorted(Path(out).glob("records/*/*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows += json.loads(line).get("index_rows") or []
    return write_index(Path(out) / "episodes", rows)


def _range(text):
    out = set()
    for part in text.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("synthetic")
    p.add_argument("--namespace", default="tracking-v1")
    p.add_argument("--split", default="train")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--out", required=True)
    p = sub.add_parser("source")
    p.add_argument("--family", required=True)
    p.add_argument("--path", required=True)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--demos", help="episode positions, e.g. 0-9,20")
    p.add_argument("--one-arm", action="store_true", help="single-arm sources: only the assigned arm")
    p.add_argument("--out", required=True)
    p = sub.add_parser("index")
    p.add_argument("out")
    a = ap.parse_args(argv)
    if a.command == "index":
        print(build_index(a.out))
        return
    k, n = (int(x) for x in a.shard.split("/"))
    if a.command == "synthetic":
        result = build_synthetic(a.namespace, a.split, a.start, a.count, a.out, (k, n), a.jobs)
    else:
        result = build_source(a.family, a.path, a.out, (k, n), _range(a.demos) if a.demos else None,
                              both_arms=not a.one_arm)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
