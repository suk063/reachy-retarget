"""Evaluate retargeting quality on local source files: tier K and (optionally) tier P per episode.

    python -m reachy_retarget.evaluate --family robomimic \\
        --path data/raw/robomimic/v1.5/can/ph/low_dim_v15.hdf5 --demos 0-19 --physics \\
        --out runs/eval/can-dev.jsonl [--write DIR] [--jobs 4]

Each episode is read through its source adapter, retargeted (:func:`retarget.retarget`, which
runs tier K) and, with ``--physics`` and a source scene, simulated (:func:`validate.physics.simulate`).
One JSON line per episode goes to ``--out`` (K/P pass flags, reasons, key metrics, timing); an
aggregate table is printed. With ``--write DIR`` every episode (failures included) is written
with :func:`schema.io.write_episode` to ``DIR/<task>/<episode_id>.h5`` plus ``DIR/index.parquet``.
Nothing here touches the network: only local files are read.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def parse_demos(spec: str | None, prefix: str = "demo_") -> list[str] | None:
    """``"0-19,25"`` -> ``["demo_0", ..., "demo_19", "demo_25"]``; None/"all" -> None."""
    if not spec or spec == "all":
        return None
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            out += [f"{prefix}{i}" for i in range(a, b + 1)]
        else:
            out.append(part if not part.isdigit() else f"{prefix}{part}")
    return out


def _short(metrics: dict, keys) -> dict:
    return {k: metrics[k] for k in keys if k in metrics}


def evaluate_episode(family: str, path: str, demo: str, *, physics: bool = True, write: str | None = None,
                     cfg=None, physics_cfg=None) -> dict:
    """Retarget one demo (+ tier P); returns a JSON-compatible record."""
    from .sources import iter_episodes

    t0 = time.perf_counter()
    src = next(iter_episodes(family, path, demos=[demo]))
    rec = process_source(src, physics=physics, write=write, cfg=cfg, physics_cfg=physics_cfg,
                         read_seconds=time.perf_counter() - t0)
    rec.update(path=str(path), demo=demo)
    return rec


def process_source(src, *, physics: bool = True, write: str | None = None, cfg=None, physics_cfg=None,
                   read_seconds: float = 0.0, layout: str = "task") -> dict:
    """Retarget one SourceEpisode, run tier K (and P when it has a scene); optionally write it.

    ``layout="task"`` writes ``<write>/<task>/<episode>.h5`` (evaluation runs);
    ``layout="dataset"`` writes ``<write>/<dataset>/<episode>.h5`` (unique across files).
    """
    from .retarget import retarget
    from .retarget.pipeline import _jsonable

    t0 = time.perf_counter()
    t_read = read_seconds
    rec = {"family": src.family, "task": src.task, "dataset": src.dataset, "episode_id": src.episode_id,
           "source_frames": src.length, "source_success": src.success}
    res = retarget(src, cfg)
    t_ret = time.perf_counter() - t0
    rec["status"] = res.status
    if res.episode is None:
        rec.update(K={"passed": False, "reasons": res.reasons}, P=None,
                   seconds={"read": t_read, "retarget": t_ret})
        return rec
    ep = res.episode
    rec["K"] = {"passed": ep.tier["K"]["passed"], "reasons": ep.tier["K"]["reasons"],
                "metrics": _short(ep.extra.get("tier_k_metrics", {}),
                                  ("left_max_pos_residual", "left_max_rot_residual", "right_max_pos_residual",
                                   "right_max_rot_residual", "max_speed_ratio", "min_self_clearance",
                                   "max_grasp_drift_pos", "max_grasp_drift_rot"))}
    rec["retarget"] = {k: res.diagnostics.get(k) for k in ("assignment", "timing", "seconds")}
    rec["retarget"]["placement"] = {k: res.diagnostics.get("placement", {}).get(k)
                                    for k in ("mobile", "pose", "offset", "flips", "cost", "grasp_offsets")}
    rec["body_parts"] = list(ep.body_parts)
    rec["output_frames"] = ep.length
    rec["P"] = None
    t_phys = 0.0
    if physics and src.scene is not None:
        from .validate import physics as P
        t1 = time.perf_counter()
        tier, rollout = P.simulate(ep, src.scene, physics_cfg)
        t_phys = time.perf_counter() - t1
        m = tier.get("metrics", {})
        rec["P"] = {"passed": tier["passed"], "reasons": tier["reasons"],
                    "metrics": _jsonable(_short(m, ("max_depth_m", "worst_contacts", "peak_arm_speed_rad_s",
                                                    "object_environment_reference_depth_m",
                                                    "object_environment_threshold_m",
                                                    "object_environment_passed_absolute",
                                                    "tcp_position_error_max_m", "tcp_rotation_error_max_rad",
                                                    "grasps", "task")))}
        ep.tier = {"K": ep.tier["K"], "P": {"passed": tier["passed"], "reasons": tier["reasons"]}}
        ep.physics = rollout
    rec["seconds"] = {"read": t_read, "retarget": t_ret, "physics": t_phys}
    if write:
        from .schema.io import index_row, write_episode
        out = Path(write) / (src.task if layout == "task" else src.dataset) / f"{src.episode_id}.h5"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_episode(out, ep)
        rec["file"] = str(out)
        rec["index_row"] = _jsonable(index_row(ep, str(out.relative_to(write))))
    return _jsonable(rec)


def _safe(args):
    family, path, demo, physics, write = args
    try:
        return evaluate_episode(family, path, demo, physics=physics, write=write)
    except Exception as e:  # keep the batch going; the failure is recorded
        import traceback
        return {"family": family, "path": str(path), "demo": demo, "status": "error",
                "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:],
                "K": {"passed": False, "reasons": [f"error: {e}"]}, "P": None}


def _demo_keys(family, path, demos):
    if demos is not None:
        return demos
    import h5py
    with h5py.File(path, "r") as f:
        return sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))


def _reason_key(reason: str) -> str:
    return reason.split(":")[0] if ":" in reason[:40] else " ".join(reason.split()[:3])


def summarize(records) -> str:
    """Aggregate table: per task, K pass, P pass (of all and of K-passing), top failure reasons."""
    from collections import Counter, defaultdict
    by = defaultdict(list)
    for r in records:
        by[r.get("task", "?")].append(r)
    lines = [f"{'task':<22}{'n':>4}{'K':>6}{'P':>6}{'K&P':>6}{'s/ep':>8}"]
    for task, rs in sorted(by.items()):
        k = sum(bool(r["K"]["passed"]) for r in rs)
        p = sum(bool(r.get("P") and r["P"]["passed"]) for r in rs)
        kp = sum(bool(r["K"]["passed"] and r.get("P") and r["P"]["passed"]) for r in rs)
        sec = sum(sum((r.get("seconds") or {}).values()) for r in rs) / len(rs)
        lines.append(f"{task:<22}{len(rs):>4}{k:>6}{p:>6}{kp:>6}{sec:>8.1f}")
        kc, pc = Counter(), Counter()
        for r in rs:
            kc.update({_reason_key(x) for x in r["K"]["reasons"]})
            if r.get("P"):
                pc.update({_reason_key(x) for x in r["P"]["reasons"]})
        if kc:
            lines.append("    K: " + ", ".join(f"{k} x{v}" for k, v in kc.most_common(6)))
        if pc:
            lines.append("    P: " + ", ".join(f"{k} x{v}" for k, v in pc.most_common(8)))
    return "\n".join(lines)


def run(family, path, demos=None, *, physics=True, out=None, write=None, jobs=1, verbose=True):
    """Evaluate demos of one file; returns the records (also appended as JSON lines to ``out``)."""
    keys = _demo_keys(family, path, demos)
    tasks = [(family, str(path), d, physics, write) for d in keys]
    records = []
    fh = None
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fh = open(out, "w")
    try:
        it = map(_safe, tasks) if jobs <= 1 else ProcessPoolExecutor(jobs).map(_safe, tasks)
        for r in it:
            records.append(r)
            if fh:
                fh.write(json.dumps(r) + "\n")
                fh.flush()
            if verbose:
                k, p = r["K"], r.get("P")
                ps = "-" if p is None else ("PASS" if p["passed"] else "fail")
                sec = sum((r.get("seconds") or {}).values())
                print(f"{r.get('task', '?')} {r['demo']}: K {'PASS' if k['passed'] else 'fail'} P {ps} "
                      f"({sec:.0f} s)", flush=True)
                for x in k["reasons"][:4]:
                    print(f"    K {x}", flush=True)
                for x in (p or {}).get("reasons", [])[:6]:
                    print(f"    P {x}", flush=True)
                if r.get("status") == "error":
                    print(r.get("traceback", ""), flush=True)
    finally:
        if fh:
            fh.close()
    if write:
        from .schema.io import write_index
        rows = [r["index_row"] for r in records if r.get("index_row")]
        if rows:
            write_index(write, rows)
    if verbose:
        print(summarize(records))
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", required=True)
    ap.add_argument("--path", required=True, nargs="+")
    ap.add_argument("--demos", default=None, help="e.g. 0-19,25 (default: all)")
    ap.add_argument("--physics", action="store_true")
    ap.add_argument("--out", default=None, help="JSON-lines output")
    ap.add_argument("--write", default=None, help="write episodes (HDF5) into this directory")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args(argv)
    demos = parse_demos(a.demos)
    records = []
    for p in a.path:
        records += run(a.family, p, demos, physics=a.physics, out=None if len(a.path) > 1 else a.out,
                       write=a.write, jobs=a.jobs, verbose=len(a.path) == 1)
    if len(a.path) > 1:
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in records))
        print(summarize(records))
    return 0 if records else 1


if __name__ == "__main__":
    sys.exit(main())
