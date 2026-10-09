"""Tracking build summary: K pass counts per scenario axis, retries, reasons, cost.

``python -m reachy_retarget.tracking.report runs/tracking/pilot [--md out.md] [--json out.json]``
reads ``records/tracking/*.jsonl``. Counts are per scenario (or source episode); a scenario passes
when its last attempt passes tier K. Reasons are bucketed by :func:`reachy_retarget.report.reason_kind`.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..report import reason_kind

AXES = ("cell", "neck", "length", "speed", "start")


def load(roots):
    recs = []
    for root in roots:
        for p in sorted(Path(root).glob("records/tracking/*.jsonl")):
            recs += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return recs


def summarize(recs) -> dict:
    out = {"total": len(recs), "K": 0, "errors": 0, "by": {a: defaultdict(Counter) for a in AXES + ("source",)},
           "attempts": Counter(), "reasons_first": Counter(), "reasons_any": Counter(), "prescreen": 0,
           "seconds": 0.0, "duration_s": 0.0, "K_duration_s": 0.0, "files": 0, "bytes": 0}
    for r in recs:
        ok = bool((r.get("K") or {}).get("passed"))
        out["K"] += ok
        out["errors"] += r.get("status") in ("error", "read_error")
        sc = r.get("scenario")
        keys = {a: (sc or {}).get(a) for a in AXES} if sc else {"source": r.get("source_family")}
        for a, v in keys.items():
            if v is not None:
                out["by"][a][str(v)]["n"] += 1
                out["by"][a][str(v)]["K"] += ok
        out["attempts"][len(r.get("attempts") or [1])] += 1
        out["prescreen"] += len(r.get("prescreen") or [])
        kinds = [reason_kind(x) for x in (r.get("K") or {}).get("reasons") or []]
        if kinds and not ok:
            out["reasons_first"][kinds[0]] += 1
            out["reasons_any"].update(set(kinds))
        out["seconds"] += r.get("seconds") or 0.0
        out["duration_s"] += r.get("duration") or 0.0
        out["K_duration_s"] += (r.get("duration") or 0.0) if ok else 0.0
        for a in r.get("attempts") or ([{"file": r.get("file")}] if r.get("file") else []):
            f = a.get("file")
            if f and Path(f).exists():
                out["files"] += 1
                out["bytes"] += Path(f).stat().st_size
    out["by"] = {a: {k: dict(v) for k, v in sorted(c.items())} for a, c in out["by"].items() if c}
    for k in ("attempts", "reasons_first", "reasons_any"):
        out[k] = dict(out[k].most_common())
    return out


def markdown(s) -> str:
    lines = ["# Tracking build", "",
             f"Scenarios {s['total']}, tier K passed {s['K']} ({100 * s['K'] / max(s['total'], 1):.1f} %), "
             f"errors {s['errors']}.",
             f"Duration: {s['duration_s'] / 3600:.2f} h tracked, {s['K_duration_s'] / 3600:.2f} h passing K; "
             f"compute {s['seconds'] / 3600:.2f} CPU-h; {s['files']} files, "
             f"{s['bytes'] / max(s['files'], 1) / 1e6:.2f} MB per file.",
             f"IK attempts per scenario: {s['attempts']}; pre-screen regenerations: {s['prescreen']}.", ""]
    for axis, rows in s["by"].items():
        lines += [f"## By {axis}", "", "| value | n | K | K % |", "|---|---|---|---|"]
        lines += [f"| {k} | {v.get('n', 0)} | {v.get('K', 0)} | {100 * v.get('K', 0) / max(v.get('n', 1), 1):.0f} |"
                  for k, v in rows.items()]
        lines.append("")
    lines += ["## Failure reasons (first / any)", ""]
    lines += [f"* {k}: {s['reasons_first'].get(k, 0)} / {v}" for k, v in s["reasons_any"].items()]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    s = summarize(load(a.roots))
    text = markdown(s)
    if a.md:
        Path(a.md).write_text(text)
    if a.json:
        Path(a.json).write_text(json.dumps(s, indent=1))
    print(text)


if __name__ == "__main__":
    main()
