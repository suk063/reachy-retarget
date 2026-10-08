"""Aggregate build records into a results table (markdown + JSON).

``python -m reachy_retarget.report <build output root>... [--out results.json] [--md results.md]``
reads every ``records/<family>/*.jsonl`` under the given roots. Counts are reported
separately for every stage; K (kinematic) and P (physics) passes are never merged.
Independent demonstrations are counted by lineage seed when an episode is a generated
variant, so generated data never inflates the number of distinct human demonstrations.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

STAGES = ("source", "retargeted", "K", "P_tested", "P", "K_and_P", "errors")


def load_records(roots) -> list[dict]:
    records = []
    for root in roots:
        for path in sorted(Path(root).glob("records/*/*.jsonl")):
            records += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return records


def _stage_flags(rec: dict) -> dict:
    k = (rec.get("K") or {}).get("passed", False)
    p = rec.get("P")
    return {"source": True, "retargeted": rec.get("status") == "ok", "K": bool(k), "P_tested": p is not None,
            "P": bool(p and p.get("passed")), "K_and_P": bool(k and p and p.get("passed")),
            "errors": rec.get("status") in ("error", "read_error")}


def summarize(records: list[dict]) -> dict:
    by_dataset = defaultdict(Counter)
    by_family = defaultdict(Counter)
    body_parts, regimes, reasons = Counter(), Counter(), Counter()
    for rec in records:
        flags = _stage_flags(rec)
        for stage, value in flags.items():
            by_dataset[rec.get("dataset", "?")][stage] += value
            by_family[rec.get("family", "?")][stage] += value
        if flags["K"]:
            body_parts[" + ".join(sorted(rec.get("body_parts") or []))] += 1
            regimes[rec.get("regime") or "unknown"] += 1
        for reason in ((rec.get("K") or {}).get("reasons") or [])[:1] + ((rec.get("P") or {}).get("reasons") or [])[:1]:
            reasons[reason.split(":")[0].split(" ")[0] if ":" in reason else reason[:40]] += 1
    totals = Counter()
    for counts in by_family.values():
        totals.update(counts)
    return {"totals": dict(totals), "families": {k: dict(v) for k, v in sorted(by_family.items())},
            "datasets": {k: dict(v) for k, v in sorted(by_dataset.items())},
            "body_parts_K": dict(body_parts.most_common()), "regimes_K": dict(regimes.most_common()),
            "top_failure_reasons": dict(reasons.most_common(25))}


def markdown(summary: dict) -> str:
    def table(rows: dict) -> list[str]:
        out = ["| name | " + " | ".join(STAGES) + " |", "|---" * (len(STAGES) + 1) + "|"]
        out += [f"| {name} | " + " | ".join(str(c.get(s, 0)) for s in STAGES) + " |" for name, c in rows.items()]
        return out

    lines = ["# Build results", "", "## Families", ""] + table(summary["families"])
    lines += ["", f"Total: " + ", ".join(f"{s} {summary['totals'].get(s, 0)}" for s in STAGES), "",
              "## Datasets", ""] + table(summary["datasets"])
    lines += ["", "## Body parts used (K passes)", ""]
    lines += [f"* {k or 'none'}: {v}" for k, v in summary["body_parts_K"].items()]
    lines += ["", "## Most common first failure reasons", ""]
    lines += [f"* {k}: {v}" for k, v in summary["top_failure_reasons"].items()]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", help="write the JSON summary here")
    ap.add_argument("--md", help="write the markdown table here")
    a = ap.parse_args(argv)
    summary = summarize(load_records(a.roots))
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=1))
    text = markdown(summary)
    if a.md:
        Path(a.md).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
