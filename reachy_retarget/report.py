"""Aggregate build records into a results table (markdown + JSON).

``python -m reachy_retarget.report <build output root>... [--out results.json] [--md results.md]``
reads every ``records/<family>/*.jsonl`` under the given roots. Counts are reported
separately for every stage; K (kinematic) and P (physics) passes are never merged.
Independent demonstrations are counted by lineage seed when an episode is a generated
variant, so generated data never inflates the number of distinct human demonstrations.
``excluded`` counts source episodes left out of the dataset (``status: "excluded"``, e.g. ``no_meshes``:
their scene meshes cannot be obtained), ``written`` the episodes stored with their meshes; the
exclusion reasons are listed per kind and the stored size per episode (with and without the asset
library) per dataset. Failure reasons are bucketed by kind (:func:`reason_kind`: the tier-P gate name, or the K reason
text with sides, object names and numbers stripped), separately for K and P, both for the first
reason of each episode and for any reason.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

STAGES = ("source", "excluded", "retargeted", "K", "P_tested", "P", "K_and_P", "written", "errors")


_GATE = re.compile(r"^([a-z][a-z0-9_]*):")
_HAND_OBJECT = re.compile(r"\bhand-\S+")
_SIDE = re.compile(r"\b(left|right)\b\s*")
_EXCL_NUM = re.compile(r"\d+")
_STOP = {"by", "at", "of", "is", "to", "for", "in", "with", "during", "from", "on", "x"}


def reason_kind(reason: str) -> str:
    """Bucket of a K or P failure reason with numbers, sides and object names stripped.

    ``"grasp_drift: right/cubeA: 0.0148 m, 0.0490 rad"`` -> ``"grasp_drift"`` (tier-P gates and other
    ``identifier: details`` reasons such as ``scene:`` or ``error:`` keep the identifier);
    ``"left TCP rotation residual 0.115 rad > 0.05 rad at frame 12 (3 frames)"`` -> ``"TCP rotation
    residual"``; ``"right hand-SquareNut relative pose drifts 18 mm / 0.3 rad during frames 3-9"`` ->
    ``"hand-object relative pose drifts"``; ``"speed limit exceeded: joint 5 at ..."`` -> ``"speed limit
    exceeded"``. Free text keeps its words up to the first number or comparison."""
    text = str(reason).strip()
    m = _GATE.match(text)
    if m:
        return m.group(1)
    if ":" in text:
        text = text.split(":", 1)[0]
    text = _HAND_OBJECT.sub("hand-object", _SIDE.sub("", text))
    words = []
    for w in text.split():
        if any(c.isdigit() for c in w) or w in ("<", ">", "=", "<=", ">=", "/") or w.startswith("("):
            break
        words.append(w)
    while words and words[-1].lower() in _STOP:
        words.pop()
    return " ".join(words) or "other"


def load_records(roots) -> list[dict]:
    records = []
    for root in roots:
        for path in sorted(Path(root).glob("records/*/*.jsonl")):
            records += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return records


def _stage_flags(rec: dict) -> dict:
    k = (rec.get("K") or {}).get("passed", False)
    p = rec.get("P")
    return {"source": True, "excluded": rec.get("status") == "excluded",
            "retargeted": rec.get("status") == "ok" or (rec.get("status") == "excluded" and rec.get("K") is not None),
            "K": bool(k), "P_tested": p is not None,
            "P": bool(p and p.get("passed")), "K_and_P": bool(k and p and p.get("passed")),
            "written": bool(rec.get("file")), "errors": rec.get("status") in ("error", "read_error")}


def summarize(records: list[dict]) -> dict:
    by_dataset = defaultdict(Counter)
    by_family = defaultdict(Counter)
    body_parts, regimes = Counter(), Counter()
    first = {"K": Counter(), "P": Counter()}
    any_ = {"K": Counter(), "P": Counter()}
    excluded = Counter()
    sizes = defaultdict(lambda: Counter())
    for rec in records:
        if rec.get("status") == "excluded":
            excluded[f"{rec.get('excluded')}: {_EXCL_NUM.sub('#', str(rec.get('reason')))[:120]}"] += 1
        sc = rec.get("scene")
        if rec.get("file") and sc:
            s = sizes[rec.get("dataset", "?")]
            s["episodes"] += 1
            s["episode_bytes"] += sc.get("episode_bytes", 0)
            s["assets_referenced_bytes"] += sc.get("assets_referenced_bytes", 0)
            s["assets_new_bytes"] += sc.get("assets_new_bytes", 0)
        flags = _stage_flags(rec)
        for stage, value in flags.items():
            by_dataset[rec.get("dataset", "?")][stage] += value
            by_family[rec.get("family", "?")][stage] += value
        if flags["K"]:
            body_parts[" + ".join(sorted(rec.get("body_parts") or []))] += 1
            regimes[rec.get("regime") or "unknown"] += 1
        for tier in ("K", "P"):
            kinds = [reason_kind(x) for x in ((rec.get(tier) or {}).get("reasons") or [])]
            if kinds:
                first[tier][kinds[0]] += 1
                any_[tier].update(set(kinds))
    totals = Counter()
    for counts in by_family.values():
        totals.update(counts)
    return {"totals": dict(totals), "families": {k: dict(v) for k, v in sorted(by_family.items())},
            "datasets": {k: dict(v) for k, v in sorted(by_dataset.items())},
            "body_parts_K": dict(body_parts.most_common()), "regimes_K": dict(regimes.most_common()),
            "excluded_reasons": dict(excluded.most_common(25)),
            "storage_mb_per_episode": {k: {"episodes": v["episodes"],
                                           "episode_only": v["episode_bytes"] / v["episodes"] / 1e6,
                                           "with_library_share": (v["episode_bytes"] + v["assets_new_bytes"])
                                           / v["episodes"] / 1e6,
                                           "with_own_assets": (v["episode_bytes"] + v["assets_referenced_bytes"])
                                           / v["episodes"] / 1e6}
                                       for k, v in sorted(sizes.items()) if v["episodes"]},
            "top_failure_reasons": {t: dict(c.most_common(25)) for t, c in first.items()},
            "failure_reasons_any": {t: dict(c.most_common(25)) for t, c in any_.items()}}


def markdown(summary: dict) -> str:
    def table(rows: dict) -> list[str]:
        out = ["| name | " + " | ".join(STAGES) + " |", "|---" * (len(STAGES) + 1) + "|"]
        out += [f"| {name} | " + " | ".join(str(c.get(s, 0)) for s in STAGES) + " |" for name, c in rows.items()]
        return out

    lines = ["# Build results", "", "## Families", ""] + table(summary["families"])
    lines += ["", f"Total: " + ", ".join(f"{s} {summary['totals'].get(s, 0)}" for s in STAGES), "",
              "## Datasets", ""] + table(summary["datasets"])
    if summary.get("excluded_reasons"):
        lines += ["", "## Excluded episodes (not written)", ""]
        lines += [f"* {k}: {v}" for k, v in summary["excluded_reasons"].items()]
    if summary.get("storage_mb_per_episode"):
        lines += ["", "## Storage per written episode (MB: episode file; + its share of new library files; "
                      "+ every library file it references)", ""]
        lines += [f"* {k}: {v['episode_only']:.3f} / {v['with_library_share']:.3f} / {v['with_own_assets']:.3f} "
                  f"({v['episodes']} episodes)" for k, v in summary["storage_mb_per_episode"].items()]
    lines += ["", "## Body parts used (K passes)", ""]
    lines += [f"* {k or 'none'}: {v}" for k, v in summary["body_parts_K"].items()]
    for tier in ("K", "P"):
        lines += ["", f"## Tier {tier}: failure reasons by kind (episodes; first reason / any reason)", ""]
        first, any_ = summary["top_failure_reasons"].get(tier, {}), summary["failure_reasons_any"].get(tier, {})
        lines += [f"* {k}: {first.get(k, 0)} / {any_.get(k, 0)}" for k in sorted(set(first) | set(any_),
                                                                               key=lambda k: (-first.get(k, 0), -any_.get(k, 0), k))]
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
