"""Fetch job manifests: one JSONL line per cluster job, batching catalog entries by size.

Each line is ``{"id", "argv", "publish": "data", "timeout_s"}`` with ``argv`` =
``-m reachy_retarget.cli fetch <selectors> --root {out} --strip-images [--workers N]``; the
job fetches into its private ``{out}`` and :mod:`cluster.job` merges it no-overwrite into
the shared data root. Selectors are catalog ids or directory prefixes ending in ``/``
(a prefix is used only when every usable catalog entry below it is part of the job), so
argv stays short for families with thousands of small files.

Batching: entries are grouped by catalog path, groups are split until each fits
``max_bytes`` of transfer, and consecutive groups are packed into jobs that move about
``target_bytes`` (1-5 GB) and never need more than ``scratch_bytes`` (40 GB) of local
scratch (upper bound of what the job writes: file size, non-image tar members <= tar size,
inflated package size). A single entry larger than the transfer target is its own job.
Job ids are derived from the selectors (``fetch-<family>-<first selector>-<hash>``), so the
same catalog gives the same ids.

    python -m cluster.manifests behavior --out runs/fetch-w2-behavior-v1.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reachy_retarget.acquire import CatalogEntry, load_catalog  # noqa: E402

GB = 1_000_000_000
FAMILIES = {  # family: catalog prefixes of the full-scale fetch
    "behavior": ["behavior/2025-challenge-demos/", "behavior/omnigibson-robot-assets/"],
    "mobilemanibench": ["mobilemanibench/"],
    "robocasa": ["robocasa/"],
    "roboverse": ["roboverse/"],
    "molmobot": ["molmobot/"],
}


def transfer_bytes(e: CatalogEntry) -> int:
    """Bytes moved over the network to acquire ``e``."""
    s = e.source
    if e.transport == "range_member":
        return (s.get("stored_size") or e.size or 0) + 512
    if e.transport == "range_package":
        return s["stored_size"]
    return e.size or 0


def scratch_bytes(e: CatalogEntry) -> int:
    """Upper bound of the bytes written to the job's private root for ``e``."""
    if e.transport == "range_package":
        return e.source.get("inflated_size") or 4 * e.source["stored_size"]
    if e.transport == "file_images_embedded":
        return 2 * (e.size or 0)  # original + state-only copy until the original is deleted
    return e.size or 0


def _units(entries, catalog, max_bytes):
    """``[(selector, [entries])]``: directory prefixes whose usable entries are all selected
    and fit ``max_bytes``, else smaller prefixes, else single ids; in id order."""
    chosen = {e.id for e in entries}
    usable_below = {}
    for e in catalog.values():
        if e.usable:
            parts = e.id.split("/")
            for i in range(1, len(parts)):
                usable_below.setdefault("/".join(parts[:i]) + "/", []).append(e.id)

    def split(prefix, members):
        total = sum(transfer_bytes(e) for e in members)
        complete = all(i in chosen for i in usable_below.get(prefix, ()))
        if prefix and complete and total <= max_bytes and len(members) > 1:
            return [(prefix, members)]
        children, leaves = {}, []
        depth = prefix.count("/")
        for e in members:
            parts = e.id.split("/")
            if len(parts) > depth + 1:
                children.setdefault("/".join(parts[:depth + 1]) + "/", []).append(e)
            else:
                leaves.append(e)
        out = [(e.id, [e]) for e in leaves]
        for child, sub in children.items():
            out.extend(split(child, sub))
        return sorted(out, key=lambda u: u[0])

    return split("", sorted(entries, key=lambda e: e.id))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def plan(family: str, *, catalog=None, prefixes=None, target_bytes=3 * GB, max_bytes=5 * GB,
         scratch_limit=40 * GB, max_selectors=200, skip=()) -> list[dict]:
    """Jobs (manifest lines plus ``bytes``/``scratch``/``entries`` stats) for ``family``."""
    catalog = catalog or load_catalog()
    prefixes = prefixes or FAMILIES[family]
    skip = set(skip)
    entries = [e for e in catalog.values() if e.usable and e.id not in skip
               and any(e.id.startswith(p) for p in prefixes)]
    units = _units(entries, catalog, max_bytes)
    jobs, cur = [], []

    def close():
        if cur:
            jobs.append(list(cur))
            cur.clear()

    for unit in units:
        t = sum(transfer_bytes(e) for e in unit[1])
        s = sum(scratch_bytes(e) for e in unit[1])
        ct = sum(transfer_bytes(e) for u in cur for e in u[1])
        cs = sum(scratch_bytes(e) for u in cur for e in u[1])
        if cur and (ct + t > max_bytes or (ct >= target_bytes) or cs + s > scratch_limit or len(cur) >= max_selectors):
            close()
        cur.append(unit)
    close()
    if len(jobs) > 1:  # a small remainder joins the previous job when both fit the limits
        last, prev = jobs[-1], jobs[-2]
        size = lambda js, f: sum(f(e) for u in js for e in u[1])  # noqa: E731
        if size(last, transfer_bytes) < GB and size(prev + last, transfer_bytes) <= max_bytes \
                and size(prev + last, scratch_bytes) <= scratch_limit and len(prev + last) <= max_selectors:
            jobs[-2:] = [prev + last]
    out = []
    for units_ in jobs:
        sel = [u[0] for u in units_]
        ents = [e for u in units_ for e in u[1]]
        t = sum(transfer_bytes(e) for e in ents)
        s = sum(scratch_bytes(e) for e in ents)
        if s > scratch_limit and len(ents) > 1:
            raise ValueError(f"job {sel[0]} needs {s} bytes of scratch")
        requests = sum(1 for e in ents if e.transport in ("range_member", "file", "file_images_embedded"))
        workers = 8 if requests > 50 else 1
        timeout = int(min(12 * 3600, max(3600, 1800 + t / 10e6 + requests * 1.0 / workers)))
        digest = hashlib.sha1("\n".join(sel).encode()).hexdigest()[:8]
        job_id = f"fetch-{family}-{_slug(sel[0].split('/', 1)[1] if '/' in sel[0] else sel[0])}-{digest}"
        argv = ["-m", "reachy_retarget.cli", "fetch", *sel, "--root", "{out}", "--strip-images"]
        if workers > 1:
            argv += ["--workers", str(workers)]
        out.append({"id": job_id, "argv": argv, "publish": "data", "timeout_s": timeout,
                    "_stats": {"entries": len(ents), "transfer_bytes": t, "scratch_bytes": s}})
    ids = [j["id"] for j in out]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate job ids")
    return out


def write_manifest(family: str, path, **kw) -> dict:
    """Write the manifest of ``family`` to ``path`` (JSONL); return a summary."""
    jobs = plan(family, **kw)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for j in jobs:
            f.write(json.dumps({k: v for k, v in j.items() if not k.startswith("_")}) + "\n")
    t = [j["_stats"]["transfer_bytes"] for j in jobs]
    s = [j["_stats"]["scratch_bytes"] for j in jobs]
    return {"family": family, "jobs": len(jobs), "entries": sum(j["_stats"]["entries"] for j in jobs),
            "transfer_gb": round(sum(t) / GB, 2), "scratch_gb_upper": round(sum(s) / GB, 2),
            "job_transfer_gb": [round(min(t) / GB, 3), round(sorted(t)[len(t) // 2] / GB, 3), round(max(t) / GB, 3)],
            "job_scratch_gb_max": round(max(s) / GB, 2)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("families", nargs="+", choices=sorted(FAMILIES))
    ap.add_argument("--out", help="manifest path (one family) or a directory (default runs/)")
    ap.add_argument("--wave", default="w2", help="manifest name: fetch-<wave>-<family>-<version>.jsonl")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--skip-root", help="data root whose ledger records mark entries as already fetched")
    a = ap.parse_args(argv)
    skip = ()
    if a.skip_root:
        from reachy_retarget.acquire import read_ledger
        skip = set(read_ledger(a.skip_root))
    for fam in a.families:
        out = Path(a.out) if a.out and a.out.endswith(".jsonl") else Path(a.out or ROOT / "runs") \
            / f"fetch-{a.wave}-{fam}-{a.version}.jsonl"
        print(json.dumps({**write_manifest(fam, out, skip=skip), "path": str(out)}))


if __name__ == "__main__":
    main()
