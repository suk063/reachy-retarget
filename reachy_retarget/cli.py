"""Command line entry point: ``reachy-retarget {catalog,fetch,verify,index}``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _catalog(args):
    from .acquire import load_catalog

    for entry in load_catalog().values():
        if args.family in (None, entry.family):
            size = "?" if entry.size is None else f"{entry.size / 1e6:.1f} MB"
            flag = "" if entry.usable else "\tunusable"
            print(f"{entry.id}\t{size}\t{entry.kind}\t{entry.content or ''}\t{entry.license}{flag}")


def _fetch(args):
    from .acquire import fetch, load_catalog, select

    catalog = load_catalog()
    entries = select(catalog, args.ids, families=args.family, match=args.match)
    if not entries:
        raise SystemExit("nothing selected")
    log = (lambda e, recs: print(f"# {e.id}: {len(recs)} records", file=sys.stderr, flush=True)) if args.verbose \
        else None
    for rec in fetch(entries, args.root, catalog=catalog, strip_images=args.strip_images, workers=args.workers,
                     max_episodes=args.max_episodes, log=log):
        print(json.dumps(rec, sort_keys=True))


def _verify(args):
    from .acquire import load_catalog, read_ledger, record_digest, sha256_file

    catalog, problems = load_catalog(), 0
    ledger = read_ledger(args.root)
    for entry_id, rec in sorted(ledger.items()):
        path = Path(args.root) / rec["local_path"]
        if not path.exists():
            problem = "missing"
        elif sha256_file(path) != record_digest(rec):
            problem = "sha256 mismatch"
        elif entry_id in catalog and catalog[entry_id].sha256 not in (None, rec["sha256_verified"]):
            problem = "ledger differs from catalog"
        else:
            continue
        problems += 1
        print(f"{problem}\t{entry_id}")
    selected = [e for e in catalog.values() if e.family in args.family and e.usable]
    absent = [e.id for e in selected if not _fetched(e, ledger)]
    for entry_id in absent:
        print(f"not fetched\t{entry_id}")
    print(f"{len(ledger)} ledger records, {problems} problems, {len(absent)} catalogued entries not fetched")
    raise SystemExit(1 if problems or absent else 0)


def _fetched(e, ledger) -> bool:
    """An entry is fetched when its record exists; range packages fetched by earlier versions
    have only their (catalogued) member records."""
    if e.id in ledger:
        return True
    members = e.source.get("members") if e.transport == "range_package" else None
    return bool(members) and all(e.member_id(m["name"]) in ledger for m in members)


def index_rows(root: Path) -> list[dict]:
    """Index rows of a dataset directory: the ``index_row`` of every build record (``records/**/*.jsonl``)
    whose episode file exists, else rows read from every ``*.h5``/``*.hdf5`` episode below ``root``."""
    import json

    from .schema.io import index_row, read_episode

    records = sorted((root / "records").rglob("*.jsonl")) if (root / "records").is_dir() else []
    if records:
        rows = []
        for f in records:
            for line in f.read_text().splitlines():
                row = json.loads(line).get("index_row") if line.strip() else None
                if row and (root / row["file"]).is_file():
                    rows.append(row)
        return sorted(rows, key=lambda r: r["file"])
    paths = sorted({*root.rglob("*.h5"), *root.rglob("*.hdf5")})
    return [index_row(read_episode(path), str(path.relative_to(root))) for path in paths]


def _index(args):
    from .schema.io import write_index

    root = Path(args.dir)
    rows = index_rows(root)
    out = Path(args.out) if args.out else root
    out.mkdir(parents=True, exist_ok=True)
    print(write_index(out, rows), len(rows))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="reachy-retarget", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("catalog", help="list catalogued source files")
    p.add_argument("--family")
    p.set_defaults(run=_catalog)
    p = sub.add_parser("fetch", help="acquire catalogued entries (explicit network access)")
    p.add_argument("ids", nargs="*", help="catalog ids '<family>/<path>', or prefixes ending in '/' "
                                          "(every usable entry below)")
    p.add_argument("--family", action="append", default=[], help="fetch every usable entry of a family")
    p.add_argument("--match", help="substring filter applied to --family selections")
    p.add_argument("--root", required=True, help="data root; files land in <root>/raw/<family>/")
    p.add_argument("--strip-images", action="store_true", help="keep a state-only copy of image-bearing files")
    p.add_argument("--workers", type=int, default=1, help="entries fetched concurrently")
    p.add_argument("--max-episodes", type=int, help="tar_stream: keep extras/ of the first N episodes only")
    p.add_argument("-v", "--verbose", action="store_true", help="progress on stderr")
    p.set_defaults(run=_fetch)
    p = sub.add_parser("verify", help="re-hash every ledger record under a data root")
    p.add_argument("--root", required=True)
    p.add_argument("--family", action="append", default=[], help="also report unfetched files of a family")
    p.set_defaults(run=_verify)
    p = sub.add_parser("index", help="write index.parquet for a dataset directory (from its build records, else "
                                     "from its episode files)")
    p.add_argument("dir")
    p.add_argument("--out", help="directory to write index.parquet to (default: dir; file paths stay relative to dir)")
    p.set_defaults(run=_index)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
