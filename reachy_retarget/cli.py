"""Command line entry point: ``reachy-retarget {catalog,fetch,index}``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _catalog(args):
    from .acquire import load_catalog

    for entry in load_catalog().values():
        if args.family in (None, entry.family):
            print(f"{entry.id}\t{entry.size / 1e6:.1f} MB\t{entry.kind}\t{entry.license}")


def _fetch(args):
    from .acquire import fetch, load_catalog

    catalog = load_catalog()
    ids = list(args.ids) + [e.id for e in catalog.values()
                            if e.family in args.family and (not args.match or args.match in e.id)]
    if not ids:
        raise SystemExit("nothing selected")
    for record in fetch(ids, args.root, catalog=catalog, strip_images=args.strip_images):
        print(json.dumps(record, sort_keys=True))


def _index(args):
    from .schema.io import index_row, read_episode, write_index

    root = Path(args.dir)
    rows = [index_row(read_episode(path), str(path.relative_to(root))) for path in sorted(root.rglob("*.hdf5"))]
    print(write_index(root, rows), len(rows))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="reachy-retarget", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("catalog", help="list catalogued source files")
    p.add_argument("--family")
    p.set_defaults(run=_catalog)
    p = sub.add_parser("fetch", help="download catalogued files (explicit network access)")
    p.add_argument("ids", nargs="*", help="catalog ids '<family>/<path>'")
    p.add_argument("--family", action="append", default=[], help="fetch every file of a family")
    p.add_argument("--match", help="substring filter applied to --family selections")
    p.add_argument("--root", required=True, help="data root; files land in <root>/raw/<family>/")
    p.add_argument("--strip-images", action="store_true", help="keep a state-only copy of image-bearing files")
    p.set_defaults(run=_fetch)
    p = sub.add_parser("index", help="write index.parquet for a directory of episodes")
    p.add_argument("dir")
    p.set_defaults(run=_index)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
