"""Reproducible collection and representative conversion commands."""

import argparse
import importlib
import json
from .store import Store, ROOT


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(ROOT / "configs/default.json"))
    preliminary, _ = pre.parse_known_args()
    config = json.loads(open(preliminary.config).read())
    p = argparse.ArgumentParser(
        description="Reachy 2 state acquisition, conversion and validation",
        parents=[pre],
    )
    p.add_argument("--root", default=str(ROOT))
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--group", choices=["core", "survey", "oxe", "all"], default="all")
    d.add_argument(
        "--locked",
        metavar="MANIFEST_GZ",
        help="Restore the exact public source selection without network discovery",
    )
    f = sub.add_parser("fetch")
    f.add_argument("--source", action="append")
    f.add_argument("--workers", type=int, default=config["fetch"]["workers"])
    f.add_argument("--limit", type=int)
    f.add_argument(
        "--transport",
        choices=["auto", "http", "xet"],
        default=config["fetch"]["transport"],
    )
    for cmd in ("normalize", "retarget", "validate", "report"):
        q = sub.add_parser(cmd)
        q.add_argument("--source", action="append")
        q.add_argument(
            "--limit", type=int, default=config["normalize"]["sequences_per_source"]
        )
        if cmd == "validate":
            q.add_argument(
                "--physics",
                action="store_true",
                help="Replay fixed Can demo_0..9 with actual dynamic contact",
            )
        if cmd == "report":
            q.add_argument(
                "--refresh-inventory",
                action="store_true",
                help="Scan acquired episodes and recompute content deduplication",
            )
    args = p.parse_args()
    store = Store(args.root)
    if config.get("storage_reserve_bytes") != 50_000_000_000:
        p.error("The approved free-space reserve is fixed at 50 decimal GB.")
    if args.command == "discover":
        if args.locked:
            from .manifest import restore_lock

            print(restore_lock(store, args.locked))
        else:
            from .discovery import discover

            discover(store, args.group)
    elif args.command == "fetch":
        from .acquire import fetch

        if args.transport in ("auto", "xet"):
            from .bulk import fetch_xet

            rows = store.rows(
                "files",
                "status IN ('queued','failed','storage_wait','downloading','retry_wait')",
            )
            if args.source:
                rows = [x for x in rows if x["source_id"] in args.source]
            if args.limit:
                rows = rows[: args.limit]
            print(fetch_xet(store, rows, config["fetch"]["xet_batch_size"]), flush=True)
        if args.transport in ("auto", "http"):
            fetch(store, args.source, args.workers, args.limit)
    else:
        if args.command == "validate" and args.physics:
            from .contact import run

            c = config["contact"]
            import math

            run(
                store,
                limit=c["sequences"],
                label=c["label"],
                time_scale=c["time_scale"],
                heading=math.degrees(c["heading_rad"]),
                depth=c["grasp_depth_m"],
                render=c["render"],
            )
        if args.command == "report" and args.refresh_inventory:
            from .object_inventory import run as objects

            objects(store)
            from .inventory import run

            run(store)
        importlib.import_module("reachy_retarget." + args.command).run(
            store, sources=args.source, limit=args.limit
        )


if __name__ == "__main__":
    main()
