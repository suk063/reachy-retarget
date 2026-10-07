"""Drain the existing acquisition queue with durable provider cooldowns.

This runs immediately as a local process, never submits forms or creates a cloud
job. A stop sentinel is checked between batches. Failed payloads already exhaust
bounded retries inside acquire.download and are not retried forever.
"""

import argparse
import collections
import fcntl
import json
import os
import time
from urllib.parse import urlparse
from .store import Store, json_write, now
from .acquire import fetch, host_paused, file_lock
from .bulk import fetch_xet, inventory
from .disk import available

PENDING = "status IN ('queued','downloading','retry_wait','storage_wait')"


def run(store):
    directory = store.root / "runs"
    directory.mkdir(exist_ok=True)
    lock = (directory / "collector.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = directory / "collector-status.json"
    stop = directory / "collector.stop"
    failures = {}
    last_source = None

    def status(phase, **extra):
        json_write(
            status_path, dict(pid=os.getpid(), phase=phase, updated_at=now(), **extra)
        )

    def final_report(phase, **extra):
        status("refreshing_final_report", next_phase=phase)
        from .object_inventory import run as objects
        from .inventory import run as sequences
        from .validate import run as validate
        from .report import run as report

        objects(store)
        sequences(store)
        validate(store)
        status(phase, **extra)
        report(store)

    while not stop.exists():
        if available(store.root) < 50_000_000_000 + 64 * 1024 * 1024:
            final_report(
                "storage_decision_required", reason="50 GB reserve; queue retained"
            )
            return
        rows = store.rows("files", PENDING)
        if not rows:
            final_report(
                "finished_acquisition_attempts",
                failed_files=len(store.rows("files", "status='failed'")),
            )
            return
        groups = collections.defaultdict(list)
        paused = {}
        externally_owned = 0
        for row in rows:
            if row["status"] == "downloading":
                ownership = file_lock(store, row)
                if ownership is None:
                    externally_owned += 1
                    continue
                ownership.close()
            host = urlparse(row["url"]).netloc
            if host not in paused:
                paused[host] = host_paused(store, row["url"])
            if not paused[host]:
                groups[row["source_id"]].append(row)
        if not groups:
            status(
                "waiting_for_active_workers"
                if externally_owned == len(rows)
                else "provider_cooldown",
                pending_files=len(rows),
                externally_owned_files=externally_owned,
            )
            time.sleep(30)
            continue
        # Prioritize rare sources and state projections, then rotate among large
        # collections. Every selected file remains in the durable manifest.
        sizes = {s: len(v) for s, v in groups.items()}
        choices = sorted(
            groups,
            key=lambda s: (
                s == last_source,
                not any("#state-" in r["url"] for r in groups[s]),
                sizes[s],
            ),
        )
        sid = choices[0]
        last_source = sid
        batch = groups[sid][:2048]
        status(
            "acquiring", source=sid, selected_files=len(batch), pending_files=len(rows)
        )
        hashes = inventory(store)
        xet = [
            r
            for r in batch
            if (r["source_id"], r["path"]) in hashes
            and "#" not in r["url"]
            and r.get("expected_sha256")
        ]
        try:
            if xet:
                outcome = fetch_xet(store, xet, batch_size=512)
            else:
                # fetch() uses a bounded process pool for HDF5/RLDS projection.
                outcome = fetch(
                    store,
                    [sid],
                    workers=4,
                    limit=min(16, len(batch)),
                    retry_failed=False,
                )
            print(now(), sid, outcome, flush=True)
            if outcome.get("storage_wait"):
                final_report(
                    "storage_decision_required",
                    source=sid,
                    reason="50 GB reserve; queue retained",
                )
                return
            if outcome.get("connection_failed"):
                failures[sid] = failures.get(sid, 0) + 1
                if failures[sid] >= 3:
                    store.update_files(
                        [
                            (
                                r["source_id"],
                                r["path"],
                                {
                                    "status": "failed",
                                    "error": "Three public connection attempts failed; explicit fetch can retry.",
                                },
                            )
                            for r in batch
                        ]
                    )
            if not outcome.get("downloaded"):
                time.sleep(5)
        except Exception as e:
            # Avoid leaking signed transport URLs into the public log.
            failures[sid] = failures.get(sid, 0) + 1
            status("batch_error", source=sid, error_type=type(e).__name__)
            print(now(), sid, type(e).__name__, flush=True)
            if failures[sid] >= 3:
                store.update_files(
                    [
                        (
                            r["source_id"],
                            r["path"],
                            {
                                "status": "failed",
                                "error": "Three batch exceptions: " + type(e).__name__,
                            },
                        )
                        for r in batch
                    ]
                )
            time.sleep(5)
    status("stopped_by_sentinel", reason="Current committed files and queue preserved")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    a = p.parse_args()
    run(Store(a.root))
