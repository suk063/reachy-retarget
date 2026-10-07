"""Cross-process reservations without holding the disk mutex over network I/O."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import uuid

RESERVE = 50_000_000_000


def root_for(path):
    path = Path(path)
    for parent in (path, *path.parents):
        if (parent / "catalog/ledger.sqlite").exists():
            return parent
    return path if path.is_dir() else path.parent


def active(root):
    directory = Path(root) / "catalog/disk-reservations"
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for p in directory.glob("*.json"):
        try:
            item = json.loads(p.read_text())
            os.kill(item["pid"], 0)
            entries.append(item)
        except ProcessLookupError:
            p.unlink(missing_ok=True)
        except (ValueError, FileNotFoundError, PermissionError):
            continue
    return entries


def available(path):
    root = root_for(path)
    return shutil.disk_usage(path).free - sum(r["bytes"] for r in active(root))


@contextmanager
def allocation(root, size, reserve=RESERVE):
    root = Path(root)
    directory = root / "catalog"
    directory.mkdir(parents=True, exist_ok=True)
    token = directory / "disk-reservations" / f"{os.getpid()}-{uuid.uuid4().hex}.json"
    with (directory / "disk-write.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (
            shutil.disk_usage(root).free - sum(r["bytes"] for r in active(root)) - size
            < reserve
        ):
            raise RuntimeError("50 GB reserve before allocation; queue preserved")
        token.write_text(json.dumps({"pid": os.getpid(), "bytes": int(size)}))
    try:
        yield
    finally:
        token.unlink(missing_ok=True)
