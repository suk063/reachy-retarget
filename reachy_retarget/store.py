"""Durable acquisition ledger. One connection per operation/thread."""

from pathlib import Path
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open('w') as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


class Store:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.path = self.root / "catalog/ledger.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as c:
            c.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS sources (
              id TEXT PRIMARY KEY, category TEXT, url TEXT, status TEXT,
              details TEXT NOT NULL, updated TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS files (
              source_id TEXT, path TEXT, url TEXT, size INTEGER, expected_sha256 TEXT,
              status TEXT DEFAULT 'queued', bytes INTEGER DEFAULT 0, sha256 TEXT,
              error TEXT, updated TEXT, PRIMARY KEY(source_id,path));
            CREATE TABLE IF NOT EXISTS episodes (
              id TEXT PRIMARY KEY, source_id TEXT, source_sequence TEXT, split TEXT,
              duration REAL, frames INTEGER, objects INTEGER, path TEXT,
              status TEXT, details TEXT, updated TEXT);
            """)

    def connect(self):
        c = sqlite3.connect(self.path, timeout=60)
        c.row_factory = sqlite3.Row
        return c

    def source(self, id, category, url, status="catalogued", **details):
        with self.connect() as c:
            old = c.execute("SELECT details FROM sources WHERE id=?", (id,)).fetchone()
            merged = json.loads(old[0]) if old else {}
            merged.update(details)
            c.execute(
                "INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?)",
                (id, category, url, status, json.dumps(merged), now()),
            )

    def file(self, source, path, url, size=None, expected_sha256=None):
        p = Path(path)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("Unsafe relative path")
        with self.connect() as c:
            c.execute(
                "INSERT OR IGNORE INTO files(source_id,path,url,size,expected_sha256,updated) VALUES(?,?,?,?,?,?)",
                (source, path, url, size, expected_sha256, now()),
            )

    def rows(self, table, where="1", args=()):
        if table not in ("sources", "files", "episodes"):
            raise ValueError(table)
        with self.connect() as c:
            return [
                dict(x) for x in c.execute(f"SELECT * FROM {table} WHERE {where}", args)
            ]

    def files(self, source, entries):
        """Insert a discovered page in one durable transaction."""
        values = []
        for path, url, size, digest in entries:
            p = Path(path)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("Unsafe relative path")
            values.append((source, path, url, size, digest, now()))
        with self.connect() as c:
            c.executemany(
                "INSERT OR IGNORE INTO files(source_id,path,url,size,expected_sha256,updated) VALUES(?,?,?,?,?,?)",
                values,
            )

    def update_file(self, source, path, **values):
        values["updated"] = now()
        if set(values) - {"status", "bytes", "sha256", "error", "size", "updated"}:
            raise ValueError("Unknown file field")
        with self.connect() as c:
            c.execute(
                "UPDATE files SET "
                + ",".join(f"{k}=?" for k in values)
                + " WHERE source_id=? AND path=?",
                (*values.values(), source, path),
            )

    def update_files(self, updates):
        """Commit a reconstruction batch atomically instead of fsync per file."""
        allowed = {"status", "bytes", "sha256", "error", "size", "updated"}
        with self.connect() as c:
            for source, path, values in updates:
                values = dict(values, updated=now())
                if set(values) - allowed:
                    raise ValueError("Unknown file field")
                c.execute(
                    "UPDATE files SET "
                    + ",".join(f"{k}=?" for k in values)
                    + " WHERE source_id=? AND path=?",
                    (*values.values(), source, path),
                )

    def episode(
        self,
        id,
        source_id,
        sequence,
        split,
        duration,
        frames,
        objects,
        path,
        status,
        details,
    ):
        with self.connect() as c:
            c.execute(
                "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    id,
                    source_id,
                    sequence,
                    split,
                    duration,
                    frames,
                    objects,
                    str(path),
                    status,
                    json.dumps(details),
                    now(),
                ),
            )
