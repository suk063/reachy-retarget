"""Helpers shared by the catalog generators (network only when a generator is invoked).

* :func:`hf_tree` - the Hugging Face tree API at a pinned revision (paginated).
* :func:`hf_digests` - publisher digests of a tree item: LFS SHA-256 or the git blob SHA-1.
* :func:`index_remote_tar` - every member header of a remote uncompressed tar, read by
  small HTTP ranges in parallel segments (each segment starts at a header found by
  scanning; every segment boundary is confirmed by the walk of the previous segment).
* :func:`write_table` - the gzip TSV row tables referenced by catalog ``table:`` keys.
"""
from __future__ import annotations

import gzip
import json
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..transports import KeepAliveOpener, find_tar_header, make_request, read_range, walk_tar

HF = "https://huggingface.co"


def hf_get_json(url, opener):
    with opener(make_request(url)) as r:
        return json.loads(r.read()), (getattr(r, "headers", None) or {}).get("Link")


def hf_tree(repo: str, revision: str, path: str = "", *, recursive=True, opener=None, repo_type="datasets"):
    """Every item of ``repo`` under ``path`` at ``revision`` (files and folders)."""
    opener = opener or KeepAliveOpener()
    q = urllib.parse.quote(path)
    url = f"{HF}/api/{repo_type}/{repo}/tree/{revision}/{q}?recursive={str(recursive).lower()}&expand=false"
    out = []
    while url:
        items, link = hf_get_json(url, opener)
        out.extend(items)
        m = re.search(r'<([^>]+)>;\s*rel="next"', link or "")
        url = m[1] if m else None
    return out


def hf_resolve(repo: str, revision: str, path: str, repo_type="datasets") -> str:
    return f"{HF}/{repo_type}/{repo}/resolve/{revision}/{urllib.parse.quote(path)}"


def hf_digests(item: dict) -> dict:
    """``{"sha256": ...}`` for LFS files, else ``{"git_blob_sha1": ...}`` (git object id)."""
    if item.get("lfs"):
        return {"sha256": item["lfs"]["oid"]}
    return {"git_blob_sha1": item["oid"]}


def index_remote_tar(url: str, size: int, *, opener=None, segment=1 << 30, workers=64, lookahead=16384,
                     scan=4 << 20, scan_limit=256 << 20, progress=None):
    """All members ``[(data_offset, size, name, typeflag)]`` of the tar at ``url``.

    The archive is cut into ``segment``-byte pieces; the first header at or after each
    cut is found by scanning ``scan``-byte blocks for a checksummed ustar header, then the
    pieces are walked in parallel with 512-byte header reads (plus ``lookahead``). A
    piece's walk must land exactly on the next piece's first header, which confirms the
    scan; otherwise the index is rejected.
    """
    opener = opener or KeepAliveOpener()

    def read(o, n):
        n = min(n, size - o)
        return read_range(url, o, n, opener) if n > 0 else b""

    def anchor(cut):
        if cut == 0:
            return 0
        pos = cut
        while pos < min(size, cut + scan_limit):
            data = read(pos, scan + 512)
            hit = find_tar_header(data, pos)
            if hit is not None:
                return hit
            pos += scan
        return None

    cuts = list(range(0, size, segment))
    with ThreadPoolExecutor(min(workers, len(cuts))) as ex:
        anchors = sorted({a for a in ex.map(anchor, cuts) if a is not None})

    def walk(i):
        stop = anchors[i + 1] if i + 1 < len(anchors) else None
        rows, last = [], anchors[i]
        for off, sz, name, typ in walk_tar(read, start=anchors[i], stop=stop, lookahead=lookahead):
            rows.append((off, sz, name, typ.decode() if isinstance(typ, bytes) else str(typ)))
            last = off + ((sz + 511) // 512) * 512
        if stop is not None and last != stop:
            raise IOError(f"{url}: walk of segment {i} ended at {last}, next segment starts at {stop}")
        if progress:
            progress(i, len(anchors), len(rows))
        return rows

    with ThreadPoolExecutor(min(workers, len(anchors))) as ex:
        parts = list(ex.map(walk, range(len(anchors))))
    return [r for p in parts for r in p]


def write_table(path, columns: list[str], rows: list[dict]) -> None:
    """Write ``rows`` as a gzip TSV with a header line (empty cell = absent)."""
    def cell(v):
        if v is None:
            return ""
        s = str(v)
        if "\t" in s or "\n" in s:
            raise ValueError(f"table cell contains a tab/newline: {s!r}")
        return s
    known_int = {"size", "offset", "stored_size", "inflated_size", "crc32", "archive_size"}
    ints = {c for c in columns if c not in known_int and any(isinstance(r.get(c), int) and not isinstance(
        r.get(c), bool) for r in rows) and all(r.get(c) is None or isinstance(r.get(c), int) for r in rows)}
    header = [f"{c}:int" if c in ints else c for c in columns]
    lines = ["\t".join(header)] + ["\t".join(cell(r.get(c)) for c in columns) for r in rows]
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as f:
        f.write(("\n".join(lines) + "\n").encode())  # mtime 0: identical rows give identical bytes


def write_catalog(path, header: str, doc: dict) -> None:
    import yaml
    Path(path).write_text(header + yaml.safe_dump(doc, sort_keys=False, width=120))


__all__ = ["hf_tree", "hf_resolve", "hf_digests", "index_remote_tar", "write_table", "write_catalog"]
