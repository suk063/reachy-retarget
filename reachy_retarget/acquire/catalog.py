"""Pinned catalog of source files (``catalog/<family>.yaml`` plus optional gzip TSV tables).

A catalog file is ``{"sources": [group, ...]}``. A group carries shared fields
(``family``, ``revision``, ``license`` and the defaults ``kind``, ``content``, ``usable``),
optional ``url_base`` (entry URL = ``url_base`` + entry ``src``, or + ``archive`` for rows that carry
their own ``archive_sha256``/``archive_size``), optional ``archives`` (name -> ``{url, sha256,
size, format}`` for archive-backed kinds), optional ``path_prefix``
(entry path = ``path_prefix`` + ``src`` when a row gives no path), inline ``files`` and an
optional ``table`` (gzip TSV next to the yaml, one entry per row, header = field names with
``:int`` marking integer columns, empty cell = absent). Other group keys (``repo``, ``note``, ``totals``, ...) are
documentation. Other top-level keys of a catalog file are documentation too.

Entry ``kind`` says how bytes are acquired (all through :func:`reachy_retarget.acquire.fetch`):

``file``
    the whole file at ``url``.
``file_images_embedded``
    a whole HDF5 file that also stores images; fetched only with ``strip_images`` and
    replaced by its state-only copy (:mod:`.strip`).
``tar_stream``
    an uncompressed tar streamed once; image/video members are discarded in memory and
    the other members written under the entry's directory with ``tar_members.json``.
``range_member``
    one member of an archive read by HTTP byte range (``archive``, ``offset``, ``size``;
    ``compression`` ``none`` | ``deflate`` (zip) | ``zstd``).
``range_package``
    a zstd-compressed tar stored as a byte range of an archive; members whose name ends
    with one of ``keep`` are written under the entry's directory (others discarded in
    memory) with ``package_members.json``.

``content`` is what the bytes are (``low_dim``, ``metadata``, ``assets``, ``docs``, ...).
Publisher digests: ``sha256`` and, in ``digests``, ``sha1`` (Box) or ``git_blob_sha1``
(non-LFS git files). Unknown entry keys are kept in ``meta``.
"""
from __future__ import annotations

import gzip
import posixpath
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path

import yaml

KINDS = ("file", "file_images_embedded", "tar_stream", "range_member", "range_package")
LEGACY_KINDS = {"images_embedded": "file_images_embedded"}  # earlier 'kind' values = content classes
DIGEST_KEYS = ("sha1", "git_blob_sha1")
SOURCE_KEYS = ("archive", "archive_sha256", "archive_size", "member", "offset", "stored_size", "compression", "crc32",
               "inflated_size", "range_sha256", "keep", "members")
CORE_KEYS = ("path", "url", "src", "revision", "sha256", "size", "license", "kind", "content", "dataset", "note",
             "asset_marker", "usable", "family", "id")
GROUP_DEFAULTS = ("family", "revision", "license", "kind", "content", "usable", "compression", "keep")
INT_KEYS = ("size", "offset", "stored_size", "inflated_size", "crc32", "archive_size")
CATALOG_DIR = Path(str(resources.files(__package__).joinpath("catalog")))


@dataclass(frozen=True)
class CatalogEntry:
    id: str                    # "<family>/<path>"
    family: str                # top-level folder under raw/
    path: str                  # local path under raw/<family>/ (a directory for tar_stream/range_package
                               # outputs: tar_stream -> dirname(path), range_package -> path)
    url: str                   # pinned URL of the file or of the archive holding it
    revision: str              # publisher revision (commit, tag or release version)
    sha256: str | None         # publisher-declared (or catalog-time) digest of the bytes stored
    size: int | None           # bytes (tar_stream: the tar; range_package: the compressed range)
    license: str
    kind: str = "file"         # acquisition kind, see KINDS
    content: str | None = None # low_dim | metadata | assets | docs | index | robot_model ...
    dataset: str | None = None # logical dataset name, e.g. "robomimic/can/ph"
    note: str | None = None
    asset_marker: str | None = None  # assets: recorded MJCF paths containing this marker
                                     # resolve to the archive members after it
    usable: bool = True        # False: catalogued for completeness, not selected by prefix fetches
    digests: dict = field(default_factory=dict, hash=False)  # other publisher digests (sha1, git_blob_sha1)
    source: dict = field(default_factory=dict, hash=False)   # kind-specific transport fields
    meta: dict = field(default_factory=dict, hash=False)     # family-specific descriptive fields

    @property
    def transport(self) -> str:
        """The acquisition kind; entries built with an earlier content-class ``kind`` map to
        ``file`` (``images_embedded`` to ``file_images_embedded``)."""
        return self.kind if self.kind in KINDS else LEGACY_KINDS.get(self.kind, "file")

    def local_path(self, root) -> Path:
        return Path(root) / "raw" / self.family / self.path

    def output_dir(self, root) -> Path:
        """Directory receiving the members of a ``tar_stream``/``range_package`` entry."""
        p = self.local_path(root)
        return p.parent if self.transport == "tar_stream" else p

    def member_id(self, name: str) -> str:
        """Ledger id of a ``range_package`` member (written flat into the package directory)."""
        return f"{self.id}/{posixpath.basename(name)}"


def _cast(key, value):
    if value is None or value == "":
        return None
    if key in INT_KEYS:
        return int(value)
    if key == "usable" and isinstance(value, str):
        return value.lower() in ("1", "true", "yes")
    return value


def _rows(table: Path):
    """Rows of a gzip TSV table; a ``name:int`` header casts that column to int."""
    with gzip.open(table, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        cols = [(c.split(":")[0], c.endswith(":int")) for c in header]
        for line in f:
            row = {}
            for (c, is_int), v in zip(cols, line.rstrip("\n").split("\t")):
                if v != "":
                    row[c] = int(v) if is_int else _cast(c, v)
            yield row


def make_entry(f: dict, group: dict) -> CatalogEntry:
    """One entry from a row ``f`` and its ``group`` (shared fields, ``archives``, ``url_base``)."""
    f = {**{k: group[k] for k in GROUP_DEFAULTS if k in group}, **f}
    archives = group.get("archives") or {}
    source = {k: f.pop(k) for k in SOURCE_KEYS if k in f}
    if isinstance(source.get("keep"), str):  # table cells: comma-separated suffixes
        source["keep"] = source["keep"].split(",")
    digests = {k: f.pop(k) for k in DIGEST_KEYS if k in f}
    if "archive" in source:
        a = archives.get(source["archive"])
        if a is not None:
            f.setdefault("url", a["url"])
            source.update(archive_sha256=a.get("sha256"), archive_size=a.get("size"), archive_format=a.get("format"))
        elif "url_base" in group and "archive_sha256" in source:  # row carries its archive's digest
            f.setdefault("url", group["url_base"] + source["archive"])
            source.setdefault("archive_format", group.get("archive_format"))
        else:
            raise ValueError(f"{f.get('path')}: archive {source['archive']!r} not in the group's archives")
    src = f.pop("src", None)
    if "path" not in f and src is not None and "path_prefix" in group:
        f["path"] = group["path_prefix"] + src
    if "url" not in f:
        if src is None or "url_base" not in group:
            raise ValueError(f"{f.get('path')}: no url (and no src + url_base)")
        f["url"] = group["url_base"] + src
    core = {k: f.pop(k) for k in list(f) if k in CORE_KEYS}
    core.setdefault("sha256", None)
    core.setdefault("size", None)
    entry = CatalogEntry(id=f"{core['family']}/{core['path']}", digests=digests, source=source, meta=f,
                         **{k: v for k, v in core.items() if k != "id"})
    if entry.kind not in KINDS:
        raise ValueError(f"{entry.id}: unknown kind {entry.kind!r} (one of {KINDS})")
    return entry


def parse_catalog(text: str, base: Path | None = None, *, tables: bool = True) -> list[CatalogEntry]:
    """Entries of one catalog file; ``tables=False`` skips the rows of gzip TSV tables."""
    doc = yaml.load(text, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    out = []
    for group in doc.get("sources", []):
        for f in group.get("files") or []:
            out.append(make_entry(dict(f), group))
        if group.get("table") and tables:
            if base is None:
                raise ValueError("a catalog with a table needs its directory")
            out.extend(make_entry(row, group) for row in _rows(base / group["table"]))
    return out


def _index(entries) -> dict[str, CatalogEntry]:
    out = {}
    for e in entries:
        if e.id in out:
            raise ValueError(f"duplicate catalog id {e.id}")
        out[e.id] = e
    return out


@lru_cache(maxsize=2)
def _packaged(tables: bool = True) -> dict[str, CatalogEntry]:
    files = sorted(p for p in CATALOG_DIR.iterdir() if p.name.endswith(".yaml"))
    return _index(e for p in files for e in parse_catalog(p.read_text(), p.parent, tables=tables))


def load_catalog(path=None, *, tables: bool = True) -> dict[str, CatalogEntry]:
    """Entries keyed by id, from one catalog file or every packaged ``catalog/*.yaml``.

    ``tables=False`` leaves out the rows of gzip TSV tables (the per-member listings of
    BEHAVIOR, MobileManiBench, MolmoBot and RoboVerse, about 210k entries and several
    hundred MB of Python objects); readers whose files and asset archives are all listed
    inline use it. The packaged catalog is parsed once per process (per ``tables``
    value); callers get their own dict."""
    if path:
        p = Path(path)
        return _index(parse_catalog(p.read_text(), p.parent, tables=tables))
    return dict(_packaged(tables))


@lru_cache(maxsize=8)
def _family_file(name: str) -> dict[str, CatalogEntry]:
    p = CATALOG_DIR / f"{name}.yaml"
    return _index(parse_catalog(p.read_text(), p.parent))


def load_family_catalog(family: str) -> dict[str, CatalogEntry]:
    """Entries of one packaged ``catalog/<family>.yaml`` only, its tables included (BEHAVIOR:
    30k entries, about 60 MB, instead of the 210k entries and about 400 MB of
    :func:`load_catalog`). For readers that look up only their own family's files (provenance by
    SHA-256 or path). Parsed once per process; callers get their own dict."""
    if not (CATALOG_DIR / f"{family}.yaml").is_file():
        raise KeyError(f"no packaged catalog for family {family!r}")
    return dict(_family_file(family))


def select(catalog: dict, patterns, *, families=(), match=None) -> list[CatalogEntry]:
    """Entries for ``patterns``: an exact id, or a prefix ending in ``/`` (every usable entry
    below it), plus every usable entry of ``families`` (optionally containing ``match``)."""
    out, seen = [], set()

    def add(e):
        if e.id not in seen:
            seen.add(e.id)
            out.append(e)

    for p in patterns:
        if isinstance(p, CatalogEntry):
            add(p)
        elif p in catalog:
            add(catalog[p])
        elif p.endswith("/"):
            hits = [e for k, e in catalog.items() if k.startswith(p) and e.usable]
            if not hits:
                raise KeyError(f"no usable catalog entry below {p!r}")
            for e in hits:
                add(e)
        else:
            raise KeyError(f"unknown catalog id {p!r}")
    for e in catalog.values():
        if e.family in families and e.usable and (not match or match in e.id):
            add(e)
    return out


__all__ = ["CATALOG_DIR", "KINDS", "CatalogEntry", "load_catalog", "load_family_catalog", "make_entry", "parse_catalog", "select"]
