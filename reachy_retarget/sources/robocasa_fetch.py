"""Explicit acquisition of RoboCasa365 LeRobot packages without storing images.

The official RoboCasa365 datasets (``robocasa/scripts/download_datasets.py``) are one
uncompressed ``lerobot.tar`` per task, split and source, served as Box shared files
(``robocasa/models/assets/box_links/box_links_ds.json`` at the pinned RoboCasa commit).
Each tar holds a LeRobot v2.1 dataset: ``data/`` (per-episode parquet: 16-d state, 12-d
action, reward/done), ``meta/``, ``extras/`` (``dataset_meta.json`` and per episode
``ep_meta.json``, ``model.xml.gz`` = the recorded MJCF, ``states.npz`` = MuJoCo states)
and ``videos/`` (three 256x256 MP4 camera streams per episode). Member order differs
between human tars (data, extras, meta, videos) and MimicGen tars (meta, videos, data,
extras; the one MimicGen tar inspected, OpenDrawer ``mg/demo/2025-08-20-21-55-00``, has no
``extras/`` at all), so the whole tar is streamed once:

* every byte is hashed (SHA-1, the digest Box publishes, plus SHA-256) and the total
  size is compared with the catalogue, so the tar is verified although it is never
  stored;
* image members (``videos/``, ``images/`` or an image/video file extension) are
  discarded in memory and never touch the disk;
* every other member is written to ``<root>/raw/robocasa/<tar dir>/<member>`` and listed
  with its tar offset, size and SHA-256 in ``<tar dir>/tar_members.json``;
* dropped connections are resumed with HTTP ``Range`` from the last byte received.

The tar table lives in ``reachy_retarget/acquire/catalog/robocasa.yaml`` under
``robocasa_tars``. Box publishes no SHA-256, so the pinned identity of a tar is its Box
file id, file version id and SHA-1. Asset archives are ordinary ``sources`` entries of
the same file and go through :func:`reachy_retarget.acquire.fetch`.

:func:`fetch_asset_subset` copies selected members of catalogued asset zips (read over
HTTP ``Range`` from the zip central directory, CRC-32 checked) into
``<root>/raw/robocasa/asset_subset/``, for local verification when the full asset
archives (about 11 GB) do not fit. :func:`generate_catalog` rebuilds the catalog file
from publisher metadata.

Nothing here touches the network on import.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import posixpath
import re
import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile
import zlib
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path

import yaml

from ..acquire.fetch import RESERVE_BYTES, USER_AGENT, CatalogEntry, ChecksumMismatch, InsufficientDisk, _record

CHUNK = 1 << 20
FAMILY = "robocasa"
IMAGE_EXT = (".mp4", ".avi", ".mkv", ".webm", ".mov", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif",
             ".tiff", ".exr")
MANIFEST = "tar_members.json"
ASSET_MARKER = "robocasa/models/assets/"
SUBSET_DIR = "asset_subset"


@dataclass(frozen=True)
class TarEntry:
    id: str                    # "robocasa/<path>"
    path: str                  # "v1.0/<split>/<atomic|composite>/<Task>/<date>[/mg/demo/<stamp>]/lerobot.tar"
    dataset: str               # "robocasa/v1.0/..." (path without /lerobot.tar)
    task: str
    split: str                 # "pretrain" | "target"
    task_type: str             # "atomic" | "composite"
    source: str                # "human" | "mg" (MimicGen-generated)
    variant: str               # registry key: "human_path", "mg_path", "mg_5x5_path", ...
    url: str                   # Box shared/static download URL
    shared_link: str
    box_file_id: str
    box_file_version: str
    box_sha1: str
    size: int
    seed: str | None = None    # generated tars: dataset of the human seed demonstrations
    horizon: int | None = None
    note: str | None = None
    revision: str = ""
    license: str = ""

    def local_dir(self, root) -> Path:
        return Path(root) / "raw" / FAMILY / posixpath.dirname(self.path)


def _catalog_text(path=None) -> str:
    return Path(path).read_text() if path else \
        resources.files("reachy_retarget.acquire").joinpath("catalog/robocasa.yaml").read_text()


def load_tars(path=None) -> dict[str, TarEntry]:
    """Parse the ``robocasa_tars`` section of the RoboCasa catalog file."""
    sec = yaml.load(_catalog_text(path), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))["robocasa_tars"]
    out = {}
    for t in sec["tars"]:
        e = TarEntry(id=f"{FAMILY}/{t['path']}", revision=sec["revision"], license=sec["license"], **t)
        if e.id in out:
            raise ValueError(f"duplicate tar id {e.id}")
        out[e.id] = e
    return out


def is_image_member(name: str) -> bool:
    parts = name.split("/")
    return any(p in ("videos", "images") for p in parts[:-1]) or name.lower().endswith(IMAGE_EXT)


# ---------------------------------------------------------------- streaming

class _ResumingStream(io.RawIOBase):
    """Sequential reader of ``url`` that hashes every byte and resumes with Range."""

    def __init__(self, url, open_url, retries=8):
        self.url, self.open_url, self.retries = url, open_url, retries
        self.pos, self.total = 0, None
        self.sha1, self.sha256 = hashlib.sha1(), hashlib.sha256()
        self.resp = None
        self._open()

    def _open(self):
        req = urllib.request.Request(self.url, headers={"User-Agent": USER_AGENT})
        if self.pos:
            req.add_header("Range", f"bytes={self.pos}-")
        self.resp = self.open_url(req)
        status = getattr(self.resp, "status", 200)
        if self.pos and status != 206:
            raise RuntimeError(f"{self.url}: server ignored Range on resume (status {status})")
        rng = self.resp.headers.get("Content-Range") if hasattr(self.resp, "headers") else None
        length = self.resp.headers.get("Content-Length") if hasattr(self.resp, "headers") else None
        if rng:
            self.total = int(rng.rsplit("/", 1)[1])
        elif length and not self.pos:
            self.total = int(length)

    def readable(self):
        return True

    def readinto(self, b):
        for attempt in range(self.retries + 1):
            try:
                n = self.resp.readinto(b) if hasattr(self.resp, "readinto") else None
                if n is None:
                    data = self.resp.read(len(b))
                    n = len(data)
                    b[:n] = data
                break
            except (OSError, urllib.error.URLError):
                if attempt == self.retries:
                    raise
                try:
                    self.resp.close()
                finally:
                    self._open()
        view = memoryview(b)[:n]
        self.sha1.update(view)
        self.sha256.update(view)
        self.pos += n
        return n

    def drain(self):
        while self.readinto(bytearray(CHUNK)):
            pass

    def close(self):
        if self.resp is not None:
            self.resp.close()
        super().close()


def _tar_record_entry(e: TarEntry) -> CatalogEntry:
    return CatalogEntry(id=e.id, family=FAMILY, path=e.path, url=e.url, revision=e.box_file_version,
                        sha256=None, size=e.size, license=e.license, kind="images_embedded",
                        dataset=e.dataset,
                        note=f"Box file {e.box_file_id} version {e.box_file_version} sha1 {e.box_sha1}; "
                             "only non-image members are stored (see tar_members.json)")


def fetch_tars(ids, root, *, tars=None, reserve=RESERVE_BYTES, opener=None, max_episodes=None) -> list[dict]:
    """Stream catalogued RoboCasa tars (ids or :class:`TarEntry`) and keep non-image members.

    ``max_episodes`` keeps ``extras/`` of only the first N episodes met in tar order
    (data parquet and metadata are always kept; the whole tar is still streamed and
    verified). Returns ledger records (one per tar; ``stripped.path`` is the manifest).
    A tar whose manifest already exists and matches the catalogued SHA-1 is skipped.
    """
    tars = tars or load_tars()
    entries = [tars[i] if isinstance(i, str) else i for i in ids]
    open_url = opener or urllib.request.urlopen
    root = Path(root)
    out = []
    for e in entries:
        dest = e.local_dir(root)
        manifest = dest / MANIFEST
        if manifest.exists():
            man = json.loads(manifest.read_text())
            if man.get("box_sha1") == e.box_sha1 and man.get("tar_sha1_verified"):
                out.append(_ledger(root, e, man, manifest))
                continue
        dest.mkdir(parents=True, exist_ok=True)
        # A conservative bound: non-image members are at most the tar size.
        if shutil.disk_usage(dest).free - e.size < reserve:
            raise InsufficientDisk(f"{e.id}: storing up to {e.size} bytes would leave < {reserve} bytes free")
        stream = _ResumingStream(e.url, open_url)
        kept, dropped, episodes, written = [], {"members": 0, "bytes": 0}, [], []
        try:
            with tarfile.open(fileobj=io.BufferedReader(stream, CHUNK), mode="r|") as tar:
                for mem in tar:
                    if not mem.isfile():
                        continue
                    if is_image_member(mem.name):
                        dropped["members"] += 1
                        dropped["bytes"] += mem.size
                        continue  # tarfile skips the member's data in stream mode
                    mt = re.search(r"/extras/(episode_\d+)/", mem.name)
                    if mt and mt[1] not in episodes:
                        if max_episodes is not None and len(episodes) >= max_episodes:
                            continue
                        episodes.append(mt[1])
                    if shutil.disk_usage(dest).free - mem.size < reserve:
                        raise InsufficientDisk(f"{e.id}: free space fell below reserve at {mem.name}")
                    p = dest / mem.name
                    if not p.resolve().is_relative_to(dest.resolve()):
                        raise ValueError(f"{e.id}: unsafe member path {mem.name}")
                    p.parent.mkdir(parents=True, exist_ok=True)
                    h = hashlib.sha256()
                    with tar.extractfile(mem) as src, open(p, "wb") as dst:
                        for block in iter(lambda: src.read(CHUNK), b""):
                            h.update(block)
                            dst.write(block)
                    written.append(p)
                    kept.append({"member": mem.name, "tar_offset": mem.offset_data, "size": mem.size,
                                 "sha256": h.hexdigest()})
            stream.drain()  # trailing zero blocks
        finally:
            stream.close()
        if stream.pos != e.size or stream.sha1.hexdigest() != e.box_sha1:
            for p in written:
                p.unlink(missing_ok=True)
            raise ChecksumMismatch(f"{e.id}: streamed {stream.pos} bytes sha1 {stream.sha1.hexdigest()}, "
                                   f"expected {e.size} bytes sha1 {e.box_sha1}; members removed")
        man = {"tar_id": e.id, "dataset": e.dataset, "url": e.url, "shared_link": e.shared_link,
               "box_file_id": e.box_file_id, "box_file_version": e.box_file_version, "box_sha1": e.box_sha1,
               "tar_size": e.size, "tar_sha1_verified": True, "tar_sha256": stream.sha256.hexdigest(),
               "selection": "non-image members" + ("" if max_episodes is None else
                                                   f"; extras of the first {max_episodes} episodes in tar order"),
               "extras_episodes": len(episodes), "dropped_image_members": dropped, "members": kept}
        tmp = manifest.with_name(f".{MANIFEST}.tmp-{os.getpid()}")
        tmp.write_text(json.dumps(man, indent=1))
        os.replace(tmp, manifest)
        out.append(_ledger(root, e, man, manifest))
    return out


def _ledger(root, e: TarEntry, man: dict, manifest: Path) -> dict:
    info = {"path": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "bytes": sum(m["size"] for m in man["members"]), "original_deleted": True,
            "original_bytes": e.size, "dropped_datasets": man["dropped_image_members"]["members"],
            "dropped_names": ["videos/**"], "dropped_stored_bytes": man["dropped_image_members"]["bytes"],
            "kept_datasets": len(man["members"]), "rule": "tar members under videos/ or images/ or with an "
                                                          "image/video extension are never written",
            "box_sha1_verified": man["box_sha1"]}
    return _record(root, _tar_record_entry(e), man["tar_sha256"], e.size, "tofu", stripped=info)


def read_manifest(dataset_dir) -> dict | None:
    """``tar_members.json`` of an extracted tar directory (or its ``lerobot`` child)."""
    d = Path(dataset_dir)
    for p in (d / MANIFEST, d.parent / MANIFEST):
        if p.exists():
            return json.loads(p.read_text())
    return None


# ---------------------------------------------------------------- asset subsets

class _HTTPFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP Range requests."""

    def __init__(self, url, open_url):
        self.open_url, self.pos = open_url, 0
        with open_url(urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})) as r:
            self.url = r.geturl() if hasattr(r, "geturl") else url
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"User-Agent": USER_AGENT,
                                                        "Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with self.open_url(req) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch_asset_subset(rels, root, *, catalog=None, reserve=RESERVE_BYTES, opener=None) -> dict:
    """Copy the asset members ``rels`` (paths relative to ``robocasa/models/assets/``)
    out of the catalogued RoboCasa asset zips into ``<root>/raw/robocasa/asset_subset``.

    Each member is read over HTTP Range and checked against the CRC-32 of the zip
    central directory. ``asset_subset/manifest.json`` records, per member, the source
    archive id, its pinned digest, and the member's CRC and SHA-256. Members found in
    no archive are listed as missing. Returns the manifest.
    """
    from ..acquire import load_catalog
    from .robocasa import asset_sources_from_catalog

    catalog = catalog or load_catalog()
    open_url = opener or urllib.request.urlopen
    base = Path(root) / "raw" / FAMILY / SUBSET_DIR
    base.mkdir(parents=True, exist_ok=True)
    mpath = base / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"members": {}, "missing": []}
    todo = sorted({posixpath.normpath(r) for r in rels} - set(man["members"]))
    zips = {}
    for spec in asset_sources_from_catalog(catalog):
        if spec["kind"] != "zip":
            continue
        e = catalog[spec["id"]]
        names = None
        for rel in list(todo):
            member = spec["member"](rel)
            if member is None:
                continue
            if names is None:
                zips[e.id] = zipfile.ZipFile(io.BufferedReader(_HTTPFile(e.url, open_url), 1 << 16))
                names = {i.filename: i for i in zips[e.id].infolist()}
            info = names.get(member)
            if info is None:
                continue
            if shutil.disk_usage(base).free - info.file_size < reserve:
                raise InsufficientDisk(f"{rel}: would leave < {reserve} bytes free")
            data = zips[e.id].read(info)
            if zlib.crc32(data) != info.CRC:
                raise ChecksumMismatch(f"{e.id}:{member}: CRC mismatch")
            p = base / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            man["members"][rel] = {"archive_id": e.id, "archive_sha256": e.sha256, "archive_note": e.note,
                                   "member": member, "crc32": info.CRC, "size": info.file_size,
                                   "sha256": hashlib.sha256(data).hexdigest()}
            todo.remove(rel)
    man["missing"] = sorted(set(man.get("missing", [])) - set(man["members"]) | set(todo))
    tmp = mpath.with_name(f".manifest.json.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(man, indent=1, sort_keys=True))
    os.replace(tmp, mpath)
    return man


# ---------------------------------------------------------------- catalog generation

ROBOCASA_REPO = "https://github.com/robocasa/robocasa"
CATALOG_HEADER = """\
# Pinned RoboCasa365 files. Generated by reachy_retarget.sources.robocasa_fetch.generate_catalog
# from publisher metadata (RoboCasa dataset registry and Box link tables at the pinned commit,
# Box file metadata, HF LFS sha256, PyPI digests); edit by regenerating, not by hand.
# 'sources' are whole files usable with acquire.fetch() (asset archives; kind: assets).
# 'robocasa_tars' are the per-task LeRobot tars on Box. They embed MP4 videos and must go
# through sources.robocasa_fetch.fetch_tars(), which streams and verifies the whole tar
# (size + Box sha1) and stores only non-image members. Box publishes no sha256.
# MimicGen tars (source: mg) checked so far carry no extras/ (no MJCF, no MuJoCo states): only
# parquet robot observations/actions, so the robocasa adapter cannot replay them.
"""
ASSET_REPO = "robocasa/robocasa-assets"


def zip_marker(name: str) -> str:
    """Recorded-path fragment after which a reference equals the asset zip's member name.

    Object zips (``objaverse/``, ``aigen_objs/``, ``lightwheel/`` members) are unpacked
    by RoboCasa into ``models/assets/objects/``, the others into ``models/assets/``.
    """
    stem = posixpath.basename(name).removesuffix(".zip")
    return ASSET_MARKER + ("objects/" if stem in ("objaverse", "aigen_objs", "objects_lightwheel") else "")


def _get(url, open_url, headers=None) -> bytes:
    with open_url(urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})) as r:
        return r.read()


def box_metadata(shared_link, open_url=None) -> dict:
    """File id, version id, SHA-1, name and size of a public Box shared file (page scrape)."""
    html = _get(shared_link, open_url or urllib.request.urlopen).decode("utf-8", "replace")
    m = re.search(r'"preview_metadata":\{"type":"file","id":"(\d+)".*?"sha1":"([0-9a-f]{40})","file_version":'
                  r'\{"type":"file_version","id":"(\d+)","sha1":"([0-9a-f]{40})"\},"name":"([^"]*)","size":(\d+)', html)
    if not m:
        raise ValueError(f"{shared_link}: no Box file metadata found")
    return {"box_file_id": m[1], "box_sha1": m[2], "box_file_version": m[3], "name": m[5], "size": int(m[6])}


def static_url(shared_link: str, ext: str) -> str:
    """``https://utexas.box.com/s/<id>`` -> ``https://utexas.box.com/shared/static/<id>.<ext>``."""
    return f"{shared_link.split('/s/')[0]}/shared/static/{shared_link.rstrip('/').rsplit('/', 1)[1]}.{ext}"


def parse_registry(text: str) -> dict:
    """``{tar key: (task, variant, horizon)}`` from ``robocasa/utils/dataset_registry.py``."""
    out = {}
    for block in re.finditer(r"\n    (\w+)=dict\((.*?)\n    \),", text, re.S):
        task, body = block[1], block[2]
        hz = re.search(r"horizon=(\d+)", body)
        for var, p in re.findall(r'(\w+_path)="(v1\.0/[^"]+)"', body):
            out[p.split("v1.0/", 1)[1].rstrip("/") + "/lerobot.tar"] = (task, var, int(hz[1]) if hz else None)
    return out


def generate_catalog(out_path, *, commit, asset_revision, robosuite_version="1.5.2", opener=None,
                     workers=8) -> dict:
    """Rebuild ``robocasa.yaml`` from publisher metadata (network; explicit call only).

    Reads the dataset registry and Box link tables at RoboCasa ``commit``, scrapes Box
    metadata of every tar and Box-only asset zip, the HF LFS digests of
    ``robocasa/robocasa-assets`` at ``asset_revision``, the PyPI digest of the robosuite
    wheel and hashes the GitHub source archive of ``commit``.
    """
    from concurrent.futures import ThreadPoolExecutor

    open_url = opener or urllib.request.urlopen
    raw = f"https://raw.githubusercontent.com/robocasa/robocasa/{commit}/robocasa"
    registry = parse_registry(_get(f"{raw}/utils/dataset_registry.py", open_url).decode())
    links_ds = json.loads(_get(f"{raw}/models/assets/box_links/box_links_ds.json", open_url))
    links_assets = json.loads(_get(f"{raw}/models/assets/box_links/box_links_assets.json", open_url))
    humans = {(t, k.split("/")[0]): f"robocasa/v1.0/{posixpath.dirname(k)}"
              for k, (t, var, _) in registry.items() if var == "human_path"}
    with ThreadPoolExecutor(workers) as ex:
        metas = dict(zip(links_ds, ex.map(lambda v: box_metadata(v, open_url), links_ds.values())))
    tars = []
    for key, link in sorted(links_ds.items()):
        md = metas[key]
        split, task_type, task = key.split("/")[:3]
        reg = registry.get(key)
        variant = reg[1] if reg else ("mg_path" if "/mg/" in key else "human_path")
        source = "mg" if "/mg/" in key else "human"
        path = f"v1.0/{key}"
        seed, note = None, None
        if source == "mg":
            seed = f"robocasa/v1.0/{key.split('/mg/')[0]}"
            human = humans.get((task, split))
            if human and human != seed:
                note = (f"MimicGen folder is not under its task's human dataset; seed taken from the registry's "
                        f"human_path ({human}), the source demos may be an earlier collection")
                seed = human
        tars.append({k: v for k, v in dict(
            path=path, dataset=f"robocasa/{posixpath.dirname(path)}", task=task, split=split,
            task_type=task_type, source=source, variant=variant if reg else f"{variant} (not in registry)",
            url=static_url(link, "tar"), shared_link=link, box_file_id=md["box_file_id"],
            box_file_version=md["box_file_version"], box_sha1=md["box_sha1"], size=md["size"],
            seed=seed, horizon=reg[2] if reg else None, note=note).items() if v is not None})
    hf = json.loads(_get(f"https://huggingface.co/api/datasets/{ASSET_REPO}/tree/{asset_revision}", open_url))
    hf_files = [{"path": f"assets/{f['path']}", "url": f"https://huggingface.co/datasets/{ASSET_REPO}/resolve/"
                 f"{asset_revision}/{f['path']}", "sha256": f["lfs"]["oid"], "size": f["size"], "kind": "assets",
                 "asset_marker": zip_marker(f["path"])}
                for f in hf if f["path"].endswith(".zip")]
    hf_names = {posixpath.basename(f["path"]) for f in hf_files}
    box_files = []
    for name, link in sorted(links_assets.items()):
        if f"{name}.zip" in hf_names:
            continue
        md = box_metadata(link, open_url)
        stream = _ResumingStream(static_url(link, "zip"), open_url)
        try:
            stream.drain()
        finally:
            stream.close()
        if stream.pos != md["size"] or stream.sha1.hexdigest() != md["box_sha1"]:
            raise ChecksumMismatch(f"{name}.zip: streamed bytes do not match Box size/sha1")
        box_files.append({"path": f"assets/{name}.zip", "url": static_url(link, "zip"),
                          "sha256": stream.sha256.hexdigest(), "size": md["size"], "kind": "assets",
                          "asset_marker": zip_marker(f"{name}.zip"),
                          "note": f"Box file {md['box_file_id']} version {md['box_file_version']} sha1 "
                                  f"{md['box_sha1']}; sha256 computed at catalog time from a stream whose size "
                                  "and sha1 matched Box (Box publishes no SHA-256)"})
    code = _get(f"https://codeload.github.com/robocasa/robocasa/zip/{commit}", open_url)
    pypi = json.loads(_get(f"https://pypi.org/pypi/robosuite/{robosuite_version}/json", open_url))
    whl = next(u for u in pypi["urls"] if u["filename"].endswith("-py3-none-any.whl"))
    doc = {"sources": [
        {"family": FAMILY, "repo": f"https://huggingface.co/datasets/{ASSET_REPO}", "revision": asset_revision,
         "license": "CC-BY-4.0", "files": hf_files},
        {"family": FAMILY, "repo": "box_links_assets.json", "revision": commit, "license": "CC-BY-4.0",
         "files": box_files},
        {"family": FAMILY, "repo": ROBOCASA_REPO, "revision": commit, "license": "MIT",
         "files": [{"path": f"robocasa-{commit}.zip", "url": f"https://codeload.github.com/robocasa/robocasa/zip/{commit}",
                    "sha256": hashlib.sha256(code).hexdigest(), "size": len(code), "kind": "assets",
                    "asset_marker": ASSET_MARKER,
                    "note": "GitHub source archive; digest computed at catalog time (GitHub publishes none). "
                            "Code MIT; the models/assets tree inside is CC-BY-4.0 (README)."}]},
        {"family": "robosuite", "repo": "https://pypi.org/project/robosuite/", "revision": robosuite_version,
         "license": "MIT", "files": [{"path": whl["filename"], "url": whl["url"], "sha256": whl["digests"]["sha256"],
                                      "size": whl["size"], "kind": "assets",
                                      "asset_marker": "robosuite/models/assets/"}]}],
        "robocasa_tars": {"revision": commit, "license": "CC-BY-4.0", "tars": tars,
                          "registry_without_download": sorted(f"v1.0/{k}" for k in registry if k not in links_ds)}}
    Path(out_path).write_text(CATALOG_HEADER + yaml.safe_dump(doc, sort_keys=False, width=120))
    return doc


__all__ = ["TarEntry", "load_tars", "fetch_tars", "read_manifest", "fetch_asset_subset", "generate_catalog",
           "box_metadata", "parse_registry", "static_url", "is_image_member"]
