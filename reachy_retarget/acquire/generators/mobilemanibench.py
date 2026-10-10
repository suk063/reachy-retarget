"""Catalog generator for MobileManiBench (``arnoldland/MobileManiBench``), G1 robot only.

Network use only when invoked::

    python -m reachy_retarget.acquire.generators.mobilemanibench --cache <dir> [--out <catalog.yaml>]

Every G1 per-object tar listed in ``object_manifest.jsonl`` is indexed by reading its tar
headers with small HTTP ranges (:func:`..common.index_remote_tar`); the member count must
equal the publisher's ``nfiles``. Selected members (``range_member`` rows of
``mobilemanibench.members.tsv.gz``): every episode's ``state_infos.pkl`` and the per-run
configuration (``params/{env,agent}.yaml``, ``git/*.diff``, run ``log.txt``) and per
trajectory batch ``scene_infos.json``/``log.txt``. Never videos (``*.mp4``), policy
checkpoints (``*.pt``/``*.onnx``) or tensorboard events. Member digests are not published
(only whole-tar SHA-256); members first fetched by the earlier subset keep the SHA-256
computed then, the others are verified at fetch time by the tar header that precedes
their bytes (name, size, header checksum) and recorded trust-on-first-use.

PartNet-Mobility object assets (``--assets``, network: the zip central directory of
``Assets/Assets.zip`` read over HTTP Range): for every PartNet-Mobility object referenced by a
catalogued episode, the members ``Assets/partnet/dataset/<id>/`` ``mobility.urdf``,
``textured_objs/*`` (OBJ meshes and MTL materials), ``images/*`` (their textures: mesh assets, not
observations), the small metadata files (``meta.json``, ``result.json``, ``semantics.txt``,
``mobility_v2.json``, ``bounding_box.json``) and ``Assets/partnet/process/<group>/<id>/config.yaml``
(``fix_base``) as ``range_member`` rows (deflate or stored, zip CRC-32) of
``mobilemanibench.assets.tsv.gz``. Rendered part images, point samples and the Isaac USD files are
never catalogued.
"""
from __future__ import annotations

import argparse
import json
import posixpath
import re
import struct
import sys
import time
from pathlib import Path

import yaml

from ..transports import KeepAliveOpener, is_image_name, read_range
from .common import hf_resolve, index_remote_tar, write_table

FAMILY = "mobilemanibench"
REPO = "arnoldland/MobileManiBench"
REVISION = "88bc86eed0162c3b9a27bc962ce2d4815cbf2e59"
TABLE = "mobilemanibench.members.tsv.gz"
SKIP_SUFFIX = (".mp4", ".pt", ".onnx", ".npz", ".png", ".jpg")
ASSETS_TABLE = "mobilemanibench.assets.tsv.gz"
ASSETS_ZIP = "Assets/Assets.zip"
PARTNET_META = ("mobility.urdf", "meta.json", "result.json", "semantics.txt", "mobility_v2.json", "bounding_box.json")
PARTNET_LICENSE = ("PartNet-Mobility terms of use (SAPIEN: non-commercial research and educational use only; ShapeNet "
                   "terms apply), re-hosted in arnoldland/MobileManiBench Assets/Assets.zip")
EPISODE_OBJECT = re.compile(r"/partnet/([^/]+)/\d+/(\d+)-joint_\d+-")


def zip_directory(url: str, size: int, opener=None) -> list[tuple]:
    """``[(name, method, crc32, compressed size, size, local header offset)]`` of a remote (zip64) zip."""
    opener = opener or KeepAliveOpener()
    tail = read_range(url, size - 65536, 65536, opener)
    i = tail.rfind(b"PK\x05\x06")
    _, _, _, _, n, cdsize, cdoff, _ = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    if 0xFFFFFFFF in (cdoff, cdsize) or n == 0xFFFF:
        j = tail.rfind(b"PK\x06\x06")
        *_, n, cdsize, cdoff = struct.unpack("<IQHHIIQQQQ", tail[j:j + 56])
    cd = read_range(url, cdoff, cdsize, opener)
    rows, p = [], 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        h = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
        method, crc, csz, usz, fl, el, cl, off = h[4], h[7], h[8], h[9], h[10], h[11], h[12], h[16]
        name = cd[p + 46:p + 46 + fl].decode("utf-8", "replace")
        extra, q = cd[p + 46 + fl:p + 46 + fl + el], 0
        while q + 4 <= len(extra):
            hid, hsz = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                vals, k = extra[q + 4:q + 4 + hsz], 0
                for field in ("usz", "csz", "off"):
                    if {"usz": usz, "csz": csz, "off": off}[field] == 0xFFFFFFFF:
                        v = struct.unpack("<Q", vals[k:k + 8])[0]
                        k += 8
                        usz, csz, off = (v, csz, off) if field == "usz" else (usz, v, off) if field == "csz" else (usz, csz, v)
            q += 4 + hsz
        rows.append((name, method, crc, csz, usz, off))
        p += 46 + fl + el + cl
    if len(rows) != n:
        raise IOError(f"{url}: parsed {len(rows)} central directory entries, the end record says {n}")
    return rows


def partnet_member(name: str, objects: set) -> str | None:
    """Content class of an ``Assets.zip`` member to catalogue for the PartNet objects ``objects``
    (``{(group, id)}``), or ``None``."""
    m = re.fullmatch(r"Assets/partnet/dataset/(\d+)/(.+)", name)
    if m and m[1] in {i for _, i in objects} and not name.endswith("/"):
        rest = m[2]
        if rest in PARTNET_META:
            return "assets" if rest.endswith(".urdf") else "metadata"
        if rest.startswith("textured_objs/") and rest.lower().endswith((".obj", ".mtl")):
            return "assets"
        if rest.startswith("images/") and rest.lower().endswith((".png", ".jpg", ".jpeg")):
            return "assets"
        return None
    m = re.fullmatch(r"Assets/partnet/process/([^/]+)/(\d+)/config\.yaml", name)
    return "metadata" if m and (m[1], m[2]) in objects else None


def build_assets(out_yaml=None, opener=None) -> dict:
    """Add (or rebuild) the PartNet-Mobility asset group: members of ``Assets/Assets.zip`` for every
    PartNet object referenced by a catalogued episode."""
    from ..catalog import CATALOG_DIR, load_catalog

    out_yaml = Path(out_yaml or CATALOG_DIR / f"{FAMILY}.yaml")
    doc = yaml.safe_load(out_yaml.read_text())
    cat = load_catalog(out_yaml)
    objects = {(m[1], m[2]) for e in cat.values() if e.family == FAMILY and (m := EPISODE_OBJECT.search(e.path))}
    known = {e.path: e.sha256 for e in cat.values() if e.sha256}
    members = next(g for g in doc["sources"] if g.get("table") == TABLE)
    archive = members["archives"][ASSETS_ZIP]
    rows, counts = [], {"objects": len(objects), "files": 0, "bytes": 0, "stored_bytes": 0}
    for name, method, crc, csz, usz, off in zip_directory(archive["url"], archive["size"], opener):
        content = partnet_member(name, objects)
        if content is None:
            continue
        if method not in (0, 8):
            raise ValueError(f"{name}: unsupported zip method {method}")
        parts = name.split("/")
        group, oid = (parts[3], parts[4]) if parts[2] == "process" else (next(g for g, i in sorted(objects)
                                                                              if i == parts[3]), parts[3])
        rows.append({"path": name, "archive": ASSETS_ZIP, "offset": off, "size": usz, "stored_size": csz,
                     "compression": "deflate" if method == 8 else "stored", "crc32": crc, "content": content,
                     "dataset": f"{FAMILY}/partnet/{group}/{oid}", "sha256": known.get(name)})
        counts["files"] += 1
        counts["bytes"] += usz
        counts["stored_bytes"] += csz
    rows.sort(key=lambda r: r["path"])
    write_table(out_yaml.parent / ASSETS_TABLE, ["path", "archive", "offset", "size", "stored_size", "compression",
                                                 "crc32", "content", "dataset", "sha256"], rows)
    groups = [g for g in doc["sources"] if g.get("table") != ASSETS_TABLE]
    groups.append({
        "family": FAMILY, "repo": f"https://huggingface.co/datasets/{REPO}", "revision": REVISION,
        "license": PARTNET_LICENSE, "kind": "range_member",
        "note": "PartNet-Mobility object assets of the catalogued episodes' objects: URDF, OBJ meshes, MTL materials and "
                "their textures (mesh assets, not observations), metadata and the Isaac process config.yaml (fix_base); "
                "members of Assets/Assets.zip read by byte range (local header offset), zip CRC-32 checked, sha256 "
                "recorded trust-on-first-use. Rendered part images, point samples and USD files are not catalogued.",
        "archives": {ASSETS_ZIP: archive}, "table": ASSETS_TABLE, "totals": counts})
    header = "".join(line + "\n" for line in out_yaml.read_text().splitlines() if line.startswith("#"))
    out_yaml.write_text(header + yaml.safe_dump({**doc, "sources": groups}, sort_keys=False, width=120))
    return counts


def wanted(name: str) -> str | None:
    """Content class of a member to catalogue, or ``None``."""
    base = posixpath.basename(name)
    if name.endswith(SKIP_SUFFIX) or base.startswith("events.out.tfevents") or is_image_name(name):
        return None
    if base == "state_infos.pkl":
        return "low_dim"
    if base in ("env.yaml", "agent.yaml", "scene_infos.json", "log.txt") or base.endswith(".diff"):
        return "metadata"
    return None


def manifest_rows(path=None) -> list[dict]:
    """G1 rows of the publisher's ``object_manifest.jsonl`` (local copy or the pinned URL)."""
    if path and Path(path).exists():
        text = Path(path).read_text()
    else:
        with KeepAliveOpener()(__import__("urllib.request").request.Request(
                hf_resolve(REPO, REVISION, "MobileManiDataset/object_manifest.jsonl"))) as r:
            text = r.read().decode()
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [r for r in rows if r["robot"] == "G1_Robot"]


def index_tars(cache, manifest=None, workers=48, parallel_tars=6, log=print) -> None:
    """Index every G1 tar into ``<cache>/<tar name>.json`` (skips tars already indexed)."""
    from concurrent.futures import ThreadPoolExecutor

    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    opener = KeepAliveOpener()

    def one(row):
        out = cache / (row["repo_path"].replace("/", "__") + ".json")
        if out.exists():
            return
        url = hf_resolve(REPO, REVISION, row["repo_path"])
        size = row["tar_bytes"]
        t = time.time()
        members = index_remote_tar(url, size, opener=opener, segment=256 << 20, workers=workers, scan=1 << 20)
        files = [m for m in members if m[3] in ("0", "\x00", "7")]
        if len(files) != row["nfiles"]:
            raise IOError(f"{row['repo_path']}: indexed {len(files)} files, manifest nfiles {row['nfiles']}")
        out.write_text(json.dumps({"archive": row["repo_path"], "size": size, "sha256": row["sha256"],
                                   "nfiles": row["nfiles"], "members": files}))
        log(f"{row['repo_path']}: {len(files)} files in {time.time() - t:.0f} s")

    rows = sorted(manifest_rows(manifest), key=lambda r: -r["tar_bytes"])  # largest first
    with ThreadPoolExecutor(parallel_tars) as ex:
        for fut in [ex.submit(one, r) for r in rows]:
            fut.result()


def build(cache, out_yaml=None, manifest=None) -> dict:
    """Write ``mobilemanibench.yaml`` (whole-file groups kept) and the member table."""
    from ..catalog import CATALOG_DIR, load_catalog

    out_yaml = Path(out_yaml or CATALOG_DIR / f"{FAMILY}.yaml")
    old = yaml.safe_load(out_yaml.read_text())
    groups = [g for g in old["sources"] if g.get("kind", "file") == "file"]
    assets = [g for g in old["sources"] if g.get("table") == ASSETS_TABLE]   # kept; rebuilt by --assets
    members = next(g for g in old["sources"] if g.get("table") == TABLE)
    known = {e.path: e.sha256 for e in load_catalog(out_yaml).values() if e.sha256}  # digests of earlier fetches
    zip_members = [f for f in members.get("files", []) if f.get("compression") == "deflate"]
    zip_archives = {f["archive"]: members["archives"][f["archive"]] for f in zip_members}
    rows, archives, counts = [], {}, {"tars": 0, "episodes": 0, "bytes": 0, "files": 0}
    for row in manifest_rows(manifest):
        idx = json.loads((Path(cache) / (row["repo_path"].replace("/", "__") + ".json")).read_text())
        archives[row["repo_path"]] = {"url": hf_resolve(REPO, REVISION, row["repo_path"]), "sha256": row["sha256"],
                                      "size": row["tar_bytes"], "format": "tar", "nfiles": row["nfiles"]}
        counts["tars"] += 1
        group = f"{FAMILY}/{posixpath.dirname(row['repo_path']).split('MobileManiDataset/')[1]}/{row['object']}"
        for off, size, name, _ in idx["members"]:
            content = wanted(name)
            if content is None:
                continue
            rows.append({"path": name, "archive": row["repo_path"], "offset": off, "size": size, "content": content,
                         "dataset": group, "sha256": known.get(name)})
            counts["files"] += 1
            counts["bytes"] += size
            counts["episodes"] += content == "low_dim"
    rows.sort(key=lambda r: r["path"])
    write_table(out_yaml.parent / TABLE, ["path", "archive", "offset", "size", "content", "dataset", "sha256"], rows)
    groups.append({
        "family": FAMILY, "repo": f"https://huggingface.co/datasets/{REPO}", "revision": REVISION, "license": "MIT",
        "kind": "range_member",
        "note": "Single members of the per-object G1 tars (byte ranges; the tars also hold MP4 videos, policy "
                "checkpoints and tensorboard events, which are never requested) and the robot URDF from "
                "Assets/Assets.zip (deflate member, CRC-32 checked). Archive sha256 = publisher "
                "(object_manifest.jsonl = HF LFS oid). Member sha256 is published for none of them: rows with a "
                "sha256 were hashed when the earlier subset was fetched; the others are checked against the tar "
                "header preceding their bytes and recorded trust-on-first-use.",
        "archives": {**archives, **zip_archives},
        "files": zip_members,
        "table": TABLE,
        "totals": counts})
    header = (f"# Pinned MobileManiBench files (family '{FAMILY}'). Generated by\n"
              "# reachy_retarget.acquire.generators.mobilemanibench (tar headers of every G1 tar read over HTTP Range at\n"
              "# the pinned HF revision, member counts checked against object_manifest.jsonl); edit by regenerating.\n"
              "# kind: file = whole files; range_member = one archive member read by byte range (videos never).\n"
              "# The XHand (dexterous hand) tars are excluded. The member rows are in the gzip TSV named by 'table'.\n")
    out_yaml.write_text(header + yaml.safe_dump({"sources": groups + assets}, sort_keys=False, width=120))
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", help="directory for per-tar header indexes (resumable; required unless --assets)")
    ap.add_argument("--manifest", help="local copy of object_manifest.jsonl")
    ap.add_argument("--out", help="catalog yaml to write (default: the packaged catalog)")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--index-only", action="store_true")
    ap.add_argument("--assets", action="store_true", help="only (re)build the PartNet-Mobility asset group")
    a = ap.parse_args(argv)
    if a.assets:
        print(json.dumps(build_assets(a.out)))
        return
    if not a.cache:
        ap.error("--cache is required unless --assets")
    index_tars(a.cache, a.manifest, a.workers, log=lambda s: print(s, flush=True))
    if not a.index_only:
        print(json.dumps(build(a.cache, a.out, a.manifest)))


if __name__ == "__main__":
    sys.exit(main())
