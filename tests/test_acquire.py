"""Offline tests of the catalog, the ledger and every fetch kind (an in-process fake opener
replaces HTTP; nothing here touches the network)."""
import gzip
import hashlib
import importlib
import io
import json
import re
import tarfile
import zipfile
import zlib

import pytest

from reachy_retarget.acquire import (KINDS, CatalogEntry, ChecksumMismatch, ImageRefused, InsufficientDisk, fetch,
                                     find_entry, is_image_name, ledger_record_path, load_catalog, locate_entry,
                                     migrate_ledger, read_ledger, read_ledger_record, same_record, select)
from reachy_retarget.acquire.catalog import parse_catalog
from reachy_retarget.acquire.transports import walk_tar
fetch_mod = importlib.import_module("reachy_retarget.acquire.fetch")  # module, not the function

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB


def entry(**kw):
    base = dict(id="demo/a/b.bin", family="demo", path="a/b.bin", url="https://example.invalid/b.bin",
                revision="r1", sha256=hashlib.sha256(PAYLOAD).hexdigest(), size=len(PAYLOAD),
                license="MIT", kind="file", content="low_dim")
    return CatalogEntry(**{**base, **kw})


class FakeResponse(io.BytesIO):
    def __init__(self, data, status, headers=None):
        super().__init__(data)
        self.status, self.headers = status, headers or {}


def opener(log, blobs=None, honour_range=True, status=None):
    """Fake ``urlopen``: serves ``blobs[url]`` (default PAYLOAD), honouring ``Range``."""
    def open_url(req):
        blob = (blobs or {}).get(req.full_url, PAYLOAD)
        rng = req.get_header("Range")
        log.append(rng)
        if rng and honour_range:
            a, b = re.match(r"bytes=(\d+)-(\d*)", rng).groups()
            a, b = int(a), int(b) if b else len(blob) - 1
            return FakeResponse(blob[a:b + 1], status or 206, {"Content-Range": f"bytes {a}-{b}/{len(blob)}"})
        return FakeResponse(blob, status or 200, {"Content-Length": str(len(blob))})
    return open_url


@pytest.fixture
def plenty_of_disk(monkeypatch):
    monkeypatch.setattr(fetch_mod, "_free", lambda p: 10**15)


# ---------------------------------------------------------------- catalog

def test_packaged_catalog_is_pinned_and_explicit():
    cat = load_catalog()
    for e in cat.values():
        assert e.id == f"{e.family}/{e.path}" and e.kind in KINDS and e.license and e.content
        if "huggingface.co" in e.url:
            assert f"/resolve/{e.revision}/" in e.url and re.fullmatch(r"[0-9a-f]{40}", e.revision)
        if e.kind in ("file", "file_images_embedded"):
            assert e.size > 0 and (e.sha256 or e.digests.get("git_blob_sha1"))
        elif e.kind == "tar_stream":
            assert re.fullmatch(r"[0-9a-f]{40}", e.digests["sha1"]) and e.size > 0
        elif e.kind == "range_member":
            assert e.source["archive_sha256"] and e.source["offset"] >= 0 and e.size > 0
        else:
            assert e.source["compression"] == "zstd" and e.source["stored_size"] > 0 and e.source["keep"]
        # never images or videos, except still textures catalogued as mesh assets (AGENTS.md)
        assert not (e.kind in ("file", "range_member") and is_image_name(e.path) and not fetch_mod.is_texture_asset(e))
    datasets = {e.dataset for e in cat.values() if e.family == "robomimic"}
    for task in ["lift", "can", "square", "transport"]:
        assert {f"robomimic/{task}/ph", f"robomimic/{task}/mh"} <= datasets
    for fam in ("mimicgen", "libero", "dexmimicgen"):
        assert all(e.kind == "file_images_embedded" for e in cat.values() if e.family == fam and e.path.endswith(".hdf5"))
    assert all(e.asset_marker for e in cat.values() if e.content == "assets" and e.path.endswith((".zip", ".whl")))
    assert {e.content for e in cat.values() if e.family == "robosuite"} == {"assets"}


def test_catalog_groups_tables_archives_and_select(tmp_path):
    rows = ["path\tarchive\toffset\tsize\tcontent\tsha256", "x/a.pkl\tarc.tar\t512\t10\tlow_dim\t",
            "x/b.yaml\tarc.tar\t2048\t5\tmetadata\t" + "a" * 64]
    with gzip.open(tmp_path / "t.tsv.gz", "wt") as f:
        f.write("\n".join(rows) + "\n")
    (tmp_path / "c.yaml").write_text(
        "sources:\n- family: fam\n  revision: r\n  license: MIT\n  kind: range_member\n"
        "  archives: {arc.tar: {url: 'https://x.invalid/arc.tar', sha256: '" + "b" * 64 + "', size: 9999, format: tar}}\n"
        "  table: t.tsv.gz\n  files:\n  - {path: y/c.bin, archive: arc.tar, offset: 4096, size: 3, content: low_dim, "
        "task: t1, usable: false}\n"
        "- family: fam\n  revision: r\n  license: MIT\n  url_base: 'https://x.invalid/base/'\n"
        "  files:\n  - {path: y/d.json, src: d.json, kind: file, content: metadata, git_blob_sha1: '" + "c" * 40 + "'}\n")
    cat = load_catalog(tmp_path / "c.yaml")
    a = cat["fam/x/a.pkl"]
    assert a.kind == "range_member" and a.sha256 is None and a.source["offset"] == 512 and a.url.endswith("arc.tar")
    assert a.source["archive_sha256"] == "b" * 64 and a.source["archive_format"] == "tar"
    assert cat["fam/x/b.yaml"].sha256 == "a" * 64
    c = cat["fam/y/c.bin"]
    assert c.meta == {"task": "t1"} and c.usable is False
    d = cat["fam/y/d.json"]
    assert d.url == "https://x.invalid/base/d.json" and d.digests == {"git_blob_sha1": "c" * 40}
    assert [e.id for e in select(cat, ["fam/x/"])] == ["fam/x/a.pkl", "fam/x/b.yaml"]
    assert "fam/y/c.bin" not in {e.id for e in select(cat, ["fam/y/"])}  # unusable: only by exact id
    assert [e.id for e in select(cat, ["fam/y/c.bin"])] == ["fam/y/c.bin"]
    with pytest.raises(KeyError):
        select(cat, ["fam/nothing"])
    with pytest.raises(ValueError, match="unknown kind"):
        parse_catalog("sources:\n- {family: f, revision: r, license: x, files: [{path: p, url: u, kind: low_dim}]}\n")


# ---------------------------------------------------------------- kind: file

def test_fetch_resumes_verifies_and_records(tmp_path, plenty_of_disk):
    e = entry()
    part = e.local_path(tmp_path).with_name("b.bin.part")
    part.parent.mkdir(parents=True)
    part.write_bytes(PAYLOAD[:1000])
    log = []
    (rec,) = fetch([e], tmp_path, opener=opener(log))
    assert log == ["bytes=1000-"]
    assert e.local_path(tmp_path).read_bytes() == PAYLOAD and not part.exists()
    assert rec["sha256_verified"] == e.sha256 and rec["sha256_source"] == "catalog"
    assert rec["verified_by"] == ["sha256", "size"] and rec["kind"] == "file" and rec["content"] == "low_dim"
    assert read_ledger(tmp_path)[e.id]["bytes"] == len(PAYLOAD)
    fetch([e], tmp_path, opener=opener(log))  # already present: verified again, no transfer
    assert len(log) == 1
    assert find_entry(e.local_path(tmp_path), {e.id: e}) == e


def test_fetch_restarts_when_range_is_ignored(tmp_path, plenty_of_disk):
    e = entry()
    part = e.local_path(tmp_path).with_name("b.bin.part")
    part.parent.mkdir(parents=True)
    part.write_bytes(b"garbage")
    fetch([e], tmp_path, opener=opener([], honour_range=False))
    assert e.local_path(tmp_path).read_bytes() == PAYLOAD


def test_fetch_rejects_bad_checksum(tmp_path, plenty_of_disk):
    e = entry(sha256="0" * 64)
    with pytest.raises(ChecksumMismatch):
        fetch([e], tmp_path, opener=opener([]))
    assert not e.local_path(tmp_path).exists()
    assert e.local_path(tmp_path).with_name("b.bin.sha256-mismatch").exists()
    assert read_ledger(tmp_path) == {}


def test_fetch_verifies_git_blob_sha1(tmp_path, plenty_of_disk):
    blob = hashlib.sha1(b"blob %d\0" % len(PAYLOAD) + PAYLOAD).hexdigest()
    (rec,) = fetch([entry(sha256=None, digests={"git_blob_sha1": blob})], tmp_path, opener=opener([]))
    assert rec["sha256_source"] == "tofu" and "git_blob_sha1" in rec["verified_by"]
    with pytest.raises(ChecksumMismatch):
        fetch([entry(id="demo/c", path="c", sha256=None, digests={"git_blob_sha1": "0" * 40})], tmp_path,
              opener=opener([]))


def test_fetch_keeps_50gb_reserve(tmp_path, monkeypatch):
    e = entry()
    monkeypatch.setattr(fetch_mod, "_free", lambda p: 50_000_000_000 + len(PAYLOAD) - 1)
    log = []
    with pytest.raises(InsufficientDisk):
        fetch([e], tmp_path, opener=opener(log))
    assert log == []  # refused before any transfer


def test_fetch_refuses_images_and_videos(tmp_path, plenty_of_disk):
    log = []
    with pytest.raises(PermissionError):  # image-embedding HDF5 without stripping
        fetch([entry(kind="file_images_embedded")], tmp_path, opener=opener(log))
    with pytest.raises(PermissionError):  # entries built with the earlier content-class kind
        fetch([entry(kind="images_embedded")], tmp_path, opener=opener(log))
    for name in ("videos/ep0.mp4", "cam/frame_0001.PNG", "rgb.webm"):
        with pytest.raises(ImageRefused):
            fetch([entry(id=f"demo/{name}", path=name)], tmp_path, opener=opener(log))
        with pytest.raises(ImageRefused):
            fetch([entry(id=f"demo/m/{name}", path=f"m/{name}", kind="range_member",
                         source={"offset": 0, "archive": "a.tar", "archive_format": "tar"})],
                  tmp_path, opener=opener(log))
    assert log == [] and not list(tmp_path.rglob("*.*"))
    for name, content in (("cam/frame_0001.png", "low_dim"), ("tex/wood.mp4", "assets"),
                          ("images/wood.png", "assets")):  # observations, videos and image folders stay refused
        with pytest.raises(ImageRefused):
            fetch([entry(id=f"demo/{name}", path=name, content=content)], tmp_path, opener=opener(log))
    assert log == []


def test_fetch_keeps_texture_assets(tmp_path, plenty_of_disk):
    log = []
    e = entry(id="demo/table/textures/wood.png", path="table/textures/wood.png", content="assets")
    fetch([e], tmp_path, opener=opener(log))
    assert (tmp_path / "raw" / "demo" / "table" / "textures" / "wood.png").read_bytes() == PAYLOAD


# ---------------------------------------------------------------- ledger

def test_ledger_is_one_json_record_per_file(tmp_path, plenty_of_disk):
    (rec,) = fetch([entry()], tmp_path, opener=opener([]))
    path = tmp_path / "raw" / "ledger" / "demo" / "a" / "b.bin.json"
    assert ledger_record_path(tmp_path, "demo/a/b.bin") == path
    assert json.loads(path.read_text()) == rec
    assert not (tmp_path / "raw" / "ledger.json").exists()


def test_ledgers_of_private_roots_merge_by_copy(tmp_path, plenty_of_disk):
    """Concurrent jobs write private roots; a no-overwrite copy merges their ledgers."""
    import shutil
    a, b, shared = tmp_path / "a", tmp_path / "b", tmp_path / "shared"
    fetch([entry()], a, opener=opener([]))
    fetch([entry(id="demo/c.bin", path="c.bin")], b, opener=opener([]))
    for job in (a, b):
        shutil.copytree(job, shared, dirs_exist_ok=True, copy_function=shutil.copy2)
    assert set(read_ledger(shared)) == {"demo/a/b.bin", "demo/c.bin"}


def test_legacy_single_file_ledger_is_read_and_migrated(tmp_path):
    rec = {"id": "demo/x.bin", "bytes": 1}
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "ledger.json").write_text(json.dumps({rec["id"]: rec}))
    assert read_ledger_record(tmp_path, "demo/x.bin") == rec
    assert migrate_ledger(tmp_path) == 1 and migrate_ledger(tmp_path) == 0
    assert json.loads(ledger_record_path(tmp_path, "demo/x.bin").read_text()) == rec
    assert read_ledger(tmp_path) == {"demo/x.bin": rec}


def test_records_of_earlier_versions_stay_readable_and_match(tmp_path, plenty_of_disk):
    """A record written by the previous fetcher (content class in 'kind', no 'content') is read,
    verified by `verify`, and identified as the same content as a new record of the file."""
    from reachy_retarget.cli import main
    e = entry(id="demo/old.bin", path="old.bin")
    (new,) = fetch([e], tmp_path / "new", opener=opener([]))
    old = {"asset_marker": None, "bytes": len(PAYLOAD), "dataset": None, "family": "demo",
           "fetched_at": "2026-10-07T21:03:34+00:00", "id": e.id, "kind": "low_dim", "license": "MIT",
           "local_path": "raw/demo/old.bin", "note": None, "path": "old.bin", "revision": "r1", "sha256": e.sha256,
           "sha256_source": "catalog", "sha256_verified": e.sha256, "size": len(PAYLOAD), "url": e.url}
    p = ledger_record_path(tmp_path / "old", e.id)
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(old))
    (tmp_path / "old/raw/demo").mkdir(parents=True)
    (tmp_path / "old/raw/demo/old.bin").write_bytes(PAYLOAD)
    assert read_ledger(tmp_path / "old")[e.id] == old and same_record(old, new)
    with pytest.raises(SystemExit) as done:
        main(["verify", "--root", str(tmp_path / "old")])
    assert done.value.code == 0
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from cluster.job import same_ledger_record
    assert same_ledger_record(p, ledger_record_path(tmp_path / "new", e.id))


# ---------------------------------------------------------------- kind: tar_stream

def _tar(members: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for name, data in members.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


TAR_MEMBERS = {"ds/meta/info.json": b'{"fps": 20}', "ds/videos/cam/episode_000000.mp4": b"\x00" * 40000,
               "ds/data/episode_000000.parquet": b"PAR1" + b"x" * 3000, "ds/frame.png": b"\x89PNG" * 10,
               "ds/extras/episode_000000/states.npz": b"npz" * 100}


def tar_entry(blob, **kw):
    return entry(**{**dict(id="demo/t/x.tar", path="t/x.tar", url="https://x.invalid/x.tar", sha256=None,
                           size=len(blob), kind="tar_stream", digests={"sha1": hashlib.sha1(blob).hexdigest()}), **kw})


def test_tar_stream_keeps_only_non_image_members(tmp_path, plenty_of_disk):
    blob = _tar(TAR_MEMBERS)
    e = tar_entry(blob)
    log = []
    (rec,) = fetch([e], tmp_path, opener=opener(log, {e.url: blob}))
    d = e.output_dir(tmp_path)
    assert not list(d.rglob("*.mp4")) and not list(d.rglob("*.png"))
    man = json.loads((d / "tar_members.json").read_text())
    assert {m["member"] for m in man["members"]} == {"ds/meta/info.json", "ds/data/episode_000000.parquet",
                                                     "ds/extras/episode_000000/states.npz"}
    assert man["dropped_image_members"] == {"members": 2, "bytes": 40040} and man["tar_sha1_verified"]
    for m in man["members"]:
        assert blob[m["tar_offset"]:m["tar_offset"] + m["size"]] == (d / m["member"]).read_bytes()
    assert rec["kind"] == "tar_stream" and rec["local_path"].endswith("t/tar_members.json")
    assert rec["stripped"]["sha1_verified"] == e.digests["sha1"]
    log.clear()
    fetch([e], tmp_path, opener=opener(log, {e.url: blob}))  # verified manifest: no transfer
    assert log == []
    bad = tar_entry(blob, id="demo/u/x.tar", path="u/x.tar", digests={"sha1": "0" * 40})
    with pytest.raises(ChecksumMismatch):
        fetch([bad], tmp_path, opener=opener([], {e.url: blob}))
    assert not [p for p in bad.output_dir(tmp_path).rglob("*") if p.is_file()]


def test_tar_stream_resumes_a_dropped_connection(tmp_path, plenty_of_disk):
    blob = _tar(TAR_MEMBERS)
    e = tar_entry(blob)
    state, log = {"dropped": False}, []

    class Dropping(FakeResponse):
        def readinto(self, b):
            if self.tell() >= 20000:
                raise ConnectionResetError("dropped")
            return super().readinto(memoryview(b)[:max(1, 20000 - self.tell())])

    base = opener(log, {e.url: blob})

    def open_url(req):
        if req.get_header("Range") is None and not state["dropped"]:
            state["dropped"] = True
            log.append(None)
            return Dropping(blob, 200, {"Content-Length": str(len(blob))})
        return base(req)

    fetch([e], tmp_path, opener=open_url)
    assert log == [None, "bytes=20000-"]
    assert (e.output_dir(tmp_path) / "ds/data/episode_000000.parquet").exists()


def test_tar_stream_refuses_without_disk(tmp_path, monkeypatch):
    blob = _tar(TAR_MEMBERS)
    monkeypatch.setattr(fetch_mod, "_free", lambda p: 50_000_000_000 + len(blob) - 1)
    log = []
    with pytest.raises(InsufficientDisk):
        fetch([tar_entry(blob)], tmp_path, opener=opener(log))
    assert log == []


# ---------------------------------------------------------------- kind: range_member

def _archive():
    pkl = b"state" * 100
    tar_b = _tar({"run/ep0/rgb_head.mp4": b"\x00" * 3000, "run/ep0/state_infos.pkl": pkl})
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Assets/robot.urdf", b"<robot name='g'/>" * 50)
    return tar_b, zbuf.getvalue(), pkl


def test_range_member_reads_only_the_member(tmp_path, plenty_of_disk):
    tar_b, zip_b, pkl = _archive()
    blobs = {"https://x.invalid/t.tar": tar_b, "https://x.invalid/a.zip": zip_b}
    walked = list(walk_tar(lambda o, n: tar_b[o:o + n]))
    assert [w[2] for w in walked] == ["run/ep0/rgb_head.mp4", "run/ep0/state_infos.pkl"]
    off, size = walked[1][:2]
    src = {"archive": "t.tar", "archive_format": "tar", "archive_sha256": "0" * 64, "offset": off}
    tofu = entry(id="mmb/run/ep0/state_infos.pkl", family="mmb", path="run/ep0/state_infos.pkl",
                 url="https://x.invalid/t.tar", sha256=None, size=size, kind="range_member", source=src)
    zinfo = zipfile.ZipFile(io.BytesIO(zip_b)).getinfo("Assets/robot.urdf")
    data_z = b"<robot name='g'/>" * 50
    zipped = entry(id="mmb/Assets/robot.urdf", family="mmb", path="Assets/robot.urdf", url="https://x.invalid/a.zip",
                   sha256=hashlib.sha256(data_z).hexdigest(), size=len(data_z), kind="range_member",
                   content="robot_model", source={"archive": "a.zip", "archive_format": "zip",
                                                  "offset": zinfo.header_offset, "stored_size": zinfo.compress_size,
                                                  "compression": "deflate", "crc32": zinfo.CRC})
    log = []
    r1, r2 = fetch([tofu, zipped], tmp_path, opener=opener(log, blobs))
    assert (tmp_path / "raw/mmb/run/ep0/state_infos.pkl").read_bytes() == pkl
    assert (tmp_path / "raw/mmb/Assets/robot.urdf").read_bytes() == data_z
    assert r1["sha256_source"] == "tofu" and r1["verified_by"] == ["size", "tar_header"]
    assert r2["sha256_source"] == "catalog" and {"crc32", "sha256"} <= set(r2["verified_by"])
    assert log[0] == f"bytes={off - 512}-{off + size - 1}"  # the header and the member, nothing else
    assert not list(tmp_path.rglob("*.mp4")) and not list(tmp_path.rglob("*.part"))
    assert locate_entry(tmp_path / "raw/mmb/run/ep0/state_infos.pkl", {tofu.id: tofu}) == tofu  # via the ledger
    other_bytes = tmp_path / "x" / "raw/mmb/run/ep0/state_infos.pkl"
    other_bytes.parent.mkdir(parents=True)
    other_bytes.write_bytes(b"subset")
    assert locate_entry(other_bytes, {tofu.id: tofu}) is None  # catalogued place, unrecorded content
    log.clear()
    fetch([tofu], tmp_path, opener=opener(log, blobs))  # matches its first-use record: no transfer
    assert log == []
    wrong = entry(**{**tofu.__dict__, "id": "mmb/w", "path": "w", "source": {**src, "offset": off + 512}})
    with pytest.raises(ChecksumMismatch, match="header"):  # bytes not preceded by the member's header
        fetch([wrong], tmp_path, opener=opener([], blobs))
    other = entry(**{**tofu.__dict__, "id": "mmb/y", "path": "y"})
    with pytest.raises(RuntimeError, match="ignored the byte range"):
        fetch([other], tmp_path, opener=opener([], blobs, honour_range=False))
    assert not (tmp_path / "raw/mmb/y").exists() and not (tmp_path / "raw/mmb/w").exists()


# ---------------------------------------------------------------- kind: range_package

def _zstd_available():
    import shutil
    try:
        import zstandard  # noqa: F401
        return True
    except ImportError:
        return shutil.which("zstd") is not None


def _compress(data: bytes) -> bytes:
    try:
        import zstandard
        return zstandard.ZstdCompressor().compress(data)
    except ImportError:
        import subprocess
        return subprocess.run(["zstd", "-c", "-q"], input=data, capture_output=True, check=True).stdout


@pytest.mark.skipif(not _zstd_available(), reason="needs zstandard or the zstd CLI")
def test_range_package_keeps_only_listed_suffixes(tmp_path, plenty_of_disk):
    h5 = b"\x89HDF\r\n" + b"h" * 5000
    inner = _tar({"house_1/trajectories_batch_1_of_1.h5": h5, "house_1/episode_0_head_camera.mp4": b"\x00" * 4096,
                  "house_1/notes.txt": b"x"})
    blob = _compress(inner)
    shard = b"X" * 100 + blob + b"Y" * 50
    members_total = len(h5) + 4096 + 1  # the publisher's inflated_size: sum of the member sizes
    src = {"archive": "s.tar", "archive_sha256": "0" * 64, "offset": 100, "stored_size": len(blob),
           "inflated_size": members_total, "compression": "zstd", "keep": [".h5"]}
    e = entry(id="mb/Cfg/val/part0/house_1", family="mb", path="Cfg/val/part0/house_1", url="https://x.invalid/s.tar",
              sha256=None, size=None, kind="range_package", source=src, meta={"config": "Cfg"})
    log = []
    recs = fetch([e], tmp_path, opener=opener(log, {e.url: shard}))
    out = tmp_path / "raw/mb/Cfg/val/part0/house_1"
    assert (out / "trajectories_batch_1_of_1.h5").read_bytes() == h5 and log == [f"bytes=100-{100 + len(blob) - 1}"]
    assert sorted(p.name for p in out.iterdir()) == ["package_members.json", "trajectories_batch_1_of_1.h5"]
    member, pkg = recs
    assert member["id"] == "mb/Cfg/val/part0/house_1/trajectories_batch_1_of_1.h5" and member["sha256_source"] == "tofu"
    assert pkg["id"] == e.id and pkg["members"] == [member["id"]] and "inflated_size" in pkg["verified_by"]
    man = json.loads((out / "package_members.json").read_text())
    assert man["range_sha256"] == hashlib.sha256(blob).hexdigest() and man["dropped_image_members"]["members"] == 1
    fetch([e], tmp_path, opener=lambda r: pytest.fail("network used"))  # complete package: no transfer
    found = locate_entry(out / "trajectories_batch_1_of_1.h5", {e.id: e})
    assert found.id == member["id"]
    pinned = entry(**{**e.__dict__, "id": "mb/Cfg/val/part0/house_2", "path": "Cfg/val/part0/house_2",
                      "source": {**src, "range_sha256": "f" * 64}})
    with pytest.raises(ChecksumMismatch):
        fetch([pinned], tmp_path, opener=opener([], {e.url: shard}))
    assert not list((tmp_path / "raw/mb/Cfg/val/part0/house_2").glob("*.h5*"))
    short = entry(**{**e.__dict__, "id": "mb/Cfg/val/part0/house_3", "path": "Cfg/val/part0/house_3",
                     "source": {**src, "inflated_size": members_total + 1}})
    with pytest.raises(ChecksumMismatch, match="inflated"):
        fetch([short], tmp_path, opener=opener([], {e.url: shard}))


@pytest.mark.skipif(not _zstd_available(), reason="needs zstandard or the zstd CLI")
def test_range_package_listed_members_and_refusals(tmp_path, plenty_of_disk, monkeypatch):
    h5 = b"\x89HDF\r\n" + b"k" * 3000
    blob = _compress(_tar({"house_1/trajectories_batch_1_of_1.h5": h5}))
    member = {"name": "house_1/trajectories_batch_1_of_1.h5", "sha256": hashlib.sha256(h5).hexdigest(),
              "size": len(h5), "trajectories": 3}
    src = {"archive": "s.tar", "offset": 0, "stored_size": len(blob), "compression": "zstd", "keep": [".h5"],
           "range_sha256": hashlib.sha256(blob).hexdigest(), "members": [member]}
    e = entry(id="mb/p/house_1", family="mb", path="p/house_1", url="https://x.invalid/s.tar", sha256=None,
              size=None, kind="range_package", source=src)
    member_rec, pkg_rec = fetch([e], tmp_path, opener=opener([], {e.url: blob}))
    assert member_rec["sha256_source"] == "catalog" and member_rec["meta"] == {"trajectories": 3}
    assert pkg_rec["sha256_source"] == "catalog"
    with pytest.raises(RuntimeError, match="ignored the byte range"):
        fetch([entry(**{**e.__dict__, "id": "mb/p/house_9", "path": "p/house_9"})], tmp_path,
              opener=opener([], {e.url: blob}, honour_range=False))
    monkeypatch.setattr(fetch_mod, "_free", lambda p: 0)
    with pytest.raises(InsufficientDisk):
        fetch([entry(**{**e.__dict__, "id": "mb/p/house_8", "path": "p/house_8"})], tmp_path,
              opener=opener([], {e.url: blob}))


# ---------------------------------------------------------------- transports

def test_walk_tar_handles_long_names_and_lookahead():
    long = "a/" + "b" * 150 + "/state_infos.pkl"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for name, data in [(long, b"1" * 700), ("short.json", b"{}")]:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    blob = buf.getvalue()
    reads = []

    def read(o, n):
        reads.append((o, n))
        return blob[o:o + n]

    got = [(n, s) for _, s, n, _ in walk_tar(read, lookahead=4096)]
    assert got == [(long, 700), ("short.json", 2)] and len(reads) <= 2
    for off, size, name, _ in walk_tar(read):
        assert blob[off:off + size] == (b"1" * 700 if name == long else b"{}")


def test_crc_and_image_names():
    assert is_image_name("lerobot/videos/chunk-000/x/episode_000000.mp4")
    assert is_image_name("foo/bar.PNG") and not is_image_name("lerobot/extras/episode_0/states.npz")
    assert zlib.crc32(b"x") >= 0
