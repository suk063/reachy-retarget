"""Offline tests of image stripping and of ``fetch(strip_images=True)`` (fake opener)."""
import hashlib
import importlib
import io
import json

import h5py
import numpy as np
import pytest

from reachy_retarget.acquire import (STRIP_ATTR, CatalogEntry, ChecksumMismatch, fetch, identify,
                                     read_ledger, read_strip_record, sha256_file, strip_images,
                                     stripped_path)
from reachy_retarget.acquire.strip import is_image

fetch_mod = importlib.import_module("reachy_retarget.acquire.fetch")
T = 6


def write_source(path):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.attrs["top"] = "kept"
        d = f.create_group("data")
        d.attrs["env_args"] = json.dumps({"env_name": "Stack_D0"})
        for k in ("demo_0", "demo_1"):
            g = d.create_group(k)
            g.attrs["model_file"] = "<mujoco/>"
            g.attrs["num_samples"] = T
            g.create_dataset("states", data=rng.normal(size=(T, 5)))
            g.create_dataset("actions", data=rng.normal(size=(T, 7)).astype(np.float32), compression="gzip")
            o = g.create_group("obs")
            o.create_dataset("robot0_eef_pos", data=rng.normal(size=(T, 3)))
            o.create_dataset("agentview_image", data=rng.integers(0, 255, (T, 8, 8, 3), dtype=np.uint8),
                             compression="gzip", chunks=(1, 8, 8, 3))
            o.create_dataset("eye_in_hand_rgb", data=rng.integers(0, 255, (T, 8, 8, 3), dtype=np.uint8))
            o.create_dataset("agentview_depth", data=rng.random((T, 8, 8)).astype(np.float32))
            o.create_dataset("unnamed_volume", data=np.zeros((T, 4, 4), np.uint8))   # uint8 ndim 3
            o.create_dataset("gripper_flags", data=np.zeros((T, 2), np.uint8))       # uint8 ndim 2: kept
            o["eef_alias"] = h5py.SoftLink(o.name + "/robot0_eef_pos")
        m = f.create_group("mask")
        m.create_dataset("train", data=np.array([b"demo_0", b"demo_1"]))


def test_image_rule_matches_episode_writer_guard(tmp_path):
    p = tmp_path / "a.hdf5"
    write_source(p)
    with h5py.File(p) as f:
        o = f["data/demo_0/obs"]
        flags = {k: is_image(o[k].name, o[k]) for k in o if k != "eef_alias"}
    assert flags == {"robot0_eef_pos": False, "agentview_image": True, "eye_in_hand_rgb": True,
                     "agentview_depth": True, "unnamed_volume": True, "gripper_flags": False}


def test_strip_keeps_everything_but_images(tmp_path):
    src, dst = tmp_path / "a.hdf5", tmp_path / "a.state.hdf5"
    write_source(src)
    before = sha256_file(src)
    summary = strip_images(src, dst, original={"sha256": "x" * 64, "catalog_id": "demo/a.hdf5"})
    assert sha256_file(src) == before  # source untouched, not deleted here
    assert summary["dropped_datasets"] == 8 and summary["kept_datasets"] == 9
    assert summary["dropped_names"] == ["/data/*/obs/agentview_depth", "/data/*/obs/agentview_image",
                                        "/data/*/obs/eye_in_hand_rgb", "/data/*/obs/unnamed_volume"]
    with h5py.File(src) as a, h5py.File(dst) as b:
        assert b.attrs["top"] == "kept" and b["data"].attrs["env_args"] == a["data"].attrs["env_args"]
        for k in ("demo_0", "demo_1"):
            ga, gb = a["data"][k], b["data"][k]
            assert gb.attrs["model_file"] == "<mujoco/>"
            np.testing.assert_array_equal(ga["states"][:], gb["states"][:])
            assert gb["actions"].compression == "gzip" and gb["actions"].dtype == np.float32
            assert set(gb["obs"]) == {"robot0_eef_pos", "gripper_flags", "eef_alias"}
            assert isinstance(gb["obs"].get("eef_alias", getlink=True), h5py.SoftLink)
        assert list(b["mask/train"][:]) == [b"demo_0", b"demo_1"]
    rec = read_strip_record(dst)
    assert rec["original"]["sha256"] == "x" * 64 and read_strip_record(src) is None
    assert stripped_path(tmp_path / "x" / "stack.hdf5") == tmp_path / "x" / "stack.state.hdf5"



@pytest.fixture
def source_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_mod, "_free", lambda p: 10**15)
    p = tmp_path / "publisher.hdf5"
    write_source(p)
    payload = p.read_bytes()
    e = CatalogEntry(id="demo/source/stack.hdf5", family="demo", path="source/stack.hdf5",
                     url="https://example.invalid/stack.hdf5", revision="r1",
                     sha256=hashlib.sha256(payload).hexdigest(), size=len(payload), license="CC-BY-4.0",
                     kind="images_embedded", dataset="demo/source/stack")
    log = []

    def opener(req):
        log.append(req.full_url)
        resp = io.BytesIO(payload)
        resp.status = 200
        return resp
    return e, opener, log, tmp_path / "root"


def test_fetch_refuses_images_unless_stripping(source_entry):
    e, opener, log, root = source_entry
    with pytest.raises(PermissionError, match="strip_images"):
        fetch([e], root, opener=opener)
    assert log == []


def test_fetch_strips_records_both_digests_and_deletes_original(source_entry):
    e, opener, log, root = source_entry
    (rec,) = fetch([e], root, opener=opener, strip_images=True)
    orig, out = e.local_path(root), stripped_path(e.local_path(root))
    assert not orig.exists() and out.exists() and len(log) == 1
    assert rec["sha256_verified"] == e.sha256 and rec["bytes"] == e.size and rec["url"] == e.url
    s = rec["stripped"]
    assert s["sha256"] == sha256_file(out) and s["bytes"] == out.stat().st_size < e.size
    assert s["original_deleted"] and s["path"] == rec["local_path"] == "raw/demo/source/stack.state.hdf5"
    assert read_ledger(root)[e.id] == rec
    # The stripped file names its original, so catalog lookup works without the ledger.
    entry, info = identify(out, {e.id: e})
    assert entry == e and info["sha256"] == e.sha256 and info["file_sha256"] == s["sha256"]
    assert json.loads(h5py.File(out).attrs[STRIP_ATTR])["original"]["url"] == e.url
    # Re-fetch: verified against the ledger, no transfer.
    assert fetch([e], root, opener=opener, strip_images=True) == [rec] and len(log) == 1
    with h5py.File(out, "a") as f:
        f.attrs["tampered"] = 1
    with pytest.raises(ChecksumMismatch):
        fetch([e], root, opener=opener, strip_images=True)


def test_fetch_strips_an_already_present_original_without_transfer(source_entry, tmp_path):
    e, opener, log, root = source_entry
    dest = e.local_path(root)
    dest.parent.mkdir(parents=True)
    dest.write_bytes((tmp_path / "publisher.hdf5").read_bytes())
    (rec,) = fetch([e], root, opener=opener, strip_images=True)
    assert log == [] and not dest.exists() and rec["stripped"]["dropped_datasets"] == 8
