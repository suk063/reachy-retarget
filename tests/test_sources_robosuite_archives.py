"""Offline tests of the robosuite reader's multi-archive asset resolution, stripped-file
identity, side hints and MimicGen lineage (fixtures generated in the test)."""
import hashlib
import zipfile
from pathlib import Path

import h5py
import mujoco
import numpy as np
import pytest

from reachy_retarget.acquire import CatalogEntry, load_catalog, strip_images
from reachy_retarget.sources import iter_episodes
from reachy_retarget.sources.robosuite import (AssetArchive, _default_lineage, _seed_task,
                                               _side_hints, resolve_mjcf)

FIXTURE = Path(__file__).parent / "fixtures" / "robomimic_can_ph_demo0_sub.hdf5"

TEX_XML = """<mujoco><asset>
  <texture name="a" type="2d" file="/home/u/robosuite/robosuite/models/assets/textures/ceramic.png"/>
  <texture name="b" type="2d" file="/home/u/mg/mimicgen_envs/models/robosuite/assets/objects/../textures/ceramic.png"/>
  <texture name="c" type="2d" file="/home/u/old/chiliocosm/assets/textures/wood.png"/>
  <texture name="d" type="2d" file="/home/u/robosuite/robosuite/models/assets/textures/wood.png"/>
</asset><worldbody/></mujoco>"""


def _png(rgb: bytes) -> bytes:
    import struct
    import zlib
    chunk = lambda t, b: struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\0" + rgb)) + chunk(b"IEND", b""))


def _zip(path, members):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return path


def test_multi_archive_resolution_aliases_and_basename_collisions(tmp_path):
    gray, red = _png(b"\x80\x80\x80"), _png(b"\xff\x00\x00")
    wheel = AssetArchive(_zip(tmp_path / "rs.whl", {
        "robosuite/models/assets/textures/ceramic.png": gray,
        "robosuite/models/assets/textures/wood.png": gray}))
    mg = AssetArchive(_zip(tmp_path / "mg.zip", {
        "mimicgen-abc/mimicgen_envs/models/robosuite/assets/textures/ceramic.png": gray}),
        "mimicgen_envs/models/robosuite/assets/", prefix="mimicgen_envs/models/robosuite/assets/")
    lib = AssetArchive(_zip(tmp_path / "lib.zip", {"LIBERO-x/libero/libero/assets/textures/wood.png": red}),
                       "libero/libero/assets/", aliases=("chiliocosm/assets/",), prefix="libero/libero/assets/")
    xml, assets, missing = resolve_mjcf(TEX_XML, [mg, lib, wheel])
    assert missing == []
    files = {t.split('name="')[1][0]: t.split('file="')[1].split('"')[0] for t in xml.split("<texture ")[1:]}
    # Same base name and same bytes: one asset; same base name, other bytes: flattened name.
    assert files["a"] == files["b"] == "textures/ceramic.png" and len(assets) == 3
    assert files["c"] == "libero/libero/assets/textures/wood.png"
    assert files["d"] == "textures__wood.png" and assets["textures__wood.png"] == gray
    m = mujoco.MjModel.from_xml_string(xml, assets)
    assert m.ntex == 4
    # A missing archive drops to placeholders.
    _, assets, missing = resolve_mjcf(TEX_XML, [wheel])
    assert assets == {} and len(missing) == 2


def test_side_hints_from_base_layout():
    grippers = ["gripper0_right", "gripper1_right"]
    parallel = {"robot0_base": [-0.56, -0.25, 0.0], "robot1_base": [-0.56, 0.25, 0.0]}
    hints, why = _side_hints(parallel, grippers)
    assert hints == {"gripper0_right": "right", "gripper1_right": "left"} and why
    rotated = {"robot0_base": [0.25, -0.56, np.pi / 2], "robot1_base": [-0.25, -0.56, np.pi / 2]}
    assert _side_hints(rotated, grippers)[0] == {"gripper0_right": "right", "gripper1_right": "left"}
    opposed = {"robot0_base": [0, -0.81, np.pi / 2], "robot1_base": [0, 0.81, -np.pi / 2]}
    assert _side_hints(opposed, grippers) == ({}, None)
    assert _side_hints({"robot0_base": [0, 0, 0]}, ["gripper0"]) == ({}, None)


def test_mimicgen_lineage_points_at_catalogued_source_seeds():
    cat = load_catalog()
    assert _seed_task("square_d0_iiwa") == "square" and _seed_task("mug_cleanup_o1") == "mug_cleanup"
    assert _seed_task("three_piece_assembly_d1") == "three_piece_assembly"
    groups = set()
    for e in cat.values():
        if e.family != "mimicgen" or e.kind != "file_images_embedded":
            continue
        lin = _default_lineage(e, e.dataset)
        group = e.dataset.split("/")[1]
        groups.add(group)
        if group == "source":
            assert lin == {"generated": False}
        else:
            assert lin["generated"] and lin["variant_group"] == group
            assert f"mimicgen/{lin['seed'].split('/', 1)[1]}.hdf5" in cat, lin["seed"]
    assert groups == {"source", "core", "object", "robot", "large_interpolation"}


def test_stripped_copy_reads_identically_and_keeps_catalog_identity(tmp_path):
    raw = FIXTURE.read_bytes()
    entry = CatalogEntry(id="robomimic/can_sub.hdf5", family="robomimic", path="can_sub.hdf5",
                         url="https://example.invalid/can_sub.hdf5", revision="r", size=len(raw),
                         sha256=hashlib.sha256(raw).hexdigest(), license="MIT", kind="low_dim",
                         dataset="robomimic/can/sub")
    out = tmp_path / "can_sub.state.hdf5"
    strip_images(FIXTURE, out, original={"sha256": entry.sha256, "catalog_id": entry.id})
    a = next(iter_episodes("robomimic", FIXTURE, catalog={entry.id: entry}))
    b = next(iter_episodes("robomimic", out, catalog={entry.id: entry}))
    assert a.dataset == b.dataset == "robomimic/can/sub" and a.license == b.license == "MIT"
    assert b.provenance["sha256"] == entry.sha256 != b.provenance["file_sha256"]
    assert b.provenance["stripped"] and not a.provenance["stripped"]
    for k, e in a.effectors.items():
        np.testing.assert_array_equal(e.pose, b.effectors[k].pose)
        np.testing.assert_array_equal(e.opening, b.effectors[k].opening)
    for k, o in a.objects.items():
        np.testing.assert_array_equal(o.pose, b.objects[k].pose)


@pytest.mark.parametrize("name", ["stack", "coffee", "mug_cleanup", "kitchen", "pick_place"])
def test_real_mimicgen_source_full_scene(name):
    """Fetched (stripped) MimicGen sources compile with MimicGen/task-zoo assets."""
    repo = Path(__file__).resolve().parents[1]
    path = repo / f"data/raw/mimicgen/source/{name}.state.hdf5"
    if not path.exists() or not (repo / "data/raw/robosuite/robosuite-1.4.1-py3-none-any.whl").exists():
        pytest.skip("stripped MimicGen source and robosuite 1.4.1 wheel not fetched")
    ep = next(iter_episodes("mimicgen", path, demos=["demo_0"]))
    if ep.provenance["missing_assets"]:
        pytest.skip("MimicGen / task-zoo asset archives not fetched")
    assert ep.provenance["state_route"] == "mjcf_states" and ep.scene is not None
    assert ep.provenance["stripped"] and ep.lineage == {"generated": False} and ep.license == "CC-BY-4.0"
