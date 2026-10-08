"""Offline tests of the robosuite adapter on a tiny robomimic fixture.

Fixture: robomimic v1.5 can/ph ``demo_0`` (MIT), 21 of 118 state rows plus the
recorded MJCF, ``obs/robot0_eef_pos`` and ``obs/robot0_gripper_qpos``; its source
URL, revision and sha256 are stored in the file's ``fixture_note`` attribute.
"""
import re
import struct
import zipfile
import zlib
from pathlib import Path

import h5py
import mujoco
import numpy as np
import pytest

from reachy_retarget.sources import families, iter_episodes, register
from reachy_retarget.sources.robosuite import ASSET_MARKER, _asset_rel, resolve_mjcf

FIXTURE = Path(__file__).parent / "fixtures" / "robomimic_can_ph_demo0_sub.hdf5"
REPO = Path(__file__).resolve().parents[1]


def episode(**kw):
    return next(iter_episodes("robomimic", FIXTURE, **kw))


def recorded(key):
    with h5py.File(FIXTURE) as f:
        return f["data/demo_0"][key][:]


def test_registry():
    assert {"robomimic", "mimicgen"} <= set(families())
    with pytest.raises(KeyError):
        next(iter_episodes("nope", FIXTURE))
    with pytest.raises(ValueError):
        register("robomimic")(lambda p, **k: iter(()))


def test_kinematic_fallback_matches_recorded_observations():
    ep = episode()
    assert ep.length == 21 and ep.time[0] == 0 and ep.task == "PickPlaceCan"
    assert ep.provenance["state_route"] == "mjcf_kinematic_only" and ep.scene is None
    assert ep.provenance["missing_assets"] and ep.license == "unknown"
    np.testing.assert_allclose(ep.base_hint, [-0.5, -0.1, 0.0], atol=1e-9)
    (eff,) = ep.effectors.values()
    # robosuite eef_pos is the grip site; our grasp center sits 3.6 mm toward the palm.
    delta = recorded("obs/robot0_eef_pos") - eff.pose[:, :3, 3]
    np.testing.assert_allclose(np.einsum("ti,ti->t", delta, eff.pose[:, :3, 2]), 0.0036, atol=2e-4)
    np.testing.assert_allclose(np.cross(delta, eff.pose[:, :3, 2]), 0, atol=1e-6)
    # Width is the pad-face gap: the finger joint separation minus 1 mm (the Panda pads overlap
    # by 1 mm at q = 0).
    q = recorded("obs/robot0_gripper_qpos")
    np.testing.assert_allclose(eff.width, np.maximum(q[:, 0] - q[:, 1] - 0.001, 0), atol=1e-6)
    assert 0.4 < eff.opening.min() < 0.6 and eff.opening.max() > 0.98
    np.testing.assert_allclose(np.linalg.det(eff.pose[:, :3, :3]), 1, atol=1e-9)


def test_effector_axes_from_geometry():
    """+z points from the palm to the fingertips, +y from finger 1 to finger 2."""
    ep = episode()
    (eff,) = ep.effectors.values()
    with h5py.File(FIXTURE) as f:
        xml, states = f["data/demo_0"].attrs["model_file"], f["data/demo_0/states"][:]
    model_xml, assets, _ = resolve_mjcf(xml, None)
    m = mujoco.MjModel.from_xml_string(model_xml, assets)
    d = mujoco.MjData(m)
    for t, row in enumerate(states):
        d.qpos[:], d.qvel[:] = row[1:1 + m.nq], row[1 + m.nq:]
        mujoco.mj_forward(m, d)
        T = eff.pose[t]
        local = lambda p: T[:3, :3].T @ (p - T[:3, 3])
        palm = local(d.body("gripper0_right_right_gripper").xpos)
        tip1 = local(d.body("gripper0_right_finger_joint1_tip").xpos)
        tip2 = local(d.body("gripper0_right_finger_joint2_tip").xpos)
        assert palm[2] < -0.08 and abs(palm[0]) < 1e-6 and abs(palm[1]) < 1e-6
        assert tip1[2] > palm[2] + 0.08 and tip2[2] > palm[2] + 0.08
        assert tip1[1] < 0 < tip2[1] and abs(tip1[0]) < 1e-6 and abs(tip2[0]) < 1e-6


def test_objects_and_fixtures():
    ep = episode()
    assert set(ep.objects) == {"Can", "bin1", "bin2"}
    assert ep.provenance["inactive_free_bodies"] == ["Bread", "Cereal", "Milk"]
    can = ep.objects["Can"]
    assert can.role == "manipulated" and can.valid.all()
    np.testing.assert_allclose(np.linalg.norm(can.pose[:, 3:], axis=1), 1, atol=1e-9)
    assert np.ptp(can.pose[:, 1]) > 0.3  # carried from bin1 to bin2
    assert ep.objects["bin1"].role == "receptacle" and np.ptp(ep.objects["bin1"].pose, axis=0).max() == 0


def _synthetic_archive(xml: str, path: Path) -> Path:
    """Placeholder files for every referenced asset (tetrahedra, 1x1 PNG)."""
    v = np.array([[0, 0, 0], [.01, 0, 0], [0, .01, 0], [0, 0, .01]], np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    stl = b"\0" * 80 + struct.pack("<I", 4) + b"".join(
        struct.pack("<12fH", 0, 0, 0, *v[a], *v[b], *v[c], 0) for a, b, c in faces)
    obj = "".join(f"v {x} {y} {z}\n" for x, y, z in v) + "".join(f"f {a+1} {b+1} {c+1}\n" for a, b, c in faces)
    msh = struct.pack("<4i", 4, 0, 0, 4) + v.tobytes() + faces.astype(np.int32).tobytes()
    chunk = lambda t, b: struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(b"\0\x80\x80\x80")) + chunk(b"IEND", b""))
    data = {".stl": stl, ".obj": obj.encode(), ".msh": msh, ".png": png}
    rels = {_asset_rel(f) for f in re.findall(r'file="([^"]+)"', xml)}
    with zipfile.ZipFile(path, "w") as z:
        for rel in rels:
            z.writestr(ASSET_MARKER + rel, data[Path(rel).suffix])
    return path


def test_full_scene_route_with_asset_archive(tmp_path):
    with h5py.File(FIXTURE) as f:
        xml = f["data/demo_0"].attrs["model_file"]
    ep = episode(asset_archive=_synthetic_archive(xml, tmp_path / "assets.zip"))
    assert ep.provenance["state_route"] == "mjcf_states" and not ep.provenance["missing_assets"]
    scene = ep.scene
    assert scene.robot_prefixes == ["fixed_mount0_", "gripper0_", "robot0_"]
    np.testing.assert_allclose(scene.initial_qpos["Can_joint0"], ep.objects["Can"].pose[0], atol=1e-12)
    m = mujoco.MjModel.from_xml_string(scene.mjcf, scene.assets)
    assert m.nq == 37 and m.nmesh > 0
    # Parked objects are declared inactive; the source reference covers the Can only.
    assert scene.inactive_bodies == ["Bread_main", "Cereal_main", "Milk_main"]
    assert set(scene.reference["per_object"]) == {"Can"} and scene.reference["frames"] == ep.length
    assert scene.reference["object_environment_depth_m"] >= 0 and ep.provenance["scene_reference"] == scene.reference
    # Kinematics do not depend on mesh shapes: identical to the fallback route.
    np.testing.assert_allclose(next(iter(ep.effectors.values())).pose,
                               next(iter(episode().effectors.values())).pose, atol=1e-12)


@pytest.mark.skipif(not (REPO / "data/raw/robomimic/v1.5/can/ph/low_dim_v15.hdf5").exists()
                    or not (REPO / "data/raw/robosuite/robosuite-1.5.1-py3-none-any.whl").exists(),
                    reason="fetched robomimic can/ph and robosuite 1.5.1 wheel not present")
def test_real_can_ph_file():
    ep = next(iter_episodes("robomimic", REPO / "data/raw/robomimic/v1.5/can/ph/low_dim_v15.hdf5",
                            demos=["demo_0"]))
    assert ep.dataset == "robomimic/can/ph" and ep.license == "MIT" and ep.success
    assert ep.provenance["state_route"] == "mjcf_states" and ep.scene is not None
    assert ep.provenance["revision"] == "74fa018461f479cd9fd15b924a16103012096203"
    assert ep.length == 118 and ep.lineage == {"generated": False}
    # source reference: the Can sinks 2.1 mm into bin1 (soft contacts), Milk/Bread/Cereal parked
    ref = ep.scene.reference
    assert 0.0015 < ref["object_environment_depth_m"] < 0.003 and ref["worst_contact"]["bodies"] == ["bin1", "Can_main"]
    assert ep.scene.inactive_bodies == ["Bread_main", "Cereal_main", "Milk_main"]
    can = ep.objects["Can"].geometry
    assert can["kind"] == "cylinder" and can["axis"] == "z" and abs(can["radius"] - 0.0261) < 5e-4
    (eff,) = ep.effectors.values()
    with h5py.File(REPO / "data/raw/robomimic/v1.5/can/ph/low_dim_v15.hdf5") as f:
        np.testing.assert_array_equal(eff.command, (f["data/demo_0/actions"][:, -1] + 1) / 2)


def test_gripper_command_from_actions(tmp_path):
    import shutil
    path = tmp_path / "can.hdf5"
    shutil.copy(FIXTURE, path)
    with h5py.File(path, "a") as f:
        g = f["data/demo_0"]
        a = np.zeros((len(g["states"]), 7))
        a[:, 6] = np.where(np.arange(len(a)) >= 8, 1.0, -1.0)
        g["actions"] = a
    (eff,) = next(iter_episodes("robomimic", path)).effectors.values()
    np.testing.assert_array_equal(eff.command, np.arange(len(a)) >= 8)
    with h5py.File(path, "a") as f:
        f["data/demo_0/actions"][:, 6] = 0.3  # not a binary -1/+1 command: not used
    ep = next(iter_episodes("robomimic", path))
    assert next(iter(ep.effectors.values())).command is None
    assert "not all -1 / +1" in ep.provenance["gripper_command"]["reason"]


def test_fixture_with_articulated_children_keeps_its_static_geometry():
    # MimicGen Kitchen/HammerCleanup hang the stove buttons and a cabinet drawer below the table body:
    # the table is still a fixture, described by its static geoms only.
    from reachy_retarget.sources.robosuite import _Model
    xml = """<mujoco><worldbody>
    <body name="table" pos="0 0 0.8"><geom name="table_collision" type="box" size="0.4 0.4 0.025"/>
      <body name="drawer" pos="0 0 0.1"><joint name="drawer_slide" type="slide" axis="1 0 0"/>
        <geom type="box" size="0.1 0.1 0.05"/></body>
    </body>
    <body name="cube_main" pos="0 0 0.9"><freejoint/><geom type="box" size="0.02 0.02 0.02"/></body>
    </worldbody></mujoco>"""
    model = _Model(xml, None)
    model.mj.mj_forward(model.m, model.d)
    assert [model.body_names[b] for b in model.fixtures] == ["table"]
    np.testing.assert_allclose(model.body_aabb(model.fixtures[0], static=True)["half_extents"], [0.4, 0.4, 0.025])
    np.testing.assert_allclose(model.body_aabb(model.fixtures[0])["half_extents"], [0.4, 0.4, 0.0875])


def test_union_of_boxes_records_each_part_oriented_box():
    # LIBERO bowl walls: thin box geoms turned in the body; their body-frame AABB is much thicker
    from reachy_retarget.sources.robosuite import _Model
    xml = """<mujoco><worldbody>
    <body name="bowl_main" pos="0 0 0.9"><freejoint/>
      <geom type="box" size="0.0007 0.01 0.02" pos="0.03 0 0" euler="0 0 45"/>
      <geom type="box" size="0.02 0.02 0.002" pos="0 0 -0.02"/></body>
    </worldbody></mujoco>"""
    model = _Model(xml, None)
    model.mj.mj_forward(model.m, model.d)
    g = model.body_aabb(model.m.body("bowl_main").id, parts=True)
    wall = g["boxes"][0]
    assert wall["half_extents"][0] > 0.007                       # the AABB part, as before
    np.testing.assert_allclose(wall["obb"]["half_extents"], [0.0007, 0.01, 0.02], atol=1e-6)
    np.testing.assert_allclose(wall["obb"]["center"], [0.03, 0, 0], atol=1e-6)
    np.testing.assert_allclose(np.abs(wall["obb"]["quat"]), [np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)], atol=1e-6)
