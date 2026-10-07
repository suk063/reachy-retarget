"""Primitive MuJoCo scenes for non-MuJoCo sources (sources.primitive_scene, ManiSkill3 scenes).

Offline: ManiSkill fixtures (tests/fixtures/maniskill) and synthetic resting poses of every
catalogued task's actors.
"""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from reachy_retarget.schema.source import ObjectTrack
from reachy_retarget.sources import iter_episodes
from reachy_retarget.sources import primitive_scene as ps
from reachy_retarget.sources.maniskill import (PHYSICAL, STATIC_ACTORS, TASKS, TABLE_HEIGHT, WORLD_Z_OFFSET,
                                               peg_insertion_geometry, scene_for)
from reachy_retarget.validate.scene import build_scene

FIX = Path(__file__).parent / "fixtures" / "maniskill"
URDFS = {"panda": FIX / "panda_v2.urdf", "panda_wristcam": FIX / "panda_v3.urdf"}
TABLE_POSE = [-0.12, 0.0, -TABLE_HEIGHT + WORLD_Z_OFFSET, np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]


def track(pose, geometry, role="manipulated", T=3):
    return ObjectTrack(pose=np.tile(np.asarray(pose, float), (T, 1)), valid=np.ones(T, bool), role=role,
                       geometry=dict(geometry))


def episodes(name):
    return list(iter_episodes("maniskill", FIX / name / "trajectory.h5", urdfs=URDFS))


def resting_objects(env_id):
    """Each task's actors placed at rest on the table (stacked cubes on top of each other)."""
    objs = {"table-workspace": track(TABLE_POSE, STATIC_ACTORS["table-workspace"][1], "support")}
    x = -0.3
    for name, (role, geom) in TASKS[env_id]["objects"].items():
        if isinstance(geom, str):
            peg, box = peg_insertion_geometry(7)
            geom = peg if name == "peg" else box
        if name == "receptacle":       # kinematic, held in the air in PlugCharger
            pose = [0.3, 0.2, TABLE_HEIGHT + 0.1, 1, 0, 0, 0]
        elif name == "box_with_hole":
            pose = [0.0, 0.3, TABLE_HEIGHT + geom["aabb"]["half_extents"][2], 1, 0, 0, 0]
        else:
            lo = np.array(geom["aabb"]["center"]) - geom["aabb"]["half_extents"] if geom["kind"] == "boxes" else None
            z = (-lo[2] if lo is not None else geom["radius"] if geom["kind"] == "sphere"
                 else geom["half_extents"][2])
            pose = [x, 0.0, TABLE_HEIGHT + z, 1, 0, 0, 0]
            x += 0.35
        objs[name] = track(pose, geom, role)
    if env_id in ("StackCube-v1", "StackPyramid-v1"):   # cubeA on cubeB
        objs["cubeA"].pose[:, :3] = objs["cubeB"].pose[0, :3] + [0, 0, 0.04]
    return objs


@pytest.mark.parametrize("env_id", sorted(TASKS))
def test_task_scene_rests_stably(env_id):
    objs = resting_objects(env_id)
    ref, note = scene_for(env_id, objs)
    assert ref is not None, note
    assert ref.robot_prefixes == [ps.NO_ROBOT_PREFIX] and not ref.assets
    m = mujoco.MjModel.from_xml_string(ref.mjcf)
    dynamic = {k for k, v in PHYSICAL[env_id].items() if v["body_type"] == "dynamic"}
    assert {m.body(m.jnt_bodyid[j]).name for j in range(m.njnt)} == dynamic
    assert set(ref.initial_qpos) == {f"{k}_freejoint" for k in dynamic}
    r = ps.rest_check(ref, seconds=2.0)
    assert r["max_penetration_m"] < 1e-3, r["worst_contact"]
    assert max(r["drift_m"].values()) < 1e-3, r["drift_m"]
    assert max(r["rotation_rad"].values()) < 1e-2, r["rotation_rad"]


def test_masses_and_materials_from_maniskill_defaults():
    prov = ps.scene_provenance(scene_for("PullCubeTool-v1", resting_objects("PullCubeTool-v1"))[0])
    cube, tool = prov["objects"]["cube"], prov["objects"]["l_shape_tool"]
    assert cube["mass_kg"] == pytest.approx(0.04 ** 3 * 1000)
    assert tool["density_kg_m3"] == [500.0, 1000.0]
    assert tool["mass_kg"] == pytest.approx(0.2 * 0.05 * 0.05 * 500 + 0.05 * 0.1 * 0.05 * 1000)
    assert cube["dynamic_friction"] == cube["static_friction"] == 0.3 and cube["restitution"] == 0.0
    assert prov["floor"]["z"] == 0.0 and not prov["warnings"]
    assert prov["objects"]["table-workspace"]["body_type"] == "static"
    m = mujoco.MjModel.from_xml_string(scene_for("PickCube-v1", resting_objects("PickCube-v1"))[0].mjcf)
    assert m.body("cube").mass[0] == pytest.approx(0.064)
    g = m.geom("cube/geom0")
    np.testing.assert_allclose(g.friction, [0.3, 0, 0])
    assert g.condim[0] == 3 and m.geom("floor").pos[2] == 0.0
    # Floor at z = 0 (SourceEpisode contract), table top at the table height.
    t = mujoco.MjData(m)
    mujoco.mj_forward(m, t)
    gid = m.geom("table-workspace/geom0").id
    assert t.geom_xpos[gid][2] + m.geom_size[gid][2] == pytest.approx(0.9196429, abs=1e-9)
    assert t.geom_xpos[gid][2] - m.geom_size[gid][2] == pytest.approx(0.0, abs=1e-9)


def test_fixture_episodes_scene_matches_frame0_and_builds():
    for name in ("pickcube_mp", "peginsertion_mp", "tworobot_rl"):
        for ep in episodes(name):
            assert ep.scene is not None, ep.provenance["scene"]
            m = mujoco.MjModel.from_xml_string(ep.scene.mjcf)
            d = mujoco.MjData(m)
            for j, v in ep.scene.initial_qpos.items():
                d.qpos[m.joint(j).qposadr[0]:][:7] = v
            mujoco.mj_forward(m, d)
            for oid, tr in ep.objects.items():
                assert tr.geometry["body"] == oid
                b = m.body(oid).id
                np.testing.assert_allclose(d.xpos[b], tr.pose[0, :3], atol=1e-9)
                np.testing.assert_allclose(abs(d.xquat[b] @ tr.pose[0, 3:]), 1.0, atol=1e-9)
            # qpos0 of the MJCF equals frame 0 as well
            np.testing.assert_allclose(m.qpos0, d.qpos, atol=1e-9)
            scene = build_scene(ep.scene)
            assert set(scene.free_bodies.values()) == {o for o, t in ep.objects.items()
                                                       if PHYSICAL[ep.task].get(o, {}).get("body_type") == "dynamic"}
            assert scene.info["removed"]["bodies"] == [] and scene.initial_qpos.keys() == ep.scene.initial_qpos.keys()
            assert ep.provenance["world_z_offset_m"] == pytest.approx(0.9196429)
            np.testing.assert_allclose(ep.objects["table-workspace"].pose[0, 2], 0.0, atol=1e-6)
            prov = ep.provenance["scene_physical"]
            assert prov["source"]["maniskill_commit"].startswith("baab60ed") and prov["assumptions"]
            r = ps.rest_check(ep.scene, seconds=2.0)
            assert r["max_penetration_m"] < 1e-3 and max(r["drift_m"].values()) < 1e-3


def test_unrepresentable_objects_keep_no_scene():
    objs = resting_objects("PickCube-v1")
    objs["mystery"] = track([0, 0, 0.1, 1, 0, 0, 0], {})
    ref, note = scene_for("PickCube-v1", objs)
    assert ref is None and "mystery" in note
    with pytest.raises(ps.UnrepresentableObject):
        ps.build({"m": track([0, 0, 0.1, 1, 0, 0, 0], {"kind": "mesh"})}, physical={
            "defaults": {"static_friction": .3, "dynamic_friction": .3, "restitution": 0, "density": 1000},
            "objects": {"m": {"body_type": "dynamic"}}})


def test_cylinder_axis_and_warnings():
    objs = {"c": track([0, 0, 0.02, 1, 0, 0, 0], {"kind": "cylinder", "radius": 0.02, "half_length": 0.05})}
    ref = ps.build(objs, physical={
        "defaults": {"static_friction": .5, "dynamic_friction": .3, "restitution": 0.1, "density": 1000},
        "objects": {"c": {"body_type": "dynamic"}}})
    m = mujoco.MjModel.from_xml_string(ref.mjcf)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    axis = d.geom_xmat[m.geom("c/geom0").id].reshape(3, 3)[:, 2]
    np.testing.assert_allclose(abs(axis), [1, 0, 0], atol=1e-9)        # SAPIEN cylinders lie along x
    assert m.body("c").mass[0] == pytest.approx(np.pi * 0.02 ** 2 * 0.1 * 1000)
    w = ps.scene_provenance(ref)["warnings"]
    assert any("static friction" in x for x in w) and any("restitution" in x for x in w)


def test_crop_supports_xy_only_and_recorded():
    table = track(TABLE_POSE, STATIC_ACTORS["table-workspace"][1], "support")
    T = 3
    cube = track([0.0, 0.0, TABLE_HEIGHT + 0.02, 1, 0, 0, 0], {"kind": "box", "half_extents": [0.02] * 3}, T=T)
    cube.pose[2, :3] = [0.1, 0.3, TABLE_HEIGHT + 0.2]          # moves during the episode
    objs, rec = ps.crop_supports({"table-workspace": table, "cube": cube}, 0.10)
    assert objs["cube"] is cube and table.geometry["half_extents"] == STATIC_ACTORS["table-workspace"][1]["half_extents"]
    (r,) = rec
    g = objs["table-workspace"].geometry
    assert r["object"] == "table-workspace" and r["margin_m"] == 0.10 and r["rule"] == ps.CROP_RULE
    assert r["original"]["half_extents"] == STATIC_ACTORS["table-workspace"][1]["half_extents"]
    assert g["half_extents"][2] == r["original"]["half_extents"][2] and g["center"][2] == r["original"]["center"][2]
    # World AABB of the cropped table top = cube footprint over all frames + 0.10 m.
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])           # table yaw +90 deg
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T
    corners = TABLE_POSE[:3] + (np.array(g["center"]) + signs * g["half_extents"]) @ R.T
    np.testing.assert_allclose(corners[:, :2].min(0), [-0.02 - 0.10, -0.02 - 0.10], atol=1e-9)
    np.testing.assert_allclose(corners[:, :2].max(0), [0.1 + 0.02 + 0.10, 0.3 + 0.02 + 0.10], atol=1e-9)
    np.testing.assert_allclose(corners[:, 2].max(), TABLE_HEIGHT, atol=1e-9)   # top surface kept
    # Never beyond the original extent.
    far = track([5.0, 0.0, TABLE_HEIGHT + 0.02, 1, 0, 0, 0], {"kind": "box", "half_extents": [0.02] * 3})
    g2 = ps.crop_supports({"table-workspace": table, "cube": cube, "far": far}, 0.10)[0]["table-workspace"].geometry
    orig = STATIC_ACTORS["table-workspace"][1]
    assert g2["center"][1] - g2["half_extents"][1] == pytest.approx(-orig["half_extents"][1])  # world +x = body -y


def test_maniskill_table_cropped_in_track_and_scene():
    ep = episodes("pickcube_mp")[0]
    (rec,) = ep.provenance["scene_adaptations"]
    g = ep.objects["table-workspace"].geometry
    assert rec["cropped"]["half_extents"] == g["half_extents"] and g["uncropped"]["half_extents"] == rec["original"]["half_extents"]
    assert g["half_extents"][0] < 0.3 and g["half_extents"][1] < 0.3
    m = mujoco.MjModel.from_xml_string(ep.scene.mjcf)
    np.testing.assert_allclose(m.geom("table-workspace/geom0").size, g["half_extents"], atol=1e-8)
    np.testing.assert_allclose(m.geom("table-workspace/geom0").pos, g["center"], atol=1e-8)
