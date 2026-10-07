"""BiGym adapter: offline tests on tiny replay-record fixtures (rows subsampled from real
replays of the pinned release; no mesh assets, so the kinematic-only route is used)."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from reachy_retarget.acquire import load_catalog
from reachy_retarget.sources import bigym, bigym_replay, families, iter_episodes

FIX = Path(__file__).parent / "fixtures" / "bigym"
DRAWER = FIX / "DrawerTopOpen" / "04fad25d18b2455ba349df0302fd51e8.npz"
PLATE_OK = FIX / "MovePlate" / "c795f62efb4f4d20ab899af96d766a2f.npz"
PLATE_FAIL = FIX / "MovePlate" / "09431449a6cf4931b5b85fa5b1c7426a.npz"


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


@pytest.fixture(scope="module")
def drawer(catalog):
    (ep,) = iter_episodes("bigym", DRAWER, catalog=catalog)
    return ep


@pytest.fixture(scope="module")
def plate(catalog):
    (ep,) = iter_episodes("bigym", PLATE_OK, catalog=catalog)
    return ep


def test_registered_and_catalogued(catalog):
    assert "bigym" in families()
    z = catalog[bigym.DEMO_ZIP_ID]
    assert z.sha256 == "a7fd1e7f1bd60962ec73dc23a08e6b9b0ffa2711dbbfb8c4de914b819e1186d2"
    assert z.size == 123362114 and z.revision == "v0.9.0"
    code = [e for e in catalog.values() if e.id.startswith(bigym.CODE_ID_PREFIX)]
    assert len(code) == 1 and code[0].revision == "d98124454dfa3bc697f75536daec5989a47bfa8d"


def test_task_table_covers_catalog():
    import yaml
    from importlib import resources
    doc = yaml.safe_load(resources.files("reachy_retarget.acquire").joinpath("catalog/bigym.yaml").read_text())
    tasks = doc["tasks"]
    assert len(tasks) == 40 and sum(t["recordings"] for t in tasks.values()) == 1933
    grouped = {t for ts in bigym.TASK_GROUPS.values() for t in ts}
    assert grouped == set(tasks)
    assert all(t["group"] == bigym.task_group(name) for name, t in tasks.items())


def test_drawer_episode_contract(drawer):
    ep = drawer
    assert ep.family == "bigym" and ep.task == "DrawerTopOpen"
    assert ep.dataset == "bigym/v0.9.0/DrawerTopOpen" and ep.episode_id == "04fad25d18b2455ba349df0302fd51e8"
    assert ep.instruction == "Open top drawer of the cupboard"
    assert ep.regime == "mobile_manipulation" and ep.license == "Apache-2.0"
    assert set(ep.effectors) == {"left", "right"}
    assert {k: e.side_hint for k, e in ep.effectors.items()} == {"left": "left", "right": "right"}
    T = ep.length
    assert ep.base.shape == (T, 3) and ep.torso_height.shape == (T,)
    assert np.all(np.diff(ep.time) > 0) and ep.time[0] == 0.0
    # mjData.time of 500 Hz control steps
    steps = np.asarray(ep.provenance["frame_steps"])
    np.testing.assert_allclose(ep.time[1:], (steps[1:] + 1) * 0.002, atol=1e-9)
    for e in ep.effectors.values():
        R = e.pose[:, :3, :3]
        np.testing.assert_allclose(R @ R.transpose(0, 2, 1), np.broadcast_to(np.eye(3), R.shape), atol=1e-9)
        assert 0 <= e.width.min() and e.width.max() < 0.09
    # The drawer is opened by the replayed actions and the env reports success.
    art = ep.articulations["base_cabinet_600"]
    assert art.joint_names[0] == "base_cabinet_600/drawer_big_1"
    assert art.qpos[0].max() < 1e-6 and art.qpos[-1].max() > 0.3
    assert ep.success is True and ep.provenance["success_any"] is True
    assert ep.provenance["first_success_step"] == 1872
    # Base motion of the floating base (pelvis): it moves and stays near the floor plane.
    assert np.linalg.norm(np.diff(ep.base[:, :2], axis=0), axis=1).sum() > 0.2
    assert ep.base_hint is None


def test_drawer_provenance(drawer):
    p = drawer.provenance
    assert p["zip_sha256_matches_catalog"] and p["zip_catalog_id"] == bigym.DEMO_ZIP_ID
    assert p["member_sha256"] == "34089950c6dd944ddb7912b76596566d61d4cac4205f05801f485aab2c2ec75d"
    assert p["recorded_package_versions"] == {"mujoco": "3.1.5", "bigym": "4.0.0"}
    assert p["replay_runtime_versions"]["bigym"] == "4.1.0"
    assert p["replay_runtime_versions"]["mujoco"] == "3.1.5"
    assert p["code"]["revision"] == "d98124454dfa3bc697f75536daec5989a47bfa8d"
    assert p["model_export_check"]["ok"] is True
    assert p["state_route"] == "bigym_replay_kinematic_only" and p["missing_assets"]
    assert drawer.scene is None  # no assets in the fixture -> no physics scene
    assert p["task_group"] == "articulated"
    assert p["floor_z_in_source_m"] == 0.0 and p["world_offset_m"] == [0.0, 0.0, 0.0]
    # Pelvis starts 1 m above the floor; grippers are above the floor at all times.
    assert drawer.torso_height.max() == pytest.approx(1.0, abs=1e-6)
    assert all(e.pose[:, 2, 3].min() > 0.3 for e in drawer.effectors.values())
    # The delta-mode file of the same recording is lineage, not a second demo.
    assert drawer.lineage["generated"] is False
    assert drawer.lineage["other_versions"] == []  # DrawerTopOpen has no delta-mode copies


def test_gripper_frame_from_geometry(drawer):
    """Grasp center = pad midpoint at the open reference; +y = left pad -> right pad."""
    import mujoco
    meta, arrays = bigym.read_record(DRAWER)
    xml, assets, _ = bigym.resolve_assets(meta["_mjcf"], None)
    m = mujoco.MjModel.from_xml_string(xml, assets)
    d = mujoco.MjData(m)
    d.qpos[:] = arrays["qpos"][0]
    mujoco.mj_kinematics(m, d)
    for side, e in drawer.effectors.items():
        pre = f"h1/robotiq_2f85_{side}/"
        pads = [np.mean([d.geom_xpos[m.geom(pre + f"{s}_pad{i}").id] for i in (1, 2)], axis=0)
                for s in ("left", "right")]
        assert e.opening[0] > 0.99  # open at reset
        np.testing.assert_allclose(e.pose[0][:3, 3], (pads[0] + pads[1]) / 2, atol=1e-6)
        y = (pads[1] - pads[0]) / np.linalg.norm(pads[1] - pads[0])
        assert e.pose[0][:3, 1] @ y > 0.999
        base = d.xpos[m.body(pre + "base").id]
        z = e.pose[0][:3, 3] - base
        assert e.pose[0][:3, 2] @ (z / np.linalg.norm(z)) > 0.999
        assert e.width[0] == pytest.approx(0.0854, abs=5e-4)  # 2F-85: 85 mm stroke
    eff = drawer.provenance["effectors"]["left"]
    assert eff["palm_body"] == "h1/robotiq_2f85_left/base"
    np.testing.assert_allclose(eff["grasp_center_in_palm_frame_m"], [0, 0, 0.13067], atol=1e-5)


def test_pick_place_plate_moves(plate):
    assert plate.task == "MovePlate" and plate.success is True
    (other,) = plate.lineage["other_versions"]  # the delta-mode copy of the same recording
    assert "_delta/" in other and other.endswith("c795f62efb4f4d20ab899af96d766a2f.safetensors")
    assert plate.provenance["task_group"] == "pick_place"
    o = plate.objects["plate"]
    assert o.role == "manipulated" and o.valid.all()
    assert np.linalg.norm(o.pose[-1, :3] - o.pose[0, :3]) > 0.3
    np.testing.assert_allclose(np.linalg.norm(o.pose[:, 3:], axis=1), 1, atol=1e-9)
    assert {"table", "dish_drainer", "dish_drainer_1"} <= set(plate.objects)
    assert plate.objects["table"].role == "support"
    assert plate.objects["dish_drainer"].role == "receptacle"
    assert "floor" not in plate.objects
    # One gripper closes on the plate (pads do not touch: the plate is between them).
    closed = [k for k, e in plate.effectors.items() if e.opening.min() < 0.8]
    assert closed
    assert all(plate.effectors[k].width.min() > 0.002 for k in closed)


def test_failed_source_replay_is_reported(catalog):
    (ep,) = iter_episodes("bigym", PLATE_FAIL, catalog=catalog)
    assert ep.success is False
    assert ep.provenance["success_any"] is False and ep.provenance["first_success_step"] is None
    # The recording itself has termination flags, which do not imply success.
    assert ep.provenance["recorded_first_termination_step"] == 2002


def test_partial_and_empty_records(tmp_path, catalog):
    meta, arrays = bigym.read_record(PLATE_OK)
    mjcf = meta.pop("_mjcf")

    def write(name, meta, arrays):
        a = dict(arrays)
        a["mjcf"] = np.frombuffer(mjcf.encode(), dtype=np.uint8)
        a["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        np.savez_compressed(tmp_path / name, **a)

    partial = {**meta, "error": {"type": "ValueError", "message": "action out of bounds"},
               "replayed_actions": 100}
    write("partial.npz", partial, {k: v[:5] for k, v in arrays.items()})
    write("empty.npz", {**partial, "replayed_actions": 0}, {})
    eps = list(iter_episodes("bigym", tmp_path, catalog=catalog))
    assert len(eps) == 1 and eps[0].success is None
    assert eps[0].provenance["replay_error"]["type"] == "ValueError"
    assert list(bigym.read_bigym(tmp_path, catalog=catalog, include_failed=False)) == []


def test_with_scene_false_skips_assets(catalog):
    (ep,) = bigym.read_bigym(PLATE_OK, catalog=catalog, with_scene=False)
    assert ep.scene is None and ep.provenance["assets_loaded"] is False


def test_record_version_checked(tmp_path):
    p = tmp_path / "x.npz"
    np.savez(p, meta=np.frombuffer(json.dumps({"record_version": "other"}).encode(), dtype=np.uint8))
    with pytest.raises(ValueError, match="record version"):
        bigym.read_record(p)


def test_resolve_assets_placeholders_and_found(tmp_path):
    xml = ('<mujoco><asset><mesh name="m" file="m-1.stl"/><texture name="t" type="2d" file="t-1.png"/>'
           '<material name="mat" texture="t"/></asset><worldbody><body><geom type="mesh" mesh="m"/>'
           '</body></worldbody></mujoco>')
    out, assets, missing = bigym.resolve_assets(xml, tmp_path)
    assert sorted(missing) == ["m-1.stl", "t-1.png"] and assets == {}
    assert 'type="sphere"' in out and "texture=" not in out
    (tmp_path / "m-1.stl").write_bytes(b"x")
    (tmp_path / "t-1.png").write_bytes(b"y")
    out, assets, missing = bigym.resolve_assets(xml, tmp_path)
    assert missing == [] and set(assets) == {"m-1.stl", "t-1.png"} and out == xml


def test_replay_member_selection(tmp_path):
    """One replay per recording UUID: absolute preferred, delta only when it is alone."""
    zp = tmp_path / "demos.zip"
    names = ["A/JointPositionActionMode_floating_x_absolute/lightweight/u1.safetensors",
             "A/JointPositionActionMode_floating_x_delta/lightweight/u1.safetensors",
             "A/JointPositionActionMode_floating_x_delta/lightweight/u2.safetensors",
             "B/JointPositionActionMode_floating_x_absolute/lightweight/u3.safetensors",
             "B/JointPositionActionMode_floating_x_absolute/lightweight/u4.safetensors"]
    with zipfile.ZipFile(zp, "w") as z:
        for n in names:
            z.writestr(n, b"")
    assert bigym_replay.list_members(zp) == [names[0], names[2], names[3], names[4]]
    assert bigym_replay.list_members(zp, tasks={"B"}, limit=1) == [names[3]]
    assert bigym_replay.member_versions(zp)[("A", "u1")] == names[:2]
    rec = bigym.list_recordings(zp)
    assert {k: (len(v["uuids"]), v["files"]) for k, v in rec.items()} == {"A": (2, 3), "B": (2, 2)}
