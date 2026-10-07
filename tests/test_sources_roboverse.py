"""Offline tests of the RoboVerse adapter on tiny subsets of the pinned files.

Fixtures (``tests/fixtures/roboverse``, paths mirror ``RoboVerseOrg/roboverse_data`` at
revision ``fab63cc``):

* robot files ``franka_panda.urdf``, ``panda.xml`` (menagerie MJCF, pad geometry),
  ``panda_longer_finger.urdf`` (CALVIN) and ``ann_dict.npy``: byte copies of the catalogued files.
* ``calvin_traj_ann/env_D_val_out/task_288_v2.pkl``: byte copy ("pick up the blue block", one
  window, torch-tensor pickle), so it matches its catalog entry.
* ``calvin_traj_ann/env_A_out/task_5_v2.pkl``: first window of "lift the red block" of the
  catalogued env_A file, re-pickled as plain lists (tests the scene-A colour remap).
* ``rlbench/{pick_and_lift,close_drawer}/v2/franka_v2.pkl.gz``: episode 0 of the catalogued
  files, re-pickled (``init_state``, ``actions``, ``states`` unchanged).
"""
import gzip
import pickle
from pathlib import Path

import numpy as np
import pytest

import reachy_retarget.sources.roboverse as rv
from reachy_retarget.acquire import load_catalog
from reachy_retarget.sources import families, iter_episodes
from reachy_retarget.sources.roboverse_pickle import load, load_npy_object

FIX = Path(__file__).parent / "fixtures" / "roboverse"
FILES = {rv.FRANKA_URDF: FIX / rv.FRANKA_URDF, rv.FRANKA_MJCF: FIX / rv.FRANKA_MJCF,
         rv.CALVIN_URDF: FIX / rv.CALVIN_URDF, rv.ANN_DICT: FIX / rv.ANN_DICT}
RL = FIX / "trajs/rlbench"
CV = FIX / "trajs/calvin/calvin_traj_ann"


def episodes(path, **kw):
    return list(iter_episodes("roboverse", path, robot_files=FILES, **kw))


def test_registry_and_catalog():
    assert "roboverse" in families()
    cat = {k: e for k, e in load_catalog().items() if e.family == "roboverse"}
    assert all(e.sha256 and len(e.sha256) == 64 and e.size and e.url.startswith(
        "https://huggingface.co/datasets/RoboVerseOrg/roboverse_data/resolve/" + rv.HF_REVISION) for e in cat.values())
    rl = [e for e in cat.values() if e.path.startswith("trajs/rlbench/") and e.kind == "low_dim"]
    assert len(rl) == 80 and all(e.path.endswith("/v2/franka_v2.pkl.gz") for e in rl)
    cv = [e for e in cat.values() if e.path.startswith("trajs/calvin/") and e.kind == "low_dim"]
    assert len(cv) == 389 + 299 and {e.dataset for e in cv} == {"roboverse/calvin/env_A", "roboverse/calvin/env_D_val"}
    for rel in FILES:   # fixtures are byte copies of catalogued files
        assert any(e.path == rel for e in cat.values())
    # overlapping benchmarks and cross-embodiment variants are not catalogued
    assert not any(s in e.path for e in cat.values() for s in ("/libero", "/maniskill/", "sawyer_v2", "ur5e_2f85"))


def test_restricted_unpickler_refuses_globals(tmp_path):
    bad = tmp_path / "x.pkl.gz"
    bad.write_bytes(gzip.compress(pickle.dumps({"franka": [np.float64(1.0)]})))
    with pytest.raises(pickle.UnpicklingError):
        load(bad)
    evil = tmp_path / "y.pkl"
    evil.write_bytes(b"cos\nsystem\n(S'true'\ntR.")
    with pytest.raises(pickle.UnpicklingError):
        load(evil)
    d = load(CV / "env_D_val_out/task_288_v2.pkl")       # torch tensors decoded without torch
    st = d["franka"][0]["reset_state"][0]["robots"]["franka"]
    assert isinstance(st["pos"], np.ndarray) and st["pos"].shape == (3,)
    ann = load_npy_object(FILES[rv.ANN_DICT])
    assert len(ann) == 389 and ann["lift the red block"] == 5 and ann["pick up the blue block"] == 288


def test_franka_grasp_model():
    gr = rv.FrankaGrasp(FILES[rv.FRANKA_URDF], pad_center=rv.mjcf_pad_center(FILES[rv.FRANKA_MJCF]))
    np.testing.assert_allclose(gr.center_in_hand, [0, 0, 0.0584 + 0.0445], atol=1e-9)
    q = np.array([[0, -0.3, 0, -2.2, 0, 1.9, 0.785, 0.03, 0.01]])
    out = gr.forward(q, np.eye(4)[None])
    G, H = out["grasp"][0], out["hand"][0]
    np.testing.assert_allclose(G[:3, :3], H[:3, :3] @ np.diag([-1, -1, 1]), atol=1e-12)
    assert out["width"][0] == pytest.approx(0.04)
    # asymmetric fingers move the pad midpoint along the closing axis (+y = finger1 -> finger2)
    shift = (np.linalg.inv(H) @ np.r_[G[:3, 3], 1])[:3]
    np.testing.assert_allclose(shift, [0, 0.01, 0.1029], atol=1e-9)
    xc = rv.fk_crosscheck_mujoco(gr, q, np.eye(4)[None], out["grasp"], mjcf_path=FILES[rv.FRANKA_MJCF])
    if xc["available"]:
        assert xc["max_grasp_center_error_m"] < 1e-6
    cal = rv.FrankaGrasp(FILES[rv.CALVIN_URDF], tcp_link="tcp")
    np.testing.assert_allclose(cal.center_in_hand, [0, 0, 0.14], atol=1e-9)


def test_rlbench_pick_and_lift():
    (ep,) = episodes(RL / "pick_and_lift/v2/franka_v2.pkl.gz", crosscheck=True)
    assert ep.family == "roboverse" and ep.task == "rlbench/pick_and_lift" and ep.success is True
    assert ep.length == 158 and np.allclose(np.diff(ep.time), rv.RLBENCH_DT)
    assert ep.lineage["upstream"] == "RLBench" and ep.lineage["human"] is False
    obj = ep.objects["pick_and_lift_target"]
    assert obj.role == "manipulated" and obj.geometry["kind"] == "box"
    np.testing.assert_allclose(obj.geometry["half_extents"], [0.025] * 3)
    # floor at z = 0: resting cube center on the 0.75 m table top
    assert obj.pose[0, 2] == pytest.approx(rv.RLBENCH_TABLE_HEIGHT + 0.025, abs=1e-3)
    assert ep.objects["table"].role == "support" and "success_visual" in ep.provenance["goal_markers"]
    top = ep.objects["table"].pose[0, 2] + ep.objects["table"].geometry["center"][2] + \
        ep.objects["table"].geometry["half_extents"][2]
    assert top == pytest.approx(rv.RLBENCH_TABLE_HEIGHT)
    assert ep.scene is not None and ep.scene.initial_qpos   # all primitives -> MuJoCo scene
    eff = ep.effectors["franka"]
    assert eff.width.max() == pytest.approx(0.08, abs=1e-3) and eff.width.min() == pytest.approx(0.0489, abs=1e-3)
    ck = ep.provenance["state_checks"]
    held, raw = ck["held_objects"]["pick_and_lift_target"], ck["held_objects_uncalibrated"]["pick_and_lift_target"]
    # calibration makes the parented cube rigid in the grasp frame and centred on the closing axis
    assert max(held["spread_m"]) < 0.003 and max(raw["spread_m"]) > 0.02
    assert abs(held["median_offset_m"][1]) < 0.004
    assert ck["fk_crosscheck"].get("max_grasp_center_error_m", 0) < 1e-6
    assert ep.base_hint[2] == 0.0 and ep.base_hint[0] == pytest.approx(-0.2677, abs=1e-3)
    (raw_ep,) = episodes(RL / "pick_and_lift/v2/franka_v2.pkl.gz", rlbench_calibration=False)
    assert raw_ep.provenance["calibration"] == {"applied": False}


def test_rlbench_articulation_and_mesh_scene():
    (ep,) = episodes(RL / "close_drawer/v2/franka_v2.pkl.gz")
    art = ep.articulations["drawer_frame"]
    assert art.joint_names == ["drawer_frame/drawer_joint_bottom", "drawer_frame/drawer_joint_middle",
                               "drawer_frame/drawer_joint_top"]
    assert art.qpos[0, 0] == pytest.approx(0.1, abs=1e-3) and abs(art.qpos[-1, 0]) < 0.02   # bottom drawer closed
    g = ep.objects["drawer_frame"].geometry
    assert g["kind"] == "mesh" and g["asset"].endswith("drawer_frame.usd")
    assert ep.scene is None and ep.provenance["scene"].startswith("none")


def test_calvin_validation_window():
    (ep,) = episodes(CV / "env_D_val_out/task_288_v2.pkl", crosscheck=True)
    assert ep.provenance["catalog_id"] == "roboverse/trajs/calvin/calvin_traj_ann/env_D_val_out/task_288_v2.pkl"
    assert ep.instruction == "pick up the blue block" and ep.dataset == "roboverse/calvin/env_D_val"
    assert ep.length == 64 and np.allclose(np.diff(ep.time), 1 / 30)
    assert ep.lineage["human"] is True and len(ep.lineage["frames"]) == 2
    blue = ep.objects["block_blue"]
    assert blue.role == "manipulated"
    np.testing.assert_allclose(blue.geometry["half_extents"], [0.02, 0.02, 0.02])   # scene D: small x 0.8
    assert ep.articulations["table"].joint_names == [f"table/{j}" for j in
                                                     ["base__button", "base__drawer", "base__slide", "base__switch"]]
    held = ep.provenance["state_checks"]["held_objects"]["block_blue"]
    assert abs(held["median_offset_m"][1]) < 0.003 and max(held["spread_m"]) < 0.005
    assert ep.scene is None and np.allclose(ep.base_hint, [-0.34, -0.46, 0.0])
    assert ep.objects["table"].geometry["kind"] == "aabb" and ep.objects["table"].role == "fixture"
    assert ep.provenance["state_checks"]["fk_crosscheck"].get("max_grasp_center_error_m", 0) < 1e-6


def test_calvin_scene_a_colour_remap():
    (ep,) = episodes(CV / "env_A_out/task_5_v2.pkl")
    assert ep.instruction == "lift the red block" and ep.provenance["calvin_scene"] == "A"
    assert ep.provenance["block_colour_remap"]["pink_cube"] == "block_red"
    moved = {k: np.ptp(o.pose[:, 2]) for k, o in ep.objects.items() if k.startswith("block_")}
    assert max(moved, key=moved.get) == "block_red"      # the language and the motion agree
    np.testing.assert_allclose(ep.objects["block_red"].geometry["half_extents"], [0.028, 0.02, 0.02])


def test_unsupported_file(tmp_path):
    p = tmp_path / "trajs/maniskill/pick_cube/v2/franka_v2.pkl.gz"
    p.parent.mkdir(parents=True)
    p.write_bytes(gzip.compress(pickle.dumps({"franka": []})))
    with pytest.raises(ValueError):
        episodes(p)
