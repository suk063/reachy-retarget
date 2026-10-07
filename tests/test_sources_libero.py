"""Offline tests of the LIBERO adapter.

Fixture ``libero90_kitchen5_close_drawer_demo0_sub.hdf5``: 12 of 65 state rows of
``demo_0`` of LIBERO-90 KITCHEN_SCENE5 "close the top drawer of the cabinet" (HF
``yifengzhu-hf/LIBERO-datasets``, Apache-2.0), its recorded MJCF, ``problem_info`` and
``bddl_file_name`` attributes and the ``obs/ee_pos``/``obs/gripper_states`` rows; source
URL, revision and sha256 are in the file's ``fixture_note`` attribute. Asset archives are
synthetic placeholders.
"""
from pathlib import Path

import h5py
import mujoco
import numpy as np

from _robosuite_fixtures import archive_entry
from reachy_retarget.sources import families, iter_episodes
from reachy_retarget.sources.libero import parse_bddl

FIXTURE = Path(__file__).parent / "fixtures" / "libero90_kitchen5_close_drawer_demo0_sub.hdf5"

BDDL = """(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
  (:domain robosuite)
  (:language close the top drawer of the cabinet)
  (:regions (top_region (:target white_cabinet_1)))
  (:fixtures
    kitchen_table - kitchen_table
    white_cabinet_1 - white_cabinet
  )
  (:objects
    akita_black_bowl_1 - akita_black_bowl
    plate_1 - plate
    ketchup_1 - ketchup
  )
  (:obj_of_interest
    white_cabinet_1
  )
  (:init (Open white_cabinet_1_top_region))
  (:goal
    (And (Close white_cabinet_1_top_region))
  )
)"""


def test_parse_bddl():
    b = parse_bddl(BDDL)
    assert b["language"] == "close the top drawer of the cabinet"
    assert b["objects_of_interest"] == ["white_cabinet_1"]
    assert b["fixtures"] == ["kitchen_table", "white_cabinet_1"]
    assert b["objects"] == ["akita_black_bowl_1", "plate_1", "ketchup_1"]
    assert b["goal"] == "(And (Close white_cabinet_1_top_region))"


def test_kinematic_route_instruction_success_and_lineage():
    assert "libero" in families()
    ep = next(iter_episodes("libero", FIXTURE, root=Path("/nonexistent")))
    assert ep.instruction == "close the top drawer of the cabinet" and ep.success is True
    assert ep.lineage == {"generated": False} and ep.task == "Libero_Kitchen_Tabletop_Manipulation"
    assert ep.provenance["state_route"] == "mjcf_kinematic_only" and ep.scene is None
    assert ep.provenance["env_version"] is None and ep.provenance["env_version_assumed"] == "1.4.1"
    assert ep.length == 12 and ep.time[0] == 0 and np.all(np.diff(ep.time) > 0)
    assert set(ep.objects) >= {"akita_black_bowl_1", "plate_1", "ketchup_1"}
    assert all(ep.objects[k].role == "manipulated" for k in ("akita_black_bowl_1", "plate_1", "ketchup_1"))
    (art,) = ep.articulations.values()
    assert art.joint_names == ["white_cabinet_1_top_level", "white_cabinet_1_middle_level",
                               "white_cabinet_1_bottom_level"]
    (eff,) = ep.effectors.values()
    assert eff.side_hint is None and 0.8 < eff.opening.min() and eff.opening.max() <= 1
    with h5py.File(FIXTURE) as f:
        g = f["data/demo_0"]
        q = g["obs/gripper_states"][:]
        ee = g["obs/ee_pos"][:]
    # LIBERO obs lag the states by one control step (obs[t] ~ states[t + 1], verified
    # on the full demo to 0.43 mm), so only a loose bound holds on subsampled rows.
    site = eff.pose[:, :3, 3] + 0.0036 * eff.pose[:, :3, 2]
    assert np.abs(ee - site).max() < 0.025
    assert np.abs(eff.width - (q[:, 0] - q[:, 1])).max() < 0.01 and eff.width.max() <= 0.08


def test_full_scene_with_libero_and_robosuite_archives(tmp_path):
    with h5py.File(FIXTURE) as f:
        xml = f["data/demo_0"].attrs["model_file"]
        bddl_name = f["data"].attrs["bddl_file_name"]
    wheel = archive_entry(tmp_path, "robosuite", "robosuite-1.4.1-py3-none-any.whl", xml,
                          "robosuite/models/assets/", "", "robosuite/models/assets/")
    lib = archive_entry(tmp_path, "libero", "code/LIBERO-test.zip", xml, "chiliocosm/assets/",
                        "LIBERO-test/", "libero/libero/assets/", extra={bddl_name: BDDL})
    cat = {wheel.id: wheel, lib.id: lib}
    ep = next(iter_episodes("libero", FIXTURE, root=tmp_path, catalog=cat))
    pv = ep.provenance
    assert pv["state_route"] == "mjcf_states" and not pv["missing_assets"], pv["missing_assets"][:3]
    archives = {a["id"]: a for a in pv["asset_archives"]}
    assert set(archives) == {lib.id, wheel.id} and archives[lib.id]["aliases"] == ["chiliocosm/assets/"]
    assert pv["bddl"]["objects_of_interest"] == ["white_cabinet_1"] and pv["bddl"]["archive"] == lib.id
    m = mujoco.MjModel.from_xml_string(ep.scene.mjcf, ep.scene.assets)
    assert m.nmesh > 0 and ep.scene.robot_prefixes == ["gripper0_", "mount0_", "robot0_"]
    ref = next(iter_episodes("libero", FIXTURE, root=Path("/nonexistent")))
    np.testing.assert_allclose(next(iter(ep.effectors.values())).pose,
                               next(iter(ref.effectors.values())).pose, atol=1e-12)
