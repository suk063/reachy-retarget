"""Offline tests of the DexMimicGen adapter (bimanual Panda, parallel jaws).

Fixture ``dexmimicgen_threading_demo0_sub.hdf5``: 12 of 196 state rows of ``demo_0`` of
DexMimicGen ``generated/two_arm_threading.hdf5`` (HF ``MimicGen/dexmimicgen_datasets``,
CC-BY-NC-SA-4.0; read with HTTP range requests, so the whole-file digest was not
verified), its recorded MJCF and both arms' ``eef_pos``/``gripper_qpos`` rows; see the
``fixture_note`` attribute. The robosuite 1.5.1 asset archive is a synthetic placeholder.
"""
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from _robosuite_fixtures import archive_entry
from reachy_retarget.sources import families, iter_episodes

FIXTURE = Path(__file__).parent / "fixtures" / "dexmimicgen_threading_demo0_sub.hdf5"


def test_two_effectors_with_side_hints_and_generated_lineage(tmp_path):
    assert "dexmimicgen" in families()
    with h5py.File(FIXTURE) as f:
        xml = f["data/demo_0"].attrs["model_file"]
    wheel = archive_entry(tmp_path, "robosuite", "robosuite-1.5.1-py3-none-any.whl", xml,
                          "robosuite/models/assets/", "", "robosuite/models/assets/")
    ep = next(iter_episodes("dexmimicgen", FIXTURE, root=tmp_path, catalog={wheel.id: wheel}))
    assert ep.provenance["state_route"] == "mjcf_states" and ep.scene is not None
    assert ep.task == "TwoArmThreading" and ep.provenance["env_configuration"] == "single-arm-parallel"
    assert {k: e.side_hint for k, e in ep.effectors.items()} == {"gripper0_right": "right",
                                                                  "gripper1_right": "left"}
    assert ep.provenance["robot_bases_xy_yaw"]["robot0_base"][1] < 0 < \
        ep.provenance["robot_bases_xy_yaw"]["robot1_base"][1]
    assert ep.lineage["generated"] is True and ep.lineage["seed"] is None
    assert ep.success is None  # generated files carry no rewards
    assert set(ep.objects) >= {"needle_obj", "tripod_obj"}
    with h5py.File(FIXTURE) as f:
        for i, k in enumerate(["gripper0_right", "gripper1_right"]):
            eff = ep.effectors[k]
            site = eff.pose[:, :3, 3] + 0.0036 * eff.pose[:, :3, 2]
            # Recorded eef_pos agrees with the state replay to < 1 mm (recording jitter).
            assert np.abs(f[f"data/demo_0/obs/robot{i}_eef_pos"][:] - site).max() < 1e-3
            q = f[f"data/demo_0/obs/robot{i}_gripper_qpos"][:]
            np.testing.assert_allclose(eff.width, q[:, 0] - q[:, 1], atol=1e-9)
            np.testing.assert_allclose(np.linalg.det(eff.pose[:, :3, :3]), 1, atol=1e-9)


def test_dexterous_hand_tasks_are_refused(tmp_path):
    p = tmp_path / "hands.hdf5"
    shutil.copy(FIXTURE, p)
    with h5py.File(p, "a") as f:
        args = json.loads(f["data"].attrs["env_args"])
        args["env_name"] = "TwoArmBoxCleanup"
        f["data"].attrs["env_args"] = json.dumps(args)
    with pytest.raises(ValueError, match="parallel-jaw"):
        next(iter_episodes("dexmimicgen", p))


def test_catalog_lists_only_parallel_jaw_tasks():
    from reachy_retarget.acquire import load_catalog
    files = sorted(e.path for e in load_catalog().values() if e.family == "dexmimicgen")
    assert files == ["generated/two_arm_threading.hdf5", "generated/two_arm_three_piece_assembly.hdf5",
                     "generated/two_arm_transport.hdf5"]
