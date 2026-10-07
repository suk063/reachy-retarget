import json

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget.episodes import read_episode
from reachy_retarget.maniskill import SOURCE_COMMIT, URDF_URL, maniskill, panda_tcp_matrices, verify_pinocchio
from reachy_retarget.store import Store, sha256


def synthetic_panda(path):
    xml = ['<robot name="panda"><link name="panda_link0"/>']
    for i in range(1, 8):
        xml += [f'<link name="panda_link{i}"/>',
                f'<joint name="panda_joint{i}" type="revolute"><parent link="panda_link{i-1}"/><child link="panda_link{i}"/>'
                '<origin xyz="0 0 0.1"/><axis xyz="0 0 1"/><limit effort="100" velocity="3" lower="-3.2" upper="3.2"/></joint>']
    xml += ['<link name="panda_hand_tcp"/><joint name="tcp" type="fixed"><parent link="panda_link7"/><child link="panda_hand_tcp"/><origin xyz="0.1 0 0"/></joint>']
    for i in (1, 2):
        xml += [f'<link name="finger{i}"/><joint name="panda_finger_joint{i}" type="prismatic"><parent link="panda_link7"/><child link="finger{i}"/>'
                '<axis xyz="0 1 0"/><limit effort="100" velocity="1" lower="0" upper="0.04"/></joint>']
    xml.append('</robot>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(xml))


def test_tcp_fk_respects_root_rotation_and_measured_joint_state(tmp_path):
    urdf = tmp_path / "panda.urdf"
    synthetic_panda(urdf)
    qpos = np.zeros((2, 9))
    qpos[1, 0] = np.pi / 2
    quat = Rotation.from_euler("z", np.pi / 2).as_quat()[[3, 0, 1, 2]]
    root = np.tile(np.r_[1, 2, 3, quat], (2, 1))
    matrices = panda_tcp_matrices(urdf, qpos, root)
    np.testing.assert_allclose(matrices[0, :3, 3], [1, 2.1, 3.7], atol=1e-10)
    np.testing.assert_allclose(matrices[1, :3, 3], [0.9, 2, 3.7], atol=1e-10)
    pytest.importorskip("pinocchio")
    assert verify_pinocchio(urdf, qpos, root, matrices)["max_matrix_error"] < 1e-10


def fixture_release(tmp_path, *, bad_commit=False, truncate_state=False):
    store = Store(tmp_path)
    source = "hf__haosulab__ManiSkill_Demonstrations"
    directory = tmp_path / "data/raw" / source / "demos/PickCube-v1/teleop"
    directory.mkdir(parents=True)
    info = {"env_info": {"env_id": "PickCube-v1", "env_kwargs": {"obs_mode": "none", "control_mode": "pd_joint_pos"}},
            "commit_info": {"commit_id": "unknown" if bad_commit else SOURCE_COMMIT},
            "episodes": [{"episode_id": 0, "success": True}]}
    (directory / "trajectory.json").write_text(json.dumps(info))
    with h5py.File(directory / "trajectory.h5", "w") as f:
        group = f.create_group("traj_0")
        group["actions"] = np.zeros((2, 8))
        actor = np.zeros((3, 13)); actor[:, 3] = 1
        state = np.zeros((3, 31)); state[:, 3] = 1; state[1:, 13] = np.pi / 2
        group["env_states/actors/cube"] = actor
        group["env_states/articulations/panda"] = state[:-1] if truncate_state else state
        group["success"] = np.array([False, True])
    urdf = tmp_path / "data/raw/maniskill_panda_assets/panda_v2.urdf"
    synthetic_panda(urdf)
    store.file("maniskill_panda_assets", "panda_v2.urdf", URDF_URL)
    store.update_file("maniskill_panda_assets", "panda_v2.urdf", status="downloaded", sha256=sha256(urdf))
    return store, source, directory


def test_normalize_preserves_terminal_state_without_fabricating_action(tmp_path):
    store, source, _ = fixture_release(tmp_path)
    path = maniskill(store, source, 1)[0]
    arrays, metadata = read_episode(path)
    assert arrays["time_s"].shape == (3,)
    assert arrays["source/action"].shape == (2, 8)
    assert arrays["objects/cube/pose"].shape == (3, 7)
    assert arrays["hand/right_pose"].shape == (3, 7)
    np.testing.assert_allclose(arrays["hand/right_pose"][1, :3], [0, 0.1, 0.7], atol=1e-10)
    assert "contact" not in arrays and not metadata["physics_validated"]
    assert metadata["hand_pose_derivation"]["tcp_frame"] == "panda_hand_tcp"


@pytest.mark.parametrize("kwargs,match", [({"bad_commit": True}, "Unverified ManiSkill"),
                                        ({"truncate_state": True}, "T\\+1")])
def test_reject_unverified_schema_and_missing_terminal_state(tmp_path, kwargs, match):
    store, source, _ = fixture_release(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=match):
        maniskill(store, source, 1)


def test_reject_state_order_when_first_named_command_disagrees(tmp_path):
    store, source, directory = fixture_release(tmp_path)
    with h5py.File(directory / "trajectory.h5", "r+") as f:
        f["traj_0/actions"][0, 0] = 0.2
    with pytest.raises(ValueError, match="qpos ordering"):
        maniskill(store, source, 1)
