"""Physical replay must preserve contact deflections instead of command targets."""

import h5py
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mjviser")
from reachy_retarget.viewer import PhysicalPlayback


def test_replay_preserves_actual_fingers_and_free_object(tmp_path):
    xml = '''<mujoco><worldbody>
      <body><joint name="r_hand_finger"/><geom size=".01" mass=".1"/></body>
      <body pos=".1 0 0"><joint name="r_hand_finger_distal_mimic"/><geom size=".01" mass=".1"/></body>
      <body name="Can" pos="0 0 1"><freejoint name="Can_joint"/><geom size=".02" mass=".1"/></body>
      </worldbody><actuator><position name="finger" joint="r_hand_finger"/></actuator></mujoco>'''
    (tmp_path / "scene.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    state = np.tile(model.qpos0, (2, 1))
    state[:, :2] = [[.5, -.48], [.6, -.56]]
    state[1, 2:5] = [.1, .2, 1.1]
    path = tmp_path / "replay.h5"
    with h5py.File(path, "w") as f:
        for name, values in {
            "time_s": [.01, .02], "simulation/qpos": state,
            "simulation/qvel": np.zeros((2, model.nv)),
            "simulation/actuator_control": [[-.06], [-.06]],
            "simulation/hand_pose_world": np.zeros((2, 2, 7)),
            "target/hand_pose_world": np.zeros((2, 2, 7)),
            "simulation/Can_pose": state[:, 2:],
            "simulation/contacts_hand_force_open": np.zeros((2, 3)),
            "metrics/hand_can_penetration_m": [.0005, .0008],
            "command/gripper_position": [-.06, -.06],
            "simulation/model_jnt_qposadr": model.jnt_qposadr,
        }.items():
            f[name] = values
        f["simulation/model_joint_names"] = np.asarray(
            [model.joint(i).name for i in range(model.njnt)], dtype=h5py.string_dtype())
    playback = PhysicalPlayback(path, model, {"mimics": {"r_hand_finger_distal_mimic": ["r_hand_finger", -1, 0]}})
    playback.set_frame(1)
    np.testing.assert_array_equal(playback.data.qpos, state[1])
    assert playback.channels["r_hand_finger"][1] == .6
    assert playback.data.joint("r_hand_finger_distal_mimic").qpos[0] == -.56
    assert playback.data.ctrl[0] == -.06
    np.testing.assert_allclose(playback.data.body("Can").xpos, [.1, .2, 1.1])
    model.body_pos[1, 0] += .01
    with pytest.raises(AssertionError, match="physical body_pos"):
        PhysicalPlayback(path, model, {})
