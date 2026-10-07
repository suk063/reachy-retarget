import sys
from pathlib import Path

import numpy as np
import pytest

from reachy_retarget.schema import DT, JOINTS, MODES, SIDES, compute_modes
from reachy_retarget.schema import rotations as rot
from reachy_retarget.schema.control_modes import V8_ACTION_NAMES, V8_STATE_NAMES

from test_schema import make_episode

REACHY_AGENT = Path(__file__).resolve().parents[2] / "reachy-agent"


@pytest.fixture(scope="module")
def ep():
    return make_episode(T=60)


@pytest.fixture(scope="module")
def modes(ep):
    return compute_modes(ep)


@pytest.fixture(scope="module")
def agent():
    if not (REACHY_AGENT / "robot" / "actions.py").exists():
        pytest.skip("reachy-agent checkout not found")
    sys.path.insert(0, str(REACHY_AGENT))
    try:
        from policy import inputs
        from robot import actions, geometry
    except ImportError as e:
        pytest.skip(f"reachy-agent not importable: {e}")
    finally:
        sys.path.remove(str(REACHY_AGENT))
    return actions, inputs, geometry


def test_shapes_and_docs(ep, modes):
    T = ep.length
    assert set(modes) >= {"joint_pos_abs", "joint_pos_delta", "joint_vel", "ee_abs_base", "ee_delta_base",
                          "ee_abs_world", "head_rot_abs", "head_rot_delta", "base_twist_body",
                          "base_se2_delta", "gripper_continuous", "gripper_binary", "reachy_agent_v8"}
    for name, arrays in modes.items():
        spec = MODES[name]
        assert spec.frame and spec.units and spec.semantics
        for k, a in arrays.items():
            rows = T if (name, k) == ("reachy_agent_v8", "state") else T - 1
            assert a.shape == (rows, len(spec.columns[k])), (name, k)
    assert modes["reachy_agent_v8"]["action"].shape == (T - 1, 29)
    assert modes["reachy_agent_v8"]["state"].shape == (T, 30)
    assert modes["joint_pos_abs"]["action"].shape == (T - 1, 22)
    assert set(np.unique(modes["gripper_binary"]["open"])) <= {0.0, 1.0}


def test_joint_delta_integrates(ep, modes):
    q = ep.q[0] + np.r_[np.zeros((1, 22)), np.cumsum(modes["joint_pos_delta"]["action"], axis=0)]
    q[:, 2] = rot.wrap_angle(q[:, 2])
    expected = ep.q.copy()
    expected[:, 2] = rot.wrap_angle(expected[:, 2])
    np.testing.assert_allclose(q, expected, atol=1e-9)
    np.testing.assert_allclose(modes["joint_pos_abs"]["action"], ep.q[1:])


def test_se2_integrates(ep, modes):
    b = [ep.q[0, :3]]
    for d, tw in zip(modes["base_se2_delta"]["delta"], modes["base_twist_body"]["twist"]):
        np.testing.assert_allclose(rot.se2_exp(tw, DT), d, atol=1e-12)
        b.append(rot.se2_compose(b[-1], d))
    b = np.array(b)
    np.testing.assert_allclose(b[:, :2], ep.q[:, :2], atol=1e-9)
    np.testing.assert_allclose(rot.wrap_angle(b[:, 2] - ep.q[:, 2]), 0, atol=1e-9)


def test_se3_integrates(ep, modes):
    d = modes["ee_delta_base"]
    for s in SIDES:
        P = ep.tcp_base[s]
        tool, base, twist = [P[0]], [P[0]], [P[0]]
        for t in range(ep.length - 1):
            D = np.eye(4)
            D[:3, :3], D[:3, 3] = rot.so3_exp(d[f"{s}_tool"][t, 3:]), d[f"{s}_tool"][t, :3]
            tool.append(tool[-1] @ D)
            B = np.eye(4)
            B[:3, :3] = rot.so3_exp(d[f"{s}_base"][t, 3:]) @ base[-1][:3, :3]
            B[:3, 3] = base[-1][:3, 3] + d[f"{s}_base"][t, :3]
            base.append(B)
            twist.append(twist[-1] @ rot.se3_exp(modes["ee_twist_tool"][s][t] * DT))
        for track in (tool, base, twist):
            np.testing.assert_allclose(np.array(track), P, atol=1e-9)


def test_head_and_ee_abs(ep, modes):
    np.testing.assert_allclose(rot.rot6d_to_matrix(modes["head_rot_abs"]["rot6d"]), ep.head_base[1:, :3, :3], atol=1e-12)
    H = [ep.head_base[0, :3, :3]]
    for r in rot.rot6d_to_matrix(modes["head_rot_delta"]["rot6d"]):
        H.append(H[-1] @ r)
    np.testing.assert_allclose(np.array(H), ep.head_base[:, :3, :3], atol=1e-9)
    for frame, poses in (("base", ep.tcp_base), ("world", ep.tcp_world)):
        a = modes[f"ee_abs_{frame}"]
        for s in SIDES:
            np.testing.assert_allclose(rot.vec7_to_pose(a[f"{s}_pos_quat"]), poses[s][1:], atol=1e-12)
            np.testing.assert_allclose(rot.rot6d_to_matrix(a[f"{s}_pos_rot6d"][:, 3:]), poses[s][1:, :3, :3], atol=1e-12)


def test_v8_names_match_reachy_agent(agent):
    actions, inputs, _ = agent
    assert V8_ACTION_NAMES == actions.ACTION_NAMES
    assert V8_STATE_NAMES == inputs.policy_state_names(actions.HANDS)
    assert MODES["reachy_agent_v8"].columns == {"action": actions.ACTION_NAMES,
                                                "state": inputs.policy_state_names(actions.HANDS)}
    assert actions.ACTION_CONTRACT in MODES["reachy_agent_v8"].semantics
    assert inputs.POLICY_STATE_CONTRACT in MODES["reachy_agent_v8"].semantics


def test_v8_values_match_reachy_agent(agent, ep, modes):
    actions, _, geometry = agent
    v8 = modes["reachy_agent_v8"]
    for t in (0, 17, ep.length - 2):
        n = t + 1
        expected = np.r_[actions.pose_values(ep.tcp_base["left"][n]), actions.pose_values(ep.tcp_base["right"][n]),
                         actions.rotation_values(ep.head_base[n, :3, :3]),
                         geometry.relative_planar(ep.q[t, :3], ep.q[n, :3]),
                         ep.gripper_opening[n] >= 0.5]
        expected[26] = geometry.wrap_angle(expected[26])
        np.testing.assert_allclose(v8["action"][t], expected, atol=1e-6)
        actions.checked_action(v8["action"][t])
        decoded = actions.decode_action(v8["action"][t])
        np.testing.assert_allclose(decoded[0]["left"], ep.tcp_base["left"][n], atol=1e-5)
        b = ep.q[t, :3]
        state = np.r_[actions.pose_values(ep.tcp_base["left"][t]), actions.pose_values(ep.tcp_base["right"][t]),
                      actions.rotation_values(ep.head_base[t, :3, :3]), b[0], b[1], np.sin(b[2]), np.cos(b[2]),
                      ep.gripper_opening[t]]
        np.testing.assert_allclose(v8["state"][t], state, atol=1e-6)
    assert v8["action"].dtype == np.float32


def test_joint_names_order():
    assert JOINTS[:3] == ("base_x", "base_y", "base_yaw")
    assert JOINTS[3] == "l_shoulder_pitch" and JOINTS[16] == "r_wrist_yaw"
    assert JOINTS[17:] == ("neck_roll", "neck_pitch", "neck_yaw", "l_hand_finger", "r_hand_finger")
