"""Control-mode views must preserve frames, clocks and evidence boundaries."""

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from reachy_retarget.control_views import build_control_views, joint_transition_views, pose_matrices, pose_transition_views, se3_log


def poses(xyz, yaw):
    xyz = np.asarray(xyz, float)
    quat = Rotation.from_euler("z", np.asarray(yaw)[:, None]).as_quat()[:, [3, 0, 1, 2]]
    return np.column_stack((xyz, quat))


def fixture():
    names = ["right_1", "neck_pan", "left_1", "base_x", "base_y", "base_yaw"]
    q = np.array([[0, 0, 0, 0, 0, 0], [.1, .2, .3, .4, .5, .6], [.2, .4, .6, .8, 1, 1.2]])
    arrays = {
        "timestamp": np.array([0, .1, .4]),
        "observation/joint_position": q,
        "observation/joint_velocity": np.full_like(q, 7),
        "observation/base_pose": poses([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [0, 0, 0]),
        "observation/left_tcp_pose": poses([[1, 0, 1], [2, 0, 1], [3, 0, 1]], [0, .1, .2]),
        "observation/right_tcp_pose": poses([[1, -1, 1], [2, -1, 1], [3, -1, 1]], [0, .1, .2]),
        "observation/head_pose": poses([[0, 0, 1.5], [1, 0, 1.5], [2, 0, 1.5]], [0, 0, 0]),
    }
    metadata = {"joint_names": names, "joint_groups": {"left_arm": [2], "right_arm": [0], "neck": [1], "base": [3, 4, 5]}}
    return arrays, metadata


def test_joint_wrap_is_explicit_and_raw_difference_survives():
    q = np.deg2rad([[179, 179], [-179, -179]])
    views = joint_transition_views(q, [0, .2], [2 * np.pi, 0])
    np.testing.assert_allclose(np.rad2deg(views["raw_delta"]), [[-358, -358]])
    np.testing.assert_allclose(np.rad2deg(views["coordinate_delta"]), [[2, -358]])
    np.testing.assert_allclose(views["interval_velocity"], views["coordinate_delta"] / .2)


def test_irregular_clock_and_noncontiguous_group_order_are_preserved():
    arrays, metadata = fixture()
    result = build_control_views(arrays, metadata)
    prefix = "observed/joints/left_arm/"
    np.testing.assert_allclose(result["arrays"][prefix + "absolute"], arrays["observation/joint_position"][:, [2]])
    np.testing.assert_allclose(result["arrays"][prefix + "interval_velocity"][:, 0], [3, 1])
    np.testing.assert_allclose(result["arrays"][prefix + "recorded_velocity"], 7)
    np.testing.assert_allclose(result["arrays"]["transition_dt_s"], [.1, .3])
    assert result["metadata"]["views"][prefix + "observed_next_state"]["role"] == "observed_next_state"
    assert result["metadata"]["joint_groups"]["base"]["joint_names"] == ["base_x", "base_y", "base_yaw"]


@pytest.mark.parametrize("clock", [[0, 0], [1, 0], [0, np.nan]])
def test_unknown_or_invalid_interval_cannot_be_used_for_velocity(clock):
    with pytest.raises(ValueError, match="timestamps"):
        joint_transition_views([[0], [1]], clock)


def test_pose_delta_is_group_composition_not_euler_subtraction():
    p = poses([[1, 0, 0], [1, 1, 0]], [np.pi / 2, np.pi])
    views = pose_transition_views(p, [0, .25])
    current, future = pose_matrices(p)
    body = pose_matrices(views["delta_body"])[0]
    world = pose_matrices(views["delta_world_left"])[0]
    np.testing.assert_allclose(current @ body, future, atol=1e-12)
    np.testing.assert_allclose(world @ current, future, atol=1e-12)
    assert not np.allclose(world[:3, 3], p[1, :3] - p[0, :3])
    np.testing.assert_allclose(views["interval_linear_velocity_world"], [[0, 4, 0]])
    np.testing.assert_allclose(views["interval_linear_velocity_body_axes"], [[4, 0, 0]], atol=1e-12)


def test_quaternion_sign_change_is_not_a_physical_rotation():
    p = poses([[0, 0, 0], [0, 0, 0]], [np.pi - 1e-10] * 2)
    p[1, 3:] *= -1
    views = pose_transition_views(p, [0, .1])
    np.testing.assert_allclose(views["interval_se3_log_rate_body"], 0, atol=1e-12)


@pytest.mark.parametrize("angle", [1e-9, .8, np.pi])
def test_se3_log_reconstructs_transform_even_near_zero_or_pi(angle):
    transform = pose_matrices(poses([[.1, -.2, .3]], [angle]))[0]
    omega, rho = np.split(se3_log(transform[None])[0], 2)
    cross = np.array([[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]])
    algebra = np.zeros((4, 4))
    algebra[:3, :3], algebra[:3, 3] = cross, rho
    np.testing.assert_allclose(expm(algebra), transform, atol=1e-12)


def test_bad_quaternion_is_rejected_without_silent_normalization():
    with pytest.raises(ValueError, match="unit wxyz"):
        pose_matrices([[0, 0, 0, 2, 0, 0, 0]])


def test_moving_base_pose_is_relative_not_just_rotated_velocity():
    arrays, metadata = fixture()
    result = build_control_views(arrays, metadata)
    a = result["arrays"]
    np.testing.assert_allclose(a["observed/poses/neck/base/interval_linear_velocity_reference"], 0)
    assert np.any(a["observed/poses/neck/world/interval_linear_velocity_world"])
    assert result["metadata"]["views"]["observed/poses/neck/base/absolute"]["reference_frame"] == "moving_base_at_each_sample"


def test_observed_states_and_actuator_controls_do_not_invent_ee_commands():
    arrays, metadata = fixture()
    arrays["observation/actuator_control"] = np.zeros((3, 2))
    metadata["actuator_names"] = ["motor_a", "motor_b"]
    result = build_control_views(arrays, metadata)
    assert "recorded_commands/observation/actuator_control" in result["arrays"]
    assert "recorded_command/neck/pose" in result["missing_fields"]
    assert "recorded_command/base/joint_velocity" in result["missing_fields"]
    assert all(not kinds for kinds in result["metadata"]["command_coverage"].values())


def test_explicit_recorded_commands_remain_separate_from_next_state_targets():
    arrays, metadata = fixture()
    arrays["command/left_q"] = np.array([[9.], [8.]])
    arrays["command/right_pose"] = arrays["observation/right_tcp_pose"].copy()
    metadata["command_semantics"] = {
        "command/left_q": {"kind": "joint_position", "representation": "absolute", "timing": "pre_action", "joint_names": ["left_1"]},
        "command/right_pose": {"kind": "pose", "representation": "absolute", "timing": "pre_action", "group": "right_arm", "frame": "world", "quaternion_order": "wxyz"},
    }
    result = build_control_views(arrays, metadata)
    np.testing.assert_array_equal(result["arrays"]["recorded_commands/command/left_q"], [[9], [8]])
    assert result["metadata"]["command_coverage"]["left_arm"] == ["joint_position"]
    assert result["metadata"]["command_coverage"]["right_arm"] == ["pose"]
    assert "recorded_command/neck/joint_position" in result["missing_fields"]
    assert result["metadata"]["views"]["recorded_commands/command/left_q"]["timeline"] == "intervals"
    np.testing.assert_allclose(result["arrays"]["command_views/command/left_q/delta_from_observed_current"], [[9], [7.7]])
    identity = pose_matrices(result["arrays"]["command_views/command/right_pose/delta_body"])
    np.testing.assert_allclose(identity, np.broadcast_to(np.eye(4), identity.shape), atol=1e-12)


def test_unknown_delta_reference_is_not_accepted_as_a_command():
    arrays, metadata = fixture()
    arrays["command/qdelta"] = np.ones((3, 1))
    metadata["command_semantics"] = {"command/qdelta": {"kind": "joint_position", "representation": "delta", "timing": "pre_action", "joint_names": ["left_1"]}}
    with pytest.raises(ValueError, match="reference"):
        build_control_views(arrays, metadata)


def test_untimed_absolute_views_do_not_synthesize_rates():
    arrays, metadata = fixture()
    del arrays["timestamp"]
    result = build_control_views(arrays, metadata)
    assert result["metadata"]["timed"] is False
    assert not any("interval_" in key or "delta" in key for key in result["arrays"])
    assert "observed/joints/left_arm/recorded_velocity" in result["arrays"]


def test_group_names_are_explicit_and_reject_duplicate_or_unknown_indices():
    arrays, metadata = fixture()
    metadata["joint_groups"]["left_arm"] = ["left_1", "neck_pan"]
    result = build_control_views(arrays, metadata)
    assert result["metadata"]["joint_groups"]["left_arm"]["indices"] == [2, 1]
    metadata["joint_groups"]["left_arm"] = [2, 2]
    with pytest.raises(ValueError, match="duplicate"):
        build_control_views(arrays, metadata)


def test_recorded_axes_only_and_relative_base_twists_keep_distinct_semantics():
    arrays, metadata = fixture()
    arrays["observation/left_tcp_twist_base"] = np.ones((3, 6))
    arrays["observation/left_tcp_twist_relative_base"] = np.zeros((3, 6))
    result = build_control_views(arrays, metadata)
    views = result["metadata"]["views"]
    assert views["observed/poses/left_arm/recorded_twist_base"]["frame"] == "absolute_velocity_in_base_axes"
    assert views["observed/poses/left_arm/recorded_twist_relative_base"]["frame"] == "motion_relative_to_moving_base"


def test_actual_body_delta_command_can_reconstruct_its_absolute_target():
    arrays, metadata = fixture()
    arrays["command/left_delta"] = poses([[.1, 0, 0]] * 3, [.2] * 3)
    metadata["command_semantics"] = {"command/left_delta": {
        "kind": "pose", "representation": "delta_body", "timing": "pre_action", "reference": "observed_current",
        "group": "left_arm", "frame": "world", "quaternion_order": "wxyz"}}
    result = build_control_views(arrays, metadata)
    actual = pose_matrices(result["arrays"]["command_views/command/left_delta/absolute"])
    expected = pose_matrices(arrays["observation/left_tcp_pose"]) @ pose_matrices(arrays["command/left_delta"])
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_base_planar_velocity_command_keeps_its_three_component_contract():
    arrays, metadata = fixture()
    arrays["observation/base_twist"] = np.ones((3, 3))
    arrays["command/base_velocity"] = np.full((3, 3), .2)
    metadata["command_semantics"] = {"command/base_velocity": {
        "kind": "base_twist", "representation": "absolute", "timing": "pre_action", "group": "base",
        "frame": "base_axes", "component_order": "vx_vy_wz"}}
    result = build_control_views(arrays, metadata)
    assert result["arrays"]["observed/base/recorded_twist_local"].shape == (3, 3)
    assert result["metadata"]["command_coverage"]["base"] == ["base_twist"]
    assert "recorded_command/base/twist" in result["missing_fields"]  # No invented full 6D twist command.
