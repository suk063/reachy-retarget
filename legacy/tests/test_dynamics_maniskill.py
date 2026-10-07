import copy

import numpy as np
import pytest

from reachy_retarget import dynamics_maniskill as adapter


def example():
    times = np.arange(6) * .05
    actions = np.zeros((5, 8)); actions[:, 7] = [1, 1, -1, -1, -1]
    poses = np.tile([0, 0, .02, 1, 0, 0, 0], (6, 1)).astype(float)
    poses[:, 2] = [.02, .02, .02, .04, .08, .10]
    hands = poses.copy(); hands[:, 0] += .012
    arrays = {"time_s": times, "source/action_time_s": times[:-1].copy(),
              "source/action": actions, "hand/right_pose": hands, "objects/cube/pose": poses,
              "source/env_states/actors/goal_site": np.tile([.1, .1, .12, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], (6, 1))}
    metadata = {"source_format": "ManiSkill-HDF5", "robot_type": "Panda", "source_commit": adapter.SOURCE_COMMIT,
                "env_args": {"env_name": "PickCube-v1", "env_kwargs": {"control_mode": "pd_joint_pos"}}}
    return arrays, metadata


def test_gripper_sign_clock_terminal_state_and_center():
    arrays, metadata = example()
    original = copy.deepcopy(arrays)
    E, O, grip, anchor, local, region = adapter.source_grasp(arrays, metadata)
    assert E.shape == O.shape == (6, 4, 4)
    np.testing.assert_array_equal(grip, [-1, -1, 1, 1, 1, 1])
    assert anchor == 4
    np.testing.assert_array_equal(local, [0, 0, 0])
    assert "unchanged source box" in region
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], original[key])
    assert arrays["source/action"].shape == (5, 8)
    measured_local = adapter.source_grasp(arrays, metadata, geometry_center=False)[4]
    np.testing.assert_allclose(measured_local, [.012, 0, 0])


def test_official_goal_transforms_with_scene_and_is_not_final_demo_pose():
    arrays, metadata = example()
    placement = np.array([[0, -1, 0, .5], [1, 0, 0, -.1], [0, 0, 1, .8], [0, 0, 0, 1]])
    context = adapter.task_context(arrays, metadata, placement)
    np.testing.assert_allclose(context["goal_world_m"], [.4, 0, .92], atol=1e-12)
    assert context["release_required"] is False
    assert not np.allclose(context["goal_source_m"], arrays["objects/cube/pose"][-1, :3])


def test_official_static_uses_peak_arm_speed_and_adds_explicit_base_rule():
    arrays, metadata = example()
    context = adapter.task_context(arrays, metadata, np.eye(4))
    goal = np.asarray(context["goal_world_m"])
    # The source uses max absolute joint velocity, not the vector norm.
    assert adapter.task_metrics(goal, np.full(14, .19), [0, 0, 0], context)["success"]
    assert not adapter.task_metrics(goal, [.201] * 14, [0, 0, 0], context)["success"]
    assert not adapter.task_metrics(goal + [.026, 0, 0], np.zeros(14), [0, 0, 0], context)["success"]
    assert not adapter.task_metrics(goal, np.zeros(14), [.019, .019, 0], context)["success"]
    assert not adapter.task_metrics(goal, np.zeros(14), [0, 0, .201], context)["success"]
    assert not adapter.task_metrics(goal, [np.nan] * 14, [0, 0, 0], context)["success"]


@pytest.mark.parametrize("issue", ["wrong_sign_contract", "missing_terminal_state", "wrong_clock", "wrong_revision", "no_lift"])
def test_unsupported_source_contract_fails_without_guessing(issue):
    arrays, metadata = example()
    if issue == "wrong_sign_contract": arrays["source/action"][2, 7] = .4
    if issue == "missing_terminal_state": arrays["source/action"] = np.vstack([arrays["source/action"], arrays["source/action"][-1]])
    if issue == "wrong_clock": arrays["source/action_time_s"] += .05
    if issue == "wrong_revision": metadata["source_commit"] = "unknown"
    if issue == "no_lift": arrays["objects/cube/pose"][:, 2] = .02
    with pytest.raises(ValueError):
        adapter.source_grasp(arrays, metadata)


def test_moving_goal_requires_a_different_adapter():
    arrays, metadata = example()
    arrays["source/env_states/actors/goal_site"][-1, 0] += .1
    with pytest.raises(ValueError, match="stationary source goal"):
        adapter.task_context(arrays, metadata, np.eye(4))
