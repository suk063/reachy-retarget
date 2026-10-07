import numpy as np
import pytest

from reachy_retarget.retarget import gripper_trajectories


def can_metadata():
    return {"source_format": "robomimic-hdf5", "robot_type": "Panda",
            "env_args": {"env_name": "PickPlaceCan", "env_version": "1.5.1",
                         "env_kwargs": {"controller_configs": {"body_parts": {
                             "right": {"type": "OSC_POSE", "gripper": {"type": "GRIP"}}}}}}}


def test_gripper_switches_follow_dilated_clock_without_early_closing():
    source = np.array([0., 2.6, 5.2, 5.85])*3
    target = np.array([0., source[1]-.01, source[1], source[2]-.01, source[2], source[-1]])
    action = np.zeros((4, 7))
    action[:, -1] = [-1, 1, -1, -1]
    values, mapping = gripper_trajectories({"source/action": action}, can_metadata(), source, target)
    np.testing.assert_array_equal(values["right"], [2., 2., -.06, -.06, 2., 2.])
    assert "left" not in values
    assert mapping["right"]["observed_reachy_joint"] is False
    assert mapping["right"]["contact_validated"] is False


def test_unknown_or_analog_gripper_semantics_are_not_guessed():
    action = np.zeros((2, 7))
    with pytest.raises(ValueError, match="Unverified"):
        gripper_trajectories({"source/action": action}, can_metadata(), np.array([0., 1.]), np.array([0., 1.]))
    assert gripper_trajectories({}, {}, np.array([0., 1.]), np.array([0., 1.])) == ({}, {})
