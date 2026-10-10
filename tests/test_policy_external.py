"""reachy-policy in external/ (README, Policy): its robot model must be this repository's URDF."""
import pytest

from reachy_retarget.robot import URDF_FILE

model = pytest.importorskip("policy.robot.model", reason="reachy-policy is not installed (README, Policy)")


def test_policy_robot_model_is_this_urdf():
    assert model.MODEL_FILE.read_bytes() == URDF_FILE.read_bytes()
