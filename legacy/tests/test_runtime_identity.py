"""Offline checks that local previews cannot replace the native identity record."""

import json

import pytest
import numpy as np
from types import SimpleNamespace

from reachy_retarget import robot


def test_reference_identity_is_separate_and_rejects_changes(tmp_path, monkeypatch):
    control = tmp_path / "control_checkout"
    for name in ("control/reachy.py", "control/collision.py", "asset/reachy.urdf", "asset/collision_spheres.json"):
        path = control / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    root = tmp_path / "preview"
    (root / "catalog").mkdir(parents=True)
    native = root / "catalog/controller_identity.json"
    native.write_text('{"native": "preserve"}\n')
    monkeypatch.setattr(robot, "CONTROL", control)
    monkeypatch.setattr(robot, "BACKEND", "reference")
    first = robot.identity(root)
    assert robot.identity(root) == first
    assert json.loads(native.read_text()) == {"native": "preserve"}
    (control / "asset/reachy.urdf").write_text("modified model")
    with pytest.raises(RuntimeError, match="identity changed"):
        robot.identity(root)


def test_native_identity_does_not_fall_back_to_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(robot, "CONTROL", tmp_path / "missing_native_checkout")
    monkeypatch.setattr(robot, "BACKEND", "native")
    with pytest.raises(FileNotFoundError):
        robot.identity(tmp_path)


@pytest.mark.parametrize("camera_api", [False, True])
def test_world_targets_use_controller_target_frame(camera_api):
    pin = pytest.importorskip("pinocchio")
    base = pin.SE3(pin.rpy.rpyToMatrix(.1, .2, .3), np.array([.2, -.3, .4]))
    camera = pin.SE3(pin.rpy.rpyToMatrix(-.5, .7, 1.), np.array([.6, .1, 1.2]))
    head = pin.SE3(pin.rpy.rpyToMatrix(.3, -.2, .1), np.zeros(3))
    controller = SimpleNamespace(data=SimpleNamespace(oMi=[base], oMf=[camera, head]),
                                 base_joint=0, head=1, update=lambda q: None,
                                 control=lambda q, left, right, neck: (left, right, neck))
    if camera_api:
        controller.camera = 0
    adapter = object.__new__(robot.Robot)
    adapter.r, adapter.pin = controller, pin
    goals = [pin.SE3.Identity(), pin.SE3(np.eye(3), np.array([.8, -.4, 1.]))]
    left, right, neck = adapter.control(np.zeros(3), [g.homogeneous for g in goals])
    frame = camera if camera_api else base
    for local, world in zip((left, right), goals):
        np.testing.assert_allclose((frame * local).homogeneous, world.homogeneous, atol=1e-14)
    np.testing.assert_allclose(frame.rotation @ neck, head.rotation, atol=1e-14)


def test_body_attached_hand_motion_is_expressed_in_camera_axes():
    pin = pytest.importorskip("pinocchio")
    base = pin.SE3(pin.rpy.rpyToMatrix(0., 0., .6), np.array([.2, -.3, 0.]))
    camera = pin.SE3(pin.rpy.rpyToMatrix(-.5, .7, 1.), np.array([.6, .1, 1.2]))
    controller = SimpleNamespace(data=SimpleNamespace(oMi=[base], oMf=[camera]),
                                 base_joint=0, camera=0, head=0, update=lambda q: None,
                                 control=lambda *args, **kwargs: kwargs["hand_motion"])
    adapter = object.__new__(robot.Robot)
    adapter.r, adapter.pin = controller, pin
    local_left = pin.SE3(np.eye(3), np.array([.4, .2, .8]))
    goals = [(base * local_left).homogeneous, pin.SE3.Identity().homogeneous]
    velocity = np.array([[0., 0., 0., 0., 0., 0.], [.1, .2, .3, -.1, .4, .2]])
    motion = adapter.control(np.zeros(3), goals, body_attached_hands=(0,), world_goal_velocity=velocity)
    np.testing.assert_allclose(motion["jacobian"][1], 0.)
    for axis in range(3):
        moved = base.copy()
        if axis < 2:
            moved.translation += base.rotation[:, axis] * 1e-7
        else:
            moved.rotation = pin.rpy.rpyToMatrix(0., 0., .6+1e-7)
        derivative = ((moved * local_left).translation - goals[0][:3, 3]) / 1e-7
        np.testing.assert_allclose(camera.rotation @ motion["jacobian"][0, :3, axis], derivative, atol=3e-8)
    np.testing.assert_allclose(motion["velocity"][:, :3] @ camera.rotation.T, velocity[:, :3], atol=1e-14)
    np.testing.assert_allclose(motion["velocity"][:, 3:] @ camera.rotation.T, velocity[:, 3:], atol=1e-14)
