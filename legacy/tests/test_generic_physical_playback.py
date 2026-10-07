"""Dynamic multi-object replay keeps source state, geometry and failure evidence."""

import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mjviser")
from reachy_retarget.viewer import PhysicalPlayback, hide_redundant_display_geometry


@pytest.fixture
def recording(tmp_path):
    xml = '''<mujoco><worldbody>
      <body name="finger"><joint name="r_hand_finger"/><geom size=".01" mass=".1"/></body>
      <body name="finger_mimic"><joint name="finger_mimic_joint"/><geom size=".01" mass=".1"/></body>
      <body name="cubeA" pos="0 0 1"><freejoint name="cubeA_joint"/>
        <geom name="collider" type="box" size=".02 .03 .04" mass=".1"/>
        <geom name="visual" type="box" size=".02 .03 .04" contype="0" conaffinity="0" mass="0" rgba="1 0 0 1"/>
        <geom name="unmatched_collider" pos=".05 0 0" size=".01" mass=".1"/>
      </body>
      <body name="cubeB" pos=".1 0 1"><freejoint name="cubeB_joint"/><geom size=".03" mass=".2"/></body>
      </worldbody><actuator><position name="finger" joint="r_hand_finger"/></actuator></mujoco>'''
    (tmp_path / "scene.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    state = np.tile(model.qpos0, (2, 1))
    state[:, :2] = [[.5, -.48], [.6, -.56]]
    state[1, 2:5] = [.2, .3, 1.1]
    state[1, 9:12] = [.4, -.2, 1.2]
    path = tmp_path / "replay.h5"
    with h5py.File(path, "w") as f:
        for name, values in {
            "time_s": [.01, .02], "simulation/qpos": state,
            "simulation/qvel": np.ones((2, model.nv)) * .123,
            "simulation/actuator_control": [[-.06], [-.04]],
            "simulation/hand_pose_world": np.zeros((2, 2, 7)),
            "target/right_hand_pose_world": np.zeros((2, 7)),
            "objects/cubeA/pose": state[:, 2:9],
            "objects/cubeB/pose": state[:, 9:16],
            "metrics/bilateral_contact": [False, True],
            "metrics/hand_object_penetration_m": [.0005, .0008],
            "metrics/grasp_drift_m_rad": [[np.nan, np.nan], [.001, .01]],
            "command/gripper_position": [-.06, -.04],
        }.items():
            f[name] = values
    manifest = {"objects": {name: {"body": name, "joint": name + "_joint"}
                            for name in ("cubeA", "cubeB")}}
    return path, model, state, manifest


def test_multi_object_replay_without_joint_metadata(recording):
    path, model, state, manifest = recording
    playback = PhysicalPlayback(path, model, manifest)
    playback.set_frame(1)
    assert playback.generic_objects
    assert set(playback.objects) == {"cubeA", "cubeB"}
    assert playback.targets is None
    assert playback.bilateral_contact[1]
    assert np.isnan(playback.drift[0]).all()  # No fabricated pregrasp anchor.
    np.testing.assert_array_equal(playback.data.qpos, state[1])
    np.testing.assert_array_equal(playback.data.qvel, np.full(model.nv, .123))
    np.testing.assert_array_equal(playback.data.ctrl, [-.04])
    assert playback.data.joint("finger_mimic_joint").qpos[0] == -.56
    np.testing.assert_allclose(playback.data.body("cubeA").xpos, [.2, .3, 1.1])
    np.testing.assert_allclose(playback.data.body("cubeB").xpos, [.4, -.2, 1.2])


@pytest.mark.parametrize("field,bad,match", [
    ("simulation/qvel", np.zeros((2, 1)), "qvel"),
    ("simulation/actuator_control", [[np.nan], [0.]], "actuator control"),
    ("objects/cubeA/pose", np.zeros((2, 7)), "quaternion"),
    ("time_s", [.02, .01], "timestamps"),
])
def test_invalid_recordings_are_rejected(recording, field, bad, match):
    path, model, _, manifest = recording
    with h5py.File(path, "r+") as f:
        del f[field]
        f[field] = bad
    with pytest.raises(ValueError, match=match):
        PhysicalPlayback(path, model, manifest)


def test_object_manifest_must_cover_all_recorded_objects(recording):
    path, model, _, manifest = recording
    del manifest["objects"]["cubeB"]
    with pytest.raises(ValueError, match="Recorded objects differ"):
        PhysicalPlayback(path, model, manifest)


def test_visibility_preserves_physics_and_unmatched_geometry(recording):
    path, model, _, manifest = recording
    fields = ("body_mass", "body_inertia", "geom_type", "geom_size", "geom_friction",
              "geom_contype", "geom_conaffinity", "geom_rgba", "geom_pos", "geom_quat")
    before = {name: getattr(model, name).copy() for name in fields}
    urdf = ET.fromstring('<robot><link name="finger"><visual/></link></robot>')
    hide_redundant_display_geometry(model, urdf)
    assert model.geom_group[0] == 3
    assert model.geom("collider").group[0] == 3
    assert model.geom("visual").group[0] != 3
    assert model.geom("unmatched_collider").group[0] != 3
    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(model, name), expected)
    PhysicalPlayback(path, model, manifest).set_frame(1)
