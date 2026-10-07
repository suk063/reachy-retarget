"""Real state recording must describe its target clock, model and controls."""

import copy
import hashlib
import json

import h5py
import mujoco
import numpy as np
import pytest

from reachy_retarget.state_recording import RolloutRecorder, _recording_metadata


XML = """<mujoco><option timestep=".002"/><default><joint armature=".1" damping="1"/></default><worldbody>
<body name="base_link" pos="0 0 1"><freejoint/><geom type="sphere" size=".05"/>
 <body name="left_arm" pos="0 .2 0"><joint name="l_shoulder_pitch"/>
  <geom type="sphere" size=".02"/><site name="l_arm_tip_tcp" pos=".1 0 0"/>
 </body>
 <body name="right_arm" pos="0 -.2 0"><joint name="r_shoulder_pitch"/>
  <geom type="sphere" size=".02"/><site name="r_arm_tip_tcp" pos=".1 0 0"/>
 </body>
 <body name="head" pos="0 0 .2"><joint name="neck_yaw"/>
  <geom type="sphere" size=".02"/><camera name="left_head"/>
 </body>
</body>
<body name="Can_main" pos="1 0 .3"><freejoint/><geom type="cylinder" size=".03 .05"/></body>
<body name="bin1" pos="2 0 0"><geom type="box" size=".1 .1 .1"/></body>
<body name="bin2" pos="3 0 0"><geom type="box" size=".1 .1 .1"/></body>
</worldbody><actuator>
<position name="left" joint="l_shoulder_pitch" kp="10"/>
<position name="right" joint="r_shoulder_pitch" kp="10"/>
<position name="neck" joint="neck_yaw" kp="10"/>
</actuator></mujoco>"""
RAW_SHA = hashlib.sha256(b"offline raw source fixture").hexdigest()
SOURCE = {"source_id": "source-dataset", "source_sequence": "demo-1", "source_group": "group/demo-1",
          "source_format": "original-source", "fps": 20, "robot_type": "Panda",
          "model_xml": "original Panda scene", "action_semantics": "source OSC",
          "env_args": {"robots": ["Panda"]}, "hdf5_sha256": "source-normalized-hash",
          "content_fingerprint": "source-fingerprint", "episode_id": "original-episode",
          "provenance": [{"path": "raw.hdf5", "sha256": RAW_SHA}]}
PROVENANCE = {"source_urls": ["https://example.invalid/pinned/raw.hdf5"],
              "source_revision": "pinned-revision",
              "provenance": [{"url": "https://example.invalid/pinned/raw.hdf5", "sha256": RAW_SHA}]}
REPORT = {"physics": {"simulation_assumptions": ["Offline test model"], "source_urdf": "reference.urdf"},
          "status": "physical_fail", "physics_validated": False}


def recorder(tmp_path):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    manifest = {"objects": {"Can": {"body": "Can_main"}},
                "scene_sha256": hashlib.sha256(XML.encode()).hexdigest(),
                "source_urdf_sha256": hashlib.sha256(b"offline reference fixture").hexdigest()}
    return model, data, RolloutRecorder(model, manifest, {"task": "PickPlaceCan", "object_id": "Can"}, tmp_path)


def test_real_recording_replaces_source_semantics_preserves_provenance_and_state(tmp_path):
    model, data, recording = recorder(tmp_path)
    original = copy.deepcopy(SOURCE)
    for frame in range(3):
        data.ctrl[:] = [frame * .01, -frame * .01, .02]
        state = {"identity": {"test": "offline fixture"}, "format": "test-servo-state",
                 "joints": {}, "targets": {"joint_position": data.ctrl.tolist()},
                 "memory": {"source_step": frame}, "servo_targets": {}}
        recording.begin(data, state)
        for _ in range(5):
            recording.before_step(data)
            mujoco.mj_step(model, data)
        recording.end(data)
    before = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.time)
    result = recording.finish(data, SOURCE, REPORT, "source-episode", source_provenance=PROVENANCE)
    np.testing.assert_array_equal(data.qpos, before[0])
    np.testing.assert_array_equal(data.qvel, before[1])
    np.testing.assert_array_equal(data.ctrl, before[2])
    assert data.time == before[3]
    assert SOURCE == original
    with h5py.File(result["archive"], "r") as f:
        meta = json.loads(f.attrs["metadata_json"])
        assert meta["source_metadata"] == original
        assert meta["robot_type"] == "Reachy2"
        assert meta["fps"] == pytest.approx(100.)
        assert meta["clock"]["sampling_rate_hz"] == pytest.approx(100.)
        assert meta["clock"]["uniform_sampling"] is True
        assert meta["source_revision"] == "pinned-revision"
        assert meta["source_urls"] == PROVENANCE["source_urls"]
        assert meta["provenance_missing_fields"] == []
        assert meta["provenance"] == original["provenance"] + PROVENANCE["provenance"]
        assert meta["source_normalized_hdf5_sha256"] == original["hdf5_sha256"]
        assert not set(meta) & {"model_xml", "env_args", "hdf5_sha256", "content_fingerprint"}
        assert "Reachy2 joint position" in meta["action_semantics"]
        assert "command/joint_position" in meta["control_labels"]["desired_joint_targets"]
        assert "controller_state_json" in meta["control_labels"]["controller_state"]
        assert "applied_controls" in meta["control_labels"]["substep_controls"]
        assert meta["target_model"]["scene_sha256"] == recording.metadata["model_identity"]["scene_sha256"]
        assert meta["target_model"]["source_urdf_path"] == "reference.urdf"
        assert meta["physics_validated"] is False
        assert f["applied_controls"].shape == (3, 5, 3)
        np.testing.assert_array_equal(f["command/joint_position"], f["applied_controls"][:, 0])
        np.testing.assert_array_equal(f["terminal/observation/qpos"], data.qpos)
        assert f["terminal/timestamp"][()] == data.time


@pytest.mark.parametrize("timestamps", [[0., .01, .025], [0.]])
def test_irregular_or_unknown_clock_never_inherits_source_fps(tmp_path, timestamps):
    _, _, recording = recorder(tmp_path)
    meta = _recording_metadata(recording.metadata, SOURCE, timestamps, REPORT, "episode", tmp_path)
    assert meta["fps"] is None
    assert meta["clock"]["sampling_rate_hz"] is None
    assert meta["source_metadata"]["fps"] == 20
    assert meta["source_revision"] is None and meta["source_urls"] == []
    assert meta["provenance_missing_fields"] == ["source_urls", "source_revision"]


def test_external_provenance_requires_exact_existing_raw_checksum(tmp_path):
    _, _, recording = recorder(tmp_path)
    bad = copy.deepcopy(PROVENANCE)
    bad["provenance"][0]["sha256"] = "a" * 64
    with pytest.raises(ValueError, match="SHA256 does not match"):
        _recording_metadata(recording.metadata, SOURCE, [0., .01], REPORT, "episode", tmp_path,
                            source_provenance=bad)
    conflicting = dict(SOURCE, source_revision="different-revision")
    with pytest.raises(ValueError, match="revision conflicts"):
        _recording_metadata(recording.metadata, conflicting, [0., .01], REPORT, "episode", tmp_path,
                            source_provenance=PROVENANCE)


def test_metadata_failure_preserves_actual_state_controls_and_terminal_boundary(tmp_path):
    model,data,recording=recorder(tmp_path)
    for frame in range(2):
        data.ctrl[:]=[.1,-.2,.03]
        recording.begin(data,{'format':'test','memory':{'source_step':frame}})
        for _ in range(5):
            recording.before_step(data);mujoco.mj_step(model,data)
        recording.end(data)
    bad=copy.deepcopy(PROVENANCE);bad['provenance'][0]['sha256']='a'*64
    with pytest.raises(ValueError,match='SHA256'):
        recording.finish(data,SOURCE,REPORT,'source-episode',source_provenance=bad)
    with h5py.File(tmp_path/'state-attempt.hdf5') as f:
        metadata=json.loads(f.attrs['metadata_json'])
        assert not metadata['metadata_complete'] and not metadata['physics_validated']
        assert 'metadata_error' in metadata
        assert len(f['timestamp'])==2 and f['applied_controls'].shape==(2,5,3)
        np.testing.assert_array_equal(f['terminal/observation/qpos'],data.qpos)
        assert f['terminal/timestamp'][()]==data.time
    assert not (tmp_path/'common').exists()
