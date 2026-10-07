"""Uniform archives preserve source semantics and cannot masquerade as v5."""

import json

import h5py
import numpy as np
import pytest

from reachy_retarget.agent_dataset import (
    ACTION_CONTRACT, ACTION_NAMES, SCHEMA, STAGES, V5_SCHEMA,
    assess_v5, export_normalized, import_v5, inspect_archive, write_archive,
)
from reachy_retarget.store import sha256


def normalized_record():
    pose = np.tile([0., 0., 0., 1., 0., 0., 0.], (3, 1))
    pose[1] = np.nan
    return {"arrays": {"time_s": np.array([0., .03, .06]),
                       "objects/can/pose": pose, "objects/can/valid": np.array([True, False, True]),
                       "hand/right_pose": np.tile([0., 0., 0., 1., 0., 0., 0.], (3, 1)),
                       "source/action": np.ones((3, 7)), "human/joint_names": np.array(["wrist", "elbow"])},
            "metadata": {"source_id": "example", "episode_id": "one", "source_sequence": "recording/crop",
                         "source_group": "recording", "objects": {"can": {"pose_frame": "world"}},
                         "source_urls": ["https://example.test/data"], "source_revision": "v1"},
            "status": "normalized", "missing_fields": ["contact_force"]}


def native_fixture(path, mutate=None):
    # Synthetic *format* fixture only: this test never asserts physical success.
    t = 3
    pose = np.tile([0., 0., 0., 1., 0., 0., 0.], (t, 1))
    arrays = {"timestamp": np.arange(t)*.1, "action": np.zeros((t, 26), np.float32),
              "applied_controls": np.zeros((t, 50, 1), np.float32), "physics_state": np.zeros((t, 8)),
              "stage_index": np.zeros(t, np.uint8), "state_tick": np.arange(t)*50,
              "image_tick": np.arange(t)*50, "observation/joint_position": np.zeros((t, 1)),
              "observation/joint_velocity": np.zeros((t, 1)), "observation/base_twist": np.zeros((t, 3)),
              "observation/body_poses": pose[:, None, :], "observation/cameras/torso/pose": pose,
              "observation/cameras/torso/rgb": np.zeros((t, 256, 256, 3), np.uint8)}
    for name in ("base", "pickup", "receptacle", "left_tcp", "right_tcp"):
        arrays[f"observation/{name}_pose"] = pose.copy()
    for offset in (3, 12, 18):
        arrays["action"][:, offset:offset+6] = [1, 0, 0, 0, 1, 0]
    state = {"base": [0, 0, 0], "joints": {"arm": 0}, "targets": {"arm": "synthetic-test-only"},
             "memory": {"last_velocity": [0]}, "servo_targets": {"joints": {"arm": 0}},
             "identity": "synthetic-test-only", "format": "reachy-control-state-v3"}
    arrays["controller_state_json"] = np.array([json.dumps(state)]*t)
    metadata = {"schema": V5_SCHEMA, "action_contract": ACTION_CONTRACT, "action_names": list(ACTION_NAMES),
                "observation_timing": "synchronized-pre-action-v1", "fps": 10,
                "physics_state_spec": "mjSTATE_INTEGRATION", "quaternion_order": "wxyz", "stages": list(STAGES),
                "joint_names": ["arm"], "body_names": ["can"], "actuator_names": ["arm"],
                "cameras": {"torso": {"K": np.eye(3).tolist(), "D": [0]*5, "size": [256, 256],
                                      "preprocessing": "synthetic-test-only"}},
                "provenance": {"controller": {"mode": "reachy-control", "revision": "test"},
                               "manifest_sha256": "a"*64, "simulation": {"timestep": .002}}}
    if mutate:
        mutate(arrays, metadata)
    with h5py.File(path, "w") as handle:
        handle.attrs.update(schema=V5_SCHEMA, action_contract=ACTION_CONTRACT, success=True,
                            episode_id="original", metadata_json=json.dumps(metadata))
        for key, value in arrays.items():
            if value.dtype.kind in "USO":
                handle.create_dataset(key, data=value.astype(object), dtype=h5py.string_dtype())
            else:
                handle[key] = value
    return path


def test_normalized_export_preserves_missing_observations_and_never_invents_v5(tmp_path):
    record = normalized_record()
    path = export_normalized(record, tmp_path)
    assert path == tmp_path / "datasets/example/episodes/one.hdf5"
    report = inspect_archive(path)
    assert report["schema"] == SCHEMA and report["policy_ready"] is False
    assert {"action", "controller_state_json", "observation/cameras/torso/rgb"} <= set(report["missing_native_v5_fields"])
    assert "observation/cameras/torso/rgb" not in report["missing_fields"]
    assert report["rgb_required"] is False
    with h5py.File(path, "r") as handle:
        assert "action" not in handle and "observation/right_tcp_pose" not in handle
        np.testing.assert_equal(handle["observation/objects/can/pose"][:], record["arrays"]["objects/can/pose"])
        np.testing.assert_equal(handle["source/time_s"][:], handle["timestamp"][:])
        assert np.isnan(handle["observation/objects/can/pose"][1]).all()
        meta = json.loads(handle.attrs["metadata_json"])
        assert meta["source_group"] == "recording"
        assert meta["derived_fields"]["observation/objects/can/pose"]["operation"] == "identity"
    assert np.isnan(record["arrays"]["objects/can/pose"][1]).all()
    assert assess_v5(path)["structurally_compatible"] is False


def test_schema_identical_across_sources_and_source_groups_are_preserved(tmp_path):
    a = export_normalized(normalized_record(), tmp_path, "human", "crop1")
    b = export_normalized(normalized_record(), tmp_path, "robot", "crop2")
    with h5py.File(a) as first, h5py.File(b) as second:
        assert first.attrs["schema"] == second.attrs["schema"] == SCHEMA
        assert json.loads(first.attrs["metadata_json"])["source_group"] == json.loads(second.attrs["metadata_json"])["source_group"]
    with pytest.raises(FileExistsError):
        export_normalized(normalized_record(), tmp_path, "human", "crop1")


def test_untimed_waypoints_keep_same_schema_without_fabricated_timestamps(tmp_path):
    record = normalized_record()
    del record["arrays"]["time_s"]
    record["arrays"]["source/waypoint_index"] = np.arange(3)
    record["missing_fields"].append("timestamp")
    record["metadata"]["derived_fields"] = {"waypoints": {"status": "planned_not_executed"}}
    path = export_normalized(record, tmp_path)
    report = inspect_archive(path)
    assert report["schema"] == SCHEMA and report["timed"] is False and report["rows"] == 3
    with h5py.File(path) as handle:
        assert "timestamp" not in handle
        np.testing.assert_array_equal(handle["source/waypoint_index"][:], np.arange(3))
        metadata = json.loads(handle.attrs["metadata_json"])
        assert metadata["derived_fields"]["waypoints"]["status"] == "planned_not_executed"
        assert metadata["index_channel"] == "source/waypoint_index"
    assert not assess_v5(path)["structurally_compatible"]


def test_untimed_export_requires_explicit_missing_timing_and_monotonic_indices(tmp_path):
    record = normalized_record()
    del record["arrays"]["time_s"]
    record["arrays"]["source/frame_index"] = np.arange(3)
    with pytest.raises(ValueError, match="explicit missing timestamp"):
        export_normalized(record, tmp_path)
    record["missing_fields"].append("timestamp")
    record["arrays"]["source/frame_index"] = np.array([0, 0, 1])
    with pytest.raises(ValueError, match="increasing"):
        export_normalized(record, tmp_path)


def test_nonfinite_observations_require_an_honest_mask(tmp_path):
    record = normalized_record()
    record["arrays"]["objects/can/valid"][1] = True
    with pytest.raises(ValueError, match="nonfinite samples valid"):
        export_normalized(record, tmp_path)
    del record["arrays"]["objects/can/valid"]
    with pytest.raises(ValueError, match="explicit validity mask"):
        export_normalized(record, tmp_path)


def test_declared_marker_masks_survive_source_and_observation_aliases(tmp_path):
    record = normalized_record()
    pose = record["arrays"].pop("objects/can/pose")
    mask = record["arrays"].pop("objects/can/valid")
    record["arrays"].update({"objects/can/marker_frame_pose_wxyz": pose,
                             "objects/can/marker_frame_valid": mask,
                             "objects/can/marker_fit_rmse_m": np.array([0., np.nan, .001]),
                             "markers/position_m": np.zeros((3, 2, 3)),
                             "markers/valid": np.ones((3, 2), dtype=bool)})
    record["arrays"]["markers/position_m"][1, 0] = np.nan
    record["arrays"]["markers/valid"][1, 0] = False
    masks = {"objects/can/marker_frame_pose_wxyz": "objects/can/marker_frame_valid",
             "objects/can/marker_fit_rmse_m": "objects/can/marker_frame_valid",
             "markers/position_m": "markers/valid"}
    record["metadata"]["validity_masks"] = masks
    path = export_normalized(record, tmp_path)
    with h5py.File(path) as handle:
        metadata = json.loads(handle.attrs["metadata_json"])
        for key, mask_name in masks.items():
            assert metadata["validity_masks"]["source/" + key] == "source/" + mask_name
        assert metadata["validity_masks"]["observation/objects/can/marker_frame_pose_wxyz"] == "observation/objects/can/marker_frame_valid"
        assert metadata["validity_masks"]["observation/objects/can/marker_fit_rmse_m"] == "observation/objects/can/marker_frame_valid"
        np.testing.assert_array_equal(handle["source/markers/valid"][:], record["arrays"]["markers/valid"])
    record["arrays"]["markers/valid"][1, 0] = True
    with pytest.raises(ValueError, match="nonfinite samples valid"):
        export_normalized(record, tmp_path, episode_id="bad-mask")


def test_hash_bound_sidecar_detects_changed_data(tmp_path):
    path = export_normalized(normalized_record(), tmp_path)
    with h5py.File(path, "a") as handle:
        handle["source/source/action"][0, 0] = 42
    with pytest.raises(ValueError, match="checksum"):
        inspect_archive(path)


@pytest.mark.parametrize("problem", ["robot_only", "backward_time", "path", "absent_group"])
def test_invalid_archives_are_rejected_without_publication(tmp_path, problem):
    record = normalized_record()
    if problem == "robot_only":
        record["metadata"]["objects"] = {}
    if problem == "backward_time":
        record["arrays"]["time_s"] = np.array([0., 1., .5])
    if problem == "path":
        record["metadata"]["episode_id"] = "../escape"
    if problem == "absent_group":
        record["metadata"]["source_group"] = ""
    with pytest.raises(ValueError):
        export_normalized(record, tmp_path)
    assert not list(tmp_path.rglob("*.hdf5"))


def test_complete_structure_does_not_assert_dynamics_or_trainability(tmp_path):
    path = native_fixture(tmp_path / "recorded.hdf5")
    report = assess_v5(path)
    assert report["structurally_compatible"] and not report["missing_fields"] and not report["errors"]
    assert report["physics_validated"] is False and report["policy_ready"] is False
    result = import_v5(path, tmp_path / "export", "format_fixture")
    assert result["native_episode"] is None
    with h5py.File(result["archive"]) as handle:
        assert handle.attrs["schema"] == SCHEMA
        assert json.loads(handle.attrs["metadata_json"])["independent_new_demonstration"] is False
        np.testing.assert_array_equal(handle["action"][:], h5py.File(path)["action"][:])


@pytest.mark.parametrize("mutation, message", [
    (lambda a, m: a.pop("controller_state_json"), "controller_state_json"),
    (lambda a, m: a["action"].__setitem__((0, 25), .3), "binary"),
    (lambda a, m: a["action"].__setitem__((0, slice(3, 9)), 0), "rotation6d"),
    (lambda a, m: a["image_tick"].__setitem__(0, 1), "synchronized"),
    (lambda a, m: a["timestamp"].__setitem__(1, .15), "10 Hz"),
    (lambda a, m: a.__setitem__("controller_state_json", np.array(["{}"]*3)), "snapshot"),
    (lambda a, m: m["provenance"]["controller"].__setitem__("mode", "joint-pd"), "reachy-control"),
    (lambda a, m: a.__setitem__("applied_controls", np.zeros((3, 1))), "shape"),
])
def test_incompatible_v5_is_not_imported(tmp_path, mutation, message):
    path = native_fixture(tmp_path / "bad.hdf5", mutation)
    assessment = assess_v5(path)
    assert not assessment["structurally_compatible"]
    assert message in str(assessment)
    with pytest.raises(ValueError, match="Incomplete/incompatible"):
        import_v5(path, tmp_path, "bad")


def test_native_publication_requires_exact_external_evidence(tmp_path):
    source = native_fixture(tmp_path / "recorded.hdf5")
    digest = sha256(source)
    evidence = tmp_path / "validation.json"
    payload = {"validator": "simulation.validate", "status": "passed", "episode_sha256": "b"*64,
               "checks": [{"name": "synthetic bridge test only", "ok": True}]}
    evidence.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash-bound"):
        import_v5(source, tmp_path / "bad_export", "fixture", validation_evidence=evidence)
    assert not (tmp_path / "bad_export").exists()
    payload["episode_sha256"] = digest
    evidence.write_text(json.dumps(payload))
    result = import_v5(source, tmp_path / "export", "fixture", validation_evidence=evidence)
    assert sha256(result["native_episode"]) == sha256(source) == digest
    assert result["assessment"]["physics_validated"] is False
    assert result["native_episode"].endswith("policy-v5/episodes/000000.hdf5")
    with pytest.raises(ValueError, match="already published"):
        import_v5(source, tmp_path / "export", "fixture", "duplicate", validation_evidence=evidence)
