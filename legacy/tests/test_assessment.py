import json

import h5py
import numpy as np

from reachy_retarget.assessment import assess, digest
from reachy_retarget.episodes import write_episode
from reachy_retarget.store import Store


def episode(root, *, object_pose=True, masked=False, source="sample", sequence="demo_0"):
    store = Store(root)
    pose = np.tile([0.1, 0.2, 0.3, 1, 0, 0, 0], (3, 1))
    arrays = {"time_s": np.arange(3) / 20, "hand/right_pose": pose}
    if object_pose:
        arrays["objects/Cube/pose"] = pose
        if masked:
            arrays["objects/Cube/valid"] = np.zeros(3, dtype=bool)
    path = write_episode(store, source, sequence, arrays,
                         {"objects": {"Cube": {}}, "task": "PickCube"})
    return path


def test_historical_catalog_never_becomes_local_download(tmp_path):
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "datasets.json").write_text(json.dumps([
        {"id": "historical", "downloaded_files": 123, "status": "downloaded"}]))
    result = assess([tmp_path])
    assert result["summary"]["locally_downloaded_source_ids"] == 0
    assert result["roots"][0]["sources"][0]["historical_downloaded_files"] == 123
    assert not (catalog / "ledger.sqlite").exists()


def test_object_name_or_fully_invalid_track_is_not_object_evidence(tmp_path):
    episode(tmp_path, object_pose=False, sequence="robot_only")
    episode(tmp_path, masked=True, sequence="invalid_object")
    report = assess([tmp_path])
    assert report["summary"]["object_aware_source_groups"] == 0
    assert all(e["object_pipeline_status"] == "unsupported_no_valid_object_pose"
               for e in report["roots"][0]["episodes"])


def test_duplicate_roots_and_contact_variations_do_not_inflate_demos(tmp_path):
    path = episode(tmp_path / "a")
    episode(tmp_path / "b")
    eid = path.stem
    for attempt in ("first", "retry"):
        directory = tmp_path / "a/runs/contact" / attempt / "demo_0"
        directory.mkdir(parents=True)
        scene = directory / "scene.xml"
        scene.write_text("<mujoco/>")
        with h5py.File(directory / "replay.h5", "w") as f:
            f["time_s"] = np.arange(3)
            f["simulation/qpos"] = np.zeros((3, 7))
            f["simulation/Cube_pose"] = np.tile([0, 0, 0, 1, 0, 0, 0], (3, 1))
        (directory / "result.json").write_text(json.dumps({
            "source_id": "sample", "source_episode": eid, "source_sequence": "demo_0",
            "success": True, "status": "physical_pass", "scene_sha256": digest(scene),
            "object_state_assignment": "reset only; dynamic free joint for every physics step"}))
    report = assess([tmp_path / "a", tmp_path / "b"])
    assert report["summary"]["normalized_artifact_copies"] == 2
    assert report["summary"]["distinct_recorded_source_groups"] == 1
    assert report["summary"]["contact_attempts"] == 2
    assert report["summary"]["contact_validated_source_groups"] == 1


def test_claimed_contact_pass_without_replay_is_not_validation(tmp_path):
    path = episode(tmp_path)
    directory = tmp_path / "runs/contact/claim/demo_0"
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(json.dumps({
        "source_id": "sample", "source_episode": path.stem,
        "status": "physical_pass", "success": True}))
    report = assess([tmp_path])
    assert report["summary"]["contact_reported_passes"] == 1
    assert report["summary"]["contact_validated_source_groups"] == 0


def test_missing_ledger_payload_does_not_count_as_download(tmp_path):
    store = Store(tmp_path)
    store.source("missing", "human", "https://example.invalid")
    store.file("missing", "states.h5", "https://example.invalid/states.h5")
    store.update_file("missing", "states.h5", status="downloaded")
    report = assess([tmp_path])
    source = report["roots"][0]["sources"][0]
    assert source["downloaded_files"] == 0
    assert source["stale_downloaded_entries"] == 1
