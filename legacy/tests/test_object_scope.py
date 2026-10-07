"""Scope enforcement must preserve unrelated data and prevent reacquisition."""

import hashlib
import json
import os
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from reachy_retarget import acquire, bulk
from reachy_retarget.episodes import write_episode
from reachy_retarget.scope import exclusion_reason, object_tracks, remove_source, require_objects
from reachy_retarget.store import Store, sha256


EXCLUDED_SOURCE = "hf__glannuzel__reachy2_pick_place"


def _arrays(with_object=False):
    pose = np.tile([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0], (4, 1))
    arrays = {"time_s": np.arange(4) * 0.1, "hand/right_pose": pose.copy()}
    if with_object:
        arrays["objects/can/pose"] = pose.copy()
        arrays["objects/can/pose"][:, 0] = np.arange(4) * 0.01
    return arrays


def _source(store, source, *, with_object=False, sequence="demo_0"):
    url = f"https://example.invalid/datasets/{source}/revision123/state.npz"
    payload = b"immutable source selection: " + source.encode()
    raw = store.root / "data/raw" / source / "state.npz"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    store.source(source, "robot", url, "downloaded", revision="revision123")
    store.file(source, "state.npz", url, len(payload), digest)
    store.update_file(source, "state.npz", status="downloaded", bytes=len(payload), sha256=digest)
    normalized = write_episode(
        store,
        source,
        sequence,
        _arrays(with_object),
        {"objects": {"can": {"mesh": "can.obj"}} if with_object else {}},
    )
    derived = store.root / "data/retargeted" / source / "motion.h5"
    derived.parent.mkdir(parents=True, exist_ok=True)
    derived.write_bytes(b"derived test artifact")
    return raw, normalized, derived


def _blob(store, payload):
    digest = hashlib.sha256(payload).hexdigest()
    path = store.root / "data/blobs" / digest[:2] / digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_object_track_validation_preserves_read_only_source_arrays():
    arrays = _arrays(True)
    poses = arrays["objects/can/pose"]
    poses[1, 0] = np.nan
    poses[2, 3:] = 0.0
    mask = np.ones(4, dtype=bool)
    mask.setflags(write=False)
    poses.setflags(write=False)
    arrays["objects/can/valid"] = mask
    before = poses.copy()

    assert object_tracks(arrays) == {"can": 2}
    assert require_objects(arrays) == {"can": 2}
    np.testing.assert_array_equal(mask, np.ones(4, dtype=bool))
    np.testing.assert_array_equal(poses, before)


@pytest.mark.parametrize(
    "objects",
    [
        {"objects": {"can": {"mesh": "can.obj", "pose_frame": "world"}}},
        {"objects/can/name": np.array(["can"] * 4)},
        {"objects/can/pose": np.ones((4, 3))},
        {"objects/can/pose": np.ones((3, 7))},
        {"objects/can/pose": np.zeros((4, 7))},
        {"objects/can/pose": np.full((4, 7), np.nan)},
        {"objects/can/pose": _arrays(True)["objects/can/pose"], "objects/can/valid": np.zeros(4, bool)},
        {"objects/can/pose": _arrays(True)["objects/can/pose"], "objects/can/valid": np.array([True, False, False, False])},
        {"objects/can/pose": _arrays(True)["objects/can/pose"], "objects/can/valid": np.ones((4, 1), bool)},
    ],
    ids=["metadata-only", "name-only", "xyz-only", "wrong-time-length", "zero-quaternion", "nan-pose", "all-invalid", "one-valid-frame", "wrong-mask-shape"],
)
def test_object_names_metadata_and_unusable_tracks_are_insufficient(objects):
    arrays = _arrays()
    arrays.update(objects)
    with pytest.raises(ValueError, match="object pose track"):
        require_objects(arrays)


def test_explicit_source_removal_records_provenance_and_preserves_other_data(tmp_path):
    store = Store(tmp_path / "store")
    raw, normalized, derived = _source(store, EXCLUDED_SOURCE)
    other = _source(store, "object-aware", with_object=True)
    sibling = tmp_path / "sibling-repository" / "original.npz"
    sibling.parent.mkdir()
    os.link(raw, sibling)
    unrelated = store.root / "notes.txt"
    unrelated.write_text("unrelated work")
    old_hashes = {str(p.relative_to(store.root)): sha256(p) for p in (raw, normalized, normalized.with_suffix(".json"), derived)}
    protected_hashes = {p: sha256(p) for p in (*other, sibling, unrelated)}

    result = remove_source(store, EXCLUDED_SOURCE, include_raw=True)

    assert result["source"] == EXCLUDED_SOURCE
    assert result["files_removed"] == len(old_hashes)
    assert all(not p.exists() for p in (raw, normalized, derived))
    assert {p: sha256(p) for p in protected_hashes} == protected_hashes
    manifest = json.loads(open(result["manifest"]).read())
    assert manifest["completed"]
    assert manifest["source_id"] == EXCLUDED_SOURCE
    assert {row["path"]: row["sha256"] for row in manifest["files"]} == old_hashes
    file_row = manifest["ledger_files"][0]
    assert file_row["url"].endswith("/revision123/state.npz")
    assert file_row["expected_sha256"] == old_hashes[str(raw.relative_to(store.root))]
    assert json.loads(manifest["ledger_sources"][0]["details"])["revision"] == "revision123"
    assert not store.rows("episodes", "source_id=?", (EXCLUDED_SOURCE,))
    assert store.rows("episodes", "source_id=?", ("object-aware",))
    assert store.rows("files", "source_id=?", (EXCLUDED_SOURCE,))[0]["status"] == "excluded_objectless"


def test_raw_source_is_preserved_without_explicit_include_raw(tmp_path):
    store = Store(tmp_path)
    raw, normalized, derived = _source(store, "inspected-objectless")
    digest = sha256(raw)

    result = remove_source(store, "inspected-objectless")

    assert sha256(raw) == digest
    assert not normalized.exists() and not derived.exists()
    manifest = json.loads(open(result["manifest"]).read())
    assert not manifest["include_raw"]
    assert all(not row["path"].startswith("data/raw/") for row in manifest["files"])


def test_raw_removal_prunes_only_owned_orphan_blobs_and_records_checksums(tmp_path):
    store = Store(tmp_path)
    source = "owned-orphan-blobs"
    raw, _, _ = _source(store, source)
    blob = _blob(store, raw.read_bytes())
    raw.unlink()
    os.link(blob, raw)
    unrelated = _blob(store, b"unrelated cached payload")
    expected_only = _blob(store, b"source did not acquire this expected payload")
    store.file(source, "never-downloaded.npz", "https://example.invalid/never-downloaded", expected_sha256=expected_only.name)

    result = remove_source(store, source, include_raw=True)

    assert not raw.exists() and not blob.exists()
    assert unrelated.exists() and expected_only.exists()
    assert result["blobs_removed"] == 1
    manifest = json.loads(open(result["manifest"]).read())
    removed_blob = [entry for entry in manifest["files"] if entry.get("kind") == "raw_blob"]
    assert len(removed_blob) == 1
    assert removed_blob[0]["path"] == str(blob.relative_to(tmp_path))
    assert removed_blob[0]["sha256"] == blob.name
    assert removed_blob[0]["bytes"] > 0
    assert manifest["blob_cleanup"]["candidate_hashes"] == [blob.name]
    assert not manifest["blob_cleanup"]["preserved"]
    assert any(entry["url"].endswith("/revision123/state.npz") and entry["sha256"] == blob.name
               for entry in manifest["ledger_files"])


def test_removing_only_derived_data_preserves_its_raw_blob(tmp_path):
    store = Store(tmp_path)
    source = "raw-blob-removal-not-authorized"
    raw, _, _ = _source(store, source)
    blob = _blob(store, raw.read_bytes())

    result = remove_source(store, source)

    assert raw.exists() and blob.exists()
    assert result["blobs_removed"] == 0
    manifest = json.loads(open(result["manifest"]).read())
    assert not manifest["blob_cleanup"]["requested"]
    assert not any(entry.get("kind") == "raw_blob" for entry in manifest["files"])


def test_rerun_cleans_legacy_orphan_from_excluded_source_ledger(tmp_path):
    store = Store(tmp_path)
    source = "previously-removed-objectless-selection"
    raw, _, _ = _source(store, source)
    payload = raw.read_bytes()
    remove_source(store, source, include_raw=True)
    # Recreate only the orphan to model the old removal behavior, which left
    # content-addressed blobs behind after removing raw and normalized files.
    blob = _blob(store, payload)
    assert not store.rows("episodes", "source_id=?", (source,))
    assert not raw.parent.exists()

    result = remove_source(Store(tmp_path), source, include_raw=True)

    assert not blob.exists()
    assert result["files_removed"] == result["blobs_removed"] == 1
    manifest = json.loads(open(result["manifest"]).read())
    assert manifest["files"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["ledger_files"][0]["status"] == "excluded_objectless"
    assert manifest["completed"]


@pytest.mark.parametrize("reference_field", ["sha256", "expected_sha256"])
def test_blob_referenced_by_other_source_ledger_is_retained(tmp_path, reference_field):
    store = Store(tmp_path)
    source = "removed-blob-owner"
    raw, _, _ = _source(store, source)
    blob = _blob(store, raw.read_bytes())
    store.source("shared-source", "human_object", "https://example.invalid/shared")
    expected = blob.name if reference_field == "expected_sha256" else None
    store.file("shared-source", "shared.npz", "https://example.invalid/shared.npz", expected_sha256=expected)
    if reference_field == "sha256":
        store.update_file("shared-source", "shared.npz", sha256=blob.name)

    result = remove_source(store, source, include_raw=True)

    assert not raw.exists()
    assert sha256(blob) == blob.name
    assert result["blobs_removed"] == 0
    manifest = json.loads(open(result["manifest"]).read())
    preserved = manifest["blob_cleanup"]["preserved"]
    assert len(preserved) == 1
    assert preserved[0]["sha256"] == blob.name
    assert preserved[0]["references"] == [{"source_id": "shared-source", "path": "shared.npz"}]


@pytest.mark.parametrize("reference_kind", ["hardlink", "copy"])
def test_blob_referenced_by_unregistered_raw_path_is_retained(tmp_path, reference_kind):
    store = Store(tmp_path)
    source = "removed-blob-owner"
    raw, _, _ = _source(store, source)
    blob = _blob(store, raw.read_bytes())
    other = tmp_path / "data/raw/unregistered-source/state.npz"
    other.parent.mkdir(parents=True)
    if reference_kind == "hardlink":
        os.link(blob, other)
    else:
        other.write_bytes(blob.read_bytes())

    result = remove_source(store, source, include_raw=True)

    assert not raw.exists()
    assert sha256(blob) == sha256(other) == blob.name
    assert result["blobs_removed"] == 0
    manifest = json.loads(open(result["manifest"]).read())
    assert manifest["blob_cleanup"]["preserved"][0]["references"] == [{"path": str(other.relative_to(tmp_path))}]


def test_linked_raw_inventory_conservatively_preserves_candidate_blobs(tmp_path):
    store = Store(tmp_path / "store")
    source = "removed-blob-owner"
    raw, _, _ = _source(store, source)
    blob = _blob(store, raw.read_bytes())
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    original = sibling / "state.npz"
    original.write_bytes(blob.read_bytes())
    (store.root / "data/raw/linked-source").symlink_to(sibling, target_is_directory=True)

    result = remove_source(store, source, include_raw=True)

    assert sha256(blob) == sha256(original) == blob.name
    assert result["blobs_removed"] == 0
    manifest = json.loads(open(result["manifest"]).read())
    assert "Linked raw path" in manifest["blob_cleanup"]["preserved"][0]["reason"]


@pytest.mark.parametrize("linked_component", ["blob", "shard", "blob-root"])
def test_linked_blob_paths_abort_before_any_source_deletion(tmp_path, linked_component):
    store = Store(tmp_path / "store")
    raw, normalized, derived = _source(store, EXCLUDED_SOURCE)
    payload = raw.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    blob_root = store.root / "data/blobs"
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    if linked_component == "blob-root":
        original = sibling / digest[:2] / digest
        original.parent.mkdir()
        blob_root.symlink_to(sibling, target_is_directory=True)
    elif linked_component == "shard":
        original = sibling / digest
        blob_root.mkdir()
        (blob_root / digest[:2]).symlink_to(sibling, target_is_directory=True)
    else:
        original = sibling / digest
        (blob_root / digest[:2]).mkdir(parents=True)
        (blob_root / digest[:2] / digest).symlink_to(original)
    original.write_bytes(payload)

    with pytest.raises(ValueError, match="linked raw blob"):
        remove_source(store, EXCLUDED_SOURCE, include_raw=True)

    assert raw.exists() and normalized.exists() and derived.exists()
    assert original.read_bytes() == payload
    assert not list((store.root / "runs/removals").glob("*.json"))


def test_corrupt_content_address_aborts_before_deleting_source(tmp_path):
    store = Store(tmp_path)
    raw, normalized, derived = _source(store, EXCLUDED_SOURCE)
    blob = _blob(store, raw.read_bytes())
    blob.write_bytes(b"corrupt content under a valid hash name")

    with pytest.raises(ValueError, match="checksum"):
        remove_source(store, EXCLUDED_SOURCE, include_raw=True)

    assert raw.exists() and normalized.exists() and derived.exists() and blob.exists()


def test_robot_category_without_inspected_episodes_cannot_authorize_deletion(tmp_path):
    store = Store(tmp_path)
    source = "uninspected-robot"
    store.source(source, "robot", "https://example.invalid", "catalogued")
    raw = tmp_path / "data/raw" / source / "state.npz"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"unknown content must survive")

    with pytest.raises(ValueError, match="inspected|evidence"):
        remove_source(store, source, include_raw=True)

    assert raw.read_bytes() == b"unknown content must survive"


@pytest.mark.parametrize("source", ["../sibling", "/absolute", "nested/source", "..", ".", "nested\\source"])
def test_removal_rejects_nonlocal_source_names(tmp_path, source):
    with pytest.raises(ValueError, match="safe path component"):
        remove_source(Store(tmp_path), source, include_raw=True)


def test_mixed_source_cannot_be_removed_even_if_source_is_hardcoded_excluded(tmp_path):
    store = Store(tmp_path)
    raw, objectless, derived = _source(store, EXCLUDED_SOURCE)
    object_aware = write_episode(store, EXCLUDED_SOURCE, "object_demo", _arrays(True), {"objects": {"can": {}}})
    hashes = {p: sha256(p) for p in (raw, objectless, object_aware, derived)}

    with pytest.raises(ValueError, match="object state|object.*track"):
        remove_source(store, EXCLUDED_SOURCE, include_raw=True)

    assert {p: sha256(p) for p in hashes} == hashes
    assert len(store.rows("episodes", "source_id=?", (EXCLUDED_SOURCE,))) == 2
    assert not list((tmp_path / "runs/removals").glob("*.json"))


def test_unregistered_object_episode_prevents_whole_source_removal(tmp_path):
    store = Store(tmp_path)
    source = "partially-inventoried"
    raw, _, derived = _source(store, source)
    object_aware = write_episode(store, source, "unregistered", _arrays(True), {"objects": {"can": {}}})
    with store.connect() as connection:
        connection.execute("DELETE FROM episodes WHERE source_sequence='unregistered'")

    with pytest.raises(ValueError):
        remove_source(store, source, include_raw=True)

    assert raw.exists() and derived.exists() and object_aware.exists()


def test_missing_inspection_payload_does_not_authorize_raw_removal(tmp_path):
    store = Store(tmp_path)
    source = "missing-normalized-payload"
    raw, normalized, derived = _source(store, source)
    normalized.unlink()

    with pytest.raises(ValueError):
        remove_source(store, source, include_raw=True)

    assert raw.exists() and derived.exists()


def test_nested_symlink_aborts_removal_without_touching_sibling(tmp_path):
    store = Store(tmp_path / "store")
    raw, normalized, derived = _source(store, EXCLUDED_SOURCE)
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    original = sibling / "source.bin"
    original.write_bytes(b"preserve sibling source")
    (raw.parent / "linked-source").symlink_to(sibling, target_is_directory=True)

    with pytest.raises(ValueError, match="linked|outside"):
        remove_source(store, EXCLUDED_SOURCE, include_raw=True)

    assert original.read_bytes() == b"preserve sibling source"
    assert raw.exists() and normalized.exists() and derived.exists()


@pytest.mark.parametrize("transport", ["http", "xet"])
@pytest.mark.parametrize("source", [EXCLUDED_SOURCE, "explicitly-removed-selection"])
def test_removed_sources_cannot_be_reacquired_even_with_stale_queue_item(tmp_path, monkeypatch, transport, source):
    store = Store(tmp_path)
    raw, _, _ = _source(store, source)
    stale_item = dict(store.rows("files")[0], status="queued")
    remove_source(store, source, include_raw=True)
    calls = []

    def forbidden_network(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Excluded source reached a network transport")

    monkeypatch.setattr(acquire.requests, "get", forbidden_network)
    monkeypatch.setattr(acquire.requests, "head", forbidden_network)
    monkeypatch.setitem(sys.modules, "hf_xet", SimpleNamespace(XetSession=forbidden_network, XetFileInfo=object))
    if transport == "http":
        result = acquire.download(store, stale_item, attempts=1, reserve=0)
        assert result == "excluded_objectless"
    else:
        assert bulk.fetch_xet(store, [stale_item]) == {"excluded_objectless": 1}

    assert not calls
    assert not raw.exists()
    assert store.rows("files")[0]["status"] == "excluded_objectless"


@pytest.mark.parametrize("transport", ["http", "xet"])
def test_removal_exclusion_survives_rediscovery_and_store_reopen(tmp_path, monkeypatch, transport):
    store = Store(tmp_path)
    source = "inspected-selection-rediscovered-later"
    _source(store, source)
    removal = remove_source(store, source, include_raw=True)

    # Discovery may refresh source status and discover a file absent from the
    # old ledger. Neither operation may revoke the user's persistent exclusion.
    url = f"https://example.invalid/datasets/{source}/revision456/new.npz"
    store.source(source, "robot", url, "catalogued", revision="revision456")
    store.file(source, "new.npz", url, size=4)
    reopened = Store(tmp_path)
    assert reopened.rows("sources", "id=?", (source,))[0]["status"] == "catalogued"
    assert exclusion_reason(reopened, source)
    tombstone = json.loads((tmp_path / "catalog/objectless-exclusions.json").read_text())[source]
    assert tombstone["manifest"] == removal["manifest"]
    queued = reopened.rows("files", "source_id=? AND path=?", (source, "new.npz"))[0]
    assert queued["status"] == "queued"
    calls = []

    def forbidden_network(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Rediscovery revoked a persistent source exclusion")

    monkeypatch.setattr(acquire.requests, "get", forbidden_network)
    monkeypatch.setattr(acquire.requests, "head", forbidden_network)
    monkeypatch.setitem(sys.modules, "hf_xet", SimpleNamespace(XetSession=forbidden_network, XetFileInfo=object))
    if transport == "http":
        assert acquire.download(reopened, queued, attempts=1, reserve=0) == "excluded_objectless"
    else:
        assert bulk.fetch_xet(reopened, [queued]) == {"excluded_objectless": 1}

    assert not calls
    assert not (tmp_path / "data/raw" / source).exists()
    assert reopened.rows("files", "source_id=? AND path=?", (source, "new.npz"))[0]["status"] == "excluded_objectless"


def test_retarget_rejects_metadata_only_objects_before_controller_creation(tmp_path, monkeypatch):
    from reachy_retarget import retarget

    store = Store(tmp_path)
    source = "metadata-only-objects"
    store.source(source, "robot", "https://example.invalid", "downloaded")
    write_episode(store, source, "demo", _arrays(), {"objects": {"can": {"mesh": "can.obj"}}})
    calls = []

    def forbidden_controller(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Objectless episode reached controller initialization")

    monkeypatch.setattr(retarget, "Robot", forbidden_controller)
    with pytest.raises(ValueError, match="object pose track"):
        retarget.one(store, store.rows("episodes")[0])
    assert not calls
    assert not (tmp_path / "data/retargeted" / source).exists()


@pytest.mark.parametrize("invalid_kind", ["mask", "zero-quaternion", "nonunit-quaternion", "nan"])
def test_retarget_preserves_gaps_in_object_observations(tmp_path, monkeypatch, invalid_kind):
    """Unknown object states must not become usable interpolation targets."""
    from reachy_retarget import retarget

    store = Store(tmp_path)
    source = "partially-observed-object"
    arrays = _arrays(True)
    arrays["robot/joint_names"] = np.array(["test_joint"])
    arrays["robot/joint_position"] = np.zeros((4, 1))
    poses = arrays["objects/can/pose"]
    poses[1, 0] = 10.0  # A finite but unavailable estimate must not be retained.
    if invalid_kind == "mask":
        arrays["objects/can/valid"] = np.array([True, False, True, True])
    elif invalid_kind == "zero-quaternion":
        poses[1, 3:] = 0.0
    elif invalid_kind == "nonunit-quaternion":
        poses[1, 3:] *= 2.0
    else:
        poses[1] = np.nan
    store.source(source, "robot", "https://example.invalid", "downloaded")
    normalized = write_episode(store, source, "demo", arrays, {"objects": {"can": {}}})
    original_digest = sha256(normalized)

    # Hold a valid native robot configuration fixed so this test isolates object
    # resampling rather than controller installation, IK, or hardware access.
    model = SimpleNamespace(lowerPositionLimit=np.full(20, -2.0), upperPositionLimit=np.full(20, 2.0))
    controller = SimpleNamespace(
        q_indices={"test_joint": 0}, _joint_q_indices=np.arange(20),
        model=model, caps=np.ones(20), geometry=lambda q: (1.0, 1.0, {}),
    )
    robot = SimpleNamespace(
        q=np.zeros(20), arm_ids=list(range(14)), neck_ids=[14, 15, 16],
        r=controller, pin=SimpleNamespace(difference=lambda model, x, y: y - x),
        fk=lambda q: np.tile(np.eye(4), (2, 1, 1)), base=lambda q: np.zeros(3),
        pack=lambda arm, base: np.zeros(20), identity={}, backend="test-fixture",
    )
    monkeypatch.setattr(retarget, "Robot", lambda root: robot)
    monkeypatch.setattr(retarget, "export_tracking", lambda *args: None)
    row = store.rows("episodes")[0]

    report = retarget.one(store, row)

    assert report["status"] == "kinematic_pass"
    assert sha256(normalized) == original_digest
    with h5py.File(tmp_path / "data/retargeted" / source / row["id"] / "motion.h5") as output:
        times = output["time_s"][:]
        gap_index = np.flatnonzero(np.isclose(times, 0.15))[0]
        known_index = np.flatnonzero(np.isclose(times, 0.25))[0]
        valid = output["objects/can/valid"][:]
        exported = output["objects/can/pose"][:]
    assert not valid[gap_index]
    assert np.isnan(exported[gap_index]).all()
    assert valid[known_index]
    np.testing.assert_allclose(exported[known_index], [0.025, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0])
