import hashlib
import json
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from reachy_retarget.store import Store
from reachy_retarget.acquire import download
from reachy_retarget.episodes import (
    write_episode,
    pose_to_matrices,
    matrices_to_pose,
    interpolate_pose,
    read_episode,
)


@pytest.fixture
def http_source():
    payload = b"bounded resumable state data\x00" * 1000
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.headers.get("Range"))
            offset = (
                int(self.headers["Range"].split("=")[1].split("-")[0])
                if self.headers.get("Range")
                else 0
            )
            self.send_response(206 if offset else 200)
            if offset:
                self.send_header(
                    "Content-Range", f"bytes {offset}-{len(payload) - 1}/{len(payload)}"
                )
            self.send_header("Content-Length", str(len(payload) - offset))
            self.end_headers()
            self.wfile.write(payload[offset:])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield payload, requests, f"http://127.0.0.1:{server.server_port}/states.bin"
    server.shutdown()
    thread.join()


def test_resume_integrity_and_content_dedup(tmp_path, http_source):
    payload, requests, url = http_source
    s = Store(tmp_path)
    digest = hashlib.sha256(payload).hexdigest()
    s.source("a", "robot", url, "queued")
    s.file("a", "states.bin", url, len(payload), digest)
    p = tmp_path / "data/raw/a/states.bin"
    p.parent.mkdir(parents=True)
    p.with_name(p.name + ".part").write_bytes(payload[:517])
    assert download(s, s.rows("files")[0], attempts=1, reserve=0) == "downloaded"
    assert requests == ["bytes=517-"] and p.read_bytes() == payload
    s.source("b", "robot", url, "queued")
    s.file("b", "states.bin", url, len(payload), digest)
    assert (
        download(s, s.rows("files", "source_id=?", ("b",))[0], attempts=1, reserve=0)
        == "downloaded"
    )
    assert p.stat().st_ino == (tmp_path / "data/raw/b/states.bin").stat().st_ino


def test_corrupt_response_not_committed(tmp_path, http_source):
    payload, _, url = http_source
    s = Store(tmp_path)
    s.source("x", "robot", url, "queued")
    s.file("x", "bad.bin", url, len(payload), "0" * 64)
    assert download(s, s.rows("files")[0], attempts=1, reserve=0) == "failed"
    assert not (tmp_path / "data/raw/x/bad.bin").exists()
    assert list((tmp_path / "data/raw/x").glob("*.corrupt-*"))


def test_storage_reserve_preserves_queue(tmp_path, http_source):
    payload, requests, url = http_source
    s = Store(tmp_path)
    s.source("x", "robot", url, "queued")
    s.file("x", "x.bin", url, len(payload))
    assert download(s, s.rows("files")[0], attempts=1, reserve=10**30) == "storage_wait"
    assert not requests and s.rows("files")[0]["status"] == "storage_wait"


def test_shared_transform_preserves_grasp_and_quaternion_order():
    hand = np.eye(4)
    hand[:3, 3] = [0.4, -0.2, 0.9]
    hand[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.7, 0.6]).as_matrix()
    obj = np.eye(4)
    obj[:3, 3] = [0.5, -0.18, 0.85]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", 1.2).as_matrix()
    T[:3, 3] = [1, 2, 0]
    np.testing.assert_allclose(
        np.linalg.inv(T @ hand) @ (T @ obj), np.linalg.inv(hand) @ obj, atol=1e-12
    )
    np.testing.assert_allclose(
        pose_to_matrices(matrices_to_pose(hand)), hand, atol=1e-12
    )
    np.testing.assert_allclose(matrices_to_pose(np.eye(4)), [0, 0, 0, 1, 0, 0, 0])


def test_duplicate_split_and_missing_contact(tmp_path):
    s = Store(tmp_path)
    t = np.arange(10) * 0.01
    pose = np.tile([0, 0, 0, 1, 0, 0, 0], (10, 1))
    arrays = {"time_s": t, "hand/right_pose": pose}
    for source in ["original", "reformat"]:
        s.source(source, "robot", "https://example.invalid", "queued")
        write_episode(s, source, "same", arrays, {"objects": {}})
    rows = s.rows("episodes")
    assert rows[0]["split"] == rows[1]["split"]
    a, m = read_episode(tmp_path / rows[0]["path"])
    assert "contact" not in a and not m["physics_validated"]


def test_time_interpolation_uses_rotation_slerp():
    p = np.array([[0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1]], float)
    middle = interpolate_pose(np.array([0, 1]), p, np.array([0.5]))[0]
    np.testing.assert_allclose(
        middle, [0.5, 0, 0, np.sqrt(0.5), 0, 0, np.sqrt(0.5)], atol=1e-12
    )


@pytest.fixture
def range_server():
    files = {}
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.path, self.headers.get("Range")))
            if self.path.startswith("/resolve/"):
                self.send_response(302)
                self.send_header("Location", self.path.replace("/resolve/", "/cdn/"))
                self.end_headers()
                return
            payload = files[self.path]
            first, last = 0, len(payload) - 1
            if self.headers.get("Range"):
                first, end = self.headers["Range"].split("=")[1].split("-")
                first = int(first)
                last = min(int(end) if end else last, last)
                self.send_response(206)
                self.send_header(
                    "Content-Range", f"bytes {first}-{last}/{len(payload)}"
                )
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(last - first + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            try:
                self.wfile.write(payload[first : last + 1])
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield files, calls, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    t.join()


def test_remote_state_projection_resolves_once_and_omits_pixels(tmp_path, range_server):
    import h5py, io
    from reachy_retarget.extract import hdf5_state

    files, calls, base = range_server
    b = io.BytesIO()
    with h5py.File(b, "w") as f:
        f["data/demo_0/states"] = np.arange(120).reshape(20, 6)
        f["data/demo_0/obs/agentview_image"] = np.ones((20, 128, 128, 3), dtype="u1")
        f["data/demo_0/obs/eef_pos"] = np.ones((20, 3))
        f["data"].attrs["env_args"] = '{"frame":"world"}'
    files["/cdn/source.h5"] = b.getvalue()
    dest = tmp_path / "state.h5"
    size, digest = hdf5_state(base + "/resolve/source.h5", dest, reserve=0)
    with h5py.File(dest) as f:
        assert "data/demo_0/states" in f and "data/demo_0/obs/eef_pos" in f
        assert "agentview_image" not in f["data/demo_0/obs"]
        assert f["data"].attrs["env_args"] == '{"frame":"world"}'
    assert len([x for x in calls if x[0].startswith("/resolve/")]) == 1
    assert json.loads(dest.with_suffix(".h5.extraction.json").read_text())[
        "source_integrity"
    ].startswith("Pinned")
    assert digest == hashlib.sha256(dest.read_bytes()).hexdigest()


def test_selective_archive_rejects_traversal_and_keeps_object_geometry(
    tmp_path, range_server
):
    import io, zipfile
    from reachy_retarget.extract import archive_state

    files, _, base = range_server
    for name, unsafe in [("good", False), ("bad", True)]:
        b = io.BytesIO()
        with zipfile.ZipFile(b, "w") as z:
            z.writestr("sequence/objects.json", '{"pose":[0,0,0,1,0,0,0]}')
            z.writestr("objects/can.obj", "v 0 0 0\n")
            z.writestr("sequence/rgb/frame.jpg", b"pixels")
            if unsafe:
                z.writestr("../escape.pkl", b"bad")
        files[f"/cdn/{name}.zip"] = b.getvalue()
    archive_state(base + "/resolve/good.zip", tmp_path / "good.zip", reserve=0)
    with zipfile.ZipFile(tmp_path / "good.zip") as z:
        assert set(z.namelist()) == {"sequence/objects.json", "objects/can.obj"}
    with pytest.raises(ValueError, match="Unsafe"):
        archive_state(base + "/resolve/bad.zip", tmp_path / "bad.zip", reserve=0)
    assert not (tmp_path / "bad.zip").exists()


def test_cross_process_file_ownership_does_not_overwrite(tmp_path, http_source):
    from reachy_retarget.acquire import file_lock

    payload, calls, url = http_source
    s = Store(tmp_path)
    s.file("a", "state.bin", url, len(payload))
    row = s.rows("files")[0]
    lock = file_lock(s, row)
    try:
        assert download(s, row, attempts=1, reserve=0) == "in_progress"
        assert not calls
    finally:
        lock.close()
    assert download(s, row, attempts=1, reserve=0) == "downloaded"


def test_reservations_include_other_inflight_batches(tmp_path, monkeypatch):
    import shutil
    from reachy_retarget.disk import allocation, available

    Store(tmp_path)
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        "reachy_retarget.disk.shutil.disk_usage", lambda _: type(usage)(100, 0, 100)
    )
    with allocation(tmp_path, 30, reserve=50):
        assert available(tmp_path) == 70
        with pytest.raises(RuntimeError, match="reserve"):
            with allocation(tmp_path, 30, reserve=50):
                pass
    assert available(tmp_path) == 100


def test_partition_independent_sequence_dedup(tmp_path):
    import pyarrow as pa, pyarrow.parquet as pq
    from reachy_retarget.inventory import run
    from reachy_retarget.store import sha256

    s = Store(tmp_path)
    for source, chunks in [
        ("one", [np.arange(6)]),
        ("two", [np.arange(3), np.arange(3, 6)]),
    ]:
        s.source(source, "robot", "https://example.invalid", "queued")
        for i, t in enumerate(chunks):
            path = f"data/chunk-000/file-{i:03d}.parquet"
            p = tmp_path / "data/raw" / source / path
            p.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.table(
                    {
                        "episode_index": np.zeros(len(t), int),
                        "timestamp": t * 0.1,
                        "observation.state": [[float(x), 1.0, 2.0] for x in t],
                        "action": [[float(x)] for x in t],
                    }
                ),
                p,
            )
            s.file(source, path, "https://example.invalid/" + path, p.stat().st_size)
            s.update_file(
                source,
                path,
                status="downloaded",
                bytes=p.stat().st_size,
                sha256=sha256(p),
            )
    result = run(s)
    assert (
        result["unique_state_fingerprints"] == 1
        and result["exact_duplicate_sequences"] == 1
    )


def test_locked_manifest_recreates_selection_not_download_claim(tmp_path):
    from reachy_retarget.manifest import export_lock, restore_lock

    s = Store(tmp_path / "one")
    s.source("x", "robot", "https://example.invalid", "queued", revision="abc")
    s.file("x", "state.npz", "https://example.invalid/abc/state.npz", 8, "1" * 64)
    s.update_file("x", "state.npz", status="downloaded", bytes=8, sha256="1" * 64)
    lock = export_lock(s)
    dest = Store(tmp_path / "two")
    restore_lock(dest, lock)
    row = dest.rows("files")[0]
    assert row["status"] == "queued" and row["expected_sha256"] == "1" * 64
    assert json.loads(dest.rows("sources")[0]["details"])["revision"] == "abc"


def test_nested_npz_projection_omits_pixels_and_preserves_states(tmp_path):
    import io, zipfile
    from reachy_retarget.extract import archive_state

    frame = io.BytesIO()
    np.savez(
        frame,
        robot_obs=np.arange(15),
        scene_obs=np.arange(24),
        rgb_static=np.ones((4, 4, 3), dtype="u1"),
    )
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as z:
        z.writestr("episode_000000.npz", frame.getvalue())
    target = tmp_path / "states.zip"
    archive_state("https://example.invalid/source.zip", target, local=source, reserve=0)
    with (
        zipfile.ZipFile(target) as z,
        np.load(io.BytesIO(z.read("episode_000000.npz")), allow_pickle=False) as f,
    ):
        assert set(f.files) == {"robot_obs", "scene_obs"}
        np.testing.assert_equal(f["scene_obs"], np.arange(24))


def test_discovery_resumes_pinned_page_and_preserves_xet_hash(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import reachy_retarget.discovery as d

    store = Store(tmp_path)
    calls = []
    broken = [True]
    info = {"sha": "revision1", "tags": ["license:cc-by-4.0"]}

    def get(url):
        calls.append(url)
        if url.endswith("/api/datasets/owner/repo"):
            data = info
            links = {}
        elif url == "https://example.invalid/page2":
            if broken[0]:
                raise RuntimeError("interrupted listing")
            data = [
                {
                    "type": "file",
                    "path": "data/b.parquet",
                    "size": 7,
                    "lfs": {"oid": "b" * 64},
                    "xetHash": "second",
                }
            ]
            links = {}
        else:
            data = [
                {
                    "type": "file",
                    "path": "data/a.parquet",
                    "size": 7,
                    "lfs": {"oid": "a" * 64},
                    "xetHash": "first",
                }
            ]
            links = {"next": {"url": "https://example.invalid/page2"}}
        return SimpleNamespace(
            json=lambda: data, raise_for_status=lambda: None, links=links
        )

    monkeypatch.setattr(d, "get", get)
    d.hf_discover(store, "owner/repo")
    assert len(store.rows("files")) == 1
    broken[0] = False
    calls.clear()
    d.hf_discover(store, "owner/repo")
    assert calls == ["https://example.invalid/page2"]
    data = json.loads((tmp_path / "catalog/evidence/hf__owner__repo.json").read_text())
    assert data["discovery_complete"] and [
        x["xetHash"] for x in data["selected_files"]
    ] == ["first", "second"]


def test_gzip_tar_state_resume_skips_committed_members(tmp_path):
    import io, tarfile, zipfile
    from reachy_retarget.extract import archive_state

    state = io.BytesIO()
    np.save(state, np.arange(6))
    frame = io.BytesIO()
    np.savez(frame, pose_y=np.eye(4), seg=np.ones((32, 32), dtype="u1"))
    source = tmp_path / "source.tar.gz"
    target = tmp_path / "state.zip"
    with tarfile.open(source, "w:gz") as tar:
        for name, value in [
            ("poses/a.npy", state.getvalue()),
            ("rgb/image.png", b"pixel payload"),
            ("poses/b.npz", frame.getvalue()),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(value)
            tar.addfile(info, io.BytesIO(value))
    with zipfile.ZipFile(target.with_name(target.name + ".part"), "w") as z:
        z.writestr("poses/a.npy", state.getvalue())
    archive_state(
        "https://example.invalid/source.tar.gz",
        target,
        kind="tgz",
        local=source,
        reserve=0,
    )
    with zipfile.ZipFile(target) as z:
        assert z.namelist() == ["poses/a.npy", "poses/b.npz"]
        with np.load(io.BytesIO(z.read("poses/b.npz")), allow_pickle=False) as f:
            assert f.files == ["pose_y"]


def test_legacy_numpy_pickle_allows_bytes_but_rejects_code():
    import pickle, os
    from reachy_retarget.human import array_pickle

    x = {"pose": np.arange(12, dtype=np.float32).reshape(3, 4)}
    np.testing.assert_equal(
        array_pickle(pickle.dumps(x, protocol=2))["pose"], x["pose"]
    )

    class Forbidden:
        def __reduce__(self):
            return os.system, ("this-command-must-never-execute",)

    with pytest.raises(pickle.UnpicklingError):
        array_pickle(pickle.dumps(Forbidden()))
