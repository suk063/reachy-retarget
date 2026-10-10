import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest

from reachy_retarget.schema import (
    JOINTS, SIDES, ObjectTrack, PhysicsRollout, ReachyEpisode, Reference, Articulation,
    index_row, read_episode, recompute_modes, write_episode, write_index,
)
from reachy_retarget.schema import rotations as rot
from reachy_retarget.schema.io import check_not_image


def _smooth_poses(rng, T, scale=0.3):
    t = np.linspace(0, 1, T)[:, None]
    w = rng.normal(size=3) + np.sin(3 * t + rng.normal(size=3)) * scale * 2
    P = np.zeros((T, 4, 4))
    P[:, :3, :3] = rot.so3_exp(w)
    P[:, :3, 3] = rng.normal(size=3) * 0.3 + np.cos(2 * t + rng.normal(size=3)) * scale
    P[:, 3, 3] = 1
    return P


def make_episode(T=40, seed=0, **kw):
    rng = np.random.default_rng(seed)
    t = np.arange(T) / 50
    q = np.cumsum(rng.normal(scale=0.01, size=(T, 22)), axis=0)
    q[:, 2] = np.linspace(3.0, 3.4, T)          # crosses +pi to exercise yaw wrapping
    pose = np.c_[rng.normal(size=(T, 3)), rot.matrix_to_quat(rot.so3_exp(rng.normal(size=(T, 3))))]
    valid = np.ones(T, bool)
    valid[5:8] = False
    pose[5:8] = np.nan
    fields = dict(
        family="robomimic", dataset="robomimic/can/ph", episode_id="demo_3", task="pick can",
        time=t, q=q, qd=np.gradient(q, t, axis=0),
        tcp_base={s: _smooth_poses(rng, T) for s in SIDES}, head_base=_smooth_poses(rng, T, 0.1),
        gripper_opening=np.clip(rng.uniform(size=(T, 2)), 0, 1), gripper_width=rng.uniform(0, 0.08, (T, 2)),
        source_time=np.linspace(0, 1.2, T), retarget_config="sha256:abc",
        tier={"K": {"passed": False, "reasons": ["tcp residual 7 mm"]}, "P": None},
        instruction="pick up the can", license="MIT", provenance={"url": "https://example.org", "sha256": "00"},
        lineage={"seed": "robomimic/can/ph/demo_3", "generated": False}, variant_of=None,
        body_parts=("right_arm", "head", "base"),
        reference=Reference(tcp={"right": _smooth_poses(rng, T)}, head=_smooth_poses(rng, T), base=q[:, :3] + 0.01),
        objects={"can": ObjectTrack(pose, valid, "manipulated", {"kind": "cylinder", "radius": 0.03})},
        articulations={"drawer": Articulation(["slide"], rng.normal(size=(T, 1)))},
        validation={"tcp_pos_residual": rng.uniform(size=(T, 2)), "self_clearance": rng.uniform(size=T),
                    "grasp": rng.uniform(size=T) > 0.5},
        physics=PhysicsRollout(np.arange(3 * T) / 500, rng.normal(size=(3 * T, 4)), rng.normal(size=(3 * T, 3)),
                               rng.normal(size=(3 * T, 2)), ["a", "b", "c", "d"], ["a", "b", "c"], ["u0", "u1"],
                               info={"mujoco": "3.2", "timestep": 0.002}),
        extra={"assumptions": {"friction": 1.0}})
    fields.update(kw)
    return ReachyEpisode(**fields)


def _equal(a, b, path="ep"):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        assert np.asarray(a).shape == np.asarray(b).shape, path
        np.testing.assert_allclose(a, b, atol=1e-12, equal_nan=True, err_msg=path)
    elif isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _equal(a[k], b[k], f"{path}.{k}")
    elif hasattr(a, "__dataclass_fields__"):
        for k in a.__dataclass_fields__:
            _equal(getattr(a, k), getattr(b, k), f"{path}.{k}")
    else:
        assert a == b, path


def test_round_trip(tmp_path):
    ep = make_episode()
    path = write_episode(tmp_path / "ep.h5", ep)
    assert not list(tmp_path.glob(".*tmp*"))
    back = read_episode(path)
    _equal(ep, back)
    assert back.uid == "robomimic/can/ph/demo_3" and back.body_parts == ("right_arm", "head", "base")
    with h5py.File(path) as f:
        assert f.attrs["schema"] == "reachy-retarget-episode-v2"
        assert f["state/tcp/left/world"].shape == (40, 7)
        assert f["state/q"].compression == "gzip"
        np.testing.assert_allclose(rot.vec7_to_pose(f["state/head/world"][()]), ep.head_world, atol=1e-12)
        np.testing.assert_allclose(f["state/base_pose_world"][()], ep.q[:, :3])
        assert "reachy_agent_v8" in f["actions"]


def test_recompute_modes(tmp_path):
    path = write_episode(tmp_path / "ep.h5", make_episode())
    with h5py.File(path, "r+") as f:
        del f["actions/joint_vel"]
        f["actions/base_se2_delta/delta"][0] = 99
    recompute_modes(path)
    with h5py.File(path) as f:
        assert "joint_vel" in f["actions"]
        assert abs(f["actions/base_se2_delta/delta"][0, 0]) < 1


def test_tcp_world_composes_base():
    ep = make_episode()
    b = ep.q[5]
    expected = rot.planar_pose(b) @ ep.tcp_base["left"][5]
    np.testing.assert_allclose(ep.tcp_world["left"][5], expected)


def test_no_image_guard(tmp_path):
    for name in ("/obs/rgb", "/cameras/torso", "/depth", "/validation/image_mask"):
        with pytest.raises(ValueError):
            check_not_image(name, np.zeros(3))
    with pytest.raises(ValueError):
        check_not_image("/validation/x", np.zeros((4, 8, 8), np.uint8))
    check_not_image("/validation/x", np.zeros((4, 8), np.uint8))
    with pytest.raises(ValueError):
        write_episode(tmp_path / "a.h5", make_episode(validation={"frames": np.zeros((40, 8, 8), np.uint8)}))
    with pytest.raises(ValueError):
        write_episode(tmp_path / "b.h5", make_episode(validation={"rgb_mean": np.zeros(40)}))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("change", [
    dict(time=np.arange(40) / 30), dict(q=np.zeros((40, 21))), dict(regime="space"),
    dict(body_parts=("tail",)), dict(tier={"K": {"passed": True, "reasons": []}}),
    dict(gripper_opening=np.full((40, 2), 1.5)), dict(head_base=np.zeros((40, 4, 4))),
    dict(source_time=np.linspace(1, 0, 40)),
])
def test_validation_rejects(change):
    with pytest.raises(ValueError):
        make_episode(**change)


def test_index(tmp_path):
    a, b = make_episode(), make_episode(episode_id="demo_3_mirror", variant_of="robomimic/can/ph/demo_3",
                                        tier={"K": {"passed": True, "reasons": []}, "P": {"passed": True, "reasons": []}})
    path = write_index(tmp_path, [index_row(a, "a.h5"), index_row(b, "b.h5")])
    rows = pq.read_table(path).to_pylist()
    assert [r["uid"] for r in rows] == [a.uid, b.uid]
    assert rows[0]["tier_p_passed"] is None and rows[1]["tier_p_passed"] is True
    assert rows[1]["variant_of"] == a.uid and rows[0]["lineage_seed"] == rows[1]["lineage_seed"]
    assert rows[0]["length"] == 40 and rows[0]["duration"] == pytest.approx(39 / 50)


def test_rotation_helpers():
    rng = np.random.default_rng(1)
    R = rot.so3_exp(rng.normal(size=(50, 3)))
    np.testing.assert_allclose(rot.rot6d_to_matrix(rot.rot6d(R)), R, atol=1e-12)
    np.testing.assert_allclose(rot.rot6d_to_matrix(rot.rot6d(R) * 3), R, atol=1e-12)
    np.testing.assert_allclose(rot.quat_to_matrix(rot.matrix_to_quat(R)), R, atol=1e-12)
    assert rot.rot6d(np.eye(3)).tolist() == [1, 0, 0, 0, 1, 0]
    xi = np.c_[rng.normal(size=(50, 3)), rng.normal(size=(50, 3))]
    xi[0, 3:] = 0
    xi[1, 3:] = 1e-9
    xi[2, 3:] = [np.pi - 1e-4, 0, 0]
    np.testing.assert_allclose(rot.se3_log(rot.se3_exp(xi))[3:], xi[3:], atol=1e-9)
    np.testing.assert_allclose(rot.se3_log(rot.se3_exp(xi))[:2], xi[:2], atol=1e-9)
    np.testing.assert_allclose(rot.se3_exp(rot.se3_log(rot.se3_exp(xi))), rot.se3_exp(xi), atol=1e-9)
    tw = rng.normal(size=(50, 3))
    tw[0, 2] = 0
    np.testing.assert_allclose(rot.se2_log(rot.se2_exp(tw, 0.02), 0.02), tw, atol=1e-9)
    assert rot.wrap_angle(np.pi + 0.1) == pytest.approx(-np.pi + 0.1)
    assert len(JOINTS) == 22


def test_index_command_reads_records_or_episode_files(tmp_path):
    import json

    from reachy_retarget.cli import index_rows
    a, b = make_episode(), make_episode(episode_id="demo_4")
    (tmp_path / "episodes" / "d").mkdir(parents=True)
    write_episode(tmp_path / "episodes" / "d" / "a.h5", a)
    write_episode(tmp_path / "episodes" / "d" / "b.h5", b)
    assert [r["file"] for r in index_rows(tmp_path)] == ["episodes/d/a.h5", "episodes/d/b.h5"]   # .h5 files read
    (tmp_path / "records" / "f").mkdir(parents=True)
    recs = [{"status": "ok", "index_row": index_row(a, "d/a.h5")},                # builds: relative to episodes/
            {"status": "excluded", "excluded": "no_meshes"},                       # not written: no row
            {"status": "ok", "index_row": index_row(b, "d/gone.h5")}]              # file missing: skipped
    (tmp_path / "records" / "f" / "x.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    rows = index_rows(tmp_path)
    assert [(r["uid"], r["file"]) for r in rows] == [(a.uid, "episodes/d/a.h5")]
