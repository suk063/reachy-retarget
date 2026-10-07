"""Offline tests of the MobileManiBench adapter and its member fetcher.

Fixture (``tests/fixtures/mobilemanibench``, dataset MIT / code BSD-3-Clause) mirrors the
extracted tar layout under ``raw/mobilemanibench``: two episodes' ``state_infos.pkl`` with
every 8th row (plus the last) of the published files (G1 close microwave 7119, G1 pick YCB
021_bleach_cleanser), their ``scene_infos.json``, a key subset of each ``params/env.yaml``,
the pinned ``G1_120s.urdf`` without visual/collision/inertial elements and one entry of each
prompt table. Source digests are in ``fixture_note.json``.
"""
import hashlib
import io
import pickle
import shutil
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import reachy_retarget.sources.mobilemanibench as mmb  # noqa: F401  (registers the family)
from reachy_retarget.acquire import load_catalog
from reachy_retarget.acquire.fetch import ChecksumMismatch, read_ledger
from reachy_retarget.sources import families, iter_episodes
from reachy_retarget.sources.mobilemanibench_fetch import MemberEntry, fetch_members, load_members, walk_tar

FIX = Path(__file__).parent / "fixtures" / "mobilemanibench"
RAW = FIX / "raw" / "mobilemanibench"
MICRO = RAW / ("G1_Robot/Close/partnet/microwave/0000/7119-joint_0-REVOLUTE-link_0-handle_0/train_0/"
               "trajectories/traj_000/episode_000/state_infos.pkl")
YCB = RAW / "G1_Robot/Open/ycb/ycb/0018/021_bleach_cleanser/train_0/trajectories/traj_000/episode_000/state_infos.pkl"


def one(path, **kw):
    eps = list(iter_episodes("mobilemanibench", path, **kw))
    assert len(eps) == 1
    return eps[0]


def test_registry_and_catalog():
    assert "mobilemanibench" in families()
    cat = {k: e for k, e in load_catalog().items() if e.family == "mobilemanibench"}
    assert "mobilemanibench/MobileManiDataset/object_manifest.jsonl" in cat
    assert "mobilemanibench/code/unimanip/configs/data/analysis_partnet.yaml" in cat
    members = load_members()
    pkls = [m for m in members.values() if m.path.endswith("state_infos.pkl")]
    assert len(pkls) == 40 and len({m.dataset for m in pkls}) == 5
    assert all(m.path.startswith("G1_Robot/") for m in pkls)  # dexterous-hand tars excluded
    assert not any(m.path.endswith((".mp4", ".npz")) for m in members.values())
    for m in members.values():
        assert len(m.sha256) == 64 and len(m.archive_sha256) == 64 and m.size > 0
        assert f"/resolve/{m.revision}/" in m.url and m.url.endswith(m.archive)
    urdf = members["mobilemanibench/Assets/g1_robot_rotate/G1_120s.urdf"]
    assert urdf.compression == "deflate" and urdf.archive == "Assets/Assets.zip" and urdf.crc32
    for p in pkls:  # every episode has its env.yaml and scene_infos.json catalogued
        train = p.path.split("/trajectories/")[0]
        traj = p.path.rsplit("/", 2)[0]
        assert f"mobilemanibench/{train}/params/env.yaml" in members
        assert f"mobilemanibench/{traj}/scene_infos.json" in members


def test_close_microwave_episode():
    ep = one(MICRO)
    T = ep.length
    assert T == 23 and ep.regime == "mobile_manipulation" and ep.scene is None and ep.success is True
    assert ep.task == "close_microwave" and ep.instruction == "close microwave"
    assert ep.dataset == "mobilemanibench/G1_Robot/Close/partnet/microwave"
    assert ep.episode_id == "0000/7119-joint_0-REVOLUTE-link_0-handle_0/traj_000/episode_000"
    assert np.isclose(ep.time[0], 1 / 30) and np.allclose(np.diff(ep.time)[:-1], 8 / 30)
    assert set(ep.effectors) == {"right"} and ep.effectors["right"].side_hint == "right"
    assert ep.base.shape == (T, 3) and np.allclose(ep.base_hint, ep.base[0])
    assert np.allclose(ep.torso_height, ep.torso_height[0], atol=1e-3) and 0.9 < ep.torso_height[0] < 1.1
    p = ep.provenance
    assert p["fk_check"]["right"]["max_position_error_m"] < 1e-5 and p["fk_check"]["right"]["max_rotation_error_rad"] < 1e-5
    assert p["base_check"]["max_xy_error_m"] < 1e-5 and p["base_check"]["root_motion_m"] == 0
    assert p["catalog_id"] is None  # fixture rows are a subset, not the catalogued file
    assert {k: o.role for k, o in ep.objects.items()} == {"microwave_7119_handle_0": "manipulated",
                                                         "microwave_7119": "fixture"}
    root = ep.objects["microwave_7119"]
    assert root.valid[0] and not root.valid[1:].any()
    art = ep.articulations["microwave_7119"]
    assert art.joint_names == ["microwave_7119:joint_0"]
    assert abs(art.qpos[0, 0] - 1.094) < 1e-3 and abs(art.qpos[-1, 0]) < 0.01  # closed at the end
    assert p["articulation"]["label"].startswith("derived") and p["articulation"]["off_axis_residual"] < 1e-4
    assert ep.lineage["generated"] and not ep.lineage["human"] and ep.lineage["variant_of"] is None


def test_grasp_center_convention():
    ep = one(MICRO)
    d = mmb.load_state(MICRO)
    e = ep.effectors["right"]
    R = e.pose[:, :3, :3]
    assert np.allclose(np.einsum("tji,tjk->tik", R, R), np.eye(3), atol=1e-6)
    pads = d["robot_body"][:, [mmb.B["gripper_r_inner_link5"], mmb.B["gripper_r_outer_link5"]], :3].astype(float)
    mid = pads.mean(1)
    assert np.linalg.norm(mid - e.pose[:, :3, 3], axis=1).max() < 0.01  # point between the pads
    sep = pads[:, 0] - pads[:, 1]
    assert np.all(np.einsum("ti,ti->t", sep, R[:, :, 1]) > 0.95 * np.linalg.norm(sep, axis=1))  # closing axis = +y
    center = d["robot_hand"][:, :3].astype(float)
    depth = np.einsum("ti,ti->t", e.pose[:, :3, 3] - center, R[:, :, 2])
    assert 0.08 < depth.min() and depth.max() < 0.11  # approach axis = +z, pads ahead of the center link
    assert np.allclose(e.width, np.linalg.norm(sep, axis=1)) and 0 <= e.opening.min() and e.opening.max() <= 1


def test_pick_ycb_episode():
    ep = one(YCB)
    assert ep.task == "pick_bleach_cleanser" and ep.instruction == "pick bleach cleanser"
    assert list(ep.objects) == ["bleach_cleanser"] and ep.objects["bleach_cleanser"].role == "manipulated"
    assert not ep.articulations and ep.success
    obj = ep.objects["bleach_cleanser"].pose
    assert obj[-1, 2] - obj[0, 2] > 0.15  # lifted
    assert ep.provenance["fk_check"]["right"]["max_position_error_m"] < 1e-5


def test_folder_and_limit():
    eps = list(iter_episodes("mobilemanibench", RAW / "G1_Robot"))
    assert sorted(e.task for e in eps) == ["close_microwave", "pick_bleach_cleanser"]
    assert len(list(iter_episodes("mobilemanibench", RAW / "G1_Robot", limit=1))) == 1


def test_without_urdf_no_fk(tmp_path):
    dst = tmp_path / "x" / MICRO.relative_to(RAW).parents[3]
    shutil.copytree(MICRO.parents[3], dst)
    ep = one(dst / "trajectories/traj_000/episode_000/state_infos.pkl")
    assert ep.provenance["fk_check"] is None and ep.instruction == "close microwave"  # fallback prompt


def test_xhand_refused(tmp_path):
    rel = MICRO.relative_to(RAW / "G1_Robot")
    dst = tmp_path / "XHand_Robot" / rel
    shutil.copytree(MICRO.parents[3], dst.parents[3])
    with pytest.raises(ValueError, match="XHand"):
        one(dst)


def test_unsafe_pickle_refused(tmp_path):
    class Evil:
        def __reduce__(self):
            return (print, ("x",))
    p = tmp_path / "state_infos.pkl"
    p.write_bytes(pickle.dumps({"time": Evil()}))
    with pytest.raises(pickle.UnpicklingError):
        mmb.load_state(p)


def test_parse_and_prompt():
    assert mmb.parse_object_name("19179-joint_0-PRISMATIC-link_0-handle_0")["joint_type"] == "PRISMATIC"
    assert mmb.parse_object_name("100491-joint_4-None-link_4-handle_0")["joint_type"] == "None"
    assert mmb.parse_object_name("021_bleach_cleanser")["joint"] is None
    table = {"oven": {"7120/joint_1/REVOLUTE/link_1/handle_0": "train/oven/bottom"}}
    assert mmb.instruction_for("open", "partnet", "oven", "7120-joint_1-REVOLUTE-link_1-handle_0", table)[0] == \
        "open oven at bottom"
    assert mmb.instruction_for("push", "partnet", "cart", "x", None)[0] == "push cart"


def test_derive_articulation_revolute_and_prismatic():
    T = 50
    th = np.linspace(0, 1.2, T)
    hinge, r = np.array([0.3, -0.2, 0.0]), 0.4
    H = np.tile(np.eye(4), (T, 1, 1))
    H[:, :3, :3] = Rotation.from_euler("z", th[:, None]).as_matrix()
    H[:, :3, 3] = hinge + np.column_stack([r * np.cos(th), r * np.sin(th), np.full(T, 0.8)])
    out = mmb.derive_articulation(H, "REVOLUTE")
    assert np.allclose(out["theta"], th, atol=1e-9) and np.allclose(out["axis_world"], [0, 0, 1])
    assert np.allclose(out["hinge_point_world"][:2], hinge[:2], atol=1e-6) and abs(out["handle_radius_m"] - r) < 1e-6
    P = np.tile(np.eye(4), (T, 1, 1))
    P[:, :3, 3] = np.outer(np.linspace(0, 0.5, T), [0, -1, 0])
    out = mmb.derive_articulation(P, "PRISMATIC")
    assert np.isclose(out["theta"][-1], 0.5) and out["rotation_residual_rad"] == 0


# ---------------------------------------------------------------- fetcher (fake HTTP)

class _Resp(io.BytesIO):
    def __init__(self, data, status=206):
        super().__init__(data)
        self.status = status


def _archive_bytes():
    tbuf = io.BytesIO()
    with tarfile.open(fileobj=tbuf, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for name, data in [("R/a/episode_000/rgb_image_head.mp4", b"\x00" * 3000),
                           ("R/a/" + "long" * 40 + "/state_infos.pkl", b"state" * 100)]:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("Assets/robot.urdf", b"<robot name='g'/>" * 50)
    return tbuf.getvalue(), zbuf.getvalue()


def _opener(blobs, seen, status=206):
    def open_url(req):
        a, b = map(int, req.get_header("Range").split("=")[1].split("-"))
        seen.append((req.full_url, a, b))
        return _Resp(blobs[req.full_url][a:b + 1], status)
    return open_url


def test_walk_tar_and_fetch_members(tmp_path):
    tar_b, zip_b = _archive_bytes()
    blobs = {"https://x.invalid/t.tar": tar_b, "https://x.invalid/a.zip": zip_b}
    seen = []
    walked = list(walk_tar("https://x.invalid/t.tar", open_url=_opener(blobs, seen)))
    assert [n.rsplit("/", 1)[1] for _, _, n in walked] == ["rgb_image_head.mp4", "state_infos.pkl"]
    off, size, name = walked[1]
    assert tar_b[off:off + size] == b"state" * 100 and max(b - a + 1 for _, a, b in seen) <= 2048
    zinfo = zipfile.ZipFile(io.BytesIO(zip_b)).getinfo("Assets/robot.urdf")
    data_z = b"<robot name='g'/>" * 50
    common = dict(revision="r", license="MIT", archive_sha256="0" * 64, archive_size=1)
    m_tar = MemberEntry(id=f"mobilemanibench/{name}", path=name, archive="t.tar", url="https://x.invalid/t.tar",
                        offset=off, stored_size=size, size=size, sha256=hashlib.sha256(b"state" * 100).hexdigest(),
                        **common)
    m_zip = MemberEntry(id="mobilemanibench/Assets/robot.urdf", path="Assets/robot.urdf", archive="a.zip",
                        url="https://x.invalid/a.zip", offset=zinfo.header_offset, stored_size=zinfo.compress_size,
                        size=len(data_z), sha256=hashlib.sha256(data_z).hexdigest(), compression="deflate",
                        crc32=zinfo.CRC, kind="robot_model", **common)
    seen.clear()
    recs = fetch_members([m_tar, m_zip], tmp_path, reserve=0, opener=_opener(blobs, seen))
    assert (tmp_path / "raw/mobilemanibench" / name).read_bytes() == b"state" * 100
    assert (tmp_path / "raw/mobilemanibench/Assets/robot.urdf").read_bytes() == data_z
    assert not list(tmp_path.rglob("*.mp4")) and not list(tmp_path.rglob("*.part"))
    assert all(r["sha256_source"] == "catalog" for r in recs) and len(read_ledger(tmp_path)) == 2
    assert sum(b - a + 1 for _, a, b in seen) < len(tar_b) // 2  # only the member ranges were read
    # re-fetch: verified locally, no transfer
    seen.clear()
    fetch_members([m_tar], tmp_path, reserve=0, opener=_opener(blobs, seen))
    assert seen == []
    # corrupted digest and ignored range are refused
    bad = MemberEntry(**{**m_tar.__dict__, "id": "mobilemanibench/z", "path": "z", "sha256": "1" * 64})
    with pytest.raises(ChecksumMismatch):
        fetch_members([bad], tmp_path, reserve=0, opener=_opener(blobs, []))
    other = MemberEntry(**{**m_tar.__dict__, "id": "mobilemanibench/y", "path": "y"})
    with pytest.raises(RuntimeError, match="ignored the byte range"):
        fetch_members([other], tmp_path, reserve=0, opener=_opener(blobs, [], status=200))
    assert not (tmp_path / "raw/mobilemanibench/y").exists()


def test_fixture_notes_record_sources():
    import json
    note = json.loads((FIX / "fixture_note.json").read_text())
    for rel, info in note["files"].items():
        assert (RAW / rel).exists() and len(info["source_sha256"]) == 64
    assert sum(f.stat().st_size for f in FIX.rglob("*") if f.is_file()) < 300_000
