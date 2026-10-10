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
from reachy_retarget.acquire import CatalogEntry, ChecksumMismatch, fetch, load_catalog, read_ledger
from reachy_retarget.acquire.transports import read_range, walk_tar
from reachy_retarget.sources import families, iter_episodes

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
    members = {k: e for k, e in cat.items() if e.kind == "range_member"}
    pkls = [m for m in members.values() if m.path.endswith("state_infos.pkl")]
    assert len(pkls) >= 40 and len({m.dataset for m in pkls}) >= 5
    assert all(m.path.startswith("G1_Robot/") for m in pkls)  # dexterous-hand tars excluded
    assert not any(m.path.endswith((".mp4", ".npz", ".pt", ".onnx")) for m in members.values())
    for m in members.values():
        assert len(m.source["archive_sha256"]) == 64 and m.size > 0 and m.source["offset"] > 0
        assert f"/resolve/{m.revision}/" in m.url and m.url.endswith(m.source["archive"])
    hashed = [m for m in pkls if m.sha256]  # members of the earlier subset keep their digests
    assert len(hashed) >= 40 and all(len(m.sha256) == 64 for m in hashed)
    urdf = members["mobilemanibench/Assets/g1_robot_rotate/G1_120s.urdf"]
    assert urdf.source["compression"] == "deflate" and urdf.source["archive"] == "Assets/Assets.zip"
    assert urdf.source["crc32"]
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
    open_url = _opener(blobs, seen)
    walked = list(walk_tar(lambda o, n: read_range("https://x.invalid/t.tar", o, min(n, len(tar_b) - o), open_url)))
    assert [n.rsplit("/", 1)[1] for _, _, n, _ in walked] == ["rgb_image_head.mp4", "state_infos.pkl"]
    off, size, name, _ = walked[1]
    assert tar_b[off:off + size] == b"state" * 100 and max(b - a + 1 for _, a, b in seen) <= 2048
    zinfo = zipfile.ZipFile(io.BytesIO(zip_b)).getinfo("Assets/robot.urdf")
    data_z = b"<robot name='g'/>" * 50
    common = dict(family="mobilemanibench", revision="r", license="MIT", kind="range_member")
    m_tar = CatalogEntry(id=f"mobilemanibench/{name}", path=name, url="https://x.invalid/t.tar", size=size,
                         sha256=hashlib.sha256(b"state" * 100).hexdigest(), content="low_dim",
                         source={"archive": "t.tar", "archive_format": "tar", "archive_sha256": "0" * 64,
                                 "member": name, "offset": off}, **common)
    m_zip = CatalogEntry(id="mobilemanibench/Assets/robot.urdf", path="Assets/robot.urdf", url="https://x.invalid/a.zip",
                         size=len(data_z), sha256=hashlib.sha256(data_z).hexdigest(), content="robot_model",
                         source={"archive": "a.zip", "archive_format": "zip", "offset": zinfo.header_offset,
                                 "stored_size": zinfo.compress_size, "compression": "deflate", "crc32": zinfo.CRC},
                         **common)
    seen.clear()
    recs = fetch([m_tar, m_zip], tmp_path, reserve=0, opener=_opener(blobs, seen))
    assert (tmp_path / "raw/mobilemanibench" / name).read_bytes() == b"state" * 100
    assert (tmp_path / "raw/mobilemanibench/Assets/robot.urdf").read_bytes() == data_z
    assert not list(tmp_path.rglob("*.mp4")) and not list(tmp_path.rglob("*.part"))
    assert all(r["sha256_source"] == "catalog" for r in recs) and len(read_ledger(tmp_path)) == 2
    assert "tar_header" in recs[0]["verified_by"]  # long (GNU) name: truncated header name still matches
    assert sum(b - a + 1 for _, a, b in seen) < len(tar_b) // 2  # only the member ranges were read
    # re-fetch: verified locally, no transfer
    seen.clear()
    fetch([m_tar], tmp_path, reserve=0, opener=_opener(blobs, seen))
    assert seen == []
    # corrupted digest and ignored range are refused
    bad = CatalogEntry(**{**m_tar.__dict__, "id": "mobilemanibench/z", "path": "z", "sha256": "1" * 64})
    with pytest.raises(ChecksumMismatch):
        fetch([bad], tmp_path, reserve=0, opener=_opener(blobs, []))
    other = CatalogEntry(**{**m_tar.__dict__, "id": "mobilemanibench/y", "path": "y"})
    with pytest.raises(RuntimeError, match="ignored the byte range"):
        fetch([other], tmp_path, reserve=0, opener=_opener(blobs, [], status=200))
    assert not (tmp_path / "raw/mobilemanibench/y").exists()


def test_fixture_notes_record_sources():
    import json
    note = json.loads((FIX / "fixture_note.json").read_text())
    for rel, info in note["files"].items():
        assert (RAW / rel).exists() and len(info["source_sha256"]) == 64
    assert sum(f.stat().st_size for f in FIX.rglob("*") if f.is_file()) < 300_000


# ---------------------------------------------------------------- PartNet-Mobility scene (synthetic object)

def _png() -> bytes:
    import struct
    import zlib

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    raw = b"".join(b"\x00" + b"\x80\x40\x20" * 2 for _ in range(2))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _box_obj(lo, hi, mtl="m") -> str:
    v = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    f = [(1, 2, 4), (1, 4, 3), (5, 7, 8), (5, 8, 6), (1, 5, 6), (1, 6, 2), (3, 4, 8), (3, 8, 7), (1, 3, 7), (1, 7, 5),
         (2, 6, 8), (2, 8, 4)]
    return ("mtllib mat.mtl\n" + "".join(f"v {a} {b} {c}\n" for a, b, c in v) + "vt 0 0\nvt 1 0\nvt 1 1\n"
            + f"usemtl {mtl}\n" + "".join(f"f {a}/1 {b}/2 {c}/3\n" for a, b, c in f))


def partnet_object(root: Path, oid="7119", group="microwave", fix_base=True, joint_type="revolute"):
    """Synthetic PartNet-Mobility object: a body box and a door on ``joint_0`` (hinge about z at x = 0.3)."""
    d = root / "raw/mobilemanibench/Assets/partnet/dataset" / oid
    (d / "textured_objs").mkdir(parents=True)
    (d / "images").mkdir()
    (d / "images/texture_0.png").write_bytes(_png())
    (d / "textured_objs/mat.mtl").write_text("newmtl m\nKd 0.5 0.5 0.5\nmap_Kd ../images/texture_0.png\n"
                                             "newmtl g\nKd 0.2 0.2 0.2\nmap_Kd ../images/texture_1.jpg\n")
    (d / "textured_objs/body.obj").write_text(_box_obj((-0.3, -0.2, -0.2), (0.3, 0.2, 0.2)))
    (d / "textured_objs/door.obj").write_text(_box_obj((-0.6, -0.22, -0.2), (0.0, -0.2, 0.2), "g"))
    limit = '<limit lower="0" upper="1.5708"/>' if joint_type == "revolute" else '<limit lower="0" upper="0.3"/>'
    (d / "mobility.urdf").write_text(
        '<robot name="o"><link name="base"/><link name="link_1"><visual><geometry><mesh filename="textured_objs/body.obj"/>'
        '</geometry></visual><collision><geometry><mesh filename="textured_objs/body.obj"/></geometry></collision></link>'
        '<joint name="joint_1" type="fixed"><parent link="base"/><child link="link_1"/></joint>'
        '<link name="link_0"><visual><geometry><mesh filename="textured_objs/door.obj"/></geometry></visual>'
        '<collision><geometry><mesh filename="textured_objs/door.obj"/></geometry></collision></link>'
        f'<joint name="joint_0" type="{joint_type}"><origin xyz="0.3 0 0"/><axis xyz="0 0 1"/><parent link="link_1"/>'
        f'<child link="link_0"/>{limit}</joint></robot>')
    c = root / "raw/mobilemanibench/Assets/partnet/process" / group / oid
    c.mkdir(parents=True)
    (c / "config.yaml").write_text(f"fix_base: {str(fix_base).lower()}\n")
    a = root / "raw/mobilemanibench/code/unimanip/configs/data"
    a.mkdir(parents=True, exist_ok=True)
    (a / "analysis_scene.yaml").write_text(f"object:\n  partnet:\n    {group}:\n      scale: 0.5\n      place: tabletop\n"
                                           "      init_height: 0.2\n")
    return d


def test_partnet_scene_solves_the_grasp_joint_and_a_free_root(tmp_path):
    import mujoco

    from reachy_retarget.sources import partnet_scene as ps
    partnet_object(tmp_path)
    base = tmp_path / "raw/mobilemanibench/"

    def read(rel):
        return (base / rel).read_bytes()
    root0 = [0.2, 0.1, 0.8, np.cos(0.3), 0, 0, np.sin(0.3)]
    ref, obj, model, prov = ps.build_scene(read, "Assets/partnet/dataset/7119", name="microwave_7119", scale=0.5,
                                           root_pose=root0, free=False, place="tabletop", stage_top=0.63,
                                           grasp_joint="joint_0", grasp_q0=1.2)
    assert prov["jpeg_textures_dropped"] == ["Assets/partnet/dataset/7119/images/texture_1.jpg"] and model.ntex == 1
    assert prov["stage"]["top_z"] == 0.63 and prov["ground"]["top_z"] == 0.01
    init = obj.initial_joints("joint_0", 1.2)
    assert init == {"microwave_7119:joint_0": 1.2}
    # handle poses generated by forward kinematics are recovered exactly
    d = mujoco.MjData(model)
    q_true = np.linspace(1.2, 0.05, 30)
    H = np.zeros((30, 4, 4))
    offset = np.eye(4)
    offset[:3, 3] = [-0.25, -0.21, 0.1]
    for t, q in enumerate(q_true):
        d.qpos[model.joint("microwave_7119:joint_0").qposadr[0]] = q
        mujoco.mj_forward(model, d)
        L = np.eye(4)
        L[:3, :3], L[:3, 3] = d.xmat[model.body("microwave_7119:link_0").id].reshape(3, 3), d.xpos[model.body("microwave_7119:link_0").id]
        H[t] = L @ offset
    q, check = ps.solve_grasp_joint(model, H, joint="microwave_7119:joint_0", link="microwave_7119:link_0",
                                    initial=init, joint_type="REVOLUTE")
    np.testing.assert_allclose(q, q_true, atol=1e-9)
    assert check["max_off_axis_rad"] < 1e-9 and check["handle_offset_in_link_spread_m"] < 1e-9
    # a free root carrying the grasp link (cart): root(t) = handle(t) . offset^-1
    ref, obj, model, _ = ps.build_scene(read, "Assets/partnet/dataset/7119", name="cart_7119", scale=0.5, root_pose=root0,
                                        free=True, place="space", stage_top=0.0, grasp_joint=None, grasp_q0=None)
    d = mujoco.MjData(model)
    d.qpos[:7] = root0
    mujoco.mj_forward(model, d)
    L = np.eye(4)
    L[:3, :3], L[:3, 3] = d.xmat[model.body("cart_7119:link_1").id].reshape(3, 3), d.xpos[model.body("cart_7119:link_1").id]
    moved = np.tile(L @ offset, (30, 1, 1))   # handle on the root's fixed link, the cart pushed 0.3 m along x
    moved[:, 0, 3] += np.linspace(0, 0.3, 30)
    pose, check = ps.cart_root_track(model, moved, np.array(root0), link="cart_7119:link_1",
                                     initial=obj.initial_joints(None, None))
    np.testing.assert_allclose(pose[0], root0, atol=1e-9)
    np.testing.assert_allclose(pose[:, 0] - pose[0, 0], np.linspace(0, 0.3, 30), atol=1e-9)


def test_close_microwave_with_partnet_scene(tmp_path):
    from reachy_retarget.evaluate import process_source
    from reachy_retarget.schema.scene_assets import read_scene
    shutil.copytree(FIX / "raw", tmp_path / "raw")
    partnet_object(tmp_path)
    ep = one(tmp_path / MICRO.relative_to(FIX))
    assert ep.scene is not None and ep.provenance["scene"].startswith("partnet_scene")
    root = ep.objects["microwave_7119"]
    assert root.valid.all() and root.geometry["body"] == "microwave_7119" and np.ptp(root.pose, axis=0).max() == 0
    p = ep.provenance["articulation"]
    assert p["label"].startswith("solved") and p["solved_t0"] == pytest.approx(ep.articulations["microwave_7119"].qpos[0, 0])
    assert ep.scene_qpos.joint_names == ["microwave_7119:joint_0"]
    assert ep.provenance["scene_omitted"][0]["component"] == "room" and ep.provenance["scene_build"]["stage"]["top_z"] == 0.63
    rec = process_source(ep, physics=False, write=str(tmp_path / "out"), layout="dataset")
    assert rec["status"] == "ok" and rec["file"], rec.get("reason")
    scene, _ = read_scene(rec["file"])
    names = {c["name"] for c in scene.components}
    assert {"microwave_7119", "microwave_7119:link_0", "stage", "ground"} <= names and scene.valid.all()
