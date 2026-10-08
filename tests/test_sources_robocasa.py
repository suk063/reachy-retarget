"""Offline tests of the RoboCasa adapter and its acquisition helpers (tiny synthetic fixtures)."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import struct
import tarfile
import zipfile

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

from reachy_retarget.acquire import (TAR_MANIFEST, CatalogEntry, ChecksumMismatch, InsufficientDisk,  # noqa: E402
                                     fetch, is_image_name, load_catalog)
from reachy_retarget.acquire.assets import fetch_robocasa_asset_subset  # noqa: E402
from reachy_retarget.acquire.generators import robocasa as gen  # noqa: E402
from reachy_retarget.sources.robocasa import RobocasaAssets, read_robocasa  # noqa: E402
from reachy_retarget.sources.registry import iter_episodes  # noqa: E402

MESH_REF = "/root/robocasa/robocasa/models/assets/objects/objaverse/cube/model.stl"

XML = f"""
<mujoco model="toy_kitchen">
  <compiler angle="radian" meshdir="meshes/"/>
  <asset><mesh name="obj_mesh" file="{MESH_REF}"/></asset>
  <worldbody>
    <body name="robot0_base" pos="10 10 0">
      <body name="mobilebase0_base" pos="0 0 0">
        <joint name="mobilebase0_joint_mobile_forward" type="slide" axis="1 0 0"/>
        <joint name="mobilebase0_joint_mobile_side" type="slide" axis="0 1 0"/>
        <joint name="mobilebase0_joint_mobile_yaw" type="hinge" axis="0 0 1"/>
        <geom type="box" size="0.3 0.3 0.1" pos="0 0 0.1" mass="10"/>
        <body name="mobilebase0_support" pos="0 0 0.2">
          <joint name="mobilebase0_joint_torso_height" type="slide" axis="0 0 1" range="0 0.34"/>
          <site name="mobilebase0_center" pos="0 0 0"/>
          <geom type="box" size="0.05 0.05 0.3" pos="0 0 0.3" mass="2"/>
          <body name="gripper0_right_right_gripper" pos="0.5 0 1.0" euler="0 1.5707963 0">
            <geom type="box" size="0.02 0.05 0.02" mass="0.5"/>
            <site name="gripper0_right_grip_site" pos="0 0 0.1"/>
            <body name="gripper0_right_leftfinger" pos="0 0 0.05">
              <joint name="gripper0_right_finger_joint1" type="slide" axis="0 1 0" range="0 0.04"/>
              <geom name="gripper0_right_finger1_pad_collision" type="box" size="0.005 0.002 0.005"
                    pos="0 0.005 0.045" mass="0.05"/>
            </body>
            <body name="gripper0_right_rightfinger" pos="0 0 0.05">
              <joint name="gripper0_right_finger_joint2" type="slide" axis="0 1 0" range="-0.04 0"/>
              <geom name="gripper0_right_finger2_pad_collision" type="box" size="0.005 0.002 0.005"
                    pos="0 -0.005 0.045" mass="0.05"/>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="obj_main" pos="1 0 0.9"><freejoint name="obj_joint0"/>
      <geom type="mesh" mesh="obj_mesh" mass="0.1"/></body>
    <body name="distr_counter_main" pos="1 0.5 0.9"><freejoint name="distr_counter_joint0"/>
      <geom type="box" size="0.03 0.03 0.03" mass="0.1"/></body>
    <body name="cab_1_main" pos="2 0 1">
      <geom type="box" size="0.3 0.3 0.3"/>
      <body name="cab_1_hingedoor" pos="0.3 0 0">
        <joint name="cab_1_doorhinge" type="hinge" axis="0 0 1" range="-1.57 0"/>
        <geom type="box" size="0.01 0.3 0.3" mass="1"/>
      </body>
    </body>
    <body name="counter_1_main" pos="1 0 0.4"><geom type="box" size="0.5 0.5 0.4"/></body>
    <body name="floor_room_main" pos="0 0 -0.02"><geom type="box" size="20 20 0.02"/></body>
    <body name="floor_backing_room_main" pos="0 0 -0.14"><geom type="box" size="20 20 0.1"/></body>
  </worldbody>
</mujoco>
"""

EP_META = {"lang": "Pick the cube from the counter and place it in the cabinet.", "layout_id": 1, "style_id": 2,
           "object_cfgs": [{"name": "obj", "graspable": True, "info": {"cat": "cube", "mjcf_path": "x"}},
                           {"name": "distr_counter", "info": {"cat": "bowl"}}],
           "fixture_refs": {"cab": "cab_1", "counter": "counter_1"},
           "fixtures": {"cab_1": {"cls": "HingeCabinet"}, "counter_1": {"cls": "Counter"}},
           "init_robot_base_pos": [0, 0, 0], "init_robot_base_ori": [0, 0, 0]}


def tetra_stl() -> bytes:
    v = np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]], np.float32)
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    out = bytearray(b"\0" * 80 + struct.pack("<I", len(faces)))
    for f in faces:
        out += struct.pack("<3f", 0, 0, 0) + b"".join(struct.pack("<3f", *v[i]) for i in f) + b"\0\0"
    return bytes(out)


def _model_without_mesh():
    xml = XML.replace(f'<asset><mesh name="obj_mesh" file="{MESH_REF}"/></asset>', "") \
             .replace('type="mesh" mesh="obj_mesh"', 'type="sphere" size="0.02"')
    return mujoco.MjModel.from_xml_string(xml)


def make_dataset(root, *, moving: bool, name="Toy", with_obs=True, n=6):
    """A one-episode RoboCasa-layout dataset under ``root/raw/robocasa/v1.0/...``."""
    m = _model_without_mesh()
    d = mujoco.MjData(m)
    adr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): m.jnt_qposadr[j] for j in range(m.njnt)}
    rows, obs = [], []
    site = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in ("mobilebase0_center",
                                                                            "gripper0_right_grip_site")}
    for t in range(n):
        d.qpos[:] = m.qpos0
        f = t / (n - 1)
        if moving:
            d.qpos[adr["mobilebase0_joint_mobile_forward"]] = 0.5 * f
            d.qpos[adr["mobilebase0_joint_mobile_yaw"]] = 0.3 * f
        d.qpos[adr["mobilebase0_joint_torso_height"]] = 0.1 * f
        d.qpos[adr["gripper0_right_finger_joint1"]] = 0.04 * (1 - f)
        d.qpos[adr["gripper0_right_finger_joint2"]] = -0.04 * (1 - f)
        d.qpos[adr["obj_joint0"]:adr["obj_joint0"] + 3] = [1, 0, 0.9 + 0.2 * f]
        d.qpos[adr["cab_1_doorhinge"]] = -1.0 * f
        mujoco.mj_forward(m, d)
        rows.append(np.r_[0.5 + 0.05 * t, d.qpos, d.qvel])
        c, R = d.site_xpos[site["mobilebase0_center"]], d.site_xmat[site["mobilebase0_center"]].reshape(3, 3)
        rel = R.T @ (d.site_xpos[site["gripper0_right_grip_site"]] - c)
        obs.append(np.r_[c, 0, 0, 0, 1, rel, 0, 0, 0, 1, d.qpos[adr["gripper0_right_finger_joint1"]],
                         d.qpos[adr["gripper0_right_finger_joint2"]]])
    lerobot = root / "raw" / "robocasa" / "v1.0" / "pretrain" / "atomic" / name / "20250101" / "lerobot"
    ex = lerobot / "extras" / "episode_000000"
    ex.mkdir(parents=True)
    (ex / "model.xml.gz").write_bytes(gzip.compress(XML.encode()))
    np.savez(ex / "states.npz", states=np.array(rows))
    (ex / "ep_meta.json").write_text(json.dumps(EP_META))
    (lerobot / "extras" / "dataset_meta.json").write_text(json.dumps(
        {"env": name, "robocasa_version": "0.5.1", "robosuite_version": "1.5.2", "mujoco_version": "3.3.1",
         "env_args": {"env_kwargs": {"robots": "PandaOmron",
                                     "controller_configs": {"type": "HYBRID_MOBILE_BASE",
                                                            "body_parts": {"base": {"type": "JOINT_VELOCITY"}}}}}}))
    (lerobot / "meta").mkdir()
    (lerobot / "meta" / "info.json").write_text(json.dumps({"fps": 20}))
    table = {"next.reward": pa.array([0.0] * (n - 1) + [1.0], pa.float32())}
    if with_obs:
        table["observation.state"] = pa.array([list(o) for o in obs], pa.list_(pa.float64(), 16))
    (lerobot / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.table(table), lerobot / "data" / "chunk-000" / "episode_000000.parquet")
    return lerobot.parent


def subset_assets(root):
    p = root / "raw" / "robocasa" / "asset_subset" / "objects" / "objaverse" / "cube" / "model.stl"
    p.parent.mkdir(parents=True)
    p.write_bytes(tetra_stl())


# ---------------------------------------------------------------- adapter

def test_kinematic_only_without_assets(tmp_path):
    ds = make_dataset(tmp_path, moving=False)
    (ep,) = read_robocasa(ds, root=tmp_path, catalog={}, tars={})
    p = ep.provenance
    assert p["state_route"] == "mjcf_kinematic_only" and ep.scene is None
    assert p["missing_assets"] == [MESH_REF]
    assert ep.length == 6 and np.allclose(np.diff(ep.time), 0.05)
    assert ep.regime == "tabletop" and ep.base is None and ep.base_hint is not None
    assert np.allclose(ep.base_hint, [10, 10, 0])
    assert ep.instruction == EP_META["lang"] and ep.success is True
    assert ep.license == "CC-BY-4.0" and ep.lineage == {"generated": None}
    assert np.allclose(ep.torso_height, np.linspace(0, 0.1, 6))
    # Task objects with roles from the task definition; distractors are not task objects.
    assert ep.objects["obj"].role == "manipulated" and ep.objects["obj"].geometry["category"] == "cube"
    assert ep.objects["cab_1"].role == "receptacle" and ep.objects["counter_1"].role == "support"
    assert "distr_counter" not in ep.objects and "distr_counter" in p["distractors"]
    assert np.allclose(ep.objects["obj"].pose[:, 2], 0.9 + 0.2 * np.linspace(0, 1, 6))
    # Articulations keyed by fixture name with joint names.
    art = ep.articulations["cab_1"]
    assert art.joint_names == ["cab_1_doorhinge"] and np.isclose(art.qpos[-1, 0], -1.0)
    assert p["task_articulations"] == ["cab_1"] and p["articulation_motion"]["cab_1"] == pytest.approx(1.0)
    # Gripper: grasp-center frame (+z approach, +y closing), width from the finger joints.
    e = ep.effectors["gripper0_right"]
    assert np.allclose(e.width, 0.08 * (1 - np.linspace(0, 1, 6)), atol=1e-9)
    assert np.isclose(e.opening[0], 1) and np.isclose(e.opening[-1], 0)
    assert np.allclose(np.linalg.det(e.pose[:, :3, :3]), 1)
    # Recorded observations reproduce exactly.
    oc = p["observation_check"]
    assert oc["compared"] and max(oc["base_position_max_err_m"], oc["eef_position_relative_max_err_m"],
                                  oc["gripper_qpos_max_err"]) < 1e-9


def test_floor_must_be_at_z0(tmp_path):
    ds = make_dataset(tmp_path, moving=False)
    (ep,) = read_robocasa(ds, root=tmp_path, catalog={}, tars={})
    assert ep.provenance["world_frame"]["floor_top_z_m"] == pytest.approx(0.0)
    p = ds / "lerobot" / "extras" / "episode_000000" / "model.xml.gz"
    p.write_bytes(gzip.compress(XML.replace('name="floor_room_main" pos="0 0 -0.02"',
                                            'name="floor_room_main" pos="0 0 0.08"').encode()))
    with pytest.raises(ValueError, match="floor"):
        next(read_robocasa(ds, root=tmp_path, catalog={}, tars={}))


def test_grasp_center_convention(tmp_path):
    ds = make_dataset(tmp_path, moving=False)
    (ep,) = read_robocasa(ds, root=tmp_path, catalog={}, tars={})
    m = _model_without_mesh()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    states = np.load(ds / "lerobot" / "extras" / "episode_000000" / "states.npz")["states"]
    d.qpos[:] = states[0, 1:1 + m.nq]
    mujoco.mj_forward(m, d)
    pads = [d.geom_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"gripper0_right_finger{i}_pad_collision")]
            for i in (1, 2)]
    hand = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper0_right_right_gripper")]
    T = ep.effectors["gripper0_right"].pose[0]
    assert np.allclose(T[:3, 3], (pads[0] + pads[1]) / 2, atol=1e-9)        # pad midpoint
    z = (pads[0] + pads[1]) / 2 - hand
    assert np.allclose(T[:3, 2], z / np.linalg.norm(z), atol=1e-6)            # approach: palm -> pads
    assert np.allclose(T[:3, 1], (pads[1] - pads[0]) / np.linalg.norm(pads[1] - pads[0]), atol=1e-6)


def test_mobile_base_path_and_regime(tmp_path):
    ds = make_dataset(tmp_path, moving=True)
    (ep,) = iter_episodes("robocasa", ds, root=tmp_path, catalog={}, tars={})
    assert ep.regime == "mobile_manipulation" and ep.base.shape == (6, 3)
    assert np.allclose(ep.base[:, 0], 10 + 0.5 * np.linspace(0, 1, 6))
    assert np.allclose(ep.base[:, 1], 10) and np.allclose(ep.base[:, 2], 0.3 * np.linspace(0, 1, 6))
    assert ep.provenance["base"]["travel_m"] == pytest.approx(0.5, abs=1e-4)
    assert np.allclose(ep.base_hint, ep.base[0])


def test_navigation_regime_without_task_objects(tmp_path):
    ds = make_dataset(tmp_path, moving=True, name="NavigateKitchen")
    meta_p = ds / "lerobot" / "extras" / "episode_000000" / "ep_meta.json"
    meta = {**EP_META, "object_cfgs": [], "fixture_refs": {"target_fixture": "counter_1"}}
    meta_p.write_text(json.dumps(meta))
    (ep,) = read_robocasa(ds, root=tmp_path, catalog={}, tars={})
    assert ep.regime == "navigation" and ep.base is not None and not any(
        o.role == "manipulated" for o in ep.objects.values())


def test_assets_from_subset_dir_give_physics_scene(tmp_path):
    ds = make_dataset(tmp_path, moving=False)
    subset_assets(tmp_path)
    (ep,) = read_robocasa(ds, root=tmp_path, catalog={}, tars={})
    p = ep.provenance
    assert p["state_route"] == "mjcf_states" and p["missing_assets"] == [] and p["assets_from_subset"]
    assert ep.scene is not None and "mobilebase0_" in ep.scene.robot_prefixes
    assert "robocasa/objects__objaverse__cube__model.stl" in ep.scene.assets
    assert ep.objects["obj"].geometry["kind"] == "aabb"
    assert set(ep.scene.initial_qpos) >= {"obj_joint0", "distr_counter_joint0", "cab_1_doorhinge"}
    m = mujoco.MjModel.from_xml_string(ep.scene.mjcf, ep.scene.assets)
    assert m.nmesh == 1


def test_assets_from_catalogued_zip(tmp_path):
    ds = make_dataset(tmp_path, moving=False)
    z = tmp_path / "raw" / "robocasa" / "assets" / "objaverse.zip"
    z.parent.mkdir(parents=True)
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("objaverse/cube/model.stl", tetra_stl())
    e = CatalogEntry(id="robocasa/assets/objaverse.zip", family="robocasa", path="assets/objaverse.zip", url="x",
                     revision="r", sha256="0" * 64, size=1, license="CC-BY-4.0", kind="file",
                     content="assets", asset_marker="robocasa/models/assets/objects/")
    (ep,) = read_robocasa(ds, root=tmp_path, catalog={e.id: e}, tars={})
    assert ep.provenance["state_route"] == "mjcf_states" and not ep.provenance["assets_from_subset"]
    used = {s["id"]: s["members_used"] for s in ep.provenance["asset_sources"]}
    assert used == {"robocasa/assets/objaverse.zip": 1}


def test_manifest_mismatch_and_lineage(tmp_path):
    ds = make_dataset(tmp_path, moving=False, name="OpenDrawer")
    rel = "v1.0/pretrain/atomic/OpenDrawer/20250101/mg/demo/x/lerobot.tar"
    seed = "robocasa/v1.0/pretrain/atomic/OpenDrawer/20250101"
    tar = CatalogEntry(id=f"robocasa/{rel}", family="robocasa", path=rel, url="u", revision="c", sha256=None, size=1,
                       license="CC-BY-4.0", kind="tar_stream", content="low_dim", dataset="robocasa/" + rel.rsplit("/", 1)[0],
                       usable=False, digests={"sha1": "a" * 40},
                       meta=dict(task="OpenDrawer", split="pretrain", task_type="atomic", source="mg", variant="mg_path",
                                 shared_link="s", box_file_id="1", box_file_version="2", seed=seed))
    # Point the tar entry at the fixture directory and lineage follows the catalogue.
    tar = CatalogEntry(**{**tar.__dict__, "path": "v1.0/pretrain/atomic/OpenDrawer/20250101/lerobot.tar"})
    (ep,) = read_robocasa(ds, root=tmp_path, catalog={}, tars={tar.id: tar})
    assert ep.lineage == {"generated": True, "seed": seed, "seed_demo": None, "variant_group": "mg_path"}
    assert ep.dataset == tar.dataset and ep.provenance["revision"]["box_sha1"] == "a" * 40
    (ds / TAR_MANIFEST).write_text(json.dumps({"members": [
        {"member": "lerobot/extras/episode_000000/states.npz", "sha256": "0" * 64, "size": 1, "tar_offset": 0}]}))
    with pytest.raises(ValueError, match="differ from the extraction manifest"):
        next(read_robocasa(ds, root=tmp_path, catalog={}, tars={}))


def test_observation_check_skipped_without_state(tmp_path):
    ds = make_dataset(tmp_path, moving=False, with_obs=False)
    (ep,) = read_robocasa(ds, root=tmp_path, catalog={}, tars={})
    assert ep.provenance["observation_check"]["compared"] is False and ep.success is True


# ---------------------------------------------------------------- acquisition

class _Resp(io.BytesIO):
    def __init__(self, data, status, headers, fail_after=None):
        super().__init__(data)
        self.status, self.headers, self.fail_after = status, headers, fail_after

    def readinto(self, b):
        if self.fail_after is not None and self.tell() >= self.fail_after:
            raise ConnectionResetError("dropped")
        if self.fail_after is not None:
            b = memoryview(b)[:max(1, self.fail_after - self.tell())]
        return super().readinto(b)

    def geturl(self):
        return "https://example.test/x"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def range_opener(blob, log, drop_first_at=None):
    state = {"dropped": False}

    def open_url(req):
        rng = req.get_header("Range")
        log.append(rng)
        if rng:
            a, b = re.match(r"bytes=(\d+)-(\d*)", rng).groups()
            a, b = int(a), int(b) if b else len(blob) - 1
            return _Resp(blob[a:b + 1], 206, {"Content-Range": f"bytes {a}-{b}/{len(blob)}"})
        fail = None
        if drop_first_at is not None and not state["dropped"]:
            state["dropped"], fail = True, drop_first_at
        return _Resp(blob, 200, {"Content-Length": str(len(blob))}, fail)
    return open_url


def toy_tar() -> bytes:
    buf = io.BytesIO()
    members = {"lerobot/meta/info.json": b'{"fps": 20}',
               "lerobot/videos/chunk-000/robot0_eye_in_hand/episode_000000.mp4": b"\x00" * 40000,
               "lerobot/data/chunk-000/episode_000000.parquet": b"PAR1" + b"x" * 3000,
               "lerobot/extras/episode_000000/ep_meta.json": b'{"lang": "a"}',
               "lerobot/extras/episode_000001/ep_meta.json": b'{"lang": "b"}',
               "README.md": b"readme"}
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for name, data in members.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def tar_entry(blob, **kw):
    path = "v1.0/pretrain/atomic/Toy/20250101/lerobot.tar"
    return CatalogEntry(**{**dict(
        id=f"robocasa/{path}", family="robocasa", path=path, url="https://example.test/x.tar", revision="c",
        sha256=None, size=len(blob), license="CC-BY-4.0", kind="tar_stream", content="low_dim",
        dataset="robocasa/v1.0/pretrain/atomic/Toy/20250101", digests={"sha1": hashlib.sha1(blob).hexdigest()},
        meta=dict(task="Toy", split="pretrain", task_type="atomic", source="human", variant="human_path",
                  shared_link="s", box_file_id="1", box_file_version="2")), **kw})


def test_fetch_tar_keeps_only_non_image_members_and_resumes(tmp_path):
    blob = toy_tar()
    e = tar_entry(blob)
    log = []
    (rec,) = fetch([e], tmp_path, opener=range_opener(blob, log, drop_first_at=20000), reserve=0)
    assert log[0] is None and log[1] == "bytes=20000-"           # resumed after the drop
    d = e.output_dir(tmp_path)
    assert not list(d.rglob("*.mp4")) and (d / "lerobot/data/chunk-000/episode_000000.parquet").exists()
    man = json.loads((d / TAR_MANIFEST).read_text())
    assert man["tar_sha1_verified"] and man["dropped_image_members"] == {"members": 1, "bytes": 40000}
    assert man["box_sha1"] == e.digests["sha1"] and man["box_file_id"] == "1"  # earlier manifest layout kept
    assert {m["member"] for m in man["members"]} == {
        "lerobot/meta/info.json", "lerobot/data/chunk-000/episode_000000.parquet",
        "lerobot/extras/episode_000000/ep_meta.json", "lerobot/extras/episode_000001/ep_meta.json", "README.md"}
    for m in man["members"]:
        assert blob[m["tar_offset"]:m["tar_offset"] + m["size"]] == (d / m["member"]).read_bytes()
    assert rec["stripped"]["box_sha1_verified"] == e.digests["sha1"] and rec["kind"] == "tar_stream"
    assert (tmp_path / "raw" / "ledger" / f"{e.id}.json").exists()
    # A second call is satisfied by the verified manifest without any transfer.
    log2 = []
    fetch([e], tmp_path, opener=range_opener(blob, log2), reserve=0)
    assert log2 == []


def test_fetch_tar_max_episodes_and_sha1_mismatch(tmp_path):
    blob = toy_tar()
    fetch([tar_entry(blob)], tmp_path, opener=range_opener(blob, []), reserve=0, max_episodes=1)
    d = tar_entry(blob).output_dir(tmp_path)
    assert (d / "lerobot/extras/episode_000000/ep_meta.json").exists()
    assert not (d / "lerobot/extras/episode_000001").exists()
    bad = tar_entry(blob, digests={"sha1": "0" * 40}, path="v1.0/pretrain/atomic/Bad/20250101/lerobot.tar",
                    id="robocasa/v1.0/pretrain/atomic/Bad/20250101/lerobot.tar")
    with pytest.raises(ChecksumMismatch):
        fetch([bad], tmp_path, opener=range_opener(blob, []), reserve=0)
    assert not list(bad.output_dir(tmp_path).rglob("*.json"))


def test_fetch_tar_refuses_without_disk(tmp_path):
    blob = toy_tar()
    with pytest.raises(InsufficientDisk):
        fetch([tar_entry(blob)], tmp_path, opener=range_opener(blob, []), reserve=10 ** 18)


def test_fetch_asset_subset_reads_members_by_range(tmp_path):
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("objaverse/cube/model.stl", tetra_stl())
        zf.writestr("objaverse/other/model.stl", b"y" * 100)
    blob = zbuf.getvalue()
    e = CatalogEntry(id="robocasa/assets/objaverse.zip", family="robocasa", path="assets/objaverse.zip",
                     url="https://example.test/o.zip", revision="r", sha256=hashlib.sha256(blob).hexdigest(),
                     size=len(blob), license="CC-BY-4.0", kind="file", content="assets",
                     asset_marker="robocasa/models/assets/objects/")
    log = []
    man = fetch_robocasa_asset_subset(["objects/objaverse/cube/model.stl", "textures/none.png"], tmp_path,
                                      catalog={e.id: e}, opener=range_opener(blob, log), reserve=0)
    assert all(r and r.startswith("bytes=") for r in log)       # never a whole-file request
    got = tmp_path / "raw" / "robocasa" / "asset_subset" / "objects" / "objaverse" / "cube" / "model.stl"
    assert got.read_bytes() == tetra_stl()
    assert man["members"]["objects/objaverse/cube/model.stl"]["archive_id"] == e.id
    assert man["missing"] == ["textures/none.png"]


def test_helpers():
    assert gen.static_url("https://utexas.box.com/s/abc123", "tar") == "https://utexas.box.com/shared/static/abc123.tar"
    assert is_image_name("lerobot/videos/chunk-000/x/episode_000000.mp4")
    assert is_image_name("foo/bar.PNG") and not is_image_name("lerobot/extras/episode_0/states.npz")
    assert gen.zip_marker("assets/objaverse.zip") == "robocasa/models/assets/objects/"
    assert gen.zip_marker("assets/textures.zip") == "robocasa/models/assets/"
    reg = gen.parse_registry('X = OrderedDict(\n    Foo=dict(\n        pretrain=dict(\n'
                             '            mg_path="v1.0/pretrain/atomic/Foo/1/mg/demo/2",\n'
                             '            human_path="v1.0/pretrain/atomic/Foo/1",\n        ),\n'
                             '        horizon=450,\n    ),\n)')
    assert reg == {"pretrain/atomic/Foo/1/mg/demo/2/lerobot.tar": ("Foo", "mg_path", 450),
                   "pretrain/atomic/Foo/1/lerobot.tar": ("Foo", "human_path", 450)}
    assets = RobocasaAssets([])
    assert assets.rel("/root/robocasa/robocasa/models/assets/fixtures/../textures/a.png") == "textures__a.png"
    assert assets.rel("/x/robosuite/models/assets/a.stl") is None


def test_catalog_tars_are_pinned():
    tars = {k: e for k, e in load_catalog().items() if e.family == "robocasa" and e.kind == "tar_stream"}
    assert len(tars) == 410
    for e in tars.values():
        assert re.fullmatch(r"[0-9a-f]{40}", e.digests["sha1"]) and e.size > 0 and e.license == "CC-BY-4.0"
        assert e.url.startswith("https://utexas.box.com/shared/static/") and e.url.endswith(".tar")
        assert e.dataset == "robocasa/" + e.path.rsplit("/", 1)[0]
        mg = e.meta["source"] == "mg"
        assert mg == ("/mg/" in e.path) and ("seed" in e.meta) == mg
        assert e.usable is not mg  # MimicGen tars carry no simulator extras: catalogued, never selected by prefix
    human = [e for e in tars.values() if e.usable]
    assert len(human) == 350 and {(e.meta["split"], e.meta["task_type"]) for e in human} == {
        ("pretrain", "atomic"), ("pretrain", "composite"), ("target", "atomic"), ("target", "composite")}
