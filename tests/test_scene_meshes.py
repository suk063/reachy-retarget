"""Scene components, the asset library and the episode reader (offline, synthetic MuJoCo scene)."""
import json

import mujoco
import numpy as np
import pytest

from _robosuite_fixtures import placeholder
from reachy_retarget.meshes.components import NoMeshes, build_scene_components
from reachy_retarget.meshes.library import AssetLibrary
from reachy_retarget.meshes.reachy_links import link_poses, reachy_template
from reachy_retarget.robot import Reachy
from reachy_retarget.schema import rotations as rot
from reachy_retarget.schema.episode import DT, SIDES, ReachyEpisode
from reachy_retarget.schema.io import read_episode, write_episode
from reachy_retarget.schema.scene_assets import (decode_msh, encode_msh, load_component_meshes, primitive_mesh,
                                                 read_scene, scene_mjcf, surface_area)
from reachy_retarget.schema.source import Articulation, Effector, ObjectTrack, SceneRef, SourceEpisode

CUBE_OBJ = b"""v -1 -1 -1\nv 1 -1 -1\nv 1 1 -1\nv -1 1 -1\nv -1 -1 1\nv 1 -1 1\nv 1 1 1\nv -1 1 1
vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1
f 1/1 3/3 2/2\nf 1/1 4/4 3/3\nf 5/1 6/2 7/3\nf 5/1 7/3 8/4\nf 1/1 2/2 6/3\nf 1/1 6/3 5/4\nf 2/1 3/2 7/3\nf 2/1 7/3 6/4
f 3/1 4/2 8/3\nf 3/1 8/3 7/4\nf 4/1 1/2 5/3\nf 4/1 5/3 8/4\n"""
PNG = placeholder(".png")
TABLE = np.array([0.6, -0.1, 0.35])
BOX0 = np.array([0.5, -0.2, 0.75])


def scene_xml(deco_file="/rec/robosuite/models/assets/objects/meshes/deco.obj"):
    return f"""<mujoco model="synthetic"><compiler angle="radian"/>
<asset><texture name="wood" type="2d" file="/rec/robosuite/models/assets/textures/wood.png"/>
<texture name="checker" type="2d" builtin="checker" rgb1="1 1 1" rgb2="0 0 0" width="32" height="32"/>
<material name="wood_mat" texture="wood" texrepeat="2 2" specular="0.3"/>
<material name="checker_mat" texture="checker"/>
<mesh name="leg" file="/rec/robosuite/models/assets/objects/meshes/leg.obj" scale="0.02 0.02 0.1"/>
<mesh name="deco" file="{deco_file}" scale="0.05 0.05 0.05"/>
<mesh name="robot0_link_mesh" file="/rec/robosuite/models/assets/robots/link0.stl"/></asset>
<worldbody><geom name="floor" type="plane" size="3 3 .1" material="checker_mat"/>
<body name="table" pos="{TABLE[0]} {TABLE[1]} {TABLE[2]}">
  <geom name="table_collision" type="box" size="0.2 0.3 0.35" group="0"/>
  <geom name="table_visual" type="box" size="0.2 0.3 0.35" group="1" contype="0" conaffinity="0" material="wood_mat"/>
  <geom name="table_leg" type="mesh" mesh="leg" pos="0.1 0.2 -0.2" quat="0.7071068 0 0 0.7071068" group="1"
        contype="0" conaffinity="0" rgba="0.2 0.2 0.2 1"/>
  <body name="table_drawer" pos="0.2 0 0.1"><joint name="drawer_slide" type="slide" axis="1 0 0" range="0 0.2"/>
    <geom name="drawer_box" type="box" size="0.02 0.1 0.05" rgba="0 0 1 1"/></body></body>
<body name="box_main" pos="{BOX0[0]} {BOX0[1]} {BOX0[2]}"><freejoint name="box_joint"/>
  <geom name="box_collision" type="box" size="0.02 0.02 0.05" mass="0.1" group="0"/>
  <geom name="box_visual" type="mesh" mesh="deco" group="1" contype="0" conaffinity="0" material="wood_mat"/>
  <body name="box_handle" pos="0 0 0.06"><geom name="handle_visual" type="capsule" size="0.005 0.02" group="1"
        contype="0" conaffinity="0" rgba="1 0 0 1"/></body></body>
<body name="far_crate" pos="9 9 0.2"><geom name="crate" type="box" size="0.1 0.1 0.1"/></body>
<body name="robot0_base" pos="-0.5 0 0.9"><joint name="robot0_j" type="hinge"/>
  <geom type="mesh" mesh="robot0_link_mesh" mass="1"/></body>
</worldbody></mujoco>"""


ASSETS = {"objects/meshes/leg.obj": CUBE_OBJ, "objects/meshes/deco.obj": CUBE_OBJ, "textures/wood.png": PNG,
          "robots/link0.stl": placeholder(".stl")}


def episode_pair(T=30, assets=ASSETS, scene=True):
    """(ReachyEpisode, SourceEpisode): the box slides 10 cm in x and turns, the drawer opens."""
    t = np.arange(T) * DT
    q = np.zeros((T, 22))
    q[:, 0] = -0.3
    fkb = Reachy.load().fk_base(q)
    s = np.linspace(0, 1, T)
    box = np.c_[BOX0 + np.outer(s, [0.1, 0, 0]), rot.matrix_to_quat(rot.so3_exp(np.outer(s, [0, 0, 0.5])))]
    table = np.tile([*TABLE, 1, 0, 0, 0], (T, 1))
    drawer = 0.15 * s[:, None]
    objects = {"box": ObjectTrack(box, np.ones(T, bool), "manipulated", {"body": "box_main"}),
               "table": ObjectTrack(table, np.ones(T, bool), "support", {"body": "table"})}
    ep = ReachyEpisode(
        family="synthetic", dataset="synthetic/scene", episode_id="0", task="t", time=t, q=q, qd=np.zeros_like(q),
        tcp_base={k: fkb[f"{k}_tcp"] for k in SIDES}, head_base=fkb["head"],
        gripper_opening=np.ones((T, 2)), gripper_width=np.full((T, 2), 0.05), source_time=t,
        tier={"K": {"passed": True, "reasons": []}, "P": None}, retarget_config="test", objects=objects,
        articulations={"table": Articulation(["drawer_slide"], drawer)})
    ref = SceneRef(mjcf=scene_xml(), robot_prefixes=["robot0_"], initial_qpos={"box_joint": [*BOX0, 1, 0, 0, 0]},
                   assets=dict(assets)) if scene else None
    src = SourceEpisode(family="synthetic", dataset="synthetic/scene", episode_id="0", task="t", time=t,
                        effectors={"g": Effector(np.tile(np.eye(4), (T, 1, 1)), np.ones(T))}, objects=objects,
                        articulations={"table": Articulation(["drawer_slide"], drawer)}, scene=ref,
                        scene_qpos=Articulation([f"box_joint/{c}" for c in ("x", "y", "z", "qw", "qx", "qy", "qz")]
                                                + ["drawer_slide"], np.c_[box, drawer]),
                        provenance={"state_route": "mjcf_states" if scene else "mjcf_kinematic_only"})
    return ep, src


def test_msh_round_trip_and_mujoco_reads_it():
    v, n, f = primitive_mesh("box", [0.1, 0.2, 0.3])
    uv = np.c_[np.linspace(0, 1, len(v)), np.linspace(1, 0, len(v))]
    data = encode_msh(v, f, n, uv)
    m = decode_msh(data)
    np.testing.assert_allclose(m["vertices"], v, atol=1e-7)
    np.testing.assert_array_equal(m["faces"], f)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><asset><mesh name="m" file="a.msh"/></asset><worldbody><geom type="mesh" mesh="m"/></worldbody></mujoco>',
        {"a.msh": data})
    np.testing.assert_allclose(model.mesh_texcoord, uv, atol=1e-7)  # used as stored (no v flip)


@pytest.mark.parametrize("kind,size,area", [
    ("box", [0.1, 0.2, 0.3], 8 * (0.1 * 0.2 + 0.2 * 0.3 + 0.1 * 0.3)), ("sphere", [0.1], 4 * np.pi * 0.01),
    ("cylinder", [0.1, 0.2], 2 * np.pi * 0.1 * 0.4 + 2 * np.pi * 0.01),
    ("capsule", [0.1, 0.2], 2 * np.pi * 0.1 * 0.4 + 4 * np.pi * 0.01), ("plane", [1, 2, 0], 8.0)])
def test_primitive_meshes_are_closed_outward_and_sized(kind, size, area):
    v, n, f = primitive_mesh(kind, size)
    assert surface_area(v, f) == pytest.approx(area, rel=0.01)
    tri = v[f]
    if kind != "plane":
        assert np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() > 0  # outward faces
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1, atol=1e-9)


def test_library_is_content_addressed_and_never_overwrites(tmp_path):
    lib = AssetLibrary(tmp_path / "assets")
    a = lib.put(b"mesh bytes", "msh")
    b = AssetLibrary(tmp_path / "assets").put(b"mesh bytes", "msh")   # another writer, same content
    assert a == b and lib.new_files == 1 and a["bytes"] == 10
    path = lib.path(a)
    assert path.name == f"{a['sha256']}.msh" and path.read_bytes() == b"mesh bytes"
    assert lib.put(PNG, name="x.png")["format"] == "png"
    assert sorted(p.suffix for p in (tmp_path / "assets").iterdir()) == [".msh", ".png"]  # no temporary left
    path.write_bytes(b"corrupted, longer content")
    with pytest.raises(RuntimeError):
        AssetLibrary(tmp_path / "assets").put(b"mesh bytes", "msh")


def test_scene_components_round_trip_and_rebuild(tmp_path):
    ep, src = episode_pair()
    lib = AssetLibrary(tmp_path / "out" / "assets")
    scene, stats = build_scene_components(ep, src, lib)
    names = scene.names
    assert {"box", "table", "table_drawer", "worldbody"} <= set(names) and "far_crate" not in names
    assert {o["component"] for o in scene.info["omitted"]} == {"far_crate"}
    assert sum(n.startswith("reachy/") for n in names) == len(reachy_template()[0])
    comp = {c["name"]: c for c in scene.components}
    assert comp["box"]["pose_source"] == "object_track" and comp["table_drawer"]["role"] == "articulated_part"
    assert comp["table_drawer"]["articulation"] == "table" and comp["table_drawer"]["visual_from_collision"]
    assert [p["geom"] for p in comp["box"]["visual"]] == ["box_visual", "handle_visual"]   # nested body merged
    assert [p["geom"] for p in comp["box"]["collision"]] == ["box_collision"]
    assert [p["geom"] for p in comp["worldbody"]["visual"]] == ["floor"]
    assert scene.textures["checker"]["builtin"] == "checker" and scene.textures["checker"]["files"] == {}
    assert stats["assets_new"] == stats["assets_referenced"] > 0

    out = tmp_path / "out" / "episodes" / "synthetic" / "0.h5"
    out.parent.mkdir(parents=True)
    scene.info["library"] = lib.relative_to(out.parent)
    ep.scene = scene
    write_episode(out, ep)
    back = read_episode(out)
    np.testing.assert_allclose(back.scene.poses, scene.poses.astype(np.float32), atol=0)
    assert back.scene.components == json.loads(json.dumps(scene.components))
    sc, library = read_scene(out)
    assert library == (tmp_path / "out" / "assets").resolve()

    # poses: the box follows its track, the drawer opens along +x of the table, statics stay
    i = names.index("box")
    np.testing.assert_allclose(sc.poses[:, i], ep.objects["box"].pose, atol=1e-6)
    j = names.index("table_drawer")
    np.testing.assert_allclose(sc.poses[:, j, 0] - sc.poses[0, j, 0], ep.articulations["table"].qpos[:, 0], atol=1e-6)
    np.testing.assert_allclose(scene.info["track_vs_scene_state"]["box"]["max_position_m"], 0, atol=1e-9)

    # meshes against the source: the leg is the 2 m OBJ cube scaled (0.02, 0.02, 0.1) about its centre,
    # rotated 90 deg about z and placed at (0.1, 0.2, -0.2) in the table frame
    parts = load_component_meshes(out, kind="visual")
    leg = next(p for p, s in zip(parts["table"], comp["table"]["visual"]) if s["geom"] == "table_leg")
    assert surface_area(leg["vertices"], leg["faces"]) == pytest.approx(2 * 0.04 * 0.04 + 4 * 0.04 * 0.2,
                                                                        rel=1e-6)
    np.testing.assert_allclose(leg["vertices"].min(0), [0.1 - 0.02, 0.2 - 0.02, -0.2 - 0.1], atol=1e-6)
    np.testing.assert_allclose(leg["vertices"].max(0), [0.1 + 0.02, 0.2 + 0.02, -0.2 + 0.1], atol=1e-6)
    assert leg["rgba"] == [0.2, 0.2, 0.2, 1.0] and leg["uv"] is not None
    deco = parts["box"][0]
    assert deco["texture"]["path"].endswith(".png") and open(deco["texture"]["path"], "rb").read() == PNG
    assert deco["material"]["texrepeat"] == [2.0, 2.0] and deco["material"]["specular"] == pytest.approx(0.3)
    top = next(p for p in parts["table"] if p["source"] == "table_visual")
    assert surface_area(top["vertices"], top["faces"]) == pytest.approx(8 * (.2 * .3 + .3 * .35 + .2 * .35))

    # the rebuilt MJCF compiles and puts every visual geom where the source scene has it
    for t in (0, len(ep.time) - 1):
        m = mujoco.MjModel.from_xml_string(scene_mjcf(out, frame=t))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        ref = mujoco.MjModel.from_xml_string(*_source_model())
        rd = mujoco.MjData(ref)
        rd.qpos[ref.joint("box_joint").qposadr[0]:][:7] = ep.objects["box"].pose[t]
        rd.qpos[ref.joint("drawer_slide").qposadr[0]] = ep.articulations["table"].qpos[t, 0]
        mujoco.mj_forward(ref, rd)
        for c in ("box", "table", "table_drawer"):
            for k, p in enumerate(comp[c]["visual"]):
                g = m.geom(f"{c}/visual{k}").id
                np.testing.assert_allclose(d.geom_xpos[g], rd.geom_xpos[ref.geom(p["geom"]).id], atol=1e-6)
                np.testing.assert_allclose(d.geom_xmat[g], rd.geom_xmat[ref.geom(p["geom"]).id], atol=1e-5)
    # component-local rebuild (frame=None) of one component, as a per-component map renderer needs
    m = mujoco.MjModel.from_xml_string(scene_mjcf(out, frame=None, components=["box"], kinds=("visual",)))
    assert m.nbody == 2 and m.ngeom == 2 and np.allclose(m.body_pos[1], 0)
    # Reachy link poses are its forward kinematics
    k = names.index("reachy/head")
    np.testing.assert_allclose(rot.vec7_to_pose(sc.poses[:, k]), Reachy.load().fk(ep.q)["head"], atol=1e-6)


def _source_model():
    from reachy_retarget.validate.scene import _resolve_assets
    import xml.etree.ElementTree as ET
    root = ET.fromstring(scene_xml())
    vfs, _, _ = _resolve_assets(root, ASSETS, None, [])
    return ET.tostring(root, encoding="unicode"), vfs


def test_episodes_without_meshes_are_excluded(tmp_path):
    ep, src = episode_pair(scene=False)
    with pytest.raises(NoMeshes, match="no source scene"):
        build_scene_components(ep, src, AssetLibrary(tmp_path))
    ep, src = episode_pair(assets={k: v for k, v in ASSETS.items() if "deco" not in k})
    with pytest.raises(NoMeshes, match="visual assets missing"):
        build_scene_components(ep, src, AssetLibrary(tmp_path))
    from reachy_retarget.evaluate import process_source
    _, src = episode_pair(scene=False)
    rec = process_source(src, physics=False, write=str(tmp_path / "episodes"))
    assert rec["status"] == "excluded" and rec["excluded"] == "no_meshes" and not (tmp_path / "episodes").exists()


def test_physics_poses_follow_the_rollout(tmp_path):
    from reachy_retarget.robot.reachy import JOINTS
    from reachy_retarget.schema.episode import PhysicsRollout
    ep, src = episode_pair(T=10)
    N = 12
    names = ([f"box_joint/{c}" for c in ("x", "y", "z", "qw", "qx", "qy", "qz")] + ["drawer_slide"]
             + [f"reachy/{j}" for j in JOINTS])
    qpos = np.zeros((N, len(names)))
    qpos[:, :3] = BOX0 + np.outer(np.arange(N), [0, 0.01, 0])
    qpos[:, 3] = 1
    qpos[:, 7] = 0.05
    qpos[:, 8 + 2] = np.linspace(0, 1, N)       # base_yaw
    roll = PhysicsRollout(np.arange(N) * DT, qpos, np.zeros((N, 1)), np.zeros((N, 22)), names, ["v"],
                          [f"reachy/{j}" for j in JOINTS], info={"reachy_prefix": "reachy/"})
    scene, _ = build_scene_components(ep, src, AssetLibrary(tmp_path), rollout=roll)
    P = scene.physics_poses
    assert P.shape == (N, len(scene.names), 7)
    np.testing.assert_allclose(P[:, scene.names.index("box"), :3], qpos[:, :3], atol=1e-9)
    base = P[:, scene.names.index("reachy/base_link")]
    np.testing.assert_allclose(2 * np.arctan2(base[:, 6], base[:, 3]), np.linspace(0, 1, N), atol=1e-9)


def test_reachy_links_cover_the_urdf_visuals():
    comps, materials, files = reachy_template()
    links = {c["source_body"] for c in comps}
    assert {"base_link", "torso", "head", "l_hand_palm_link", "r_hand_distal_link", "antenna_left_link"} <= links
    for c in comps:
        for p in c["visual"]:
            assert p["material"] is None or p["material"] in materials
            data = files[(p.get("mesh") or p.get("generated_mesh"))[1]]
            assert len(decode_msh(data)["vertices"]) >= 4
    q = np.zeros((2, 22))
    q[:, 0], q[:, 2] = 1.0, 0.5
    P = link_poses(q, ["base_link", "torso"])
    np.testing.assert_allclose(rot.vec7_to_pose(P[:, 1]), Reachy.load().fk(q)["torso"], atol=1e-9)
    np.testing.assert_allclose(P[0, 0, :3], [1, 0, 0])


def test_family_mesh_policy(tmp_path):
    from reachy_retarget.build import build
    from reachy_retarget.sources import MESHES, require_meshes
    for fam in ("robomimic", "mimicgen", "libero", "dexmimicgen", "robocasa", "bigym", "maniskill"):
        require_meshes(fam)
    assert MESHES["roboverse"][0] == MESHES["mobilemanibench"][0] == "pending"
    for fam in ("behavior", "molmobot", "roboverse", "nope"):
        with pytest.raises(ValueError, match="scene meshes"):
            build(fam, str(tmp_path / "missing.h5"), (0, 1), str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_report_counts_excluded():
    from reachy_retarget.report import markdown, summarize
    recs = [{"family": "f", "dataset": "d", "status": "excluded", "excluded": "no_meshes", "K": None, "P": None,
             "reason": "source scene assets missing (25 files, state_route mjcf_kinematic_only)"},
            {"family": "f", "dataset": "d", "status": "ok", "K": {"passed": True, "reasons": []}, "P": None,
             "file": "x.h5", "scene": {"episode_bytes": 2_000_000, "assets_referenced_bytes": 10_000_000,
                                       "assets_new_bytes": 4_000_000}}]
    s = summarize(recs)
    assert s["totals"]["excluded"] == 1 and s["totals"]["written"] == 1 and s["totals"]["retargeted"] == 1
    assert s["storage_mb_per_episode"]["d"] == {"episodes": 1, "episode_only": 2.0, "with_library_share": 6.0,
                                                "with_own_assets": 12.0}
    assert "no_meshes: source scene assets missing (# files" in markdown(s)
