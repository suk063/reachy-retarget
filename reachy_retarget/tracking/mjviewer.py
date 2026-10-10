"""mjviser replay of episodes with their scenes on the MuJoCo model of Reachy.

The robot is the MuJoCo model of tier P (:func:`reachy_retarget.robot.mjcf.reachy_mjcf`) rendered by
mjviser, as reachy-control's mjlab tooling does, plus display-only visual geoms: the URDF visual
meshes (COLLADA, one mesh per material colour, group 2, no contact) and the tripod extensions of
:func:`.visual.tripod_extensions`.

Episodes with a ``/scene`` (manipulation builds, ``reachy_retarget.build``) add their scene
components from the asset library (:func:`reachy_retarget.schema.scene_assets.scene_mjcf`): every
component except the Reachy links (the robot is drawn from q) becomes a mocap body whose pose is
set per frame from ``/scene/poses`` (kinematic) or ``/scene/physics_poses`` with the robot from
``/physics/qpos`` (tier-P rollout, "Poses: physics"). A component without a valid pose at a frame
is moved out of sight. Tracking episodes have no scene. The model is rebuilt only when the scene
components change between episodes. mjviser textures only mesh geoms with texture coordinates, so
textured primitive visual geoms (robosuite tables, cubes, walls, floors) are turned into meshes with
MuJoCo-style texture coordinates first (:func:`uv_textured_primitives`).

Collision geoms (group 3) and frame sites (group 4) start hidden; mjviser's Groups tab shows them
and its Visualization tab adds frames and contacts. The world frame is fixed by default; "Follow
base" switches on mjviser's camera tracking, which shifts the whole world (scene, floor grid and
overlays) by minus the base position each frame so that the robot stays at the origin.

Each frame writes q into qpos (mimic joints included) and runs ``mj_kinematics``; nothing is
simulated. A kinematic replay is asserted to reproduce the episode's stored TCPs (1e-5 m).

    python -m reachy_retarget.tracking.mjviewer --run runs/tracking/pilot [--port 8088]
    python -m reachy_retarget.tracking.mjviewer --run runs/lift

``--run`` is a build or tracking run directory, or a dataset folder copied from the PVC (episode
paths written on a pod resolve below its ``episodes/``). Requires the ``viz`` extra (viser,
trimesh, pycollada, mjviser).
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..robot.mjcf import reachy_mjcf, set_reachy_state
from ..robot.reachy import JOINTS
from ..robot.urdf import pose
from ..schema.rotations import matrix_to_quat
from .viewer import (
    BASE_COLOR,
    GAZE_COLOR,
    LEFT_ACTUAL,
    LEFT_GOAL,
    RIGHT_ACTUAL,
    RIGHT_GOAL,
    _resolve,
    load_catalog,
)
from .visual import URDF, mesh_path, tripod_extensions

VISUAL_GROUP = 2
DEFAULT_RGBA = (.72, .75, .79, 1.)
ROBOT_ROLE = "robot_link"
SCENE_PREFIX = "scene/"
HIDDEN = (0., 0., -100.)  # mocap position of a component without a valid pose at a frame
SIDES = ("left", "right")
TIPS = ("l_arm_tip", "r_arm_tip")


def _origin(e):
    o = e.find("origin")
    xyz = np.fromstring(o.get("xyz", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
    rpy = np.fromstring(o.get("rpy", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
    return pose(xyz, rpy)


@functools.cache
def mesh_parts(path) -> tuple:
    """((vertices (N, 3), faces (M, 3), rgba (4,)), ...) of a COLLADA file, one entry per colour."""
    import trimesh
    by_color = {}
    for m in trimesh.load_scene(path, process=False).dump():
        material = getattr(m.visual, "material", None)
        color = getattr(material, "main_color", None)
        rgba = tuple(np.asarray(color) / 255.) if color is not None else DEFAULT_RGBA
        by_color.setdefault(rgba, []).append(m)
    out = []
    for rgba, meshes in by_color.items():
        m = trimesh.util.concatenate(meshes)
        out.append((np.asarray(m.vertices, float), np.asarray(m.faces, np.int32), np.array(rgba)))
    return tuple(out)


def display_spec():
    """``MjSpec`` of Reachy's tier-P model plus the URDF visual geometry (module docstring)."""
    import mujoco
    xml, assets = reachy_mjcf()
    spec = mujoco.MjSpec.from_string(xml, assets=assets)
    root = ET.parse(URDF).getroot()
    materials = {e.get("name"): e.find("color") for e in root.findall("material")}
    meshes = {}
    for link in root.findall("link"):
        name = link.get("name")
        body = spec.body(name)
        for i, visual in enumerate(link.findall("visual")):
            T = _origin(visual)
            place = {"pos": T[:3, 3], "quat": matrix_to_quat(T[:3, :3])}
            common = {"contype": 0, "conaffinity": 0, "group": VISUAL_GROUP, "mass": 0}
            shape = visual.find("geometry")[0]
            if shape.tag == "mesh":
                scale = np.fromstring(shape.get("scale", "1 1 1"), sep=" ")
                path = mesh_path(shape.get("filename"))
                for k, (v, f, rgba) in enumerate(mesh_parts(path)):
                    key = (path.stem, k, tuple(scale))
                    if key not in meshes:
                        meshes[key] = f"visual_{path.stem}_{k}_{len(meshes)}"
                        spec.add_mesh(name=meshes[key], uservert=(v * scale).ravel(), userface=f.ravel(),
                                      inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
                    body.add_geom(name=f"{name}_v{i}_{k}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=meshes[key],
                                  rgba=rgba, **place, **common)
                continue
            material = visual.find("material")
            color = None
            if material is not None:
                color = material.find("color")
                if color is None:
                    color = materials.get(material.get("name"))
            rgba = np.fromstring(color.get("rgba"), sep=" ") if color is not None else np.array(DEFAULT_RGBA)
            if shape.tag == "box":
                geom = {"type": mujoco.mjtGeom.mjGEOM_BOX, "size": np.fromstring(shape.get("size"), sep=" ") / 2}
            elif shape.tag == "cylinder":
                geom = {"type": mujoco.mjtGeom.mjGEOM_CYLINDER,
                        "size": [float(shape.get("radius")), float(shape.get("length")) / 2, 0]}
            elif shape.tag == "sphere":
                geom = {"type": mujoco.mjtGeom.mjGEOM_SPHERE, "size": [float(shape.get("radius")), 0, 0]}
            else:
                raise ValueError(f"Unsupported visual shape: {shape.tag}")
            body.add_geom(name=f"{name}_v{i}", rgba=rgba, **geom, **place, **common)
            if i == 0 and name in tripod_extensions():
                size, P = tripod_extensions()[name]
                body.add_geom(name=f"{name}_tripod_extension", type=mujoco.mjtGeom.mjGEOM_BOX, size=size / 2,
                              pos=P[:3, 3], quat=matrix_to_quat(P[:3, :3]), rgba=rgba, **common)
    return spec


def display_model():
    return display_spec().compile()


# OpenGL cube-map faces in MuJoCo's storage order (+X, -X, +Y, -Y, +Z, -Z): (s axis, s sign, t axis, t sign)
CUBE_FACES = ((2, -1, 1, -1), (2, 1, 1, -1), (0, 1, 2, 1), (0, 1, 2, -1), (0, 1, 1, -1), (0, -1, 1, -1))


def primitive_uv(kind: str, size, cube: bool, texrepeat=(1., 1.), texuniform: bool = False):
    """``(vertices (3n, 3), faces (n, 3), uv (3n, 2))``: the surface of a MuJoCo primitive
    (:func:`reachy_retarget.schema.scene_assets.primitive_mesh`), one vertex per triangle corner, with
    texture coordinates (MuJoCo convention: v = 0 at the first texture row).

    Each triangle is projected along its dominant normal axis. ``cube``: into a 2-D atlas of the six
    cube faces stacked in storage order, with the OpenGL cube-map face axes over the geom normalized to
    [-1, 1]; else a 2-D texture repeated ``texrepeat`` times per metre (``texuniform``) or across the
    geom (MuJoCo's material attributes)."""
    from ..schema.scene_assets import primitive_mesh
    v, _, f = primitive_mesh(kind, size)
    V = np.asarray(v, float)[np.asarray(f)].reshape(-1, 3)
    F = np.arange(len(V)).reshape(-1, 3)
    n = np.cross(V[1::3] - V[0::3], V[2::3] - V[0::3])
    axis = np.abs(n).argmax(1)
    half = np.abs(V).max(0)
    q = V / np.where(half > 0, half, 1.)
    rows, uv = np.arange(len(V)), np.empty((len(V), 2))
    if cube:
        c = q.reshape(-1, 3, 3).mean(1)[np.arange(len(F)), axis]
        sign = np.where(np.abs(c) > 1e-9, np.sign(c), np.sign(n[np.arange(len(F)), axis]))
        face = np.repeat(2 * axis + (sign < 0), 3)
        for k, (sa, ss, ta, ts) in enumerate(CUBE_FACES):
            sel = face == k
            uv[sel, 0] = np.clip((ss * q[sel, sa] + 1) / 2, 0, 1)
            uv[sel, 1] = (k + np.clip((ts * q[sel, ta] + 1) / 2, 0, 1)) / 6
    else:
        other = np.array([[1, 2], [0, 2], [0, 1]])[np.repeat(axis, 3)]
        coord = V if texuniform else (q + 1) / 2
        uv[:, 0] = coord[rows, other[:, 0]] * texrepeat[0]
        uv[:, 1] = coord[rows, other[:, 1]] * texrepeat[1]
    return V, F, uv


def uv_textured_primitives(spec) -> int:
    """Rewrite every textured primitive visual geom (no contact) of ``spec`` as a mesh geom with
    texture coordinates (:func:`primitive_uv`); returns how many were rewritten.

    mjviser textures only mesh geoms with texture coordinates and draws primitives in the flat
    material colour, which is white for most textured materials. A cube texture is re-added as a 2-D
    atlas (the six faces as MuJoCo stores them) under a copy of the material. Display only: the
    geoms do not collide."""
    import mujoco
    G = mujoco.mjtGeom
    kinds = {G.mjGEOM_PLANE: "plane", G.mjGEOM_SPHERE: "sphere", G.mjGEOM_CAPSULE: "capsule",
             G.mjGEOM_ELLIPSOID: "ellipsoid", G.mjGEOM_CYLINDER: "cylinder", G.mjGEOM_BOX: "box"}
    rgb, rgba = int(mujoco.mjtTextureRole.mjTEXROLE_RGB), int(mujoco.mjtTextureRole.mjTEXROLE_RGBA)
    m = spec.compile()
    atlases, meshes, count = {}, {}, 0
    for g in spec.geoms:
        if g.contype or g.conaffinity or g.type not in kinds or not g.material or not g.name:
            continue
        gid = m.geom(g.name).id
        mat = m.geom_matid[gid]
        tex = m.mat_texid[mat, rgb] if m.mat_texid[mat, rgb] >= 0 else m.mat_texid[mat, rgba]
        cube = tex >= 0 and m.tex_type[tex] == mujoco.mjtTexture.mjTEXTURE_CUBE
        if tex < 0 or not (cube or m.tex_type[tex] == mujoco.mjtTexture.mjTEXTURE_2D):
            continue
        if cube and g.material not in atlases:
            w, h, nc = m.tex_width[tex], m.tex_height[tex], m.tex_nchannel[tex]
            name = f"{g.material}__atlas"
            spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=w, height=h, nchannel=nc,
                             data=m.tex_data[m.tex_adr[tex]:m.tex_adr[tex] + w * h * nc].tolist())
            src = spec.material(g.material)
            copy = spec.add_material(name=name, rgba=src.rgba, specular=src.specular, shininess=src.shininess,
                                     reflectance=src.reflectance, emission=src.emission)
            textures = list(copy.textures)
            textures[rgb] = name
            copy.textures = textures
            atlases[g.material] = name
        key = (kinds[g.type], tuple(np.round(m.geom_size[gid], 9)), cube, tuple(m.mat_texrepeat[mat]),
               bool(m.mat_texuniform[mat]))
        if key not in meshes:
            V, F, uv = primitive_uv(key[0], m.geom_size[gid], cube, m.mat_texrepeat[mat], key[4])
            meshes[key] = f"uv_mesh_{len(meshes)}"
            spec.add_mesh(name=meshes[key], uservert=V.ravel(), userface=F.ravel(), usertexcoord=uv.ravel(),
                          inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        g.type, g.meshname = G.mjGEOM_MESH, meshes[key]
        if cube:
            g.material = atlases[g.material]
        count += 1
    return count


def scene_names(scene) -> list[str]:
    """Scene components drawn as mocap bodies: all but the Reachy links."""
    return [c["name"] for c in scene.components if c["role"] != ROBOT_ROLE]


def scene_key(scene, library) -> str:
    """Digest of what :func:`scene_model` builds from ``scene``: equal keys share one model."""
    comps = [c for c in scene.components if c["role"] != ROBOT_ROLE]
    text = json.dumps([str(library), comps, scene.materials, scene.textures], sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def scene_model(path, scene):
    """:func:`display_spec` plus the scene components of episode ``path`` (:func:`scene_names`) as
    mocap bodies named ``scene/<component>``, compiled."""
    import warnings

    import mujoco

    from ..schema.scene_assets import scene_mjcf
    spec = display_spec()
    names = scene_names(scene)
    if names:
        child = mujoco.MjSpec.from_string(scene_mjcf(path, frame=None, components=names))
        uv_textured_primitives(child)
        for body in child.worldbody.bodies:
            body.mocap = True
        with warnings.catch_warnings():
            # the scene MJCF references library files by absolute path: no asset dictionary
            warnings.filterwarnings("ignore", message="Attaching a child without asset dict")
            spec.attach(child, prefix=SCENE_PREFIX, frame=spec.worldbody.add_frame())
    return spec.compile()


def episode_states(model, ep, scene=None, physics: bool = False) -> dict:
    """Per-frame replay arrays of an episode (``physics``: of its tier-P rollout, on the rollout clock
    shifted to episode time): ``time`` (T,), ``q`` (T, 22), ``qpos`` (T, nq), ``xpos`` (T, nbody, 3),
    ``xmat`` (T, nbody, 3, 3), ``mocap_pos`` (T, nmocap, 3), ``mocap_quat`` (T, nmocap, 4). A kinematic
    replay raises if the TCPs differ from the episode's stored ones."""
    import mujoco
    if physics:
        ph = ep.physics
        prefix = ph.info.get("reachy_prefix", "reachy/")
        q = np.stack([ph.qpos[:, ph.qpos_names.index(prefix + j)] for j in JOINTS], 1)
        time_ = ph.time - ph.info.get("episode_time_offset_s", 0.) + ep.time[0]
        poses = None if scene is None else scene.physics_poses
        valid = None if poses is None else np.isfinite(poses).all(-1)
    else:
        q, time_ = ep.q, ep.time
        poses, valid = (None, None) if scene is None else (scene.poses, scene.valid)
    T = len(q)
    mocap_pos = np.zeros((T, model.nmocap, 3))
    mocap_quat = np.tile([1., 0., 0., 0.], (T, model.nmocap, 1))
    if poses is not None and model.nmocap:
        names = scene_names(scene)
        ids = [model.body(SCENE_PREFIX + n).mocapid[0] for n in names]
        cols = [scene.names.index(n) for n in names]
        ok = valid[:, cols, None]
        mocap_pos[:, ids] = np.where(ok, poses[:, cols, :3], HIDDEN)
        mocap_quat[:, ids] = np.where(ok, poses[:, cols, 3:], (1., 0., 0., 0.))
    data = mujoco.MjData(model)
    qpos = np.empty((T, model.nq))
    xpos = np.empty((T, model.nbody, 3))
    xmat = np.empty((T, model.nbody, 3, 3))
    for t in range(T):
        set_reachy_state(model, data, q[t])
        data.mocap_pos[:], data.mocap_quat[:] = mocap_pos[t], mocap_quat[t]
        mujoco.mj_kinematics(model, data)
        qpos[t], xpos[t], xmat[t] = data.qpos, data.xpos, data.xmat.reshape(-1, 3, 3)
    if not physics:
        world = ep.tcp_world
        for link, side in zip(TIPS, SIDES):
            if not np.allclose(xpos[:, model.body(link).id], world[side][:, :3, 3], atol=1e-5, rtol=0):
                raise ValueError("Recorded motion does not match the MuJoCo model; regenerate the episode.")
    return {"time": time_, "q": q, "qpos": qpos, "xpos": xpos, "xmat": xmat, "mocap_pos": mocap_pos,
            "mocap_quat": mocap_quat}


def moving_components(mocap_pos, tol: float = 1e-4) -> list[np.ndarray]:
    """The visible positions (n, 3) of every scene component that moves during the episode (the
    camera frames the hands, the head and these; static fixtures such as a kitchen are left out)."""
    out = []
    for k in range(mocap_pos.shape[1]):
        p = mocap_pos[:, k]
        p = p[p[:, 2] > HIDDEN[2] / 2]
        if len(p) and np.ptp(p, axis=0).max() > tol:
            out.append(p)
    return out


def load_rows(run: Path) -> list[dict]:
    """Viewer rows of every written episode: tracking records (:func:`.viewer.load_catalog`) and
    build records (``records/<family>/*.jsonl``, one written file per record)."""
    rows = load_catalog(run)
    for p in sorted(run.glob("records/*/*.jsonl")):
        if p.parent.name == "tracking":
            continue
        for line in p.read_text().splitlines():
            rec = json.loads(line) if line.strip() else {}
            if rec.get("file"):
                rows.append({"file": rec["file"], "cell": rec.get("task") or "?", "regime": rec.get("regime") or "?",
                             "K": bool((rec.get("K") or {}).get("passed")), "final": True,
                             "name": f"{rec.get('dataset')}/{rec.get('episode_id')}", "source": rec.get("family")})
    return rows


def run_viewer(run, port=8088, initial=None):
    import viser
    from mjviser import ViserMujocoScene

    from ..schema.io import read_episode
    from ..schema.scene_assets import read_scene
    from .neck import HEAD_TIP

    run = Path(run)
    catalog = load_rows(run)
    if not catalog:
        raise ValueError(f"No written episodes under {run}")
    by_name = {r["name"]: r for r in catalog}
    selected = initial if initial in by_name else next((r["name"] for r in catalog if r["final"]), catalog[0]["name"])
    server = viser.ViserServer(host="127.0.0.1", port=port, label=f"Reachy mjviser · {run.name}")
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(control_layout="floating", control_width="medium", show_logo=False,
                               show_share_button=False, brand_color=(40, 100, 145))
    tabs = server.gui.add_tab_group()
    with tabs.add_tab("Episode", icon=viser.Icon.PLAYER_PLAY):
        title = server.gui.add_markdown("")
        cell = server.gui.add_dropdown("Task / cell", ("all", *sorted({r["cell"] for r in catalog})),
                                       initial_value="all")
        status = server.gui.add_dropdown("Tier K", ("all", "passed", "failed"), initial_value="all")
        attempts = server.gui.add_dropdown("Attempts", ("final", "all"), initial_value="final")
        names = [r["name"] for r in catalog if r["final"]]
        selector = server.gui.add_dropdown("Episode", tuple(names or [selected]), initial_value=selected)
        previous_button = server.gui.add_button("Previous")
        next_button = server.gui.add_button("Next")
        source = server.gui.add_dropdown("Poses", ("kinematic", "physics"), initial_value="kinematic")
        play = server.gui.add_checkbox("Play", True)
        slider = server.gui.add_slider("Time (s)", min=0., max=1., step=.02, initial_value=0.)
        speed = server.gui.add_dropdown("Speed", ("0.25x", ".5x", "1x", "2x", "4x"), initial_value="1x")
        follow = server.gui.add_checkbox("Follow base", False)
        show_paths = server.gui.add_checkbox("Paths", True)
        show_targets = server.gui.add_checkbox("Targets", True)
        fit = server.gui.add_button("Fit camera")
        stats = server.gui.add_markdown("")
        server.gui.add_markdown("Teal/blue: left/right targets. Orange/red: actual hands. Purple: actual base. "
                                "Magenta: gaze (head_tip +X), light: target. Collision geoms: Groups → G3. "
                                "Follow base keeps the robot at the origin and moves the world instead.")

    ui = {"key": None, "scene": None, "model": None, "tabs": []}
    h = {"paths": [], "dynamic": []}
    pending = [selected]
    state = {}

    def build(model):
        """A fresh mjviser scene, its GUI tabs and the overlay nodes for ``model``."""
        server.scene.reset()
        for tab in ui["tabs"]:
            tab.remove()
        scene = ViserMujocoScene(server, model, num_envs=1)
        scene.camera_tracking_enabled = follow.value
        ui.update(scene=scene, model=model, tabs=[])
        for label, icon, make in (("Visualization", viser.Icon.EYE, scene.create_overlay_gui),
                                  ("Groups", viser.Icon.LAYERS_INTERSECT, scene.create_groups_gui)):
            tab = tabs.add_tab(label, icon=icon)
            with tab:
                make()
            ui["tabs"].append(tab)
        # overlays live under one frame that follows mjviser's camera-tracking offset
        h["overlay"] = server.scene.add_frame("/overlay", show_axes=False)
        server.scene.add_grid("/overlay/floor", width=12, height=12, cell_size=.1, section_size=1)
        h["actual"] = [server.scene.add_frame(f"/overlay/hand/{s}/actual", axes_length=.075, axes_radius=.002)
                       for s in SIDES]
        h["target"] = [server.scene.add_frame(f"/overlay/hand/{s}/target", axes_length=.11, axes_radius=.003)
                       for s in SIDES]
        h["head_target"] = server.scene.add_frame("/overlay/head/target", axes_length=.14, axes_radius=.003)
        h["paths"], h["dynamic"] = [], []  # removed with the scene

    def load(name):
        path = _resolve(run, by_name[name]["file"])
        ep = read_episode(path)
        scene, library = read_scene(path)
        key = "reachy" if scene is None else scene_key(scene, library)
        if key != ui["key"]:
            build(display_model() if scene is None else scene_model(path, scene))
            ui["key"] = key
        physics = source.value == "physics" and ep.physics is not None and (
            scene is None or scene.physics_poses is not None)
        st = episode_states(ui["model"], ep, scene, physics)
        model = ui["model"]
        head = np.zeros((len(st["q"]), 4, 4))
        head[:, :3, :3], head[:, :3, 3], head[:, 3, 3] = st["xmat"][:, model.body("head").id], \
            st["xpos"][:, model.body("head").id], 1.
        desired = np.full((ep.length, 2, 4, 4), np.nan)
        for i, s in enumerate(SIDES):
            if s in ep.reference.tcp:
                desired[:, i] = ep.reference.tcp[s]
        state.update(
            ep=ep, physics=physics, scene=scene, **st,
            ep_index=np.clip(np.searchsorted(ep.time, st["time"], side="right") - 1, 0, ep.length - 1),
            actual=np.stack([st["xpos"][:, model.body(t).id] for t in TIPS], 1),
            actual_R=np.stack([st["xmat"][:, model.body(t).id] for t in TIPS], 1),
            desired=desired[..., :3, 3], desired_R=desired[..., :3, :3],
            head_xyz=head[:, :3, 3], head_R=head[:, :3, :3], head_tip=(head @ HEAD_TIP)[:, :3, 3],
            head_ref=None if ep.reference.head is None else ep.reference.head[:, :3, :3],
            res_p=np.nan_to_num(ep.validation.get("tcp_pos_residual", np.zeros((ep.length, 2)))),
            res_r=np.nan_to_num(ep.validation.get("tcp_rot_residual", np.zeros((ep.length, 2)))),
            focus=np.concatenate([st["xpos"][:, model.body(t).id] for t in TIPS] + [head[:, :3, 3]]
                                 + moving_components(st["mocap_pos"])))

    def camera(client):
        # position first: viser moves look_at along with a new position
        if follow.value:
            client.camera.position = (1.7, -2.1, 1.7)
            client.camera.look_at = (0., 0., .8)
        else:
            pts = state["focus"]
            center = (pts.min(0) + pts.max(0)) / 2
            radius = max(.9, np.linalg.norm(pts.max(0) - pts.min(0)) * .6)
            client.camera.position = center + radius * np.array([2., -2.5, 1.4])
            client.camera.look_at = center
        client.camera.up_direction = (0., 0., 1.)

    @server.on_client_connect
    def connect(client):
        if state:
            camera(client)

    def filtered():
        return [r["name"] for r in catalog
                if (attempts.value == "all" or r["final"]) and cell.value in ("all", r["cell"])
                and (status.value == "all" or (status.value == "passed") == r["K"])]

    @cell.on_update
    @status.on_update
    @attempts.on_update
    def change_filter(_):
        options = filtered()
        if not options:
            stats.content = "No episodes match these filters."
            return
        selector.options = tuple(options)
        if selector.value not in options:
            selector.value = options[0]
        pending[0] = selector.value

    @selector.on_update
    @source.on_update
    def select(_):
        pending[0] = selector.value

    @previous_button.on_click
    def previous(_):
        selector.value = selector.options[(selector.options.index(selector.value) - 1) % len(selector.options)]

    @next_button.on_click
    def next_case(_):
        selector.value = selector.options[(selector.options.index(selector.value) + 1) % len(selector.options)]

    @follow.on_update
    def follow_base(_):
        ui["scene"].camera_tracking_enabled = follow.value
        ui["scene"].request_update()
        for client in server.get_clients().values():
            camera(client)

    @fit.on_click
    def fit_camera(_):
        for client in server.get_clients().values():
            camera(client)

    def segments(name, points, color, handles, thickness=.004):
        points = np.asarray(points)
        points = points[np.isfinite(points).all(1)]
        if len(points) >= 2:
            handles.append(server.scene.add_line_segments(name, np.stack((points[:-1], points[1:]), axis=1), color,
                                                          thickness=thickness))

    def setup_scene():
        ep = state["ep"]
        for handle in h["paths"]:
            handle.remove()
        h["paths"].clear()
        step = max(1, len(state["q"]) // 600)
        for i, (goal, actual) in enumerate(((LEFT_GOAL, LEFT_ACTUAL), (RIGHT_GOAL, RIGHT_ACTUAL))):
            segments(f"/overlay/paths/hand{i}/goal", state["desired"][::step, i], goal, h["paths"])
            segments(f"/overlay/paths/hand{i}/actual", state["actual"][::step, i], actual, h["paths"])
        q = state["q"][::step]
        segments("/overlay/paths/base", np.c_[q[:, :2], np.zeros(len(q))], BASE_COLOR, h["paths"])
        sc = ep.extra.get("tracking", {}).get("scenario") or {}
        tier = ep.tier
        parts = " + ".join(ep.body_parts) or "static"
        lines = [f"## {ep.episode_id}", f"{ep.dataset} · {ep.task} · {ep.regime} · {parts}"]
        if sc:
            lines.append(f"neck {sc.get('neck')} · {sc.get('extent')} · {sc.get('speed')} · start {sc.get('start')}")
        lines.append(f"**K {'PASS' if tier['K']['passed'] else 'FAIL'}**"
                     + ("" if tier.get("P") is None else f" · **P {'PASS' if tier['P']['passed'] else 'FAIL'}**")
                     + f" · {ep.length} states · {ep.duration:.1f} s")
        lines += [f"K: {r}" for r in tier["K"]["reasons"][:3]]
        lines += [f"P: {r}" for r in ((tier.get("P") or {}).get("reasons") or [])[:3]]
        if state["scene"] is not None:
            lines.append(f"Scene: {len(scene_names(state['scene']))} components"
                         + (" · physics rollout" if state["physics"] else " · kinematic poses"))
        elif source.value == "physics":
            lines.append("No physics rollout: kinematic poses")
        title.content = "\n\n".join(lines)

    def show(index):
        ep, k = state["ep"], state["ep_index"][index]
        ui["scene"].update_from_arrays(state["xpos"][index][None], state["xmat"][index][None],
                                       mocap_pos=state["mocap_pos"][index][None],
                                       mocap_quat=state["mocap_quat"][index][None],
                                       qpos=state["qpos"][index][None], qvel=np.zeros((1, ui["model"].nv)))
        for handle in h["dynamic"]:
            handle.remove()
        h["dynamic"].clear()
        tip, R = state["head_tip"][index], state["head_R"][index]
        segments("/overlay/gaze/ray", [tip, tip + R[:, 0]], GAZE_COLOR, h["dynamic"], .003)
        if state["head_ref"] is not None:
            segments("/overlay/gaze/target_ray", [tip, tip + state["head_ref"][k][:, 0]], (230, 150, 230),
                     h["dynamic"], .002)
        for i in (0, 1):
            h["actual"][i].position = state["actual"][index, i]
            h["actual"][i].wxyz = matrix_to_quat(state["actual_R"][index, i])
            if np.isfinite(state["desired"][k, i]).all():
                h["target"][i].position = state["desired"][k, i]
                h["target"][i].wxyz = matrix_to_quat(state["desired_R"][k, i])
        h["head_target"].position = state["head_xyz"][index]
        if state["head_ref"] is not None:
            h["head_target"].wxyz = matrix_to_quat(state["head_ref"][k])
        rp, rr = state["res_p"][k], state["res_r"][k]
        q = state["q"][index]
        stats.content = (f"**{state['time'][index]:.2f} s** ({state['time'][0]:.2f} … {state['time'][-1]:.2f})\n\n"
                         + ("" if state["physics"] else "\n\n".join(
                             f"{n}: {rp[i] * 1000:.2f} mm / {np.rad2deg(rr[i]):.2f}°"
                             for i, n in enumerate(("Left", "Right"))) + "\n\n")
                         + "Gripper opening: " + " / ".join(f"{v:.2f}" for v in ep.gripper_opening[k])
                         + f"\n\nBase: {q[0]:.2f}, {q[1]:.2f} m, {np.rad2deg(q[2]):.0f}°")

    elapsed, last_value, last_frame = 0., 0., -1
    previous_time = time.perf_counter()
    print(f"Reachy mjviser viewer ready: http://127.0.0.1:{server.get_port()} ({len(catalog)} episodes)", flush=True)
    try:
        while True:
            if pending[0] is not None:
                name, pending[0] = pending[0], None
                load(name)
                t0, t1 = float(state["time"][0]), float(state["time"][-1])
                slider.min, slider.max = t0, max(t0 + .01, t1)
                slider.value = elapsed = last_value = t0
                last_frame = -1
                setup_scene()
                for client in server.get_clients().values():
                    camera(client)
            times = state["time"]
            now = time.perf_counter()
            if slider.value != last_value:
                elapsed = slider.value
            elif play.value:
                elapsed += (now - previous_time) * float(speed.value.removesuffix("x"))
            previous_time = now
            if elapsed > times[-1]:
                elapsed = times[0]
            index = int(np.clip(np.searchsorted(times, elapsed, side="right") - 1, 0, len(times) - 1))
            offset = -state["xpos"][index, ui["model"].body("base_link").id] if follow.value else np.zeros(3)
            with server.atomic():
                h["overlay"].position = offset
                for handle in h["paths"]:
                    handle.visible = show_paths.value
                for i, handle in enumerate(h["target"]):
                    handle.visible = show_targets.value and bool(np.isfinite(state["desired"][:, i]).any())
                h["head_target"].visible = show_targets.value and state["head_ref"] is not None
                if index != last_frame:
                    show(index)
                    slider.value = float(times[index])
                    last_frame = index
                last_value = slider.value
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", type=Path, required=True, help="build or tracking run directory, or pulled dataset folder")
    p.add_argument("--port", type=int, default=8088)
    p.add_argument("--episode", help="initial episode (tracking: file stem; builds: <dataset>/<episode_id>)")
    a = p.parse_args(argv)
    run_viewer(a.run, a.port, a.episode)


if __name__ == "__main__":
    main()
