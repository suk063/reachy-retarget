"""MuJoCo scene of a MobileManiBench PartNet-Mobility episode: the object, the support stage and the ground.

The source simulator (Isaac Lab, ``unimanip/utils/env_model.py`` / ``partnet_model.py`` at
:data:`.mobilemanibench.CODE_COMMIT`) spawns, per episode:

* the PartNet-Mobility object (``Assets/partnet/dataset/<id>/mobility.urdf`` with its
  ``textured_objs/*.obj`` meshes, converted to USD), scaled by the group's ``scale`` of
  ``configs/data/analysis_scene.yaml``; fixed base except carts (``config.yaml`` ``fix_base``);
  every joint at ``lower + 0.001`` except the grasp joint at its recorded initial value;
* a kinematic, colliding stage cuboid 4 x 2 x 1 m (``room_space`` 4), its top at the room height for
  ``tabletop`` groups and at ``higher_random_object`` otherwise, shifted by the object's rotated
  bounding box (lower-limit pose): x, y = root + centre, y += (2 - size_y) / 2;
* a kinematic ground cuboid 4 x 4 x 0.02 m (top at z = 0.01) 1 m behind the stage centre in -y, and
  a terrain plane at z = 0. Stage, ground and terrain are hidden in the recorded videos but collide;
* a room (GenieSim / IsaacSim USD) with collision disabled: a visual backdrop, not representable
  here (USD only; IsaacSim rooms are not even in the release). It is recorded as an omitted
  component in the episode provenance, not built.

This module rebuilds the object as MJCF from the URDF (links -> bodies, revolute -> hinge, continuous
-> unlimited hinge, prismatic -> slide; OBJ files split per MTL material, PNG diffuse textures;
MuJoCo cannot decode JPEG, so JPEG-textured materials keep their colour only; visual sub-meshes with
fewer than 4 distinct vertices and degenerate (coplanar) collision meshes are skipped and listed),
adds the stage and ground boxes and the terrain plane, and derives every joint at every step:
the grasp joint from the recorded handle orientation (revolute; the handle frame is the grasp link
rotated by identity) or position (prismatic), the other joints constant at their initial value, and
for carts (free base) the root from the handle pose and its fixed offset at t0.
"""
from __future__ import annotations

import posixpath
import xml.etree.ElementTree as ET
from collections.abc import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from ..schema.source import SceneRef
from .calvin_scene import rpy_to_quat

ROOM_SPACE = 4.0
# Source prismatic joint coordinates are in URDF (unscaled) units and scale with the object: in
# dishwasher 11826 (scale 0.4) the racks recorded at lower + 0.001 (0.077, 0.065) sit 0.4 mm from their
# designed closed position when multiplied by the scale, but would protrude 4 cm into the closed door
# if read as metres (the door closes in that episode). Revolute coordinates are angles (unchanged).
PRISMATIC_UNITS = "URDF units x object scale (checked on dishwasher 11826 racks vs the closed door)"
GROUND_THICKNESS = 0.02
GROUND_TOP = 0.01


def _floats(s, default="0 0 0") -> list[float]:
    return [float(v) for v in (s or default).split()]


def _fmt(v) -> str:
    return " ".join(f"{float(x):.9g}" for x in v)


def parse_mtl(text: str) -> dict:
    mats, cur = {}, None
    for line in text.splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "newmtl":
            cur = t[1] if len(t) > 1 else ""
            mats[cur] = {"Kd": [0.8, 0.8, 0.8], "d": 1.0, "map_Kd": None}
        elif cur is not None and t[0] == "Kd" and len(t) >= 4:
            mats[cur]["Kd"] = [float(x) for x in t[1:4]]
        elif cur is not None and t[0] == "d" and len(t) >= 2:
            mats[cur]["d"] = float(t[1])
        elif cur is not None and t[0] == "map_Kd" and len(t) >= 2:
            mats[cur]["map_Kd"] = t[-1]
    return mats


def split_obj_by_material(text: str) -> tuple[str | None, dict[str | None, str]]:
    """``(mtllib, {material: OBJ text with that material's faces})``; vertex, texcoord and normal
    indices re-numbered per part."""
    src = ([], [], [])
    groups, cur, mtllib = {}, None, None
    for line in text.splitlines():
        if line.startswith("v "):
            src[0].append(line)
        elif line.startswith("vt "):
            src[1].append(line)
        elif line.startswith("vn "):
            src[2].append(line)
        elif line.startswith("usemtl"):
            cur = line.split()[1] if len(line.split()) > 1 else None
        elif line.startswith("mtllib"):
            mtllib = line.split()[1]
        elif line.startswith("f "):
            groups.setdefault(cur, []).append(line.split()[1:])
    out = {}
    for mat, faces in groups.items():
        maps, lists, lines = ({}, {}, {}), ([], [], []), []
        for f in faces:
            idx = []
            for c in f:
                new = []
                for k, s in enumerate((c.split("/") + ["", ""])[:3]):
                    if not s:
                        new.append("")
                        continue
                    i = int(s)
                    i = i - 1 if i > 0 else len(src[k]) + i
                    if i not in maps[k]:
                        maps[k][i] = len(lists[k]) + 1
                        lists[k].append(src[k][i])
                    new.append(str(maps[k][i]))
                while new and new[-1] == "":
                    new.pop()
                idx.append("/".join(new))
            lines.append("f " + " ".join(idx))
        out[mat] = "\n".join(lists[0] + lists[1] + lists[2] + lines) + "\n"
    return mtllib, out


def _vertices(text: str) -> np.ndarray:
    return np.array([[float(x) for x in line.split()[1:4]] for line in text.splitlines() if line.startswith("v ")],
                    float).reshape(-1, 3)


def degenerate(P: np.ndarray) -> str | None:
    if len(np.unique(P.round(9), axis=0)) < 4:
        return "fewer than 4 distinct vertices"
    s = np.linalg.svd(P - P.mean(0), compute_uv=False)
    return "coplanar" if s[2] <= 1e-6 * s[0] else None


class PartNetObject:
    """MJCF elements of one PartNet-Mobility object (``mobility.urdf``), named ``<name>`` (root body),
    ``<name>:<link>`` (bodies) and ``<name>:<joint>`` (joints)."""

    def __init__(self, read: Callable[[str], bytes], obj_dir: str, scale: float, name: str, free: bool):
        self.read, self.dir, self.scale, self.name, self.free = read, obj_dir, float(scale), name, free
        self.assets, self.meshes, self.textures, self.materials = {}, {}, {}, {}
        self.notes = {"skipped_visual": [], "skipped_collision": [], "jpeg_textures_dropped": []}
        self.urdf = ET.fromstring(read(f"{obj_dir}/mobility.urdf"))
        self.links = {link.get("name"): link for link in self.urdf.findall("link")}
        self.joints = self.urdf.findall("joint")
        self.children = {}
        for j in self.joints:
            self.children.setdefault(j.find("parent").get("link"), []).append(j)
        child_links = {j.find("child").get("link") for j in self.joints}
        roots = [n for n in self.links if n not in child_links]
        if len(roots) != 1:
            raise ValueError(f"{obj_dir}/mobility.urdf: expected one root link, got {roots}")
        self.root_link = roots[0]
        self._mtl = {}
        self.dof_joints = [j for j in self.joints if j.get("type") in ("revolute", "continuous", "prismatic")]

    # ---------------------------------------------------------------- assets
    def _material(self, mtllib: str | None, mat: str | None) -> str:
        key = (mtllib, mat)
        if key in self.materials:
            return self.materials[key][0]
        m = {"Kd": [0.8, 0.8, 0.8], "d": 1.0, "map_Kd": None}
        if mtllib:
            rel = posixpath.normpath(f"{self.dir}/textured_objs/{mtllib}")
            if rel not in self._mtl:
                self._mtl[rel] = parse_mtl(self.read(rel).decode("utf-8", "replace"))
            m = self._mtl[rel].get(mat, m)
        tex = None
        if m["map_Kd"]:
            rel = posixpath.normpath(f"{self.dir}/textured_objs/{m['map_Kd']}")
            if rel.lower().endswith(".png"):
                tex = f"{self.name}:tex{len(self.textures)}"
                if rel not in self.textures:
                    self.assets[rel] = self.read(rel)
                    self.textures[rel] = tex
                tex = self.textures[rel]
            elif rel not in self.notes["jpeg_textures_dropped"]:
                self.notes["jpeg_textures_dropped"].append(rel)
        name = f"{self.name}:mat{len(self.materials)}"
        self.materials[key] = (name, [*m["Kd"], m["d"]], tex)
        return name

    def _mesh(self, key: str, data: bytes) -> str:
        name = f"{self.name}:mesh{len(self.meshes)}"
        self.assets[key] = data
        self.meshes[name] = key
        return name

    # ---------------------------------------------------------------- bodies
    def _geoms(self, body: ET.Element, link: str) -> None:
        el = self.links[link]
        for k, v in enumerate(el.findall("visual")):
            fn = v.find("geometry/mesh").get("filename")
            o = v.find("origin")
            pos = [x * self.scale for x in _floats(o.get("xyz") if o is not None else None)]
            quat = rpy_to_quat(_floats(o.get("rpy") if o is not None else None))
            rel = posixpath.normpath(f"{self.dir}/{fn}")
            mtllib, parts = split_obj_by_material(self.read(rel).decode("utf-8", "replace"))
            for mat, text in parts.items():
                d = degenerate(_vertices(text))
                if d and d.startswith("fewer"):
                    self.notes["skipped_visual"].append({"file": rel, "material": mat, "reason": d})
                    continue
                mesh = self._mesh(f"{rel}#{mat}.obj", text.encode())
                ET.SubElement(body, "geom", type="mesh", mesh=mesh, material=self._material(mtllib, mat),
                              pos=_fmt(pos), quat=_fmt(quat), contype="0", conaffinity="0", group="1")
        for k, c in enumerate(el.findall("collision")):
            fn = c.find("geometry/mesh").get("filename")
            o = c.find("origin")
            pos = [x * self.scale for x in _floats(o.get("xyz") if o is not None else None)]
            quat = rpy_to_quat(_floats(o.get("rpy") if o is not None else None))
            rel = posixpath.normpath(f"{self.dir}/{fn}")
            data = self.read(rel)
            d = degenerate(_vertices(data.decode("utf-8", "replace")))
            if d:
                self.notes["skipped_collision"].append({"file": rel, "reason": d})
                continue
            ET.SubElement(body, "geom", type="mesh", mesh=self._mesh(f"{rel}#collision.obj", data), pos=_fmt(pos),
                          quat=_fmt(quat), group="3", rgba="0.5 0.5 0.5 0")
        for j in self.children.get(link, []):
            child = j.find("child").get("link")
            o = j.find("origin")
            b = ET.SubElement(body, "body", name=f"{self.name}:{child}",
                              pos=_fmt([x * self.scale for x in _floats(o.get("xyz") if o is not None else None)]),
                              quat=_fmt(rpy_to_quat(_floats(o.get("rpy") if o is not None else None))))
            ET.SubElement(b, "inertial", pos="0 0 0", mass="0.1", diaginertia="1e-4 1e-4 1e-4")
            jt = j.get("type")
            if jt in ("revolute", "continuous", "prismatic"):
                ax = j.find("axis")
                attrs = {"name": f"{self.name}:{j.get('name')}", "type": "slide" if jt == "prismatic" else "hinge",
                         "axis": _fmt(_floats(ax.get("xyz") if ax is not None else "1 0 0"))}
                lim = j.find("limit")
                if jt != "continuous" and lim is not None:
                    k = self.scale if jt == "prismatic" else 1.0
                    attrs["range"] = _fmt([float(lim.get("lower", 0)) * k, float(lim.get("upper", 0)) * k])
                    attrs["limited"] = "true"
                else:
                    attrs["limited"] = "false"
                ET.SubElement(b, "joint", **attrs)
            elif jt != "fixed":
                raise ValueError(f"{self.dir}/mobility.urdf: unsupported joint type {jt!r}")
            self._geoms(b, child)

    def body(self, pose) -> ET.Element:
        b = ET.Element("body", name=self.name, pos=_fmt(pose[:3]), quat=_fmt(pose[3:7]))
        if self.free:
            ET.SubElement(b, "freejoint", name=f"{self.name}:root")
        ET.SubElement(b, "inertial", pos="0 0 0", mass="0.1", diaginertia="1e-4 1e-4 1e-4")
        self._geoms(b, self.root_link)
        return b

    def asset_elements(self, asset: ET.Element) -> None:
        for name, key in self.meshes.items():
            ET.SubElement(asset, "mesh", name=name, file=key, scale=_fmt([self.scale] * 3), inertia="shell",
                          content_type="model/obj")
        for rel, tex in self.textures.items():
            ET.SubElement(asset, "texture", name=tex, type="2d", file=rel)
        for name, rgba, tex in self.materials.values():
            el = ET.SubElement(asset, "material", name=name, rgba=_fmt(rgba))
            if tex:
                el.set("texture", tex)

    def initial_joints(self, grasp_joint: str | None, grasp_q0: float | None) -> dict[str, float]:
        """Initial joint positions as the source sets them: ``lower + 0.001`` (``0.001`` for continuous
        joints), the grasp joint at its recorded initial value ``grasp_q0``. Source joint coordinates
        are URDF (unscaled) values; prismatic ones scale with the geometry, so they are multiplied by
        the object scale here (:data:`PRISMATIC_UNITS`)."""
        out, types = {}, {}
        for j in self.dof_joints:
            lim = j.find("limit")
            lower = float(lim.get("lower", 0)) if (lim is not None and j.get("type") != "continuous") else 0.0
            out[j.get("name")] = lower + 0.001
            types[j.get("name")] = j.get("type")
        if grasp_joint is not None and grasp_q0 is not None:
            out[grasp_joint] = float(grasp_q0)
        return {f"{self.name}:{n}": v * (self.scale if types[n] == "prismatic" else 1.0) for n, v in out.items()}


def _pose_matrix(p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(p[3:7])[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = p[:3]
    return T


def _pose7(T: np.ndarray) -> np.ndarray:
    q = Rotation.from_matrix(T[..., :3, :3]).as_quat()
    return np.concatenate([T[..., :3, 3], q[..., [3, 0, 1, 2]]], axis=-1)


def build_scene(read: Callable[[str], bytes], obj_dir: str, *, name: str, scale: float, root_pose, free: bool,
                place: str, stage_top: float, grasp_joint: str | None, grasp_q0: float | None) -> tuple:
    """``(SceneRef, PartNetObject, mujoco model, provenance)``: object at ``root_pose`` (xyz + wxyz),
    stage, ground and terrain. ``stage_top`` is the stage top height (room height for tabletop
    groups, ``higher_random_object`` otherwise)."""
    import mujoco

    obj = PartNetObject(read, obj_dir, scale, name, free)
    root = ET.Element("mujoco", model=f"mobilemanibench_{name}")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false", autolimits="true")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    body = obj.body(np.asarray(root_pose, float))
    obj.asset_elements(asset)
    world.append(body)
    init = obj.initial_joints(grasp_joint, grasp_q0)
    # rotated bounding box of the object at its lower-limit pose (partnet_model.process_obj_sim_config)
    probe = ET.tostring(root, encoding="unicode")
    m = mujoco.MjModel.from_xml_string(probe, obj.assets)
    d = mujoco.MjData(m)
    if free:
        d.qpos[:7] = root_pose
    for j in range(m.njnt):
        if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            d.qpos[m.jnt_qposadr[j]] = m.jnt_range[j][0] if m.jnt_limited[j] else 0.0
    mujoco.mj_forward(m, d)
    V = []
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 1:
            k = m.geom_dataid[g]
            a = m.mesh_vertadr[k]
            V.append(m.mesh_vert[a:a + m.mesh_vertnum[k]] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    V = np.concatenate(V) - np.asarray(root_pose[:3], float)
    lo, hi = V.min(0), V.max(0)
    centre, size = (lo + hi) / 2, hi - lo
    stage_xy = np.asarray(root_pose[:2], float) + centre[:2]
    stage_xy[1] += (ROOM_SPACE / 2 - size[1]) / 2
    stage = ET.SubElement(world, "body", name="stage", pos=_fmt([*stage_xy, stage_top - 0.5]))
    half = [ROOM_SPACE / 2, ROOM_SPACE / 4, 0.5]
    ET.SubElement(stage, "geom", type="box", size=_fmt(half), rgba="0.1 0.1 0.1 1", contype="0", conaffinity="0",
                  group="1")
    ET.SubElement(stage, "geom", type="box", size=_fmt(half), group="3", rgba="0.5 0.5 0.5 0")
    ground_xy = stage_xy + [0.0, (ROOM_SPACE / 2 - ROOM_SPACE) / 2]
    ground = ET.SubElement(world, "body", name="ground", pos=_fmt([*ground_xy, GROUND_TOP - GROUND_THICKNESS / 2]))
    half_g = [ROOM_SPACE / 2, ROOM_SPACE / 2, GROUND_THICKNESS / 2]
    ET.SubElement(ground, "geom", type="box", size=_fmt(half_g), rgba="0.1 0.1 0.1 1", contype="0", conaffinity="0",
                  group="1")
    ET.SubElement(ground, "geom", type="box", size=_fmt(half_g), group="3", rgba="0.5 0.5 0.5 0")
    ET.SubElement(world, "geom", name="terrain", type="plane", size=_fmt([ROOM_SPACE, ROOM_SPACE, 0.1]), group="3",
                  rgba="0.5 0.5 0.5 0")
    mjcf = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(mjcf, obj.assets)
    initial = {k: [v] for k, v in init.items()}
    if free:
        initial[f"{name}:root"] = list(map(float, root_pose))
    ref = SceneRef(mjcf=mjcf, robot_prefixes=[], initial_qpos=initial, assets=dict(obj.assets), inactive_bodies=[])
    prov = {"builder": "sources.partnet_scene", "object_dir": obj_dir, "scale": scale, "free_base": free,
            "place": place, "lower_limit_bbox_rotated": {"centre_m": centre.round(6).tolist(),
                                                        "size_m": size.round(6).tolist()},
            "stage": {"size_m": [ROOM_SPACE, ROOM_SPACE / 2, 1.0], "top_z": stage_top, "xy": stage_xy.round(6).tolist()},
            "ground": {"size_m": [ROOM_SPACE, ROOM_SPACE, GROUND_THICKNESS], "top_z": GROUND_TOP,
                       "xy": ground_xy.round(6).tolist()},
            "terrain": "collision plane at z = 0", "joints": [f"{name}:{j.get('name')}" for j in obj.dof_joints],
            "initial_joints": init, "meshes": len(obj.meshes), "textures": len(obj.textures),
            "skipped_visual": obj.notes["skipped_visual"], "skipped_collision": obj.notes["skipped_collision"],
            "jpeg_textures_dropped": obj.notes["jpeg_textures_dropped"],
            "supports_note": "stage, ground and terrain collide but are hidden in the source videos",
            "prismatic_units": PRISMATIC_UNITS}
    return ref, obj, model, prov


def solve_grasp_joint(model, handle: np.ndarray, *, joint: str, link: str, initial: dict, joint_type: str,
                      root=None) -> tuple[np.ndarray, dict]:
    """The grasp joint at every step from the (T, 4, 4) handle track: revolute = rotation of the handle
    relative to the grasp link at q = 0 about the joint axis (the handle frame is the link frame
    rotated by identity); prismatic = displacement of the handle along the joint axis since t0, added
    to the initial value. Other joints are held at ``initial``. Returns ``(q, check)``."""
    import mujoco

    d = mujoco.MjData(model)
    jid = model.joint(joint).id
    adr = model.jnt_qposadr[jid]
    for name, v in initial.items():
        j = model.joint(name).id
        d.qpos[model.jnt_qposadr[j]:model.jnt_qposadr[j] + len(np.atleast_1d(v))] = v
    if root is not None:
        d.qpos[:7] = root
    d.qpos[adr] = 0.0
    mujoco.mj_forward(model, d)
    bid = model.body(link).id
    R0 = d.xmat[bid].reshape(3, 3).copy()
    axis_local = model.jnt_axis[jid].copy()
    if joint_type == "REVOLUTE":
        rv = Rotation.from_matrix(np.einsum("ji,tjk->tik", R0, handle[:, :3, :3])).as_rotvec()
        q = rv @ axis_local
        off = np.linalg.norm(rv - q[:, None] * axis_local, axis=1)
        check = {"rule": "rotvec(R_link(q=0)^T R_handle(t)) . axis", "max_off_axis_rad": float(off.max())}
    else:
        axis_world = R0 @ axis_local
        disp = (handle[:, :3, 3] - handle[0, :3, 3]) @ axis_world
        q0 = float(np.atleast_1d(initial[joint])[0])
        q = q0 + disp
        perp = handle[:, :3, 3] - handle[0, :3, 3] - disp[:, None] * axis_world
        check = {"rule": "initial + displacement of the handle along the joint axis since t0",
                 "max_off_axis_m": float(np.linalg.norm(perp, axis=1).max())}
    # the handle stays at a fixed offset in the grasp link frame
    offs = []
    for t in sorted({0, len(q) // 2, len(q) - 1}):
        d.qpos[adr] = q[t]
        mujoco.mj_forward(model, d)
        offs.append(d.xmat[bid].reshape(3, 3).T @ (handle[t, :3, 3] - d.xpos[bid]))
    check["handle_offset_in_link_spread_m"] = float(np.ptp(np.array(offs), axis=0).max())
    check["range"] = [float(q.min()), float(q.max())]
    return q, check


def cart_root_track(model, handle: np.ndarray, root0, *, link: str, initial: dict) -> tuple[np.ndarray, dict]:
    """Root poses (T, 7) of a free-base object whose grasp link is fixed to the root (carts): the handle
    keeps its t0 offset to the root, root(t) = handle(t) . (root0^-1 . handle(0))^-1."""
    import mujoco

    d = mujoco.MjData(model)
    for name, v in initial.items():
        j = model.joint(name).id
        d.qpos[model.jnt_qposadr[j]:model.jnt_qposadr[j] + len(np.atleast_1d(v))] = v
    d.qpos[:7] = root0
    mujoco.mj_forward(model, d)
    bid = model.body(link).id
    rot_err = float(Rotation.from_matrix(d.xmat[bid].reshape(3, 3).T @ handle[0, :3, :3]).magnitude())
    offset = np.linalg.inv(_pose_matrix(root0)) @ handle[0]
    R = handle @ np.linalg.inv(offset)
    tilt = np.degrees(np.arccos(np.clip(R[:, 2, 2], -1, 1)))
    return _pose7(R), {"rule": "root(t) = handle(t) . (root0^-1 . handle(0))^-1",
                       "grasp_link_vs_handle_rotation_t0_rad": rot_err, "max_tilt_deg": float(tilt.max()),
                       "root_z_range_m": [float(R[:, 2, 3].min()), float(R[:, 2, 3].max())],
                       "path_length_m": float(np.linalg.norm(np.diff(R[:, :2, 3], axis=0), axis=1).sum())}


__all__ = ["PartNetObject", "build_scene", "solve_grasp_joint", "cart_root_track", "split_obj_by_material",
           "parse_mtl", "degenerate"]
