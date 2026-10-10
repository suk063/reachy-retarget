"""MuJoCo scene of a CALVIN episode (RoboVerse migration): desk, blocks and floor from CALVIN's own assets.

CALVIN (github.com/mees/calvin_env at 797142c, MIT) loads PyBullet URDFs with ``global_scaling``
0.8; RoboVerse ships the same files under ``roboverse_data/assets/calvin``. The scene built here:

* desk ``calvin_table_<X>/urdf/calvin_table_<X>.urdf``: the static body ``table`` at the recorded table
  pose; its prismatic links (``base__slide``, ``base__drawer``, ``base__button``, ``base__switch``)
  are child bodies with slide joints named ``table/<joint>`` (the episode's articulation columns),
  its fixed links (LED, lightbulb, plank) fixed children. Link origins, mesh scales and joint ranges
  are scaled by 0.8; the recorded joint values already are (drawer max 0.2199 = 0.275 x 0.8, slide
  0.28 = 0.35 x 0.8, switch 0.088 = 0.11 x 0.8 in env A task 0), so they drive the joints unchanged;
* visual geoms (contype 0, group 1): the URDF visual meshes (OBJ/STL) with the URDF colour and, for OBJ
  files, the diffuse texture of their MTL (``map_Kd``); collision geoms (group 3, invisible): the URDF
  collision meshes, VHACD files split into their convex parts (MuJoCo collides with mesh convex hulls,
  a multi-part file would become one hull);
* blocks: free bodies ``block_<colour>`` (freejoint ``block_<colour>_joint``) with the box size and colour
  of their URDF (``blocks/block_<colour>_<size>.urdf``) scaled by 0.8;
* floor: ``plane/plane.obj`` (30 m square, checker texture) and a collision plane.

The robot is not part of the scene (it is replaced by Reachy). ``scene_qpos`` columns: the block free
joints (``<joint>/x .. /qz``, wxyz) and the table joints, from the recorded states.
"""
from __future__ import annotations

import math
import posixpath
import xml.etree.ElementTree as ET
from collections.abc import Callable

import numpy as np

from ..schema.source import Articulation, SceneRef

SCALING = 0.8
ASSETS = "assets/calvin"
SOURCE = "https://github.com/mees/calvin_env/tree/797142c588c21e76717268b7b430958dbd13bf48/data"


def split_obj(data: bytes) -> list[bytes]:
    """The ``o`` objects of an OBJ file as separate OBJ files (vertices re-indexed); one part if it has none."""
    verts, parts, cur = [], [], None
    for line in data.decode("utf-8", "replace").splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "v":
            verts.append(" ".join(tok[1:4]))
        elif tok[0] == "o":
            cur = []
            parts.append(cur)
        elif tok[0] == "f":
            if cur is None:
                cur = []
                parts.append(cur)
            cur.append([int(t.split("/")[0]) for t in tok[1:]])
    out = []
    for faces in parts:
        if not faces:
            continue
        used = sorted({i if i > 0 else len(verts) + 1 + i for f in faces for i in f})
        new = {i: k + 1 for k, i in enumerate(used)}
        lines = [f"v {verts[i - 1]}" for i in used]
        lines += ["f " + " ".join(str(new[i if i > 0 else len(verts) + 1 + i]) for i in f) for f in faces]
        out.append(("\n".join(lines) + "\n").encode())
    return out


def mtl_texture(obj: bytes, obj_rel: str, read: Callable[[str], bytes]) -> str | None:
    """Path (relative to the RoboVerse root) of the diffuse texture of an OBJ's first MTL material, if any."""
    lib = next((line.split(None, 1)[1].strip() for line in obj.decode("utf-8", "replace").splitlines()
                if line.startswith("mtllib ")), None)
    if lib is None:
        return None
    mtl_rel = posixpath.normpath(posixpath.join(posixpath.dirname(obj_rel), lib))
    for line in read(mtl_rel).decode("utf-8", "replace").splitlines():
        tok = line.split(None, 1)
        if tok and tok[0] == "map_Kd" and len(tok) > 1:
            return posixpath.normpath(posixpath.join(posixpath.dirname(mtl_rel), tok[1].strip()))
    return None


def rpy_to_quat(rpy) -> list[float]:
    """URDF fixed-axis roll, pitch, yaw -> quaternion (w, x, y, z)."""
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2),
                              math.cos(y / 2), math.sin(y / 2))
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy]


def _floats(s, default="0 0 0") -> list[float]:
    return [float(v) for v in (s or default).split()]


def _fmt(v) -> str:
    return " ".join(f"{float(x):.9g}" for x in v)


class _Builder:
    def __init__(self, read: Callable[[str], bytes]):
        self.read, self.assets, self.asset_el = read, {}, ET.Element("asset")
        self.meshes, self.textures, self.materials = {}, {}, {}

    def mesh(self, rel: str, scale: float, data: bytes | None = None, key: str | None = None) -> str:
        key = key or rel
        name = "mesh_" + key.replace("/", "_").replace("#", "_").replace(".", "_")
        if name not in self.meshes:
            self.assets[key] = data if data is not None else self.read(rel)
            ext = posixpath.splitext(rel)[1].lower()
            el = ET.SubElement(self.asset_el, "mesh", name=name, file=key, scale=_fmt([scale] * 3))
            el.set("content_type", {".obj": "model/obj", ".stl": "model/stl"}[ext])
            self.meshes[name] = key
        return name

    def material(self, rgba, texture_rel: str | None) -> str:
        tex = None
        if texture_rel:
            tex = "tex_" + texture_rel.replace("/", "_").replace(".", "_")
            if tex not in self.textures:
                self.assets[texture_rel] = self.read(texture_rel)
                ET.SubElement(self.asset_el, "texture", name=tex, type="2d", file=texture_rel)
                self.textures[tex] = texture_rel
        name = f"mat_{len(self.materials)}"
        key = (tuple(rgba), tex)
        for n, k in self.materials.items():
            if k == key:
                return n
        el = ET.SubElement(self.asset_el, "material", name=name, rgba=_fmt(rgba))
        if tex:
            el.set("texture", tex)
        self.materials[name] = key
        return name

    def visual(self, body: ET.Element, rel: str, rgba, pos=(0, 0, 0), quat=(1, 0, 0, 0), scale=SCALING):
        tex = mtl_texture(self.read(rel), rel, self.read) if rel.lower().endswith(".obj") else None
        ET.SubElement(body, "geom", type="mesh", mesh=self.mesh(rel, scale), material=self.material(rgba, tex),
                      pos=_fmt(pos), quat=_fmt(quat), contype="0", conaffinity="0", group="1")

    def collision(self, body: ET.Element, rel: str, pos=(0, 0, 0), quat=(1, 0, 0, 0), scale=SCALING):
        data = self.read(rel)
        parts = split_obj(data) if rel.lower().endswith(".obj") else [data]
        for k, part in enumerate(parts):
            key = rel if len(parts) == 1 else f"{rel}#part{k}.obj"
            ET.SubElement(body, "geom", type="mesh", mesh=self.mesh(rel, scale, part if len(parts) > 1 else None, key),
                          pos=_fmt(pos), quat=_fmt(quat), group="3", rgba="0.5 0.5 0.5 0")


def _link_geoms(b: _Builder, link: ET.Element, body: ET.Element, urdf_rel: str) -> None:
    for tag in ("visual", "collision"):
        for g in link.findall(tag):
            mesh = g.find("geometry/mesh")
            if mesh is None:
                continue
            rel = posixpath.normpath(posixpath.join(posixpath.dirname(urdf_rel), mesh.get("filename")))
            o = g.find("origin")
            pos = [v * SCALING for v in _floats(o.get("xyz") if o is not None else None)]
            quat = rpy_to_quat(_floats(o.get("rpy") if o is not None else None))
            if tag == "visual":
                c = g.find("material/color")
                b.visual(body, rel, _floats(c.get("rgba")) if c is not None else [1, 1, 1, 1], pos, quat)
            else:
                b.collision(body, rel, pos, quat)


def _inertial(body: ET.Element, link: ET.Element) -> None:
    i = link.find("inertial")
    m = float(i.find("mass").get("value")) if i is not None else 0.0
    if m <= 0:
        m = 0.1
    o = i.find("origin") if i is not None else None
    t = i.find("inertia") if i is not None else None
    diag = [max(float(t.get(k)) * SCALING ** 2, 1e-7) for k in ("ixx", "iyy", "izz")] if t is not None else [1e-4] * 3
    ET.SubElement(body, "inertial", pos=_fmt([v * SCALING for v in _floats(o.get("xyz") if o is not None else None)]),
                  mass=f"{m:.9g}", diaginertia=_fmt(diag))


def build_scene(scene: str, read: Callable[[str], bytes], table_pose, blocks: dict) -> tuple[SceneRef, dict]:
    """``(SceneRef, provenance)`` of CALVIN scene ``scene`` (A-D). ``read(rel)`` returns the bytes of a
    RoboVerse file (path below ``roboverse_data``); ``table_pose`` is the table base pose (xyz + wxyz);
    ``blocks`` maps colour -> (URDF size name, initial pose xyz + wxyz)."""
    b = _Builder(read)
    root = ET.Element("mujoco", model=f"calvin_scene_{scene}")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    root.append(b.asset_el)
    world = ET.SubElement(root, "worldbody")
    # floor
    plane = f"{ASSETS}/plane/plane.obj"
    b.visual(world, plane, [1, 1, 1, 1], scale=1.0)
    world[-1].set("name", "floor")
    b.asset_el.find(f"mesh[@name='{world[-1].get('mesh')}']").set("inertia", "shell")  # flat: no volume
    ET.SubElement(world, "geom", name="floor_collision", type="plane", size="15 15 0.1", group="3",
                  rgba="0.5 0.5 0.5 0")
    # desk
    urdf_rel = f"{ASSETS}/calvin_table_{scene}/urdf/calvin_table_{scene}.urdf"
    robot = ET.fromstring(read(urdf_rel))
    links = {link.get("name"): link for link in robot.findall("link")}
    table = ET.SubElement(world, "body", name="table", pos=_fmt(table_pose[:3]), quat=_fmt(table_pose[3:]))
    _link_geoms(b, links["base_link"], table, urdf_rel)
    joints = []
    for j in robot.findall("joint"):
        if j.find("parent").get("link") != "base_link":
            raise ValueError(f"{urdf_rel}: joint {j.get('name')} is not a child of base_link")
        child = j.find("child").get("link")
        o = j.find("origin")
        body = ET.SubElement(table, "body", name=f"table/{child}",
                             pos=_fmt([v * SCALING for v in _floats(o.get("xyz"))]),
                             quat=_fmt(rpy_to_quat(_floats(o.get("rpy")))))
        if j.get("type") == "prismatic":
            lim = j.find("limit")
            ET.SubElement(body, "joint", name=f"table/{j.get('name')}", type="slide",
                          axis=_fmt(_floats(j.find("axis").get("xyz"))),
                          range=_fmt([float(lim.get("lower")) * SCALING, float(lim.get("upper")) * SCALING]))
            joints.append(j.get("name"))
            _inertial(body, links[child])
        elif j.get("type") != "fixed":
            raise ValueError(f"{urdf_rel}: unsupported joint type {j.get('type')}")
        _link_geoms(b, links[child], body, urdf_rel)
    # blocks
    sizes = {}
    for colour, (size_name, pose) in sorted(blocks.items()):
        rel = f"{ASSETS}/blocks/block_{colour}_{size_name}.urdf"
        link = ET.fromstring(read(rel)).find("link")
        box = [float(v) * SCALING for v in link.find("visual/geometry/box").get("size").split()]
        c = link.find("visual/material/color")
        rgba = _floats(c.get("rgba")) if c is not None else [1, 1, 1, 1]
        body = ET.SubElement(world, "body", name=f"block_{colour}", pos=_fmt(pose[:3]), quat=_fmt(pose[3:]))
        ET.SubElement(body, "freejoint", name=f"block_{colour}_joint")
        m = float(link.find("inertial/mass").get("value"))
        ET.SubElement(body, "inertial", pos="0 0 0", mass=f"{m:.9g}",
                      diaginertia=_fmt([m * (box[1] ** 2 + box[2] ** 2) / 12, m * (box[0] ** 2 + box[2] ** 2) / 12,
                                        m * (box[0] ** 2 + box[1] ** 2) / 12]))
        half = [v / 2 for v in box]
        ET.SubElement(body, "geom", type="box", size=_fmt(half), rgba=_fmt(rgba), contype="0", conaffinity="0",
                      group="1")
        ET.SubElement(body, "geom", type="box", size=_fmt(half), group="3", rgba="0.5 0.5 0.5 0")
        sizes[f"block_{colour}"] = {"urdf": rel, "size_m": [round(v, 6) for v in box]}
    mjcf = ET.tostring(root, encoding="unicode")
    initial = {f"block_{c}_joint": list(map(float, pose)) for c, (_, pose) in blocks.items()}
    ref = SceneRef(mjcf=mjcf, robot_prefixes=[], initial_qpos=initial, assets=dict(b.assets), inactive_bodies=[])
    prov = {"builder": "sources.calvin_scene", "source": SOURCE, "scaling": SCALING, "table_urdf": urdf_rel,
            "table_joints": [f"table/{j}" for j in joints], "blocks": sizes, "assets": len(b.assets),
            "collision": "URDF collision meshes, VHACD files split into convex parts; collision plane for the floor"}
    return ref, prov


def scene_qpos(block_poses: dict, table_joints: list[str], table_qpos: np.ndarray) -> Articulation:
    """Per-step scene joint positions: block free joints (``block_<colour>_joint/x .. /qz``) and table joints."""
    names, cols = [], []
    for colour, pose in sorted(block_poses.items()):
        names += [f"block_{colour}_joint/{k}" for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]
        cols.append(np.asarray(pose, float))
    names += [f"table/{j}" for j in table_joints]
    cols.append(np.asarray(table_qpos, float))
    return Articulation(joint_names=names, qpos=np.concatenate(cols, axis=1))


__all__ = ["build_scene", "scene_qpos", "split_obj", "mtl_texture", "rpy_to_quat", "SCALING"]
