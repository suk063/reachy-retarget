"""URDF mesh loading and FK playback for the tracking viewer.

Port of reachy-control ``util/visualization.py``: the same URDF (``robot/assets/reachy.urdf``,
byte-identical to reachy-control ``asset/reachy.urdf``), the same mesh resolution of ROS package
paths into ``assets/packages``, the same GLB conversion and per-link visual/collision handles. Link
poses come from this package's URDF kinematics (:class:`reachy_retarget.robot.urdf.KinematicTree`,
mimic joints included) instead of pinocchio; :func:`prepare_episode` asserts, as reachy-control's
``prepare_motion`` does, that the replayed TCPs match the episode's stored TCPs.

Requires the ``viz`` extra (viser, trimesh, pycollada); nothing else imports this module.
"""
from __future__ import annotations

import functools
import xml.etree.ElementTree as ET

import numpy as np

from ..robot.reachy import FIXED, JOINTS, planar
from ..robot.resources import URDF_FILE, urdf
from ..robot.urdf import KinematicTree, pose
from ..schema.rotations import matrix_to_quat

ASSETS = URDF_FILE.parent
URDF = URDF_FILE


def mesh_path(uri):
    """Resolve the original robot's ROS paths without rewriting its URDF."""
    if uri.startswith("package://"):
        relative = uri.removeprefix("package://")
    elif "/share/" in uri:
        relative = uri.split("/share/", 1)[1]
    else:
        raise ValueError(f"Unsupported mesh URI: {uri}")
    path = ASSETS / "packages" / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing URDF mesh: {path}")
    return path


@functools.cache
def _links():
    return [e.get("name") for e in ET.parse(URDF).getroot().findall("link")]


@functools.cache
def _tree():
    # every URDF link relative to base_link; the 19 non-base joints of q (arms, neck, fingers) move
    links = [name for name in _links() if name != "base_link"]
    return KinematicTree(urdf(), "base_link", links, JOINTS[3:], FIXED)


def link_poses(q):
    """(links, positions (T, L, 3), wxyz (T, L, 4)) of every URDF link for canonical q (T, 22)."""
    q = np.asarray(q, float)
    links = _links()
    fk = _tree().fk(q[:, 3:])
    W = planar(q[:, 0], q[:, 1], q[:, 2])
    pos = np.empty((len(q), len(links), 3), np.float32)
    rot = np.empty((len(q), len(links), 4), np.float32)
    for j, name in enumerate(links):
        T = W if name == "base_link" else W @ fk[name]
        pos[:, j] = T[:, :3, 3]
        rot[:, j] = matrix_to_quat(T[:, :3, :3])
    return links, pos, rot


def prepare_episode(ep):
    """Link poses of an episode; raises if the replayed TCPs differ from the stored ones."""
    links, pos, rot = link_poses(ep.q)
    world = ep.tcp_world
    for frame, side in (("l_arm_tip", "left"), ("r_arm_tip", "right")):
        if not np.allclose(pos[:, links.index(frame)], world[side][:, :3, 3], atol=1e-5, rtol=0):
            raise ValueError("Recorded motion does not match reachy.urdf; regenerate the episode.")
    return links, pos, rot


@functools.lru_cache(maxsize=None)
def mesh_glb(path):
    import trimesh
    scene = trimesh.load_scene(path, process=False)
    if not scene.geometry:
        raise ValueError(f"Empty visual mesh: {path}")
    return scene.export(file_type="glb")


def add_robot_visuals(server, links, kind="visual"):
    """Per-link frames holding the URDF visual (or collision) geometry, as reachy-control does."""
    import trimesh
    root = ET.parse(URDF).getroot()
    materials = {e.get("name"): e.find("color") for e in root.findall("material")}
    handles = {}
    for link in root.findall("link"):
        name = link.get("name")
        visuals = link.findall(kind)
        if not visuals:
            continue
        parent = f"/{kind}/{name}"
        handles[links.index(name)] = server.scene.add_frame(parent, show_axes=False)
        for index, visual in enumerate(visuals):
            origin = visual.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            placement = dict(position=xyz, wxyz=matrix_to_quat(pose(xyz, rpy)[:3, :3]))
            shape = visual.find("geometry")[0]
            node = f"{parent}/visual_{index}"
            if shape.tag == "mesh":
                scale = tuple(np.fromstring(shape.get("scale", "1 1 1"), sep=" "))
                path = mesh_path(shape.get("filename"))
                if kind == "visual":
                    server.scene.add_glb(node, mesh_glb(path), scale=scale, **placement)
                else:
                    mesh = trimesh.load_scene(path, process=False).to_mesh()
                    server.scene.add_mesh_simple(node, mesh.vertices, mesh.faces, color=(63, 176, 216), opacity=.25,
                                                 scale=scale, **placement)
                continue
            if shape.tag == "box":
                mesh = trimesh.creation.box(extents=np.fromstring(shape.get("size"), sep=" "))
            elif shape.tag == "cylinder":
                mesh = trimesh.creation.cylinder(radius=float(shape.get("radius")), height=float(shape.get("length")))
            elif shape.tag == "sphere":
                mesh = trimesh.creation.icosphere(radius=float(shape.get("radius")), subdivisions=2)
            else:
                raise ValueError(f"Unsupported visual shape: {shape.tag}")
            material = visual.find("material")
            color = None
            if material is not None:
                color = material.find("color")
                if color is None:
                    color = materials.get(material.get("name"))
            rgba = np.fromstring(color.get("rgba"), sep=" ") if color is not None else np.array([.72, .75, .79, 1.])
            if kind == "collision":
                rgba = np.array([.25, .69, .85, .25])
            server.scene.add_mesh_simple(node, mesh.vertices, mesh.faces,
                                         color=tuple((255 * rgba[:3]).astype(int)), opacity=float(rgba[3]), **placement)
    return handles
