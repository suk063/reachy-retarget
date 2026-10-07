"""Offline visual assets for recorded-state playback, never physical validation."""

import copy
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .episodes import matrices_to_pose, pose_to_matrices
from .physics import attrs, numbers, origin, prepare_robot
from .robot import CONTROL
from .store import json_write, sha256


def first_surface_hit(triangles, start, direction):
    """Nearest positive ray/triangle hit, without an optional spatial-index dependency."""
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(direction, edge2)
    determinant = np.einsum("ij,ij->i", edge1, cross)
    valid = np.abs(determinant) > 1e-10
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1 / determinant[valid]
    delta = start - triangles[:, 0]
    u = inverse * np.einsum("ij,ij->i", delta, cross)
    q = np.cross(delta, edge1)
    v = inverse * (q @ direction)
    distance = inverse * np.einsum("ij,ij->i", edge2, q)
    hits = valid & (u >= 0) & (v >= 0) & (u+v <= 1) & (distance > 1e-8)
    if not hits.any():
        raise ValueError("Tripod support axis does not intersect the torso visual mesh")
    return float(distance[hits].min())


def extend_front_supports(root):
    """Close the remaining front support gaps at the declared fixed tripod pose."""
    supports = [(side, root.find(f"link[@name='{side}_bar_inner']/visual")) for side in ("left", "right")]
    if not any(v is not None for _, v in supports):
        return []
    torso = root.find("link[@name='torso']/visual")
    shape = torso.find("geometry/mesh")
    mesh = trimesh.load_scene(shape.get("filename"), process=False).to_mesh()
    mesh.vertices *= np.fromstring(shape.get("scale", "1 1 1"), sep=" ")
    mesh.apply_transform(origin(torso))
    changes = []
    for side, visual in supports:
        if visual is None:
            continue
        joint = root.find(f"joint[@name='{side}_bar_joint_mimic']")
        mimic = joint.find("mimic")
        if mimic.get("joint") != "tripod_joint":
            raise ValueError("Unexpected tripod support linkage")
        transform = origin(joint)
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        transform[:3, :3] = transform[:3, :3] @ Rotation.from_rotvec(axis * float(mimic.get("offset", 0))).as_matrix()
        start, direction = transform[:3, 3], transform[:3, 2]
        distance = first_surface_hit(mesh.triangles, start, direction)
        box = visual.find("geometry/box")
        o = visual.find("origin")
        size = np.fromstring(box.get("size"), sep=" ")
        xyz = np.fromstring(o.get("xyz"), sep=" ")
        if not np.allclose(origin(visual)[:3, :3], np.eye(3)):
            raise ValueError("Unsupported rotated tripod visual")
        lower = xyz[2] - size[2]/2
        corner_distances = [first_surface_hit(mesh.triangles,
            start + transform[:3, :2] @ (xyz[:2] + size[:2]*[sx, sy]/2), direction)
            for sx in (-1, 1) for sy in (-1, 1)]
        # Reach the shell across the entire tilted square end, not just its center.
        upper = max(corner_distances) + 0.002
        before = ET.tostring(visual, encoding="unicode")
        size[2], xyz[2] = upper-lower, (upper+lower)/2
        box.set("size", numbers(size))
        o.set("xyz", numbers(xyz))
        changes.append({"kind": f"{side}_bar_inner visual extension", "before": before,
                        "after": ET.tostring(visual, encoding="unicode"), "surface_distance_m": distance,
                        "corner_surface_distances_m": corner_distances, "upper_extent_m": upper,
                        "torso_overlap_m": .002, "tripod_reference_position_rad_or_m": 0.,
                        "torso_surface_point": (start+distance*direction).tolist(),
                        "policy": "Extend visual along existing support axis; preserve lower endpoint, joint and collider"})
    return changes


def visual_urdf(folder):
    """Apply reachy-control's documented display repair to a local URDF copy."""
    source = CONTROL / "asset/reachy.urdf"
    tree = ET.parse(source)
    root = tree.getroot()
    changes = []
    for mesh in root.findall(".//mesh"):
        uri = mesh.get("filename")
        if "/share/" in uri:
            relative = uri.split("/share/", 1)[1]
        elif uri.startswith("package://"):
            relative = uri.removeprefix("package://")
        else:
            raise ValueError(f"Unsupported source mesh URI: {uri}")
        path = CONTROL / "asset/packages" / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set("filename", str(path))
        changes.append({"from": uri, "to": str(path), "kind": "local mesh path"})
    support = root.find("link[@name='back_bar_inner']/visual")
    if support is not None:
        box = support.find("geometry/box")
        o = support.find("origin")
        height = float(root.find("joint[@name='torso_base']/origin").get("xyz").split()[2])
        before = ET.tostring(support, encoding="unicode")
        size = np.fromstring(box.get("size"), sep=" ")
        size[2] = height
        xyz = np.fromstring(o.get("xyz", "0 0 0"), sep=" ")
        xyz[2] = height / 2
        box.set("size", numbers(size))
        o.set("xyz", numbers(xyz))
        changes.append({"kind": "back_bar_inner visual extension", "before": before,
                        "after": ET.tostring(support, encoding="unicode"),
                        "reference": "reachy-control/util/visualization.py:add_robot_visuals"})
    changes.extend(extend_front_supports(root))
    path = folder / "reachy.visual.urdf"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return root, {"source_urdf": str(source), "source_urdf_sha256": sha256(source),
                  "visual_urdf": str(path), "visual_urdf_sha256": sha256(path), "changes": changes}


def prepare_scene(motion, report, *, physical=False):
    """Add only source Can/bin assets, with the same rigid placement as retargeting."""
    folder = motion.parent / "viewer"
    if physical:
        folder.mkdir(exist_ok=True)
        original_scene = motion.with_name("scene.xml")
        if sha256(original_scene) != report["scene_sha256"]:
            raise ValueError("Recorded physical scene checksum mismatch")
        mj = ET.parse(original_scene).getroot()
        manifest = copy.deepcopy(report["physics"])
        manifest["recorded_scene_sha256"] = report["scene_sha256"]
    else:
        mj, manifest = prepare_robot(folder, include_neck=True)
    for geom in mj.findall(".//geom"):
        if "_collision_" in geom.get("name", ""):
            geom.set("group", "3")
    for site in mj.findall(".//site"):
        site.set("group", "3")
    urdf, visual_manifest = visual_urdf(folder)
    expected = [v for k, v in report["controller_identity"].items() if k.endswith("/asset/reachy.urdf")]
    if expected != [visual_manifest["source_urdf_sha256"]]:
        raise ValueError("Viewer URDF differs from the retargeted source model")
    visual_manifest["source_repository"] = "https://github.com/suk063/reachy-control"
    visual_manifest["source_revision"] = subprocess.check_output(
        ["git", "-C", str(CONTROL), "rev-parse", "HEAD"], text=True).strip()
    visual_manifest["visualization_reference_sha256"] = sha256(CONTROL / "util/visualization.py")
    manifest["objects"] = {}
    manifest["environment"] = []
    manifest["source_assets"] = {}
    manifest["visual_mesh_projection"] = []
    with h5py.File(motion, "r") as f:
        objects = ["Can"] if physical else list(f.get("objects", {}))
        if objects and objects != ["Can"]:
            raise ValueError(f"Object assets are not supported yet: {objects}")
        if objects:
            channel = "simulation/Can_pose" if physical else "objects/Can/pose"
            poses = f[channel][:]
            valid = f["objects/Can/valid"][:] if "objects/Can/valid" in f else np.isfinite(poses).all(axis=1)
            if not valid.all() or not np.isfinite(poses).all():
                raise ValueError("Can playback requires an uninterrupted valid track; missing poses will not be filled")
            dataset_root = motion.parents[4]
            normalized = dataset_root / "data/normalized" / report["source_id"] / (report["source_episode"] + ".json")
            meta = json.loads(normalized.read_text())
            source = ET.fromstring(meta["model_xml"])
            asset = mj.find("asset")
            source_asset = source.find("asset")
            assets_root = dataset_root / "data/raw/robosuite_can_assets"
            source_bodies = [source.find(f".//body[@name='{n}']") for n in ("bin1", "bin2", "Can_main")]
            materials = {g.get("material") for b in source_bodies for g in b.iter("geom") if g.get("material")}
            textures = set()
            for name in sorted(materials):
                material = copy.deepcopy(source_asset.find(f"material[@name='{name}']"))
                if material.get("texture"):
                    textures.add(material.get("texture"))
                asset.append(material)
            for name in sorted(textures):
                texture = copy.deepcopy(source_asset.find(f"texture[@name='{name}']"))
                path = assets_root / "textures" / Path(texture.get("file")).name
                manifest["source_assets"][str(path)] = sha256(path)
                texture.set("file", str(path))
                asset.append(texture)
            mesh = copy.deepcopy(source_asset.find("mesh[@name='Can_can_mesh']"))
            mesh_path = assets_root / "meshes/can.msh"
            mesh.set("file", str(mesh_path))
            manifest["source_assets"][str(mesh_path)] = sha256(mesh_path)
            if asset.find("mesh[@name='Can_can_mesh']") is None:
                asset.append(mesh)
            placement = np.array(report["conversion"]["world_placement"] if physical else report["world_placement"])
            for b in source_bodies:
                name = b.get("name")
                if physical:
                    body = mj.find(f".//body[@name='{name}']")
                    for g in body.iter("geom"):
                        g.set("group", "3")
                    for g in b.iter("geom"):
                        if g.get("contype") == "0":
                            visual = copy.deepcopy(g)
                            visual.set("group", "2")
                            visual.set("mass", "0")
                            body.append(visual)
                else:
                    body = copy.deepcopy(b)
                if name == "Can_main":
                    if not physical:
                        body.attrib.update(attrs(pose_to_matrices(poses[0])))
                    manifest["objects"]["Can"] = {"body": name, "joint": "Can_joint0", "pose_channel": channel}
                else:
                    T = np.eye(4)
                    T[:3, 3] = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
                    if body.get("quat"):
                        T = pose_to_matrices(np.r_[T[:3, 3], np.fromstring(body.get("quat"), sep=" ")])
                    if not physical:
                        body.attrib.update(attrs(placement @ T))
                    manifest["environment"].append(name)
                for gi, g in enumerate(body.iter("geom")):
                    if not physical:
                        g.set("group", "2" if g.get("contype") == "0" else "3")
                    # mjviser omits 2D textures on primitive geoms. Supply a
                    # face-split UV mesh for display, leaving colliders intact.
                    if g.get("group") == "2" and g.get("type") == "box":
                        half_size = np.fromstring(g.get("size"), sep=" ")
                        visual_mesh = trimesh.creation.box(extents=2*half_size)
                        visual_mesh.unmerge_vertices()
                        uv = np.zeros((len(visual_mesh.vertices), 2))
                        material = source_asset.find(f"material[@name='{g.get('material')}']")
                        repeat = np.fromstring(material.get("texrepeat", "1 1"), sep=" ")
                        for face, normal in zip(visual_mesh.faces, visual_mesh.face_normals):
                            axes = [a for a in range(3) if a != np.argmax(np.abs(normal))]
                            uv[face] = (visual_mesh.vertices[face][:, axes] + half_size[axes]) * repeat
                        visual_mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
                        mesh_name = f"viewer_{name}_{gi}"
                        target = folder / (mesh_name + ".obj")
                        obj, _ = trimesh.exchange.obj.export_obj(visual_mesh, include_texture=True, return_texture=True)
                        target.write_text(obj)
                        ET.SubElement(asset, "mesh", name=mesh_name, file=str(target))
                        g.set("type", "mesh")
                        g.set("mesh", mesh_name)
                        g.attrib.pop("size")
                        manifest["visual_mesh_projection"].append({"body": name, "mesh": mesh_name,
                            "sha256": sha256(target), "policy": "face-local planar UVs multiplied by source texrepeat; display approximation"})
                for s in body.iter("site"):
                    s.set("group", "3")
                if not physical:
                    mj.find("worldbody").append(body)
            manifest["normalized_sidecar_sha256"] = sha256(normalized)
    path = folder / "scene.xml"
    ET.ElementTree(mj).write(path, encoding="unicode")
    manifest["mode"] = "recorded physics replay" if physical else "recorded kinematic replay; no physics steps"
    manifest["object_pose_policy"] = ("Replay actual simulation/qpos; retain measured finger mimic deviations"
                                      if physical else "Replay source-derived retargeted world poses; no contact success inference")
    json_write(folder / "scene-manifest.json", manifest)
    return path, manifest, urdf, visual_manifest


class RobotVisuals:
    """Preserve the source DAE materials/UVs via GLB, driven by MuJoCo body poses."""

    def __init__(self, server, model, urdf, folder, manifest):
        self.root = server.scene.add_frame("/fixed_bodies/robot_visuals", show_axes=False)
        self.handles = {}
        colors = {e.get("name"): e.find("color") for e in urdf.findall("material")}
        manifest["meshes"] = {}
        manifest["visual_count"] = 0
        cache = {}
        for link in urdf.findall("link"):
            visuals = link.findall("visual")
            if not visuals:
                continue
            name = link.get("name")
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body < 0:
                raise ValueError(f"Visual link missing from MuJoCo model: {name}")
            parent = f"/fixed_bodies/robot_visuals/{name}"
            self.handles[body] = server.scene.add_frame(parent, show_axes=False)
            for index, visual in enumerate(visuals):
                manifest["visual_count"] += 1
                pose = matrices_to_pose(origin(visual))
                kwargs = dict(position=pose[:3], wxyz=pose[3:])
                node = f"{parent}/{index}"
                shape = visual.find("geometry")[0]
                if shape.tag == "mesh":
                    path = Path(shape.get("filename"))
                    if path not in cache:
                        scene = trimesh.load_scene(path, process=False)
                        cache[path] = scene.export(file_type="glb")
                        manifest["meshes"][str(path)] = {"sha256": sha256(path), "parts": len(scene.geometry),
                                                        "glb_sha256": hashlib.sha256(cache[path]).hexdigest()}
                    scale = tuple(np.fromstring(shape.get("scale", "1 1 1"), sep=" "))
                    server.scene.add_glb(node, cache[path], scale=scale, **kwargs)
                    continue
                if shape.tag == "box":
                    mesh = trimesh.creation.box(extents=np.fromstring(shape.get("size"), sep=" "))
                elif shape.tag == "cylinder":
                    mesh = trimesh.creation.cylinder(radius=float(shape.get("radius")), height=float(shape.get("length")))
                elif shape.tag == "sphere":
                    mesh = trimesh.creation.icosphere(radius=float(shape.get("radius")), subdivisions=2)
                else:
                    raise ValueError(f"Unsupported visual: {shape.tag}")
                material = visual.find("material")
                color = None if material is None else material.find("color")
                if color is None and material is not None:
                    color = colors.get(material.get("name"))
                rgba = np.fromstring(color.get("rgba"), sep=" ") if color is not None else np.array([.72, .75, .79, 1.])
                server.scene.add_mesh_simple(node, mesh.vertices, mesh.faces,
                    color=tuple((255*rgba[:3]).astype(int)), opacity=float(rgba[3]), **kwargs)
        json_write(folder / "visual-manifest.json", manifest)

    def update(self, data):
        for body, handle in self.handles.items():
            handle.position = data.xpos[body]
            handle.wxyz = data.xquat[body]
