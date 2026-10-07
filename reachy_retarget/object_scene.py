"""Build dynamic Reachy scenes from acquired, documented source models.

Source bodies are initialized once in XML. This module never steps a simulator
or modifies a running object's state. Missing collision assets are explicit;
textures may be absent without changing collision geometry or physical values.
"""

import copy
import hashlib
import json
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .episodes import pose_to_matrices
from .physics import attrs, numbers, prepare_robot
from .store import sha256


MANISKILL_REVISION = "ecc579b7567c31bb3d7176539d90e86ae49296be"
MANISKILL_TABLE = "mani_skill/utils/scene_builder/table/assets/table.glb"
# Acquired primary evidence is hash pinned, including the exact SAPIEN version
# required by the source setup.py. No SAPIEN/ManiSkill package is imported.
MANISKILL_EVIDENCE = {
    "setup.py": "c30ce3a5ea5de13ae8b12717f6f52cf4e959c10d68ce98043a9311058aea2e97",
    "sapien-3.0.0.dev2-actor_builder.py": "cf7d1a0fdc9d0a71b0e09cd4b32f4cc63aa2ca1f8b752b750b821629a6651dba",
    "mani_skill/utils/building/actors.py": "c19abb5eb6dc0c6728cd3b8488ebf229e6f52a8477c9d343802fcf0371663ea8",
    "mani_skill/utils/building/actor_builder.py": "5dbb1fde516fc0a55b9b4c21a5fd4d7982bed7282bbe76b7e433548a771fb24c",
    "mani_skill/utils/structs/types.py": "a02b1aa5df652f0908781fae4863d40de1b4dace4deb1e533003030d81d6abe5",
    "mani_skill/envs/sapien_env.py": "23bdd2daa2e9de42b6bd7afeb8c40f3e54948fe51236ccc56ab39d8154965e48",
    "mani_skill/envs/tasks/pick_cube.py": "82041b5fde183b7a496b793315e0b9ab4aac95f6386616c5fa3fd1d986479d66",
    "mani_skill/utils/scene_builder/table/table_scene_builder.py": "4a457e83742dadf4b8b0c2895564d988211ad7b9c40ce2d34ad8757fe2059e68",
    "mani_skill/utils/building/ground.py": "181526a4c508526f854cfe4404cbeb5c4e1304d797bfd50c65ba5bcce1cf0444",
    MANISKILL_TABLE: "cb0ebd8ad6438c1160f095d902bf55415f9d643f8c6990a186a052251fe7951a",
}


class MissingSceneAssets(FileNotFoundError):
    def __init__(self, assets):
        self.assets = assets
        super().__init__("Missing collision assets; explicitly acquire: " + json.dumps(assets))


def _maniskill_evidence(root):
    folder = Path(root) / "data/raw/maniskill_scene_evidence"
    records, missing = [], []
    for relative, digest in MANISKILL_EVIDENCE.items():
        url = ("https://raw.githubusercontent.com/haosulab/SAPIEN/3.0.0.dev2/python/py_package/wrapper/actor_builder.py"
               if relative.startswith("sapien-") else
               "https://raw.githubusercontent.com/haosulab/ManiSkill/" + MANISKILL_REVISION + "/" + relative)
        path = folder / relative
        record = {"path": str(path.resolve()), "url": url, "sha256": digest,
                  "source_revision": "3.0.0.dev2" if relative.startswith("sapien-") else MANISKILL_REVISION}
        if not path.is_file():
            missing.append(record)
        elif sha256(path) != digest:
            raise ValueError("Pinned ManiSkill dynamics evidence checksum mismatch: " + str(path))
        else:
            records.append(record)
    if missing:
        raise MissingSceneAssets(missing)
    return folder, records


def _extrusion_parts(mesh):
    """Decompose the pinned table's X-extruded profiles without filling holes.

    Reuse the original positive-X cap triangulation. Each triangle becomes a
    convex prism. Tiny GLB float jitter along the extrusion axis is measured;
    source volume must agree. Convex source pieces need no decomposition.
    """
    hull = mesh.convex_hull
    source_volume = abs(float(mesh.volume))
    if source_volume <= 0:
        raise ValueError("Table component has no positive source volume")
    if abs(hull.volume - source_volume) < source_volume * 1e-8:
        return [np.asarray(mesh.vertices)], {"method": "verified convex source component",
                    "source_volume": source_volume, "decomposed_volume": float(hull.volume),
                    "max_local_vertex_adjustment_m": 0.0}
    lo, hi = mesh.bounds[:, 0]
    normals = np.asarray(mesh.face_normals)
    chosen = np.where(normals[:, 0] > .99999)[0]
    triangles = np.asarray(mesh.triangles)[chosen]
    if not len(triangles):
        raise ValueError("Table profile has no verified extrusion cap")
    adjustment = max(float(np.max(np.minimum(abs(mesh.vertices[:, 0] - lo), abs(mesh.vertices[:, 0] - hi)))),
                     float(np.max(abs(triangles[:, :, 0] - hi))))
    if adjustment > 2e-7 or np.any((abs(normals[:, 0]) > 1e-5) & (abs(normals[:, 0]) < .99999)):
        raise ValueError("Table shape is not the verified axis extrusion")
    parts = []
    for triangle in triangles:
        lower, upper = triangle.copy(), triangle.copy()
        lower[:, 0], upper[:, 0] = lo, hi
        parts.append(np.vstack([lower, upper]))
    volume = float(np.sum(mesh.area_faces[chosen] * abs(normals[chosen, 0])) * (hi - lo))
    if abs(volume - source_volume) > source_volume * 3e-6:
        raise ValueError("Table convex decomposition changes source volume")
    return parts, {"method": "source cap triangles extruded to convex prisms",
                  "source_volume": source_volume, "decomposed_volume": volume,
                  "max_local_vertex_adjustment_m": adjustment}


def _maniskill_scene(root, metadata, arrays):
    """Documented PhysX-to-MuJoCo reconstruction, not numerical equivalence."""
    import trimesh

    if metadata.get("source_commit") != MANISKILL_REVISION or metadata.get("env_args", {}).get("env_name") != "PickCube-v1":
        raise ValueError("Unverified ManiSkill task/revision; exact dynamics evidence required")
    kwargs = metadata.get("env_args", {}).get("env_kwargs", {})
    if any(key in kwargs for key in ("sim_config", "sim_cfg", "robot_uids")):
        raise ValueError("Unverified ManiSkill simulation configuration override")
    if set(metadata.get("objects", {})) != {"cube"}:
        raise ValueError("Pinned PickCube adapter requires exactly its recorded cube")
    cube = np.asarray(arrays.get("objects/cube/pose"), float)
    table = np.asarray(arrays.get("source/env_states/actors/table-workspace"), float)
    if (cube.ndim != 2 or cube.shape[1] != 7 or not len(cube) or not np.isfinite(cube).all()
            or table.ndim != 2 or table.shape != (len(cube), 13) or not np.isfinite(table).all()):
        raise ValueError("PickCube requires measured cube and static table states")
    if not np.allclose(table, table[0], atol=1e-7) or not np.allclose(table[:, 7:], 0, atol=1e-7):
        raise ValueError("Moving source table cannot be converted to a static fixture")
    if not np.allclose(np.linalg.norm(cube[:, 3:], axis=1), 1, atol=1e-6):
        raise ValueError("Invalid cube quaternion")
    if "objects/cube/valid" in arrays and not arrays["objects/cube/valid"][0]:
        raise ValueError("Invalid initial cube state")
    folder, evidence = _maniskill_evidence(root)
    scene = trimesh.load(folder / MANISKILL_TABLE, force="scene")
    table_height = float(np.ptp(scene.bounds[:, 2]) * 1.75)
    expected_table = _pose([-.12, 0, -table_height], [2**-.5, 0, 0, 2**-.5])
    if not np.allclose(pose_to_matrices(table[0, :7]), expected_table, atol=1e-6, rtol=0):
        raise ValueError("Recorded table pose differs from pinned source builder")
    source = ET.Element("mujoco", model="pinned_maniskill_pickcube_physics_conversion")
    ET.SubElement(source, "compiler", angle="radian", fusestatic="false")
    ET.SubElement(source, "option", gravity="0 0 -9.81", timestep=".002", cone="elliptic")
    # PhysX has two Coulomb coefficients; both are .3 in this source. Zero
    # torsional/rolling friction avoids adding an unrecorded source parameter.
    default = ET.SubElement(source, "default")
    ET.SubElement(default, "geom", friction=".3 0 0", condim="3", solref=".02 1",
                  solimp=".9 .95 .001 .5 2", margin="0", gap="0", density="0")
    asset, world = ET.SubElement(source, "asset"), ET.SubElement(source, "worldbody")
    ET.SubElement(world, "geom", name="maniskill_ground", type="plane", size="20 20 .1",
                  pos=numbers([0, 0, table[0, 2]]), rgba=".6 .6 .6 1")
    fixture = ET.SubElement(world, "body", name="table-workspace", **attrs(pose_to_matrices(table[0, :7])))
    shape_rotation = Rotation.from_euler("z", np.pi / 2).as_matrix()
    derived = Path(root) / "data/assets/maniskill_scene"
    decompositions, derived_assets = {}, []
    for index, node in enumerate(sorted(scene.graph.nodes_geometry)):
        node_transform, key = scene.graph[node]
        mesh = scene.geometry[key]
        parts, check = _extrusion_parts(mesh)
        def vertices_in_fixture(vertices):
            return ((np.asarray(vertices) @ node_transform[:3, :3].T + node_transform[:3, 3]) * 1.75) @ shape_rotation.T
        check.update(parts=len(parts), source_mesh=key, source_node=node,
                     source_volume_m3=check["source_volume"] * abs(np.linalg.det(node_transform[:3, :3])) * 1.75**3,
                     max_vertex_adjustment_world_m=check["max_local_vertex_adjustment_m"] * np.linalg.norm(node_transform[:3, :3], ord=2) * 1.75)
        decompositions[node] = check
        for part_index, vertices in enumerate(parts):
            name = f"ms_table_{index}_collision_{part_index}"
            ET.SubElement(asset, "mesh", name=name, vertex=numbers(vertices_in_fixture(vertices).ravel()))
            ET.SubElement(fixture, "geom", name=name, type="mesh", mesh=name, rgba=".6 .6 .6 0", group="3")
        # Keep the complete source visual triangles and embedded tabletop image.
        name = f"ms_table_{index}_visual"
        visual = ET.SubElement(asset, "mesh", name=name, vertex=numbers(vertices_in_fixture(mesh.vertices).ravel()),
                              face=" ".join(map(str, mesh.faces.ravel())))
        material = mesh.visual.material
        color = getattr(material, "baseColorFactor", None)
        rgba = np.ones(4) if color is None else np.asarray(color) / 255.0
        mat = ET.SubElement(asset, "material", name=name + "_material", rgba=numbers(rgba))
        texture = getattr(material, "baseColorTexture", None)
        if texture is not None:
            derived.mkdir(parents=True, exist_ok=True)
            path = derived / (name + ".png")
            if not path.is_file():
                texture.save(path)
            texture_name = name + "_texture"
            ET.SubElement(asset, "texture", name=texture_name, type="2d", file=str(path.resolve()))
            mat.set("texture", texture_name)
            visual.set("texcoord", numbers(np.asarray(mesh.visual.uv).ravel()))
            derived_assets.append({"path": str(path.resolve()), "sha256": sha256(path),
                                   "derivation": "unchanged embedded table.glb base-color image"})
        ET.SubElement(fixture, "geom", name=name, type="mesh", mesh=name, material=name + "_material",
                      contype="0", conaffinity="0", density="0", group="2")
    body = ET.SubElement(world, "body", name="cube_main", **attrs(pose_to_matrices(cube[0])))
    ET.SubElement(body, "joint", type="free", name="cube_joint0", damping="0", armature="0", frictionloss="0")
    ET.SubElement(body, "geom", name="cube_collision", type="box", size=".02 .02 .02", density="1000", rgba="1 0 0 1")
    _materialize_defaults(source)
    generated_xml = ET.tostring(source, encoding="unicode")
    reference = mujoco.MjModel.from_xml_string(generated_xml)
    data = mujoco.MjData(reference)
    mujoco.mj_forward(reference, data)
    expected_mass = 1000 * .04**3
    bid = reference.body("cube_main").id
    if not np.isclose(reference.body_mass[bid], expected_mass) or not np.allclose(reference.body_inertia[bid], expected_mass * .04**2 / 6):
        raise ValueError("Reconstructed cube mass/inertia differs from source density/geometry")
    return source, reference, data, {
        "source_xml_sha256": None, "generated_source_xml_sha256": hashlib.sha256(generated_xml.encode()).hexdigest(),
        "source_model_kind": "pinned PhysX source geometry and parameters converted to MuJoCo",
        "source_provenance": metadata.get("provenance", []), "acquired_assets": evidence,
        "derived_assets": derived_assets, "omitted_bodies": [{"body": "goal_site", "reason": "source-hidden noncolliding goal marker"}],
        "missing_visual_assets": [{"source_relative_path": "mani_skill/utils/building/assets/floor_tiles_06_2k.glb",
                                   "url": "https://raw.githubusercontent.com/haosulab/ManiSkill/" + MANISKILL_REVISION + "/mani_skill/utils/building/assets/floor_tiles_06_2k.glb",
                                   "source_revision": MANISKILL_REVISION,
                                   "reason": "source ground appearance only; infinite collision plane retained"}],
        "table_convex_decomposition": decompositions,
        "source_physical_parameters": {"cube_half_size_m": [.02] * 3, "cube_density_kg_m3": 1000,
            "cube_mass_kg": expected_mass, "static_friction": .3, "dynamic_friction": .3, "restitution": 0,
            "gravity_m_s2": [0, 0, -9.81], "source_simulation_frequency_hz": 100,
            "source_solver": "PhysX TGS", "source_contact_offset_m": .02, "source_bounce_threshold_m_s": 2.0},
        "conversion_assumptions": [
            "MuJoCo is not numerically equivalent to the source PhysX TGS solver",
            "Equal source static/dynamic friction .3 maps to MuJoCo sliding .3; no source torsional/rolling coefficient is invented",
            "Source restitution zero maps to critically damped MuJoCo solref .02 1; source PhysX contact offset .02 is not a surface inflation",
            "MuJoCo contact mixing and the Reachy contact material replace the source Panda contact-pair behavior",
            "Concave table profiles use source cap-triangle convex prisms; GLB float jitter and volume differences are measured",
            "Recorded static table pose and complete collision geometry are retained; no tabletop-plane substitution",
            "Source goal marker is hidden and noncolliding; its recorded goal remains in normalized arrays"],
    }


def _pose(position, quaternion):
    return pose_to_matrices(np.r_[position, quaternion])


def _set_pose(element, transform):
    for key in ("euler", "axisangle", "xyaxes", "zaxis", "fromto"):
        element.attrib.pop(key, None)
    element.attrib.update(attrs(transform))


def _asset_spec(root, node, metadata):
    original = node.get("file")
    marker = "robosuite/models/assets/"
    relative = posixpath.normpath(original.split(marker, 1)[-1])
    if relative.startswith("/") or relative.startswith("../"):
        raise ValueError("Cannot identify robosuite asset root: " + original)
    version = metadata.get("env_args", {}).get("env_version")
    if version not in {"1.4.1", "1.5.1"}:
        raise ValueError("Unverified asset revision: " + str(version))
    url = "https://raw.githubusercontent.com/ARISE-Initiative/robosuite/v" + version + "/robosuite/models/assets/" + relative
    candidates = [Path(root) / "data/raw/robosuite_assets" / ("v" + version) / relative,
                  Path(root) / "data/raw/robosuite_assets" / relative]
    if version == "1.5.1":
        if relative.startswith("objects/"):
            candidates.append(Path(root) / "data/raw/robosuite_can_assets" / relative.removeprefix("objects/"))
        elif relative.startswith("textures/"):
            candidates.append(Path(root) / "data/raw/robosuite_can_assets" / relative)
    path = next((p for p in candidates if p.is_file()), None)
    record = {"name": node.get("name"), "kind": node.tag, "source_file": original,
              "source_relative_path": relative, "source_revision": "v" + version, "url": url,
              "expected_local_path": str(candidates[0])}
    if path:
        record.update(path=str(path.resolve()), sha256=sha256(path))
    return path, record


def _materialize_defaults(source):
    """Resolve named defaults before filtering unrelated robot assets/classes."""
    classes = {"main": {}}
    def defaults(node, inherited):
        current = copy.deepcopy(inherited)
        for child in node:
            if child.tag != "default":
                current.setdefault(child.tag, {}).update(child.attrib)
        name = node.get("class", "main")
        classes[name] = current
        for child in node.findall("default"):
            defaults(child, current)
    for node in source.findall("default"):
        defaults(node, classes["main"])
    def visit(node, inherited):
        declared = dict(node.attrib)
        class_name = declared.get("class", inherited)
        if class_name not in classes:
            raise ValueError("Unresolved source default class: " + class_name)
        node.attrib.clear()
        node.attrib.update(classes[class_name].get(node.tag, {}))
        node.attrib.update(declared)
        next_class = node.get("childclass", inherited) if node.tag == "body" else inherited
        node.attrib.pop("class", None)
        node.attrib.pop("childclass", None)
        for child in node:
            visit(child, next_class)
    for name in ("worldbody", "asset", "contact", "equality", "actuator", "tendon"):
        node = source.find(name)
        if node is not None:
            visit(node, "main")
    for node in source.findall("default"):
        source.remove(node)


def _source_scene(root, metadata, arrays):
    xml = metadata.get("model_xml")
    if not xml:
        raise ValueError("No recorded source MJCF: exact source dynamics adapter required")
    source = ET.fromstring(xml)
    if source.tag != "mujoco" or source.find("worldbody") is None:
        raise ValueError("Expected complete recorded MJCF")
    for tag in ("include", "attach", "replicate", "composite", "flexcomp", "frame"):
        if source.find(".//" + tag) is not None:
            raise ValueError("Unexpanded MJCF requires explicit adapter: " + tag)
    compiler = source.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(source, "compiler", angle="radian")
    if compiler.get("angle", "degree") != "radian":
        raise ValueError("Only recorded radian MJCF is supported")
    compiler.attrib.update(fusestatic="false")
    for key in ("meshdir", "texturedir", "assetdir"):
        compiler.attrib.pop(key, None)
    _materialize_defaults(source)
    objects = metadata.get("objects", {})
    if not objects:
        raise ValueError("No explicitly normalized object bodies")
    selected = {spec.get("body"): oid for oid, spec in objects.items()}
    if None in selected or len(selected) != len(objects):
        raise ValueError("Object-to-body mapping must be explicit and unique")
    world = source.find("worldbody")
    fixtures = metadata.get("fixtures", {})
    fixture_roots = {spec["body"] for spec in fixtures.values()}
    if fixture_roots & set(selected):
        raise ValueError("Fixture and free-object roots must be distinct")
    source_robots = metadata.get("source_robot_bodies")
    omitted = []
    for body in list(world.findall("body")):
        name = body.get("name", "")
        dynamic = body.find(".//joint") is not None or body.find(".//freejoint") is not None
        source_robot = name in source_robots if source_robots is not None else name.startswith("robot")
        articulated = any(j.get("type", "hinge") != "free" for j in body.findall(".//joint"))
        if articulated and not source_robot and name not in fixture_roots and name not in selected:
            raise ValueError("Articulated scene body requires explicit fixture state; refusing omission: " + name)
        visual_only = bool(body.findall(".//geom")) and all(
            g.get("contype", "1") == "0" and g.get("conaffinity", "1") == "0"
            for g in body.findall(".//geom"))
        if source_robot or (dynamic and name not in selected and name not in fixture_roots) or (visual_only and name not in selected and name not in fixture_roots):
            world.remove(body)
            omitted.append({"body": name, "reason": "source robot replaced by Reachy" if source_robot else
                            "unselected source object outside normalized task scope" if dynamic else "display-only marker without collision"})
            continue
        if name in selected:
            oid = selected[name]
            joints = body.findall("joint") + body.findall("freejoint")
            if len(joints) != 1 or (joints[0].tag != "freejoint" and joints[0].get("type") != "free"):
                raise ValueError("Selected object must retain exactly one free joint: " + name)
            if body.get("mocap", "false") != "false" or float(body.get("gravcomp", "0")) != 0:
                raise ValueError("Object mocap/gravity compensation is forbidden: " + name)
            key = "objects/" + oid + "/pose"
            poses = np.asarray(arrays.get(key))
            if poses.ndim != 2 or poses.shape[1] != 7 or not np.isfinite(poses[0]).all():
                raise ValueError("Missing finite initial object pose: " + key)
            if "objects/" + oid + "/valid" in arrays and not arrays["objects/" + oid + "/valid"][0]:
                raise ValueError("Invalid initial object state: " + oid)
            _set_pose(body, pose_to_matrices(poses[0]))
    kept = {body.get("name") for body in world.findall(".//body")}
    if not set(selected) <= kept:
        raise ValueError("Selected objects are missing direct-world source bodies")
    if not fixture_roots <= kept:
        raise ValueError("Declared articulated fixture is missing from source scene")
    # Source actuators and equalities must only belong to the replaced robot.
    # Any object servo/weld is refused instead of silently replayed as physics.
    object_joints = {j.get("name") for b in world.findall("body") for j in b.findall(".//joint")}
    for section in ("equality", "actuator", "tendon"):
        node = source.find(section)
        if node is not None:
            for constraint in node:
                references = set(constraint.attrib.values())
                if references & (kept | object_joints):
                    raise ValueError("Source object constraint/actuator unsupported: " + ET.tostring(constraint, encoding="unicode"))
            source.remove(node)
    for tag in ("sensor", "keyframe", "custom"):
        for node in source.findall(tag):
            source.remove(node)
    # Assign deterministic names to otherwise anonymous fixture geometry.
    for i, geom in enumerate(world.findall(".//geom")):
        if not geom.get("name"):
            geom.set("name", "source_scene_geom_" + str(i))
    # Only retained assets are needed. Geometry remains unchanged; absence of a
    # texture is explicitly separate from missing collision meshes.
    asset = source.find("asset")
    if asset is None:
        asset = ET.SubElement(source, "asset")
    needed_meshes = {g.get("mesh") for g in world.findall(".//geom") if g.get("mesh")}
    needed_materials = {g.get("material") for g in world.findall(".//geom") if g.get("material")}
    needed_textures = {m.get("texture") for m in asset.findall("material") if m.get("name") in needed_materials and m.get("texture")}
    acquired, missing, missing_visual = [], [], []
    for node in list(asset):
        keep = (node.tag == "mesh" and node.get("name") in needed_meshes or
                node.tag == "material" and node.get("name") in needed_materials or
                node.tag == "texture" and node.get("name") in needed_textures)
        if not keep:
            asset.remove(node)
            continue
        if node.get("file"):
            path, record = _asset_spec(root, node, metadata)
            if path:
                node.set("file", str(path.resolve()))
                acquired.append(record)
            elif node.tag == "mesh":
                missing.append(record)
            else:
                missing_visual.append(record)
                asset.remove(node)
                for material in asset.findall("material"):
                    if material.get("texture") == node.get("name"):
                        material.attrib.pop("texture", None)
    if missing:
        raise MissingSceneAssets(missing)
    present_geoms = {g.get("name") for g in world.findall(".//geom")}
    contact = source.find("contact")
    if contact is not None:
        for pair in list(contact):
            if any(pair.get(k) not in kept for k in ("body1", "body2") if pair.get(k)) or any(
                pair.get(k) not in present_geoms for k in ("geom1", "geom2") if pair.get(k)):
                contact.remove(pair)
    reference = mujoco.MjModel.from_xml_string(ET.tostring(source, encoding="unicode"))
    data = mujoco.MjData(reference)
    fixture_states = {}
    for fid, spec in fixtures.items():
        names = spec.get("joints", [])
        values = np.asarray(arrays.get(f"fixtures/{fid}/joint_position", []), float)
        if not names or values.shape != (len(arrays["time_s"]), len(names)) or not np.isfinite(values).all():
            raise ValueError("Missing aligned finite fixture joint states: " + fid)
        root_id = reference.body(spec["body"]).id
        descendants = {root_id}
        for bid in range(root_id+1,reference.nbody):
            if int(reference.body_parentid[bid]) in descendants:
                descendants.add(bid)
        actual = {reference.joint(j).name for j in range(reference.njnt) if int(reference.jnt_bodyid[j]) in descendants}
        if actual != set(names):
            raise ValueError("Fixture joint coverage must be complete: " + fid)
        for name, value in zip(names, values[0]):
            joint = reference.joint(name).id
            if int(reference.jnt_type[joint]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                raise ValueError("Fixture adapter supports scalar passive hinge/slide joints only")
            data.qpos[reference.jnt_qposadr[joint]] = value
            fixture_states[name] = float(value)
    mujoco.mj_forward(reference, data)
    return source, reference, data, {"source_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "source_provenance": metadata.get("provenance", []), "acquired_assets": acquired,
        "missing_visual_assets": missing_visual, "omitted_bodies": omitted,
        "fixture_initial_joint_positions": fixture_states}


GEOM_TYPES = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}


def _resolved_source_values(world, reference):
    """Prevent Reachy's default contact/material parameters leaking into source."""
    for body in world.findall(".//body"):
        bid = reference.body(body.get("name")).id
        if float(reference.body_gravcomp[bid]) != 0:
            raise ValueError("Source object/environment gravity compensation unsupported")
        body.attrib.pop("childclass", None)
        body.attrib.pop("class", None)
        _set_pose(body, _pose(reference.body_pos[bid], reference.body_quat[bid]))
        inertial = body.find("inertial")
        if inertial is not None:
            body.remove(inertial)
        if reference.body_mass[bid] > 0:
            ET.SubElement(body, "inertial", mass=numbers([reference.body_mass[bid]]),
                          pos=numbers(reference.body_ipos[bid]), quat=numbers(reference.body_iquat[bid]),
                          diaginertia=numbers(reference.body_inertia[bid]))
    for geom in world.findall(".//geom"):
        gid = reference.geom(geom.get("name")).id
        geom.attrib.pop("class", None)
        kind = int(reference.geom_type[gid])
        if kind not in GEOM_TYPES:
            raise ValueError("Unsupported source geom type: " + str(kind))
        if kind != 7:
            _set_pose(geom, _pose(reference.geom_pos[gid], reference.geom_quat[gid]))
            geom.set("type", GEOM_TYPES[kind])
            # MuJoCo accepts unused trailing size components as zero.
            geom.set("size", numbers(reference.geom_size[gid]))
        for key, value in {
            "friction": reference.geom_friction[gid], "solref": reference.geom_solref[gid],
            "solimp": reference.geom_solimp[gid], "solmix": [reference.geom_solmix[gid]],
            "margin": [reference.geom_margin[gid]], "gap": [reference.geom_gap[gid]],
            "rgba": reference.geom_rgba[gid],
        }.items():
            geom.set(key, numbers(value))
        for key in ("contype", "conaffinity", "condim", "priority", "group"):
            geom.set(key, str(int(getattr(reference, "geom_" + key)[gid])))
    for joint in world.findall(".//joint") + world.findall(".//freejoint"):
        joint.attrib.pop("class", None)
        jid = reference.joint(joint.get("name")).id
        kind = int(reference.jnt_type[jid])
        if kind not in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError("Source joint type needs an explicit dynamics adapter")
        start = reference.jnt_dofadr[jid]
        if joint.tag == "freejoint":
            joint.tag = "joint"
            joint.set("type", "free")
        width = 6 if kind == mujoco.mjtJoint.mjJNT_FREE else 1
        for name in ("damping", "armature", "frictionloss"):
            values = getattr(reference, "dof_" + name)[start:start + width]
            if not np.allclose(values, values[0], rtol=0, atol=1e-12):
                raise ValueError("Nonuniform free-joint dynamics need explicit adapter")
            joint.set(name, numbers([values[0]]))
        joint.set("stiffness", numbers([reference.jnt_stiffness[jid]]))
        joint.set("solreffriction", numbers(reference.dof_solref[start]))
        joint.set("solimpfriction", numbers(reference.dof_solimp[start]))
        if width == 1:
            joint.set("type", "hinge" if kind == mujoco.mjtJoint.mjJNT_HINGE else "slide")
            for key, value in {"axis":reference.jnt_axis[jid],"pos":reference.jnt_pos[jid],
                               "range":reference.jnt_range[jid],"solreflimit":reference.jnt_solref[jid],
                               "solimplimit":reference.jnt_solimp[jid]}.items():
                joint.set(key,numbers(value))
            joint.set("limited", "true" if reference.jnt_limited[jid] else "false")
            address = reference.jnt_qposadr[jid]
            joint.set("ref",numbers([reference.qpos0[address]]))
            joint.set("springref",numbers([reference.qpos_spring[address]]))
            joint.set("margin",numbers([reference.jnt_margin[jid]]))


def initialize_fixtures(model, data, manifest):
    """Apply source passive joint positions once, before physical integration."""
    if data.time != 0:
        raise ValueError("Fixture initialization is reset-only")
    for name,value in manifest.get("fixture_initial_joint_positions", {}).items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model,data)


def build_scene(root, metadata, arrays, placement, *, state_profile=False):
    """Return XML and a source-fidelity manifest for one reset-only scene.

    ``manifest['objects'][oid]`` contains compiled body/joint IDs, names, qpos
    and qvel addresses, source mass/inertias and all collision geom IDs. The
    caller initializes robot joints separately and never assigns object state
    after simulation begins.
    """
    root = Path(root).resolve()
    placement = np.asarray(placement, float)
    if (placement.shape != (4, 4) or not np.isfinite(placement).all()
            or not np.allclose(placement[3], [0, 0, 0, 1])
            or not np.allclose(placement[:3, :3].T @ placement[:3, :3], np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(placement[:3, :3]), 1)
            or not np.allclose(placement[2, :3], [0, 0, 1], atol=1e-8)):
        raise ValueError("Scene placement must be a finite, gravity-preserving rigid transform")
    if not metadata.get("model_xml") and metadata.get("source_format") == "ManiSkill-HDF5":
        source, reference, source_data, provenance = _maniskill_scene(root, metadata, arrays)
        object_specs = {"cube": {"body": "cube_main", "joint": "cube_joint0"}}
    else:
        source, reference, source_data, provenance = _source_scene(root, metadata, arrays)
        object_specs = metadata["objects"]
    folder = root / "data/assets" / ("reachy_mujoco_state" if state_profile else "reachy_mujoco")
    if (folder / "robot.xml").exists() and (folder / "manifest.json").exists():
        robot = ET.parse(folder / "robot.xml").getroot()
        robot_manifest = json.loads((folder / "manifest.json").read_text())
    else:
        robot, robot_manifest = prepare_robot(root, state_profile=state_profile)
    original_robot = mujoco.MjModel.from_xml_string(ET.tostring(robot, encoding="unicode"))
    robot_data = mujoco.MjData(original_robot)
    mujoco.mj_forward(original_robot, robot_data)
    world = robot.find("worldbody")
    source_world = copy.deepcopy(source.find("worldbody"))
    _resolved_source_values(source_world, reference)
    # Replace the generic floor with the actual source arena floor.
    for geom in list(world.findall("geom")):
        if geom.get("name") == "floor":
            world.remove(geom)
    for body in source_world.findall("body"):
        sid = reference.body(body.get("name")).id
        _set_pose(body, placement @ _pose(reference.body_pos[sid], reference.body_quat[sid]))
    for geom in source_world.findall("geom"):
        sid = reference.geom(geom.get("name")).id
        _set_pose(geom, placement @ _pose(reference.geom_pos[sid], reference.geom_quat[sid]))
    # Cameras/lights are presentation settings; do not import source robot refs.
    for node in source_world:
        if node.tag in {"body", "geom", "site"}:
            world.append(node)
    for node in source.find("asset"):
        robot.find("asset").append(copy.deepcopy(node))
    if source.find("contact") is not None:
        for node in source.find("contact"):
            robot.find("contact").append(copy.deepcopy(node))
    option = robot.find("option")
    for key in ("density", "viscosity", "impratio"):
        option.set(key, numbers([getattr(reference.opt, key)]))
    for key in ("gravity", "wind"):
        option.set(key, numbers(placement[:3, :3] @ getattr(reference.opt, key)))
    option.set("cone", "elliptic" if reference.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC else "pyramidal")
    xml = ET.tostring(robot, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    initialize_fixtures(model, data, provenance)
    max_inertial_error = max_geom_error = max_world_error = max_robot_fk_error = max_joint_error = max_geom_world_error = 0.0
    body_records = {}
    geom_records = {}
    for sid in range(1, reference.nbody):
        name = reference.body(sid).name
        bid = model.body(name).id
        fields = ("body_mass", "body_ipos") if reference.body_mass[sid] > 0 else ("body_mass",)
        # MuJoCo's massless carrier ipos is not a center of mass and can follow
        # the world reset placement (e.g. ToolHang tool_root). Its physical
        # inertia lives on fixed child bodies, which are checked individually.
        for field in fields:
            max_inertial_error = max(max_inertial_error, float(np.max(np.abs(getattr(reference, field)[sid] - getattr(model, field)[bid]))))
        source_rotation = Rotation.from_quat(reference.body_iquat[sid][[1, 2, 3, 0]]).as_matrix()
        merged_rotation = Rotation.from_quat(model.body_iquat[bid][[1, 2, 3, 0]]).as_matrix()
        source_tensor = source_rotation @ np.diag(reference.body_inertia[sid]) @ source_rotation.T
        merged_tensor = merged_rotation @ np.diag(model.body_inertia[bid]) @ merged_rotation.T
        # Repeated principal moments admit different equivalent eigenframes;
        # compare the physical tensor, not quaternion signs or eigenvalue order.
        max_inertial_error = max(max_inertial_error, float(np.max(np.abs(source_tensor - merged_tensor))))
        expected = placement @ _pose(source_data.xpos[sid], source_data.xquat[sid])
        observed = _pose(data.xpos[bid], data.xquat[bid])
        max_world_error = max(max_world_error, float(np.max(np.abs(expected - observed))))
        body_records[name] = {"mass_kg": float(reference.body_mass[sid]), "inertia_kg_m2": reference.body_inertia[sid].tolist(),
                              "inertial_position_m": reference.body_ipos[sid].tolist(), "inertial_quaternion_wxyz": reference.body_iquat[sid].tolist(),
                              "inertia_tensor_body_frame_kg_m2": source_tensor.tolist()}
    for sid in range(reference.ngeom):
        name = reference.geom(sid).name
        gid = model.geom(name).id
        for field in ("geom_type", "geom_size", "geom_friction", "geom_solref", "geom_solimp", "geom_solmix", "geom_margin", "geom_gap", "geom_condim", "geom_contype", "geom_conaffinity"):
            max_geom_error = max(max_geom_error, float(np.max(np.abs(getattr(reference, field)[sid] - getattr(model, field)[gid]))))
        expected_position = (placement @ np.r_[source_data.geom_xpos[sid], 1])[:3]
        expected_rotation = placement[:3, :3] @ source_data.geom_xmat[sid].reshape(3, 3)
        max_geom_world_error = max(max_geom_world_error, float(np.max(np.abs(expected_position - data.geom_xpos[gid]))),
                                  float(np.max(np.abs(expected_rotation - data.geom_xmat[gid].reshape(3, 3)))))
        if reference.geom_contype[sid] or reference.geom_conaffinity[sid]:
            geom_records[name] = {"body": reference.body(int(reference.geom_bodyid[sid])).name,
                                 "type": GEOM_TYPES[int(reference.geom_type[sid])], "size": reference.geom_size[sid].tolist(),
                                 "friction": reference.geom_friction[sid].tolist(), "solref": reference.geom_solref[sid].tolist(),
                                 "solimp": reference.geom_solimp[sid].tolist(), "condim": int(reference.geom_condim[sid]),
                                 "contype": int(reference.geom_contype[sid]), "conaffinity": int(reference.geom_conaffinity[sid]),
                                 "margin": float(reference.geom_margin[sid]), "gap": float(reference.geom_gap[sid])}
    for sid in range(reference.njnt):
        jid = model.joint(reference.joint(sid).name).id
        sa, ma = int(reference.jnt_dofadr[sid]), int(model.jnt_dofadr[jid])
        width = 6 if reference.jnt_type[sid] == mujoco.mjtJoint.mjJNT_FREE else 1
        for field in ("dof_damping", "dof_armature", "dof_frictionloss", "dof_solref", "dof_solimp"):
            max_joint_error = max(max_joint_error, float(np.max(np.abs(getattr(reference, field)[sa:sa+width] - getattr(model, field)[ma:ma+width]))))
    for sid in range(1, original_robot.nbody):
        bid = model.body(original_robot.body(sid).name).id
        max_robot_fk_error = max(max_robot_fk_error, float(np.max(np.abs(_pose(robot_data.xpos[sid], robot_data.xquat[sid]) - _pose(data.xpos[bid], data.xquat[bid])))))
    if max(max_inertial_error, max_geom_error, max_world_error, max_robot_fk_error, max_joint_error, max_geom_world_error) > 1e-7:
        raise ValueError(f"Source scene fidelity verification failed: inertia={max_inertial_error}, geom={max_geom_error}, world={max_world_error}, robot={max_robot_fk_error}, joint={max_joint_error}, geom_world={max_geom_world_error}")
    objects = {}
    for oid, spec in object_specs.items():
        name = spec["body"]
        bid = model.body(name).id
        jid = int(model.body_jntadr[bid])
        descendants = {bid}
        for child in range(bid + 1, model.nbody):
            if int(model.body_parentid[child]) in descendants:
                descendants.add(child)
        geoms = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in descendants and (model.geom_contype[i] or model.geom_conaffinity[i])]
        expected = placement @ pose_to_matrices(np.asarray(arrays["objects/" + oid + "/pose"])[0])
        if not np.allclose(_pose(data.xpos[bid], data.xquat[bid]), expected, rtol=0, atol=1e-8):
            raise ValueError("Compiled object reset pose differs from normalized source: " + oid)
        objects[oid] = {"body": name, "body_id": bid, "joint": model.joint(jid).name, "joint_id": jid,
                        "qpos_address": int(model.jnt_qposadr[jid]), "qvel_address": int(model.jnt_dofadr[jid]),
                        "collision_geom_ids": geoms, "collision_geom_names": [model.geom(i).name for i in geoms],
                        "initial_pose_world": np.r_[data.xpos[bid], data.xquat[bid]].tolist(),
                        "source_root_inertial": body_records[name],
                        "mass_kg": float(model.body_subtreemass[bid]),
                        "rigid_component_masses_kg": {model.body(i).name: float(model.body_mass[i]) for i in sorted(descendants)}}
    return xml, {**robot_manifest, **provenance, "source_environment": metadata.get("env_args", {}).get("env_name"),
                 "objects": objects, "source_bodies": body_records, "source_collision_geoms": geom_records,
                 "source_environment_options": {"density": float(reference.opt.density), "viscosity": float(reference.opt.viscosity), "impratio": float(reference.opt.impratio), "cone": option.get("cone")},
                 "world_placement": placement.tolist(),
                 "scene_sha256": hashlib.sha256(xml.encode()).hexdigest(),
                 "source_material_contact_policy": ("Pinned source PhysX parameters mapped explicitly to MuJoCo; see conversion_assumptions" if provenance.get("conversion_assumptions") else
                                                    "Source-compiled physical fields explicitly resolved before merging; source body mass/inertia retained; robot-only defaults cannot alter source bodies"),
                 "object_state_assignment": "initial XML pose only; dynamic free joint for every subsequent physics step",
                 "object_welds_actuators_gravitycomp": False,
                 "validation": {"max_body_inertial_error": max_inertial_error, "max_geom_field_error": max_geom_error,
                                "max_source_world_fk_error": max_world_error, "max_robot_world_fk_error": max_robot_fk_error,
                                "max_joint_dynamics_error": max_joint_error, "max_geom_world_fk_error": max_geom_world_error},
                 "simulation_assumptions": provenance.get("conversion_assumptions", []) + ["Source-derived simulator properties are not measured real-object properties",
                                            "MuJoCo version and robot servo settings may differ from source collection runtime",
                                            "Object velocities start at zero at reset; only the recorded initial object poses are transplanted",
                                            "Missing textures affect appearance only and are enumerated; collision meshes are mandatory"]}
