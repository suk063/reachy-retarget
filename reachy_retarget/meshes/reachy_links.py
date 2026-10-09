"""Reachy 2 link components: visual meshes from the vendored COLLADA files, collision parts as in
tier P (``robot.mjcf``), and per-frame link poses from canonical ``q``.

Visuals (as reachy-agent ``simulation/prepare.py``): every URDF ``<visual>`` of a link; mesh
visuals are read from ``robot/assets/packages`` (COLLADA, Z up, unit applied), split into one part
per (instanced geometry, material) with the node transforms, the visual origin and the mesh
scale baked in, so each part is in the link frame with an identity pose. The colour of a part is
the URDF material colour when the visual names one (explicit ``<color>`` or a robot-level
material, e.g. the antennas' ``neckwhite``), else the COLLADA effect's diffuse colour; the raw
COLLADA effect values (specular, shininess, emission) are recorded with each material. The
COLLADA files carry no texture images. Box/cylinder/sphere visuals (tripod bars) are primitives
with a generated mesh. Meshes are written once per library (identical bytes, identical names).

Collision parts: URDF ``<collision>`` primitives, and for collider meshes the convex hulls of their
connected components (``robot.mjcf.collider_parts``, the shapes tier P collides with).

Poses: ``base_link`` = ``planar(q[0:3])``; every other link by forward kinematics over the arm,
neck and finger joints (``q[3:22]``; mimic joints follow their source; tripod and antennas at
0, as in ``robot.reachy``). Component names are ``reachy/<link>``.
"""
from __future__ import annotations

import functools
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull

from ..robot import mjcf as rmjcf
from ..robot.reachy import FIXED, JOINTS, planar
from ..robot.resources import ASSETS, URDF_SHA256, urdf
from ..robot.urdf import KinematicTree
from ..schema.rotations import pose_to_vec7
from ..schema.scene_assets import encode_msh, primitive_mesh

PREFIX = "reachy/"
SKIPPED: list[dict] = []  # visual parts left out (fewer than 4 distinct vertices)
VISUAL_GROUP = 2   # reachy-agent's visual group
_C = "{http://www.collada.org/2005/11/COLLADASchema}"


def _origin(e) -> np.ndarray:
    return rmjcf._origin(e)


def _floats(text) -> np.ndarray:
    return np.array([float(x) for x in (text or "").split()])


def _effects(root) -> dict:
    """{material id: {"rgba", "specular", "shininess", "emission"}} from COLLADA effects."""
    effects = {}
    for eff in root.iter(f"{_C}effect"):
        tech = eff.find(f"{_C}profile_COMMON/{_C}technique")
        shader = next(iter(tech), None) if tech is not None else None
        rec = {}
        if shader is not None:
            for key in ("diffuse", "specular", "emission"):
                c = shader.find(f"{_C}{key}/{_C}color")
                if c is not None:
                    rec[key] = _floats(c.text).round(6).tolist()
            s = shader.find(f"{_C}shininess/{_C}float")
            if s is not None:
                rec["shininess"] = float(s.text)
        effects[eff.get("id")] = rec
    out = {}
    for mat in root.iter(f"{_C}material"):
        inst = mat.find(f"{_C}instance_effect")
        if inst is not None:
            out[mat.get("id")] = effects.get(inst.get("url", "").lstrip("#"), {})
    return out


def _primitive_corners(prim, inputs, p, counts=None):
    """Per-corner index rows (n_corners, stride) of a <triangles>/<polylist>, fan-triangulated."""
    stride = max(int(i.get("offset", 0)) for i in inputs) + 1
    idx = p.reshape(-1, stride)
    if counts is None:
        return idx
    rows, start = [], 0
    for c in counts:
        poly = idx[start:start + c]
        for k in range(1, c - 1):
            rows += [poly[0], poly[k], poly[k + 1]]
        start += c
    return np.array(rows, int).reshape(-1, stride)


def read_collada_parts(path) -> list[dict]:
    """Parts ``{"vertices", "normals" | None, "faces", "material": {...}, "material_id"}`` of a COLLADA
    file, with node transforms and the unit applied (vertices split per position/normal pair)."""
    root = ET.parse(path).getroot()
    unit = root.find(f"{_C}asset/{_C}unit")
    scale = float(unit.get("meter", 1.0)) if unit is not None else 1.0
    up = root.find(f"{_C}asset/{_C}up_axis")
    if up is not None and up.text.strip() != "Z_UP":
        raise ValueError(f"{path}: only Z_UP COLLADA files are supported")
    materials = _effects(root)
    geometries = {g.get("id"): g for g in root.iter(f"{_C}geometry")}
    library_nodes = {n.get("id"): n for lib in root.iter(f"{_C}library_nodes") for n in lib.iter(f"{_C}node")}
    parts = []

    def geometry_parts(geom, T, binding):
        mesh = geom.find(f"{_C}mesh")
        arrays = {s.get("id"): _floats(s.find(f"{_C}float_array").text) for s in mesh.findall(f"{_C}source")}
        vert = mesh.find(f"{_C}vertices")
        vin = {i.get("semantic"): i.get("source").lstrip("#") for i in vert.findall(f"{_C}input")}
        positions = arrays[vin["POSITION"]].reshape(-1, 3)
        vnormals = arrays[vin["NORMAL"]].reshape(-1, 3) if "NORMAL" in vin else None
        if mesh.find(f"{_C}polygons") is not None:
            raise ValueError(f"{path}: COLLADA <polygons> primitives are not supported")
        for prim in list(mesh.findall(f"{_C}triangles")) + list(mesh.findall(f"{_C}polylist")):
            inputs = prim.findall(f"{_C}input")
            pe = prim.find(f"{_C}p")
            if pe is None or not pe.text:
                continue
            p = np.array([int(x) for x in pe.text.split()])
            counts = ([int(x) for x in prim.find(f"{_C}vcount").text.split()]
                      if prim.tag.endswith("polylist") else None)
            corners = _primitive_corners(prim, inputs, p, counts)
            sem = {i.get("semantic"): i for i in inputs}
            vi = corners[:, int(sem["VERTEX"].get("offset", 0))]
            pos = positions[vi]
            if "NORMAL" in sem:
                nrm = arrays[sem["NORMAL"].get("source").lstrip("#")].reshape(-1, 3)[
                    corners[:, int(sem["NORMAL"].get("offset", 0))]]
            elif vnormals is not None:
                nrm = vnormals[vi]
            else:
                nrm = None
            pos = (np.c_[pos, np.ones(len(pos))] @ T.T)[:, :3] * scale
            if nrm is not None:
                N = np.linalg.inv(T[:3, :3]).T
                nrm = nrm @ N.T
                nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            key = np.round(np.c_[pos, nrm if nrm is not None else np.zeros((len(pos), 0))], 9)
            uniq, inverse = np.unique(key, axis=0, return_inverse=True)
            symbol = prim.get("material")
            mid = binding.get(symbol, symbol)
            parts.append({"vertices": uniq[:, :3], "normals": uniq[:, 3:6] if nrm is not None else None,
                          "faces": inverse.reshape(-1, 3), "material_id": mid,
                          "material": materials.get(mid, {})})

    def visit(node, T):
        T = T @ rmjcf._node_transform(node)
        for inst in node.findall(f"{_C}instance_geometry"):
            binding = {im.get("symbol"): im.get("target", "").lstrip("#")
                       for im in inst.iter(f"{_C}instance_material")}
            geometry_parts(geometries[inst.get("url").lstrip("#")], T, binding)
        for inst in node.findall(f"{_C}instance_node"):
            visit(library_nodes[inst.get("url").lstrip("#")], T)
        for child in node.findall(f"{_C}node"):
            visit(child, T)

    for scene in root.iter(f"{_C}visual_scene"):
        for node in scene.findall(f"{_C}node"):
            visit(node, np.eye(4))
    return parts


def _mesh_file(uri: str):
    rel = uri.split("/share/", 1)[1] if "/share/" in uri else uri.removeprefix("package://")
    path = ASSETS / "packages" / rel
    if not path.is_file():
        raise FileNotFoundError(f"Reachy mesh {uri} not found under {ASSETS / 'packages'}")
    return path


@functools.cache
def reachy_template() -> tuple[list[dict], dict, dict]:
    """``(components, materials, files)``: link components with parts whose ``mesh`` /
    ``generated_mesh`` hold ``("file", key)`` placeholders, and ``files[key] = msh bytes``."""
    tree = ET.parse(urdf().path).getroot()
    colors = {m.get("name"): _floats(m.find("color").get("rgba")).tolist()
              for m in tree.findall("material") if m.find("color") is not None}
    files, materials, components = {}, {}, []
    SKIPPED.clear()

    def store(data: bytes) -> tuple:
        import hashlib
        key = hashlib.sha256(data).hexdigest()
        files[key] = data
        return ("file", key)

    for link in tree.findall("link"):
        name = link.get("name")
        visual, collision = [], []
        for vi, v in enumerate(link.findall("visual")):
            T = _origin(v)
            g = v.find("geometry")[0]
            mat_el = v.find("material")
            color = None
            if mat_el is not None:
                c = mat_el.find("color")
                color = _floats(c.get("rgba")).tolist() if c is not None else colors.get(mat_el.get("name"))
            if g.tag == "mesh":
                path = _mesh_file(g.get("filename"))
                S = np.diag([*(_floats(g.get("scale")) if g.get("scale") else np.ones(3)), 1.0])
                for k, p in enumerate(read_collada_parts(path)):
                    if len(np.unique(np.round(p["vertices"], 9), axis=0)) < 4:
                        # MuJoCo rejects meshes under 4 vertices (single-triangle material patches),
                        # as reachy-agent simulation/prepare.py does; recorded in provenance()
                        SKIPPED.append({"link": name, "file": path.name, "part": k, "faces": len(p["faces"])})
                        continue
                    M = T @ S
                    verts = (np.c_[p["vertices"], np.ones(len(p["vertices"]))] @ M.T)[:, :3]
                    nrm = p["normals"]
                    if nrm is not None:
                        nrm = nrm @ np.linalg.inv(M[:3, :3])
                        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
                    if color is not None:
                        mname = f"{PREFIX}urdf/{mat_el.get('name') or 'color'}"
                        materials[mname] = {"rgba": [float(x) for x in color], "texture": None,
                                            "source": {"urdf_material": mat_el.get("name"), "rgba": color}}
                    else:
                        diffuse = p["material"].get("diffuse", [0.7, 0.7, 0.7, 1.0])
                        mname = f"{PREFIX}{path.stem}/{p['material_id'] or k}"
                        materials[mname] = {"rgba": [float(x) for x in diffuse], "texture": None,
                                            "source": {"collada": path.name, "effect": p["material"]}}
                    visual.append({"geom": f"{name}/visual{vi}/{path.stem}/{k}", "type": "mesh", "size": [0, 0, 0],
                                   "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "group": VISUAL_GROUP,
                                   "contype": 0, "conaffinity": 0, "rgba": [0.5, 0.5, 0.5, 1.0], "material": mname,
                                   "mesh": store(encode_msh(verts, p["faces"], nrm)),
                                   "mesh_source": {"file": str(path.relative_to(ASSETS)), "uri": g.get("filename"),
                                                   "part": k, "collada_material": p["material_id"],
                                                   "transform": "visual origin x mesh scale x COLLADA node "
                                                                "transforms baked into the vertices"}})
            else:
                kind, size = _urdf_primitive(g)
                rgba = color or [0.5, 0.5, 0.5, 1.0]
                v_, n_, f_ = primitive_mesh(kind, size)
                visual.append({"geom": f"{name}/visual{vi}", "type": kind, "size": size,
                               "pos": T[:3, 3].tolist(), "quat": pose_to_vec7(T)[3:].tolist(), "group": VISUAL_GROUP,
                               "contype": 0, "conaffinity": 0, "rgba": [float(x) for x in rgba], "material": None,
                               "generated_mesh": store(encode_msh(v_, f_, n_))})
        for ci, c in enumerate(link.findall("collision")):
            T = _origin(c)
            g = c.find("geometry")[0]
            base = {"pos": T[:3, 3].tolist(), "quat": pose_to_vec7(T)[3:].tolist(), "group": 3,
                    "contype": 1, "conaffinity": 1, "rgba": [0.5, 0.5, 0.5, 1.0], "material": None}
            if g.tag == "mesh":
                path = rmjcf._mesh_path(g.get("filename"))
                for k, hull_pts in enumerate(rmjcf.collider_parts(str(path))):
                    hull = ConvexHull(hull_pts)
                    faces = _outward(hull_pts, hull.simplices)
                    collision.append({**base, "geom": f"{name}/collision{ci}/{k}", "type": "mesh", "size": [0, 0, 0],
                                      "mesh": store(encode_msh(hull_pts, faces)),
                                      "mesh_source": {"file": str(path.relative_to(ASSETS)), "component": k},
                                      "collision_shape": "convex hull of a collider-mesh component (tier P)"})
            else:
                kind, size = _urdf_primitive(g)
                collision.append({**base, "geom": f"{name}/collision{ci}", "type": kind, "size": size})
        if visual or collision:
            components.append({"name": f"{PREFIX}{name}", "role": "robot_link", "source_body": name,
                               "bodies": [name], "kind": "robot", "task": False, "object_id": None,
                               "articulation": None, "joints": [], "pose_source": "robot_fk",
                               "visual_from_collision": False, "visual": visual, "collision": collision})
    return components, materials, files


def _outward(points, simplices) -> np.ndarray:
    c = points.mean(0)
    f = np.array(simplices, int)
    tri = points[f]
    flip = np.einsum("ij,ij->i", np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), tri[:, 0] - c) < 0
    f[flip] = f[flip][:, [0, 2, 1]]
    return f


def _urdf_primitive(g) -> tuple[str, list[float]]:
    if g.tag == "box":
        return "box", (_floats(g.get("size")) / 2).tolist()
    if g.tag == "cylinder":
        return "cylinder", [float(g.get("radius")), float(g.get("length")) / 2, 0.0]
    if g.tag == "sphere":
        return "sphere", [float(g.get("radius")), 0.0, 0.0]
    raise ValueError(f"unsupported URDF geometry {g.tag}")


def reachy_components(library) -> tuple[list[dict], dict]:
    """Link components and materials with mesh references written to ``library``."""
    components, materials, files = reachy_template()
    refs = {}

    def ref(placeholder):
        key = placeholder[1]
        if key not in refs:
            from ..schema.scene_assets import decode_msh
            m = decode_msh(files[key])
            refs[key] = {**library.put(files[key], "msh"), "vertices": len(m["vertices"]), "faces": len(m["faces"]),
                         "normals": m["normals"] is not None, "uv": False}
        return refs[key]

    out = []
    for c in components:
        c = dict(c)
        for kind in ("visual", "collision"):
            c[kind] = [{**p, **{k: ref(p[k]) for k in ("mesh", "generated_mesh") if k in p}} for p in c[kind]]
        out.append(c)
    return out, dict(materials)


@functools.cache
def _tree(links: tuple[str, ...]) -> KinematicTree:
    return KinematicTree(urdf(), urdf().root, list(links), JOINTS[3:22], FIXED)


def link_poses(q, links) -> np.ndarray:
    """World poses (T, len(links), 7) of URDF links for canonical ``q`` (T, 22)."""
    q = np.asarray(q, float)
    B = planar(q[:, 0], q[:, 1], q[:, 2])
    root = urdf().root
    others = tuple(sorted(l for l in links if l != root))
    fk = _tree(others).fk(q[:, 3:22]) if others else {}
    out = np.empty((len(q), len(links), 7))
    for i, link in enumerate(links):
        out[:, i] = pose_to_vec7(B if link == root else B @ fk[link])
    return out


def provenance() -> dict:
    reachy_template()
    return {"urdf_sha256": URDF_SHA256, "packages": "robot/assets/packages (see sources.json)",
            "visual_group": VISUAL_GROUP, "skipped_visual_parts": list(SKIPPED)}
