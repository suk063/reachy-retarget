"""MuJoCo model of Reachy 2 built from the vendored URDF and collision meshes (no files written).

`reachy_mjcf(prefix)` returns `(xml, assets)`; `attach_reachy(spec, ...)` inserts the robot into
another `mujoco.MjSpec` at a world pose. The model is meant for physics validation (tier P):
it has collision geometry and position servos only, no visual meshes or cameras.

Structure
---------
* Root body `{prefix}base_link` on the floor (z = 0 of its attachment frame) with the planar
  base joints `base_x`, `base_y` (slides along the frame x/y axes) and `base_yaw` (hinge about
  z), so these joint values equal canonical `q[0:3]` when the frame is the world frame.
* One body per URDF link below it (`fusestatic` is off, so every URDF frame remains a body).
  Moving joints: the 17 arm/neck joints, `{l,r}_hand_finger` and every joint that mimics a
  finger joint. Mimic joints are coupled by `<equality joint>` constraints with
  `polycoef = (offset, multiplier)` of the URDF `<mimic>` tag (solref 0.004 s, i.e. two
  physics steps, as in legacy physics.py) and carry no joint range of their own: the URDF
  ranges of the proximal/distal mimics (upper 0.554 rad) are narrower than the image of the
  finger range (0.592 rad at the closed limit) and would fight the coupling. All other
  joints (tripod, antennas, the tripod bar mimics) are folded into fixed transforms at the
  tripod's 0 value, as in robot.reachy.
* Inertials are the URDF ones (`fullinertia` rotated into the body frame). All robot bodies
  have `gravcomp=1`: the real joint controllers hold the arms against their own weight, and
  soft simulated servos alone would sag by centimetres (reachy-agent simulation/prepare.py).
  Payloads are not compensated.

Collision geometry
------------------
The URDF `<collision>` elements: primitives (base cylinder, wheel cylinders, lidar cylinder,
tripod bar boxes) as MuJoCo primitives, and the manufacturer collider meshes (`*_collider.dae`:
torso, head, upper arm, forearm, palm, proximal and distal finger links) parsed from COLLADA
here (no trimesh). Each connected component of a collider mesh becomes one mesh geom; MuJoCo
collides with its convex hull, which is exact for the convex colliders and conservative for a
concave component. Components with a hull volume below 1 mm^3 (stray planar patches) cannot
form a hull and are dropped and listed in `MESH_REPORT`. The finger pads are the hulls of
`pincette_distal_collider.dae`, whose flat inner face is the plane Y = -0.0125 m used by
robot.gripper. Geoms use friction (1, 0.01, 0.001) and condim 4 (torsional friction), the
reachy-agent settings that hold a can without creep (with elliptic cones and impratio 10, see
`OPTIONS`), and the stiff contact of legacy physics.py: solref (0.004, 1) (two physics steps)
and solimp (0.95, 0.99, 0.001). `priority=1` makes these Reachy parameters govern every
Reachy-object contact instead of being averaged with the object's (MuJoCo mixes solref/solimp
of equal-priority geoms): with the default 0.02 s object solref, a 2 Nm finger squeeze sank the
pads about 2.3 mm into a held box, against 0.2 mm with priority. Objects keep their own
parameters for every other contact. Geoms have no mass (inertia comes from the URDF).

Self-collision filtering: URDF parent/child pairs plus, across the massless dummy links of the
multi-axis joints, each collision link and its nearest colliding ancestor (legacy physics.py,
reachy-agent prepare.py). Other pairs, including the two fingers of one hand, collide.
Floor contact of the wheels is not filtered here (the floor belongs to the scene): call
`floor_excludes(prefix)` for the body names to exclude against `world`.

Actuators
---------
One `<position>` actuator per canonical joint, named `{prefix}{joint}` and declared in
`JOINTS` order (base, left arm, right arm, neck, fingers), so `ctrl` is canonical `q`:

| joints | kp | damping | forcerange | source |
| --- | --- | --- | --- | --- |
| base_x, base_y | 5e4 N/m | dampratio 1 | +/-2000 N | reachy-agent Config.base_kp/base_force |
| base_yaw | 3e3 Nm/rad | dampratio 1 | +/-500 Nm | reachy-agent Config.base_yaw_kp/base_torque |
| arms, neck | 180 Nm/rad | kv 24 | +/-40 Nm | reachy-agent Config.arm_* (legacy physics.py) |
| fingers | 8 Nm/rad | kv 0.8 | +/-2 Nm | reachy-agent Config.gripper_* |

The base servos are an ideal planar base (wheels not simulated). Arm/neck/finger joints have
damping 0.01 and armature 0.005; ctrlrange equals the URDF joint range (base: unlimited).
"""
from __future__ import annotations

import functools
import re
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .reachy import FIXED, GRASP, JOINTS, LINKS, Reachy
from .resources import ASSETS, urdf

TIMESTEP = 0.002
# Global options for scenes that contain Reachy (reachy-agent simulation/prepare.py): elliptic
# friction cones with impratio 10 stop the creep that lets a held object slide out of the pads.
OPTIONS = {"timestep": TIMESTEP, "integrator": "implicitfast", "cone": "elliptic", "impratio": 10.0,
           "iterations": 100, "gravity": (0.0, 0.0, -9.81)}
GEOM = {"friction": "1 0.01 0.001", "condim": "4", "solref": "0.004 1", "solimp": "0.95 0.99 0.001",
        "priority": "1", "contype": "1", "conaffinity": "1", "group": "3", "mass": "0"}
MIMIC_SOLREF = "0.004 1"
JOINT_DAMPING, JOINT_ARMATURE = 0.01, 0.005
GAINS = {  # joint group -> actuator gains (see module docstring for sources)
    "base_xy": {"kp": 5e4, "dampratio": 1.0, "force": 2000.0},
    "base_yaw": {"kp": 3e3, "dampratio": 1.0, "force": 500.0},
    "arm": {"kp": 180.0, "kv": 24.0, "force": 40.0},
    "gripper": {"kp": 8.0, "kv": 0.8, "force": 2.0},
}
FINGER_BODIES = {"left": ("l_hand_distal_link", "l_hand_distal_mimic_link"),
                 "right": ("r_hand_distal_link", "r_hand_distal_mimic_link")}
WHEELS = ("drivewhl1_link", "drivewhl2_link", "drivewhl3_link")
MIN_HULL_VOLUME = 1e-9  # m^3
MESH_REPORT: dict[str, list] = {}  # mesh file -> dropped degenerate components (filled on build)

_COLLADA = "{http://www.collada.org/2005/11/COLLADASchema}"


def gain_group(joint: str) -> str:
    if joint in ("base_x", "base_y"):
        return "base_xy"
    if joint == "base_yaw":
        return "base_yaw"
    return "gripper" if joint.endswith("hand_finger") else "arm"


def _num(a) -> str:
    return " ".join(f"{float(v):.12g}" for v in np.ravel(a))


def _pose_attrs(T) -> dict:
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
    return {"pos": _num(T[:3, 3]), "quat": _num([q[3], q[0], q[1], q[2]])}


def _origin(e) -> np.ndarray:
    o = e.find("origin")
    T = np.eye(4)
    if o is not None:
        T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
        T[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in o.get("rpy", "0 0 0").split()]).as_matrix()
    return T


# ------------------------------------------------------------------------------- COLLADA

def _node_transform(node) -> np.ndarray:
    T = np.eye(4)
    for child in node:
        tag = child.tag.removeprefix(_COLLADA)
        v = np.array([float(x) for x in child.text.split()]) if child.text else None
        if tag == "matrix":
            T = T @ v.reshape(4, 4)
        elif tag == "translate":
            M = np.eye(4)
            M[:3, 3] = v
            T = T @ M
        elif tag == "rotate":
            M = np.eye(4)
            M[:3, :3] = Rotation.from_rotvec(np.radians(v[3]) * v[:3] / np.linalg.norm(v[:3])).as_matrix()
            T = T @ M
        elif tag == "scale":
            T = T @ np.diag([*v, 1.0])
    return T


def _geometry_triangles(geometry):
    """(vertices (n, 3), triangles (m, 3)) of one COLLADA <geometry>."""
    mesh = geometry.find(f"{_COLLADA}mesh")
    arrays = {s.get("id"): np.array([float(x) for x in s.find(f"{_COLLADA}float_array").text.split()])
              for s in mesh.findall(f"{_COLLADA}source")}
    vert = mesh.find(f"{_COLLADA}vertices")
    pos_src = next(i.get("source") for i in vert.findall(f"{_COLLADA}input") if i.get("semantic") == "POSITION")
    vertices = arrays[pos_src.lstrip("#")].reshape(-1, 3)
    tris = []
    for prim in list(mesh.findall(f"{_COLLADA}triangles")) + list(mesh.findall(f"{_COLLADA}polylist")):
        inputs = prim.findall(f"{_COLLADA}input")
        stride = max(int(i.get("offset", 0)) for i in inputs) + 1
        off = next(int(i.get("offset", 0)) for i in inputs if i.get("semantic") == "VERTEX")
        p = np.array([int(x) for x in prim.find(f"{_COLLADA}p").text.split()])
        idx = p.reshape(-1, stride)[:, off]
        if prim.tag.endswith("polylist"):
            counts = [int(x) for x in prim.find(f"{_COLLADA}vcount").text.split()]
            start = 0
            for c in counts:  # fan triangulation
                poly = idx[start:start + c]
                tris.extend([poly[0], poly[k], poly[k + 1]] for k in range(1, c - 1))
                start += c
        else:
            tris.extend(idx.reshape(-1, 3).tolist())
    if mesh.find(f"{_COLLADA}polygons") is not None:
        raise ValueError("COLLADA <polygons> primitives are not supported")
    return vertices, np.array(tris, int).reshape(-1, 3)


def read_collada(path) -> list[np.ndarray]:
    """Vertices of every instanced geometry in the visual scene (node transforms and unit applied)."""
    root = ET.parse(path).getroot()
    unit = root.find(f"{_COLLADA}asset/{_COLLADA}unit")
    scale = float(unit.get("meter", 1.0)) if unit is not None else 1.0
    up = root.find(f"{_COLLADA}asset/{_COLLADA}up_axis")
    if up is not None and up.text.strip() != "Z_UP":
        raise ValueError(f"{path}: only Z_UP COLLADA files are supported")
    geometries = {g.get("id"): g for g in root.iter(f"{_COLLADA}geometry")}
    out = []

    def visit(node, T):
        T = T @ _node_transform(node)
        for inst in node.findall(f"{_COLLADA}instance_geometry"):
            v, f = _geometry_triangles(geometries[inst.get("url").lstrip("#")])
            v = (np.c_[v, np.ones(len(v))] @ T.T)[:, :3] * scale
            out.append((v, f))
        for child in node.findall(f"{_COLLADA}node"):
            visit(child, T)

    for scene in root.iter(f"{_COLLADA}visual_scene"):
        for node in scene.findall(f"{_COLLADA}node"):
            visit(node, np.eye(4))
    return out


def _components(vertices, faces) -> list[np.ndarray]:
    """Vertex sets of the connected components (coincident vertices welded at 1e-7 m)."""
    key = np.round(vertices / 1e-7).astype(np.int64)
    _, weld = np.unique(key, axis=0, return_inverse=True)
    weld = weld.ravel()
    parent = np.arange(weld.max() + 1)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, c in weld[faces]:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[find(rc)] = ra
    roots = np.array([find(i) for i in weld[np.unique(faces)]])
    used = np.unique(faces)
    return [vertices[used[roots == r]] for r in np.unique(roots)]


@functools.cache
def collider_parts(path: str) -> tuple[np.ndarray, ...]:
    """Convex-hull vertex sets of the components of one collider mesh (degenerate ones dropped)."""
    parts, dropped = [], []
    for v, f in read_collada(path):
        for comp in _components(v, f):
            try:
                hull = ConvexHull(comp)
                volume = hull.volume
            except Exception:  # coplanar / too few points
                hull, volume = None, 0.0
            if volume < MIN_HULL_VOLUME:
                dropped.append({"vertices": len(comp), "hull_volume_m3": float(volume)})
                continue
            parts.append(comp[np.unique(hull.vertices)])
    MESH_REPORT[str(path)] = dropped
    return tuple(parts)


def _mesh_path(uri: str):
    rel = uri.removeprefix("package://")
    path = ASSETS / "packages" / rel
    if not path.is_file():
        raise FileNotFoundError(f"collision mesh {uri} not found under {ASSETS / 'packages'}")
    return path


# ------------------------------------------------------------------------------- MJCF

def _independent():
    u = urdf()
    names = set(JOINTS[3:])
    dynamic = set(names)
    for j in u.by_name:
        if u.source(j)[0] in names:
            dynamic.add(j)
    return names, dynamic


def reachy_mjcf(prefix: str = "", *, standalone: bool = True) -> tuple[str, dict]:
    """Reachy 2 MJCF and its asset dictionary.

    Meshes are inlined as vertex lists, so `assets` is empty; it is returned so callers can pass
    it unchanged to `MjSpec.from_string`/`MjModel.from_xml_string`. With `standalone=True` the
    model also has `OPTIONS`, a floor plane and the wheel/floor exclusions; otherwise it is the
    bare robot for `attach_reachy`.
    """
    u = urdf()
    tree = ET.parse(u.path).getroot()
    links = {e.get("name"): e for e in tree.findall("link")}
    children: dict[str, list] = {}
    for j in tree.findall("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(j)
    independent, dynamic = _independent()
    p = prefix

    mj = ET.Element("mujoco", model=f"{p}reachy2")
    ET.SubElement(mj, "compiler", angle="radian", autolimits="true", fusestatic="false",
                  inertiafromgeom="false", balanceinertia="false")
    if standalone:
        o = OPTIONS
        ET.SubElement(mj, "option", timestep=str(o["timestep"]), integrator=o["integrator"], cone=o["cone"],
                      impratio=str(o["impratio"]), iterations=str(o["iterations"]), gravity=_num(o["gravity"]))
    asset = ET.SubElement(mj, "asset")
    world = ET.SubElement(mj, "worldbody")
    if standalone:
        ET.SubElement(world, "geom", name=f"{p}floor", type="plane", size="5 5 0.1",
                      **{k: v for k, v in GEOM.items() if k != "mass"})
    contact = ET.SubElement(mj, "contact")
    equality = ET.SubElement(mj, "equality")
    actuator = ET.SubElement(mj, "actuator")
    meshes: dict[str, list[str]] = {}

    def mesh_names(uri):
        if uri not in meshes:
            stem = re.sub(r"\W", "_", uri.rsplit("/", 1)[-1].rsplit(".", 1)[0])
            meshes[uri] = []
            for k, verts in enumerate(collider_parts(str(_mesh_path(uri)))):
                name = f"{p}{stem}_{k}"
                ET.SubElement(asset, "mesh", name=name, vertex=_num(verts))
                meshes[uri].append(name)
        return meshes[uri]

    def geoms(body, link):
        for i, c in enumerate(links[link].findall("collision")):
            T = _origin(c)
            g = c.find("geometry")[0]
            base = {**GEOM, **_pose_attrs(T)}
            if g.tag == "mesh":
                scale = g.get("scale", "1 1 1")
                if scale.split() != ["1", "1", "1"]:
                    raise ValueError(f"scaled collision mesh on {link} is not supported")
                for k, m in enumerate(mesh_names(g.get("filename"))):
                    ET.SubElement(body, "geom", name=f"{p}{link}_c{i}_{k}", type="mesh", mesh=m, **base)
            elif g.tag == "box":
                size = np.array([float(v) for v in g.get("size").split()]) / 2
                ET.SubElement(body, "geom", name=f"{p}{link}_c{i}", type="box", size=_num(size), **base)
            elif g.tag == "cylinder":
                ET.SubElement(body, "geom", name=f"{p}{link}_c{i}", type="cylinder",
                              size=f"{g.get('radius')} {float(g.get('length')) / 2}", **base)
            elif g.tag == "sphere":
                ET.SubElement(body, "geom", name=f"{p}{link}_c{i}", type="sphere", size=g.get("radius"), **base)
            else:
                raise ValueError(f"unsupported collision geometry {g.tag} on {link}")

    def inertial(body, link):
        e = links[link].find("inertial")
        if e is None:
            return
        mass = float(e.find("mass").get("value"))
        if mass <= 0:
            return
        T = _origin(e)
        a = e.find("inertia").attrib
        I = np.array([[float(a["ixx"]), float(a["ixy"]), float(a["ixz"])],
                      [float(a["ixy"]), float(a["iyy"]), float(a["iyz"])],
                      [float(a["ixz"]), float(a["iyz"]), float(a["izz"])]])
        I = T[:3, :3] @ I @ T[:3, :3].T
        ET.SubElement(body, "inertial", pos=_num(T[:3, 3]), mass=repr(mass),
                      fullinertia=_num([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]))

    def visit(parent, link, joint=None):
        T = np.eye(4) if joint is None else _origin(joint)
        jn = None if joint is None else joint.get("name")
        if jn is not None and jn not in dynamic and joint.get("type") != "fixed":
            src, mult, off = u.source(jn)
            T = T @ u.motion(u.by_name[jn], mult * FIXED.get(src, 0.0) + off)
        body = ET.SubElement(parent, "body", name=f"{p}{link}", gravcomp="1", **_pose_attrs(T))
        if joint is None:
            for name, kind, axis in (("base_x", "slide", "1 0 0"), ("base_y", "slide", "0 1 0"),
                                     ("base_yaw", "hinge", "0 0 1")):
                ET.SubElement(body, "joint", name=f"{p}{name}", type=kind, axis=axis, limited="false",
                              damping="0", armature="0")
        elif jn in dynamic:
            spec = {"name": f"{p}{jn}", "type": "slide" if joint.get("type") == "prismatic" else "hinge",
                    "axis": joint.find("axis").get("xyz"), "damping": str(JOINT_DAMPING),
                    "armature": str(JOINT_ARMATURE)}
            mimic = joint.find("mimic")
            if mimic is None:
                lo, hi = u.limits[jn]
                spec["range"] = f"{lo!r} {hi!r}"
            else:
                spec["limited"] = "false"
                ET.SubElement(equality, "joint", name=f"{p}{jn}_mimic", joint1=f"{p}{jn}",
                              joint2=f"{p}{mimic.get('joint')}", solref=MIMIC_SOLREF,
                              polycoef=f"{float(mimic.get('offset', 0))!r} {float(mimic.get('multiplier', 1))!r} 0 0 0")
            ET.SubElement(body, "joint", **spec)
        inertial(body, link)
        geoms(body, link)
        for frame, target in LINKS.items():
            if target == link:
                ET.SubElement(body, "site", name=f"{p}{frame}", size="0.004", group="4")
        for g, tcp in GRASP.items():
            if LINKS[tcp] == link:
                side = g.split("_")[0]
                ET.SubElement(body, "site", name=f"{p}{g}", size="0.004", group="4",
                              **_pose_attrs(Reachy.load().grasp_center(side)))
        for child in children.get(link, []):
            visit(body, child.find("child").get("link"), child)

    visit(world, u.root)

    # Self-collision filtering (see module docstring).
    excluded = {tuple(sorted((j.find("parent").get("link"), j.find("child").get("link"))))
                for j in tree.findall("joint")}
    parents = {j.find("child").get("link"): j.find("parent").get("link") for j in tree.findall("joint")}
    colliding = {n for n, e in links.items() if e.findall("collision")}
    for n in colliding:
        a = parents.get(n)
        while a is not None and a not in colliding:
            a = parents.get(a)
        if a is not None:
            excluded.add(tuple(sorted((n, a))))
    for a, b in sorted(excluded):
        ET.SubElement(contact, "exclude", name=f"{p}exclude_{a}__{b}", body1=f"{p}{a}", body2=f"{p}{b}")
    if standalone:
        for wheel in floor_excludes(prefix):
            ET.SubElement(contact, "exclude", body1="world", body2=wheel)

    for name in JOINTS:
        g = GAINS[gain_group(name)]
        spec = {"name": f"{p}{name}", "joint": f"{p}{name}", "kp": repr(g["kp"]),
                "forcerange": f"{-g['force']!r} {g['force']!r}"}
        if "dampratio" in g:
            spec["dampratio"] = repr(g["dampratio"])
        else:
            spec["kv"] = repr(g["kv"])
        if name in u.limits:
            lo, hi = u.limits[name]
            spec["ctrlrange"] = f"{lo!r} {hi!r}"
        else:
            spec["ctrllimited"] = "false"
        ET.SubElement(actuator, "position", **spec)
    return ET.tostring(mj, encoding="unicode"), {}


def floor_excludes(prefix: str = "") -> list[str]:
    """Reachy bodies to exclude against the floor body (`world`): the wheels rest on z = 0."""
    return [f"{prefix}{w}" for w in WHEELS]


def mimic_joints() -> dict[str, tuple[str, float, float]]:
    """{mimic joint: (independent canonical joint, multiplier, offset)} of the finger linkages."""
    independent, dynamic = _independent()
    u = urdf()
    return {j: u.source(j) for j in sorted(dynamic - independent)}


def attach_reachy(spec, *, prefix: str = "reachy/", pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0),
                  exclude_floor_with_world: bool = True):
    """Attach Reachy into `spec` (a `mujoco.MjSpec`) at a world pose; returns the attachment frame.

    `pos`/`quat` (wxyz) place the base joints' frame: with the identity, `base_x/base_y/base_yaw`
    are world x, y, yaw. All Reachy names get `prefix`. The wheels are excluded against `world`
    (where scenes keep their floor) unless `exclude_floor_with_world` is False.
    """
    import warnings

    import mujoco

    xml, assets = reachy_mjcf("", standalone=False)
    child = mujoco.MjSpec.from_string(xml, assets=assets)
    frame = spec.worldbody.add_frame(pos=list(pos), quat=list(quat))
    with warnings.catch_warnings():
        # Reachy's meshes are inline vertex lists: the child needs no asset dictionary.
        warnings.filterwarnings("ignore", message="Attaching a child without asset dict")
        spec.attach(child, prefix=prefix, frame=frame)
    if exclude_floor_with_world:
        for wheel in floor_excludes(prefix):
            spec.add_exclude(bodyname1="world", bodyname2=wheel)
    return frame


def set_reachy_state(model, data, q, prefix: str = "") -> None:
    """Write canonical q (22,) into qpos (mimic joints included) and ctrl; does not call mj_forward."""
    q = np.asarray(q, float)
    for i, name in enumerate(JOINTS):
        data.qpos[model.joint(f"{prefix}{name}").qposadr[0]] = q[i]
        data.ctrl[model.actuator(f"{prefix}{name}").id] = q[i]
    for j, (src, mult, off) in mimic_joints().items():
        data.qpos[model.joint(f"{prefix}{j}").qposadr[0]] = mult * q[JOINTS.index(src)] + off


def reachy_q(model, data, prefix: str = "") -> np.ndarray:
    """Canonical q (22,) read from qpos."""
    return np.array([data.qpos[model.joint(f"{prefix}{n}").qposadr[0]] for n in JOINTS])
