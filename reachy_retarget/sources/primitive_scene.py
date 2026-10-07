"""MuJoCo scenes built from primitive object geometry, for sources simulated elsewhere.

Some sources are not simulated in MuJoCo but have scenes made only of primitives (ManiSkill3
tabletop tasks: a table box, cubes, pegs, a ball, box unions). ``build`` turns their
``ObjectTrack`` geometry into a complete MJCF so they can get physics tier P
(``validate.scene.build_scene`` / ``validate.physics.simulate``):

* a floor plane (``physical["floor"]["z"]``) in the world body;
* ``static`` objects (tables, kinematic fixtures that never move in the source) as bodies
  without joints, posed at their first valid frame;
* ``dynamic`` objects as bodies with a free joint ``<id>_freejoint``, posed at their first
  valid frame (MJCF body pose and ``SceneRef.initial_qpos``), mass from per-geom density.

Body names are the object ids, so ``validate.physics`` maps ``ep.objects[id]`` to the body
``id`` (also the ``geometry["body"]`` the caller should set). There is no source robot in the
scene. ``validate.scene.build_scene`` refuses an empty ``robot_prefixes`` list, so the SceneRef
carries ``NO_ROBOT_PREFIX``, a prefix that matches no element (checked here).

Geometry (``ObjectTrack.geometry``, body frame):

* ``{"kind": "box", "half_extents": [x, y, z], "center"?: [x, y, z]}``
* ``{"kind": "boxes", "boxes": [{"center", "half_extents"}, ...]}`` (union of boxes)
* ``{"kind": "sphere", "radius": r}``
* ``{"kind": "cylinder" | "capsule", "radius": r, "half_length": h, "axis"?: "x" | "y" | "z"}``;
  the default axis is ``x`` (SAPIEN/PhysX convention), MuJoCo's own axis is z and the geom is
  rotated accordingly.

Anything else (meshes, empty geometry) raises ``UnrepresentableObject``.

``physical`` (all values recorded verbatim in the scene provenance)::

    {"source": {...},                       # where the values come from
     "defaults": {"static_friction", "dynamic_friction", "restitution", "density"},
     "floor": {"z": float, **material, "source": ...},
     "objects": {id: {"body_type": "dynamic" | "static", "density": float | [per box],
                      **material overrides, "source": ...}},
     "contact": {"condim", "solref", "solimp", "margin"}  # optional overrides of CONTACT}

Friction model mapping (PhysX -> MuJoCo), recorded as assumptions:

* MuJoCo has one Coulomb coefficient per geom; it is set to the PhysX *dynamic* friction
  (with elliptic cones it bounds the tangential force at any slip speed). PhysX static
  friction is recorded but not modelled; ``build`` warns in the provenance when it differs.
* Equal-priority MuJoCo geoms combine friction with ``max``; PhysX's default combine mode is
  ``average``. They agree when both materials are equal (all ManiSkill tabletop materials are
  the scene default). Reachy geoms have ``priority=1`` and their own friction governs every
  Reachy-object contact.
* ``condim 3`` (sliding friction only): PhysX has no torsional or rolling friction parameter;
  multi-point box contacts resist spinning in both engines.
* Restitution: MuJoCo contacts are critically damped (``solref`` damping ratio 1), i.e.
  restitution 0; a source restitution > 0 is recorded as not modelled.
* PhysX rigid contacts (``rest_offset`` 0) are hard; MuJoCo's soft contacts are made stiff with
  ``solref (0.004, 1)`` (two 2 ms steps) and ``solimp (0.95, 0.99, 0.001)``, the Reachy
  contact parameters. ``margin`` 0: PhysX ``contact_offset`` is a detection distance and
  produces no force before touching.
* PhysX actor damping (SDK defaults: linear 0, angular 0.05 1/s) is not modelled (MuJoCo free
  joints have no damping here).
"""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np

from ..schema.source import ObjectTrack, SceneRef

NO_ROBOT_PREFIX = "no_source_robot/"
CONTACT = {"condim": 3, "solref": [0.004, 1.0], "solimp": [0.95, 0.99, 0.001], "margin": 0.0}
PROVENANCE_TEXT = "primitive_scene_provenance"
ASSUMPTIONS = [
    "MuJoCo sliding friction = PhysX dynamic friction; PhysX static friction not modelled",
    "MuJoCo combines equal-priority friction with max, PhysX averages: identical when materials are equal",
    "condim 3: no torsional or rolling friction (PhysX has neither parameter)",
    "restitution 0 (critically damped solref); a nonzero source restitution is not modelled",
    "stiff contacts solref (0.004, 1), solimp (0.95, 0.99, 0.001), margin 0 approximate PhysX hard contacts "
    "with rest_offset 0",
    "PhysX actor damping (linear 0, angular 0.05 1/s SDK defaults) not modelled",
    "mass from per-geom density x volume; overlapping parts of a box union count twice, as in PhysX",
    "kinematic source actors that never move become static bodies",
    "Reachy geoms (priority 1) set friction and contact softness of every Reachy-object contact",
]


class UnrepresentableObject(ValueError):
    pass


def _num(v) -> str:
    return " ".join(f"{float(x):.9g}" for x in np.atleast_1d(v))


def _axis_quat(axis: str) -> list[float]:
    """Quaternion (wxyz) rotating MuJoCo's z axis onto the body ``axis``."""
    s = float(np.sqrt(0.5))
    return {"z": [1.0, 0.0, 0.0, 0.0], "x": [s, 0.0, s, 0.0], "y": [s, -s, 0.0, 0.0]}[axis]


def geoms_of(oid: str, geometry: dict) -> list[dict]:
    """MuJoCo geom attributes (type, size, pos, quat) for one object's geometry."""
    kind = geometry.get("kind")
    if kind == "box":
        return [{"type": "box", "size": list(geometry["half_extents"]),
                 "pos": list(geometry.get("center", [0.0, 0.0, 0.0]))}]
    if kind == "boxes":
        return [{"type": "box", "size": list(b["half_extents"]), "pos": list(b["center"])}
                for b in geometry["boxes"]]
    if kind == "sphere":
        return [{"type": "sphere", "size": [geometry["radius"]], "pos": list(geometry.get("center", [0, 0, 0]))}]
    if kind in ("cylinder", "capsule"):
        axis = geometry.get("axis", "x")
        return [{"type": kind, "size": [geometry["radius"], geometry["half_length"]],
                 "pos": list(geometry.get("center", [0, 0, 0])), "quat": _axis_quat(axis)}]
    raise UnrepresentableObject(f"object {oid!r}: geometry kind {kind!r} is not a supported primitive"
                                if kind else f"object {oid!r}: no geometry")


def _volume(g: dict) -> float:
    s = g["size"]
    if g["type"] == "box":
        return 8.0 * s[0] * s[1] * s[2]
    if g["type"] == "sphere":
        return 4.0 / 3.0 * np.pi * s[0] ** 3
    if g["type"] == "cylinder":
        return np.pi * s[0] ** 2 * 2 * s[1]
    return np.pi * s[0] ** 2 * 2 * s[1] + 4.0 / 3.0 * np.pi * s[0] ** 3   # capsule


def _first_pose(oid: str, track: ObjectTrack) -> np.ndarray:
    valid = np.flatnonzero(np.asarray(track.valid, bool))
    if not len(valid):
        raise UnrepresentableObject(f"object {oid!r}: no valid pose")
    p = np.asarray(track.pose[valid[0]], float).copy()
    p[3:] /= np.linalg.norm(p[3:])
    return p


def build(objects: dict[str, ObjectTrack], *, floor: bool = True, physical: dict,
          name: str = "primitive_scene") -> SceneRef:
    """Complete MJCF scene of primitive objects (see the module docstring)."""
    defaults = dict(physical.get("defaults", {}))
    for k in ("static_friction", "dynamic_friction", "restitution", "density"):
        if k not in defaults:
            raise ValueError(f"physical['defaults'] must give {k!r}")
    contact = {**CONTACT, **physical.get("contact", {})}
    per_object = physical.get("objects", {})
    unknown = sorted(set(per_object) - set(objects))
    if unknown:
        raise ValueError(f"physical['objects'] names objects not in the scene: {unknown}")

    root = ET.Element("mujoco", model=name)
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    opt = physical.get("option", {})
    ET.SubElement(root, "option", timestep=_num(opt.get("timestep", 0.002)),
                  gravity=_num(opt.get("gravity", [0.0, 0.0, -9.81])))
    world = ET.SubElement(root, "worldbody")
    record = {"objects": {}, "floor": None, "contact": contact, "assumptions": list(ASSUMPTIONS),
              "source": physical.get("source", {}), "defaults": defaults, "warnings": [],
              "robot_prefixes_note": f"no source robot; {NO_ROBOT_PREFIX!r} matches no element "
                                     "(validate.scene.build_scene refuses an empty prefix list)"}

    def material(over: dict) -> dict:
        m = {k: float(over.get(k, defaults[k])) for k in ("static_friction", "dynamic_friction", "restitution")}
        return m

    def contact_attrs(m: dict, who: str) -> dict:
        if abs(m["static_friction"] - m["dynamic_friction"]) > 1e-12:
            record["warnings"].append(f"{who}: static friction {m['static_friction']} != dynamic "
                                      f"{m['dynamic_friction']}; MuJoCo uses the dynamic value")
        if m["restitution"] > 0:
            record["warnings"].append(f"{who}: restitution {m['restitution']} not modelled (MuJoCo: 0)")
        return {"friction": _num([m["dynamic_friction"], 0.0, 0.0]), "condim": str(int(contact["condim"])),
                "solref": _num(contact["solref"]), "solimp": _num(contact["solimp"]),
                "margin": _num(contact["margin"])}

    if floor:
        fl = physical.get("floor", {})
        z = float(fl.get("z", 0.0))
        m = material(fl)
        ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 1", pos=_num([0, 0, z]),
                      **contact_attrs(m, "floor"))
        record["floor"] = {"z": z, **m, "source": fl.get("source")}

    initial_qpos = {}
    for oid in sorted(objects):
        track = objects[oid]
        if oid.startswith(NO_ROBOT_PREFIX) or oid == "floor":
            raise ValueError(f"object id {oid!r} is reserved")
        geoms = geoms_of(oid, track.geometry)
        pose = _first_pose(oid, track)
        spec = per_object.get(oid, {})
        body_type = spec.get("body_type")
        if body_type not in ("dynamic", "static"):
            raise ValueError(f"object {oid!r}: physical body_type must be 'dynamic' or 'static', got {body_type!r}")
        m = material(spec)
        density = spec.get("density", defaults["density"])
        dens = [float(d) for d in density] if isinstance(density, (list, tuple)) else [float(density)] * len(geoms)
        if len(dens) != len(geoms):
            raise ValueError(f"object {oid!r}: {len(dens)} densities for {len(geoms)} geoms")
        body = ET.SubElement(world, "body", name=oid, pos=_num(pose[:3]), quat=_num(pose[3:]))
        if body_type == "dynamic":
            ET.SubElement(body, "freejoint", name=f"{oid}_freejoint")
            initial_qpos[f"{oid}_freejoint"] = pose.tolist()
        attrs = contact_attrs(m, oid)
        for i, (g, d) in enumerate(zip(geoms, dens)):
            el = ET.SubElement(body, "geom", name=f"{oid}/geom{i}", type=g["type"], size=_num(g["size"]),
                               pos=_num(g["pos"]), **attrs)
            if "quat" in g:
                el.set("quat", _num(g["quat"]))
            if body_type == "dynamic":
                el.set("density", _num(d))
        mass = float(sum(_volume(g) * d for g, d in zip(geoms, dens))) if body_type == "dynamic" else None
        record["objects"][oid] = {"body": oid, "body_type": body_type, "role": track.role,
                                  "joint": f"{oid}_freejoint" if body_type == "dynamic" else None,
                                  "geoms": [{k: (list(map(float, v)) if isinstance(v, list) else v)
                                             for k, v in g.items()} for g in geoms],
                                  "density_kg_m3": dens if body_type == "dynamic" else None,
                                  "mass_kg": mass, **m, "initial_pose": pose.round(9).tolist(),
                                  "source": spec.get("source")}

    for el in root.iter():
        if el.get("name", "").startswith(NO_ROBOT_PREFIX):
            raise AssertionError("an element carries the no-robot prefix")
    custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "text", name=PROVENANCE_TEXT, data=json.dumps(record, sort_keys=True))
    xml = ET.tostring(root, encoding="unicode")
    return SceneRef(mjcf=xml, robot_prefixes=[NO_ROBOT_PREFIX], initial_qpos=initial_qpos, assets={})


def scene_provenance(scene_ref: SceneRef) -> dict:
    """The parameter record embedded by :func:`build` (sources, materials, assumptions)."""
    root = ET.fromstring(scene_ref.mjcf)
    el = root.find(f"custom/text[@name='{PROVENANCE_TEXT}']")
    if el is None:
        raise ValueError("not a primitive_scene SceneRef")
    out = json.loads(el.get("data"))
    out["mjcf_sha256"] = hashlib.sha256(scene_ref.mjcf.encode()).hexdigest()
    return out


CROP_RULE = ("static support box cropped in its own x/y only (height and top surface unchanged) to the "
             "axis-aligned region, in the support's frame, covering the footprint of every other object "
             "over every valid frame plus a margin on each side, intersected with the original extent")


def _quat_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _local_points(geometry: dict) -> np.ndarray:
    """Points spanning an object's extent in its body frame (box corners; origin if unknown)."""
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T
    kind = geometry.get("kind")
    if kind == "box":
        return np.asarray(geometry.get("center", [0, 0, 0]), float) + signs * geometry["half_extents"]
    if kind == "boxes":
        a = geometry["aabb"]
        return np.asarray(a["center"], float) + signs * a["half_extents"]
    if kind == "sphere":
        return np.asarray(geometry.get("center", [0, 0, 0]), float) + signs * geometry["radius"]
    if kind in ("cylinder", "capsule"):
        r, h = geometry["radius"], geometry["half_length"] + (geometry["radius"] if kind == "capsule" else 0)
        half = {"x": [h, r, r], "y": [r, h, r], "z": [r, r, h]}[geometry.get("axis", "x")]
        return np.asarray(geometry.get("center", [0, 0, 0]), float) + signs * half
    return np.zeros((1, 3))


def crop_supports(objects: dict[str, ObjectTrack], margin: float = 0.10, ids=None) -> tuple[dict, list[dict]]:
    """Crop static support boxes in x/y to the region the episode uses (``CROP_RULE``).

    ``ids``: support ids to crop (default: every ``support`` with ``box`` geometry whose pose
    never moves). Returns ``(objects, adaptations)``: a new dict in which each cropped support
    is a new ``ObjectTrack`` with updated ``half_extents``/``center`` (z untouched; the original
    geometry kept under ``geometry["uncropped"]``), and one record per support with the
    original and cropped extents, the margin and the rule. Other tracks are passed through.
    """
    if ids is None:
        ids = [k for k, o in objects.items() if o.role == "support" and o.geometry.get("kind") == "box"
               and np.asarray(o.valid).any()
               and np.ptp(np.asarray(o.pose)[np.asarray(o.valid, bool)], axis=0).max() < 1e-9]
    out, records = dict(objects), []
    for sid in ids:
        sup = objects[sid]
        g = sup.geometry
        if g.get("kind") != "box":
            raise ValueError(f"support {sid!r}: only box supports can be cropped")
        valid = np.flatnonzero(np.asarray(sup.valid, bool))
        if np.ptp(sup.pose[valid], axis=0).max() > 1e-9:
            raise ValueError(f"support {sid!r} moves; only static supports can be cropped")
        p0 = sup.pose[valid[0]]
        R0 = _quat_matrix(p0[3:])
        center = np.asarray(g.get("center", [0, 0, 0]), float)
        half = np.asarray(g["half_extents"], float)
        pts = []
        for oid, o in objects.items():
            if oid == sid:
                continue
            v = np.asarray(o.valid, bool)
            if not v.any():
                continue
            local = _local_points(o.geometry)
            for p in np.asarray(o.pose)[v]:
                world = p[:3] + local @ _quat_matrix(p[3:]).T
                pts.append((world - p0[:3]) @ R0)          # into the support frame
        if not pts:
            continue
        pts = np.concatenate(pts)
        lo = np.maximum(pts[:, :2].min(0) - margin, center[:2] - half[:2])
        hi = np.minimum(pts[:, :2].max(0) + margin, center[:2] + half[:2])
        if np.any(hi <= lo):
            raise ValueError(f"support {sid!r}: the objects' footprint does not overlap the support")
        new_center = np.r_[(lo + hi) / 2, center[2]]
        new_half = np.r_[(hi - lo) / 2, half[2]]
        geometry = {**g, "center": new_center.round(9).tolist(), "half_extents": new_half.round(9).tolist(),
                    "uncropped": {k: g[k] for k in ("center", "half_extents") if k in g}}
        out[sid] = ObjectTrack(pose=sup.pose, valid=sup.valid, role=sup.role, geometry=geometry)
        records.append({"object": sid, "kind": "crop_support_xy", "rule": CROP_RULE, "margin_m": margin,
                        "frame": "support body frame", "covered_objects": sorted(k for k in objects if k != sid),
                        "original": {"center": center.tolist(), "half_extents": half.tolist()},
                        "cropped": {"center": geometry["center"], "half_extents": geometry["half_extents"]}})
    return out, records


def rest_check(scene_ref: SceneRef, seconds: float = 2.0, options: dict | None = None) -> dict:
    """Simulate the scene without any robot and report object drift and penetration.

    Uses the tier-P global options (``robot.mjcf.OPTIONS``) unless ``options`` is given.
    Returns ``{"max_penetration_m", "worst_contact", "drift_m": {body: m},
    "rotation_rad": {body: rad}, "seconds"}``.
    """
    import mujoco

    from ..robot import mjcf as rmjcf

    o = {**rmjcf.OPTIONS, **(options or {})}
    spec = mujoco.MjSpec.from_string(scene_ref.mjcf)
    spec.option.timestep = o["timestep"]
    spec.option.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{o['integrator'].upper()}")
    spec.option.cone = getattr(mujoco.mjtCone, f"mjCONE_{o['cone'].upper()}")
    spec.option.impratio = o["impratio"]
    spec.option.iterations = o["iterations"]
    spec.option.gravity = list(o["gravity"])
    m = spec.compile()
    d = mujoco.MjData(m)
    for j, v in scene_ref.initial_qpos.items():
        adr = m.joint(j).qposadr[0]
        d.qpos[adr:adr + 7] = v
    mujoco.mj_forward(m, d)
    bodies = {m.body(m.jnt_bodyid[j]).name: m.jnt_bodyid[j] for j in range(m.njnt)
              if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
    start = {b: (d.xpos[i].copy(), d.xquat[i].copy()) for b, i in bodies.items()}
    worst, event = 0.0, None
    for _ in range(int(round(seconds / m.opt.timestep))):
        mujoco.mj_step(m, d)
        for c in d.contact[:d.ncon]:
            if -c.dist > worst:
                worst = float(-c.dist)
                event = {"time_s": float(d.time), "geoms": [m.geom(int(g)).name for g in c.geom]}
    drift, rot = {}, {}
    for b, i in bodies.items():
        drift[b] = float(np.linalg.norm(d.xpos[i] - start[b][0]))
        q = np.zeros(4)
        mujoco.mju_negQuat(q, start[b][1])
        r = np.zeros(4)
        mujoco.mju_mulQuat(r, d.xquat[i], q)
        rot[b] = float(2 * np.arccos(min(1.0, abs(r[0]))))
    return {"max_penetration_m": worst, "worst_contact": event, "drift_m": drift, "rotation_rad": rot,
            "seconds": seconds, "options": {k: (list(v) if isinstance(v, tuple) else v) for k, v in o.items()}}


__all__ = ["build", "scene_provenance", "rest_check", "crop_supports", "CROP_RULE", "geoms_of", "UnrepresentableObject", "NO_ROBOT_PREFIX",
           "CONTACT", "ASSUMPTIONS"]
