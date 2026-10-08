"""Floor-plane clearance between Reachy's body and scene geometry, by height band.

Reachy is modelled as a stack of discs around ``base_link`` (``BODY_PROFILE``): the mobile
base (radius ``BASE_FOOTPRINT_RADIUS``) up to 0.30 m, the tripod column up to 0.95 m and the
torso/head up to 1.45 m (head top 1.41 m); ``REST_ARM_BAND`` is added for an arm that rests the
whole episode (``Obstacles.bands``). The column and torso radii are the largest horizontal distance from the
base axis of the MuJoCo collision geometry (``robot.mjcf``) of the non-arm links in each band,
rounded up (base 0.245 m, column 0.136 m, torso 0.183 m at zero posture). Arms are excluded:
they reach over tables by design and are checked by self-clearance and tier P.

Static obstacles are the ``support`` and ``fixture`` objects with box geometry (see
:func:`box_geometry`; a cylinder counts as its bounding box, a union of ``boxes`` as its
enclosing box; pose = first valid row): their floor projection is the convex hull of
the eight projected corners and they occupy the height interval of their corners. Boxes whose
top lies below ``cfg.floor_support_height`` are the floor itself and are ignored. A body disc
is tested against an obstacle only when their height intervals overlap, so the base may pass
under a table top whose underside is above the base (source tables whose legs are visual-only
geometry, e.g. robosuite, have no collision geometry below the top). Manipulated objects and
receptacles are moving obstacles: discs (radius = horizontal circumradius of the box, else 0)
with their height interval at every valid position of their trajectory. Other geometry kinds
are not modelled and are reported as notes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull

from ..robot import BASE_FOOTPRINT_RADIUS
from ..schema.rotations import quat_to_matrix
from .config import RetargetConfig

STATIC_ROLES = ("support", "fixture")
# (z_low, z_high, radius) in m around the base_link axis; see the module docstring.
BODY_PROFILE = ((-10.0, 0.30, BASE_FOOTPRINT_RADIUS), (0.30, 0.95, 0.14), (0.95, 1.45, 0.19))
# An arm resting in the ``rest`` posture for a whole episode hangs beside the torso: its collision
# spheres span z 0.44-1.21 m out to 0.32 m from the base axis (robot.collision sphere model);
# :func:`arm_band` measures it for any posture (the navigation ``stow`` posture: 0.26 m).
REST_ARM_BAND = (0.44, 1.22, 0.33)


def arm_band(q, margin: float = 0.005):
    """(z_low, z_high, radius) of the arm collision spheres of posture q (22,) around the base axis
    (+ ``margin`` on the radius)."""
    from ..robot import SelfCollision
    sc = SelfCollision.load()
    c = sc.sphere_centers(np.asarray(q, float)[None])[0][0]
    arm = np.array([sc.links[sc.link_index[i]].startswith(("l_", "r_")) for i in range(len(c))])
    r = sc.radii[arm]
    return (float((c[arm, 2] - r).min()), float((c[arm, 2] + r).max()),
            float((np.linalg.norm(c[arm, :2], axis=1) + r).max() + margin))


def box_geometry(geometry: dict):
    """(center (3,), half extents (3,)) in the object frame of a ``box`` or body-frame ``aabb``
    geometry record, the bounding box of a body-frame ``cylinder`` (``radius``,
    ``half_length``, ``axis`` "x"/"y"/"z", default "z"; optional ``center``, default 0) or the
    enclosing box of a body-frame union of ``boxes`` (each ``{"center", "half_extents"}``, e.g.
    the ManiSkill PullCubeTool L tool). Else None. Like an ``aabb``, the enclosing box of
    ``boxes`` is an envelope: :func:`box_parts` gives the exact parts."""
    if geometry.get("frame", "body") != "body":
        return None
    kind = geometry.get("kind")
    if kind == "boxes":
        parts = box_parts(geometry)
        if not parts:
            return None
        lo = np.min([c - h for c, h in parts], axis=0)
        hi = np.max([c + h for c, h in parts], axis=0)
        return (lo + hi) / 2, (hi - lo) / 2
    center = np.asarray(geometry.get("center", (0.0, 0.0, 0.0)), float)
    if kind == "cylinder":
        axis = "xyz".find(str(geometry.get("axis", "z")))
        if axis < 0 or "radius" not in geometry or "half_length" not in geometry:
            return None
        half = np.full(3, float(geometry["radius"]))
        half[axis] = float(geometry["half_length"])
    elif kind in ("box", "aabb"):
        half = np.asarray(geometry.get("half_extents", ()), float)
    else:
        return None
    if half.shape != (3,) or center.shape != (3,):
        return None
    return center, half


def box_parts(geometry: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Exact solid boxes [(center (3,), half extents (3,)), ...] in the object frame: the parts of
    a body-frame ``boxes`` union (empty if any part is malformed), else ``[box_geometry]`` (one
    box, or [] without box geometry)."""
    if geometry.get("kind") != "boxes":
        box = box_geometry(geometry)
        return [] if box is None else [box]
    if geometry.get("frame", "body") != "body":
        return []
    out = []
    for b in geometry.get("boxes") or []:
        c = np.asarray(b.get("center", (0.0, 0.0, 0.0)), float)
        h = np.asarray(b.get("half_extents", ()), float)
        if c.shape != (3,) or h.shape != (3,):
            return []
        out.append((c, h))
    return out


def cylinder_geometry(geometry: dict):
    """(center (3,), axis index, radius, half_length) of a body-frame ``cylinder`` record, else None."""
    box = box_geometry(geometry)
    if box is None or geometry.get("kind") != "cylinder":
        return None
    axis = "xyz".index(str(geometry.get("axis", "z")))
    return box[0], axis, float(geometry["radius"]), float(geometry["half_length"])


@dataclass
class Obstacles:
    polygons: list[np.ndarray] = field(default_factory=list)  # (n, 2) counter-clockwise hulls
    heights: list[tuple[float, float]] = field(default_factory=list)  # (z_low, z_high) per polygon
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # moving object samples
    point_radii: np.ndarray = field(default_factory=lambda: np.zeros(0))
    point_heights: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    notes: list[str] = field(default_factory=list)
    bands: tuple = BODY_PROFILE  # body discs (z_low, z_high, radius); + REST_ARM_BAND when an arm rests


def _box_corners(pose7, center, half):
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T
    return pose7[:3] + (center + signs * half) @ quat_to_matrix(pose7[3:]).T


def obstacles(objects, cfg: RetargetConfig, *, static_only=False, held=None) -> Obstacles:
    """Collect floor obstacles from ``{id: ObjectTrack}`` (see the module docstring).

    ``held``: optional {object id: (T,) bool}; samples of a moving object while a hand holds it
    are skipped (the robot carries it)."""
    out = Obstacles()
    pts, radii, heights = [], [], []
    for name, o in sorted(objects.items()):
        valid = np.flatnonzero(o.valid)
        if not len(valid):
            continue
        kind = o.geometry.get("kind")
        box = box_geometry(o.geometry)
        if kind in ("box", "aabb", "boxes") and box is None:
            out.notes.append(f"object {name}: {kind} without a body-frame center/three half_extents ignored")
        if o.role in STATIC_ROLES:
            if box is None:
                out.notes.append(f"object {name}: {o.role} geometry kind {kind!r} not modelled for the base footprint")
                continue
            corners = _box_corners(o.pose[valid[0]], *box)
            if corners[:, 2].max() < cfg.floor_support_height:
                continue
            xy = corners[:, :2]
            out.polygons.append(xy[ConvexHull(xy).vertices])
            out.heights.append((float(corners[:, 2].min()), float(corners[:, 2].max())))
        elif not static_only:
            if held is not None and name in held:
                valid = valid[~np.asarray(held[name], bool)[valid]]
                if not len(valid):
                    continue
            if box is None:
                c, r, dz = np.zeros(3), 0.0, (0.0, 0.0)
            else:
                corners = _box_corners(np.r_[0, 0, 0, o.pose[valid[0], 3:]], *box)
                c = corners.mean(axis=0)
                r = float(np.linalg.norm(corners[:, :2] - c[:2], axis=1).max())
                dz = (corners[:, 2].min(), corners[:, 2].max())
            R = quat_to_matrix(o.pose[valid, 3:])
            centre = o.pose[valid, :3] + (R @ (box[0] if box is not None else np.zeros(3)))
            pts.append(centre[:, :2])
            radii.append(np.full(len(valid), r))
            heights.append(np.c_[o.pose[valid, 2] + dz[0], o.pose[valid, 2] + dz[1]])
    if pts:
        out.points, out.point_radii = np.concatenate(pts), np.concatenate(radii)
        out.point_heights = np.concatenate(heights)
    return out


def _polygon_distance(xy, poly):
    """Signed distance (N,) from points (N, 2) to a convex counter-clockwise polygon."""
    a, b = poly, np.roll(poly, -1, axis=0)
    e = b - a
    rel = xy[:, None, :] - a[None]
    t = np.clip(np.sum(rel * e, -1) / np.sum(e * e, -1), 0, 1)
    dist = np.linalg.norm(rel - t[..., None] * e, axis=-1).min(axis=1)
    inside = np.all(e[None, :, 0] * rel[..., 1] - e[None, :, 1] * rel[..., 0] >= 0, axis=1)
    return np.where(inside, -dist, dist)


def body_radius(z_low, z_high, bands=BODY_PROFILE):
    """Largest radius of the body ``bands`` (default BODY_PROFILE) overlapping [z_low, z_high]
    (array-friendly); 0 where no band overlaps."""
    z_low, z_high = np.asarray(z_low, float), np.asarray(z_high, float)
    r = np.zeros(np.broadcast(z_low, z_high).shape)
    for lo, hi, rad in bands:
        r = np.where((z_low < hi) & (z_high > lo), np.maximum(r, rad), r)
    return r


def _packed(obs: Obstacles):
    """(vertices (P, V, 2) padded by repeating the last vertex, body radius (P,)) of the polygons,
    cached on ``obs`` while its polygon list is unchanged."""
    key = (id(obs.polygons), len(obs.polygons), tuple(obs.bands))
    cache = getattr(obs, "_packed_cache", None)
    if cache is None or cache[0] != key:
        V = max(len(p) for p in obs.polygons)
        verts = np.stack([np.concatenate([p, np.repeat(p[-1:], V - len(p), axis=0)]) for p in obs.polygons])
        rad = np.array([body_radius(lo, hi, obs.bands) for lo, hi in obs.heights], float)
        cache = (key, verts, rad)
        obs._packed_cache = cache
    return cache[1], cache[2]


def _polygons_distance(xy, verts, chunk: int = 256):
    """Signed distances (N, P) from points (N, 2) to convex counter-clockwise polygons (P, V, 2)
    (padded with repeated vertices: zero-length edges count as inside-tests that always pass)."""
    a, b = verts, np.roll(verts, -1, axis=1)
    e = b - a                                              # (P, V, 2)
    ee = np.maximum(np.sum(e * e, -1), 1e-18)
    out = np.empty((len(xy), len(verts)))
    for i in range(0, len(xy), chunk):
        rel = xy[i:i + chunk, None, None, :] - a[None]     # (n, P, V, 2)
        t = np.clip(np.sum(rel * e, -1) / ee, 0, 1)
        dist = np.linalg.norm(rel - t[..., None] * e, axis=-1).min(axis=-1)
        cross = e[None, ..., 0] * rel[..., 1] - e[None, ..., 1] * rel[..., 0]
        inside = np.all(cross >= -1e-15, axis=-1)
        out[i:i + chunk] = np.where(inside, -dist, dist)
    return out


def clearance(base_xy, obs: Obstacles) -> np.ndarray:
    """Clearance (N,) of Reachy's body discs at base positions (N, 2): distance to the nearest
    obstacle minus the radius of the body band at the obstacle's height (negative = overlap)."""
    xy = np.atleast_2d(np.asarray(base_xy, float))
    d = np.full(len(xy), np.inf)
    if obs.polygons:
        verts, rad = _packed(obs)
        d = np.minimum(d, (_polygons_distance(xy, verts) - rad).min(axis=1))
    if len(obs.points):
        rad = body_radius(obs.point_heights[:, 0], obs.point_heights[:, 1], obs.bands)
        dist = np.linalg.norm(xy[:, None] - obs.points[None], axis=-1) - obs.point_radii - rad
        d = np.minimum(d, dist.min(axis=1))
    return d


def scene_obstacles(scene_ref, cfg: RetargetConfig, near_xy=None, radius: float = 1.5) -> Obstacles:
    """Static floor obstacles from a source MuJoCo scene (``SceneRef``): every colliding geom of
    the environment (bodies that are neither Reachy, the removed source robot, nor under a free
    joint: walls, counters, cabinets, appliances, articulated doors and drawers at their initial
    joint state), as the convex hull of its oriented bounding box projected on the floor with its
    height interval (the same records as box objects). Planes and geoms whose top is below
    ``cfg.floor_support_height`` are the floor. With ``near_xy`` (n, 2) only geoms whose hull
    comes within ``radius`` of one of those points are kept (house-scale scenes hold thousands).

    The scene is compiled as tier P builds it (:func:`validate.scene.build_scene`, inactive bodies
    declared by the adapter removed); compile failures give no polygons and a note."""
    import mujoco

    from ..validate.scene import build_scene, reset, subtree
    out = Obstacles()
    try:
        scene = build_scene(scene_ref, drop_bodies=sorted(scene_ref.inactive_bodies or ()))
    except Exception as err:  # missing assets, unsupported MJCF: no scene geometry
        out.notes.append(f"scene geometry unavailable: {type(err).__name__}: {err}")
        return out
    m = scene.model
    d = reset(scene, np.zeros(22))
    skip = set(scene.reachy_bodies())
    for b in scene.free_bodies.values():
        skip |= subtree(m, m.body(b).id)
    pts = None if near_xy is None else np.asarray(near_xy, float).reshape(-1, 2)
    if pts is not None and len(pts) > 400:
        pts = pts[np.linspace(0, len(pts) - 1, 400).round().astype(int)]
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T
    kept = 0
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) in skip or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        corners = d.geom_xpos[g] + (c + signs * h) @ d.geom_xmat[g].reshape(3, 3).T
        if corners[:, 2].max() < cfg.floor_support_height:
            continue
        xy = corners[:, :2]
        try:
            hull = xy[ConvexHull(xy).vertices]
        except Exception:  # degenerate (zero-area) projection: a thin vertical plate seen edge-on
            lo, hi = xy.min(axis=0) - 1e-3, xy.max(axis=0) + 1e-3
            hull = np.array([lo, (hi[0], lo[1]), hi, (lo[0], hi[1])])
        if pts is not None and _polygon_distance(pts, hull).min() > radius:
            continue
        out.polygons.append(hull)
        out.heights.append((float(corners[:, 2].min()), float(corners[:, 2].max())))
        kept += 1
    out.notes.append(f"scene geometry: {kept} colliding environment geoms near the path")
    return out


def merge_scene(objects: Obstacles, scene: Obstacles, top_tol: float = 0.02) -> Obstacles:
    """``objects`` (from :func:`obstacles`) merged with source-scene geometry (:func:`scene_obstacles`),
    without the static object envelopes the scene geometry already resolves.

    A static ``support``/``fixture`` object is the enclosing box of all its collision geoms: a table
    with collision legs becomes a block down to the floor (LIBERO's study table: z 0-0.88 m, its
    top is a 0.81-0.89 m slab over legs-free space), which keeps the base disc 0.1 m farther back than
    the tripod column needs. An envelope polygon is dropped when a scene polygon whose centroid lies
    inside it reaches the envelope's top within ``top_tol`` (the same body, resolved per geom)."""
    if not scene.polygons or not objects.polygons:
        return merge(objects, scene)
    cents = np.array([p.mean(axis=0) for p in scene.polygons])
    tops = np.array([h[1] for h in scene.heights])
    keep, dropped = [], 0
    for poly, (lo, hi) in zip(objects.polygons, objects.heights):
        inside = _polygon_distance(cents, poly) <= 0
        if np.any(inside & (np.abs(tops - hi) <= top_tol)):
            dropped += 1
            continue
        keep.append((poly, (lo, hi)))
    rest = Obstacles(polygons=[p for p, _ in keep], heights=[h for _, h in keep], points=objects.points,
                     notes=list(objects.notes), bands=objects.bands)
    if len(objects.points):
        rest.point_radii, rest.point_heights = objects.point_radii, objects.point_heights
    if dropped:
        rest.notes.append(f"{dropped} static object envelopes replaced by their scene geoms")
    return merge(rest, scene)


def merge(*parts: Obstacles) -> Obstacles:
    """One obstacle set holding every polygon and point of ``parts``."""
    out = Obstacles(bands=tuple(sorted(set().union(*(p.bands for p in parts))))) if parts else Obstacles()
    for p in parts:
        out.polygons += list(p.polygons)
        out.heights += list(p.heights)
        out.notes += list(p.notes)
    pts = [p for p in parts if len(p.points)]
    if pts:
        out.points = np.concatenate([p.points for p in pts])
        out.point_radii = np.concatenate([p.point_radii for p in pts])
        out.point_heights = np.concatenate([p.point_heights for p in pts])
    return out
