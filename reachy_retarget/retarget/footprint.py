"""Floor-plane clearance between Reachy's body and scene geometry, by height band.

Reachy is modelled as a stack of discs around ``base_link`` (``BODY_PROFILE``): the mobile
base (radius ``BASE_FOOTPRINT_RADIUS``) up to 0.30 m, the tripod column up to 0.95 m and the
torso/head above. The column and torso radii are the largest horizontal distance from the
base axis of the MuJoCo collision geometry (``robot.mjcf``) of the non-arm links in each band,
rounded up (base 0.245 m, column 0.136 m, torso 0.183 m at zero posture). Arms are excluded:
they reach over tables by design and are checked by self-clearance and tier P.

Static obstacles are the ``support`` and ``fixture`` objects with box geometry (see
:func:`box_geometry`; a cylinder counts as its bounding box; pose = first valid row): their floor projection is the convex hull of
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
BODY_PROFILE = ((-np.inf, 0.30, BASE_FOOTPRINT_RADIUS), (0.30, 0.95, 0.14), (0.95, np.inf, 0.19))


def box_geometry(geometry: dict):
    """(center (3,), half extents (3,)) in the object frame of a ``box`` or body-frame ``aabb``
    geometry record, or the bounding box of a body-frame ``cylinder`` (``radius``,
    ``half_length``, ``axis`` "x"/"y"/"z", default "z"); optional ``center``, default 0. Else None."""
    if geometry.get("frame", "body") != "body":
        return None
    center = np.asarray(geometry.get("center", (0.0, 0.0, 0.0)), float)
    kind = geometry.get("kind")
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
        if kind in ("box", "aabb") and box is None:
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


def body_radius(z_low, z_high):
    """Largest BODY_PROFILE radius over the bands overlapping [z_low, z_high] (array-friendly);
    0 where no band overlaps."""
    z_low, z_high = np.asarray(z_low, float), np.asarray(z_high, float)
    r = np.zeros(np.broadcast(z_low, z_high).shape)
    for lo, hi, rad in BODY_PROFILE:
        r = np.where((z_low < hi) & (z_high > lo), np.maximum(r, rad), r)
    return r


def clearance(base_xy, obs: Obstacles) -> np.ndarray:
    """Clearance (N,) of Reachy's body discs at base positions (N, 2): distance to the nearest
    obstacle minus the radius of the body band at the obstacle's height (negative = overlap)."""
    xy = np.atleast_2d(np.asarray(base_xy, float))
    d = np.full(len(xy), np.inf)
    for poly, (lo, hi) in zip(obs.polygons, obs.heights):
        d = np.minimum(d, _polygon_distance(xy, poly) - body_radius(lo, hi))
    if len(obs.points):
        rad = body_radius(obs.point_heights[:, 0], obs.point_heights[:, 1])
        dist = np.linalg.norm(xy[:, None] - obs.points[None], axis=-1) - obs.point_radii - rad
        d = np.minimum(d, dist.min(axis=1))
    return d
