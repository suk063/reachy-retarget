"""Floor-plane clearance between Reachy's base disc and scene geometry.

Static obstacles are the ``support`` and ``fixture`` objects with ``box`` geometry
(``half_extents`` in the object frame, pose = first valid row): their floor projection is
the convex hull of the eight projected corners. Boxes whose top lies below
``cfg.floor_support_height`` are the floor itself and are ignored. Manipulated objects and
receptacles are moving obstacles: discs (radius = horizontal circumradius of the box, else
0) at every valid position of their trajectory. Other geometry kinds are not modelled and
are reported as notes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull

from ..robot import BASE_FOOTPRINT_RADIUS
from ..schema.rotations import quat_to_matrix
from .config import RetargetConfig

STATIC_ROLES = ("support", "fixture")


@dataclass
class Obstacles:
    polygons: list[np.ndarray] = field(default_factory=list)  # (n, 2) counter-clockwise hulls
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # moving object samples
    point_radii: np.ndarray = field(default_factory=lambda: np.zeros(0))
    notes: list[str] = field(default_factory=list)


def _box_corners(pose7, half):
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T
    return pose7[:3] + (signs * half) @ quat_to_matrix(pose7[3:]).T


def obstacles(objects, cfg: RetargetConfig, *, static_only=False) -> Obstacles:
    """Collect floor obstacles from ``{id: ObjectTrack}`` (see the module docstring)."""
    out = Obstacles()
    pts, radii = [], []
    for name, o in sorted(objects.items()):
        valid = np.flatnonzero(o.valid)
        if not len(valid):
            continue
        kind = o.geometry.get("kind")
        half = np.asarray(o.geometry.get("half_extents", ()), float) if kind == "box" else None
        if half is not None and half.shape != (3,):
            out.notes.append(f"object {name}: box without three half_extents ignored")
            half, kind = None, None
        if o.role in STATIC_ROLES:
            if half is None:
                out.notes.append(f"object {name}: {o.role} geometry kind {kind!r} not modelled for the base footprint")
                continue
            corners = _box_corners(o.pose[valid[0]], half)
            if corners[:, 2].max() < cfg.floor_support_height:
                continue
            xy = corners[:, :2]
            out.polygons.append(xy[ConvexHull(xy).vertices])
        elif not static_only:
            pts.append(o.pose[valid, :2])
            if half is None:
                r = 0.0
            else:
                corners = _box_corners(np.r_[0, 0, 0, o.pose[valid[0], 3:]], half)
                r = float(np.linalg.norm(corners[:, :2], axis=1).max())
            radii.append(np.full(len(valid), r))
    if pts:
        out.points, out.point_radii = np.concatenate(pts), np.concatenate(radii)
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


def clearance(base_xy, obs: Obstacles) -> np.ndarray:
    """Clearance (N,) of the base disc (BASE_FOOTPRINT_RADIUS) at base positions (N, 2):
    distance to the nearest obstacle minus the disc radius (negative = overlap)."""
    xy = np.atleast_2d(np.asarray(base_xy, float))
    d = np.full(len(xy), np.inf)
    for poly in obs.polygons:
        d = np.minimum(d, _polygon_distance(xy, poly))
    if len(obs.points):
        dist = np.linalg.norm(xy[:, None] - obs.points[None], axis=-1) - obs.point_radii
        d = np.minimum(d, dist.min(axis=1))
    return d - BASE_FOOTPRINT_RADIUS
