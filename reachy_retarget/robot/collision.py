"""Self-collision spheres of Reachy 2 (collision_spheres.json, a per-link visual-surface fit).

Sphere pairs are checked between different links except the SRDF-derived disabled pairs, with
every cross-arm pair (one `l_` and one `r_` link) re-enabled, as reachy-agent does
(robot/collision.py): the original exclusions assumed arms that never meet. The tripod stays
at 0; finger links follow the gripper joints of q. Signed distance between two spheres is
|c_a - c_b| - r_a - r_b (negative = penetration).
"""
import functools

import numpy as np

from .reachy import BODY, FIXED, JOINTS, planar
from .resources import collision_spec, profile, urdf
from .urdf import KinematicTree

BASE_FOOTPRINT_RADIUS = float(profile()["base"]["radius"])  # m, disc around base_link origin


class SelfCollision:
    """Sphere set and enabled sphere pairs; use `SelfCollision.load()` (cached)."""

    def __init__(self):
        spec = collision_spec()
        self.links = list(spec["spheres"])
        rows = [(i, s) for i, link in enumerate(self.links) for s in spec["spheres"][link]]
        self.link_index = np.array([i for i, _ in rows])
        self.local = np.array([s[:3] + [1.0] for _, s in rows])
        self.radii = np.array([s[3] for _, s in rows])
        cross = lambda a, b: {a[:2], b[:2]} == {"l_", "r_"}  # noqa: E731
        disabled = {frozenset(p) for p in spec["disabled_pairs"] if not cross(*p)}
        link_pairs = [(a, b) for a in range(len(self.links)) for b in range(a + 1, len(self.links))
                      if frozenset((self.links[a], self.links[b])) not in disabled]
        self.link_pairs = np.array(link_pairs)
        a, b = [], []
        for la, lb in link_pairs:
            ia, ib = np.flatnonzero(self.link_index == la), np.flatnonzero(self.link_index == lb)
            a.append(np.repeat(ia, len(ib)))
            b.append(np.tile(ib, len(ia)))
        self.a, self.b = np.concatenate(a), np.concatenate(b)
        self.pair_radii = self.radii[self.a] + self.radii[self.b]
        self.tree = KinematicTree(urdf(), "base_link", self.links, JOINTS[BODY.start:], FIXED)

    @classmethod
    @functools.cache
    def load(cls):
        return cls()

    def sphere_centers(self, q):
        """(world centres (..., N, 3), radii (N,), link index (N,) into self.links)."""
        q = np.asarray(q, float)
        poses = self.tree.fk(q[..., BODY.start:])
        T = np.stack([poses[link] for link in self.links], axis=-3)[..., self.link_index, :, :]
        T = planar(q[..., 0], q[..., 1], q[..., 2])[..., None, :, :] @ T
        return (T @ self.local[..., None])[..., :3, 0], self.radii, self.link_index

    def distances(self, q):
        """Signed distances (..., P) of all enabled sphere pairs."""
        c = self.sphere_centers(q)[0]
        return np.linalg.norm(c[..., self.a, :] - c[..., self.b, :], axis=-1) - self.pair_radii

    def self_clearance(self, q):
        """(minimum signed distance in m, (link_a, link_b)) for one configuration."""
        d = self.distances(q)
        k = int(np.argmin(d))
        return float(d[k]), (self.links[self.link_index[self.a[k]]], self.links[self.link_index[self.b[k]]])

    def min_clearance(self, q, chunk: int = 64):
        """Minimum signed distance (...,) for a batch of configurations (..., 22).

        Evaluated ``chunk`` configurations at a time: the pair distances of one configuration
        take ~0.8 MB, so a whole long episode at once (MimicGen kitchen: 2463 frames) would
        need ~2 GB of temporaries."""
        q = np.asarray(q, float)
        flat = q.reshape(-1, q.shape[-1])
        out = np.empty(len(flat))
        for i in range(0, len(flat), chunk):
            out[i:i + chunk] = self.distances(flat[i:i + chunk]).min(axis=-1)
        return out[0] if q.ndim == 1 else out.reshape(q.shape[:-1])


def sphere_centers(q):
    return SelfCollision.load().sphere_centers(q)


def self_clearance(q):
    return SelfCollision.load().self_clearance(q)


def min_clearance(q):
    return SelfCollision.load().min_clearance(q)


sphere_centers.__doc__ = SelfCollision.sphere_centers.__doc__
self_clearance.__doc__ = SelfCollision.self_clearance.__doc__
min_clearance.__doc__ = SelfCollision.min_clearance.__doc__
