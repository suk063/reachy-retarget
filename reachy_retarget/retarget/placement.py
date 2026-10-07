"""Base placement (docs/design.md, step 3).

Fixed-base sources get one floor pose ``(x, y, yaw)``; mobile sources get a constant SE(2)
offset composed with the source base path (``se2_compose(source_base, offset)``). Both are
chosen by the same score on 8-12 keyframes:

1. candidates: around ``base_hint`` (fixed base), on rings around the keyframe target
   centroid (fixed base), or on a body-frame offset grid (mobile);
2. a cheap reach proxy (shoulder-to-grasp-center distance, targets in front) ranks them;
3. the best few are scored by bounded IK on the keyframes (both gripper half-turn
   symmetries per side, the better one kept): normalized TCP residual (mean and max),
   joint-limit margin and manipulability; a base footprint overlapping scene geometry
   (see :mod:`.footprint`) is a hard constraint (large penalty growing with the overlap);
4. Nelder-Mead refines the best candidate.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ..robot import LOWER, UPPER, Reachy
from ..schema.rotations import se2_compose
from . import footprint
from .config import RetargetConfig
from .targets import tcp_targets
from .wbik import ARM_COLUMNS, FrameSolver, normalized_error, tcp_errors

INFEASIBLE = 100.0      # cost added when the base footprint overlaps scene geometry
REACH = (0.25, 0.62)    # comfortable shoulder-to-grasp-center distance, m
LIMIT_SOFT = 0.15       # joint-limit margin below which the score is penalized, rad


@functools.cache
def shoulders():
    """{side: shoulder pivot position (3,) in base_link}."""
    u = Reachy.load().urdf
    return {side: u.transform("base_link", u.by_name[f"{side[0]}_shoulder_pitch"]["child"])[:3, 3]
            for side in ("left", "right")}


@functools.cache
def _reference_manipulability(nominal_key):
    q = np.asarray(nominal_key)
    J = Reachy.load().jacobian(q, "right_tcp")[1][:, ARM_COLUMNS["right"]]
    return float(np.sqrt(np.linalg.det(J @ J.T)))


def keyframes(T, labels, n):
    """About ``n`` frame indices: evenly spaced plus the first/last frame of each grasp phase."""
    idx = set(np.linspace(0, T - 1, n).round().astype(int).tolist())
    for lab in labels:
        change = np.flatnonzero(np.diff(lab) != 0)
        idx.update(change.tolist() + (change + 1).tolist())
    idx = np.array(sorted(idx))
    if len(idx) > n + 2:
        idx = idx[np.linspace(0, len(idx) - 1, n + 2).round().astype(int)]
    return idx


@dataclass
class Placement:
    """Chosen base reference and its score."""

    base: np.ndarray              # (T, 3) base reference path (constant rows for a fixed base)
    mobile: bool
    pose: np.ndarray              # (3,) fixed base pose, or the offset from the source base
    flips: dict[str, bool]        # gripper half-turn per side
    cost: float
    seed: np.ndarray              # (22,) IK solution at frame 0 (active arms, base)
    diagnostics: dict = field(default_factory=dict)


@dataclass
class Score:
    cost: float
    flips: dict[str, bool]
    details: dict
    seed: np.ndarray              # (22,) nominal posture with the scored arms' first-keyframe solution


class PlacementProblem:
    """Score base placements for the effector-to-side map ``sides`` of a source episode."""

    def __init__(self, src, sides, cfg: RetargetConfig, nominal, labels):
        self.src, self.sides, self.cfg = src, dict(sides), cfg
        self.nominal = np.asarray(nominal, float)
        self.mobile = src.base is not None
        self.kf = keyframes(src.length, list(labels.values()), cfg.placement_keyframes)
        self.targets = {}
        for flip in (False, True):
            full = tcp_targets(src, sides, dict.fromkeys(sides.values(), flip))
            self.targets[flip] = {s: X[self.kf] for s, X in full.items()}
        points = [src.effectors[k].pose[self.kf, :3, 3] for k in sides]
        self.points = np.concatenate(points) if points else np.zeros((0, 3))
        self.obstacles = footprint.obstacles(src.objects, cfg, static_only=self.mobile)
        self.solvers = {s: FrameSolver(cfg, (s,), False, self.nominal, collisions=False) for s in sides.values()}
        self.manip_ref = _reference_manipulability(tuple(self.nominal))

    def path(self, pose, rows=None):
        """Base path (n, 3) for a placement parameter at source rows (default: keyframes)."""
        rows = self.kf if rows is None else rows
        if self.mobile:
            return se2_compose(self.src.base[rows], np.asarray(pose, float))
        return np.tile(np.asarray(pose, float), (len(rows), 1))

    def footprint_clearance(self, pose):
        """Minimum base-disc clearance minus the configured margin along the path."""
        rows = np.arange(self.src.length) if self.mobile else self.kf[:1]
        clear = footprint.clearance(self.path(pose, rows)[:, :2], self.obstacles)
        return float(clear.min()) - self.cfg.footprint_margin

    def proxy(self, pose):
        """Cheap reach score: shoulder distance outside REACH, targets behind the shoulder."""
        base = self.path(pose)
        cost = 0.0
        for key, side in self.sides.items():
            p = self.src.effectors[key].pose[self.kf, :3, 3]
            c, s = np.cos(base[:, 2]), np.sin(base[:, 2])
            d = p[:, :2] - base[:, :2]
            local = np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], p[:, 2]] - shoulders()[side]
            r = np.linalg.norm(local, axis=1)
            cost += np.mean(np.maximum(0, r - REACH[1]) ** 2 + np.maximum(0, REACH[0] - r) ** 2
                            + np.maximum(0, 0.1 - local[:, 0]) ** 2)
        clear = self.footprint_clearance(pose)
        return cost + (INFEASIBLE - clear if clear < 0 else 0.0)

    def _side_score(self, side, base, flip, seed):
        """(cost, details, q at the first keyframe) of one arm along the keyframes.

        Without a ``seed`` the first keyframe is a multi-start cold solve; with one (the first
        keyframe solution of a nearby placement) it is warm started like the later keyframes.
        """
        solver, X = self.solvers[side], self.targets[flip][side]
        qs = np.empty((len(base), 22))
        for k in range(len(base)):
            frame = {side: X[k]}
            if k == 0 and seed is None:
                q = solver.cold_solve(self.nominal, frame, base[k], self.cfg.placement_iter)[0]
            else:
                q0 = seed if k == 0 else qs[k - 1]
                q = solver.solve(q0, frame, q0, base[k], self.cfg.placement_iter * (2 if k == 0 else 1))[0]
            qs[k] = q
        e = normalized_error(tcp_errors(qs, {side: X}), self.cfg)
        cols = ARM_COLUMNS[side]
        margin = np.minimum(qs[:, cols] - LOWER[cols], UPPER[cols] - qs[:, cols]).min(axis=1)
        manip = np.mean([solver.manipulability(qk) for qk in qs]) / self.manip_ref
        cost = (np.mean(e) + np.max(e) + 0.5 * np.mean(np.maximum(0, LIMIT_SOFT - margin) / LIMIT_SOFT)
                - 0.2 * min(manip, 1.0))
        return cost, {"max_normalized_residual": float(np.max(e)), "min_limit_margin": float(margin.min()),
                      "manipulability": float(manip)}, qs[0]

    def score(self, pose, flips=None, seed=None) -> Score:
        """Score a placement parameter; ``flips=None`` tries both half turns per side and
        ``seed`` (22,) warm starts the first keyframe (see :meth:`_side_score`)."""
        base = self.path(pose)
        out = Score(0.0, {}, {}, self.nominal.copy())
        for side in self.sides.values():
            options = [flips[side]] if flips is not None else [False, True]
            results = [(self._side_score(side, base, f, seed), f) for f in options]
            (c, d, q), f = min(results, key=lambda r: r[0][0])
            out.cost += c
            out.flips[side], out.details[side] = f, d
            out.seed[ARM_COLUMNS[side]] = q[ARM_COLUMNS[side]]
        out.seed[:3] = base[0]
        clear = self.footprint_clearance(pose)
        out.details["footprint_clearance"] = clear
        if clear < 0:
            out.cost += INFEASIBLE - 10 * clear
        return out

    def candidates(self):
        """Candidate placement parameters (n, 3) before ranking."""
        cfg = self.cfg
        if self.mobile:
            g = np.array(cfg.placement_mobile_offsets)
            return np.array([(dx, dy, 0.0) for dx in g for dy in g])
        out = []
        lateral = np.mean([shoulders()[s][1] for s in self.sides.values()]) if self.sides else 0.0
        if self.src.base_hint is not None:
            h = self.src.base_hint
            out += [se2_compose(h, (dx, dy, 0.0)) for dx in cfg.placement_hint_offsets
                    for dy in cfg.placement_hint_offsets]
        if len(self.points):
            c = self.points[:, :2].mean(axis=0)
            for yaw in np.linspace(-np.pi, np.pi, cfg.placement_yaws, endpoint=False):
                R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                out += [np.r_[c - R @ (r, lateral), yaw] for r in cfg.placement_radii]
        if not out:
            out.append(np.zeros(3))
        return np.array(out)

    def search(self):
        """Best candidate as a :class:`Placement` (not yet refined)."""
        cand = self.candidates()
        proxy = np.array([self.proxy(p) for p in cand])
        order = np.argsort(proxy, kind="stable")[: self.cfg.placement_candidates]
        if self.src.base_hint is not None and not self.mobile:
            n_hint = len(self.cfg.placement_hint_offsets) ** 2
            best_hint = int(np.argmin(proxy[:n_hint]))
            if best_hint not in order:
                order = np.r_[order, best_hint]
        if self.sides:
            scored = [(self.score(cand[i]), i) for i in order]
        else:
            scored = [(Score(float(proxy[i]), {}, {}, self.nominal.copy()), i) for i in order]
        best, i = min(scored, key=lambda s: s[0].cost)
        diag = {"candidates": len(cand), "scored": len(order), "proxy": float(proxy[i]), "details": best.details,
                "obstacle_notes": self.obstacles.notes, "keyframes": self.kf.tolist()}
        return self._placement(cand[i], best, diag)

    def _placement(self, pose, score: Score, diag):
        return Placement(self.path(pose, np.arange(self.src.length)), self.mobile, np.asarray(pose, float),
                         score.flips, float(score.cost), score.seed, diag)

    def refine(self, placement: Placement):
        """Nelder-Mead refinement of ``placement`` (flips fixed, warm started from its seed)."""
        if not self.sides or self.cfg.placement_refine_evals <= 0:
            return placement
        def fn(p):
            return self.score(p, placement.flips, placement.seed).cost

        simplex = placement.pose + np.vstack([np.zeros(3), np.diag([0.05, 0.05, 0.1])])
        res = minimize(fn, placement.pose, method="Nelder-Mead",
                       options={"maxfev": self.cfg.placement_refine_evals, "xatol": 0.005, "fatol": 1e-3,
                                "initial_simplex": simplex})
        best = self.score(res.x, placement.flips, placement.seed)
        if best.cost >= placement.cost:
            return placement
        diag = dict(placement.diagnostics, details=best.details, refined_from=placement.pose.tolist(),
                    refine_evals=int(res.nfev))
        return self._placement(res.x, best, diag)


def diagnostics(p: Placement) -> dict:
    """JSON-compatible summary of a placement for ``episode.extra``."""
    key = "offset" if p.mobile else "pose"
    return {"mobile": p.mobile, key: np.asarray(p.pose).tolist(), "flips": p.flips, "cost": p.cost, **p.diagnostics}
