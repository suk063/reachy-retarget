"""Base placement (docs/design.md, step 3).

Fixed-base sources get one floor pose ``(x, y, yaw)``; mobile sources get a constant SE(2)
offset composed with the source base path (``se2_compose(source_base, offset)``). Both are
chosen by the same score on 8-12 keyframes:

1. candidates: around ``base_hint`` (fixed base), on rings around the keyframe target
   centroid (fixed base), or on a body-frame offset grid (mobile);
2. a cheap reach proxy (shoulder-to-grasp-center distance, targets in front) ranks them;
3. the best few are scored by bounded IK on the keyframes (grasp offset candidates per side,
   see :mod:`.targets`, the first acceptable or best one kept, tilts penalized by
   ``cfg.grasp_tilt_cost``; offsets whose fingers would sink into scene boxes along the path
   (:func:`.targets.finger_penetration`) more than ``cfg.finger_depth_slack`` beyond the best
   offset are discarded before any IK):
   normalized TCP residual (mean and max),
   joint-limit margin, manipulability and self-clearance below the IK margin (the scoring IK has
   no repulsion term); a base footprint overlapping scene geometry
   (see :mod:`.footprint`) is a hard constraint (large penalty growing with the overlap);
4. Nelder-Mead refines the best candidate.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ..robot import LOWER, UPPER, Reachy, gripper
from ..robot.resources import profile
from ..schema.rotations import se2_compose
from . import footprint
from .config import RetargetConfig
from .targets import (approach_width, finger_angles, finger_penetration, grasp_segments, grasp_symmetry,
                      hand_object_distance, held_masks, offset_candidates, orientation_weight, source_closed,
                      tcp_targets, touch_labels)
from .wbik import ARM_COLUMNS, Clearance, FrameSolver, tcp_errors

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
    offsets: dict[str, tuple]     # grasp offset (flip, theta_deg, beta_deg) per side
    cost: float
    seed: np.ndarray              # (22,) IK solution at frame 0 (active arms, base)
    diagnostics: dict = field(default_factory=dict)
    segments: dict = field(default_factory=dict)  # side -> [(a, b, offset)] per grasp segment

    @property
    def flips(self) -> dict[str, bool]:
        return {s: bool(o[0]) for s, o in self.offsets.items()}


@dataclass
class Score:
    cost: float
    offsets: dict[str, tuple]
    details: dict
    seed: np.ndarray              # (22,) nominal posture with the scored arms' first-keyframe solution


class PlacementProblem:
    """Score base placements for the effector-to-side map ``sides`` of a source episode."""

    def __init__(self, src, sides, cfg: RetargetConfig, nominal, labels):
        self.src, self.sides, self.cfg = src, dict(sides), cfg
        self.nominal = np.asarray(nominal, float)
        self.mobile = src.base is not None
        self.kf = keyframes(src.length, list(labels.values()), cfg.placement_keyframes)
        self.symmetry = {side: grasp_symmetry(src, key, labels.get(key, np.full(src.length, -1)), cfg)
                         for key, side in sides.items()}
        self.options, self.finger_depth, self.rot_weight = {}, {}, {}
        self.labels = {side: np.asarray(labels.get(key, np.full(src.length, -1))) for key, side in sides.items()}
        for key, side in sides.items():
            lab = labels.get(key, np.full(src.length, -1))
            dist = hand_object_distance(src, key) if cfg.orientation_by_distance else None
            self.rot_weight[side] = orientation_weight(src.time, lab, cfg, dist)[self.kf]
            finger = finger_angles(src.effectors[key], lab >= 0, cfg)
            opts = offset_candidates(self.symmetry[side], cfg)
            # objects a closed hand touches are in contact by design, like held ones (StackPyramid:
            # the closed fingers push cubeA; counting that contact as depth rejected every tilt)
            touch = touch_labels(src, key, lab, cfg, source_closed(src.effectors[key], cfg, src.time))
            def opened(o, finger=finger, key=key, lab=lab):  # the fingers as commanded (narrowed approach)
                if not cfg.approach_narrow:
                    return finger
                return np.minimum(finger, gripper.width_to_angle(approach_width(src, key, o, lab, cfg)))
            depth = {o: float(finger_penetration(src, key, side, o, touch, opened(o), cfg).max()) for o in opts}
            best = min(depth.values())
            self.options[side] = [o for o in opts if depth[o] <= best + cfg.finger_depth_slack]
            self.finger_depth[side] = {f"{int(o[0])}/{o[1]:g}/{o[2]:g}": round(v, 4) for o, v in depth.items()}
        self.targets = {}
        for key, side in sides.items():
            for o in self.options[side]:
                self.targets[side, o] = tcp_targets(src, {key: side}, {side: o})[side][self.kf]
        points = [src.effectors[k].pose[self.kf, :3, 3] for k in sides]
        self.points = np.concatenate(points) if points else np.zeros((0, 3))
        self.obstacles = footprint.obstacles(src.objects, cfg, static_only=self.mobile,
                                             held=held_masks(src.objects, labels.values()))
        self.solvers = {s: FrameSolver(cfg, (s,), False, self.nominal, collisions=False) for s in sides.values()}
        self.clearance = {s: Clearance((s,)) for s in sides.values()}
        rest = profile()["postures"]["rest"]
        self.rest = np.zeros(22)
        self.rest[ARM_COLUMNS["left"]], self.rest[ARM_COLUMNS["right"]] = np.radians(rest["left"]), np.radians(rest["right"])
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

    def _target(self, side, offset):
        if (side, offset) not in self.targets:
            key = next(k for k, s in self.sides.items() if s == side)
            self.targets[side, offset] = tcp_targets(self.src, {key: side}, {side: offset})[side][self.kf]
        return self.targets[side, offset]

    def segment_offsets(self, placement: Placement) -> Placement:
        """Per-segment grasp offsets for hands with several grasp segments (robomimic Transport:
        one arm opens the lid, then takes the payload; no single offset suits both objects).

        Every segment re-selects among its own candidates (:func:`.targets.grasp_symmetry` and
        the finger-depth screen on that segment only), scored on the keyframes of its window
        (``cfg.approach_window_s`` before to ``cfg.retreat_window_s`` after) at the chosen base
        placement; the episode-wide offset is kept unless a candidate scores lower there."""
        cfg = self.cfg
        if not cfg.segment_offsets:
            return placement
        times = np.asarray(self.src.time, float)
        tk = times[self.kf]
        base = self.path(placement.pose)
        segments, diag = {}, {}
        for key, side in self.sides.items():
            segs = grasp_segments(self.labels[side])
            if len(segs) < 2:
                continue
            chosen, info = [], []
            for a, b in segs:
                rows = np.flatnonzero((tk >= times[a] - cfg.approach_window_s) & (tk <= times[b - 1] + cfg.retreat_window_s))
                default = placement.offsets[side]
                if not len(rows):
                    chosen.append((a, b, default))
                    continue
                lab = np.where((np.arange(len(times)) >= a) & (np.arange(len(times)) < b), self.labels[side], -1)
                sym = grasp_symmetry(self.src, key, lab, cfg)
                finger = finger_angles(self.src.effectors[key], lab >= 0, cfg)
                opts = offset_candidates(sym, cfg)
                depth = {o: float(finger_penetration(self.src, key, side, o, lab, finger, cfg)[a:b].max()) for o in opts}
                best_depth = min(depth.values())
                opts = [o for o in opts if depth[o] <= best_depth + cfg.finger_depth_slack]
                c0, _, q0 = self._side_score(side, base, default, None, rows)
                best = (c0, default)
                for o in opts:
                    if o == default:
                        continue
                    self._target(side, o)
                    c = self._side_score(side, base, o, q0, rows)[0]
                    if c < best[0]:
                        best = (c, o)
                chosen.append((a, b, best[1]))
                info.append({"rows": [int(a), int(b)], "offset": list(best[1]), "cost": float(best[0]),
                             "default_cost": float(c0), "candidates": len(opts)})
            if any(o != placement.offsets[side] for _, _, o in chosen):
                segments[side] = chosen
            diag[side] = info
        placement.segments = segments
        placement.diagnostics = dict(placement.diagnostics, segment_offsets=diag)
        return placement

    def _side_score(self, side, base, offset, seed, rows=None, iters=None):
        """(cost, details, q at the first keyframe) of one arm along the keyframes (or the
        keyframe subset ``rows``).

        Without a ``seed`` the first keyframe is a multi-start cold solve; with one (the first
        keyframe solution of a nearby placement or grasp offset) it is warm started like the
        later keyframes.
        """
        solver, X = self.solvers[side], self._target(side, offset)
        iters = iters or self.cfg.placement_iter
        rows = np.arange(len(base)) if rows is None else np.asarray(rows)
        X, base, w = X[rows], base[rows], self.rot_weight[side][rows]
        qs = np.empty((len(base), 22))
        for k in range(len(base)):
            frame = {side: X[k]}
            if k == 0 and seed is None:
                q = solver.cold_solve(self.nominal, frame, base[k], iters, rot_scale={side: w[k]})[0]
            else:
                q0 = seed if k == 0 else qs[k - 1]
                q = solver.solve(q0, frame, q0, base[k], iters * (2 if k == 0 else 1), rot_scale={side: w[k]})[0]
            qs[k] = q
        pos, rot = tcp_errors(qs, {side: X})[side]
        e = np.maximum(pos / self.cfg.tcp_pos_tol, w * rot / self.cfg.tcp_rot_tol)
        cols = ARM_COLUMNS[side]
        margin = np.minimum(qs[:, cols] - LOWER[cols], UPPER[cols] - qs[:, cols]).min(axis=1)
        manip = np.mean([solver.manipulability(qk) for qk in qs]) / self.manip_ref
        qc = qs.copy()  # other arms hang in the rest posture during the trajectory
        for other, cols in ARM_COLUMNS.items():
            if other not in self.sides.values():
                qc[:, cols] = self.rest[cols]
        clear = self.clearance[side].minimum(qc)  # the scoring IK has no repulsion term
        crowd = np.maximum(0.0, self.cfg.self_clearance_margin - clear) / self.cfg.self_clearance_margin
        cost = (np.mean(e) + np.max(e) + 0.5 * np.mean(np.maximum(0, LIMIT_SOFT - margin) / LIMIT_SOFT)
                - 0.2 * min(manip, 1.0) + self.cfg.grasp_tilt_cost * abs(offset[2]) / 30.0
                + np.mean(crowd) + np.max(crowd))
        return cost, {"max_normalized_residual": float(np.max(e)), "min_limit_margin": float(margin.min()),
                      "manipulability": float(manip), "min_self_clearance": float(clear.min())}, qs[0]

    def _best_offset(self, side, base, seed):
        """Grasp offset of one side at one placement. The first keyframe is cold-solved once with
        the source frame (unless ``seed`` is given) and every candidate is warm started from it.
        Candidates are tried in order of preference (smallest tilt first, see
        :func:`.targets.offset_candidates`) on ``cfg.placement_quick_keyframes`` keyframes; the
        first whose quick and full scores keep every keyframe residual below
        ``cfg.placement_accept`` of the tolerance is taken. Otherwise the
        ``cfg.placement_offsets_full`` best quick scores are scored on all keyframes."""
        options = self.options[side]
        if seed is None:
            seed = self._side_score(side, base, options[0], None, [0])[2]
        if len(options) == 1:
            return self._side_score(side, base, options[0], seed), options[0]
        rows = np.unique(np.linspace(0, len(base) - 1, min(len(base), self.cfg.placement_quick_keyframes))
                         .round().astype(int))
        quick = []
        for i, o in enumerate(options):
            c, d, _ = self._side_score(side, base, o, seed, rows, self.cfg.placement_quick_iter)
            quick.append((c, i))
            if d["max_normalized_residual"] < self.cfg.placement_accept:
                full = self._side_score(side, base, o, seed)
                if full[1]["max_normalized_residual"] < self.cfg.placement_accept:
                    return full, o
        full = [(self._side_score(side, base, options[i], seed), options[i])
                for _, i in sorted(quick)[: self.cfg.placement_offsets_full]]
        return min(full, key=lambda r: r[0][0])

    def score(self, pose, offsets=None, seed=None) -> Score:
        """Score a placement parameter; ``offsets=None`` searches the grasp offset candidates
        per side (:meth:`_best_offset`) and ``seed`` (22,) warm starts the first keyframe (see
        :meth:`_side_score`)."""
        base = self.path(pose)
        out = Score(0.0, {}, {}, self.nominal.copy())
        for side in self.sides.values():
            if offsets is not None:
                (c, d, q), o = self._side_score(side, base, offsets[side], seed), offsets[side]
            else:
                (c, d, q), o = self._best_offset(side, base, seed)
            out.cost += c
            out.offsets[side], out.details[side] = o, d
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
                "grasp_symmetry": self.symmetry, "finger_depth_by_offset": self.finger_depth,
                "obstacle_notes": self.obstacles.notes, "keyframes": self.kf.tolist()}
        return self._placement(cand[i], best, diag)

    def _placement(self, pose, score: Score, diag):
        return Placement(self.path(pose, np.arange(self.src.length)), self.mobile, np.asarray(pose, float),
                         score.offsets, float(score.cost), score.seed, diag)

    def refine(self, placement: Placement):
        """Nelder-Mead refinement of ``placement`` (offsets fixed, warm started from its seed)."""
        if not self.sides or self.cfg.placement_refine_evals <= 0:
            return placement
        def fn(p):
            return self.score(p, placement.offsets, placement.seed).cost

        simplex = placement.pose + np.vstack([np.zeros(3), np.diag([0.05, 0.05, 0.1])])
        res = minimize(fn, placement.pose, method="Nelder-Mead",
                       options={"maxfev": self.cfg.placement_refine_evals, "xatol": 0.005, "fatol": 1e-3,
                                "initial_simplex": simplex})
        best = self.score(res.x, placement.offsets, placement.seed)
        if best.cost >= placement.cost:
            return placement
        diag = dict(placement.diagnostics, details=best.details, refined_from=placement.pose.tolist(),
                    refine_evals=int(res.nfev))
        return self._placement(res.x, best, diag)


def diagnostics(p: Placement) -> dict:
    """JSON-compatible summary of a placement for ``episode.extra``."""
    key = "offset" if p.mobile else "pose"
    return {"mobile": p.mobile, key: np.asarray(p.pose).tolist(), "flips": p.flips,
            "grasp_offsets": {s: {"flip": bool(o[0]), "theta_deg": o[1], "tilt_deg": o[2]} for s, o in p.offsets.items()},
            "segment_grasp_offsets": {s: [{"rows": [int(a), int(b)], "flip": bool(o[0]), "theta_deg": o[1],
                                           "tilt_deg": o[2]} for a, b, o in segs] for s, segs in p.segments.items()},
            "cost": p.cost, **p.diagnostics}
