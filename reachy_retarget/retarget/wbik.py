"""Whole-body trajectory IK over the base and arm joints of ``q`` (docs/design.md, step 4).

Each frame is solved by a few warm-started bounded damped least-squares steps
(``scipy.optimize.lsq_linear``, BVLS) on stacked, weighted residuals:

* active TCP poses (``{l,r}_arm_tip``): position error and the rotation-vector error,
* posture regularization toward the previous frame and toward a nominal posture,
* base deviation from its reference path (only when the base is a variable),
* self-collision repulsion for the worst sphere pairs closer than the clearance margin,
* Levenberg damping.

Increments are bounded by the joint limits minus a margin and by a per-step trust region.
Inactive arms, the neck (see :mod:`.gaze`) and the fingers (see :mod:`.targets`) are not
variables. With a fixed base the base columns are excluded entirely.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from ..robot import LOWER, UPPER, VELOCITY, Reachy, SelfCollision
from ..robot.reachy import BODY
from .config import RetargetConfig

ARM_COLUMNS = {"left": np.arange(3, 10), "right": np.arange(10, 17)}
BASE_COLUMNS = np.arange(3)


def _rotvec(R):
    """Rotation vector of one or more rotation matrices."""
    return Rotation.from_matrix(R).as_rotvec()


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def tcp_errors(q, targets):
    """Per-frame TCP residual norms for a batch: q (T, 22), targets {side: (T, 4, 4)}.

    Returns ``{side: (position error (T,) in m, rotation error (T,) in rad)}``.
    """
    fk = Reachy.load().fk(q)
    out = {}
    for side, X in targets.items():
        P = fk[f"{side}_tcp"]
        pos = np.linalg.norm(X[:, :3, 3] - P[:, :3, 3], axis=-1)
        rot = np.linalg.norm(_rotvec(np.swapaxes(P[:, :3, :3], 1, 2) @ X[:, :3, :3]), axis=-1)
        out[side] = (pos, rot)
    return out


def normalized_error(errors, cfg: RetargetConfig):
    """Worst residual over sides as a fraction of the K tolerances: (T,)."""
    return np.max([np.maximum(p / cfg.tcp_pos_tol, r / cfg.tcp_rot_tol) for p, r in errors.values()], axis=0)


class Clearance:
    """Self-clearance restricted to sphere pairs that involve at least one moving arm link."""

    def __init__(self, sides):
        self.sc = SelfCollision.load()
        prefixes = tuple(f"{s[0]}_" for s in sides)
        moving = np.array([link.startswith(prefixes) for link in self.sc.links])
        la, lb = self.sc.link_index[self.sc.a], self.sc.link_index[self.sc.b]
        keep = moving[la] | moving[lb]
        self.a, self.b = self.sc.a[keep], self.sc.b[keep]
        self.radii = self.sc.pair_radii[keep]

    def distances(self, q):
        """Signed distances (..., P) of the selected pairs and the base-frame sphere centres."""
        q = np.array(q, float)
        q[..., :3] = 0.0
        c = self.sc.sphere_centers(q)[0]
        return np.linalg.norm(c[..., self.a, :] - c[..., self.b, :], axis=-1) - self.radii, c

    def minimum(self, q):
        """Minimum signed distance (...,) over the selected pairs."""
        return self.distances(q)[0].min(axis=-1)

    def gradients(self, q, margin, limit):
        """(distances (k,), gradients (k, 22) w.r.t. q) of the worst pairs below ``margin``.

        At most one sphere pair per link pair and ``limit`` link pairs; the gradient uses
        point Jacobians of the two sphere centres (base columns are zero: the base moves
        the whole body rigidly).
        """
        d, c = self.distances(q)
        hits = np.flatnonzero(d < margin)
        if not len(hits):
            return np.zeros(0), np.zeros((0, 22))
        li = self.sc.link_index
        hits = hits[np.argsort(d[hits])]
        chosen, seen = [], set()
        for k in hits:
            key = (li[self.a[k]], li[self.b[k]])
            if key not in seen:
                seen.add(key)
                chosen.append(k)
            if len(chosen) == limit:
                break
        qb = np.asarray(q, float)[BODY.start:]
        jac = {}

        def point_jacobian(sphere):
            link = self.sc.links[li[sphere]]
            if link not in jac:
                jac[link] = self.sc.tree.jacobian(qb, link)
            T, J = jac[link]
            return J[:3] - _skew(c[sphere] - T[:3, 3]) @ J[3:]

        grads = np.zeros((len(chosen), 22))
        for i, k in enumerate(chosen):
            n = c[self.a[k]] - c[self.b[k]]
            n /= max(np.linalg.norm(n), 1e-9)
            grads[i, BODY.start:] = n @ (point_jacobian(self.a[k]) - point_jacobian(self.b[k]))
        return d[chosen], grads


class FrameSolver:
    """Bounded damped least-squares IK of one frame for a fixed set of variables.

    ``active``: sides whose TCP is tracked (their arm joints are variables); ``base_free``:
    whether base x, y, yaw are variables; ``nominal``: (22,) posture regularization target
    (its base entries are ignored); ``collisions``: whether to add the repulsion term.
    """

    def __init__(self, cfg: RetargetConfig, active, base_free: bool, nominal, *, collisions=True):
        self.cfg, self.active = cfg, tuple(active)
        self.robot = Reachy.load()
        cols = ([BASE_COLUMNS] if base_free else []) + [ARM_COLUMNS[s] for s in self.active]
        self.free = np.concatenate(cols) if cols else np.zeros(0, int)
        self.base_free = base_free
        m = cfg.joint_limit_margin
        self.lower, self.upper = LOWER[self.free] + m, UPPER[self.free] - m
        self.nominal = np.asarray(nominal, float)
        self.is_arm = self.free >= 3
        self.clearance = Clearance(self.active) if collisions and self.active else None

    def _system(self, q, targets, q_prev, base_ref):
        cfg, f = self.cfg, self.free
        rows, rhs, worst = [], [], 0.0
        for side in self.active:
            T, J = self.robot.jacobian(q, f"{side}_tcp")
            X = targets[side]
            ep = X[:3, 3] - T[:3, 3]
            er = _rotvec(X[:3, :3] @ T[:3, :3].T)
            worst = max(worst, np.linalg.norm(ep) / cfg.tcp_pos_tol, np.linalg.norm(er) / cfg.tcp_rot_tol)
            rows += [cfg.w_pos * J[:3, f], cfg.w_rot * J[3:, f]]
            rhs += [cfg.w_pos * ep, cfg.w_rot * er]
        eye = np.eye(len(f))
        rows.append(cfg.w_prev * eye)
        rhs.append(cfg.w_prev * (q_prev[f] - q[f]))
        rows.append(cfg.w_nominal * eye[self.is_arm])
        rhs.append(cfg.w_nominal * (self.nominal[f] - q[f])[self.is_arm])
        if self.base_free:
            rows.append(cfg.w_base * eye[~self.is_arm])
            rhs.append(cfg.w_base * (base_ref - q[:3]))
        collided = False
        if self.clearance is not None:
            d, g = self.clearance.gradients(q, cfg.self_clearance_margin, cfg.collision_pairs)
            if len(d):
                collided = True
                rows.append(cfg.w_collision * g[:, f])
                rhs.append(cfg.w_collision * (cfg.self_clearance_margin - d))
        rows.append(cfg.w_damping * eye)
        rhs.append(np.zeros(len(f)))
        return np.vstack(rows), np.concatenate(rhs), worst, collided

    def solve(self, q, targets, q_prev, base_ref, max_iter, box=None):
        """Refine ``q`` (22,) toward ``targets`` {side: 4x4 TCP}; returns (q, iterations used).

        ``q_prev`` is the posture-regularization reference, ``base_ref`` (3,) the base target
        (also the fixed base pose when the base is not a variable); ``box`` = optional
        absolute (lower (22,), upper (22,)) bounds intersected with the joint limits.
        """
        q = np.array(q, float)
        if not self.base_free:
            q[:3] = base_ref
        if not len(self.free):
            return q, 0
        f = self.free
        lower, upper = self.lower, self.upper
        if box is not None:
            lower, upper = np.maximum(lower, box[0][f]), np.minimum(upper, box[1][f])
            upper = np.maximum(upper, lower)
        q[f] = np.clip(q[f], lower, upper)
        step = self.cfg.ik_step
        for it in range(max_iter):
            A, b, worst, collided = self._system(q, targets, q_prev, base_ref)
            if worst < 0.05 and not collided and it:
                return q, it
            lo = np.maximum(lower - q[f], -step)
            hi = np.minimum(upper - q[f], step)
            dq = lsq_linear(A, b, bounds=(lo, np.maximum(hi, lo)), method="bvls", tol=1e-8, max_iter=50).x
            q[f] += dq
            if np.max(np.abs(dq)) < 1e-7:
                return q, it + 1
        return q, max_iter

    def error(self, q, targets):
        """Worst normalized TCP residual of one configuration."""
        return float(normalized_error(tcp_errors(q[None], {s: X[None] for s, X in targets.items()}), self.cfg)[0])

    def cold_solve(self, q, targets, base_ref, max_iter):
        """Multi-start solve for frames without a warm start: ``q`` with its active arms set to
        the nominal posture and to ``cfg.ik_seeds - 1`` deterministic pseudo-random postures,
        ``cfg.ik_seed_iter`` iterations each; the best is then solved to ``max_iter``."""
        cfg, rng = self.cfg, np.random.default_rng(0)
        cols = self.free[self.is_arm]
        best = None
        for k in range(cfg.ik_seeds):
            seed = np.array(q, float)
            seed[cols] = self.nominal[cols] if k == 0 else rng.uniform(-1.2, 1.2, len(cols))
            seed = self.solve(seed, targets, seed, base_ref, cfg.ik_seed_iter)[0]
            e = self.error(seed, targets)
            if best is None or e < best[0]:
                best = (e, seed)
        return self.solve(best[1], targets, best[1], base_ref, max_iter)

    def manipulability(self, q):
        """Smallest Yoshikawa measure sqrt(det(J J^T)) of the active arms' 6x7 TCP Jacobians."""
        vals = [np.sqrt(max(np.linalg.det(J @ J.T), 0.0))
                for J in (self.robot.jacobian(q, f"{s}_tcp")[1][:, ARM_COLUMNS[s]] for s in self.active)]
        return min(vals) if vals else 0.0


def solve_trajectory(cfg: RetargetConfig, targets, base_ref, base_free, q0, nominal):
    """Frame-by-frame IK along the source clock.

    targets: {side: (T, 4, 4) world TCP}; base_ref: (T, 3) base path (the fixed base when
    ``base_free`` is False); q0: (22,) seed of the first frame (its non-variable entries are
    kept for every frame). Returns (q (T, 22), iterations per frame (T,)).
    """
    solver = FrameSolver(cfg, tuple(targets), base_free, nominal)
    T = len(base_ref)
    q = np.empty((T, 22))
    iters = np.zeros(T, int)
    cur = np.array(q0, float)
    cur[:3] = base_ref[0]
    for t in range(T):
        frame = {s: X[t] for s, X in targets.items()}
        limit = cfg.ik_first_max_iter if t == 0 else cfg.ik_max_iter
        cur, iters[t] = solver.solve(cur, frame, cur, base_ref[t], limit)
        q[t] = cur
    return q, iters


def smooth(cfg: RetargetConfig, q, targets, base_free):
    """Light Gaussian smoothing of the IK variables, kept per frame only where it does not
    push the TCP residual above max(previous residual, cfg.refine_fraction of the tolerance)
    nor lower the self-clearance below min(previous clearance, the IK margin).

    Returns (smoothed q, fraction of frames that kept the smoothed value).
    """
    if not targets or cfg.smoothing_sigma <= 0 or len(q) < 3:
        return q, 0.0
    cols = np.concatenate(([BASE_COLUMNS] if base_free else []) + [ARM_COLUMNS[s] for s in targets])
    qs = q.copy()
    qs[:, cols] = gaussian_filter1d(q[:, cols], cfg.smoothing_sigma, axis=0, mode="nearest")
    e_old, e_new = normalized_error(tcp_errors(q, targets), cfg), normalized_error(tcp_errors(qs, targets), cfg)
    clear = Clearance(tuple(targets))
    c_old, c_new = clear.minimum(q), clear.minimum(qs)
    keep = (e_new <= np.maximum(e_old, cfg.refine_fraction)) & (c_new >= np.minimum(c_old, cfg.self_clearance_margin))
    return np.where(keep[:, None], qs, q), float(keep.mean())


def refine(cfg: RetargetConfig, q, targets, base_ref, base_free, nominal, dt):
    """Re-solve output frames whose residual exceeds ``cfg.refine_fraction`` of the tolerance.

    Frames are processed in order, warm-started and regularized at their current value, and
    boxed so that the change to each neighbour stays within ``VELOCITY * dt``: refinement
    never breaks the speed limits. Returns (q, refined frame count).
    """
    if not targets:
        return q, 0
    bad = np.flatnonzero(normalized_error(tcp_errors(q, targets), cfg) > cfg.refine_fraction)
    if not len(bad):
        return q, 0
    solver = FrameSolver(cfg, tuple(targets), base_free, nominal)
    step = VELOCITY * dt * (1 - 1e-6)
    q = q.copy()
    for t in bad:
        nb = q[[i for i in (t - 1, t + 1) if 0 <= i < len(q)]]
        box = (nb.max(axis=0) - step, nb.min(axis=0) + step)
        q[t] = solver.solve(q[t], {s: X[t] for s, X in targets.items()}, q[t], base_ref[t], cfg.ik_max_iter, box)[0]
    return q, len(bad)
