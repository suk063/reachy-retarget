"""Whole-body trajectory IK over the base and arm joints of ``q`` (docs/design.md, step 4).

Each frame is solved by a few warm-started bounded damped least-squares steps
(:func:`box_lsq`, an active-set solver of the box-bounded normal equations; the
Levenberg damping rows keep them positive definite) on stacked, weighted residuals:

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
from scipy.spatial.transform import Rotation

from ..robot import LOWER, UPPER, VELOCITY, Reachy, SelfCollision
from ..robot.reachy import BODY
from . import footprint
from .config import RetargetConfig

ARM_COLUMNS = {"left": np.arange(3, 10), "right": np.arange(10, 17)}
BASE_COLUMNS = np.arange(3)


def _rotvec(R):
    """Rotation vector of one or more rotation matrices."""
    R = np.asarray(R, float)
    if R.shape == (3, 3):  # fast path for single frames (IK inner loop); scipy near pi
        c = (R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0
        if c > -0.99:
            angle = np.arccos(min(1.0, c))
            v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
            return v * (0.5 if angle < 1e-8 else angle / (2.0 * np.sin(angle)))
    return Rotation.from_matrix(R).as_rotvec()


def box_lsq(A, b, lo, hi, max_iter=None):
    """argmin |A x - b| subject to lo <= x <= hi, for full column rank A (small problems).

    Active-set method on the normal equations: solve with the free variables, clamp the worst
    violator to its bound, repeat; then release clamped variables whose gradient points into
    the box (KKT check). Exact at convergence; ``max_iter`` defaults to 3 n.
    """
    H, g = A.T @ A, A.T @ b
    n = len(g)
    x = np.clip(np.zeros(n), lo, hi)
    fixed = np.zeros(n, bool)
    for _ in range(max_iter or 3 * n):
        free = ~fixed
        y = x.copy()
        if free.any():
            rhs = g[free] - H[np.ix_(free, fixed)] @ x[fixed]
            y[free] = np.linalg.solve(H[np.ix_(free, free)], rhs)
        viol = free & ((y < lo - 1e-12) | (y > hi + 1e-12))
        if viol.any():
            # step from x toward y until the first bound is hit, then fix that variable
            d = y - x
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(d < 0, (lo - x) / d, np.where(d > 0, (hi - x) / d, np.inf))
            t = np.where(viol, t, np.inf)
            k = int(np.argmin(t))
            x = x + max(0.0, min(1.0, float(t[k]))) * d
            x[k] = lo[k] if d[k] < 0 else hi[k]
            fixed[k] = True
            x = np.clip(x, lo, hi)
            continue
        x = y
        grad = H @ x - g  # KKT: at lower bound grad >= 0, at upper bound grad <= 0
        release = fixed & (((x <= lo + 1e-12) & (grad < -1e-12)) | ((x >= hi - 1e-12) & (grad > 1e-12)))
        if not release.any():
            break
        fixed[int(np.argmax(np.where(release, np.abs(grad), -1)))] = False
    return x


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

    def minimum(self, q, chunk: int = 64):
        """Minimum signed distance (...,) over the selected pairs, ``chunk`` configurations at a
        time (see :meth:`SelfCollision.min_clearance`)."""
        q = np.asarray(q, float)
        flat = q.reshape(-1, q.shape[-1])
        out = np.empty(len(flat))
        for i in range(0, len(flat), chunk):
            out[i:i + chunk] = self.distances(flat[i:i + chunk])[0].min(axis=-1)
        return out[0] if q.ndim == 1 else out.reshape(q.shape[:-1])

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

    def __init__(self, cfg: RetargetConfig, active, base_free: bool, nominal, *, collisions=True, obstacles=None):
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
        self.obstacles = obstacles if base_free and obstacles is not None and (
            obstacles.polygons or len(obstacles.points)) else None

    def _system(self, q, targets, q_prev, base_ref, rot_scale=None):
        cfg, f = self.cfg, self.free
        rows, rhs, worst = [], [], 0.0
        for side in self.active:
            T, J = self.robot.jacobian(q, f"{side}_tcp")
            X = targets[side]
            ep = X[:3, 3] - T[:3, 3]
            er = _rotvec(X[:3, :3] @ T[:3, :3].T)
            w = 1.0 if rot_scale is None else float(rot_scale.get(side, 1.0))
            worst = max(worst, np.linalg.norm(ep) / cfg.tcp_pos_tol, w * np.linalg.norm(er) / cfg.tcp_rot_tol)
            rows += [cfg.w_pos * J[:3, f], w * cfg.w_rot * J[3:, f]]
            rhs += [cfg.w_pos * ep, w * cfg.w_rot * er]
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

    def solve(self, q, targets, q_prev, base_ref, max_iter, box=None, rot_scale=None):
        """Refine ``q`` (22,) toward ``targets`` {side: 4x4 TCP}; returns (q, iterations used).

        ``q_prev`` is the posture-regularization reference, ``base_ref`` (3,) the base target
        (also the fixed base pose when the base is not a variable); ``box`` = optional
        absolute (lower (22,), upper (22,)) bounds intersected with the joint limits;
        ``rot_scale`` = optional {side: factor} on the orientation weight.
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
            A, b, worst, collided = self._system(q, targets, q_prev, base_ref, rot_scale)
            if worst < 0.05 and not collided and it and (
                    self.obstacles is None or self._footprint(q[:2])[0] >= self.cfg.footprint_margin):
                return q, it
            lo = np.maximum(lower - q[f], -step)
            hi = np.minimum(upper - q[f], step)
            dq = box_lsq(A, b, lo, np.maximum(hi, lo))
            if self.obstacles is not None:
                # Base footprint clearance >= cfg.footprint_margin as linearized constraints: when
                # the step would end below it, re-solve with stiff constraint rows linearized at
                # the current and at the violating end point (a penalty on the current clearance
                # alone chatters across the boundary). Up to three rounds.
                extra_A, extra_b = [], []
                for xy in (q[:2], None, None):
                    end = q[:2] + dq[:2]
                    if xy is None:
                        if footprint.clearance(end[None], self.obstacles)[0] >= self.cfg.footprint_margin:
                            break
                        xy = end
                    c, g = self._footprint(xy)
                    if xy is q[:2] and c + g @ dq[:2] >= self.cfg.footprint_margin:
                        break
                    row = np.zeros(len(f))
                    row[:2] = g
                    w = self.cfg.w_footprint
                    extra_A.append(w * row)
                    extra_b.append(w * (self.cfg.footprint_margin - c + g @ (xy - q[:2])))
                    dq = box_lsq(np.vstack([A] + [r[None] for r in extra_A]), np.r_[b, extra_b], lo,
                                 np.maximum(hi, lo))
            q[f] += dq
            if np.max(np.abs(dq)) < 1e-7:
                return q, it + 1
        return q, max_iter

    def _footprint(self, xy, h=1e-4):
        """(clearance, gradient (2,)) of the base footprint at base position ``xy``."""
        c = footprint.clearance(np.array([xy, xy + (h, 0.0), xy + (0.0, h)]), self.obstacles)
        return float(c[0]), (c[1:] - c[0]) / h

    def error(self, q, targets, rot_scale=None):
        """Worst normalized TCP residual of one configuration (rotation scaled by ``rot_scale``)."""
        errs = tcp_errors(q[None], {s: X[None] for s, X in targets.items()})
        if rot_scale:
            errs = {s: (p, r * rot_scale.get(s, 1.0)) for s, (p, r) in errs.items()}
        return float(normalized_error(errs, self.cfg)[0])

    def cold_solve(self, q, targets, base_ref, max_iter, rot_scale=None):
        """Multi-start solve for frames without a warm start: ``q`` with its active arms set to
        the nominal posture and to ``cfg.ik_seeds - 1`` deterministic pseudo-random postures,
        ``cfg.ik_seed_iter`` iterations each; the best is then solved to ``max_iter``."""
        cfg, rng = self.cfg, np.random.default_rng(0)
        cols = self.free[self.is_arm]
        best = None
        for k in range(cfg.ik_seeds):
            seed = np.array(q, float)
            seed[cols] = self.nominal[cols] if k == 0 else rng.uniform(-1.2, 1.2, len(cols))
            seed = self.solve(seed, targets, seed, base_ref, cfg.ik_seed_iter, rot_scale=rot_scale)[0]
            e = self.error(seed, targets, rot_scale)
            if best is None or e < best[0]:
                best = (e, seed)
        return self.solve(best[1], targets, best[1], base_ref, max_iter, rot_scale=rot_scale)

    def manipulability(self, q):
        """Smallest Yoshikawa measure sqrt(det(J J^T)) of the active arms' 6x7 TCP Jacobians."""
        vals = [np.sqrt(max(np.linalg.det(J @ J.T), 0.0))
                for J in (self.robot.jacobian(q, f"{s}_tcp")[1][:, ARM_COLUMNS[s]] for s in self.active)]
        return min(vals) if vals else 0.0


def solve_trajectory(cfg: RetargetConfig, targets, base_ref, base_free, q0, nominal, rot_scale=None,
                     max_iter=None, box=None, obstacles=None):
    """Frame-by-frame IK along the source clock.

    targets: {side: (T, 4, 4) world TCP}; base_ref: (T, 3) base path (the fixed base when
    ``base_free`` is False); q0: (22,) seed of the first frame (its non-variable entries are
    kept for every frame); rot_scale: optional {side: (T,)} orientation weight factors;
    max_iter: iterations per frame after the first (default ``cfg.ik_max_iter``); box: optional
    absolute (lower (22,), upper (22,)) bounds for every frame; obstacles: footprint obstacles
    repelling a free base (``cfg.footprint_margin``).
    Returns (q (T, 22), iterations per frame (T,)).
    """
    solver = FrameSolver(cfg, tuple(targets), base_free, nominal, obstacles=obstacles)
    T = len(base_ref)
    q = np.empty((T, 22))
    iters = np.zeros(T, int)
    cur = np.array(q0, float)
    cur[:3] = base_ref[0]
    for t in range(T):
        frame = {s: X[t] for s, X in targets.items()}
        limit = cfg.ik_first_max_iter if t == 0 else (max_iter or cfg.ik_max_iter)
        scale = None if rot_scale is None else {s: w[t] for s, w in rot_scale.items()}
        cur, iters[t] = solver.solve(cur, frame, cur, base_ref[t], limit, box=box, rot_scale=scale)
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


def refine(cfg: RetargetConfig, q, targets, base_ref, base_free, nominal, dt, obstacles=None):
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
    solver = FrameSolver(cfg, tuple(targets), base_free, nominal, obstacles=obstacles)
    step = VELOCITY * dt * (1 - 1e-6)
    q = q.copy()
    for t in bad:
        nb = q[[i for i in (t - 1, t + 1) if 0 <= i < len(q)]]
        box = (nb.max(axis=0) - step, nb.min(axis=0) + step)
        q[t] = solver.solve(q[t], {s: X[t] for s, X in targets.items()}, q[t], base_ref[t], cfg.ik_max_iter, box)[0]
    return q, len(bad)
