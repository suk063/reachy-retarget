"""Tier K: kinematic validation of a retargeted episode (docs/design.md, "Validation tiers").

Checks, all on the stored canonical state:

* TCP residual against ``reference.tcp`` on every frame of every tracked side
  (``cfg.tcp_pos_tol`` / ``cfg.tcp_rot_tol``; the stricter ``grasp_*`` tolerances while the
  hand holds an object),
* joint limits (URDF, no margin) and joint speed limits (``robot.VELOCITY``) from ``q`` and
  ``time``,
* self-clearance of the collision-sphere model >= ``cfg.min_self_clearance``,
* the base footprint disc clear of static support/fixture box geometry,
* grasp-phase consistency: while a hand holds an object, the object pose expressed in the
  Reachy grasp-center frame stays constant within ``cfg.grasp_rel_*_tol``, i.e. the
  retargeted hand motion reproduces the source object motion.

Grasp phases come from ``validation["grasp_object"]`` (T, 2): per side, an index into
``extra["grasp_object_ids"]`` or -1, written by the retargeting pipeline from the source.
"""
from __future__ import annotations

import numpy as np

from ..retarget import footprint
from ..retarget.config import RetargetConfig
from ..robot import LOWER, UPPER, VELOCITY, Reachy, min_clearance
from ..schema.episode import SIDES
from ..schema.rotations import se3_inv, so3_log, vec7_to_pose, wrap_angle


def _segments(mask):
    """(start, stop) pairs of the True runs of a boolean array."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def frame_metrics(ep) -> dict[str, np.ndarray]:
    """Per-frame diagnostics stored under ``/validation``.

    ``tcp_pos_residual`` and ``tcp_rot_residual`` (T, 2) are NaN for untracked sides;
    ``self_clearance`` (T,) in m; ``joint_limit_margin`` (T,) = smallest distance of an arm,
    neck or finger joint to its URDF limit (negative = outside).
    """
    T = ep.length
    pos, rot = np.full((T, 2), np.nan), np.full((T, 2), np.nan)
    world = ep.tcp_world
    for i, side in enumerate(SIDES):
        if side in ep.reference.tcp:
            X, P = ep.reference.tcp[side], world[side]
            pos[:, i] = np.linalg.norm(X[:, :3, 3] - P[:, :3, 3], axis=1)
            rot[:, i] = np.linalg.norm(so3_log(np.swapaxes(P[:, :3, :3], 1, 2) @ X[:, :3, :3]), axis=1)
    body = slice(3, 22)
    margin = np.minimum(ep.q[:, body] - LOWER[body], UPPER[body] - ep.q[:, body]).min(axis=1)
    return {"tcp_pos_residual": pos, "tcp_rot_residual": rot, "self_clearance": min_clearance(ep.q),
            "joint_limit_margin": margin}


def _grasp_drift(ep, labels, ids):
    """Worst (position, rotation) drift of held objects in the Reachy grasp-center frames,
    and per-side reasons."""
    robot = Reachy.load()
    world = ep.tcp_world
    worst_p, worst_r, events = 0.0, 0.0, []
    for i, side in enumerate(SIDES):
        G = world[side] @ robot.grasp_center(side)
        for k in np.unique(labels[:, i][labels[:, i] >= 0]):
            obj = ep.objects[ids[k]]
            for a, b in _segments((labels[:, i] == k) & obj.valid):
                rel = se3_inv(G[a:b]) @ vec7_to_pose(obj.pose[a:b])
                dp = np.linalg.norm(rel[:, :3, 3] - rel[0, :3, 3], axis=1).max()
                dr = np.linalg.norm(so3_log(np.swapaxes(rel[0, :3, :3], 0, 1) @ rel[:, :3, :3]), axis=1).max()
                worst_p, worst_r = max(worst_p, dp), max(worst_r, dr)
                events.append((side, ids[k], int(a), int(b), float(dp), float(dr)))
    return worst_p, worst_r, events


def check(ep, cfg: RetargetConfig | None = None) -> dict:
    """Run tier K (default configuration if ``cfg`` is None).

    Returns ``{"passed", "reasons", "metrics", "frames"}``; ``frames`` holds the per-frame
    arrays of :func:`frame_metrics`."""
    cfg = cfg or RetargetConfig()
    frames = frame_metrics(ep)
    reasons, metrics = [], {}
    labels = np.asarray(ep.validation.get("grasp_object", np.full((ep.length, 2), -1)))
    ids = list(ep.extra.get("grasp_object_ids", []))
    holding = labels >= 0
    for i, side in enumerate(SIDES):
        if side not in ep.reference.tcp:
            continue
        pos, rot = frames["tcp_pos_residual"][:, i], frames["tcp_rot_residual"][:, i]
        tol_p = np.where(holding[:, i], cfg.grasp_tcp_pos_tol, cfg.tcp_pos_tol)
        tol_r = np.where(holding[:, i], cfg.grasp_tcp_rot_tol, cfg.tcp_rot_tol)
        metrics[f"{side}_max_pos_residual"] = float(pos.max())
        metrics[f"{side}_max_rot_residual"] = float(rot.max())
        for name, err, tol, unit, scale in (("position", pos, tol_p, "mm", 1e3), ("rotation", rot, tol_r, "rad", 1)):
            bad = err > tol
            if bad.any():
                j = int(np.argmax(err - tol))
                reasons.append(f"{side} TCP {name} residual {err[j] * scale:.3g} {unit} > {tol[j] * scale:.3g} {unit} "
                               f"at frame {j} ({int(bad.sum())} frames)")
    q = ep.q
    margin = frames["joint_limit_margin"]
    metrics["min_joint_limit_margin"] = float(margin.min())
    if margin.min() < -1e-9:
        reasons.append(f"joint limit exceeded by {-margin.min():.4g} rad at frame {int(np.argmin(margin))}")
    dq = np.diff(q, axis=0)
    dq[:, 2] = wrap_angle(dq[:, 2])
    ratio = np.abs(dq) / np.diff(ep.time)[:, None] / VELOCITY
    metrics["max_speed_ratio"] = float(ratio.max())
    if ratio.max() > 1 + 1e-6:
        t, j = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
        reasons.append(f"speed limit exceeded: joint {j} at {ratio[t, j]:.3g} x limit (frame {t})")
    clear = frames["self_clearance"]
    metrics["min_self_clearance"] = float(clear.min())
    if clear.min() < cfg.min_self_clearance:
        reasons.append(f"self-clearance {clear.min() * 1e3:.3g} mm < {cfg.min_self_clearance * 1e3:.3g} mm "
                       f"at frame {int(np.argmin(clear))}")
    obs = footprint.obstacles(ep.objects, cfg, static_only=True)
    if obs.polygons:
        fp = footprint.clearance(q[:, :2], obs)
        metrics["min_footprint_clearance"] = float(fp.min())
        if fp.min() < 0:
            reasons.append(f"base footprint overlaps static scene geometry by {-fp.min() * 1e3:.3g} mm "
                           f"at frame {int(np.argmin(fp))}")
    metrics["footprint_notes"] = obs.notes
    dp, dr, events = _grasp_drift(ep, labels, ids)
    metrics.update(grasp_phases=len(events), max_grasp_drift_pos=dp, max_grasp_drift_rot=dr)
    for side, obj, a, b, p, r in events:
        if p > cfg.grasp_rel_pos_tol or r > cfg.grasp_rel_rot_tol:
            reasons.append(f"{side} hand-{obj} relative pose drifts {p * 1e3:.3g} mm / {r:.3g} rad "
                           f"during frames {a}-{b}")
    return {"passed": not reasons, "reasons": reasons, "metrics": metrics, "frames": frames}
