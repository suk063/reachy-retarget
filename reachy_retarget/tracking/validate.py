"""Tier K of tracking episodes: the manipulation tier K plus the head.

:func:`reachy_retarget.validate.kinematic.check` (TCP residuals on every frame and side, joint and
speed limits -- fingers and neck included --, self-clearance, the navigation destination) with the
tracking tolerances, plus the head rotation residual against ``reference.head``, which no other
check covers. The base deviation from the nominal path is reported, not gated (the base is free).
"""
from __future__ import annotations

import numpy as np

from ..schema.rotations import so3_log
from ..validate import kinematic
from .config import TrackingConfig


def check(ep, cfg: TrackingConfig | None = None) -> dict:
    cfg = cfg or TrackingConfig()
    k = kinematic.check(ep, cfg.ik())
    reasons, metrics, frames = list(k["reasons"]), dict(k["metrics"]), dict(k["frames"])
    if ep.reference.base is not None:
        yaw = np.abs(np.angle(np.exp(1j * (ep.q[:, 2] - ep.reference.base[:, 2]))))
        metrics["max_base_yaw_deviation_rad"] = float(yaw.max())
    if ep.reference.head is not None:
        R, X = ep.head_world[:, :3, :3], ep.reference.head[:, :3, :3]
        err = np.linalg.norm(so3_log(np.swapaxes(R, 1, 2) @ X), axis=1)
        frames["head_rot_residual"] = err
        metrics["head_max_rot_residual"] = float(err.max())
        metrics["head_median_rot_residual"] = float(np.median(err))
        bad = err > cfg.head_rot_tol
        if bad.any():
            j = int(np.argmax(err))
            reasons.append(f"head rotation residual {err[j]:.3g} rad > {cfg.head_rot_tol:.3g} rad at frame {j} "
                           f"({int(bad.sum())} frames)")
    return {"passed": not reasons, "reasons": reasons, "metrics": metrics, "frames": frames}
