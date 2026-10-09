"""Tracking configuration: tolerances, IK weights and timing in one frozen, hashed record.

The digest is stored in each tracking episode as ``retarget_config`` and the values in
``extra["tracking_config"]``. The whole-body IK and tier K of :mod:`reachy_retarget.retarget` and
:mod:`reachy_retarget.validate.kinematic` read a :class:`RetargetConfig`; :meth:`TrackingConfig.ik`
builds one with the tracking values.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace

from ..retarget.config import RetargetConfig
from ..schema.episode import HZ


@dataclass(frozen=True)
class TrackingConfig:
    """Units: metres, radians, seconds."""

    # Tier K. TCP tolerances are those of the manipulation retargeting; the head is checked too.
    tcp_pos_tol: float = 0.005
    tcp_rot_tol: float = 0.05
    head_rot_tol: float = 0.05           # head rotation residual against reference.head
    # Self-clearance gate: reachy-control keeps every sphere pair >= 10 mm (its CBF ``MARGIN``), so
    # tracking episodes are held to the same 10 mm (the manipulation retargeting gates at 9 mm).
    min_self_clearance: float = 0.010

    # Whole-body IK (reachy_retarget.retarget.wbik.FrameSolver), strict on every frame and side.
    joint_limit_margin: float = 0.045
    self_clearance_margin: float = 0.02  # repulsion is active below this clearance
    ik_max_iter: int = 10
    ik_first_max_iter: int = 80
    w_pos: float = 100.0
    w_rot: float = 3.0
    w_prev: float = 0.05
    w_nominal: float = 0.01
    w_base: float = 1.0                  # base deviation from the nominal base path
    base_box_xy: float = 0.5             # m, the base stays this close to its nominal path (each axis)
    base_box_yaw: float = 0.8            # rad
    smoothing_sigma: float = 1.0         # frames
    refine_fraction: float = 0.5

    # Neck (tracking.neck).
    neck_margin: float = 0.03            # rad inside the URDF limits
    neck_iterations: int = 25

    # Timing: synthetic references are generated within the speed limits and are not retimed;
    # source references are slowed down where needed (reachy_retarget.retarget.timing).
    velocity_scale: float = 0.95
    rate_hz: float = float(HZ)

    def __post_init__(self):
        if self.rate_hz != HZ:
            raise ValueError(f"the episode schema fixes the output rate at {HZ} Hz")

    def ik(self) -> RetargetConfig:
        """The :class:`RetargetConfig` used by the whole-body IK, timing and tier K."""
        return replace(RetargetConfig(), tcp_pos_tol=self.tcp_pos_tol, tcp_rot_tol=self.tcp_rot_tol,
                       min_self_clearance=self.min_self_clearance, joint_limit_margin=self.joint_limit_margin,
                       self_clearance_margin=self.self_clearance_margin, ik_max_iter=self.ik_max_iter,
                       ik_first_max_iter=self.ik_first_max_iter, w_pos=self.w_pos, w_rot=self.w_rot,
                       w_prev=self.w_prev, w_nominal=self.w_nominal, w_base=self.w_base,
                       smoothing_sigma=self.smoothing_sigma, refine_fraction=self.refine_fraction,
                       velocity_scale=self.velocity_scale, gaze_margin=self.neck_margin)

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))

    def digest(self) -> str:
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(("tracking:" + text).encode()).hexdigest()
