"""Retargeting configuration: every tolerance, weight and search parameter in one frozen record.

The configuration is hashed (:meth:`RetargetConfig.digest`) and the digest is stored in each
episode as ``retarget_config``; the full values go to ``extra["retarget_config"]``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from ..schema.episode import HZ


@dataclass(frozen=True)
class RetargetConfig:
    """Parameters of :func:`reachy_retarget.retarget.retarget` and of tier-K validation.

    Units: metres, radians, seconds. IK weights multiply residuals of the stated unit
    (per metre, per radian), so a weight ratio is a unit exchange rate.
    """

    # Tier K (docs/design.md, "Validation tiers").
    tcp_pos_tol: float = 0.005           # TCP position residual on active effectors
    tcp_rot_tol: float = 0.05            # TCP rotation residual (angle of the rotation error)
    grasp_tcp_pos_tol: float = 0.003     # stricter residuals while a hand holds an object
    grasp_tcp_rot_tol: float = 0.03
    grasp_rel_pos_tol: float = 0.01      # drift of the object in the Reachy grasp-center frame
    grasp_rel_rot_tol: float = 0.1
    min_self_clearance: float = 0.009    # sphere-model signed distance
    footprint_margin: float = 0.05       # added to BASE_FOOTPRINT_RADIUS around scene geometry

    # Gripper and grasp labelling.
    closed_opening: float = 0.5          # source opening below this = commanded closed
    closed_drop: float = 0.1             # ... or this far below the episode's open level
    opening_rate: float = 0.25           # unless opening faster than this (1/s), see targets.source_closed
    grasp_contact_distance: float = 0.04  # grasp center to object surface (box) or centre
    squeeze_angle: float = 0.05          # finger angle below object contact while grasping (0.4 Nm at kp 8)
    floor_support_height: float = 0.05   # support boxes whose top is lower are the floor

    # Grasp re-selection (targets.py): offsets of the source grasp-center frame.
    grasp_tilts_deg: tuple[float, ...] = (0.0, -15.0, 15.0, -30.0, 30.0, -45.0, 45.0)  # about the closing axis
    grasp_tilt_cost: float = 0.1         # placement cost per 30 deg of tilt
    symmetry_axis_tol_deg: float = 15.0  # approach vs object principal axis for quarter turns
    symmetry_extent_tol: float = 0.1     # relative difference of the cross-section half extents
    align_max_deg: float = 20.0          # largest closing-axis to face-normal alignment applied
    align_agree_deg: float = 5.0         # grasped objects of one hand must agree on it within this
    finger_depth_slack: float = 0.003    # offsets whose finger path sinks deeper into boxes than the best + this are dropped

    # Object-centric grasps (targets.object_centric): the hand carries the object rigidly.
    object_centric: bool = True
    grasp_settle_frames: int = 2         # source frames into a grasp where the relative pose is taken
    object_centric_fraction: float = 0.5  # only segments whose source in-hand drift exceeds this x grasp_rel_*_tol

    # Orientation outside grasps (targets.orientation_weight): the hand orientation is strict
    # around grasps and free (IK-preferred, then tracked strictly) far from them.
    approach_window_s: float = 0.5
    retreat_window_s: float = 0.3
    orientation_blend_s: float = 1.0
    free_rot_weight: float = 0.05

    # Whole-body IK (bounded damped least squares, warm started).
    joint_limit_margin: float = 0.045    # physics tracking dips ~0.015 rad below it; the tier-P gate needs 0.025
    self_clearance_margin: float = 0.02  # repulsion is active below this clearance
    ik_max_iter: int = 8                 # per trajectory frame
    ik_first_max_iter: int = 80          # first frame, after the multi-start
    ik_free_iter: int = 4                # per frame in the free-orientation pass (pipeline)
    ik_seeds: int = 12                   # multi-start postures of a cold start
    ik_seed_iter: int = 15               # iterations per multi-start posture
    ik_step: float = 0.2                 # trust region per iteration (rad, m)
    w_pos: float = 100.0                 # per metre of TCP position error
    w_rot: float = 3.0                   # per radian of TCP rotation error
    w_prev: float = 0.05                 # posture regularization toward the previous frame
    w_nominal: float = 0.01              # posture regularization toward the nominal posture
    w_damping: float = 0.02              # Levenberg damping of each step
    w_base: float = 1.0                  # base deviation from its reference (mobile sources)
    w_collision: float = 300.0            # per metre below self_clearance_margin
    w_footprint: float = 1e4             # stiffness of the free-base footprint constraint row (wbik)
    collision_pairs: int = 4             # at most this many offending link pairs per step
    smoothing_sigma: float = 1.0         # Gaussian smoothing of q, in source frames
    refine_fraction: float = 0.5         # re-solve output frames above this fraction of tol

    # Base assistance for fixed-base sources (pipeline): when the fixed placement leaves frames
    # outside the tier-K tolerances, the base may move within a box around the placement.
    base_assist: bool = True
    w_base_assist: float = 1.0           # per metre / radian of base deviation from the placement
    base_assist_range: float = 0.6       # m, each axis
    base_assist_yaw: float = 1.0         # rad

    # Base placement.
    placement_keyframes: int = 10
    placement_radii: tuple[float, ...] = (0.0, 0.15, 0.3, 0.4, 0.5, 0.6)  # shoulder-to-centroid, m
    placement_yaws: int = 16             # headings around the target centroid
    placement_hint_offsets: tuple[float, ...] = (-0.2, 0.0, 0.2)  # x/y around base_hint, m
    placement_mobile_offsets: tuple[float, ...] = (-0.3, -0.15, 0.0, 0.15)  # body x/y, m
    placement_candidates: int = 5        # best proxy candidates evaluated with IK
    placement_refine_evals: int = 30     # Nelder-Mead function evaluations
    placement_iter: int = 15             # IK iterations per keyframe while scoring
    placement_quick_keyframes: int = 3   # keyframes of the grasp-offset pre-screen
    placement_quick_iter: int = 10       # IK iterations per keyframe in the pre-screen
    placement_offsets_full: int = 2      # grasp offsets scored on all keyframes
    placement_accept: float = 0.6        # first offset with keyframe residuals below this fraction of tol wins

    # Gaze.
    gaze_smoothing_s: float = 0.15
    gaze_margin: float = 0.03            # neck limit margin

    # Timing.
    velocity_scale: float = 0.95         # fraction of robot.VELOCITY used for time scaling
    dilation_window_s: float = 0.1       # max-filter half window of the time dilation
    grasp_dwell_s: float = 0.3           # minimum duration of the source step into / out of a grasp
    rate_hz: float = float(HZ)

    def __post_init__(self):
        if self.rate_hz != HZ:
            raise ValueError(f"the episode schema fixes the output rate at {HZ} Hz")
        if not 0 < self.velocity_scale <= 1:
            raise ValueError("velocity_scale must be in (0, 1]")

    def to_dict(self) -> dict:
        """Plain JSON-compatible values (tuples become lists)."""
        return json.loads(json.dumps(asdict(self)))

    def digest(self) -> str:
        """``sha256:<hex>`` of the canonical JSON of all values."""
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(text.encode()).hexdigest()
