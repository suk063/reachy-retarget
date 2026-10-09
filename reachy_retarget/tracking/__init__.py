"""Pure tracking retargeting (docs/tracking.md).

Turns tracking references -- world poses of both TCPs (``{l,r}_arm_tip``), a head rotation, a
nominal base path and gripper openings -- into Reachy joint trajectories and validates them
kinematically (tier K only). There are no objects, no grasp re-selection and no relaxation of the
reference: the reference is tracked as given, and frames Reachy cannot follow fail tier K.

This package is separate from the object-centric manipulation retargeting
(:mod:`reachy_retarget.retarget.pipeline`); it reuses only embodiment-level pieces (robot model,
whole-body IK frame solver, timing, schema, tier-K checks).

References come from :mod:`.synthetic` (procedural scenarios) and :mod:`.source` (hand paths of
the source datasets read by :mod:`reachy_retarget.sources`).
"""
from .config import TrackingConfig
from .reference import TrackingReference

__all__ = ["TrackingConfig", "TrackingReference"]
