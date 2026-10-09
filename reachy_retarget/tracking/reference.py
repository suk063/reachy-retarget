"""Tracking reference: what the robot must follow, on the reference's own clock."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..schema.episode import BODY_PARTS, REGIMES, SIDES


@dataclass
class TrackingReference:
    """World-frame targets of one tracking episode.

    * ``tcp``: both sides always; an idle hand gets a hold target (world or base-relative).
    * ``head``: world rotation of the ``head`` frame (equal to that of reachy-control's ``head_tip``),
      or None when the neck is not used (it then holds ``q_start``'s neck angles).
    * ``base``: nominal base path; the base is free in the IK but kept near it (``w_base``, box).
    * ``opening``: gripper opening in [0, 1] (1 = open), left and right.
    * ``q_start``: robot state at the first row (seed of the first IK frame); None = cold start.
    * ``retime``: False = ``time`` is already the 50 Hz output clock and is kept (synthetic);
      True = slow down where the speed limits need it and resample to 50 Hz (source paths).
    * ``witness``: optional (T, 22) joint path whose FK produced the reference (stored for audit
      only; the IK never sees it).
    """

    dataset: str
    episode_id: str
    task: str
    time: np.ndarray
    tcp: dict[str, np.ndarray]
    base: np.ndarray
    opening: np.ndarray
    head: np.ndarray | None = None
    q_start: np.ndarray | None = None
    retime: bool = False
    regime: str = "tabletop"
    body_parts: tuple[str, ...] = ()
    lineage: dict = field(default_factory=dict)
    variant_of: str | None = None
    license: str = "unknown"
    provenance: dict = field(default_factory=dict)
    instruction: str | None = None
    witness: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.time = np.asarray(self.time, float)
        T = len(self.time)
        if self.time.ndim != 1 or T < 2 or np.any(np.diff(self.time) <= 0) or not np.isfinite(self.time).all():
            raise ValueError("time must be strictly increasing with at least two rows")
        if set(self.tcp) != set(SIDES):
            raise ValueError(f"tcp needs both sides {SIDES}")
        self.tcp = {s: _rigid(f"tcp.{s}", self.tcp[s], T) for s in SIDES}
        self.base = _finite("base", self.base, (T, 3))
        self.opening = _finite("opening", self.opening, (T, 2))
        if self.opening.min() < 0 or self.opening.max() > 1:
            raise ValueError("opening must lie in [0, 1]")
        if self.head is not None:
            H = _finite("head", self.head, (T, 3, 3))
            if not np.allclose(H @ np.swapaxes(H, 1, 2), np.eye(3), atol=1e-6):
                raise ValueError("head: not rotation matrices")
            self.head = H
        if self.q_start is not None:
            self.q_start = _finite("q_start", self.q_start, (22,))
        if self.witness is not None:
            self.witness = _finite("witness", self.witness, (T, 22))
        if self.regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}")
        if not set(self.body_parts) <= set(BODY_PARTS):
            raise ValueError(f"body_parts must be a subset of {BODY_PARTS}")
        if not self.retime and not np.allclose(np.diff(self.time), np.diff(self.time)[0], atol=1e-9):
            raise ValueError("a reference that is not retimed must be uniformly sampled")

    @property
    def length(self) -> int:
        return len(self.time)


def _finite(name, a, shape):
    a = np.asarray(a, float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected finite values of shape {shape}, got {a.shape}")
    return a


def _rigid(name, a, T):
    a = _finite(name, a, (T, 4, 4))
    R = a[:, :3, :3]
    if (not np.allclose(a[:, 3], [0, 0, 0, 1], atol=1e-9)
            or not np.allclose(R @ np.swapaxes(R, 1, 2), np.eye(3), atol=1e-6)):
        raise ValueError(f"{name}: not rigid transforms")
    return a


def motion_parts(tcp, base, head, *, pos_tol=0.005, rot_tol=0.02) -> tuple[str, ...]:
    """Body parts a reference uses: an arm whose base-relative TCP target moves (more than
    ``pos_tol`` / ``rot_tol`` from its first row), the head when it is tracked, the base when its
    nominal path moves more than 1 mm / 0.01 rad."""
    from ..robot import planar
    from ..schema.rotations import so3_log

    W = planar(base[:, 0], base[:, 1], base[:, 2])
    parts = []
    for s in SIDES:
        X = np.linalg.inv(W) @ tcp[s]
        dp = np.linalg.norm(X[:, :3, 3] - X[0, :3, 3], axis=1).max()
        dr = np.linalg.norm(so3_log(np.swapaxes(X[:1, :3, :3], 1, 2) @ X[:, :3, :3]), axis=1).max()
        if dp > pos_tol or dr > rot_tol:
            parts.append(f"{s}_arm")
    if head is not None:
        parts.append("head")
    if np.ptp(base[:, :2], axis=0).max() > 1e-3 or np.ptp(base[:, 2]) > 0.01:
        parts.append("base")
    return tuple(parts)


def regime_of(parts) -> str:
    """``navigation`` (only the base and/or head move), ``mobile_manipulation`` (base and an arm),
    else ``tabletop`` (stationary base)."""
    arms = any(p.endswith("_arm") for p in parts)
    if "base" in parts:
        return "mobile_manipulation" if arms else "navigation"
    return "tabletop"


def jsonable(x):
    """Recursively convert numpy values to JSON types; non-finite floats become None."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, (str, type(None))):
        return x
    return str(x)
