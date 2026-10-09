"""Canonical, validated content of one retargeted Reachy 2 episode.

The episode stores robot and object *state* only (never images). Every control view is
derived from it by :mod:`reachy_retarget.schema.control_modes`. Robot poses are
``(T, 4, 4)`` matrices in memory; object tracks keep the source ``(T, 7)`` xyz + wxyz
form. The ``base_link`` frame lies on the floor, so world poses follow from base-frame
poses and the planar base pose ``q[:, 0:3]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rotations import planar_pose
from .source import Articulation, ObjectTrack

HZ = 50
DT = 1.0 / HZ
SIDES = ("left", "right")
_ARM = ("shoulder_pitch", "shoulder_roll", "elbow_yaw", "elbow_pitch", "wrist_roll", "wrist_pitch", "wrist_yaw")
JOINTS = ("base_x", "base_y", "base_yaw",
          *(f"l_{j}" for j in _ARM), *(f"r_{j}" for j in _ARM),
          "neck_roll", "neck_pitch", "neck_yaw", "l_hand_finger", "r_hand_finger")
BASE, NECK, FINGERS = slice(0, 3), slice(17, 20), slice(20, 22)
ARMS = {"left": slice(3, 10), "right": slice(10, 17)}
BODY_PARTS = ("left_arm", "right_arm", "head", "base")
REGIMES = ("tabletop", "mobile_manipulation", "navigation")
OBJECT_ROLES = ("manipulated", "support", "receptacle", "fixture")
COMPONENT_ROLES = (*OBJECT_ROLES, "articulated_part", "robot_link")
POSE_SOURCES = ("object_track", "scene_state", "static", "robot_fk")


def _array(name, a, shape, *, finite=True):
    a = np.asarray(a, float)
    if a.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {a.shape}")
    if finite and not np.isfinite(a).all():
        raise ValueError(f"{name}: non-finite values")
    return a


def _poses(name, a, T):
    a = _array(name, a, (T, 4, 4))
    R = a[:, :3, :3]
    if (not np.allclose(a[:, 3], [0, 0, 0, 1], atol=1e-9)
            or not np.allclose(R @ np.swapaxes(R, 1, 2), np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(R), 1, atol=1e-5)):
        raise ValueError(f"{name}: not rigid transforms")
    return a


def _key(kind, name):
    if not isinstance(name, str) or not name or "/" in name:
        raise ValueError(f"{kind} id must be a non-empty string without '/': {name!r}")


def _tier(name, t, required):
    if t is None and not required:
        return None
    if (not isinstance(t, dict) or set(t) != {"passed", "reasons"} or not isinstance(t["passed"], bool)
            or not all(isinstance(r, str) for r in t["reasons"])):
        raise ValueError(f"tier {name} must be {{'passed': bool, 'reasons': [str, ...]}}")
    return {"passed": t["passed"], "reasons": list(t["reasons"])}


@dataclass
class Reference:
    """Pre-IK targets in the world frame, resampled onto the episode clock."""

    tcp: dict[str, np.ndarray] = field(default_factory=dict)  # side -> (T, 4, 4) TCP target
    head: np.ndarray | None = None   # (T, 4, 4) head target (gaze orientation at the head frame)
    base: np.ndarray | None = None   # (T, 3) x, y, yaw base target


@dataclass
class PhysicsRollout:
    """Measured simulator state of a tier-P rollout, on the simulator's own clock."""

    time: np.ndarray                 # (N,) seconds
    qpos: np.ndarray                 # (N, nq)
    qvel: np.ndarray                 # (N, nv)
    ctrl: np.ndarray                 # (N, nu) actuator commands actually applied
    qpos_names: list[str]
    qvel_names: list[str]
    ctrl_names: list[str]
    info: dict = field(default_factory=dict)  # simulator version, timestep, assumptions

    def __post_init__(self):
        self.time = np.asarray(self.time, float)
        N = len(self.time)
        if self.time.ndim != 1 or not np.isfinite(self.time).all() or np.any(np.diff(self.time) <= 0):
            raise ValueError("physics.time must be strictly increasing and finite")
        for k in ("qpos", "qvel", "ctrl"):
            names = list(getattr(self, f"{k}_names"))
            setattr(self, f"{k}_names", names)
            setattr(self, k, _array(f"physics.{k}", getattr(self, k), (N, len(names))))


@dataclass
class SceneComponents:
    """Rigid scene components with their mesh parts and per-frame world poses (``/scene``).

    ``components[c]`` describes pose column ``c`` (see docs/schema.md: name, role, source body,
    pose source, visual and collision parts referencing the asset library by SHA-256);
    ``materials`` and ``textures`` are keyed by the names the parts use. ``poses[t, c]`` is the
    world pose (xyz + wxyz) of component ``c`` at episode row ``t`` (NaN where ``valid`` is
    False); ``physics_poses`` the same on the tier-P rollout clock (``/physics/time``).
    """

    components: list[dict]
    poses: np.ndarray                        # (T, C, 7)
    valid: np.ndarray                        # (T, C) bool
    materials: dict = field(default_factory=dict)
    textures: dict = field(default_factory=dict)
    physics_poses: np.ndarray | None = None  # (N, C, 7)
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.components = list(self.components)
        C = len(self.components)
        names = [c.get("name") for c in self.components]
        if len(set(names)) != C or not all(isinstance(n, str) and n for n in names):
            raise ValueError("scene component names must be unique non-empty strings")
        for c in self.components:
            if c.get("role") not in COMPONENT_ROLES:
                raise ValueError(f"scene component {c.get('name')!r}: role must be one of {COMPONENT_ROLES}")
            if c.get("pose_source") not in POSE_SOURCES:
                raise ValueError(f"scene component {c.get('name')!r}: pose_source must be one of {POSE_SOURCES}")
            for kind in ("visual", "collision"):
                for p in c.get(kind, []):
                    if p.get("material") is not None and p["material"] not in self.materials:
                        raise ValueError(f"scene component {c['name']!r}: unknown material {p['material']!r}")
        for name, m in self.materials.items():
            for tex in [m.get("texture"), *(m.get("layers") or {}).values()]:
                if tex is not None and tex not in self.textures:
                    raise ValueError(f"material {name!r}: unknown texture {tex!r}")
        self.poses = np.asarray(self.poses, float)
        self.valid = np.asarray(self.valid, bool)
        if self.poses.ndim != 3 or self.poses.shape[1:] != (C, 7) or self.valid.shape != self.poses.shape[:2]:
            raise ValueError(f"scene poses must be (T, {C}, 7) with valid (T, {C})")
        if not np.isfinite(self.poses[self.valid]).all():
            raise ValueError("scene poses: valid rows must be finite")
        if self.physics_poses is not None:
            self.physics_poses = np.asarray(self.physics_poses, float)
            if self.physics_poses.ndim != 3 or self.physics_poses.shape[1:] != (C, 7):
                raise ValueError(f"scene physics_poses must be (N, {C}, 7)")

    @property
    def names(self) -> list[str]:
        return [c["name"] for c in self.components]


@dataclass
class ReachyEpisode:
    """One retargeted episode at 50 Hz in the canonical :data:`JOINTS` order."""

    family: str
    dataset: str
    episode_id: str
    task: str
    time: np.ndarray                         # (T,) seconds, uniform 1/HZ steps
    q: np.ndarray                            # (T, 22) JOINTS; base x, y, yaw in world
    qd: np.ndarray                           # (T, 22) time derivative of q (base in world)
    tcp_base: dict[str, np.ndarray]          # side -> (T, 4, 4) {l,r}_arm_tip in base_link
    head_base: np.ndarray                    # (T, 4, 4) head frame in base_link
    gripper_opening: np.ndarray              # (T, 2) left, right in [0, 1], 1 = open
    gripper_width: np.ndarray                # (T, 2) finger separation in metres
    source_time: np.ndarray                  # (T,) source clock time of each target row
    tier: dict                               # {"K": {"passed", "reasons"}, "P": {...} | None}
    retarget_config: str                     # hash of the retargeting configuration
    uid: str = ""                            # defaults to "<dataset>/<episode_id>"
    instruction: str | None = None
    regime: str = "tabletop"
    license: str = "unknown"
    provenance: dict = field(default_factory=dict)
    lineage: dict = field(default_factory=dict)       # {"seed": ..., "generated": bool, ...}
    variant_of: str | None = None                      # uid of the episode this one derives from
    body_parts: tuple[str, ...] = ()                   # subset of BODY_PARTS actually used
    reference: Reference = field(default_factory=Reference)
    objects: dict[str, ObjectTrack] = field(default_factory=dict)
    articulations: dict[str, Articulation] = field(default_factory=dict)
    validation: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (T, ...) per frame
    physics: PhysicsRollout | None = None
    scene: SceneComponents | None = None               # scene components, meshes and poses
    extra: dict = field(default_factory=dict)          # JSON metadata, e.g. simulation assumptions

    def __post_init__(self):
        self.uid = self.uid or f"{self.dataset}/{self.episode_id}"
        self.time = np.asarray(self.time, float)
        T = len(self.time)
        if (self.time.ndim != 1 or T < 2 or not np.isfinite(self.time).all()
                or not np.allclose(np.diff(self.time), DT, rtol=0, atol=1e-6)):
            raise ValueError(f"time must hold at least two samples at {HZ} Hz")
        n = len(JOINTS)
        self.q, self.qd = _array("q", self.q, (T, n)), _array("qd", self.qd, (T, n))
        if set(self.tcp_base) != set(SIDES):
            raise ValueError(f"tcp_base needs exactly the sides {SIDES}")
        self.tcp_base = {s: _poses(f"tcp_base.{s}", self.tcp_base[s], T) for s in SIDES}
        self.head_base = _poses("head_base", self.head_base, T)
        self.gripper_opening = _array("gripper_opening", self.gripper_opening, (T, 2))
        self.gripper_width = _array("gripper_width", self.gripper_width, (T, 2))
        if self.gripper_opening.min() < 0 or self.gripper_opening.max() > 1 or self.gripper_width.min() < 0:
            raise ValueError("gripper opening must be in [0, 1] and width non-negative")
        self.source_time = _array("source_time", self.source_time, (T,))
        if np.any(np.diff(self.source_time) < 0):
            raise ValueError("source_time must be non-decreasing")
        if not isinstance(self.tier, dict) or set(self.tier) != {"K", "P"}:
            raise ValueError("tier must have exactly the keys 'K' and 'P'")
        self.tier = {"K": _tier("K", self.tier["K"], True), "P": _tier("P", self.tier["P"], False)}
        if self.regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}")
        if not set(self.body_parts) <= set(BODY_PARTS):
            raise ValueError(f"body_parts must be a subset of {BODY_PARTS}")
        self.body_parts = tuple(p for p in BODY_PARTS if p in self.body_parts)
        ref = self.reference
        if not set(ref.tcp) <= set(SIDES):
            raise ValueError("reference.tcp keys must be sides")
        ref.tcp = {s: _poses(f"reference.tcp.{s}", ref.tcp[s], T) for s in SIDES if s in ref.tcp}
        if ref.head is not None:
            ref.head = _poses("reference.head", ref.head, T)
        if ref.base is not None:
            ref.base = _array("reference.base", ref.base, (T, 3))
        for k, o in self.objects.items():
            _key("object", k)
            o.pose = _array(f"object {k} pose", o.pose, (T, 7), finite=False)
            o.valid = np.asarray(o.valid, bool)
            if o.valid.shape != (T,) or not np.isfinite(o.pose[o.valid]).all():
                raise ValueError(f"object {k}: valid must be (T,) and valid rows finite")
            if o.role not in OBJECT_ROLES:
                raise ValueError(f"object {k}: role must be one of {OBJECT_ROLES}")
        for k, a in self.articulations.items():
            _key("articulation", k)
            a.joint_names = list(a.joint_names)
            a.qpos = _array(f"articulation {k} qpos", a.qpos, (T, len(a.joint_names)), finite=False)
        for k in list(self.validation):
            _key("validation", k)
            v = np.asarray(self.validation[k])
            if v.ndim < 1 or v.shape[0] != T or v.dtype.kind not in "biuf":
                raise ValueError(f"validation {k}: expected a numeric per-frame array with T rows")
            self.validation[k] = v
        if self.scene is not None:
            if self.scene.poses.shape[0] != T:
                raise ValueError(f"scene poses must have {T} rows")
            if self.scene.physics_poses is not None and (
                    self.physics is None or len(self.scene.physics_poses) != len(self.physics.time)):
                raise ValueError("scene physics_poses need a physics rollout with as many rows")

    @property
    def length(self) -> int:
        return len(self.time)

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0])

    @property
    def base_pose_world(self) -> np.ndarray:
        """(T, 3) world x, y, yaw of base_link, i.e. ``q[:, 0:3]``."""
        return self.q[:, BASE]

    @property
    def tcp_world(self) -> dict[str, np.ndarray]:
        W = planar_pose(self.base_pose_world)
        return {s: W @ self.tcp_base[s] for s in SIDES}

    @property
    def head_world(self) -> np.ndarray:
        return planar_pose(self.base_pose_world) @ self.head_base
