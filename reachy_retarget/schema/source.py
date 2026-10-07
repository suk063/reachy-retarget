"""Embodiment-independent source demonstrations, as produced by source adapters.

All arrays use the source world frame (metres, z up) and the source clock. Poses are
``(T, 4, 4)`` homogeneous matrices; object tracks are ``(T, 7)`` as xyz + wxyz.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

ObjectRole = Literal["manipulated", "support", "receptacle", "fixture"]
Side = Literal["left", "right"]


@dataclass
class Effector:
    """A source gripper, expressed at its grasp center.

    The pose frame has the approach direction (from palm toward the fingertips) along
    +z and the finger closing direction along +y, independent of the source gripper.
    """

    pose: np.ndarray                 # (T, 4, 4) world pose of the grasp center
    opening: np.ndarray              # (T,) in [0, 1], 1 = fully open
    width: np.ndarray | None = None  # (T,) finger separation in metres, if known
    side_hint: Side | None = None


@dataclass
class ObjectTrack:
    pose: np.ndarray                 # (T, 7) xyz + wxyz in world
    valid: np.ndarray                # (T,) bool; no interpolation across invalid rows
    role: ObjectRole = "manipulated"
    geometry: dict = field(default_factory=dict)  # e.g. {"kind": "box", "half_extents": [...]}


@dataclass
class Articulation:
    joint_names: list[str]
    qpos: np.ndarray                 # (T, n)


@dataclass
class SceneRef:
    """A MuJoCo scene usable for physics validation (tier P)."""

    mjcf: str                        # complete MJCF XML of the source scene
    robot_prefixes: list[str]        # body-name prefixes of the source robot to remove
    initial_qpos: dict = field(default_factory=dict)  # object joint name -> qpos at t0
    assets: dict = field(default_factory=dict)        # asset file name -> bytes


@dataclass
class SourceEpisode:
    family: str                      # e.g. "robomimic"
    dataset: str                     # e.g. "robomimic/can/ph"
    episode_id: str                  # stable id within the dataset
    task: str
    time: np.ndarray                 # (T,) seconds, strictly increasing
    effectors: dict[str, Effector]
    objects: dict[str, ObjectTrack] = field(default_factory=dict)
    base: np.ndarray | None = None   # (T, 3) x, y, yaw in world; None = fixed base
    base_hint: np.ndarray | None = None  # (3,) fixed source robot base x, y, yaw (placement prior)
    torso_height: np.ndarray | None = None
    articulations: dict[str, Articulation] = field(default_factory=dict)
    scene: SceneRef | None = None
    instruction: str | None = None
    success: bool | None = None
    regime: Literal["tabletop", "mobile_manipulation", "navigation"] = "tabletop"
    license: str = "unknown"
    provenance: dict = field(default_factory=dict)  # urls, revision, sha256 of source files
    lineage: dict = field(default_factory=dict)     # {"seed": ..., "generated": bool}

    def __post_init__(self):
        self.time = np.asarray(self.time, float)
        T = len(self.time)
        if T < 2 or not np.all(np.isfinite(self.time)) or not np.all(np.diff(self.time) > 0):
            raise ValueError("time must contain at least two strictly increasing finite values")
        if not self.effectors and self.base is None:
            raise ValueError("an episode needs at least one effector or a base path")
        for name, e in self.effectors.items():
            e.pose = np.asarray(e.pose, float)
            e.opening = np.asarray(e.opening, float)
            if e.pose.shape != (T, 4, 4) or e.opening.shape != (T,):
                raise ValueError(f"effector {name}: expected pose (T,4,4) and opening (T,)")
            if not (np.all(np.isfinite(e.pose)) and np.all(np.isfinite(e.opening))):
                raise ValueError(f"effector {name}: non-finite values")
            if e.opening.min() < -1e-6 or e.opening.max() > 1 + 1e-6:
                raise ValueError(f"effector {name}: opening outside [0, 1]")
            if e.width is not None:
                e.width = np.asarray(e.width, float)
                if e.width.shape != (T,):
                    raise ValueError(f"effector {name}: width must be (T,)")
        for name, o in self.objects.items():
            o.pose = np.asarray(o.pose, float)
            o.valid = np.asarray(o.valid, bool)
            if o.pose.shape != (T, 7) or o.valid.shape != (T,):
                raise ValueError(f"object {name}: expected pose (T,7) and valid (T,)")
        if self.base is not None:
            self.base = np.asarray(self.base, float)
            if self.base.shape != (T, 3):
                raise ValueError("base must be (T, 3)")
        if self.base_hint is not None:
            self.base_hint = np.asarray(self.base_hint, float)
            if self.base_hint.shape != (3,) or not np.all(np.isfinite(self.base_hint)):
                raise ValueError("base_hint must be three finite values")
        for name, a in self.articulations.items():
            a.qpos = np.asarray(a.qpos, float)
            if a.qpos.shape != (T, len(a.joint_names)):
                raise ValueError(f"articulation {name}: qpos must be (T, n)")

    @property
    def length(self) -> int:
        return len(self.time)

    @property
    def uid(self) -> str:
        return f"{self.dataset}/{self.episode_id}"
