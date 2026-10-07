"""Reachy 2 whole-body kinematics on the canonical 22-value joint vector (see docs/design.md).

World frame: the floor plane z = 0; the mobile base is the planar pose (base_x, base_y, base_yaw)
of `base_link`. The tripod stays at 0 (its calibrated extension is baked into the URDF) and the
antennas and wheels are not modelled as joints. Frames:

* `left_tcp` / `right_tcp`: URDF `{l,r}_arm_tip` (tool reference; its +z points from the
  fingertips back toward the wrist),
* `left_grasp` / `right_grasp`: TCP @ grasp_center(side),
* `head`: URDF `head`, `torso`: URDF `torso`.
"""
import functools

import numpy as np

from . import gripper
from .resources import urdf
from .urdf import KinematicTree

ARM = ("shoulder_pitch", "shoulder_roll", "elbow_yaw", "elbow_pitch", "wrist_roll", "wrist_pitch", "wrist_yaw")
JOINTS = (("base_x", "base_y", "base_yaw") + tuple(f"l_{n}" for n in ARM) + tuple(f"r_{n}" for n in ARM)
          + ("neck_roll", "neck_pitch", "neck_yaw", "l_hand_finger", "r_hand_finger"))
BASE, LEFT_ARM, RIGHT_ARM, NECK, GRIPPERS = slice(0, 3), slice(3, 10), slice(10, 17), slice(17, 20), slice(20, 22)
BODY = slice(3, 20)  # arm and neck joints: the URDF part of the IK variables q[0:20]
SIDES = {"left": "l", "right": "r"}
FIXED = {"tripod_joint": 0.0}

LOWER = np.array([-np.inf] * 3 + [urdf().limits[n][0] for n in JOINTS[3:]])
UPPER = np.array([np.inf] * 3 + [urdf().limits[n][1] for n in JOINTS[3:]])
# Joint speed limits (m/s or rad/s). Arms 1 rad/s, neck 30 deg/s, base 0.61 m/s per axis and
# 114 deg/s (whole-body controller limits, docs/design.md). The URDF gripper velocity (10 rad/s)
# is a placeholder; 3 rad/s closes the full 2.35 rad stroke in 0.78 s, the fastest grasp/release
# duration of reachy-agent's validated scripted grasps (simulation/config.py: 0.8-1.5 s).
VELOCITY = np.array([0.61, 0.61, np.radians(114)] + [1.0] * 14 + [np.radians(30)] * 3 + [3.0] * 2)

TCP = {"left_tcp": "l_arm_tip", "right_tcp": "r_arm_tip"}
LINKS = {**TCP, "head": "head", "torso": "torso"}
GRASP = {"left_grasp": "left_tcp", "right_grasp": "right_tcp"}
FRAMES = tuple(LINKS) + tuple(GRASP)


def planar(x, y, yaw):
    """SE(2) floor pose as 4x4 (batched over the inputs' shape)."""
    x, y, yaw = np.broadcast_arrays(*(np.asarray(v, float) for v in (x, y, yaw)))
    T = np.broadcast_to(np.eye(4), x.shape + (4, 4)).copy()
    c, s = np.cos(yaw), np.sin(yaw)
    T[..., 0, 0], T[..., 0, 1], T[..., 1, 0], T[..., 1, 1] = c, -s, s, c
    T[..., 0, 3], T[..., 1, 3] = x, y
    return T


class Reachy:
    """Kinematic model. Use `Reachy.load()` (cached); all inputs are canonical q (..., 22)."""

    def __init__(self):
        self.urdf = urdf()
        self._all = KinematicTree(self.urdf, "base_link", list(LINKS.values()), JOINTS[BODY], FIXED)
        self._single = {f: KinematicTree(self.urdf, "base_link", [link], JOINTS[BODY], FIXED)
                        for f, link in LINKS.items()}
        self._grasp = {f: self._grasp_center(SIDES[f.split("_")[0]]) for f in TCP}

    @classmethod
    @functools.cache
    def load(cls):
        return cls()

    def _grasp_center(self, s):
        T_palm_tip = self.urdf.transform(f"{s}_hand_palm_link", f"{s}_arm_tip")
        T_palm_gc = np.eye(4)
        T_palm_gc[:3, 3] = gripper.pad_centers(gripper.STRAIGHT_ANGLE, "left" if s == "l" else "right").mean(0)
        return np.linalg.inv(T_palm_tip) @ T_palm_gc

    def grasp_center(self, side):
        """T_tip_gc (4x4): grasp-center frame in the `{l,r}_arm_tip` frame.

        Origin: midpoint between the centres of the two distal pad faces (see gripper.py) with
        the proximal links straight (finger angle gripper.STRAIGHT_ANGLE = 1.1815 rad), i.e. in
        the palm frame x = -0.0102 (finger pivot offset), y = 0 and z = 0.0700 (finger pivots)
        + 0.0500 (proximal length) + 0.02675 (pad centre on the distal axis) = 0.14678 m.
        Axes are those of the palm frame: +z = approach (palm toward fingertips), +y = closing
        direction of the mimic finger, i.e. from `*_hand_distal_mimic_link` toward
        `*_hand_distal_link`, +x = y x z. The tip frame is the palm frame rotated by pi about x
        and shifted 0.10 m along palm z, so for both sides
        T_tip_gc = [[1, 0, 0, -0.0102], [0, -1, 0, 0], [0, 0, -1, -0.04678]].
        The pad centre moves by at most 8.6 mm toward the palm over the gripper stroke.
        """
        return self._grasp[f"{side}_tcp"].copy()

    def tcp_from_grasp_center(self, T_gc, side):
        """TCP (`{l,r}_arm_tip`) pose(s) that put the grasp center at T_gc (..., 4, 4)."""
        return np.asarray(T_gc, float) @ np.linalg.inv(self._grasp[f"{side}_tcp"])

    def fk_base(self, q):
        """{frame: pose (..., 4, 4) in base_link} for FRAMES."""
        q = np.asarray(q, float)
        out = {f: T for f, T in zip(LINKS, self._all.fk(q[..., BODY]).values())}
        out.update({g: out[t] @ self._grasp[t] for g, t in GRASP.items()})
        return out

    def fk(self, q):
        """{frame: world pose (..., 4, 4)} for FRAMES; base_link = planar(q[0:3])."""
        q = np.asarray(q, float)
        B = planar(q[..., 0], q[..., 1], q[..., 2])
        return {f: B @ T for f, T in self.fk_base(q).items()}

    def jacobian(self, q, frame):
        """(world pose 4x4, 6x20 geometric Jacobian in the world frame) of one configuration.

        Rows: linear velocity of the frame origin, then angular velocity. Columns: q[0:20]
        (base x, y, yaw in the world, then arm and neck joints).
        """
        q = np.asarray(q, float)
        tcp = GRASP.get(frame, frame)
        tree = self._single[tcp]
        T, Jb = tree.jacobian(q[BODY], LINKS[tcp])
        if frame in GRASP:
            G = self._grasp[tcp]
            Jb = Jb.copy()
            Jb[:3] -= np.cross(T[:3, :3] @ G[:3, 3], Jb[3:], axis=0)
            T = T @ G
        B = planar(*q[:3])
        T = B @ T
        J = np.zeros((6, 20))
        J[:3, 3:], J[3:, 3:] = B[:3, :3] @ Jb[:3], B[:3, :3] @ Jb[3:]
        J[0, 0] = J[1, 1] = J[5, 2] = 1.0
        J[0, 2], J[1, 2] = -(T[1, 3] - q[1]), T[0, 3] - q[0]
        return T, J
