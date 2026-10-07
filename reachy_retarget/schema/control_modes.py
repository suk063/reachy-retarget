"""Registry of control views derived from the canonical episode state.

Every action array has ``T - 1`` rows: row ``t`` is the command issued at state row
``t`` whose target is state row ``t + 1`` (one 20 ms step at 50 Hz). There is no action
for the last state row. State views (only ``reachy_agent_v8/state``) have ``T`` rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .episode import BASE, DT, JOINTS, SIDES, ReachyEpisode
from .rotations import matrix_to_quat, rot6d, se2_log, se2_relative, se3_inv, se3_log, so3_log, wrap_angle

ROT6D = ("r00", "r10", "r20", "r01", "r11", "r21")
POS_ROT6D = ("x", "y", "z", *ROT6D)
POS_QUAT = ("x", "y", "z", "qw", "qx", "qy", "qz")
DELTA = ("dx", "dy", "dz", "rx", "ry", "rz")
TWIST = ("vx", "vy", "vz", "wx", "wy", "wz")
V8_ACTION_NAMES = (*(f"{s}_{n}" for s in SIDES for n in POS_ROT6D), *(f"head_{n}" for n in ROT6D),
                   "base_dx", "base_dy", "base_dyaw", "left_gripper_open", "right_gripper_open")
V8_STATE_NAMES = (*V8_ACTION_NAMES[:24], "base_world_x", "base_world_y", "base_world_sin_yaw",
                  "base_world_cos_yaw", "left_gripper_opening", "right_gripper_opening")


@dataclass(frozen=True)
class ModeSpec:
    name: str
    frame: str
    units: str
    semantics: str
    columns: dict[str, tuple[str, ...]]           # array name -> column names
    fn: Callable[[ReachyEpisode], dict[str, np.ndarray]]

    def compute(self, ep: ReachyEpisode) -> dict[str, np.ndarray]:
        out = self.fn(ep)
        if set(out) != set(self.columns):
            raise RuntimeError(f"mode {self.name}: arrays {sorted(out)} != {sorted(self.columns)}")
        for k, a in out.items():
            if a.ndim != 2 or a.shape[1] != len(self.columns[k]) or not np.isfinite(a).all():
                raise RuntimeError(f"mode {self.name}/{k}: bad shape {a.shape} or non-finite values")
        return out


MODES: dict[str, ModeSpec] = {}


def _mode(name, frame, units, semantics, **columns):
    def register(fn):
        MODES[name] = ModeSpec(name, frame, units, semantics, columns, fn)
        return fn
    return register


def compute_modes(ep: ReachyEpisode) -> dict[str, dict[str, np.ndarray]]:
    return {name: spec.compute(ep) for name, spec in MODES.items()}


def _pos_rot6d(T):
    return np.concatenate((T[..., :3, 3], rot6d(T[..., :3, :3])), axis=-1)


def _pos_quat(T):
    return np.concatenate((T[..., :3, 3], matrix_to_quat(T[..., :3, :3])), axis=-1)


def _joint_delta(q):
    d = np.diff(q, axis=0)
    d[:, 2] = wrap_angle(d[:, 2])
    return d


def _base_delta(ep):
    b = ep.base_pose_world
    return se2_relative(b[:-1], b[1:])


@_mode("joint_pos_abs", "joint space; base_x/base_y/base_yaw in world", "m, rad",
       "action[t] = q[t+1] (all 22 joints including base and fingers)", action=JOINTS)
def _joint_pos_abs(ep):
    return {"action": ep.q[1:]}


@_mode("joint_pos_delta", "joint space; base terms in world", "m, rad",
       "action[t] = q[t+1] - q[t], base_yaw difference wrapped to [-pi, pi)", action=JOINTS)
def _joint_pos_delta(ep):
    return {"action": _joint_delta(ep.q)}


@_mode("joint_vel", "joint space; base terms in world", "m/s, rad/s",
       "action[t] = qd[t], the trajectory velocity held over [t, t+1)", action=JOINTS)
def _joint_vel(ep):
    return {"action": ep.qd[:-1]}


def _ee_abs(poses):
    out = {}
    for s in SIDES:
        out[f"{s}_pos_rot6d"] = _pos_rot6d(poses[s][1:])
        out[f"{s}_pos_quat"] = _pos_quat(poses[s][1:])
    return out


_EE_ABS_COLUMNS = {f"{s}_{k}": tuple(f"{s}_{n}" for n in names)
                   for s in SIDES for k, names in (("pos_rot6d", POS_ROT6D), ("pos_quat", POS_QUAT))}


@_mode("ee_abs_base", "base_link at t+1", "m, rotation6d / wxyz quaternion (w >= 0)",
       "action[t] = TCP pose at t+1 in the base frame at t+1", **_EE_ABS_COLUMNS)
def _ee_abs_base(ep):
    return _ee_abs(ep.tcp_base)


@_mode("ee_abs_world", "world", "m, rotation6d / wxyz quaternion (w >= 0)",
       "action[t] = TCP world pose at t+1", **_EE_ABS_COLUMNS)
def _ee_abs_world(ep):
    return _ee_abs(ep.tcp_world)


@_mode("ee_delta_base", "<side>_tool: current TCP frame; <side>_base: base_link (arm-only motion)",
       "m, rad (rotation vector)",
       "tool: D = inv(P[t]) P[t+1], action = (D translation, log(D rotation)), so "
       "P[t+1] = P[t] D; base: action = (p[t+1] - p[t], log(R[t+1] R[t]^T)), so "
       "p[t+1] = p[t] + dp and R[t+1] = exp(r) R[t]; P = TCP pose in base_link",
       **{f"{s}_{k}": tuple(f"{s}_{n}" for n in DELTA) for s in SIDES for k in ("tool", "base")})
def _ee_delta_base(ep):
    out = {}
    for s in SIDES:
        P = ep.tcp_base[s]
        D = se3_inv(P[:-1]) @ P[1:]
        out[f"{s}_tool"] = np.concatenate((D[:, :3, 3], so3_log(D[:, :3, :3])), axis=-1)
        R = P[:, :3, :3]
        out[f"{s}_base"] = np.concatenate((np.diff(P[:, :3, 3], axis=0),
                                           so3_log(R[1:] @ np.swapaxes(R[:-1], 1, 2))), axis=-1)
    return out


@_mode("ee_twist_tool", "current TCP frame (body twist), relative to base_link", "m/s, rad/s",
       "action[t] = se3_log(inv(P[t]) P[t+1]) / dt: the constant body twist reaching P[t+1] in "
       "20 ms; P = TCP pose in base_link", **{s: tuple(f"{s}_{n}" for n in TWIST) for s in SIDES})
def _ee_twist_tool(ep):
    return {s: se3_log(se3_inv(ep.tcp_base[s][:-1]) @ ep.tcp_base[s][1:]) / DT for s in SIDES}


@_mode("head_rot_abs", "base_link at t+1", "rotation6d",
       "action[t] = head frame orientation at t+1 in the base frame at t+1",
       rot6d=tuple(f"head_{n}" for n in ROT6D))
def _head_rot_abs(ep):
    return {"rot6d": rot6d(ep.head_base[1:, :3, :3])}


@_mode("head_rot_delta", "current head frame", "rotation6d",
       "action[t] = rot6d(H[t]^T H[t+1]), H = head orientation in base_link",
       rot6d=tuple(f"head_d{n}" for n in ROT6D))
def _head_rot_delta(ep):
    H = ep.head_base[:, :3, :3]
    return {"rot6d": rot6d(np.swapaxes(H[:-1], 1, 2) @ H[1:])}


@_mode("base_twist_body", "base_link at t (body frame)", "m/s, rad/s",
       "action[t] = se2_log(base displacement t->t+1, 20 ms): constant body twist reaching "
       "the next base pose", twist=("base_vx", "base_vy", "base_wz"))
def _base_twist_body(ep):
    return {"twist": se2_log(_base_delta(ep), DT)}


@_mode("base_se2_delta", "base_link at t (body frame)", "m, rad",
       "action[t] = body-frame displacement (dx, dy, dyaw) from the base pose at t to t+1 "
       "(20 ms, as reachy-agent v8 base_dx/base_dy/base_dyaw); dyaw wrapped",
       delta=("base_dx", "base_dy", "base_dyaw"))
def _base_se2_delta(ep):
    return {"delta": _base_delta(ep)}


@_mode("gripper_continuous", "gripper", "opening in [0, 1], 1 = open",
       "action[t] = gripper opening at t+1", opening=("left_gripper_opening", "right_gripper_opening"))
def _gripper_continuous(ep):
    return {"opening": ep.gripper_opening[1:]}


@_mode("gripper_binary", "gripper", "{0, 1}, 1 = open",
       "action[t] = 1 if gripper opening at t+1 >= 0.5 else 0",
       open=("left_gripper_open", "right_gripper_open"))
def _gripper_binary(ep):
    return {"open": (ep.gripper_opening[1:] >= 0.5).astype(float)}


@_mode("reachy_agent_v8", "TCP/head: base_link at t+1; base delta: base_link at t; base state: world",
       "m, rad, rotation6d, binary/continuous gripper; float32",
       "reachy-agent contract current-base-tcp-head-rotation6d-local-base-delta-binary-grippers-v8: "
       "action[t] = (TCP poses and head rotation at t+1 in base_link, 20 ms local base "
       "displacement t->t+1, binary gripper opening at t+1); state[t] = "
       "base-tcp-head-world-base-xy-sincos-opening-v8 at t (T rows)",
       action=V8_ACTION_NAMES, state=V8_STATE_NAMES)
def _reachy_agent_v8(ep):
    def poses(rows):
        return [*(_pos_rot6d(ep.tcp_base[s][rows]) for s in SIDES), rot6d(ep.head_base[rows, :3, :3])]

    nxt, b = slice(1, None), ep.q[:, BASE]
    action = np.concatenate([*poses(nxt), _base_delta(ep), (ep.gripper_opening[nxt] >= 0.5)], axis=-1)
    base_state = np.stack((b[:, 0], b[:, 1], np.sin(b[:, 2]), np.cos(b[:, 2])), axis=-1)
    state = np.concatenate([*poses(slice(None)), base_state, ep.gripper_opening], axis=-1)
    return {"action": action.astype(np.float32), "state": state.astype(np.float32)}
