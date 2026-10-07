"""Read-only adapter to the pinned sibling controller and its native runtime."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from .store import ROOT, json_write, sha256

CONTROL = Path("/home/sunghwan/workspace/reachy-control")
CONTROL_PYTHON = CONTROL / ".venv/bin/python"
ARM_SUFFIXES = (
    "shoulder_pitch",
    "shoulder_roll",
    "elbow_yaw",
    "elbow_pitch",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
ARMS = tuple(side + n for side in ("l_", "r_") for n in ARM_SUFFIXES)
NECK = ("neck_roll", "neck_pitch", "neck_yaw")


def in_runtime():
    return Path(sys.prefix) == CONTROL / ".venv"


def dispatch(module, args):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + str(CONTROL)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    return subprocess.run(
        [str(CONTROL_PYTHON), "-m", module, *args], env=env, cwd=ROOT, check=True
    )


def identity(root=ROOT):
    files = [
        CONTROL / "control/reachy.py",
        CONTROL / "control/native.py",
        CONTROL / "control/collision.py",
        CONTROL / "asset/reachy.urdf",
        CONTROL / "asset/collision_spheres.json",
        CONTROL / "native/build/libreachy_wbc.so",
    ]
    manifest = {str(p): sha256(p) for p in files}
    path = Path(root) / "catalog/controller_identity.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != manifest:
            raise RuntimeError(
                "Controller/model identity changed; create a new explicit snapshot"
            )
    else:
        json_write(path, manifest)
    return manifest


class Robot:
    def __init__(self, root=ROOT):
        if str(CONTROL) not in sys.path:
            sys.path.insert(0, str(CONTROL))
        import pinocchio as pin
        from control import Controller, Settings

        self.pin = pin
        self.identity = identity(root)
        self.r = Controller(Settings(neck_enabled=False))
        self.q = self.r.q0.copy()
        self.arm_ids = np.array([self.r.q_indices[n] for n in ARMS])
        self.arm_v = np.array([self.r.v_indices[n] for n in ARMS])
        self.neck_ids = np.array([self.r.q_indices[n] for n in NECK])
        self.neck_v = np.array([self.r.v_indices[n] for n in NECK])
        self.active = np.r_[np.arange(3), self.arm_v]

    def fk(self, q):
        self.r.update(q)
        return np.array([self.r.data.oMf[f].homogeneous.copy() for f in self.r.hands])

    def integrate(self, q, v, dt=1):
        return self.pin.integrate(self.r.model, q, v * dt)

    def pack(self, arms, base=None, neck=None):
        q = self.r.q0.copy()
        q[self.arm_ids] = arms
        if base is not None:
            q[:4] = [base[0], base[1], np.cos(base[2]), np.sin(base[2])]
        if neck is not None:
            q[self.neck_ids] = neck
        return q

    def base(self, q):
        return np.array([q[0], q[1], np.arctan2(q[3], q[2])])

    def ik(self, targets, q=None, active_hands=(0, 1), iterations=100):
        q = self.r.q0.copy() if q is None else q.copy()
        pin = self.pin
        r = self.r
        for _ in range(iterations):
            self.fk(q)
            errors = []
            jac = []
            for h in active_hands:
                cur = r.data.oMf[r.hands[h]]
                goal = targets[h]
                err = np.r_[
                    goal[:3, 3] - cur.translation,
                    pin.log3(goal[:3, :3] @ cur.rotation.T),
                ]
                J = pin.computeFrameJacobian(
                    r.model,
                    r.data,
                    q,
                    r.hands[h],
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )
                errors.append(err)
                jac.append(J[:, self.active])
            e = np.concatenate(errors)
            J = np.concatenate(jac)
            if (
                max(np.linalg.norm(x[:3]) for x in errors) < 0.001
                and max(np.linalg.norm(x[3:]) for x in errors) < 0.01
            ):
                break
            scale = np.tile(np.r_[np.ones(3), np.full(3, 0.35)], len(active_hands))
            J = J * scale[:, None]
            e = e * scale
            weight = np.ones(len(self.active))
            weight[:3] = 0.2
            step = weight * (
                J.T @ np.linalg.solve((J * weight) @ J.T + np.eye(len(e)) * 1e-4, e)
            )
            v = np.zeros(r.model.nv)
            v[self.active] = np.clip(step, -0.1, 0.1)
            q = self.integrate(q, v)
            qi = r._joint_q_indices
            q[qi] = np.clip(
                q[qi],
                r.model.lowerPositionLimit[qi] + 0.031,
                r.model.upperPositionLimit[qi] - 0.031,
            )
        actual = self.fk(q)
        pe = max(
            np.linalg.norm(actual[h, :3, 3] - targets[h, :3, 3]) for h in active_hands
        )
        re = max(
            np.linalg.norm(pin.log3(targets[h, :3, :3] @ actual[h, :3, :3].T))
            for h in active_hands
        )
        return q, float(pe), float(re)

    def control(self, q, world_goals):
        r = self.r
        r.update(q)
        base_inv = r.data.oMi[r.base_joint].inverse()
        goals = [base_inv * self.pin.SE3(m[:3, :3], m[:3, 3]) for m in world_goals]
        neck = base_inv.rotation @ r.data.oMf[r.head].rotation
        return r.control(q, goals[0], goals[1], neck)
