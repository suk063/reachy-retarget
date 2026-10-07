"""Measured common manipulation gates and independent actuator-only replay.

Collectors never mutate MuJoCo. Capture solved contacts immediately after
``mj_step``; synchronize the caller's caches before ``update``. Replay resets
once from a recorded integration state and subsequently writes only controls.
Missing measurements are incomplete validation, never successful gates.
"""
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .dynamics_audit import _verify_assets, _verify_no_object_assistance
from .robot import ARMS
from .store import sha256


THRESHOLDS = dict(arm_speed_rad_s=1.001, base_vx_m_s=.611, base_vy_m_s=.611,
    base_wz_rad_s=float(np.deg2rad(114)+.001), joint_margin_rad=.025,
    self_clearance_m=.009, hand_object_depth_m=.001, robot_environment_depth_m=.002,
    object_environment_depth_m=.002, robot_self_depth_m=.002, acquisition_lift_m=.015,
    carry_bilateral_fraction=.95, grasp_translation_m=.003, grasp_rotation_rad=float(np.deg2rad(3)),
    final_stable_s=1., stable_linear_speed_m_s=.02, stable_angular_speed_rad_s=.2,
    positive_contact_force_n=1e-6, replay_absolute_tolerance=1e-7)


def _subtree(model, name):
    root = model.body(name).id
    if root <= 0:
        raise ValueError("An explicit nonworld body root is required")
    result = {root}
    for body in range(root+1, model.nbody):
        if int(model.body_parentid[body]) in result:
            result.add(body)
    return result


class GateObserver:
    """One free task object and explicit grasping hands; every physics step.

    Positive clearance must come from the same measured robot geometry checker
    used by the main validator. A missing clearance or task measurement blocks
    completeness. Acceleration is measured diagnostically; the common protocol
    has no invented acceleration threshold.
    """
    def __init__(self, model, initial_data, *, active_object_body, grasp_hands,
                 robot_body_root="base_link", base_body="base_link", arm_joint_names=ARMS,
                 tcp_sites=None, finger_bodies=None, initial_self_clearance_m=None,
                 require_terminal_contact=False, scene_sha256=None):
        self.model = model
        self.scene_sha256 = scene_sha256
        self.robot = _subtree(model, robot_body_root)
        self.active = _subtree(model, active_object_body)
        if self.robot & self.active:
            raise ValueError("Moving object cannot overlap robot")
        self.active_id, self.base_id = model.body(active_object_body).id, model.body(base_body).id
        if self.base_id not in self.robot:
            raise ValueError("Base body must belong to robot")
        self.grasp_hands = tuple(grasp_hands)
        if not self.grasp_hands or set(self.grasp_hands)-{"left", "right"} or len(set(self.grasp_hands)) != len(self.grasp_hands):
            raise ValueError("Explicit unique left/right grasp hands required")
        tcp_sites = tcp_sites or {"left": "l_arm_tip_tcp", "right": "r_arm_tip_tcp"}
        finger_bodies = finger_bodies or {side: (prefix+"_hand_distal_link", prefix+"_hand_distal_mimic_link")
                                           for side, prefix in (("left", "l"), ("right", "r"))}
        self.fingers, self.tcp = {}, {}
        for side in self.grasp_hands:
            self.fingers[side] = {model.body(name).id for name in finger_bodies[side]}
            self.tcp[side] = model.site(tcp_sites[side]).id
            if len(self.fingers[side]) != 2 or not self.fingers[side] <= self.robot:
                raise ValueError("Each grasp hand requires two distinct robot pad bodies")
            if int(model.site_bodyid[self.tcp[side]]) not in self.robot:
                raise ValueError("TCP site must belong to robot")
        self.hands = {i for i in self.robot if "hand_" in model.body(i).name}
        if not set().union(*self.fingers.values()) <= self.hands:
            raise ValueError("Pad mappings must be actual Reachy hand bodies")
        self.arm_names = tuple(arm_joint_names)
        if not self.arm_names or len(set(self.arm_names)) != len(self.arm_names):
            raise ValueError("Explicit unique arm joint names required")
        available_arms = {name for name in ARMS if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0}
        if not available_arms <= set(self.arm_names):
            raise ValueError("Both arms' available Reachy joints must be measured without omission")
        self.arm_ids = np.array([model.joint(name).id for name in self.arm_names])
        if any(int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE)
               or not model.jnt_limited[j] or int(model.jnt_bodyid[j]) not in self.robot for j in self.arm_ids):
            raise ValueError("Arm gates require limited scalar robot hinge joints")
        self.qadr, self.vadr = model.jnt_qposadr[self.arm_ids], model.jnt_dofadr[self.arm_ids]
        self.limits = model.jnt_range[self.arm_ids].copy()
        gravity = np.asarray(model.opt.gravity)
        if np.linalg.norm(gravity) == 0:
            raise ValueError("Acquisition lift needs explicit nonzero gravity")
        self.up = -gravity/np.linalg.norm(gravity)
        self.initial_height = float(initial_data.xpos[self.active_id] @ self.up)
        self.initial_time = self.last_time = float(initial_data.time)
        self.require_terminal_contact = bool(require_terminal_contact)
        self.missing = set()
        self.peak_arm = np.abs(initial_data.qvel[self.vadr]).copy()
        self.peak_acceleration = np.zeros(len(self.arm_names))
        self.previous_velocity = initial_data.qvel[self.vadr].copy()
        initial_base_velocity = np.empty(6)
        mujoco.mj_objectVelocity(model, initial_data, mujoco.mjtObj.mjOBJ_XBODY,
                                self.base_id, initial_base_velocity, 1)
        self.peak_base = np.abs(initial_base_velocity[[3, 4, 2]])
        self.min_margin = float(np.min(np.minimum(initial_data.qpos[self.qadr]-self.limits[:, 0],
                                                   self.limits[:, 1]-initial_data.qpos[self.qadr])))
        self.min_clearance = np.inf
        self._clearance(initial_self_clearance_m)
        self.depths = {key: 0. for key in ("hand_object", "robot_environment", "object_environment", "robot_self")}
        self.depth_events = {key: None for key in self.depths}
        self.forbidden_contacts = 0
        self.states = {side: dict(anchor=None, acquired=False, carry_s=0., bilateral_s=0.,
                                translation=0., rotation=0., previous_intent=False, phases=0)
                       for side in self.grasp_hands}
        self.steps = 0
        self.stable_s = self.best_stable_s = self.max_lift = 0.
        self.native_task_final = False
        self.finite = True
        self._consume_contacts(self.sample_contacts(initial_data), initial=True)

    def _clearance(self, value):
        if value is None or not np.isfinite(value):
            self.missing.add("measured_self_clearance")
        else:
            self.min_clearance = min(self.min_clearance, float(value))

    def sample_contacts(self, data):
        """Call immediately after mj_step, before refreshing contact caches."""
        peaks = {key: (0., None) for key in self.depths} if hasattr(self, "depths") else {
            key: (0., None) for key in ("hand_object", "robot_environment", "object_environment", "robot_self")}
        touched = {side: set() for side in self.grasp_hands}
        forbidden = 0
        for index, contact in enumerate(data.contact):
            geoms = [int(contact.geom1), int(contact.geom2)]
            pair = {int(self.model.geom_bodyid[i]) for i in geoms}
            active, robot = bool(pair & self.active), bool(pair & self.robot)
            kind = ("hand_object" if active and pair & self.hands else
                    "object_environment" if active and not robot and not pair <= self.active else
                    "robot_self" if pair <= self.robot else
                    "robot_environment" if robot and not active else None)
            depth = max(0., -float(contact.dist))
            if kind and depth > peaks[kind][0]:
                peaks[kind] = (depth, [self.model.geom(i).name for i in geoms])
            if active and pair & (self.robot-self.hands):
                forbidden += 1
            if active and pair & self.hands:
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, data, index, force)
                if force[0] > THRESHOLDS["positive_contact_force_n"]:
                    for side in touched:
                        touched[side].update(pair & self.fingers[side])
        return dict(time_s=float(data.time), depths={k: v[0] for k, v in peaks.items()},
                    depth_pairs={k: v[1] for k, v in peaks.items()}, nonhand_object_contacts=forbidden,
                    bilateral={side: touched[side] == self.fingers[side] for side in touched})

    def _consume_contacts(self, sample, *, initial=False):
        for kind, depth in sample["depths"].items():
            if depth > self.depths[kind]:
                self.depths[kind] = float(depth)
                self.depth_events[kind] = dict(depth_m=float(depth), time_s=sample["time_s"],
                                              geoms=sample["depth_pairs"][kind], initial=initial)
        self.forbidden_contacts += int(sample["nonhand_object_contacts"])

    def update(self, data, *, contact_sample, grasp_intent, task_satisfied, self_clearance_m=None):
        """Call with synchronized current poses/velocities once per physics step."""
        if task_satisfied is not None and not isinstance(task_satisfied, (bool, np.bool_)):
            raise ValueError("task_satisfied must be the explicit boolean native predicate")
        if any(not isinstance(value, (bool, np.bool_)) for value in grasp_intent.values()):
            raise ValueError("Grasp intent must be explicit per-hand booleans")
        dt = float(data.time)-self.last_time
        if (not np.isclose(dt, self.model.opt.timestep, atol=1e-10, rtol=0)
                or not np.isclose(contact_sample["time_s"], data.time, atol=1e-10, rtol=0)):
            raise ValueError("Gates require every physics step and its matching solved-contact sample")
        self.steps += 1
        self.last_time = float(data.time)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            self.finite = False
            raise ValueError("Nonfinite physical state")
        self._consume_contacts(contact_sample)
        velocity = data.qvel[self.vadr]
        self.peak_arm = np.maximum(self.peak_arm, abs(velocity))
        self.peak_acceleration = np.maximum(self.peak_acceleration, abs(velocity-self.previous_velocity)/dt)
        self.previous_velocity = velocity.copy()
        base = np.empty(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_XBODY, self.base_id, base, 1)
        self.peak_base = np.maximum(self.peak_base, abs(base[[3, 4, 2]]))
        self.min_margin = min(self.min_margin, float(np.min(np.minimum(
            data.qpos[self.qadr]-self.limits[:, 0], self.limits[:, 1]-data.qpos[self.qadr]))))
        self._clearance(self_clearance_m)
        lift = float(data.xpos[self.active_id] @ self.up)-self.initial_height
        self.max_lift = max(self.max_lift, lift)
        object_rotation = data.xmat[self.active_id].reshape(3, 3)
        for side, state in self.states.items():
            if side not in grasp_intent:
                self.missing.add("grasp_intent/"+side)
            closed = bool(grasp_intent.get(side, False))
            if closed and not state["previous_intent"]:
                state["phases"] += 1
                if state["phases"] > 1:
                    self.missing.add("unsupported_multiple_grasp_phases/"+side)
            state["previous_intent"] = closed
            bilateral = bool(contact_sample["bilateral"][side])
            site = self.tcp[side]
            hand_rotation = data.site_xmat[site].reshape(3, 3)
            relative_rotation = hand_rotation.T @ object_rotation
            relative_position = hand_rotation.T @ (data.xpos[self.active_id]-data.site_xpos[site])
            if closed and bilateral and lift > THRESHOLDS["acquisition_lift_m"] and state["anchor"] is None:
                state["anchor"] = (relative_position.copy(), relative_rotation.copy())
                state["acquired"] = True
            if closed and state["anchor"] is not None:
                position0, rotation0 = state["anchor"]
                state["translation"] = max(state["translation"], float(np.linalg.norm(relative_position-position0)))
                state["rotation"] = max(state["rotation"], float(Rotation.from_matrix(rotation0.T @ relative_rotation).magnitude()))
                state["carry_s"] += dt
                state["bilateral_s"] += dt*bilateral
        if task_satisfied is None:
            self.missing.add("native_task_predicate")
        self.native_task_final = bool(task_satisfied)
        object_velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_XBODY, self.active_id, object_velocity, 0)
        stable = (self.native_task_final and np.linalg.norm(object_velocity[3:]) < THRESHOLDS["stable_linear_speed_m_s"]
                  and np.linalg.norm(object_velocity[:3]) < THRESHOLDS["stable_angular_speed_rad_s"]
                  and (not self.require_terminal_contact or all(contact_sample["bilateral"].values())))
        self.stable_s = self.stable_s+dt if stable else 0.
        self.best_stable_s = max(self.best_stable_s, self.stable_s)

    def finish(self, *, expected_steps, expected_final_time_s, actuator_replay=None):
        hands = {side: dict(acquired=state["acquired"],
            bilateral_carry_fraction=state["bilateral_s"]/state["carry_s"] if state["carry_s"] else None,
            grasp_translation_m=state["translation"] if state["acquired"] else None,
            grasp_rotation_rad=state["rotation"] if state["acquired"] else None,
            measured_carry_duration_s=state["carry_s"], phases=state["phases"])
                 for side, state in self.states.items()}
        acquired = all(row["acquired"] for row in hands.values())
        gates = dict(task_stable_final_1s=self.stable_s >= THRESHOLDS["final_stable_s"]-1e-10,
            bilateral_lifted_grasp=acquired,
            carry_contact_95pct=acquired and all(row["bilateral_carry_fraction"] >= .95 for row in hands.values()),
            grasp_translation_3mm=acquired and all(row["grasp_translation_m"] <= .003 for row in hands.values()),
            grasp_rotation_3deg=acquired and all(row["grasp_rotation_rad"] <= THRESHOLDS["grasp_rotation_rad"] for row in hands.values()),
            hand_object_penetration_1mm=self.depths["hand_object"] <= .001,
            robot_environment_penetration_2mm=self.depths["robot_environment"] <= .002,
            object_environment_penetration_2mm=self.depths["object_environment"] <= .002,
            robot_self_penetration_2mm=self.depths["robot_self"] <= .002,
            no_nonhand_object_robot_contact=self.forbidden_contacts == 0,
            actual_arm_speed_1rad_s=bool(np.max(self.peak_arm) <= THRESHOLDS["arm_speed_rad_s"]),
            actual_base_speed_limits=bool(np.all(self.peak_base <= [THRESHOLDS[k] for k in ("base_vx_m_s", "base_vy_m_s", "base_wz_rad_s")])),
            joint_margin_25mrad=self.min_margin >= .025,
            self_clearance_9mm=bool(np.isfinite(self.min_clearance) and self.min_clearance >= .009 and "measured_self_clearance" not in self.missing),
            rollout_complete=bool(self.steps == expected_steps and self.steps > 0 and np.isclose(self.last_time, expected_final_time_s, atol=1e-9, rtol=0)),
            finite_integration=self.finite)
        missing = sorted(self.missing)
        if actuator_replay is None:
            missing.append("independent_actuator_replay")
        if not self.scene_sha256 or (actuator_replay is not None and actuator_replay.get("scene_sha256") != self.scene_sha256):
            missing.append("matching_collector_replay_scene_identity")
        gates["actuator_replay"] = bool(actuator_replay and actuator_replay.get("actuator_replay_pass")
                                        and actuator_replay.get("scene_sha256") == self.scene_sha256)
        complete = not missing
        return dict(protocol="reachy-common-physical-gates-v1", thresholds=dict(THRESHOLDS),
            gates=gates, missing_fields=missing, validation_complete=complete,
            physics_validated=bool(complete and all(gates.values())),
            failure_reasons=[key for key, passed in gates.items() if not passed],
            physical_steps=self.steps, expected_steps=int(expected_steps), final_time_s=self.last_time,
            grasp_hands=hands, native_task_final=self.native_task_final,
            final_stable_s=self.stable_s, best_stable_s=self.best_stable_s, max_lift_m=self.max_lift,
            peak_arm_velocity_rad_s=self.peak_arm.tolist(), arm_joint_names=list(self.arm_names),
            peak_arm_acceleration_rad_s2=self.peak_acceleration.tolist(), acceleration_has_acceptance_threshold=False,
            peak_base_velocity_local_vx_vy_wz=self.peak_base.tolist(), min_joint_margin_rad=self.min_margin,
            min_self_clearance_m=float(self.min_clearance) if np.isfinite(self.min_clearance) else None,
            max_contact_depths_m=dict(self.depths), worst_contact_events=dict(self.depth_events),
            forbidden_nonhand_object_contact_samples=self.forbidden_contacts,
            actuator_replay=actuator_replay,
            scope="Measured robot/task-object manipulation gates; producer must separately bind native source/model identity and task predicate. No object state, geometry, physics parameter or threshold changes.")


def verify_actuator_replay(scene, arrays, *, expected_scene_sha256, expected_assets,
                           active_object_body, active_object_joint, object_pose_key,
                           initial_state=None):
    """Verify pre-action rows, every control interval, and the terminal boundary.

    ``arrays`` is a mapping loaded from the common NPZ/HDF5 archive. All actual
    integration state at row zero must be recorded in ``physics_state``. The
    optional separately recorded initial qpos/qvel/time must match that row.
    """
    scene = Path(scene)
    if sha256(scene) != expected_scene_sha256:
        raise ValueError("Recorded scene identity changed")
    _verify_assets(scene, {"scene_asset_hashes": expected_assets})
    model = mujoco.MjModel.from_xml_path(str(scene))
    _verify_no_object_assistance(model, {"active": {"body": active_object_body, "joint": active_object_joint}})
    keys = ("timestamp", "observation/qpos", "observation/qvel", "physics_state",
            "command/joint_position", "control_interval_s", "terminal/timestamp",
            "terminal/observation/qpos", "terminal/observation/qvel", object_pose_key,
            "terminal/"+object_pose_key)
    if any(key not in arrays for key in keys):
        raise ValueError("Missing exact replay arrays: " + ", ".join(key for key in keys if key not in arrays))
    values = {key: np.asarray(arrays[key]) for key in keys}
    time = values["timestamp"]
    n = len(time)
    size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
    shapes = {"timestamp": (n,), "observation/qpos": (n, model.nq), "observation/qvel": (n, model.nv),
              "physics_state": (n, size), "command/joint_position": (n, model.nu), "control_interval_s": (n,),
              "terminal/timestamp": (), "terminal/observation/qpos": (model.nq,),
              "terminal/observation/qvel": (model.nv,), object_pose_key: (n, 7), "terminal/"+object_pose_key: (7,)}
    if not n or any(values[key].shape != shape or not np.isfinite(values[key]).all() for key, shape in shapes.items()):
        raise ValueError("Nonfinite or shape-inconsistent actual replay arrays")
    durations = values["control_interval_s"]
    count = np.rint(durations/model.opt.timestep).astype(int)
    if (not np.all(count > 0) or not np.allclose(count*model.opt.timestep, durations, atol=1e-12, rtol=0)
            or not np.allclose(np.diff(time), durations[:-1], atol=1e-9, rtol=0)
            or not np.isclose(time[-1]+durations[-1], values["terminal/timestamp"], atol=1e-9, rtol=0)):
        raise ValueError("Recorded intervals do not cover the full increasing clock and terminal boundary")
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, values["physics_state"][0], mujoco.mjtState.mjSTATE_INTEGRATION)
    if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
        raise ValueError("Unrecorded/nonactuator external forces are forbidden")
    if initial_state is not None:
        for key in ("qpos", "qvel"):
            if not np.array_equal(data.__getattribute__(key), initial_state[key]):
                raise ValueError("First recorded state differs from independently saved initial " + key)
        if not np.isclose(data.time, float(initial_state["timestamp"]), atol=1e-10, rtol=0):
            raise ValueError("Initial clock differs from separate reset recording")
    body = model.body(active_object_body).id
    errors = dict(qpos=0., qvel=0., object_pose=0., clock=0.)
    def synchronize():
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_fwdVelocity(model, data)
        mujoco.mj_fwdActuation(model, data)
    def compare(qpos, qvel, pose, timestamp):
        errors["qpos"] = max(errors["qpos"], float(np.max(abs(data.qpos-qpos), initial=0.)))
        errors["qvel"] = max(errors["qvel"], float(np.max(abs(data.qvel-qvel), initial=0.)))
        errors["object_pose"] = max(errors["object_pose"], float(np.max(abs(np.r_[data.xpos[body], data.xquat[body]]-pose))))
        errors["clock"] = max(errors["clock"], abs(float(data.time)-float(timestamp)))
    for row in range(n):
        data.ctrl[:] = values["command/joint_position"][row]
        synchronize()
        compare(values["observation/qpos"][row], values["observation/qvel"][row], values[object_pose_key][row], time[row])
        for _ in range(int(count[row])):
            mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("Nonfinite actuator-only replay")
    synchronize()
    compare(values["terminal/observation/qpos"], values["terminal/observation/qvel"],
            values["terminal/"+object_pose_key], values["terminal/timestamp"])
    return dict(actuator_replay_pass=bool(max(errors.values()) < THRESHOLDS["replay_absolute_tolerance"]),
                max_absolute_errors=errors, control_rows=n, physical_steps=int(count.sum()),
                initial_time_s=float(time[0]), terminal_time_s=float(data.time), scene_sha256=sha256(scene),
                scene_asset_hashes=expected_assets, runtime_mujoco_version=mujoco.__version__,
                object_state_assignments="one recorded integration-state reset only",
                object_assistance=False, compared_terminal=True,
                meaning="Exact actuator replay, separate from task and common physical gate success")
