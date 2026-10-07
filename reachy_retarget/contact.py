"""Replay the fixed robomimic Can ph/demo_0..9 with physical Reachy contact."""

import argparse
import json
import os
import platform
import shutil
import importlib.metadata
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
import h5py
from scipy.spatial.transform import Rotation
from .robot import Robot, ARMS, in_runtime, dispatch
from .episodes import read_episode, pose_to_matrices, matrices_to_pose, interpolate_pose
from .physics import scene, initialize, measured, BASE, GRIPPERS, prepare_robot
from .store import Store, json_write, sha256


def ramp_closure(times, commands, duration):
    """Preserve source event times and opening; soften only each closing edge."""
    if duration < 0 or not np.isfinite(duration):
        raise ValueError("Closing duration must be finite and nonnegative")
    result = np.array(commands, dtype=float, copy=True)
    if not duration:
        return result
    closed = np.asarray(commands) < 0
    for start in np.flatnonzero(closed & ~np.r_[False, closed[:-1]]):
        index = np.flatnonzero((times >= times[start]) & (times < times[start]+duration) & closed)
        x = np.clip((times[index]-times[start])/duration, 0, 1)
        result[index] = 2. + (-.06-2.) * x**3 * (10-15*x+6*x*x)
    return result


def grasp_drift(hand_poses, object_poses, commands, bilateral):
    """Measure object motion relative to the TCP from first lifted pinch to release."""
    hand = pose_to_matrices(np.asarray(hand_poses)[:, 1])
    obj = pose_to_matrices(np.asarray(object_poses))
    relative = np.linalg.inv(hand) @ obj
    eligible = (np.asarray(commands) < 0) & (obj[:, 2, 3] > obj[0, 2, 3] + .03)
    anchors = np.flatnonzero(eligible & np.asarray(bilateral))
    if not len(anchors):
        return {"available": False}, np.full((len(obj), 2), np.nan)
    start = anchors[0]
    releases = np.flatnonzero(np.asarray(commands)[start:] >= 0)
    end = start + releases[0] if len(releases) else len(obj)
    delta = np.linalg.inv(relative[start]) @ relative[start:end]
    drift = np.full((len(obj), 2), np.nan)
    drift[start:end, 0] = np.linalg.norm(delta[:, :3, 3], axis=1)
    drift[start:end, 1] = Rotation.from_matrix(delta[:, :3, :3]).magnitude()
    return {"available": True, "release_observed": bool(len(releases)), "anchor_frame": int(start), "end_frame_exclusive": int(end),
            "max_translation_m": float(np.nanmax(drift[:, 0])),
            "max_rotation_deg": float(np.rad2deg(np.nanmax(drift[:, 1]))),
            "bilateral_contact_fraction": float(np.mean(bilateral[start:end])),
            "definition": "TCP-relative drift from first bilateral contact above 3 cm lift until release command"}, drift


def convert_grasp(
    a, source_xml, root, robot, heading=0.0, time_scale=6.0, depth=0.02, tilt=0.0, grasp_height=0.0
):
    """Keep source EEF motion with a constant source/Reachy grasp attachment.

    Contact attachment comes from the source's first stable lifted interval,
    not from the moving object's simulated pose. All scene objects share one
    rigid placement, no object sizes are changed.
    """
    E = pose_to_matrices(a["hand/right_pose"])
    O = pose_to_matrices(a["objects/Can/pose"])
    command = a["source/action"][:, -1]
    held = np.flatnonzero((O[:, 2, 3] > O[0, 2, 3] + 0.03) & (command > 0))
    if not len(held):
        raise ValueError("No source lift with closed gripper")
    anchor = held[0]
    relative = np.linalg.inv(E[anchor]) @ O[anchor]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", heading, degrees=True).as_matrix()
    T[:3, 3] = [0.50, 0.0, 0.0]
    # The physical finger endpoint midpoint at a 50 mm aperture. Reachy -Z is
    # the approach axis; a documented 20 mm depth moves the can into the pads.
    xml, manifest = scene(root, source_xml, T, T @ O[0])
    import mujoco

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    initialize(model, data, robot, robot.q, manifest["mimics"], 0.63)
    tcp = model.site("r_arm_tip_tcp").id
    R = data.site_xmat[tcp].reshape(3, 3)
    p = data.site_xpos[tcp]
    fingertips = np.array(
        [
            data.site_xpos[model.site(n + "_endpoint").id]
            for n in ["r_hand_distal_link", "r_hand_distal_mimic_link"]
        ]
    )
    offset = R.T @ (fingertips.mean(0) - p)
    offset[2] += depth
    # At the nominal grasp orientation TCP +X points upward. A positive
    # offset lowers the hand on the unchanged can toward its center.
    offset[0] += grasp_height
    # Keep source object heading; positive tilt raises the wrist behind the TCP.
    tool_anchor = np.eye(4)
    tool_anchor[:3, :3] = Rotation.from_euler(
        "ZY", [heading, tilt - 90], degrees=True
    ).as_matrix()
    tool_anchor[:3, 3] = (T @ O[anchor])[:3, 3] - tool_anchor[:3, :3] @ offset
    attachment = np.linalg.inv(T @ E[anchor]) @ tool_anchor
    targets = T @ E @ attachment
    if grasp_height or tilt:
        # Refine the grasp near closure, preserving the already feasible
        # approach and its initial IK branch. Only robot targets are changed.
        nominal_tool = np.eye(4)
        nominal_tool[:3, :3] = Rotation.from_euler("ZY", [heading, -90], degrees=True).as_matrix()
        nominal_offset = offset - np.array([grasp_height, 0., depth-.035])
        nominal_tool[:3, 3] = (T @ O[anchor])[:3, 3] - nominal_tool[:3, :3] @ nominal_offset
        nominal_attachment = np.linalg.inv(T @ E[anchor]) @ nominal_tool
        nominal_targets = T @ E @ nominal_attachment
        close = np.flatnonzero(command > 0)[0]
        weight = np.clip((a["time_s"] - (a["time_s"][close] - .5)) / .5, 0, 1)
        weight = weight**3 * (10 - 15*weight + 6*weight**2)
        delta = np.linalg.inv(nominal_tool) @ tool_anchor
        targets[:, :3, 3] = nominal_targets[:, :3, 3] + np.einsum("nij,nj->ni", nominal_targets[:, :3, :3], weight[:, None]*delta[:3, 3])
        targets[:, :3, :3] = nominal_targets[:, :3, :3] @ Rotation.from_rotvec(weight[:, None]*Rotation.from_matrix(delta[:3, :3]).as_rotvec()).as_matrix()
    # Park the inactive arm in the current base frame; convert its target back
    # to world explicitly during replay. Only the demonstrated arm is tracked.
    goals = np.tile(robot.fk(robot.q), (len(E), 1, 1, 1))
    goals[:, 1] = targets
    t = a["time_s"] * time_scale
    meta = {
        "source_grasp_anchor_frame": int(anchor),
        "source_eef_to_object": relative.tolist(),
        "source_eef_to_reachy_tcp": attachment.tolist(),
        "reachy_fingertip_midpoint_in_tcp_m": offset.tolist(),
        "grasp_depth_m": depth,
        "grasp_tilt_degrees": tilt,
        "world_placement": T.tolist(),
        "time_dilation": time_scale,
        "grasp_height_offset_m": grasp_height,
        "grasp_height_policy": "constant object-center offset in TCP +X; changes hand attachment only",
        "grasp_transition": "quintic attachment blend during 0.5 source seconds before closure" if grasp_height or tilt else "constant attachment",
        "gripper_conversion": "source OSC -1=open -> 2.0rad, +1=close -> -0.06rad; derived Reachy finger command",
        "source_gripper_timing": "zero-order hold at source action timestamps",
        "extension": "hold final source hand target with open gripper for 3 seconds",
    }
    meta["inactive_arm"] = (
        "hold initial left arm pose relative to the moving base; no demonstrated left hand label"
    )
    return t, goals, command, T, xml, manifest, meta


def trial(
    store, row, label="baseline", render=True, time_scale=6.0, heading=0.0, depth=0.02, grasp_height=0.0, tilt=0.0, close_duration=0.0
):
    import mujoco

    a, m = read_episode(store.root / row["path"])
    robot = Robot(store.root)
    t, g, grip, T, xml, manifest, conversion = convert_grasp(
        a, m["model_xml"], store.root, robot, heading, time_scale, depth, tilt=tilt, grasp_height=grasp_height
    )
    out = (
        store.root / "runs/contact" / label / row["source_sequence"].rsplit("/", 1)[-1]
    )
    if (out / "result.json").exists() or (out / "replay.h5").exists():
        raise FileExistsError(f"Preserve previous trial; choose a new label: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene.xml").write_text(xml)
    for source_name in ("contact.py", "robot.py", "physics.py"):
        shutil.copy2(Path(__file__).with_name(source_name), out / source_name)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    # Initialization is the only state assignment, before the simulation starts.
    # Deterministic multistart resolves elbow/wrist IK branches. Prefer a
    # stationary initial base; only the task controller moves it afterwards.
    original_active = robot.active.copy()
    robot.active = robot.arm_v[7:]
    rng = np.random.default_rng(5)
    candidates = []
    for i in range(24):
        seed = robot.q.copy()
        if i:
            ids = robot.arm_ids[7:]
            seed[ids] = rng.uniform(
                robot.r.model.lowerPositionLimit[ids] + 0.04,
                robot.r.model.upperPositionLimit[ids] - 0.04,
            )
        solved, p_err, r_err = robot.ik(g[0], seed, active_hands=(1,), iterations=300)
        gap, margin, _ = robot.r.geometry(solved)
        candidates.append(
            (p_err + r_err * 0.3 + (max(0, 0.015 - gap) * 10), solved, p_err, r_err)
        )
        if p_err < 0.001 and r_err < 0.01 and gap > 0.015:
            break
    _, q, pe, re = min(candidates, key=lambda x: x[0])
    robot.active = original_active
    g[:, 0] = robot.fk(q)[0]
    initialize(model, data, robot, q, manifest["mimics"], 2.0)
    initial_can = data.xpos[model.body("Can_main").id].copy()
    # Exact FK agreement of independently converted MuJoCo and Pinocchio models.
    initial_fk = robot.fk(q)
    fk_error = max(
        np.linalg.norm(
            data.site_xpos[model.site(s + "_arm_tip_tcp").id] - initial_fk[h, :3, 3]
        )
        for h, s in enumerate(("l", "r"))
    )
    if fk_error > 1e-6:
        raise ValueError(f"URDF conversion FK mismatch {fk_error}")
    seconds = np.arange(0, t[-1] + 3.001, 0.01)
    goals = np.stack(
        [
            pose_to_matrices(interpolate_pose(t, matrices_to_pose(g[:, h]), seconds))
            for h in range(2)
        ],
        axis=1,
    )
    target_velocity = np.zeros((len(seconds), 2, 6))
    target_velocity[:-1, 1, :3] = np.diff(goals[:, 1, :3, 3], axis=0) / .01
    target_velocity[:-1, 1, 3:] = Rotation.from_matrix(
        goals[1:, 1, :3, :3] @ goals[:-1, 1, :3, :3].transpose(0, 2, 1)).as_rotvec() / .01
    conversion["target_motion_contract"] = ("left hand body-attached; right world trajectory with explicit velocity"
                                             if hasattr(robot.r, "camera") else "historical base-frame API")
    indices = np.clip(np.searchsorted(t, seconds, side="right") - 1, 0, len(t) - 1)
    commands = np.where(grip[indices] > 0, -0.06, 2.0)
    commands[seconds > t[-1]] = 2.0
    commands = ramp_closure(seconds, commands, close_duration)
    conversion["closing_ramp_s"] = close_duration
    conversion["closing_ramp_policy"] = "quintic command transition after source close event; opening event unchanged"
    canbody = model.body("Can_main").id
    canjoint = model.joint("Can_joint0")
    canids = set(np.where(model.geom_bodyid == canbody)[0])
    robot_bodies = {
        model.body(i).name
        for i in range(model.nbody)
        if model.body(i).name not in ("world", "Can_main", "bin1", "bin2")
    }
    handids = {
        i
        for i in range(model.ngeom)
        if "hand_" in model.body(model.geom_bodyid[i]).name
    }
    renderer = None
    writer = None
    if render:
        import imageio.v2 as imageio

        renderer = mujoco.Renderer(model, height=720, width=960)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.35, 0, 0.85]
        camera.distance = 2.5
        camera.azimuth = 140
        camera.elevation = -22
        writer = imageio.get_writer(
            out / "replay.mp4", fps=20, codec="libx264", macro_block_size=16
        )
    records = []
    joint_records = []
    base_records = []
    target_records = []
    ctrl_records = []
    poses = []
    contacts = []
    errors = []
    clearances = []
    margins = []
    forbidden = []
    qpos = []
    qvel_records = []
    full_controls = []
    hand_poses = []
    stable = 0.0
    maxstable = 0.0
    maxlift = 0.0
    grasp_contact = False
    reason = None
    peak_arm_velocity = np.zeros(14)
    peak_base_velocity = np.zeros(3)
    physics_min_gap = float("inf")
    physics_min_margin = float("inf")
    self_penetration_count = 0
    hand_can_depths = []
    bilateral_contacts = []
    max_hand_can_penetration = 0.0
    forbidden_can_penetrations = 0
    bilateral_grasp_observed = False
    fingertip_bodies = {"r_hand_distal_link", "r_hand_distal_mimic_link"}
    arm_dofs = np.array([model.joint(n).dofadr[0] for n in ARMS])
    base_dofs = np.array([model.joint(n).dofadr[0] for n in BASE])
    commanded = q.copy()
    bin_local = np.array([0.1, 0.28, 0.8])
    bin_world = (T @ np.r_[bin_local, 1])[:3]
    try:
        for step, sec in enumerate(seconds):
            measured_q = measured(model, data, robot)
            base = robot.base(measured_q)
            base_T = np.eye(4)
            base_T[:3, :3] = Rotation.from_euler("z", base[2]).as_matrix()
            base_T[:2, 3] = base[:2]
            goals[step, 0] = base_T @ g[0, 0]
            motion = dict(body_attached_hands=(0,), world_goal_velocity=target_velocity[step]) if hasattr(robot.r, "camera") else {}
            velocity, info = robot.control(measured_q, goals[step], **motion)
            # Servo targets integrate the controller's commanded velocity. Actual
            # state is never replaced by those targets or by a source trajectory.
            next_measured = robot.integrate(measured_q, velocity, 0.01)
            delta_base = robot.base(next_measured) - robot.base(measured_q)
            delta_base[2] = np.arctan2(np.sin(delta_base[2]), np.cos(delta_base[2]))
            commanded = robot.pack(
                commanded[robot.arm_ids] + velocity[robot.arm_v] * 0.01,
                robot.base(commanded) + delta_base,
            )
            joint_ids = robot.arm_ids
            commanded[joint_ids] = np.clip(
                commanded[joint_ids],
                robot.r.model.lowerPositionLimit[joint_ids] + 0.06,
                robot.r.model.upperPositionLimit[joint_ids] - 0.06,
            )
            # Anti-windup bounds the servo reference around the measured robot,
            # without changing the measured state or the dynamic object.
            commanded[joint_ids] = np.clip(
                commanded[joint_ids],
                measured_q[joint_ids] - 0.15,
                measured_q[joint_ids] + 0.15,
            )
            cmd_base = robot.base(commanded)
            for n, v in zip(ARMS, commanded[robot.arm_ids]):
                data.ctrl[model.actuator(n).id] = v
            for n, v in zip(BASE, cmd_base):
                data.ctrl[model.actuator(n).id] = v
            data.ctrl[model.actuator("r_hand_finger").id] = commands[step]
            hand_contact = False
            bad = 0
            maxforce = 0.0
            interval_depth = 0.0
            bilateral_contact = False
            for _ in range(5):
                mujoco.mj_step(model, data)
                touching_fingers = set()
                peak_arm_velocity = np.maximum(
                    peak_arm_velocity, np.abs(data.qvel[arm_dofs])
                )
                base_v = data.qvel[base_dofs].copy()
                yaw = data.qpos[model.joint("base_yaw").qposadr[0]]
                base_v[:2] = (
                    Rotation.from_euler("z", -yaw).as_matrix()[:2, :2] @ base_v[:2]
                )
                peak_base_velocity = np.maximum(peak_base_velocity, np.abs(base_v))
                physics_q = measured(model, data, robot)
                gap, margin, _ = robot.r.geometry(physics_q)
                physics_min_gap = min(physics_min_gap, gap)
                physics_min_margin = min(physics_min_margin, margin)
                for ci in range(data.ncon):
                    c = data.contact[ci]
                    pair = {c.geom1, c.geom2}
                    if pair & canids and pair & handids:
                        hand_contact = True
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, ci, force)
                        maxforce = max(maxforce, float(np.linalg.norm(force[:3])))
                    b1 = model.body(model.geom_bodyid[c.geom1]).name
                    b2 = model.body(model.geom_bodyid[c.geom2]).name
                    if pair & canids and pair & handids:
                        interval_depth = max(interval_depth, max(0., -float(c.dist)))
                        if np.linalg.norm(force[:3]) > 1e-6:
                            touching_fingers.update({b1, b2} & fingertip_bodies)
                        if c.dist < -0.001 and not ({b1, b2} & fingertip_bodies):
                            forbidden_can_penetrations += 1
                    if c.dist < -0.002 and b1 in robot_bodies and b2 in robot_bodies:
                        self_penetration_count += 1
                    if c.dist < -0.002 and (
                        (b1 in robot_bodies and b2 in ("bin1", "bin2"))
                        or (b2 in robot_bodies and b1 in ("bin1", "bin2"))
                    ):
                        bad += 1
                bilateral_contact |= touching_fingers == fingertip_bodies
            # mj_step integrates qpos last; refresh derived body poses at the
            # saved interval-end timestamp before recording or rendering them.
            mujoco.mj_forward(model, data)
            qm = measured(model, data, robot)
            fk = robot.fk(qm)
            obj = data.xpos[canbody].copy()
            quat = data.xquat[canbody].copy()
            local = T[:3, :3].T @ (obj - T[:3, 3])
            v = data.qvel[canjoint.dofadr[0] : canjoint.dofadr[0] + 6]
            lift = obj[2] - initial_can[2]
            maxlift = max(maxlift, lift)
            grasp_contact |= hand_contact and commands[step] < 0
            bilateral_grasp_observed |= bilateral_contact and commands[step] < 0 and lift >= 0.03
            max_hand_can_penetration = max(max_hand_can_penetration, interval_depth)
            hand_can_depths.append(interval_depth)
            bilateral_contacts.append(bilateral_contact)
            opened = float(data.qpos[model.joint("r_hand_finger").qposadr[0]]) > 1.7
            inside = (
                abs(local[0] - 0.1) < 0.175
                and abs(local[1] - 0.28) < 0.225
                and 0.84 < local[2] < 0.96
            )
            settled = (
                inside
                and opened
                and not hand_contact
                and np.linalg.norm(v[:3]) < 0.02
                and np.linalg.norm(v[3:]) < 0.2
            )
            stable = stable + 0.01 if settled else 0.0
            maxstable = max(maxstable, stable)
            records.append(float(data.time))
            joint_records.append(qm[robot.arm_ids])
            base_records.append(robot.base(qm))
            target_records.append(matrices_to_pose(goals[step]))
            ctrl_records.append(commanded[robot.arm_ids])
            poses.append(np.r_[obj, quat])
            contacts.append([hand_contact, maxforce, opened])
            errors.append(np.linalg.norm(fk[1, :3, 3] - goals[step, 1, :3, 3]))
            clearances.append(info["clearance"])
            margins.append(info["joint_margin"])
            forbidden.append(bad)
            qpos.append(data.qpos.copy())
            qvel_records.append(data.qvel.copy())
            full_controls.append(data.ctrl.copy())
            hand_poses.append(matrices_to_pose(fk))
            if renderer is not None and step % 5 == 0:
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
            if not np.isfinite(data.qpos).all():
                raise RuntimeError("Non-finite simulator state")
            if step % 500 == 0:
                print(
                    out.name,
                    f"{sec:.1f}s lift={maxlift:.3f} stable={maxstable:.2f} err={errors[-1]:.3f} contact={hand_contact}",
                    flush=True,
                )
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
    # Collision and source-controller limits remain independent pass conditions.
    speed_pass = bool(
        np.all(peak_arm_velocity <= 1.0 + 1e-3)
        and np.all(peak_base_velocity <= np.array([0.61, 0.61, np.deg2rad(114)]) + 1e-3)
    )
    drift_report, drift_series = grasp_drift(hand_poses, poses, commands[:len(records)], bilateral_contacts) if records else ({"available": False}, [])
    stable_grasp = bool(drift_report["available"] and drift_report["release_observed"]
                        and drift_report["max_translation_m"] <= .003
                        and drift_report["max_rotation_deg"] <= 3.
                        and drift_report["bilateral_contact_fraction"] >= .95)
    task_success = bool(
        maxlift >= 0.08 and stable >= 1.0 and grasp_contact and not reason
    )
    success = bool(
        task_success
        and speed_pass
        and sum(forbidden) == 0
        and self_penetration_count == 0
        and physics_min_gap >= 0.009
        and physics_min_margin >= 0.025
        and max_hand_can_penetration <= 0.001
        and forbidden_can_penetrations == 0
        and bilateral_grasp_observed
        and stable_grasp
    )
    reasons = []
    if reason:
        reasons.append(reason)
    if maxlift < 0.08:
        reasons.append("object_not_lifted_8cm")
    if not grasp_contact:
        reasons.append("no_closed_gripper_object_contact")
    if stable < 1.0:
        reasons.append("final_placement_not_stable_open_no_hand_contact_for_1s")
    if not speed_pass:
        reasons.append("actual_physics_velocity_limit_exceeded")
    if self_penetration_count:
        reasons.append("robot_self_penetration")
    if sum(forbidden):
        reasons.append("robot_environment_penetration")
    if physics_min_gap < 0.009:
        reasons.append("self_clearance_below_9mm")
    if physics_min_margin < 0.025:
        reasons.append("joint_margin_below_25mrad")
    if max_hand_can_penetration > 0.001:
        reasons.append("hand_can_penetration_above_1mm")
    if forbidden_can_penetrations:
        reasons.append("non_fingertip_can_penetration_above_1mm")
    if not bilateral_grasp_observed:
        reasons.append("no_bilateral_finger_contact_during_lift")
    if not stable_grasp:
        reasons.append("grasp_stability_criteria_not_met")
    report = {
        "source_episode": row["id"],
        "source_id": row["source_id"],
        "source_sequence": row["source_sequence"],
        "split": row["split"],
        "status": "physical_pass" if success else "physical_fail",
        "success": success,
        "failure_reasons": reasons,
        "grasp_drift": drift_report,
        "grasp_stability_pass": stable_grasp,
        "max_lift_m": maxlift,
        "max_stable_placement_s": maxstable,
        "grasp_contact_observed": bool(grasp_contact),
        "bilateral_grasp_contact_observed": bool(bilateral_grasp_observed),
        "max_hand_can_penetration_m": max_hand_can_penetration,
        "non_fingertip_can_penetration_contacts": forbidden_can_penetrations,
        "contact_task_success": task_success,
        "final_stable_placement_s": stable,
        "physics_peak_arm_velocity_rad_s": peak_arm_velocity.tolist(),
        "physics_peak_base_velocity_m_s_rad_s": peak_base_velocity.tolist(),
        "physics_min_clearance_m": physics_min_gap,
        "physics_min_joint_margin_rad": physics_min_margin,
        "robot_self_penetration_substeps": self_penetration_count,
        "actual_velocity_pass": speed_pass,
        "robot_environment_penetration_substeps": int(sum(forbidden)),
        "max_tracking_position_error_m": max(errors, default=None),
        "min_self_clearance_m": min(clearances, default=None),
        "min_joint_margin_rad": min(margins, default=None),
        "initial_ik_error_m": pe,
        "initial_ik_rotation_error_rad": re,
        "fk_conversion_error_m": fk_error,
        "object_mass_kg": float(model.body_mass[canbody]),
        "object_inertia_kg_m2": model.body_inertia[canbody].tolist(),
        "conversion": conversion,
        "physics": manifest,
        "controller_identity": robot.identity,
        "controller_backend": robot.backend,
        "controller_target_frame": robot.target_frame,
        "runtime": {"platform": platform.platform(), "packages": {name: importlib.metadata.version(name)
                    for name in ("mujoco", "numpy", "pin", "osqp")}},
        "implementation_sha256": {name: sha256(out / name) for name in ("contact.py", "robot.py", "physics.py")},
        "scene_sha256": sha256(out / "scene.xml"),
        "source_hdf5_sha256": sha256(store.root / row["path"]),
        "physics_hz": 500,
        "control_hz": 100,
        "object_state_assignment": "reset only; dynamic free joint for every physics step",
        "criteria": {
            "lift_m": 0.08,
            "stable_s": 1.0,
            "translation_speed_max_m_s": 0.02,
            "angular_speed_max_rad_s": 0.2,
            "gripper_open_min_rad": 1.7,
            "hand_can_penetration_max_m": 0.001,
            "bilateral_finger_contact_required_during_lift": True,
            "grasp_translation_drift_max_m": .003,
            "grasp_rotation_drift_max_deg": 3.,
            "grasp_bilateral_contact_fraction_min": .95,
            "grasp_release_observed_required": True,
        },
    }
    with h5py.File(out / "replay.h5", "w") as f:
        for k, v in {
            "time_s": records,
            "target/hand_pose_world": target_records,
            "command/joint_position": ctrl_records,
            "command/gripper_position": commands[: len(records)],
            "simulation/joint_position": joint_records,
            "simulation/base_xyyaw": base_records,
            "simulation/Can_pose": poses,
            "simulation/contacts_hand_force_open": contacts,
            "simulation/qpos": qpos,
            "simulation/qvel": qvel_records,
            "simulation/actuator_control": full_controls,
            "simulation/hand_pose_world": hand_poses,
            "metrics/hand_position_error": errors,
            "metrics/self_clearance": clearances,
            "metrics/environment_contacts": forbidden,
            "metrics/hand_can_penetration_m": hand_can_depths,
            "metrics/bilateral_finger_contact": bilateral_contacts,
            "metrics/grasp_drift_m_rad": drift_series,
        }.items():
            f.create_dataset(k, data=v, compression="gzip")
        f["command/time_s"] = seconds[: len(records)]
        object_reference = matrices_to_pose(T @ pose_to_matrices(a["objects/Can/pose"]))
        f["target/Can_reference_pose_world"] = interpolate_pose(
            t, object_reference, seconds[: len(records)]
        )
        f["goal/Can_final_source_pose_world"] = object_reference[-1]
        f["goal/container_center_world"] = bin_world
        f["simulation/joint_names"] = np.asarray(ARMS, dtype=h5py.string_dtype())
        f["simulation/model_joint_names"] = np.asarray([model.joint(i).name for i in range(model.njnt)], dtype=h5py.string_dtype())
        f["simulation/model_jnt_qposadr"] = model.jnt_qposadr
        f.attrs["object_reference_role"] = (
            "reference and goal only; never applied as a simulator state or force"
        )
        f.attrs["units"] = (
            "m, rad, s; poses xyz+wxyz; command at interval start, simulation at interval end"
        )
        f.attrs["metadata_json"] = json.dumps(report)
    json_write(out / "result.json", report)
    print(
        {
            k: v
            for k, v in report.items()
            if k not in ("conversion", "physics", "controller_identity")
        },
        flush=True,
    )
    return report


def run(
    store,
    sources=None,
    limit=10,
    label="baseline",
    render=True,
    time_scale=6.0,
    heading=0.0,
    depth=0.02,
    grasp_height=0.0,
    tilt=0.0,
    close_duration=0.0,
):
    if not in_runtime():
        args = [
            "--root",
            str(store.root),
            "--limit",
            str(limit),
            "--label",
            label,
            "--time-scale",
            str(time_scale),
            "--heading",
            str(heading),
            "--depth",
            str(depth),
            "--grasp-height",
            str(grasp_height),
            "--tilt",
            str(tilt),
            "--close-duration",
            str(close_duration),
        ]
        if not render:
            args += ["--no-render"]
        return dispatch("reachy_retarget.contact", args)
    rows = [
        r
        for r in store.rows("episodes")
        if r["source_id"] == "hf__robomimic__robomimic_datasets"
        and r["source_sequence"].startswith("v1.5/can/ph/demo_")
    ]
    rows.sort(key=lambda r: int(r["source_sequence"].rsplit("_", 1)[-1]))
    reports = []
    for row in rows[:limit]:
        reports.append(trial(store, row, label, render, time_scale, heading, depth, grasp_height, tilt, close_duration))
        json_write(store.root / "runs/contact" / label / "summary.json", reports)
    return reports


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--label", default="baseline")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--time-scale", type=float, default=6.0)
    p.add_argument("--heading", type=float, default=0.0)
    p.add_argument("--depth", type=float, default=0.02)
    p.add_argument("--grasp-height", type=float, default=0.0)
    p.add_argument("--tilt", type=float, default=0.0)
    p.add_argument("--close-duration", type=float, default=0.0)
    a = p.parse_args()
    run(
        Store(Path(a.root).resolve()),
        limit=a.limit,
        label=a.label,
        render=not a.no_render,
        time_scale=a.time_scale,
        heading=a.heading,
        depth=a.depth,
        grasp_height=a.grasp_height,
        tilt=a.tilt,
        close_duration=a.close_duration,
    )
