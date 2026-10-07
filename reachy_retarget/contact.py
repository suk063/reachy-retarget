"""Replay the fixed robomimic Can ph/demo_0..9 with physical Reachy contact."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
import h5py
from scipy.spatial.transform import Rotation
from .robot import Robot, ARMS, in_runtime, dispatch
from .episodes import read_episode, pose_to_matrices, matrices_to_pose, interpolate_pose
from .physics import scene, initialize, measured, BASE, GRIPPERS, prepare_robot
from .store import Store, json_write, sha256


def convert_grasp(
    a, source_xml, root, robot, heading=0.0, time_scale=6.0, depth=0.02, tilt=0.0
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
    # Keep source object heading at the anchor, while making tool -Z downward.
    tool_anchor = np.eye(4)
    tool_anchor[:3, :3] = Rotation.from_euler(
        "ZY", [heading, tilt - 90], degrees=True
    ).as_matrix()
    tool_anchor[:3, 3] = (T @ O[anchor])[:3, 3] - tool_anchor[:3, :3] @ offset
    attachment = np.linalg.inv(T @ E[anchor]) @ tool_anchor
    targets = T @ E @ attachment
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
        "gripper_conversion": "source OSC -1=open -> 2.0rad, +1=close -> -0.06rad; derived Reachy finger command",
        "source_gripper_timing": "zero-order hold at source action timestamps",
        "extension": "hold final source hand target with open gripper for 3 seconds",
    }
    meta["inactive_arm"] = (
        "hold initial left arm pose relative to the moving base; no demonstrated left hand label"
    )
    return t, goals, command, T, xml, manifest, meta


def trial(
    store, row, label="baseline", render=True, time_scale=6.0, heading=0.0, depth=0.02
):
    import mujoco

    a, m = read_episode(store.root / row["path"])
    robot = Robot(store.root)
    t, g, grip, T, xml, manifest, conversion = convert_grasp(
        a, m["model_xml"], store.root, robot, heading, time_scale, depth
    )
    out = (
        store.root / "runs/contact" / label / row["source_sequence"].rsplit("/", 1)[-1]
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene.xml").write_text(xml)
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
    robot.active = original_active
    _, q, pe, re = min(candidates, key=lambda x: x[0])
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
    indices = np.clip(np.searchsorted(t, seconds, side="right") - 1, 0, len(t) - 1)
    commands = np.where(grip[indices] > 0, -0.06, 2.0)
    commands[seconds > t[-1]] = 2.0
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
            velocity, info = robot.control(measured_q, goals[step])
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
            for _ in range(5):
                mujoco.mj_step(model, data)
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
                    if c.dist < -0.002 and b1 in robot_bodies and b2 in robot_bodies:
                        self_penetration_count += 1
                    if c.dist < -0.002 and (
                        (b1 in robot_bodies and b2 in ("bin1", "bin2"))
                        or (b2 in robot_bodies and b1 in ("bin1", "bin2"))
                    ):
                        bad += 1
            qm = measured(model, data, robot)
            fk = robot.fk(qm)
            obj = data.xpos[canbody].copy()
            quat = data.xquat[canbody].copy()
            local = T[:3, :3].T @ (obj - T[:3, 3])
            v = data.qvel[canjoint.dofadr[0] : canjoint.dofadr[0] + 6]
            lift = obj[2] - initial_can[2]
            maxlift = max(maxlift, lift)
            grasp_contact |= hand_contact and commands[step] < 0
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
    report = {
        "source_episode": row["id"],
        "source_sequence": row["source_sequence"],
        "split": row["split"],
        "status": "physical_pass" if success else "physical_fail",
        "success": success,
        "failure_reasons": reasons,
        "max_lift_m": maxlift,
        "max_stable_placement_s": maxstable,
        "grasp_contact_observed": bool(grasp_contact),
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
            "metrics/hand_position_error": errors,
            "metrics/self_clearance": clearances,
            "metrics/environment_contacts": forbidden,
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
        reports.append(trial(store, row, label, render, time_scale, heading, depth))
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
    a = p.parse_args()
    run(
        Store(a.root),
        limit=a.limit,
        label=a.label,
        render=not a.no_render,
        time_scale=a.time_scale,
        heading=a.heading,
        depth=a.depth,
    )
