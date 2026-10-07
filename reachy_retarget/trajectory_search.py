"""Budgeted incremental-horizon actuator optimization in unassisted MuJoCo.

Inspired by DynaRetarget (arXiv:2602.06827v3), not a reproduction of its results.
Every sampled prefix starts at the SAME initial physical state. Earlier knots
remain adjustable as the horizon grows. No reference object state is injected.
All sampled controls, states and scores are saved, including unsuccessful ones.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .store import json_write, sha256, now


@dataclass(frozen=True)
class SearchConfig:
    samples: int = 8
    iterations: int = 2
    elite: int = 2
    knot_s: float = 1.
    horizon_increment_s: float = 4.
    seed: int = 2026
    threads: int = 4
    std_fraction: float = .5
    minimum_std_fraction: float = .08

    def validate(self):
        if not (self.samples >= 3 and 1 <= self.elite < self.samples and self.iterations >= 1
                and self.knot_s > 0 and self.horizon_increment_s > 0 and self.threads >= 1
                and 0 < self.minimum_std_fraction <= self.std_fraction <= 1):
            raise ValueError("Invalid global trajectory-search budget")


def interpolate_residual(knots, knot_times, times):
    return np.stack([np.interp(times, knot_times, column) for column in np.asarray(knots).T], axis=1)


def quaternion_error(actual, target):
    dot = np.sum(actual * target, axis=-1)
    return 2 * np.arccos(np.clip(np.abs(dot), 0., 1.))


def dense_cost(model, states, sensors, controls, nominal, object_goals, hand_goals,
               closed, object_qpos, robot_qpos, robot_dof):
    """A fixed objective over object outcomes, contacts, tracking and limits.

    This score ranks proposals only. The full independent validator decides
    success and retains tighter penetration/drift/task requirements.
    """
    q = states[..., 1:1+model.nq]
    v = states[..., 1+model.nq:1+model.nq+model.nv]
    obj = q[..., object_qpos:object_qpos+7]
    pos = np.sum(((obj[..., :3]-object_goals[..., :3])/.03)**2, axis=-1)
    rot = (quaternion_error(obj[..., 3:], object_goals[..., 3:])/.3)**2
    terms = {"object_position": pos.mean(1), "object_rotation": .2*rot.mean(1),
             "terminal_object": 2*(pos[:, -1]+.2*rot[:, -1])}

    def sensor(name):
        sid = model.sensor(name).id
        a, n = model.sensor_adr[sid], model.sensor_dim[sid]
        return sensors[..., a:a+n]

    tcp = sensor("retarget_tcp_pos")
    shape = obj.shape[:-1]
    actual_rot = Rotation.from_quat(obj[..., 3:][..., [1, 2, 3, 0]].reshape(-1, 4))
    local = actual_rot.inv().apply((tcp-obj[..., :3]).reshape(-1, 3)).reshape(*shape, 3)
    desired_rot = Rotation.from_quat(object_goals[:, 3:][:, [1, 2, 3, 0]])
    desired_local = desired_rot.inv().apply(hand_goals[:, :3]-object_goals[:, :3])
    # Soft contact alignment in the object's frame, not a rigid attachment.
    terms["object_frame_contact"] = (.3*np.sum(((local-desired_local)/.03)**2, axis=-1)*closed).mean(1)
    forces = np.stack([np.linalg.norm(sensor(f"retarget_contact_{i}"), axis=-1) for i in range(2)], axis=-1)
    # Only penalize missing force when the reference is above its initial height.
    carrying = closed & (object_goals[:, 2] > object_goals[0, 2]+.015)
    terms["missing_bilateral_force"] = (3*np.clip(1-forces.min(-1)/.1, 0, 1)*carrying).mean(1)
    terms["self_contact"] = 2*np.mean(np.linalg.norm(sensor("retarget_self_contact"), axis=-1) > .01, axis=1)
    for category in ("object", "robot"):
        depth = np.maximum(-sensor(f"retarget_{category}_depth")[..., 0]-.002, 0)
        terms[category+"_penetration"] = 20*np.max((depth/.002)**2, axis=1)
    tcp_quat = sensor("retarget_tcp_quat")
    hand_rot = Rotation.from_quat(tcp_quat[..., [1,2,3,0]].reshape(-1,4))
    relative_quat = (actual_rot.inv()*hand_rot).as_quat().reshape(*shape,4)
    drift_cost = np.zeros(len(states))
    for sample in range(len(states)):
        anchors = np.flatnonzero(carrying & (forces[sample].min(-1) > 1e-6))
        if not len(anchors):
            drift_cost[sample] = 20. if np.any(carrying) else 0.
            continue
        anchor = anchors[0]
        window = (np.arange(len(closed)) >= anchor) & closed
        displacement = np.linalg.norm(local[sample]-local[sample,anchor],axis=-1)
        angle = quaternion_error(relative_quat[sample],relative_quat[sample,anchor])
        drift_cost[sample] = 20*(np.max(np.maximum(displacement[window]/.003-1,0)**2)
                                 + np.max(np.maximum(angle[window]/np.deg2rad(3)-1,0)**2))
    terms["grasp_drift"] = drift_cost
    robot_velocity = v[..., robot_dof]
    caps = np.r_[.6, .6, np.deg2rad(114), np.ones(len(robot_dof)-3)]
    terms["speed"] = np.mean(np.maximum(np.abs(robot_velocity)/caps-1, 0)**2, axis=(1, 2))
    joints = np.array([int(np.flatnonzero(model.jnt_qposadr == address)[0]) for address in robot_qpos])
    selected = model.jnt_limited[joints].astype(bool)
    if np.any(selected):
        joint_q = q[..., robot_qpos[selected]]
        ranges = model.jnt_range[joints[selected]]
        violation = np.maximum(ranges[:, 0]+.025-joint_q, 0)+np.maximum(joint_q-ranges[:, 1]+.025, 0)
        terms["joint_margin"] = 10*np.mean((violation/.025)**2, axis=(1, 2))
    terms["control_deviation"] = .02*np.mean((controls-nominal)**2, axis=(1, 2))
    terms["control_smoothness"] = .002*np.mean(np.diff(controls, axis=1)**2, axis=(1, 2)) if controls.shape[1] > 1 else np.zeros(len(states))
    score = sum(terms.values())
    valid = np.isfinite(states).all(axis=(1, 2)) & np.isfinite(score)
    valid &= np.all(np.diff(states[..., 0], axis=1) > 0, axis=1)
    return np.where(valid, score, 1e12), terms


def optimize(attempt, output, config=SearchConfig()):
    """Refine a recorded common-policy seed; retain each optimization sample."""
    import mujoco
    import mujoco.rollout
    from .physics import BASE, ARMS

    config.validate()
    attempt, output = Path(attempt), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = json.loads((attempt/"result.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(attempt/"scene.xml"))
    with h5py.File(attempt/"plan.h5") as f:
        times, reference, grip = f["time_s"][:], f["reference"][:], f["gripper"][:]
        objects, hands = f["object_goals"][:], f["hand_goals"][:]
        q0, v0, ctrl0 = f["initial/qpos"][:], f["initial/qvel"][:], f["initial/ctrl"][:]
    if len(times) < 2 or not np.allclose(np.diff(times), .01, rtol=0, atol=1e-8):
        raise ValueError("The current search requires the recorded 100 Hz command clock")
    substeps = int(round(.01/model.opt.timestep))
    if substeps < 1 or not np.isclose(substeps*model.opt.timestep, .01):
        raise ValueError("Command period must be an integer number of physical steps")
    joints = [model.joint(name).id for name in (*BASE, *ARMS)]
    robot_qpos, robot_dof = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    actuator_ids = np.array([model.actuator(name).id for name in (*BASE, *ARMS)])
    gripper = model.actuator("r_hand_finger").id
    nominal = np.tile(ctrl0, (len(times), 1))
    nominal[:, actuator_ids] = reference
    nominal[:, gripper] = grip
    # The seed's feedback commands are valid controls, including unsuccessful ones.
    if (attempt/"replay.h5").is_file():
        with h5py.File(attempt/"replay.h5") as f:
            recorded = f["simulation/actuator_control"][:]
        nominal[:len(recorded)] = recorded
    active = np.r_[actuator_ids[:3], actuator_ids[10:], gripper].astype(int)
    radius = np.r_[.04, .04, .15, np.full(7, .12), .3]
    knot_times = np.unique(np.r_[np.arange(0, times[-1], config.knot_s), times[-1]])
    mean = np.zeros((len(knot_times), len(active)))
    std = np.tile(radius*config.std_fraction, (len(knot_times), 1))
    rng = np.random.default_rng(config.seed)
    data = [mujoco.MjData(model) for _ in range(config.threads)]
    data[0].qpos[:] = q0; data[0].qvel[:] = v0; data[0].ctrl[:] = ctrl0
    initial = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(model, data[0], initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    oid = report["plan"]["object_id"]
    joint = report["physics"]["objects"][oid]["joint"]
    object_qpos = int(model.joint(joint).qposadr[0])
    if model.jnt_type[model.joint(joint).id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("This search version requires a free active rigid object")
    summary = {"created": now(), "method": "budgeted incremental-horizon CEM",
               "reference": "https://arxiv.org/html/2602.06827v3", "config": asdict(config),
               "scene_sha256": sha256(attempt/"scene.xml"), "seed_result_sha256": sha256(attempt/"result.json"),
               "object_assignment": "initial state only, for each independent candidate",
               "all_previous_knots_reoptimized": True, "samples": [],
               "reproduction": False, "physics_validated": False}
    np.savez_compressed(output/"inputs.npz", initial_state=initial, nominal=nominal,
                        object_goals=objects, hand_goals=hands, times=times, knot_times=knot_times)
    horizons = np.unique(np.r_[np.arange(config.horizon_increment_s, times[-1], config.horizon_increment_s), times[-1]])
    best = mean.copy()
    with mujoco.rollout.Rollout(nthread=config.threads) as engine:
        for stage, horizon in enumerate(horizons):
            end = min(len(times), int(np.searchsorted(times, horizon, side="right")))
            count = min(len(knot_times), int(np.searchsorted(knot_times, horizon, side="left"))+1)
            for iteration in range(config.iterations):
                knots = rng.normal(mean[:count], std[:count], (config.samples, count, len(active)))
                knots = np.clip(knots, -radius, radius)
                knots[0] = best[:count]
                knots[1] = 0  # Explicit reference alternative at every horizon.
                knots[:, 0] = 0  # Fixed initial command boundary, not a new reset pose.
                control = np.broadcast_to(nominal[:end], (config.samples, end, model.nu)).copy()
                for sample in range(config.samples):
                    residual = interpolate_residual(knots[sample], knot_times[:count], times[:end])
                    control[sample][:, active] += residual
                limited = np.flatnonzero(model.actuator_ctrllimited)
                control[..., limited] = np.clip(control[..., limited], model.actuator_ctrlrange[limited, 0], model.actuator_ctrlrange[limited, 1])
                state, sensors = engine.rollout(model, data, initial[None], np.repeat(control, substeps, axis=1))
                state, sensors = state[:, substeps-1::substeps], sensors[:, substeps-1::substeps]
                scores, terms = dense_cost(model, state, sensors, control, nominal[:end], objects[:end], hands[:end],
                                           grip[:end] < 0, object_qpos, robot_qpos, robot_dof)
                elite = np.argsort(scores, kind="stable")[:config.elite]
                best[:count] = knots[elite[0]]
                mean[:count] = .5*mean[:count]+.5*knots[elite].mean(0)
                std[:count] = np.maximum(.5*std[:count]+.5*knots[elite].std(0), radius*config.minimum_std_fraction)
                for sample in range(config.samples):
                    name = f"stage_{stage:03d}_iteration_{iteration:02d}_sample_{sample:03d}.npz"
                    np.savez_compressed(output/name, control=control[sample], state=state[sample],
                                        sensors=sensors[sample], knots=knots[sample], score=scores[sample])
                    summary["samples"].append({"path": name, "sha256": sha256(output/name),
                        "horizon_s": float(horizon), "score": float(scores[sample]),
                        "terms": {key:float(value[sample]) for key,value in terms.items()},
                        "status": "optimization_sample_not_validated"})
                summary["last_horizon_s"] = float(horizon)
                summary["wall_s"] = time.monotonic()-started
                json_write(output/"search.json", summary)
                print(json.dumps({"stage": stage, "horizon_s": float(horizon), "iteration": iteration,
                                  "best_cost": float(scores[elite[0]]), "wall_s": summary["wall_s"]}), flush=True)
    controls = nominal.copy()
    controls[:, active] += interpolate_residual(best, knot_times, times)
    limited = np.flatnonzero(model.actuator_ctrllimited)
    controls[:, limited] = np.clip(controls[:, limited], model.actuator_ctrlrange[limited, 0], model.actuator_ctrlrange[limited, 1])
    path = output/"selected-control.npz"
    np.savez_compressed(path, control=controls, scene_sha256=summary["scene_sha256"], initial_qpos=q0, contact_closed=grip < 0)
    summary.update(finished=now(), selected_control_sha256=sha256(path), wall_s=time.monotonic()-started)
    json_write(output/"search.json", summary)
    return path
