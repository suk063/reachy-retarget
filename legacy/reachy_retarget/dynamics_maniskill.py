"""Offline PickCube grasp/timing/goal adapter for measured ManiSkill states.

Commands are derived for the Reachy planner; original T actions and T+1 state
arrays remain untouched. No SAPIEN, ManiSkill or hardware SDK is imported.
"""

import numpy as np

from .episodes import pose_to_matrices
from .maniskill import SOURCE_COMMIT


TASK = "PickCube-v1"
OBJECT = "cube"
PLACEMENT_Z = .8
SOURCE_BASE = "https://github.com/haosulab/ManiSkill/blob/" + SOURCE_COMMIT + "/"
REFERENCES = {
    "task_success": SOURCE_BASE + "mani_skill/envs/tasks/pick_cube.py#L87-L96",
    "robot_static": SOURCE_BASE + "mani_skill/agents/robots/panda/panda.py#L307-L309",
    "panda_controller": SOURCE_BASE + "mani_skill/agents/robots/panda/panda.py#L154-L167",
    "normalized_gripper": SOURCE_BASE + "mani_skill/agents/controllers/pd_joint_pos.py#L97-L130",
    "action_scaling": SOURCE_BASE + "mani_skill/utils/common.py#L179-L182",
}


def supported(metadata):
    return (metadata.get("source_format") == "ManiSkill-HDF5"
            and metadata.get("source_commit") == SOURCE_COMMIT
            and metadata.get("robot_type") == "Panda"
            and metadata.get("env_args", {}).get("env_name") == TASK
            and metadata.get("env_args", {}).get("env_kwargs", {}).get("control_mode") == "pd_joint_pos")


def _verify(metadata):
    if not supported(metadata):
        raise ValueError("Unsupported ManiSkill task, revision, robot or controller contract")


def source_grasp(arrays, metadata, oid=OBJECT, geometry_center=True):
    """Return the planner's E, O, signed closure, anchor, local grasp, label.

    Panda action[7] uses +1=open and -1=close (target .04/-.01 m per
    finger). The common Reachy planner expects +1=close, -1=open. Action i
    applies at state-time i; the derived terminal command holds the last action.
    It is not appended to, or represented as, an original source action.
    """
    _verify(metadata)
    if oid != OBJECT:
        raise ValueError("PickCube only supports its recorded cube")
    time = np.asarray(arrays["time_s"], float)
    action_time = np.asarray(arrays["source/action_time_s"], float)
    actions = np.asarray(arrays["source/action"], float)
    if (time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0)
            or actions.shape != (len(time) - 1, 8) or not np.isfinite(actions).all()
            or action_time.shape != (len(time) - 1,) or not np.allclose(action_time, time[:-1], rtol=0, atol=1e-8)):
        raise ValueError("PickCube requires T actions and T+1 measured states with their original clocks")
    action_grip = actions[:, 7]
    if not np.isin(action_grip, [-1., 1.]).all():
        raise ValueError("This adapter requires the verified binary source gripper commands")
    E = pose_to_matrices(np.asarray(arrays["hand/right_pose"], float))
    O = pose_to_matrices(np.asarray(arrays["objects/cube/pose"], float))
    if E.shape != (len(time), 4, 4) or O.shape != E.shape or not np.isfinite(E).all() or not np.isfinite(O).all():
        raise ValueError("Missing complete measured hand/object poses")
    grip = -np.r_[action_grip, action_grip[-1]]
    # Lifted state i must follow a closed source transition and retain closure.
    applied = -np.r_[action_grip[0], action_grip]
    rise = O[:, 2, 3] - O[0, 2, 3]
    threshold = min(.03, .5 * float(np.max(rise)))
    lifted = np.flatnonzero((grip > 0) & (applied > 0) & (rise > threshold))
    if threshold < .005 or not len(lifted):
        raise ValueError("No measured closed-gripper source lift establishes a grasp")
    anchor = int(lifted[0])
    if geometry_center:
        local = np.zeros(3)
        region = "center of the unchanged source box collision geometry, half-size 0.02 m"
    else:
        local = (np.linalg.inv(O[anchor]) @ E[anchor])[:3, 3]
        region = "measured source TCP position in the cube frame at the source lift"
    return E, O, grip, anchor, local, region


def task_context(arrays, metadata, placement):
    """Keep the official target distinct from the final demonstrated cube pose."""
    _verify(metadata)
    goals = np.asarray(arrays["source/env_states/actors/goal_site"], float)
    if (goals.shape != (len(arrays["time_s"]), 13) or not np.isfinite(goals).all()
            or not np.allclose(goals[:, :7], goals[0, :7], rtol=0, atol=1e-7)
            or not np.allclose(goals[:, 7:], 0, rtol=0, atol=1e-7)):
        raise ValueError("PickCube needs the recorded stationary source goal")
    transform = np.asarray(placement, float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Invalid scene placement")
    world = (transform @ np.r_[goals[0, :3], 1])[:3]
    return {
        "task": TASK, "goal_source_m": goals[0, :3].tolist(), "goal_world_m": world.tolist(),
        "goal_distance_m": .025, "arm_static_threshold_rad_s": .2,
        "base_static_threshold_m_s": .02, "base_yaw_static_threshold_rad_s": .2,
        "release_required": False, "references": REFERENCES,
        "source_success": "cube-goal Euclidean distance <=0.025 m and max absolute Panda non-finger joint velocity <=0.2 rad/s",
        "retargeted_static_interpretation": "all Reachy arm joints <=0.2 rad/s; additionally require base planar speed <=0.02 m/s and yaw speed <=0.2 rad/s",
        "gripper_command_derivation": "negate source action column 7 at its original state time; hold last command at terminal T+1 state without modifying source actions",
        "goal_policy": "official recorded goal; final demonstration pose is a trajectory reference, not the success target",
    }


def task_metrics(cube_world, arm_velocity, base_velocity, context):
    """Evaluate the official geometry/static predicate plus explicit base rule."""
    cube, arms, base = map(lambda x: np.asarray(x, float), (cube_world, arm_velocity, base_velocity))
    if cube.shape != (3,) or arms.ndim != 1 or not len(arms) or base.shape != (3,):
        raise ValueError("Expected cube xyz, non-finger arm velocities, and base vx/vy/yaw velocity")
    distance = float(np.linalg.norm(cube - np.asarray(context["goal_world_m"])))
    finite = np.isfinite(cube).all() and np.isfinite(arms).all() and np.isfinite(base).all()
    arm_peak = float(np.max(np.abs(arms)))
    base_speed = float(np.linalg.norm(base[:2]))
    placed = bool(finite and distance <= .025)
    static = bool(finite and arm_peak <= .2 and base_speed <= .02 and abs(base[2]) <= .2)
    return {"goal_distance_m": distance, "is_obj_placed": placed, "is_robot_static": static,
            "arm_peak_rad_s": arm_peak, "base_speed_m_s": base_speed,
            "success": bool(placed and static)}


def task_satisfied(model, data, cube_body, context):
    """MuJoCo convenience hook; all queried joints are robot joints."""
    from .physics import ARMS, BASE
    arms = [int(model.joint(name).dofadr[0]) for name in ARMS]
    base = [int(model.joint(name).dofadr[0]) for name in BASE]
    return task_metrics(data.xpos[cube_body], data.qvel[arms], data.qvel[base], context)["success"]
