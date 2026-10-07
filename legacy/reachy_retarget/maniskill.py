"""Offline adapter for the pinned ManiSkill PickCube teleoperation release.

Only measured articulation states feed FK. Source actions remain T transitions,
while object/robot state and derived TCP poses retain all T+1 recorded frames.
The module never imports ManiSkill, SAPIEN, an SDK, or accesses the network.
"""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .episodes import matrices_to_pose, pose_to_matrices, write_episode
from .store import sha256

SOURCE_COMMIT = "ecc579b7567c31bb3d7176539d90e86ae49296be"
DATASET_REVISION = "d674485bbffdd533914e52d272fdda34c0515608"
SOURCE_BASE = "https://github.com/haosulab/ManiSkill/blob/" + SOURCE_COMMIT + "/"
URDF_URL = SOURCE_BASE.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/") + "mani_skill/assets/robots/panda/panda_v2.urdf"
ARM_JOINT_NAMES = ["panda_joint" + str(i) for i in range(1, 8)]
JOINT_NAMES = ARM_JOINT_NAMES + ["panda_finger_joint1", "panda_finger_joint2"]
EVIDENCE = {
    "actor_state": SOURCE_BASE + "mani_skill/utils/structs/actor.py#L111-L131",
    "articulation_state": SOURCE_BASE + "mani_skill/utils/structs/articulation.py#L174-L197",
    "quaternion": SOURCE_BASE + "mani_skill/utils/structs/pose.py#L124-L140",
    "joint_order_and_tcp": SOURCE_BASE + "mani_skill/agents/robots/panda/panda.py#L33-L54",
    "control_frequency": SOURCE_BASE + "mani_skill/utils/structs/types.py#L70-L78",
    "cube_dimensions": SOURCE_BASE + "mani_skill/envs/tasks/pick_cube.py#L16-L44",
    "goal_semantics": SOURCE_BASE + "mani_skill/envs/tasks/pick_cube.py#L45-L87",
    "trajectory_layout": "https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html",
}


def panda_tcp_matrices(urdf, qpos, root_pose):
    """Compose the pinned URDF chain using named measured arm positions."""
    qpos, root_pose = np.asarray(qpos, float), np.asarray(root_pose, float)
    if qpos.ndim != 2 or qpos.shape[1] != 9 or root_pose.shape != (len(qpos), 7):
        raise ValueError("Panda FK requires N x 9 measured qpos and N x 7 root pose")
    if not np.isfinite(qpos).all() or not np.isfinite(root_pose).all():
        raise ValueError("Nonfinite Panda state")
    if not np.allclose(np.linalg.norm(root_pose[:, 3:], axis=1), 1, atol=1e-3):
        raise ValueError("Invalid root quaternion")
    tree = ET.parse(urdf)
    by_child = {joint.find("child").get("link"): joint for joint in tree.findall("joint")}
    chain, link = [], "panda_hand_tcp"
    while link != "panda_link0":
        if link not in by_child or len(chain) > len(by_child):
            raise ValueError("Cannot resolve Panda root to TCP chain")
        joint = by_child[link]
        chain.append(joint)
        link = joint.find("parent").get("link")
    chain.reverse()
    if [j.get("name") for j in chain if j.get("type") != "fixed"] != ARM_JOINT_NAMES:
        raise ValueError("Unverified Panda arm joint chain")
    world = pose_to_matrices(root_pose)
    for joint in chain:
        origin = joint.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        fixed = np.eye(4)
        fixed[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        fixed[:3, 3] = xyz
        world = world @ fixed
        if joint.get("type") == "fixed":
            continue
        if joint.get("type") != "revolute":
            raise ValueError("Unsupported Panda arm joint type")
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        axis /= np.linalg.norm(axis)
        angle = qpos[:, JOINT_NAMES.index(joint.get("name"))]
        rotation = np.broadcast_to(np.eye(4), (len(qpos), 4, 4)).copy()
        rotation[:, :3, :3] = Rotation.from_rotvec(angle[:, None] * axis).as_matrix()
        world = world @ rotation
    return world


def verify_pinocchio(urdf, qpos, root_pose, derived):
    """Independent mesh-free FK check when the optional model runtime exists."""
    try:
        import pinocchio as pin
    except ImportError:
        return {"available": False, "reason": "Pinocchio is not installed"}
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    tcp = model.getFrameId("panda_hand_tcp")
    if tcp >= model.nframes:
        raise ValueError("Pinocchio cannot find panda_hand_tcp")
    root = pose_to_matrices(root_pose)
    errors = []
    for i, observed in enumerate(qpos):
        configuration = pin.neutral(model)
        for name, value in zip(JOINT_NAMES, observed):
            jid = model.getJointId(name)
            if jid == 0 or model.joints[jid].nq != 1:
                raise ValueError("Missing independent Panda joint: " + name)
            configuration[model.joints[jid].idx_q] = value
        pin.framesForwardKinematics(model, data, configuration)
        independent = root[i] @ data.oMf[tcp].homogeneous
        errors.append(float(np.max(np.abs(independent - derived[i]))))
    error = max(errors)
    if error > 1e-9:
        raise ValueError("Independent Panda FK disagrees: " + str(error))
    return {"available": True, "frames": len(errors), "max_matrix_error": error,
            "library": "Pinocchio", "version": pin.__version__}


def maniskill(store, source, limit=5):
    root = store.root / "data/raw" / source
    urdf = store.root / "data/raw/maniskill_panda_assets/panda_v2.urdf"
    if not urdf.is_file():
        raise ValueError("Missing pinned Panda URDF; explicitly fetch " + URDF_URL)
    ledger = store.rows("files", "source_id=? AND path=?", ("maniskill_panda_assets", "panda_v2.urdf"))
    if len(ledger) != 1 or ledger[0]["url"] != URDF_URL or ledger[0]["status"] != "downloaded" or ledger[0]["sha256"] != sha256(urdf):
        raise ValueError("Panda URDF provenance/checksum does not match pinned source")
    outputs = []
    for path in sorted(root.rglob("trajectory.h5")):
        metadata_path = path.with_suffix(".json")
        info = json.loads(metadata_path.read_text())
        env = info.get("env_info", {})
        kwargs = env.get("env_kwargs", {})
        if (env.get("env_id") != "PickCube-v1" or info.get("commit_info", {}).get("commit_id") != SOURCE_COMMIT
                or kwargs.get("control_mode") != "pd_joint_pos" or kwargs.get("robot_uids", "panda") != "panda"):
            raise ValueError("Unverified ManiSkill environment, robot, controller, or source commit")
        sim = kwargs.get("sim_cfg", {})
        frequency = float(sim.get("control_freq", 20))
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("Invalid ManiSkill source control frequency")
        episode_metadata = {int(e["episode_id"]): e for e in info["episodes"]}
        with h5py.File(path, "r") as f:
            for name in sorted(f, key=lambda x: int(x.removeprefix("traj_"))):
                if limit and len(outputs) >= limit:
                    break
                group = f[name]
                actions = group["actions"][()]
                frames = len(actions) + 1
                if actions.shape != (frames - 1, 8) or not np.isfinite(actions).all():
                    raise ValueError("Unverified Panda pd_joint_pos action shape")
                actor = group["env_states/actors/cube"][()]
                state = group["env_states/articulations/panda"][()]
                if actor.shape != (frames, 13) or state.shape != (frames, 31):
                    raise ValueError("Expected T+1 Cube actor13 and Panda articulation31 states")
                if not np.isfinite(actor).all() or not np.isfinite(state).all():
                    raise ValueError("Nonfinite ManiSkill simulator state")
                if not np.allclose(np.linalg.norm(actor[:, 3:7], axis=1), 1, atol=1e-3):
                    raise ValueError("Invalid object quaternion")
                qpos = state[:, 13:22]
                # The exact pinned teleop export starts with its measured arm
                # state as the first named pd_joint_pos command. This checks
                # the otherwise unnamed simulator qpos ordering against the
                # documented controller's panda_joint1..7 ordering.
                order_error = float(np.max(np.abs(actions[0, :7] - qpos[0, :7])))
                if order_error > 1e-6:
                    raise ValueError("Pinned teleop initial command cannot verify arm qpos ordering")
                tcp = panda_tcp_matrices(urdf, qpos, state[:, :7])
                check = verify_pinocchio(urdf, qpos, state[:, :7], tcp)
                arrays = {
                    "time_s": np.arange(frames) / frequency,
                    "source/action": actions,
                    "source/action_time_s": np.arange(frames - 1) / frequency,
                    "source/joint_names": JOINT_NAMES,
                    "source/joint_position": qpos,
                    "source/joint_velocity": state[:, 22:31],
                    "source/robot_root_pose": state[:, :7],
                    "source/gripper_width_m": qpos[:, 7:9].sum(axis=1),
                    "hand/right_pose": matrices_to_pose(tcp),
                    "objects/cube/pose": actor[:, :7],
                    "objects/cube/linear_velocity": actor[:, 7:10],
                    "objects/cube/angular_velocity": actor[:, 10:13],
                }
                def preserve(key, dataset):
                    if isinstance(dataset, h5py.Dataset) and dataset.dtype.kind in "bifu" and key not in {"actions"}:
                        arrays["source/" + key] = dataset[()]
                group.visititems(preserve)
                metadata = {
                    "source_format": "ManiSkill-HDF5", "robot_type": "Panda", "scenario": "PickCube-v1",
                    "env_args": {"env_name": "PickCube-v1", "env_kwargs": kwargs},
                    "source_commit": SOURCE_COMMIT, "source_dataset_revision": DATASET_REVISION,
                    "source_episode_metadata": episode_metadata[int(name.removeprefix("traj_"))],
                    "source_group": source + "/PickCube-v1/teleop/" + name,
                    "source_coordinates": "SAPIEN world, meters, right-handed Z-up, xyz+wxyz",
                    "fps": frequency, "schema_evidence": EVIDENCE,
                    "objects": {"cube": {"pose_frame": "world", "type": "cube", "geometry": {"type": "box", "half_size_m": [0.02] * 3},
                                          "source": "env_states/actors/cube", "geometry_evidence": EVIDENCE["cube_dimensions"]}},
                    "excluded_actor_roles": {"goal_site": "noncolliding visual goal marker", "table-workspace": "environment, retained in source states"},
                    "missing": ["contact", "Reachy_gripper_mapping", "Reachy_physical_validation"],
                    "hand_pose_derivation": {"method": "FK from measured Panda articulation qpos and measured root world pose",
                                             "urdf": str(urdf.relative_to(store.root)), "urdf_sha256": sha256(urdf), "urdf_url": URDF_URL,
                                             "tcp_frame": "panda_hand_tcp", "joint_order": JOINT_NAMES,
                                             "initial_command_order_check_max_rad": order_error, "independent_fk_check": check},
                    "timing": "T+1 complete state frames including terminal state; original T actions retained with source/action_time_s; no fabricated terminal action",
                    "action_semantics": "Source Panda joint-position targets and normalized gripper command; not measured pose or Reachy commands",
                    "gripper_width_semantics": "Sum of measured Panda finger prismatic positions in meters; no Reachy finger-angle conversion assumed",
                    "provenance": [{"path": str(p.relative_to(store.root)), "sha256": sha256(p)} for p in (path, metadata_path, urdf)],
                }
                sequence = str(path.relative_to(root).parent) + "/" + name
                outputs.append(write_episode(store, source, sequence, arrays, metadata))
    if not outputs:
        raise ValueError("No supported downloaded ManiSkill trajectories")
    return outputs
