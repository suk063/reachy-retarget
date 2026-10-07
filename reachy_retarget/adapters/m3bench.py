"""Offline M3Bench native JSON adapter for planned mobile-arm waypoints.

Planner convergence and attachment metadata are not measured physical success.
In particular, no object trajectory is invented from a gripper attachment.
"""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


CODE_REVISION = "97cec07e7c37cc36aec35ff79ee5d6b5db123a45"
REVISION = "5b543a8c93c82b16442b2c1c4a139b96bb313b59"
REPOSITORY = "https://github.com/TooSchoolForCool/M3Bench"
ARCHIVE_URL = "https://huggingface.co/datasets/M3Bench/M3Bench/resolve/" + REVISION + "/pick_traj.zip"
PILOT_SEQUENCE = "pick_traj/physcene_4838/book_28_link/2024-10-03-02-53-19/23"
JOINT_NAMES = ["base_y_base_x", "base_theta_base_y", "base_link_base_theta"] + ["joint_" + str(i) for i in range(1, 8)]
PILOT_MEMBERS = [
    {"path": PILOT_SEQUENCE + "/pick_vkc_return.json", "offset": 975, "compressed_bytes": 5276, "bytes": 15491, "compression": 8, "crc32": 3571944598, "sha256": "6a861c0eb038f54d92e0b88d4c28e318b32aded8fc238129f5afb55a26cecdfe"},
    {"path": PILOT_SEQUENCE + "/config.json", "offset": 6400, "compressed_bytes": 467, "bytes": 1709, "compression": 8, "crc32": 2411641855, "sha256": "cf05c7fc129086708bc1beeff0d662ea9a503d4b8b6d6ad5607e9645bf963c0c"},
    {"path": PILOT_SEQUENCE + "/vkc_request.json", "offset": 7021, "compressed_bytes": 4088, "bytes": 29770, "compression": 8, "crc32": 2623813121, "sha256": "9c293c9dd01d2e12e24947d3388964bcbd0d5ecdd94b41202c8afa91b8605289"},
    {"path": PILOT_SEQUENCE + "/trajectory/pick_vkc_caption_trajectory.json", "offset": 25930, "compressed_bytes": 5551, "bytes": 18201, "compression": 8, "crc32": 1935798397, "sha256": "3f79b57164baa6b62fb3da98b72d962863ec3b59d58711035226eb7ae6fe38b7"},
]
MISSING = [
    "timestamp",
    "physical_timestamps_or_reviewed_time_parameterization",
    "gripper_open_close_trajectory",
    "continuous_object_pose_and_contact_forces",
    "source_robot_and_scene_assets_for_fk_and_collision",
    "object_mass_inertia_and_contact_parameters",
    "retargeted_reachy_state_action_and_actuator_replay",
    "continuous_task_target_pose_and_relevant_support_or_receptacle_transforms",
]


def describe():
    return {
        "dataset": "m3bench", "source_revision": REVISION, "code_revision": CODE_REVISION,
        "source_urls": [REPOSITORY, "https://huggingface.co/datasets/M3Bench/M3Bench"],
        "license": "Apache-2.0 on publisher dataset card; asset-specific notices require preservation",
        "native_format": "config.json + named VKC return + captioned 10-DoF joint waypoints; no native physical clock",
        "storage_root": "/mnt/reachy-retarget",
        "pilot": {
            "url": ARCHIVE_URL, "archive_bytes": 2811609913,
            "method": "HTTP Range of pinned raw-DEFLATE member payloads, then CRC32 and SHA256 verification",
            "members": [dict(member) for member in PILOT_MEMBERS],
            "selected_bytes": sum(member["bytes"] for member in PILOT_MEMBERS),
            "whole_archive_checksum_verified": False,
        },
        "assets": {
            "robot_urdf": {"url": ARCHIVE_URL.rsplit("/", 1)[0] + "/robot_urdf.zip", "bytes": 56672394},
            "scene_urdf": {"url": ARCHIVE_URL.rsplit("/", 1)[0] + "/scene_urdf.zip", "bytes": 15239447414, "selection": "physcene_4838 and its referenced meshes/materials"},
        },
        "fk_asset": {
            "url": ARCHIVE_URL.rsplit("/", 1)[0] + "/robot_urdf.zip",
            "path": "robot_urdf/Mec_kinova/main.urdf", "local_name": "robot.urdf",
            "offset": 2760, "compressed_bytes": 3702, "bytes": 26532,
            "compression": 8, "crc32": 2058453746,
            "sha256": "f4534b09ef7f97ca38fee8ccdf83faccfacdf6af643afede30fa2f30aeaa22ba",
            "role": "Mesh-free FK only; full scene collision assets still required",
        },
        "pipeline": [
            {"stage": "fetch", "operation": "Fetch exact selected JSON member ranges and separately selected scene/robot assets"},
            {"stage": "inspect", "operation": "Cross-check caption and named VKC waypoints, target initial transform, request collision settings"},
            {"stage": "normalize", "operation": "Preserve 10-DoF joint waypoints, initial object transform and language; leave physical time and moving object tracks absent"},
            {"stage": "agent_review", "operation": "Review time parameterization, gripper closure, physical object properties and FK; reject planner attachment and collision-shrinking assistance"},
            {"stage": "retarget", "operation": "Map FK wrist/base references and feasible pad contact to Reachy without changing the object"},
            {"stage": "validate", "operation": "Simulate free objects with robot actuators only, record all attempts and independently test completion"},
            {"stage": "export", "operation": "Write shared archive; complete Reachy-agent v5 requires generated Reachy observations and physical rollout"},
        ],
        "source_replay": {
            "repository": REPOSITORY, "revision": CODE_REVISION,
            "entrypoint": "evaluation/tv_evaluate/evaluate_pick.py",
            "dependencies": "Isaac Sim 2023.1.1, Python 3.10.13 and TongVerse; setup references a private 10.2.31.187 release URL",
            "role": "Final-pose grasp evaluator: sets object world pose and robot to the final waypoint before close/lift. It is not full trajectory replay and cannot certify our dynamics gates.",
            "ready_for_unattended_install": False,
            "independent_reconstruction": "Use pinned robot/scene URDF for FK; explicitly parameterize waypoint time and gripper closure; simulate free objects in MuJoCo without attachments or collision shrinking",
            "pybullet_entrypoint": "evaluation/evaluate_traj.py",
            "pybullet_role": "Collision/limit/smoothness evaluation; not equivalent to physical grasp success",
        },
        "missing_fields": list(MISSING), "status": "native_waypoint_normalization_implemented_reconstruction_required",
        "physics_validated": False,
        "task_object_pose_coverage": {
            "scope": "initial_task_target_only", "all_task_object_poses_saved": False,
            "unrelated_scene_objects_required": False,
            "continuous_source_object_poses_available": False,
        },
    }


def _paths(path):
    path = Path(path)
    if path.is_file():
        root = path.parent.parent if path.parent.name == "trajectory" else path.parent
    else:
        root = path
    if not (root / "config.json").is_file():
        configs = sorted(root.rglob("config.json")) if root.is_dir() else []
        if len(configs) != 1:
            raise ValueError("Select one M3Bench instance containing config.json")
        root = configs[0].parent
    candidates = sorted(root.glob("trajectory/*_vkc_caption_trajectory.json"))
    if not candidates:
        candidates = sorted(root.glob("*_vkc_caption_trajectory.json"))
    if len(candidates) != 1:
        raise ValueError("Expected one captioned native trajectory in the selected instance")
    return root, root / "config.json", candidates[0]


def fk_end_effector(urdf, qpos):
    """Derive source gripper-base FK from named URDF joints, without meshes."""
    qpos = np.asarray(qpos, float)
    if qpos.ndim != 2 or qpos.shape[1] != 10 or not np.isfinite(qpos).all():
        raise ValueError("Expected finite N x 10 source joint positions")
    tree = ET.parse(urdf)
    by_child = {joint.find("child").get("link"): joint for joint in tree.findall("joint")}
    chain, link = [], "robotiq_arg2f_base_link"
    while link != "world":
        if link not in by_child or len(chain) > len(by_child):
            raise ValueError("Cannot resolve source gripper to world URDF chain")
        joint = by_child[link]
        chain.append(joint)
        link = joint.find("parent").get("link")
    chain.reverse()
    if [joint.get("name") for joint in chain if joint.get("type") != "fixed"] != JOINT_NAMES:
        raise ValueError("Source URDF movable joint order differs from native trajectory")
    world = np.broadcast_to(np.eye(4), (len(qpos), 4, 4)).copy()
    for joint in chain:
        origin = joint.find("origin")
        attributes = {} if origin is None else origin.attrib
        xyz = np.fromstring(attributes.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(attributes.get("rpy", "0 0 0"), sep=" ")
        if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite(np.r_[xyz, rpy]).all():
            raise ValueError("Malformed source URDF joint origin")
        fixed = np.eye(4)
        fixed[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        fixed[:3, 3] = xyz
        world = world @ fixed
        kind = joint.get("type")
        if kind == "fixed":
            continue
        axis_element = joint.find("axis")
        axis = np.fromstring("1 0 0" if axis_element is None else axis_element.get("xyz"), sep=" ")
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-10:
            raise ValueError("Invalid source URDF joint axis")
        axis /= np.linalg.norm(axis)
        value = qpos[:, JOINT_NAMES.index(joint.get("name"))]
        motion = np.broadcast_to(np.eye(4), (len(qpos), 4, 4)).copy()
        if kind in {"revolute", "continuous"}:
            motion[:, :3, :3] = Rotation.from_rotvec(value[:, None] * axis).as_matrix()
        elif kind == "prismatic":
            motion[:, :3, 3] = value[:, None] * axis
        else:
            raise ValueError("Unsupported source URDF joint type")
        world = world @ motion
    return world


def normalize(path, output_dir=None, *, robot_urdf=None):
    """Return partial shared channels; leave unknown physical timestamps absent."""
    root, config_path, trajectory_path = _paths(path)
    config, trajectory = json.loads(config_path.read_text()), json.loads(trajectory_path.read_text())
    qpos = np.asarray(trajectory["trajectory"], dtype=float)
    if qpos.ndim != 2 or qpos.shape[1] != len(JOINT_NAMES) or len(qpos) < 2 or not np.isfinite(qpos).all():
        raise ValueError("Expected finite N x 10 native joint waypoints")
    env = config["env"]
    obj = env["object"]
    target = str(obj["name"])
    if trajectory.get("link_name") != target:
        raise ValueError("Caption target and initial object configuration disagree")
    transform = np.asarray(obj["transformation_matrix"], dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("Invalid initial object homogeneous transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
        raise ValueError("Object transform contains scale or a non-rigid rotation")
    initial_base = np.asarray(env["agent"]["position"], dtype=float)
    if initial_base.shape != (3,) or not np.allclose(initial_base, qpos[0, :3], atol=1e-6):
        raise ValueError("Initial robot base and first absolute waypoint disagree")
    observed_files = [config_path, trajectory_path]
    action_type = trajectory.get("action_type")
    if action_type not in {"pick", "place"}:
        raise ValueError("Unverified M3Bench action type")
    return_path = root / (action_type + "_vkc_return.json")
    planner_converged = None
    joint_order_verified = False
    if return_path.exists():
        result = json.loads(return_path.read_text())
        if result.get("joint_names") != JOINT_NAMES:
            raise ValueError("Named VKC joint order differs from the pinned source robot")
        named = np.column_stack([result[name] for name in JOINT_NAMES])
        if named.shape != qpos.shape or not np.allclose(named, qpos, atol=1e-8):
            raise ValueError("Captioned waypoints differ from the named VKC result")
        if result.get("n_step") != len(qpos):
            raise ValueError("VKC step count disagrees with trajectory length")
        planner_converged = result.get("converged")
        joint_order_verified = True
        observed_files.append(return_path)
    request_path = root / "vkc_request.json"
    source_collision_scale = None
    if request_path.exists():
        request = json.loads(request_path.read_text())
        source_collision_scale = request.get("vkc_env", {}).get("collision_scale", {})
        observed_files.append(request_path)
    checksums = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in observed_files}
    pinned_by_name = {Path(member["path"]).name: member for member in PILOT_MEMBERS}
    verified_pilot = all(name in pinned_by_name and pinned_by_name[name]["sha256"] == digest for name, digest in checksums.items())
    source_sequence = PILOT_SEQUENCE if verified_pilot else root.as_posix().split("pick_traj/")[-1]
    source_group = "/".join(source_sequence.split("/")[:-1]) if verified_pilot or "/" in source_sequence else "unresolved-recording/" + checksums[config_path.name]
    arrays = {
        "source/waypoint_index": np.arange(len(qpos), dtype=np.int64),
        "source/joint_position": qpos,
        "source/joint_names": np.asarray(JOINT_NAMES),
        "source/base_planar_configuration": qpos[:, :3].copy(),
        "source/object_initial_transform": transform,
    }
    missing = list(MISSING)
    derived = {"source/base_planar_configuration": "First three named source joints; no base motion model was simulated"}
    robot_urdf = Path(robot_urdf) if robot_urdf else root / "robot.urdf"
    if robot_urdf.is_file():
        arrays["source/ee_transform"] = fk_end_effector(robot_urdf, qpos)
        derived["source/ee_transform"] = {
            "method": "Named source URDF forward kinematics, world to robotiq_arg2f_base_link; no gripper-pad calibration implied",
            "urdf_sha256": hashlib.sha256(robot_urdf.read_bytes()).hexdigest(),
        }
    if not joint_order_verified:
        missing.append("named_vkc_result_crosscheck")
    metadata = {
        "source_id": "m3bench", "source_revision": REVISION,
        "source_revision_verified_for_pilot": verified_pilot, "adapter_code_reference_revision": CODE_REVISION,
        "source_urls": [ARCHIVE_URL + "#member=" + pinned_by_name[name]["path"] for name in checksums] if verified_pilot else [REPOSITORY],
        "source_sequence": source_sequence, "source_group": "m3bench/" + source_group,
        "source_checksums": checksums, "scene": env["scene"]["name"],
        "task": action_type, "instruction": trajectory.get("caption", ""),
        "objects": {target: {"pose_availability": "initial_only", "source_link": target}},
        "task_object_pose_coverage": {
            "scope": "initial_task_target_only", "all_task_object_poses_saved": False,
            "required_task_object_ids": list(dict.fromkeys([target] + [trajectory[key] for key in ("supporter", "receptacle") if trajectory.get(key)])),
            "initial_object_ids": [target], "continuous_object_ids": [],
            "unrelated_scene_objects_required": False,
            "reason": "Native JSON lacks continuous target motion and related support/receptacle transforms; unrelated scene props are outside pose-logging scope",
        },
        "source_robot": env["agent"]["name"],
        "coordinate_frame": "Original source scene world frame",
        "joint_units": ["m", "m", "rad"] + ["rad"] * 7,
        "base_configuration_order": ["x", "y", "yaw"],
        "physical_time_available": False,
        "planner_converged": planner_converged,
        "source_planner_collision_scale": source_collision_scale,
        "source_attachments": config.get("attachments", {}),
        "derived_fields": derived,
        "simulation_assumptions": [],
        "prohibited_physical_shortcuts": ["source planner collision shrinking", "kinematic object attachment", "invented source object trajectory"],
        "missing_fields": missing, "physics_validated": False,
    }
    return {"arrays": arrays, "metadata": metadata, "status": "normalized_partial", "missing_fields": missing}


def inspect(path):
    result = normalize(path)
    return {
        "dataset": "m3bench", "status": "inspected_not_retargeted",
        "waypoints": len(result["arrays"]["source/waypoint_index"]),
        "physical_time_available": False,
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in result["arrays"].items()},
        "metadata": result["metadata"], "missing_fields": result["missing_fields"],
        "physics_validated": False,
    }
