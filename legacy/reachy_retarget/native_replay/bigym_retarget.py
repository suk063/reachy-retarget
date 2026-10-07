"""Conservative Reachy IK from verified native H1 replay, without object edits.

Only the source-specific channel/command contract is implemented here. The
shared pose_retarget solver preserves the complete clock and horizontal base
path. Source task failure is retained and is not an IK rejection criterion.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .bigym import SOURCE_REVISION, TASK_OBJECTS


HAND_KEYS = {side: f"source/{side}_pinch_pose" for side in ("left", "right")}
BASE_KEY = "source/base_pose"
CLOCK_KEY = "timestamp"
GRIPPER_OPEN_RAD = 2.0
GRIPPER_CLOSED_RAD = -0.06
SOURCE_BLOB = "https://github.com/NeuracoreAI/bigym/blob/" + SOURCE_REVISION + "/"


def prepare(record):
    """Verify binary native aperture polarity before deriving Reachy targets."""
    arrays, metadata = record["arrays"], record["metadata"]
    if metadata.get("source_revision") != SOURCE_REVISION or metadata.get("replay_bigym_version") != "4.1.0":
        raise ValueError("Unverified native BiGym source revision/version")
    if metadata.get("source_replay_completed") is not True:
        raise ValueError("Full native replay is required for the full-trajectory IK attempt")
    clock = np.asarray(arrays[CLOCK_KEY])
    if clock.ndim != 1 or len(clock) < 2 or not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0):
        raise ValueError("Finite increasing actual native replay timestamps are required")
    action = np.asarray(arrays.get("source/action"))
    if action.shape != (len(clock), 15) or not np.isfinite(action).all():
        raise ValueError("Missing finite aligned 15-component native actions")
    channels = metadata.get("action_channels", [])
    if len(channels) != 15 or [row.get("index") for row in channels] != list(range(15)):
        raise ValueError("Recorded native action-channel order is required")
    if any(row.get("kind") != "base_position_target_increment" for row in channels[:3]):
        raise ValueError("BiGym base channels must remain target increments")
    if any(row.get("kind") != "absolute_joint_position_target" for row in channels[3:13]):
        raise ValueError("BiGym arm channels must remain absolute source joint targets")
    for key in [BASE_KEY, *HAND_KEYS.values(), *("objects/" + name + "/pose" for name in TASK_OBJECTS)]:
        value = np.asarray(arrays.get(key))
        if value.shape != (len(clock), 7) or not np.isfinite(value).all() or not np.allclose(np.linalg.norm(value[:, 3:], axis=1), 1, atol=1e-5):
            raise ValueError("Missing/invalid source world pose: " + key)
    targets, mapping = {}, {}
    for hand, side in enumerate(("left", "right")):
        column = 13 + hand
        channel = channels[column]
        expected_actuator = f"h1/robotiq_2f85_{side}/fingers_actuator"
        if (channel.get("name") != side + "_gripper"
                or channel.get("kind") != "normalized_gripper_command"
                or channel.get("range") != [0, 1]
                or channel.get("actuators") != [expected_actuator]):
            raise ValueError("Unverified native gripper identity/range/order")
        command = action[:, column]
        # Verified pilot commands are exactly binary. Do not guess a mapping
        # for another source's continuous or reordered command vector.
        if not np.isin(command, [0., 1.]).all():
            raise ValueError("This verified pilot mapping requires binary gripper commands")
        targets[side] = np.where(command == 1, GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD)
        mapping[side] = {"source_channel": f"source/action[:,{column}]",
            "source_open": 0., "source_closed": 1., "native_actuator_open": 0., "native_actuator_closed": 255.,
            "native_gripper_mode": "discrete normalized command; GripperConfig.discrete=True",
            "reachy_open_joint_target_rad": GRIPPER_OPEN_RAD,
            "reachy_closed_joint_target_rad": GRIPPER_CLOSED_RAD,
            "reachy_joint": ("l" if side == "left" else "r") + "_hand_finger",
            "semantics": "derived desired joint target; not issued control, measured aperture, or contact label",
            "resampling": "none; original consumed source action row retained"}
    details = {"source_dataset": "bigym", "source_replay_completed": True,
        "source_task_success": metadata.get("source_task_success"),
        "source_task_success_is_not_reachy_success": True,
        "clock": "actual native mjData.time, post-action state; unchanged through IK",
        "base": "H1 pelvis world pose; only measured XY/yaw used, no base optimization",
        "hands": {side: {"channel": key, "native_site": metadata.get("pinch_sites", {}).get(side),
                  "mapping": "native pinch origin to Reachy arm_tip origin; constant initial rotation only",
                  "physical_pad_calibration_verified": False} for side, key in HAND_KEYS.items()},
        "grippers": mapping,
        "unmapped_source_channels": ["source/cameras/head/pose"],
        "head_neck_semantics": "H1 camera pose remains a source observation; no Reachy neck target inferred",
        "source_task_objects": list(TASK_OBJECTS),
        "source_gripper_evidence": [SOURCE_BLOB + path for path in (
            "bigym/action_modes.py", "bigym/robots/gripper.py", "bigym/robots/config.py",
            "bigym/robots/configs/robotiq.py", "bigym/envs/xmls/robotiq_2f85/2f85.xml")],
        "native_polarity_evidence": {"observed_left_pad_origin_distance_open_m": 0.09784520395158099,
            "observed_left_pad_origin_distance_closed_m": 0.014675922653635421,
            "scope": "same native replay, last stable binary open/closed commands; pad-origin distance, not aperture calibration"},
        "reachy_endpoint_evidence": "Existing reachy_retarget/contact.py and retarget.py convention; within pinned URDF gripper limits",
        "source_record_preserved": True, "object_geometry_or_trajectory_optimization": False}
    return targets, details


def geometric_preflight(record, robot):
    """All-frame necessary reach bound from the actual pinned kinematic chain.

    Triangle inequality gives a conservative maximum shoulder-to-tool distance.
    Exceeding it proves position infeasibility for the chosen fixed base path;
    satisfying it does not establish IK, orientation, collision, or dynamics.
    """
    from ..pose_retarget import references
    arrays = record["arrays"]
    home = robot.fk(robot.q)
    model, data = robot.r.model, robot.r.data
    base, targets, placement, attachments, _ = references(arrays, home, BASE_KEY, HAND_KEYS)
    yaw = np.arctan2(base[:, 1, 0], base[:, 0, 0])
    rotations = Rotation.from_euler("z", yaw[:, None]).as_matrix()
    base_joint = model.getJointId("root_joint")
    origin = data.oMi[base_joint]
    distances, bounds, shoulders, reports = [], [], [], {}
    for hand, side in enumerate(("left", "right")):
        shoulder_name = ("l" if hand == 0 else "r") + "_shoulder_pitch"
        shoulder = model.getJointId(shoulder_name)
        if shoulder == 0 or shoulder >= model.njoints:
            raise ValueError("Missing pinned Reachy shoulder joint")
        frame = model.frames[robot.r.hands[hand]]
        current = int(frame.parentJoint)
        length = float(np.linalg.norm(frame.placement.translation))
        chain = []
        visited = set()
        while current != shoulder:
            if current == 0 or current in visited:
                raise ValueError("Reachy tool frame is not a descendant of its shoulder")
            visited.add(current)
            chain.append(str(model.names[current]))
            length += float(np.linalg.norm(model.jointPlacements[current].translation))
            current = int(model.parents[current])
        shoulder_local = origin.rotation.T @ (data.oMi[shoulder].translation - origin.translation)
        world = np.einsum("nij,j->ni", rotations, shoulder_local)
        world[:, :2] += base[:, :2, 3]
        distance = np.linalg.norm(targets[:, hand, :3, 3] - world, axis=1)
        lower_bound = np.maximum(distance - length, 0.)
        distances.append(distance); bounds.append(length); shoulders.append(world)
        worst = int(np.argmax(distance))
        reports[side] = {"shoulder_joint": shoulder_name, "tool_frame": frame.name,
            "chain_joints_after_shoulder": chain, "maximum_geometric_reach_m": length,
            "shoulder_height_m": float(shoulder_local[2]),
            "target_height_range_m": [float(targets[:, hand, 2, 3].min()), float(targets[:, hand, 2, 3].max())],
            "maximum_required_reach_m": float(distance.max()), "outside_reach_sphere_frames": int((lower_bound > 1e-6).sum()),
            "position_error_lower_bound_max_m": float(lower_bound.max()),
            "frames_with_proven_error_over_2cm": int((lower_bound > .02).sum()),
            "worst_frame_index": worst, "worst_frame_time_s": float(arrays[CLOCK_KEY][worst])}
    raw = {"source_clock_s": np.asarray(arrays[CLOCK_KEY]), "target_hand_matrix": targets,
        "source_base_reference_matrix": base, "shoulder_world_xyz": np.stack(shoulders, axis=1),
        "required_reach_m": np.stack(distances, axis=1), "maximum_geometric_reach_m": np.asarray(bounds),
        "placement": placement, "tool_rotation_attachments": attachments}
    report = {"schema": "reachy-bigym-geometric-preflight-v1", "frames": len(arrays[CLOCK_KEY]), "hands": reports,
        "proven_infeasible_at_2cm": any(value["position_error_lower_bound_max_m"] > .02 for value in reports.values()),
        "status": "necessary_bound_only", "kinematic_passed": False, "physics_validated": False,
        "method": "sum of pinned shoulder-to-arm_tip rigid link lengths; all measured base XY/yaw frames held fixed",
        "height_optimized": False, "base_path_optimized": False, "object_trajectories_optimized": False,
        "full_ik_requested_despite_geometry_failure": True}
    return raw, report


def retarget(record, root, dataset, episode_id, attempt):
    """Preflight all frames, then retain the complete shared-engine IK attempt."""
    from ..robot import Robot
    from ..pose_retarget import retarget as solve
    path = Path(attempt)
    path.mkdir(parents=True, exist_ok=True)
    artifact, report_path = path / "geometric-preflight.npz", path / "geometric-preflight.json"
    if artifact.exists() or report_path.exists():
        raise FileExistsError("Immutable BiGym preflight artifacts already exist")
    grippers, mapping = prepare(record)
    robot = Robot(path / "preflight-runtime-identity")
    raw, preflight = geometric_preflight(record, robot)
    with artifact.open("xb") as stream:
        np.savez_compressed(stream, **raw)
    preflight["raw_artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    preflight["source_sequence"] = record["metadata"].get("source_sequence")
    preflight["source_group"] = record["metadata"].get("source_group")
    report_path.write_text(json.dumps(preflight, indent=2, allow_nan=False) + "\n")
    mapping["geometric_preflight"] = preflight
    mapping["geometric_preflight_artifact"] = str(artifact)
    result = solve(record, root, dataset, episode_id, attempt,
        clock_key=CLOCK_KEY, base_key=BASE_KEY, hand_keys=HAND_KEYS,
        gripper_targets=grippers, mapping_metadata=mapping)
    return {**result, "geometric_preflight": preflight, "geometric_preflight_artifact": str(artifact)}
