"""Offline control representations with observed states and commands separated.

No simulator, SDK, network, interpolation or implicit control frequency is used.
State differences are observed transitions, never relabelled command targets.
"""

import numpy as np
from scipy.spatial.transform import Rotation


SCHEMA = "reachy-control-views-v1"
GROUPS = ("left_arm", "right_arm", "neck", "base", "left_gripper", "right_gripper")
POSE_CHANNELS = {
    "left_arm": "observation/left_tcp_pose",
    "right_arm": "observation/right_tcp_pose",
    "neck": "observation/head_pose",
    "base": "observation/base_pose",
}


def _clock(value):
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or len(value) < 2 or not np.isfinite(value).all() or np.any(np.diff(value) <= 0):
        raise ValueError("Need at least two finite strictly increasing actual timestamps")
    return value


def pose_matrices(poses):
    """Convert XYZ + wxyz to rigid matrices; never silently fix bad quaternions."""
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 7 or not np.isfinite(poses).all():
        raise ValueError("Expected finite N x 7 XYZ/wxyz poses")
    if not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Pose quaternion must be unit wxyz")
    matrices = np.broadcast_to(np.eye(4), (len(poses), 4, 4)).copy()
    matrices[:, :3, 3] = poses[:, :3]
    matrices[:, :3, :3] = Rotation.from_quat(poses[:, [4, 5, 6, 3]]).as_matrix()
    return matrices


def _poses(matrices):
    return np.column_stack((matrices[:, :3, 3], Rotation.from_matrix(matrices[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]))


def _inverse(matrices):
    inverse = np.broadcast_to(np.eye(4), matrices.shape).copy()
    inverse[:, :3, :3] = matrices[:, :3, :3].transpose(0, 2, 1)
    inverse[:, :3, 3] = -np.einsum("nij,nj->ni", inverse[:, :3, :3], matrices[:, :3, 3])
    return inverse


def se3_log(matrices):
    """Principal SE(3) logarithm, angular-first [rotation_vector, translation_log]."""
    matrices = np.asarray(matrices, float)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or not np.isfinite(matrices).all():
        raise ValueError("Expected finite N x 4 x 4 transforms")
    if (not np.allclose(matrices[:, 3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(matrices[:, :3, :3].transpose(0, 2, 1) @ matrices[:, :3, :3], np.eye(3), atol=1e-7)
            or not np.allclose(np.linalg.det(matrices[:, :3, :3]), 1, atol=1e-7)):
        raise ValueError("SE(3) logarithm requires rigid transforms")
    omega = Rotation.from_matrix(matrices[:, :3, :3]).as_rotvec()
    theta = np.linalg.norm(omega, axis=1)
    skew = np.zeros((len(omega), 3, 3))
    skew[:, 0, 1], skew[:, 0, 2] = -omega[:, 2], omega[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = omega[:, 2], -omega[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -omega[:, 1], omega[:, 0]
    factor = np.empty(len(theta))
    small = theta < 1e-5
    factor[small] = 1 / 12 + theta[small] ** 2 / 720
    large = theta[~small]
    factor[~small] = (1 - .5 * large / np.tan(.5 * large)) / large ** 2
    inverse_jacobian = np.eye(3) - .5 * skew + factor[:, None, None] * (skew @ skew)
    rho = np.einsum("nij,nj->ni", inverse_jacobian, matrices[:, :3, 3])
    return np.column_stack((omega, rho))


def pose_transition_views(poses, timestamps):
    """Observed pose transitions; left world delta differs from XYZ subtraction."""
    timestamps = _clock(timestamps)
    transforms = pose_matrices(poses)
    if len(transforms) != len(timestamps):
        raise ValueError("Pose rows do not match timestamps")
    current, following = transforms[:-1], transforms[1:]
    body_delta = _inverse(current) @ following
    world_delta = following @ _inverse(current)
    dt = np.diff(timestamps)
    translation = following[:, :3, 3] - current[:, :3, 3]
    return {
        "absolute": np.asarray(poses, float).copy(),
        "observed_next_state": np.asarray(poses, float)[1:].copy(),
        "delta_body": _poses(body_delta),
        "delta_world_left": _poses(world_delta),
        "translation_difference_world": translation,
        "interval_linear_velocity_world": translation / dt[:, None],
        "interval_linear_velocity_body_axes": np.einsum("nij,nj->ni", current[:, :3, :3].transpose(0, 2, 1), translation) / dt[:, None],
        "interval_se3_log_rate_body": se3_log(body_delta) / dt[:, None],
        "interval_se3_log_rate_world_spatial": se3_log(world_delta) / dt[:, None],
    }


def joint_transition_views(positions, timestamps, periods=None):
    """Preserve raw coordinate differences and optional declared periodic deltas."""
    timestamps = _clock(timestamps)
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or len(positions) != len(timestamps) or not np.isfinite(positions).all():
        raise ValueError("Expected finite joint positions aligned to timestamps")
    raw = np.diff(positions, axis=0)
    delta = raw.copy()
    if periods is not None:
        periods = np.asarray(periods, float)
        if periods.shape != (positions.shape[1],) or not np.isfinite(periods).all() or np.any(periods < 0):
            raise ValueError("Joint periods must be a nonnegative finite value per column; zero means unwrapped")
        periodic = periods > 0
        delta[:, periodic] = (raw[:, periodic] + periods[periodic] / 2) % periods[periodic] - periods[periodic] / 2
    return {
        "absolute": positions.copy(), "observed_next_state": positions[1:].copy(),
        "raw_delta": raw, "coordinate_delta": delta,
        "interval_velocity": delta / np.diff(timestamps)[:, None],
    }


def _joint_groups(metadata, groups):
    names = list(metadata.get("joint_names", []))
    if len(set(names)) != len(names) or any(not isinstance(name, str) for name in names):
        raise ValueError("joint_names must be unique strings in observation column order")
    mapping, descriptions = {}, {}
    for group in groups:
        declared = metadata.get("joint_groups", {}).get(group, [])
        if not declared:
            continue
        if all(isinstance(index, (int, np.integer)) and not isinstance(index, bool) for index in declared):
            indices = [int(index) for index in declared]
        elif all(isinstance(name, str) for name in declared):
            if any(name not in names for name in declared):
                raise ValueError("Joint group references an unknown joint")
            indices = [names.index(name) for name in declared]
        else:
            raise ValueError("Joint groups must contain indices or names, not mixed values")
        if len(set(indices)) != len(indices) or any(index < 0 or index >= len(names) for index in indices):
            raise ValueError("Invalid or duplicate joint group index")
        mapping[group] = indices
        descriptions[group] = {"indices": indices, "joint_names": [names[index] for index in indices]}
    return names, mapping, descriptions


def build_control_views(arrays, metadata, *, groups=GROUPS):
    """Return numeric views and complete semantics without modifying the input.

    Recorded desired commands are optional and require command_semantics entries:
    {channel: {kind, representation, timing, joint_names OR group/frame}}.
    No observed next state, data.ctrl, or unapplied action is promoted to an EE
    position/velocity target. Missing command families remain explicit.
    """
    groups = tuple(groups)
    if len(set(groups)) != len(groups) or any(group not in GROUPS for group in groups):
        raise ValueError("Unknown or duplicate control group")
    output, semantics, missing = {}, {}, []
    time_value = arrays.get("timestamp", arrays.get("time_s"))
    timestamps = _clock(time_value) if time_value is not None else None
    count = len(timestamps) if timestamps is not None else None
    if timestamps is not None:
        output.update(timestamp=timestamps.copy(), transition_start_s=timestamps[:-1].copy(),
                      transition_end_s=timestamps[1:].copy(), transition_dt_s=np.diff(timestamps))
    else:
        missing.append("timestamp: interval deltas/rates cannot be generated from an invented clock")

    def checked(channel, width=None):
        nonlocal count
        value = np.asarray(arrays[channel], float)
        if value.ndim < 2 or not np.isfinite(value).all() or (width is not None and value.shape[-1] != width):
            raise ValueError("Invalid observed channel: " + channel)
        if count is None:
            count = len(value)
        if len(value) != count:
            raise ValueError("Observed channel row count differs: " + channel)
        return value

    names, group_indices, group_details = _joint_groups(metadata, groups)
    positions = checked("observation/joint_position", len(names)) if "observation/joint_position" in arrays else None
    velocities = checked("observation/joint_velocity", len(names)) if "observation/joint_velocity" in arrays else None
    if positions is not None and (positions.ndim != 2 or not names):
        raise ValueError("Joint observations require named scalar columns")
    if velocities is not None and velocities.ndim != 2:
        raise ValueError("Joint velocities require named scalar columns")
    period_map = metadata.get("joint_periods", {})
    if not isinstance(period_map, dict) or any(name not in names for name in period_map):
        raise ValueError("joint_periods must explicitly map known joint names to periods")
    for group in groups:
        if group not in group_indices or positions is None:
            missing.append("observation/joints/" + group)
            continue
        indexes = group_indices[group]
        group_periods = [period_map.get(names[index], 0) for index in indexes]
        values = joint_transition_views(positions[:, indexes], timestamps, group_periods) if timestamps is not None else {"absolute": positions[:, indexes].copy()}
        for name, value in values.items():
            key = "observed/joints/" + group + "/" + name
            output[key] = value
            semantics[key] = {
                "role": "observed_current_state" if name == "absolute" else "observed_next_state" if name == "observed_next_state" else "derived_observed_transition",
                "timeline": "samples" if name == "absolute" else "intervals",
                "joint_names": group_details[group]["joint_names"],
                "periods": group_periods if name in {"coordinate_delta", "interval_velocity"} else None,
                "is_recorded_command": False,
            }
        if velocities is not None:
            key = "observed/joints/" + group + "/recorded_velocity"
            output[key] = velocities[:, indexes].copy()
            semantics[key] = {"role": "observed_velocity", "timeline": "samples", "is_recorded_command": False}
        else:
            missing.append("observation/joints/" + group + "/recorded_velocity")

    base = pose_matrices(checked(POSE_CHANNELS["base"], 7)) if POSE_CHANNELS["base"] in arrays else None
    for group, channel in POSE_CHANNELS.items():
        if group not in groups:
            continue
        if channel not in arrays:
            missing.append(channel)
            continue
        poses = checked(channel, 7)
        matrices = pose_matrices(poses)
        frames = {"world": poses}
        if base is not None and group != "base":
            frames["base"] = _poses(_inverse(base) @ matrices)
        for frame, frame_poses in frames.items():
            views = pose_transition_views(frame_poses, timestamps) if timestamps is not None else {"absolute": frame_poses.copy()}
            for name, value in views.items():
                label = name.replace("world", "reference") if frame == "base" else name
                key = "observed/poses/" + group + "/" + frame + "/" + label
                output[key] = value
                semantics[key] = {
                    "role": "observed_current_state" if name == "absolute" else "observed_next_state" if name == "observed_next_state" else "derived_observed_transition",
                    "reference_frame": "world" if frame == "world" else "moving_base_at_each_sample",
                    "timeline": "samples" if name == "absolute" else "intervals",
                    "source_channel": channel, "is_recorded_command": False,
                    "quaternion_order": "wxyz" if name in {"absolute", "observed_next_state", "delta_body", "delta_world_left"} else None,
                }
        stem = channel[:-5]
        for suffix in ("_twist_world", "_twist_base", "_twist_relative_base"):
            source = stem + suffix
            if source not in arrays:
                continue
            key = "observed/poses/" + group + "/recorded" + suffix
            output[key] = checked(source, 6).copy()
            semantics[key] = {"role": "observed_velocity", "source_channel": source, "timeline": "samples", "component_order": "angular_linear",
                              "frame": {"_twist_world": "world_axes_at_body_origin", "_twist_base": "absolute_velocity_in_base_axes", "_twist_relative_base": "motion_relative_to_moving_base"}[suffix], "is_recorded_command": False}

    if "observation/base_twist" in arrays:
        source = "observation/base_twist"
        key = "observed/base/recorded_twist_local"
        output[key] = checked(source, 3).copy()
        semantics[key] = {"role": "observed_velocity", "source_channel": source, "timeline": "samples",
                          "component_order": ["vx", "vy", "wz"], "frame": "base_axes", "is_recorded_command": False}

    command_coverage = {group: set() for group in groups}
    for channel, declaration in metadata.get("command_semantics", {}).items():
        if channel not in arrays:
            missing.append(channel)
            continue
        if not isinstance(declaration, dict) or not all(declaration.get(key) for key in ("kind", "representation", "timing")):
            raise ValueError("Recorded command requires kind, representation and timing")
        kind = declaration["kind"]
        representation = declaration["representation"]
        value = np.asarray(arrays[channel], float)
        if value.ndim != 2 or not np.isfinite(value).all() or count is None or len(value) not in {count, count - 1}:
            raise ValueError("Recorded command rows must align with samples or intervals")
        covered = []
        if kind in {"joint_position", "joint_velocity"}:
            if representation not in ({"absolute", "delta"} if kind == "joint_position" else {"absolute"}):
                raise ValueError("Unverified joint command representation")
            if representation == "delta" and not declaration.get("reference"):
                raise ValueError("Delta joint command needs its recorded reference semantics")
            command_names = declaration.get("joint_names", [])
            if (len(command_names) != value.shape[1] or len(set(command_names)) != len(command_names)
                    or any(name not in names for name in command_names)):
                raise ValueError("Joint command columns need verified ordered joint_names")
            for group, description in group_details.items():
                if set(description["joint_names"]) <= set(command_names):
                    covered.append(group)
        elif kind in {"pose", "twist", "base_twist"}:
            group = declaration.get("group")
            if group not in groups or not declaration.get("frame"):
                raise ValueError("Cartesian command requires an explicit group and frame")
            if kind == "pose":
                if representation not in {"absolute", "delta_body", "delta_world_left", "delta_reference_left"}:
                    raise ValueError("Unverified pose command representation")
                if representation != "absolute" and not declaration.get("reference"):
                    raise ValueError("Delta pose command needs its recorded reference semantics")
                if declaration.get("quaternion_order") != "wxyz":
                    raise ValueError("Pose command quaternion order must be explicitly wxyz")
                pose_matrices(value)
            elif kind == "base_twist":
                if group != "base" or representation != "absolute" or value.shape[1] != 3 or declaration.get("component_order") != "vx_vy_wz":
                    raise ValueError("Planar base twist command needs base group and vx_vy_wz order")
            elif representation != "absolute" or value.shape[1] != 6 or declaration.get("component_order") not in {"angular_linear", "linear_angular"}:
                raise ValueError("Twist command requires six columns and explicit component order")
            covered.append(group)
        else:
            raise ValueError("Unsupported recorded command kind: " + str(kind))
        key = "recorded_commands/" + channel
        output[key] = value.copy()
        semantics[key] = dict(declaration, source_channel=channel, role="recorded_command", is_recorded_command=True,
                              timeline="samples" if len(value) == count else "intervals")
        # Convert an actual recorded target against its synchronous pre-action
        # observation. Future observed states are never used by this path.
        converted = {}
        if declaration["timing"] == "pre_action":
            if kind == "joint_position" and positions is not None:
                indexes = [names.index(name) for name in command_names]
                current = positions[:len(value), indexes]
                if representation == "absolute":
                    converted["delta_from_observed_current"] = value - current
                elif declaration.get("reference") == "observed_current":
                    converted["absolute"] = current + value
            elif kind == "pose" and declaration["frame"] in {"world", "base"}:
                observation_key = "observed/poses/" + declaration["group"] + "/" + declaration["frame"] + "/absolute"
                if observation_key in output:
                    current = pose_matrices(output[observation_key][:len(value)])
                    supplied = pose_matrices(value)
                    if representation == "absolute":
                        converted["delta_body"] = _poses(_inverse(current) @ supplied)
                        converted["delta_reference_left"] = _poses(supplied @ _inverse(current))
                    elif declaration.get("reference") == "observed_current":
                        converted["absolute"] = _poses(current @ supplied if representation == "delta_body" else supplied @ current)
        for mode, converted_value in converted.items():
            converted_key = "command_views/" + channel + "/" + mode
            output[converted_key] = converted_value
            semantics[converted_key] = dict(declaration, role="derived_recorded_command_representation", source_channel=channel,
                                            representation=mode, is_recorded_command=False, derived_from_recorded_command=True,
                                            reference="observed_current", timeline="samples" if len(value) == count else "intervals")
        for group in covered:
            command_coverage[group].add(kind)
    for source in ("observation/actuator_control", "applied_controls"):
        if source not in arrays:
            continue
        value = np.asarray(arrays[source], float)
        actuator_names = metadata.get("actuator_names", [])
        if (value.ndim not in (2, 3) or not np.isfinite(value).all() or not actuator_names
                or value.shape[-1] != len(actuator_names) or count is None or len(value) not in {count, count - 1}):
            raise ValueError("Actuator controls need aligned samples/substeps and actuator_names")
        key = "recorded_commands/" + source
        output[key] = value.copy()
        semantics[key] = {"role": "recorded_actuator_control", "source_channel": source, "actuator_names": list(actuator_names),
                          "is_recorded_command": True, "cartesian_or_joint_target_semantics_inferred": False}
    for group in groups:
        required = ("joint_position", "joint_velocity", "pose", "twist") if group in POSE_CHANNELS else ("joint_position", "joint_velocity")
        for kind in required:
            if kind not in command_coverage[group]:
                missing.append("recorded_command/" + group + "/" + kind)
    return {
        "arrays": output,
        "metadata": {
            "schema": SCHEMA, "joint_groups": group_details, "joint_details": metadata.get("joint_details", []),
            "source_state_contract": metadata.get("state_contract"),
            "source_model_structure_sha256": metadata.get("model_structure_sha256"),
            "source_episode_id": metadata.get("episode_id"),
            "timed": timestamps is not None, "views": semantics, "missing_fields": missing,
            "command_coverage": {group: sorted(kinds) for group, kinds in command_coverage.items()},
            "definitions": {
                "delta_body": "inverse(T_current) @ T_next",
                "delta_world_left": "T_next @ inverse(T_current); this translation is not simply p_next - p_current",
                "interval_se3_log_rate": "Principal SE(3) log(delta) / actual dt, angular-first; derived finite-interval rate, not recorded instantaneous velocity",
                "periodic_joint_delta": "Only explicitly declared joint_periods are wrapped; shortest-arc motion is ambiguous at/above half a period; raw_delta is retained",
                "base_pose_views": "Pose relative to base at each sample; includes changes of the moving reference, unlike axes-only twist rotation",
                "observed_next_state": "A future observed state, never an inferred controller target",
            },
            "physics_validated_by_this_conversion": False,
        },
        "status": "views_derived_from_saved_channels", "missing_fields": missing,
    }
