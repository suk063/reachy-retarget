"""Read-only MuJoCo observations for robot and explicitly named task objects.

The caller must update position/velocity-dependent MuJoCo caches for the current
state before capture. This module never steps, forwards, resets, renders, writes
model/data arrays, or imports a robot SDK. Full replay state can include the
compiled scene, but direct object poses are restricted to the task manifest.
"""

import copy
import hashlib
import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


STATE_CONTRACT = "reachy-retarget-state-observations-v1"
ARM_SUFFIXES = ("shoulder_pitch", "shoulder_roll", "elbow_yaw", "elbow_pitch",
                "wrist_roll", "wrist_pitch", "wrist_yaw")
REACHY_JOINT_GROUPS = {
    "left_arm": tuple("l_" + name for name in ARM_SUFFIXES),
    "right_arm": tuple("r_" + name for name in ARM_SUFFIXES),
    "neck": ("neck_roll", "neck_pitch", "neck_yaw"),
    "left_gripper": ("l_hand_finger",), "right_gripper": ("r_hand_finger",),
}
REQUIRED_REACHY_JOINTS = tuple(name for names in REACHY_JOINT_GROUPS.values() for name in names)
_JOINT_TYPES = {0: ("free", 7, 6), 1: ("ball", 4, 3), 2: ("slide", 1, 1), 3: ("hinge", 1, 1)}


def _pose(position, rotation):
    return _poses(np.asarray(position).reshape(1, 3), np.asarray(rotation).reshape(1, 3, 3))[0]


def _poses(position, rotation):
    """Validate and convert an entire pose batch with one SciPy call."""
    position, rotation = np.asarray(position), np.asarray(rotation).reshape(-1, 3, 3)
    if (not np.isfinite(position).all() or not np.isfinite(rotation).all()
            or position.shape != (len(rotation), 3)
            or not np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6).all()):
        raise ValueError("Invalid pose cache; caller must synchronize current MuJoCo kinematics")
    return np.concatenate((position, Rotation.from_matrix(rotation).as_quat()[:, [3, 0, 1, 2]]), axis=1)


def _copy_metadata(value):
    """Detach known metadata containers without deepcopy's per-scalar memo work.

    Unknown model-identity types retain normal deepcopy semantics. The private
    template is never exposed or changed after construction.
    """
    kind = type(value)
    if kind is dict:
        return {key: _copy_metadata(item) for key, item in value.items()}
    if kind is list:
        return [_copy_metadata(item) for item in value]
    if kind is tuple:
        return tuple(_copy_metadata(item) for item in value)
    if kind in (str, int, float, bool, bytes, type(None)):
        return value
    return copy.deepcopy(value)


def _names(mapping, label):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"Explicit nonempty {label} mapping is required")
    if any(not isinstance(name, str) or not name or "/" in name or name in (".", "..") for name in mapping):
        raise ValueError(f"Invalid {label} channel label")
    return dict(mapping)


class StateObserver:
    """Capture per-frame arrays; stack identical keys along axis 0 for episodes.

    ``task_objects`` maps semantic IDs to one body root or a list of body roots.
    All descendants/parts are retained. Roles map to one of these semantic IDs;
    a multi-root role additionally requires ``{object: ID, body: BODY_NAME}``.
    Robot membership must be supplied as one subtree or an explicit body-ID list.
    Every robot/task scalar joint is recorded, including passive joints; named
    Reachy groups also expose absent neck/arm/gripper coordinates explicitly.
    """

    def __init__(self, model, *, task_objects, base_body, head_body, tcp_sites,
                 robot_body_root=None, robot_body_ids=None, object_roles=None,
                 cameras=None, joint_names=None, model_identity=None,
                 required_joint_names=REQUIRED_REACHY_JOINTS):
        self.model = model
        if (robot_body_root is None) == (robot_body_ids is None):
            raise ValueError("Specify exactly one robot_body_root or robot_body_ids")
        self._body_name = lambda i: model.body(i).name or f"body_{i:04d}"

        def body_id(name):
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) if isinstance(name, str) else int(name)
            if index <= 0 or index >= model.nbody:
                raise ValueError(f"Unknown/nonphysical body mapping: {name!r}")
            return index

        def descendants(root):
            result = {root}
            for child in range(root + 1, model.nbody):
                if int(model.body_parentid[child]) in result:
                    result.add(child)
            return result

        self.robot_ids = descendants(body_id(robot_body_root)) if robot_body_root is not None else {body_id(i) for i in robot_body_ids}
        if not self.robot_ids:
            raise ValueError("Robot body membership cannot be empty")
        self.base_id, self.head_id = body_id(base_body), body_id(head_body)
        if not {self.base_id, self.head_id} <= self.robot_ids:
            raise ValueError("Base and head must belong to the declared robot")
        if set(tcp_sites) != {"left", "right"}:
            raise ValueError("Both left and right TCP sites must be declared")
        self.tcp_ids = {}
        for side, name in tcp_sites.items():
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if index < 0 or int(model.site_bodyid[index]) not in self.robot_ids:
                raise ValueError(f"Unknown or nonrobot {side} TCP site: {name}")
            self.tcp_ids[side] = index

        self.objects, self.object_roots = {}, {}
        assigned = set()
        for label, value in _names(task_objects, "task_objects").items():
            roots = [body_id(name) for name in (value if isinstance(value, (tuple, list)) else [value])]
            if not roots or len(set(roots)) != len(roots):
                raise ValueError(f"Missing/duplicate task object roots: {label}")
            members = set().union(*(descendants(index) for index in roots))
            if members & self.robot_ids or members & assigned:
                raise ValueError(f"Task object overlaps robot or another task object: {label}")
            assigned |= members
            self.objects[label], self.object_roots[label] = sorted(members), roots
        self.body_ids = sorted(self.robot_ids | assigned)
        self.roles = {}
        object_roles = object_roles or {}
        if set(object_roles) - {"pickup", "receptacle"}:
            raise ValueError("Only pickup and receptacle are role aliases")
        for role in ("pickup", "receptacle"):
            value = object_roles.get(role)
            if value is None:
                continue
            label = value.get("object") if isinstance(value, dict) else value
            if label not in self.objects:
                raise ValueError(f"Role {role} is not a declared task object")
            roots = self.object_roots[label]
            if isinstance(value, dict) and "body" in value:
                index = body_id(value["body"])
                if index not in self.objects[label]:
                    raise ValueError(f"Role {role} body lies outside its task object")
            elif len(roots) == 1:
                index = roots[0]
            else:
                raise ValueError(f"Multi-root role {role} needs an explicit body")
            self.roles[role] = (label, index)

        all_scalar_ids = [i for i in range(model.njnt) if int(model.jnt_type[i]) in (2, 3)]
        scalar_ids = [i for i in all_scalar_ids if int(model.jnt_bodyid[i]) in self.robot_ids | assigned]
        labels = {i: model.joint(i).name or f"joint_{i:04d}" for i in range(model.njnt)}
        all_names = [labels[i] for i in scalar_ids]
        if len(set(all_names)) != len(all_names):
            raise ValueError("Ambiguous generated scalar joint names")
        if joint_names is not None:
            if len(joint_names) != len(all_names) or set(joint_names) != set(all_names):
                raise ValueError("joint_names must order every robot/task scalar joint without omission")
            scalar_ids = [{labels[i]: i for i in scalar_ids}[name] for name in joint_names]
        self.joint_ids, self.joint_names = scalar_ids, [labels[i] for i in scalar_ids]
        self.qadr = np.asarray([model.jnt_qposadr[i] for i in scalar_ids], dtype=int)
        self.vadr = np.asarray([model.jnt_dofadr[i] for i in scalar_ids], dtype=int)
        joint_details = []
        for index, name in zip(scalar_ids, self.joint_names):
            kind = _JOINT_TYPES[int(model.jnt_type[index])][0]
            unit = "rad" if kind == "hinge" else "m"
            joint_details.append(dict(name=name, model_joint_id=index, model_body_id=int(model.jnt_bodyid[index]),
                type=kind, qpos_address=int(model.jnt_qposadr[index]), dof_address=int(model.jnt_dofadr[index]),
                limited=bool(model.jnt_limited[index]), range=model.jnt_range[index].tolist(),
                units={"position": unit, "velocity": unit + "/s"}))
        actuator_details = [dict(name=model.actuator(i).name or f"actuator_{i:04d}", model_actuator_id=i,
                                transmission_type=int(model.actuator_trntype[i]),
                                transmission_ids=model.actuator_trnid[i].tolist(),
                                gain_type=int(model.actuator_gaintype[i]), bias_type=int(model.actuator_biastype[i]))
                            for i in range(model.nu)]
        directly_actuated = {int(model.actuator_trnid[i, 0]) for i in range(model.nu)
                             if int(model.actuator_trntype[i]) in (0, 1)}
        groups = {name: [i for i, joint in enumerate(self.joint_names) if joint in members]
                  for name, members in REACHY_JOINT_GROUPS.items()}
        groups.update(
            robot=[i for i, j in enumerate(scalar_ids) if int(model.jnt_bodyid[j]) in self.robot_ids],
            task_objects=[i for i, j in enumerate(scalar_ids) if int(model.jnt_bodyid[j]) in assigned],
            base=[i for i, j in enumerate(scalar_ids) if int(model.jnt_bodyid[j]) == self.base_id],
            directly_actuated=[i for i, j in enumerate(scalar_ids) if j in directly_actuated],
            no_direct_joint_actuator=[i for i, j in enumerate(scalar_ids) if j not in directly_actuated])
        self.object_joints = {}
        object_details = {}
        for label, members in self.objects.items():
            joints = [i for i in range(model.njnt) if int(model.jnt_bodyid[i]) in members]
            qa, va = [], []
            details = []
            for i in joints:
                kind, nq, nv = _JOINT_TYPES[int(model.jnt_type[i])]
                qstart, vstart = int(model.jnt_qposadr[i]), int(model.jnt_dofadr[i])
                qa.extend(range(qstart, qstart+nq)); va.extend(range(vstart, vstart+nv))
                details.append(dict(name=labels[i], type=kind, model_joint_id=i,
                                    qpos_address=qstart, qpos_width=nq, dof_address=vstart, dof_width=nv))
            self.object_joints[label] = (np.asarray(qa, dtype=int), np.asarray(va, dtype=int))
            object_details[label] = dict(root_body_ids=self.object_roots[label], body_ids=members,
                body_names=[self._body_name(i) for i in members], joints=details,
                pose_frame="world", part_channels={f"body_{i:04d}": self._body_name(i) for i in members})

        if cameras is None:
            cameras = {model.camera(i).name or f"camera_{i:04d}": i for i in range(model.ncam)
                       if int(model.cam_bodyid[i]) in self.robot_ids}
        self.camera_ids = {}
        for label, name in cameras.items():
            _names({label: name}, "cameras")
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) if isinstance(name, str) else int(name)
            if index < 0 or index >= model.ncam:
                raise ValueError(f"Unknown camera: {name}")
            self.camera_ids[label] = index
        missing_joints = sorted(set(required_joint_names) - set(self.joint_names))
        missing = [f"observation/{role}_pose" for role in ("pickup", "receptacle") if role not in self.roles]
        missing += [f"required_joint/{name}" for name in missing_joints]
        if not self.camera_ids:
            missing.append("observation/cameras/*/pose")
        self._metadata = dict(state_contract=STATE_CONTRACT, rgb_stored=False,
            observation_timing="caller-synchronized-current-MuJoCo-state",
            joint_names=self.joint_names, joint_details=joint_details, joint_groups=groups,
            scalar_joint_scope="robot_and_task_object_subtrees", scalar_joint_coverage_complete=True,
            model_scalar_joint_count=len(all_scalar_ids), compiled_scalar_joint_coverage_complete=len(scalar_ids)==len(all_scalar_ids),
            required_joint_names=list(required_joint_names),
            missing_required_joints=missing_joints, required_joint_coverage_complete=not missing_joints,
            body_names=[self._body_name(i) for i in self.body_ids], body_ids=self.body_ids,
            robot_body_ids=sorted(self.robot_ids), base_body=self._body_name(self.base_id), head_body=self._body_name(self.head_id),
            tcp_sites=dict(tcp_sites), objects=object_details, task_objects=dict(task_objects),
            task_object_coverage_complete=True,
            object_roles={role: {"object": label, "body": self._body_name(index)} for role, (label, index) in self.roles.items()},
            excluded_scene_body_count=model.nbody-1-len(self.body_ids),
            cameras={label: {"model_camera_id": index, "name": model.camera(index).name,
                              "pose_convention": "world optical: x right, y down, z forward; MuJoCo xmat @ diag(1,-1,-1)",
                              "fovy_degrees": float(model.cam_fovy[index])} for label, index in self.camera_ids.items()},
            actuator_names=[row["name"] for row in actuator_details], actuator_details=actuator_details,
            model_dimensions={"nq": model.nq, "nv": model.nv, "nu": model.nu, "njnt": model.njnt, "nbody": model.nbody},
            model_identity=copy.deepcopy(model_identity), mujoco_version=mujoco.__version__,
            clock={"source": "mjData.time", "unit": "s", "physics_timestep_s": float(model.opt.timestep),
                   "sampling_rate_hz": None, "resampling_performed": False},
            quaternion_order="wxyz", physics_state_spec="mjSTATE_INTEGRATION",
            physics_state_size=mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION),
            units={"length": "m", "angle": "rad", "time": "s", "linear_velocity": "m/s", "angular_velocity": "rad/s"},
            frames={"poses": "world; *_pose_base is relative to current base",
                    "base_twist": "vx,vy,wz at base origin in base axes",
                    "twist_order": ["wx", "wy", "wz", "vx", "vy", "vz"],
                    "*_twist_world": "absolute origin velocity in world axes",
                    "*_twist_base": "absolute origin velocity expressed in base axes; no base-motion subtraction",
                    "*_twist_relative_base": "relative motion after subtracting base rotation/translation, expressed in base axes"},
            observation_sources={"joint_position": "mjData.qpos scalar addresses", "joint_velocity": "mjData.qvel scalar addresses",
                "actuator_control": "mjData.ctrl snapshot; not inferred desired joint/EEF commands",
                "poses": "current MuJoCo body/site/camera caches",
                "twists": "mj_objectVelocity: mjOBJ_XBODY at xpos/xmat body frame, mjOBJ_SITE at site frame; angular then linear"},
            control_labels={"desired_joint_targets": "not provided", "desired_ee_targets": "not provided",
                            "controller_state": "not provided", "substep_controls": "not provided"},
            missing_fields=missing, physics_validated=False, policy_ready=False)
        structural = {key: self._metadata[key] for key in ("joint_details", "body_ids", "objects", "actuator_details", "model_dimensions")}
        self._metadata["model_structure_sha256"] = hashlib.sha256(json.dumps(structural, sort_keys=True).encode()).hexdigest()
        self._metadata = copy.deepcopy(self._metadata)

    @property
    def metadata(self):
        """Independent snapshot; callers cannot mutate the observer's template."""
        return _copy_metadata(self._metadata)

    def capture(self, data):
        """Return one detached observation and metadata snapshot."""
        return {"arrays": self.capture_arrays(data), "metadata": self.metadata,
                "missing_fields": list(self._metadata["missing_fields"]), "status": "observed_simulator_state"}

    def capture_arrays(self, data):
        """Capture all numeric channels without repeating episode metadata.

        Use StateObservationBuffer for an episode with one privately owned
        metadata snapshot. The default capture API still detaches metadata for
        each independently returned frame.
        """
        model = self.model
        if data.qpos.shape != (model.nq,) or data.qvel.shape != (model.nv,) or data.xpos.shape != (model.nbody, 3):
            raise ValueError("MjData dimensions differ from observer model")
        if not np.isfinite(data.time) or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("Nonfinite simulator state")
        base_position = data.xpos[self.base_id].copy()
        base_rotation = data.xmat[self.base_id].reshape(3, 3).copy()
        endpoints = [(side + "_tcp", mujoco.mjtObj.mjOBJ_SITE, index,
                      data.site_xpos[index], data.site_xmat[index].reshape(3, 3))
                     for side, index in self.tcp_ids.items()] + [
                     ("head", mujoco.mjtObj.mjOBJ_XBODY, self.head_id,
                      data.xpos[self.head_id], data.xmat[self.head_id].reshape(3, 3))]
        cameras = list(self.camera_ids.items())
        # Include bodies, both TCPs/head, base-relative endpoints and optical
        # cameras in one checked conversion. No raw pose cache skips validation.
        positions = np.concatenate((data.xpos[self.body_ids],
            np.stack([row[3] for row in endpoints]),
            np.stack([base_rotation.T @ (row[3]-base_position) for row in endpoints]),
            np.asarray([data.cam_xpos[index] for _, index in cameras]).reshape(-1, 3)))
        rotations = np.concatenate((data.xmat[self.body_ids].reshape(-1, 3, 3),
            np.stack([row[4] for row in endpoints]),
            np.stack([base_rotation.T @ row[4] for row in endpoints]),
            np.asarray([data.cam_xmat[index].reshape(3, 3) @ np.diag([1., -1., -1.])
                        for _, index in cameras]).reshape(-1, 3, 3)))
        poses = _poses(positions, rotations)
        count = len(self.body_ids)
        pose_cache = {index: poses[row] for row, index in enumerate(self.body_ids)}
        twist_cache = {}
        for i in self.body_ids:
            value = np.empty(6)
            # mjOBJ_BODY uses the inertial/COM frame. XBODY is the body frame
            # whose origin/orientation are recorded in xpos/xmat above.
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, i, value, 0)
            twist_cache[i] = value
        base_world = twist_cache[self.base_id]
        base_local = np.empty(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, self.base_id, base_local, 1)
        physics = np.empty(self._metadata["physics_state_size"])
        mujoco.mj_getState(model, data, physics, mujoco.mjtState.mjSTATE_INTEGRATION)
        arrays = {"timestamp": np.asarray(data.time, dtype=np.float64),
            "observation/joint_position": data.qpos[self.qadr].copy(),
            "observation/joint_velocity": data.qvel[self.vadr].copy(),
            "observation/qpos": data.qpos.copy(), "observation/qvel": data.qvel.copy(),
            "observation/base_pose": pose_cache[self.base_id].copy(),
            "observation/base_twist": base_local[[3, 4, 2]].copy(),
            "observation/base_twist_world": base_world.copy(), "observation/base_twist_local": base_local.copy(),
            "observation/body_poses": poses[:count].copy(),
            "observation/body_twists_world": np.stack([twist_cache[i] for i in self.body_ids]),
            "observation/actuator_control": data.ctrl.copy(),
            "observation/actuator_force": data.actuator_force.copy(),
            "observation/actuator_velocity": data.actuator_velocity.copy(), "physics_state": physics}
        for row, (label, objtype, index, position, rotation) in enumerate(endpoints):
            world = np.empty(6)
            mujoco.mj_objectVelocity(model, data, objtype, index, world, 0)
            local = np.r_[base_rotation.T @ world[:3], base_rotation.T @ world[3:]]
            relative = np.r_[base_rotation.T @ (world[:3]-base_world[:3]),
                             base_rotation.T @ (world[3:]-base_world[3:]-np.cross(base_world[:3], position-base_position))]
            arrays[f"observation/{label}_pose"] = poses[count+row].copy()
            arrays[f"observation/{label}_pose_base"] = poses[count+len(endpoints)+row].copy()
            arrays[f"observation/{label}_twist_world"] = world
            arrays[f"observation/{label}_twist_base"] = local
            arrays[f"observation/{label}_twist_relative_base"] = relative
        for label, members in self.objects.items():
            roots = self.object_roots[label]
            if len(roots) == 1:
                arrays[f"observation/objects/{label}/pose"] = pose_cache[roots[0]].copy()
            for index in members:
                arrays[f"observation/objects/{label}/parts/body_{index:04d}/pose"] = pose_cache[index].copy()
                arrays[f"observation/objects/{label}/parts/body_{index:04d}/twist_world"] = twist_cache[index].copy()
            qa, va = self.object_joints[label]
            arrays[f"observation/objects/{label}/qpos"] = data.qpos[qa].copy()
            arrays[f"observation/objects/{label}/qvel"] = data.qvel[va].copy()
        for role, (_, index) in self.roles.items():
            arrays[f"observation/{role}_pose"] = pose_cache[index].copy()
        for row, (label, _) in enumerate(cameras):
            arrays[f"observation/cameras/{label}/pose"] = poses[count+2*len(endpoints)+row].copy()
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("Nonfinite observation cache")
        return arrays


def _stack_arrays(frames, metadata, missing_fields):
    if not frames:
        raise ValueError("At least one captured state is required")
    first = frames[0]
    if any(frame.keys() != first.keys() for frame in frames):
        raise ValueError("Observed model/task/channel contract changed between frames")
    arrays = {key: np.stack([frame[key] for frame in frames]) for key in first}
    if len(frames) > 1 and not np.all(np.diff(arrays["timestamp"]) > 0):
        raise ValueError("Captured timestamps must increase; duplicate states are not time samples")
    return {"arrays": arrays, "metadata": copy.deepcopy(metadata),
            "missing_fields": list(missing_fields), "status": "observed_simulator_sequence"}


class StateObservationBuffer:
    """Retain every numeric frame and one private episode metadata snapshot.

    Neither the metadata template nor stored rows are returned to callers.
    finish() produces independent stacked arrays and detached metadata. This
    removes identical per-frame metadata allocations without changing capture
    frequency, validation, channel names, numeric values or episode output.
    """
    def __init__(self, observer):
        self._observer = observer
        self._metadata = observer.metadata
        self._missing_fields = list(self._metadata["missing_fields"])
        self._frames = []

    def __len__(self):
        return len(self._frames)

    def capture(self, data):
        self._frames.append(self._observer.capture_arrays(data))

    def finish(self):
        return _stack_arrays(self._frames, self._metadata, self._missing_fields)


def stack_observations(frames):
    """Stack real captured rows without resampling, gap filling or fake fields."""
    frames = list(frames)
    if not frames:
        raise ValueError("At least one captured state is required")
    first = frames[0]
    for frame in frames:
        if frame["metadata"] != first["metadata"] or frame["arrays"].keys() != first["arrays"].keys():
            raise ValueError("Observed model/task/channel contract changed between frames")
    return _stack_arrays([frame["arrays"] for frame in frames], first["metadata"], first["missing_fields"])
