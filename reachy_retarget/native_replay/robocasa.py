"""Verified RoboCasa source FK from recorded states, never a dynamics rollout.

An isolated kinematic projection keeps every body/joint/site transform and joint
address. Meshes, contacts, actuators and appearance are deliberately absent.
Compiler-only inertials on otherwise massless moving bodies have no meaning as
physical parameters. This model must not be used for physical validation.
No network, renderer, robosuite import, or hardware SDK is used.
"""

import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..adapters.native_mobile import _joint_layout, _provenance, _require_recorded, normalize


BASE_SITE = "mobilebase0_center"
BASE_BODY = "mobilebase0_base"
EEF_SITE = "gripper0_right_grip_site"
EEF_BODY = "robot0_right_hand"
SOURCE_CONVENTION_URL = "https://github.com/ARISE-Initiative/robosuite/blob/v1.5.2/robosuite/robots/mobile_robot.py"
CONVERSION_URL = "https://github.com/robocasa/robocasa/blob/456174f62b89b8fca99eaaf33949c29fec9cfc2a/robocasa/scripts/dataset_scripts/convert_hdf5_lerobot.py"
MODALITY_STATE = {"base_position": (0, 3), "base_rotation": (3, 7),
                  "end_effector_position_relative": (7, 10), "end_effector_rotation_relative": (10, 14),
                  "gripper_qpos": (14, 16)}
MODALITY_ACTION = {"base_motion": (0, 4), "control_mode": (4, 5),
                   "end_effector_position": (5, 8), "end_effector_rotation": (8, 11), "gripper_close": (11, 12)}


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def compile_kinematic_model(xml):
    """Compile explicit MJCF transforms with unchanged qpos/qvel addressing.

    Reject implicit/default/frame/aligned-free transformations rather than
    silently changing their semantics. The returned model is FK-only.
    """
    source, joints, nq, nv = _joint_layout(xml)
    if source.find(".//default") is not None:
        raise ValueError("Kinematic projection requires explicit MJCF without defaults")
    if source.find(".//plugin") is not None:
        raise ValueError("Kinematic projection cannot interpret plugin transforms")
    compilers = source.findall("compiler")
    if len(compilers) > 1 or compilers and compilers[0].get("coordinate", "local") != "local":
        raise ValueError("Kinematic projection requires one local-coordinate compiler")
    attrs = dict(compilers[0].attrib) if compilers else {}
    # Preserve angle/eulerseq semantics. Other compiler behavior (asset loading,
    # inertial inference, body fusion) must not enter this FK projection.
    attrs = {key: attrs[key] for key in ("angle", "eulerseq") if key in attrs}
    attrs.update(autolimits="true", fusestatic="false", inertiafromgeom="false", alignfree="false")
    projection = ET.Element("mujoco", model="robocasa_recorded_state_fk_only")
    ET.SubElement(projection, "compiler", attrs)
    world = ET.SubElement(projection, "worldbody")
    scaffolding = []
    permitted = {"body", "joint", "freejoint", "inertial", "site", "camera"}

    def visit(original, target):
        for node in original:
            if node.tag not in permitted:
                continue
            if any(key in node.attrib for key in ("class", "childclass")):
                raise ValueError("Unresolved MJCF defaults in kinematic projection")
            child = ET.SubElement(target, node.tag, {k: v for k, v in node.attrib.items() if k != "material"})
            visit(node, child)
            if node.tag == "body" and any(x.tag in {"joint", "freejoint"} for x in child) and child.find("inertial") is None:
                ET.SubElement(child, "inertial", pos="0 0 0", mass="1", diaginertia="1 1 1")
                scaffolding.append(node.get("name"))

    visit(source.find("worldbody"), world)
    projected_xml = ET.tostring(projection, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(projected_xml)
    if (model.nq, model.nv, model.njnt) != (nq, nv, len(joints)):
        raise ValueError("Compiled projection changed recorded state dimensions")
    for index, joint in enumerate(joints):
        if (model.joint(index).name != joint["name"] or model.jnt_qposadr[index] != joint["qpos_address"]
                or model.jnt_dofadr[index] != joint["qvel_address"]):
            raise ValueError("Compiled projection changed recorded joint addressing")
    return model, {"source_model_xml_sha256": _digest(xml), "projection_xml_sha256": _digest(projected_xml),
                   "projection_xml": projected_xml, "joint_layout": joints,
                   "compiler_scaffolding_inertial_bodies": scaffolding,
                   "removed_asset_counts": {tag: len(source.findall("asset/" + tag)) for tag in ("mesh", "texture", "hfield")},
                   "physical_model": False, "source_asset_fidelity_verified": False}


def _pose(position, rotation):
    return np.r_[position, Rotation.from_matrix(np.asarray(rotation).reshape(3, 3)).as_quat()[[3, 0, 1, 2]]]


def _relative(position, rotation, base_position, base_rotation):
    return _pose(base_rotation.T @ (position - base_position), base_rotation.T @ rotation)


def _descendants(model, root):
    result = {root}
    for index in range(root + 1, model.nbody):
        if int(model.body_parentid[index]) in result:
            result.add(index)
    return sorted(result)


def _resolve(model, kind, name):
    identifier = mujoco.mj_name2id(model, kind, name)
    if identifier < 0:
        raise ValueError("Missing verified source frame: " + name)
    return identifier


def reconstruct(root, *, position_tolerance_m=1e-8, rotation_tolerance_rad=2e-6):
    """Return a common record after validating FK against genuine observations.

    The source simulator clock, including gaps, remains authoritative. Parquet's
    nominal clock and reordered action vectors are retained separately. Source
    EEF observables mix site position/body rotation; true TCP poses use the site
    for both, so their orientations must not be substituted for one another.
    """
    import pyarrow.parquet as parquet

    record = normalize(root, "robocasa")
    arrays, meta = record["arrays"], copy.deepcopy(record["metadata"])
    folder, _, provenance = _provenance(root, "robocasa")
    parquet_path = folder / "lerobot/data/chunk-000/episode_000000.parquet"
    modality_path = folder / "lerobot/meta/modality.json"
    for path in (parquet_path, modality_path):
        _require_recorded(path, provenance)
    modality = json.loads(modality_path.read_text())
    for group, expected, original_key in (("state", MODALITY_STATE, "observation.state"), ("action", MODALITY_ACTION, "action")):
        actual = modality.get(group, {})
        if set(actual) != set(expected) or any((actual[name].get("start"), actual[name].get("end")) != bounds
                                              or actual[name].get("original_key") != original_key for name, bounds in expected.items()):
            raise ValueError("Unverified source parquet modality layout")
    versions = {key: meta["source_configuration"].get(key) for key in ("robocasa_version", "robosuite_version", "mujoco_version")}
    if versions != {"robocasa_version": "0.5.1", "robosuite_version": "1.5.2", "mujoco_version": "3.3.1"}:
        raise ValueError("Source EEF observable convention requires the verified package versions")
    columns = ["observation.state", "action", "timestamp", "frame_index", "next.reward", "next.done"]
    table = parquet.read_table(parquet_path, columns=columns)
    observed = np.asarray(table["observation.state"].to_pylist(), dtype=float)
    action = np.asarray(table["action"].to_pylist(), dtype=float)
    count = len(arrays["timestamp"])
    if observed.shape != (count, 16) or action.shape != (count, 12) or not np.isfinite(observed).all() or not np.isfinite(action).all():
        raise ValueError("Source action/observation rows do not align with recorded states")
    frame_index = np.asarray(table["frame_index"].to_pylist())
    if not np.array_equal(frame_index, np.arange(count)):
        raise ValueError("Source parquet frame index does not align with state order")
    if any(not np.allclose(np.linalg.norm(observed[:, start:stop], axis=1), 1, atol=1e-5, rtol=0)
           for start, stop in ((3, 7), (10, 14))):
        raise ValueError("Invalid source observable quaternion")
    model, projection = compile_kinematic_model(meta["source_model_xml"])
    data = mujoco.MjData(model)
    base_site = _resolve(model, mujoco.mjtObj.mjOBJ_SITE, BASE_SITE)
    mobile_body = _resolve(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
    eef_site = _resolve(model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
    eef_body = _resolve(model, mujoco.mjtObj.mjOBJ_BODY, EEF_BODY)
    robot = _descendants(model, _resolve(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_base"))
    task = {name: _descendants(model, _resolve(model, mujoco.mjtObj.mjOBJ_BODY, spec["source_body"]))
            for name, spec in meta["objects"].items()}
    selected = set(robot).union(*(set(ids) for ids in task.values()))
    for index in selected:
        ancestor = index
        while ancestor:
            if model.body_mocapid[ancestor] >= 0:
                raise ValueError("Selected source frames depend on unrecorded mocap state")
            ancestor = int(model.body_parentid[ancestor])
    values = {}
    def append(name, value):
        values.setdefault(name, []).append(value.copy())
    for state in arrays["source/state"]:
        data.time = float(state[0])
        data.qpos[:] = state[1:1 + model.nq]
        data.qvel[:] = state[1 + model.nq:]
        # Isolated reference-state restoration, never a dynamic validation run.
        mujoco.mj_kinematics(model, data)
        bp, br = data.site_xpos[base_site], data.site_xmat[base_site].reshape(3, 3)
        ep, er = data.site_xpos[eef_site], data.site_xmat[eef_site].reshape(3, 3)
        legacy = data.xmat[eef_body].reshape(3, 3)
        append("source/base_pose", _pose(bp, br))
        append("source/mobile_base_body_pose", _pose(data.xpos[mobile_body], data.xmat[mobile_body]))
        append("source/right_tcp_pose", _pose(ep, er))
        append("source/right_tcp_pose_base", _relative(ep, er, bp, br))
        append("source/right_eef_observable_pose_base", _relative(ep, legacy, bp, br))
        append("source/robot_body_poses", np.asarray([_pose(data.xpos[i], data.xmat[i]) for i in robot]))
        for name, ids in task.items():
            append("objects/" + name + "/pose", _pose(data.xpos[ids[0]], data.xmat[ids[0]]))
            for index in ids:
                append("objects/" + name + "/parts/" + model.body(index).name + "/pose", _pose(data.xpos[index], data.xmat[index]))
    derived = {name: np.asarray(value) for name, value in values.items()}
    errors = {}
    for name, channel, position, quaternion in (("base", "source/base_pose", (0, 3), (3, 7)),
                                               ("eef_observable", "source/right_eef_observable_pose_base", (7, 10), (10, 14))):
        pose = derived[channel]
        errors[name + "_position_m"] = float(np.max(np.linalg.norm(pose[:, :3] - observed[:, slice(*position)], axis=1)))
        fk_rotation = Rotation.from_quat(pose[:, [4, 5, 6, 3]])
        obs_rotation = Rotation.from_quat(observed[:, slice(*quaternion)])
        errors[name + "_orientation_rad"] = float(np.max((fk_rotation.inv() * obs_rotation).magnitude()))
    gripper_names = ["gripper0_right_finger_joint1", "gripper0_right_finger_joint2"]
    robot_names = meta["source_joint_names"]
    errors["gripper_position_m"] = float(np.max(np.abs(arrays["source/joint_position"][:, [robot_names.index(name) for name in gripper_names]] - observed[:, 14:16])))
    errors["object_free_qpos_pose"] = float(np.max(np.abs(derived["objects/obj/pose"][:, :3] - arrays["objects/obj/pose"][:, :3])))
    free_rotation = Rotation.from_quat(arrays["objects/obj/pose"][:, [4, 5, 6, 3]])
    fk_rotation = Rotation.from_quat(derived["objects/obj/pose"][:, [4, 5, 6, 3]])
    errors["object_free_qpos_orientation_rad"] = float(np.max((free_rotation.inv() * fk_rotation).magnitude()))
    if any(value > (rotation_tolerance_rad if name.endswith("orientation_rad") else position_tolerance_m) for name, value in errors.items()):
        raise ValueError("Source FK disagrees with recorded observations: " + json.dumps(errors, sort_keys=True))
    arrays.update(derived)
    arrays.update({"source/observation_state": observed, "source/action": action,
                   "source/parquet_timestamp": np.asarray(table["timestamp"].to_pylist()),
                   "source/reward": np.asarray(table["next.reward"].to_pylist()),
                   "source/done": np.asarray(table["next.done"].to_pylist(), dtype=bool)})
    for name, ids in task.items():
        meta["objects"][name].update(decoded_pose_available=True, pose_frame="world", quaternion="wxyz",
                                     pose_source="recorded MuJoCo state and verified kinematic projection",
                                     part_body_names=[model.body(i).name for i in ids])
        selected_joints = [joint for joint in projection["joint_layout"] if joint["top_body"] == meta["objects"][name]["source_body"]]
        scalar = [joint for joint in selected_joints if joint["nq"] == joint["nv"] == 1]
        if scalar:
            arrays["objects/" + name + "/joint_velocity"] = arrays["source/state"][:, [1 + model.nq + joint["qvel_address"] for joint in scalar]].copy()
    meta["object_coverage"].update(decoded_object_pose_names=sorted(task), missing_task_object_pose_names=[], all_task_object_poses_decoded=True)
    meta["state_control_coverage"].update(source_actions_available=True, source_eef_poses_decoded=True,
                                        source_base_world_pose_decoded=True, source_action_quantity="Recorded reordered PandaOmron hybrid mobile-base/OSC action; not Reachy commands")
    meta.update(source_package_versions=versions, fk_runtime_mujoco_version=mujoco.__version__,
                source_base_frame={"site": BASE_SITE, "meaning": "Mobile support center; includes torso height, not chassis ground frame"},
                source_tcp_frame={"site": EEF_SITE, "position_and_rotation_from_same_site": True},
                source_eef_observable_convention={"position_site": EEF_SITE, "rotation_body": EEF_BODY,
                    "quaternion_order_in_parquet": "xyzw", "quaternion_order_in_derived_poses": "wxyz", "source_url": SOURCE_CONVENTION_URL},
                source_robot_body_names=[model.body(i).name for i in robot], source_action_modality=modality["action"],
                source_kinematic_reconstruction={**projection, "validation_max_errors": errors, "validated_observation_rows": count,
                    "position_tolerance_m": position_tolerance_m, "rotation_tolerance_rad": rotation_tolerance_rad,
                    "source_actions_executed": False, "physics_steps": 0, "original_states_and_model_unchanged": True})
    meta["derived_fields"].update({name: "FK from recorded qpos and exact source body/joint/site transforms; no physics integration" for name in derived})
    meta["derived_fields"].update({"source/action": "Unmodified acquired parquet action columns; source ordering described by source_action_modality",
                                   "source/parquet_timestamp": "Original nominal parquet clock, separate from recorded simulator time and its gaps"})
    meta["simulation_assumptions"] = ["Isolated FK-only projection retains explicit body/joint/site transforms and original state addressing",
        "Compiler-only inertials and omitted meshes/contacts are not physical simulation parameters",
        "Recorded source MuJoCo 3.3.1 states reconstructed in " + mujoco.__version__ + "; base/EEF observables verified independently",
        "Source references are not Reachy measured states, issued controls, or physical success"]
    missing = ["asset-faithful source dynamics replay", "source controller memory and per-physics-substep controls",
               "Reachy retargeting and physical validation"]
    meta["missing"] = missing
    return {"arrays": arrays, "metadata": meta, "status": "source_fk_verified_actions_preserved", "missing_fields": missing}
