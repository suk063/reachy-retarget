"""Verified source decoding and publisher predicates, separate from control search.

Task names may select a schema or success predicate here. Geometry/contact
planning and actuator optimization consume canonical numerical inputs instead.
"""

import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .episodes import pose_to_matrices
from . import dynamics_maniskill

TASK_OBJECTS = {"PickPlaceCan": "Can", "Lift": "cube", "NutAssemblySquare": "SquareNut",
                "Stack_D0": "cubeA", "Threading_D0": "needle_obj", "PickCube-v1": "cube"}



REFERENCES = {
    "PickCube-v1": dynamics_maniskill.REFERENCES["task_success"],
    "Lift": "https://github.com/ARISE-Initiative/robosuite/blob/v1.5.1/robosuite/environments/manipulation/lift.py#L382-L393",
    "Stack_D0": "https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/environments/manipulation/stack.py",
    "NutAssemblySquare": "https://github.com/ARISE-Initiative/robosuite/blob/v1.5.1/robosuite/environments/manipulation/nut_assembly.py#L335-L347",
    "Threading_D0": "https://github.com/NVlabs/mimicgen/blob/main/mimicgen/envs/robosuite/threading.py#L412-L428",
    "PickPlaceCan": "https://github.com/ARISE-Initiative/robosuite/blob/v1.5.1/robosuite/environments/manipulation/pick_place.py",
}



def _validate_source_arrays(arrays, object_ids, control_freq=None):
    times = np.asarray(arrays.get("time_s", []))
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or abs(float(times[0])) > 1e-9 or np.any(np.diff(times) <= 0)):
        raise ValueError("Source time must be finite, increasing, and start at zero")
    if control_freq is not None and not np.allclose(np.diff(times), 1. / control_freq, atol=1e-7, rtol=0):
        raise ValueError("Source time contradicts the declared control frequency")
    actions = np.asarray(arrays.get("source/action", []))
    if actions.shape != (len(times), 7):
        raise ValueError("Unverified source action shape; expected (T, 7)")
    if not np.isin(actions[:, 6], (-1., 1.)).all():
        raise ValueError("Unverified source gripper schema")
    if not np.isfinite(actions).all():
        raise ValueError("Nonfinite source action")
    prefixes = ["hand/right", *["objects/" + oid for oid in object_ids]]
    for prefix in prefixes:
        key = prefix + ("_pose" if prefix.startswith("hand/") else "/pose")
        poses = np.asarray(arrays.get(key, []))
        if (poses.shape != (len(times), 7) or not np.isfinite(poses).all()
                or not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1., atol=1e-4, rtol=0)):
            raise ValueError("Source pose requires aligned finite xyz+wxyz unit quaternions: " + key)
        validity_keys = [prefix + "/valid"]
        if prefix.startswith("hand/"):
            validity_keys.append(prefix + "_valid")
        for validity_key in validity_keys:
            if validity_key in arrays:
                valid = np.asarray(arrays[validity_key])
                if valid.shape != (len(times),) or not np.isin(valid, (True,)).all():
                    raise ValueError("Source trajectory contains invalid samples: " + validity_key)



def validate_task_contract(metadata, arrays, model, data, placement):
    """Validate known source schemas and derive predicates from the source scene.

    This deliberately supports the inspected Panda OSC_POSE releases only. The
    returned JSON record binds each predicate to the compiled source arena;
    arbitrary task labels/configurations cannot inherit unrelated constants.
    """
    import mujoco

    env = metadata.get("env_args", {})
    task, version = env.get("env_name"), env.get("env_version")
    versions = {"PickPlaceCan": "1.5.1", "Lift": "1.5.1", "NutAssemblySquare": "1.5.1",
                "Stack_D0": "1.4.1", "Threading_D0": "1.4.1"}
    if task not in versions or version != versions[task]:
        raise ValueError(f"Unverified dynamics task/version contract: {task}/{version}")
    kwargs = env.get("env_kwargs", {})
    if kwargs.get("robots") not in (["Panda"], "Panda") or kwargs.get("control_freq") != 20:
        raise ValueError("Unverified source robot or control frequency")
    controller = kwargs.get("controller_configs", {})
    if version == "1.5.1":
        if controller.get("type") != "BASIC" or set(controller.get("body_parts", {})) != {"right"}:
            raise ValueError("Unverified source composite-controller configuration")
        controller = controller["body_parts"]["right"]
        if controller.get("gripper", {}).get("type") != "GRIP":
            raise ValueError("Unverified source gripper-controller configuration")
    if controller.get("type") != "OSC_POSE":
        raise ValueError("Unverified source arm-controller configuration")
    for unsupported in ("single_object_mode", "object_type", "nut_type", "table_offset",
                        "table_height", "ring_size", "num_ring_geoms", "bin_size"):
        if unsupported in kwargs:
            raise ValueError("Unverified task-specific source configuration: " + unsupported)
    required = {"PickPlaceCan": {"Can"}, "Lift": {"cube"}, "NutAssemblySquare": {"SquareNut"},
                "Stack_D0": {"cubeA", "cubeB"}, "Threading_D0": {"needle_obj", "tripod_obj"}}[task]
    objects = metadata.get("objects", {})
    if set(objects) != required:
        raise ValueError("Source object scope differs from verified task contract")
    _validate_source_arrays(arrays, required, kwargs["control_freq"])
    for oid, spec in objects.items():
        if spec.get("pose_frame") != "world":
            raise ValueError("Unverified source object pose frame: " + oid)
        body = model.body(spec["body"])
        joint = model.joint(spec["joint"])
        if int(model.jnt_bodyid[joint.id]) != body.id or model.jnt_type[joint.id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Source object mapping must identify its free root joint: " + oid)
    placement = np.asarray(placement)
    inverse = np.linalg.inv(placement)

    def point(position):
        return (inverse @ np.r_[position, 1.])[:3]

    def vector_setting(key, default):
        values = np.asarray(kwargs.get(key, default), dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("Invalid source configuration: " + key)
        return values

    contract = {"schema": "robosuite-panda-object-task-v1", "task": task, "source_version": version,
                "reference": REFERENCES[task], "control_freq_hz": 20,
                "object_ids": sorted(required), "frame": "source world before rigid scene placement"}
    if task == "PickPlaceCan":
        size = vector_setting("table_full_size", [.39, .49, .82])
        if not np.allclose(size, [.39, .49, .82], atol=1e-8, rtol=0):
            raise ValueError("Unverified logical bin-size configuration")
        bin_body = model.body("bin2").id
        center = point(data.xpos[bin_body])
        expected = vector_setting("bin2_pos", [.1, .28, .8])
        rotation = placement[:3, :3].T @ data.xmat[bin_body].reshape(3, 3)
        if not np.allclose(center, expected, atol=1e-7, rtol=0) or not np.allclose(rotation, np.eye(3), atol=1e-7):
            raise ValueError("Source bin pose contradicts verified environment configuration")
        floor = [i for i in range(model.ngeom) if model.geom_bodyid[i] == bin_body
                 and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
                 and (model.geom_contype[i] or model.geom_conaffinity[i])
                 and np.allclose(model.geom_size[i], [.2, .25, .02], atol=1e-8, rtol=0)
                 and np.allclose(data.geom_xpos[i], data.xpos[bin_body], atol=1e-7, rtol=0)]
        if len(floor) != 1:
            raise ValueError("Unverified source bin floor geometry")
        contract.update(bin_body="bin2", logical_bin_size=size.tolist(), bin_origin=center.tolist(),
                        bin_lower=center.tolist(), bin_upper=(center + [size[0]/2, size[1]/2, .1]).tolist(),
                        bin_index=3)
    else:
        size = vector_setting("table_full_size", [.8, .8, .05])
        if not np.allclose(size, [.8, .8, .05], atol=1e-8, rtol=0):
            raise ValueError("Unverified source table-size configuration")
        table = model.geom("table_collision").id
        orientation = placement[:3, :3].T @ data.geom_xmat[table].reshape(3, 3)
        if (model.geom_type[table] != mujoco.mjtGeom.mjGEOM_BOX
                or not np.allclose(model.geom_size[table] * 2, size, atol=1e-8, rtol=0)
                or not np.allclose(orientation, np.eye(3), atol=1e-7, rtol=0)):
            raise ValueError("Unverified source tabletop geometry")
        top = float(point(data.geom_xpos[table])[2] + model.geom_size[table, 2])
        expected_top = .82 if task == "NutAssemblySquare" else .8
        if not np.isclose(top, expected_top, atol=1e-7, rtol=0):
            raise ValueError("Unverified source table height")
        contract.update(table_geom="table_collision", table_top=top, lift_height=top + .04)
        if task == "NutAssemblySquare":
            model.body("peg1")  # Name resolution must fail before an expensive rollout.
            contract.update(peg_body="peg1", peg_xy_tolerance=.03,
                            assembly_min_height=top, assembly_max_height=top + .05)
        if task == "Stack_D0":
            contract["stack_min_center_height_difference"] = .025
        if task == "Threading_D0":
            ring = [i for i in range(model.ngeom) if model.geom(i).name.startswith("tripod_obj_ring_")
                    and model.geom(i).name.removeprefix("tripod_obj_ring_").isdigit()]
            if (len(ring) != 20 or any(model.geom_type[i] != mujoco.mjtGeom.mjGEOM_BOX for i in ring)
                    or not np.allclose(model.geom_size[ring], [.005, .002, .002], atol=1e-8, rtol=0)):
                raise ValueError("Unverified source threading ring geometry")
            tripod = model.body(objects["tripod_obj"]["body"]).id
            local = (data.geom_xpos[ring] - data.xpos[tripod]) @ data.xmat[tripod].reshape(3, 3)
            centered = local - local.mean(0)
            actual = {tuple(np.round(p, 7)) for p in centered}
            expected = {(0., round(j*.004-.01, 7), round(k*.004-.01, 7))
                        for j in range(6) for k in range(6) if j in (0, 5) or k in (0, 5)}
            if actual != expected:
                raise ValueError("Unverified source threading ring pattern")
            radius = float((np.ptp(local[:, 1]) + 2*model.geom_size[ring[0], 1])/2)
            model.geom("needle_obj_needle")
            contract.update(ring_geoms=[model.geom(i).name for i in ring],
                            needle_geom="needle_obj_needle", threading_radius=radius)
    return contract



def source_grasp(a, metadata, oid, geometry_center=True):
    """Use the source lift to select a grasp region on the unchanged object."""
    if metadata.get("source_format") == "ManiSkill-HDF5":
        return dynamics_maniskill.source_grasp(a, metadata, oid, geometry_center)
    _validate_source_arrays(a, [oid])
    E = pose_to_matrices(a["hand/right_pose"])
    O = pose_to_matrices(a[f"objects/{oid}/pose"])
    grip = np.asarray(a["source/action"][:, 6])
    if not np.isin(grip, (-1., 1.)).all():
        raise ValueError("Unverified source gripper schema")
    closed = np.flatnonzero(grip > 0)
    if not len(closed):
        raise ValueError("No demonstrated closed gripper")
    threshold = min(.03, .5*float(np.max(O[:, 2, 3]-O[0, 2, 3])))
    lifted = np.flatnonzero((grip > 0) & (O[:, 2, 3] > O[0, 2, 3]+threshold))
    if not len(lifted) or threshold < .005:
        raise ValueError("No source lift establishing a grasp attachment")
    anchor = int(lifted[0])
    local = (np.linalg.inv(O[anchor]) @ E[anchor])[:3, 3]
    selected = "source end-effector position in object frame"
    if geometry_center:
        tree = ET.fromstring(metadata["model_xml"])
        body = tree.find(f'.//body[@name="{metadata["objects"][oid]["body"]}"]')
        choices = []
        for geom in body.findall("geom"):
            if geom.get("contype", "1") == "0":
                continue
            center = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
            choices.append((float(np.linalg.norm(center-local)), geom.get("name"), center))
        if choices:
            _, name, local = min(choices, key=lambda v: v[0])
            selected = "center of source collision region nearest demonstrated grasp: " + str(name)
    return E, O, grip, anchor, local, selected



def source_window(a, metadata, start_s):
    """Select a stationary pregrasp source window without modifying source arrays."""
    original_times = np.asarray(a["time_s"])
    source_start = int(np.searchsorted(original_times, start_s))
    if start_s < 0 or source_start >= len(original_times)-2:
        raise ValueError("Invalid source interval start")
    if source_start:
        if metadata.get("source_format") == "ManiSkill-HDF5":
            raise ValueError("Source cropping is not implemented for T+1 ManiSkill recordings")
        if np.any(a["source/action"][:source_start+1,6] > 0):
            raise ValueError("Source crop would omit demonstrated grasp interaction")
        for key, value in a.items():
            if key.startswith("objects/") and key.endswith("/pose"):
                speed = np.linalg.norm(np.diff(value[source_start-1:source_start+2,:3],axis=0),axis=1) / np.diff(original_times[source_start-1:source_start+2])
                if np.max(speed) > .01:
                    raise ValueError("Source crop must start with stationary objects")
                transforms = pose_to_matrices(value[source_start-1:source_start+2])
                angular = Rotation.from_matrix(np.swapaxes(transforms[:-1,:3,:3],1,2) @ transforms[1:,:3,:3]).magnitude() / np.diff(original_times[source_start-1:source_start+2])
                if np.max(angular) > .2:
                    raise ValueError("Source crop must start with rotationally stationary objects")
        a = {key: value[source_start:] if np.ndim(value) and len(value)==len(original_times) else value
             for key,value in a.items()}
        a["time_s"] = a["time_s"]-original_times[source_start]
    return a, source_start, original_times



def task_state(task, oid, model, data, object_bodies, placement, closed, hand_contact, initial, contract=None):
    """Publisher predicates plus release/support guards, in source coordinates.

    Production callers must pass the validated and persisted scene contract.
    Defaults retain the direct predicate API for small isolated test fixtures.
    """
    if contract is None:
        contract = {"task": task, "lift_height": .84, "bin_lower": [.1, .28, .8],
                    "bin_upper": [.295, .525, .9], "peg_body": "peg1", "peg_xy_tolerance": .03,
                    "assembly_min_height": .82, "assembly_max_height": .87,
                    "stack_min_center_height_difference": .025, "threading_radius": .012,
                    "needle_geom": "needle_obj_needle"}
    if contract.get("task") != task:
        raise ValueError("Task predicate and validated source contract disagree")
    if task == "PickCube-v1":
        return dynamics_maniskill.task_satisfied(model, data, object_bodies[oid], contract)
    p = data.xpos[object_bodies[oid]]
    local = placement[:3, :3].T @ (p-placement[:3, 3])
    if task == "Lift":
        return bool(local[2] > contract["lift_height"] and closed and hand_contact)
    if task == "PickPlaceCan":
        return bool(np.all(local > contract["bin_lower"]) and np.all(local < contract["bin_upper"])
                    and not closed and not hand_contact)
    if task == "NutAssemblySquare":
        peg = data.xpos[model.body(contract["peg_body"]).id]
        peg_local = placement[:3, :3].T @ (peg-placement[:3, 3])
        return bool(np.max(np.abs(local[:2]-peg_local[:2])) < contract["peg_xy_tolerance"]
                    and contract["assembly_min_height"] < local[2] < contract["assembly_max_height"]
                    and not closed and not hand_contact)
    if task == "Stack_D0":
        other = object_bodies["cubeB"]
        contact = any({int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])} == {object_bodies[oid], other}
                      for c in data.contact)
        return bool(contact and local[2] > contract["lift_height"]
                    and p[2] > data.xpos[other, 2]+contract["stack_min_center_height_difference"]
                    and not hand_contact and not closed)
    if task == "Threading_D0":
        needle = data.geom_xpos[model.geom(contract["needle_geom"]).id]
        names = contract.get("ring_geoms", [model.geom(i).name for i in range(model.ngeom)
                             if model.geom(i).name.startswith("tripod_obj_ring_")
                             and model.geom(i).name.removeprefix("tripod_obj_ring_").isdigit()])
        ring = [model.geom(name).id for name in names]
        return bool(ring and np.linalg.norm(needle-data.geom_xpos[ring].mean(0)) < contract["threading_radius"]
                    and closed and hand_contact)
    return False



def source_vertical_offset(metadata):
    """Source floor/table coordinate conversion; never an optimizer parameter."""
    return dynamics_maniskill.PLACEMENT_Z if metadata.get("source_format") == "ManiSkill-HDF5" else 0.


def scene_contract(arrays, metadata, model, data, placement):
    if metadata.get("source_format") == "ManiSkill-HDF5":
        return dynamics_maniskill.task_context(arrays, metadata, placement)
    return validate_task_contract(metadata, arrays, model, data, placement)
