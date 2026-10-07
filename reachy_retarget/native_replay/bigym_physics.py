"""Asset-faithful BiGym task scene with an actuated Reachy replacement.

Explicit native asset export runs in the isolated source runtime. Scene/rollout
functions run in the common Reachy runtime. No network, SDK or image rendering.
The free plate is initialized once and never assigned during integration.
"""

from pathlib import Path
import copy
import hashlib
import json
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def export_inertials(replay, output):
    """Preserve native compiled mass properties across MuJoCo mesh versions."""
    import mujoco
    if mujoco.__version__ != "3.1.5":
        raise ValueError("Read native binary mass properties in MuJoCo 3.1.5")
    output, replay = Path(output), Path(replay)
    if output.exists():
        raise FileExistsError("Immutable native inertia evidence exists")
    model = mujoco.MjModel.from_binary_path(str(replay/"source-model.mjb"))
    bodies = {}
    for i in range(1, model.nbody):
        name = model.body(i).name
        if name.startswith(("plate/", "table/", "dish_drainer/", "dish_drainer_1/")):
            bodies[name] = {"mass": float(model.body_mass[i]), "inertia": model.body_inertia[i].tolist(),
                           "pos": model.body_ipos[i].tolist(), "quat": model.body_iquat[i].tolist()}
    result = {"source_mjb_sha256": _sha(replay/"source-model.mjb"), "source_mujoco_version": mujoco.__version__,
              "body_inertials": bodies}
    _json(output, result)
    return result


def _mesh_vertices_body(model, geom):
    import mujoco
    mesh = int(model.geom_dataid[geom])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
    return model.mesh_vert[start:start+count]@rotation.reshape(3, 3).T+model.geom_pos[geom]


def export_mesh_geometry(replay, output):
    """Reference vertices in body frames, independent of compiler mesh centering."""
    import mujoco
    if mujoco.__version__ != "3.1.5":
        raise ValueError("Read source mesh geometry in MuJoCo 3.1.5")
    output, replay = Path(output), Path(replay)
    if output.exists():
        raise FileExistsError("Immutable native geometry evidence exists")
    model = mujoco.MjModel.from_binary_path(str(replay/"source-model.mjb"))
    vertices = {model.geom(i).name: _mesh_vertices_body(model, i) for i in range(model.ngeom)
                if model.geom(i).name.startswith(("plate/", "table/", "dish_drainer/", "dish_drainer_1/"))
                and int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_MESH)}
    with output.open("xb") as stream:
        np.savez_compressed(stream, **vertices)
    _json(output.with_suffix(".json"), {"source_mjb_sha256": _sha(replay/"source-model.mjb"),
        "source_mujoco_version": mujoco.__version__, "npz_sha256": _sha(output),
        "frame": "each geometry owning body; mesh rotation and centering undone", "geoms": list(vertices)})
    return {"mesh_geoms": len(vertices), "bytes": output.stat().st_size}


def export_assets(pilot_root, replay, source_root, output):
    """Explicit, immutable materialization of official dm_control virtual assets."""
    from .bigym import load_pilot, _create_environment, _source_provenance
    import mujoco
    output, replay = Path(output), Path(replay)
    if output.exists():
        raise FileExistsError("Immutable native asset export exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 50_000_000_000:
        raise OSError("Native asset export requires 50 decimal GB reserve")
    provenance = _source_provenance(source_root)
    native, *_ = load_pilot(pilot_root)
    env = _create_environment(native, source_root)
    try:
        xml = env.mojo.root_element.mjcf.to_xml_string()
        original = (replay/"source-model.xml").read_text()
        if xml != original:
            raise ValueError("Recreated seeded task XML differs from the archived native replay")
        assets = env.mojo.root_element.mjcf.get_assets()
        required = sum(len(v) for v in assets.values())
        if shutil.disk_usage(output.parent).free-required < 50_000_000_000:
            raise OSError("Native asset export would violate the disk reserve")
        output.mkdir()
        records = []
        for name, payload in assets.items():
            if Path(name).name != name:
                raise ValueError("Unexpected native asset path")
            dest = output/name
            dest.write_bytes(payload)
            records.append({"name": name, "bytes": len(payload), "sha256": _sha(dest)})
        model = env.mojo.model
        roots = {"plate": "plate/", "rack_start": "dish_drainer/", "rack_target": "dish_drainer_1/", "table": "table/"}
        members = {}
        for label, name in roots.items():
            first = model.body(name).id
            bodyids = {first}
            for i in range(first+1, model.nbody):
                if int(model.body_parentid[i]) in bodyids:
                    bodyids.add(i)
            members[label] = {"body": name,
                "bodies": {model.body(i).name: {"mass": float(model.body_mass[i]), "inertia": model.body_inertia[i].tolist()} for i in sorted(bodyids)},
                "geoms": {model.geom(i).name: {"type": int(model.geom_type[i]), "size": model.geom_size[i].tolist(),
                    "friction": model.geom_friction[i].tolist(), "solref": model.geom_solref[i].tolist(),
                    "solimp": model.geom_solimp[i].tolist(), "contype": int(model.geom_contype[i]),
                    "conaffinity": int(model.geom_conaffinity[i])}
                    for i in range(model.ngeom) if int(model.geom_bodyid[i]) in bodyids}}
        j = model.joint("plate/").id
        manifest = {"schema": "bigym-native-assets-v1", "source_revision": provenance["acquisition"][0].get("revision"),
            "source_tree_sha256": provenance["source_tree_sha256"], "source_xml_sha256": _sha(replay/"source-model.xml"),
            "source_mujoco_version": mujoco.__version__, "assets": records, "task_objects": members,
            "target_sites": [site.mjcf.full_identifier for site in env.rack_target.sites],
            "plate_joint": {"name": "plate/", "qpos_address": int(model.jnt_qposadr[j]), "dof_address": int(model.jnt_dofadr[j])}}
        _json(output/"manifest.json", manifest)
        export_inertials(replay, output/"native-inertials.json")
        export_mesh_geometry(replay, output/"native-mesh-vertices.npz")
        return manifest
    finally:
        env.close()


def _remove_source_robot(root):
    """Remove the source body and only entries referencing its named subtree."""
    world = root.find("worldbody")
    body = world.find("body[@name='h1/']")
    if body is None:
        raise ValueError("Expected exactly the native H1 world-body root")
    world.remove(body)
    for section in ("contact", "equality", "tendon", "actuator", "sensor"):
        parent = root.find(section)
        if parent is None:
            continue
        for item in list(parent):
            if any(str(value).startswith("h1/") for element in item.iter() for value in element.attrib.values()):
                parent.remove(item)
    return root


def build_scene(replay, native_assets, output, placement):
    """Keep original task MJCF/meshes, material coefficients, masses and inertia."""
    import mujoco
    from ..physics import prepare_robot, attrs
    from ..episodes import pose_to_matrices
    from scipy.spatial import cKDTree
    replay, native_assets, output = Path(replay), Path(native_assets), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output/"scene.xml"
    if target.exists():
        raise FileExistsError("Immutable physical scene exists")
    manifest = json.loads((native_assets/"manifest.json").read_text())
    inertials = json.loads((native_assets/"native-inertials.json").read_text())
    if inertials["source_mjb_sha256"] != _sha(replay/"source-model.mjb"):
        raise ValueError("Source binary mass-property identity mismatch")
    geometry = json.loads((native_assets/"native-mesh-vertices.json").read_text())
    if (geometry["source_mjb_sha256"] != inertials["source_mjb_sha256"]
            or geometry["npz_sha256"] != _sha(native_assets/"native-mesh-vertices.npz")):
        raise ValueError("Native mesh geometry identity mismatch")
    original_vertices = np.load(native_assets/"native-mesh-vertices.npz", allow_pickle=False)
    if _sha(replay/"source-model.xml") != manifest["source_xml_sha256"]:
        raise ValueError("Native scene identity mismatch")
    placement = np.asarray(placement)
    if (placement.shape != (4, 4) or not np.allclose(placement[2], [0, 0, 1, 0])
            or not np.allclose(placement[3], [0, 0, 0, 1])
            or not np.allclose(placement[:3, :3].T@placement[:3, :3], np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(placement[:3, :3]), 1)):
        raise ValueError("Only one rigid planar scene gauge is allowed")
    scene = _remove_source_robot(ET.parse(replay/"source-model.xml").getroot())
    for asset in manifest["assets"]:
        path = native_assets/asset["name"]
        if _sha(path) != asset["sha256"]:
            raise ValueError("Native task asset changed")
    for item in scene.find("asset"):
        if item.get("file"):
            item.set("file", str(native_assets/item.get("file")))
    # Modern MuJoCo computes mesh-derived inertia differently. Explicit native
    # compiled inertials preserve original physics instead of adopting that change.
    for body in scene.findall(".//body"):
        spec = inertials["body_inertials"].get(body.get("name"))
        if spec and spec["mass"] > 0:
            old = body.find("inertial")
            if old is not None:
                body.remove(old)
            ET.SubElement(body, "inertial", mass=str(spec["mass"]),
                          diaginertia=" ".join(map(str, spec["inertia"])),
                          pos=" ".join(map(str, spec["pos"])), quat=" ".join(map(str, spec["quat"])))
    # All native world bodies share the exact same gauge, including support.
    for body in scene.findall("worldbody/body"):
        pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
        quat = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
        matrix = pose_to_matrices(np.r_[pos, quat])
        body.attrib.update(attrs(placement@matrix))
    robot, identity = prepare_robot(output/"reachy-assets", state_profile=True)
    default = copy.deepcopy(robot.find("default")); default.set("class", "reachy_native_replacement")
    scene.find("default").append(default)
    body = copy.deepcopy(robot.find("worldbody/body[@name='base_link']"))
    body.set("childclass", "reachy_native_replacement")
    scene.find("worldbody").append(body)
    for section in ("asset", "actuator", "equality", "contact"):
        parent = scene.find(section)
        if parent is None:
            parent = ET.SubElement(scene, section)
        parent.extend(copy.deepcopy(list(robot.find(section))))
    # Preserve static link identities for complete task/robot observations.
    scene.find("compiler").set("fusestatic", "false")
    target.write_text(ET.tostring(scene, encoding="unicode"))
    model = mujoco.MjModel.from_xml_path(str(target))
    comparisons, mesh_errors = [], {}
    for label, description in manifest["task_objects"].items():
        for name, values in description["bodies"].items():
            i = model.body(name).id
            for key, actual in (("mass", model.body_mass[i]), ("inertia", model.body_inertia[i])):
                if not np.allclose(actual, values[key], rtol=1e-5, atol=1e-9):
                    raise ValueError("Native task body " + name + " changed " + key)
        for name, values in description["geoms"].items():
            i = model.geom(name).id
            for key in ("type", "size", "friction", "solref", "solimp", "contype", "conaffinity"):
                if key == "size" and int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                    # Mesh size is a compiler-dependent principal-axis bound.
                    # Compare the actual physical vertices below instead.
                    continue
                if not np.allclose(getattr(model, "geom_"+key)[i], values[key], rtol=1e-5, atol=1e-9):
                    raise ValueError("Native task geometry " + name + " changed " + key)
            if int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                before, after = original_vertices[name], _mesh_vertices_body(model, i)
                error = max(cKDTree(before).query(after)[0].max(), cKDTree(after).query(before)[0].max())
                if error > 1e-6:
                    raise ValueError("Native physical mesh vertices changed: " + name + ": " + str(error))
                mesh_errors[name] = float(error)
        comparisons.append(label)
    # No actuator may affect the free plate or fixture; no object weld allowed.
    plate = model.body("plate/").id
    for i in range(model.nu):
        joint = int(model.actuator_trnid[i, 0])
        if int(model.jnt_bodyid[joint]) == plate:
            raise ValueError("Forbidden moving-object actuation")
    if any("plate/" in str(v) for e in scene.find("equality") for v in e.attrib.values()):
        raise ValueError("Forbidden moving-object equality")
    result = {"native": manifest, "native_inertials": inertials, "reachy": identity, "scene_sha256": _sha(target),
        "scene_path": str(target), "physics_mujoco_version": mujoco.__version__,
        "native_task_properties_compared": comparisons, "placement": placement.tolist(),
        "native_mesh_max_vertex_error_m": mesh_errors,
        "simulation_assumptions": ["Original BiGym source version gap retained (recorded 4.0, public replay 4.1).",
            "Reachy uses gravity-compensated ideal planar/arm/neck position servos; mobile wheel traction is not modeled.",
            "Native task XML/mesh assets, masses, inertia, friction and contact parameters retained and checked.",
            "Simulator version differs from native 3.1.5; this is a derived Reachy simulation."]}
    _json(output/"scene-manifest.json", result)
    return model, result


def _body_members(model, name):
    root = model.body(name).id
    result = {root}
    for i in range(root+1, model.nbody):
        if int(model.body_parentid[i]) in result:
            result.add(i)
    return result


def simulation_scene(model, manifest, output, profile):
    """Explicit contact-model sensitivity; retain the source-contact scene."""
    from ..contact_profile import NAME, prepare
    if profile is None:
        return model, manifest
    if profile != NAME:
        raise ValueError("Only the declared global-contact-4ms sensitivity profile is supported")
    output = Path(output)
    scene = Path(manifest["scene_path"])
    original = scene.read_text()
    prepared = prepare((model, original, manifest, None, None, {}), output/"contact-profile", timestep=.001)
    derived, xml, updated = prepared[:3]
    retained = output/"source-contact-scene.xml"
    scene.rename(retained)
    _json(output/"source-contact-scene-manifest.json", dict(manifest, scene_path=str(retained)))
    scene.write_text(xml)
    updated.update(scene_path=str(scene), scene_sha256=_sha(scene),
                   source_contact_scene_path=str(retained), source_contact_scene_sha256=_sha(retained))
    _json(output/"scene-manifest.json", updated)
    return derived, updated


def contact_metrics(model, data):
    """Actual collision pairs; source plate/rack contacts are not robot defects."""
    robot = _body_members(model, "base_link")
    plate = _body_members(model, "plate/")
    target = _body_members(model, "dish_drainer_1/")
    table = _body_members(model, "table/")
    floor = _body_members(model, "floor")
    pads = {model.body(prefix+suffix).id for prefix in ("l", "r")
            for suffix in ("_hand_distal_link", "_hand_distal_mimic_link")}
    robot_pairs = []
    flags = dict(plate_target_contact=False, plate_table_contact=False, plate_floor_contact=False, plate_pad_contact=False)
    for c in data.contact[:data.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        if c.dist > 1e-8:  # Pinned BiGym physics_utils._DEFAULT_COLLISION_MARGIN.
            continue
        for label, members in (("target", target), ("table", table), ("floor", floor), ("pad", pads)):
            if (b1 in plate and b2 in members) or (b2 in plate and b1 in members):
                flags["plate_"+label+"_contact"] = True
        if b1 in robot or b2 in robot:
            robot_pairs.append({"geom1": model.geom(g1).name, "geom2": model.geom(g2).name,
                                "body1": model.body(b1).name, "body2": model.body(b2).name,
                                "penetration_m": float(max(0., -c.dist)), "self_collision": b1 in robot and b2 in robot})
    return dict(flags, robot_contacts=robot_pairs,
                max_robot_penetration_m=max((p["penetration_m"] for p in robot_pairs), default=0.))


def task_success(model, data, target_sites, placement, contact=None):
    """Pinned MovePlate geometry/contact predicate with Reachy pad-body contact.

    Native release tests any pad collision (not bilateral force). The converted
    task uses the corresponding Reachy distal collision bodies. Source world
    right is rotated by the common scene gauge, not hardcoded in derived axes.
    """
    contact = contact or contact_metrics(model, data)
    plate = model.body("plate/").id
    distance = min(float(np.linalg.norm(data.xpos[plate]-data.site_xpos[model.site(name).id])) for name in target_sites)
    up = data.xmat[plate].reshape(3, 3)@np.array([0., 0., 1.])
    right = np.asarray(placement)[:3, :3]@np.array([0., -1., 0.])
    angle = float(np.arccos(np.clip(up@right, -1., 1.)))
    passed = distance <= .05 and angle <= np.deg2rad(20) and contact["plate_target_contact"] and not contact["plate_table_contact"] and not contact["plate_pad_contact"]
    return dict(success=bool(passed), target_slot_distance_m=distance, upright_error_rad=angle,
                **{key: value for key, value in contact.items() if key.startswith("plate_")})


def initialize_scene(model, manifest, replay, raw, robot, initial_grips):
    """The single reset assignment of source plate pose/velocity, then free dynamics."""
    import mujoco
    from ..physics import initialize, GRIPPERS
    from ..episodes import pose_to_matrices, matrices_to_pose
    data = mujoco.MjData(model)
    native = np.load(Path(replay)/"states.npz", allow_pickle=False)
    placement = np.asarray(manifest["placement"])
    joint = model.joint("plate/").id
    qadr, vadr = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    source_vadr = manifest["native"]["plate_joint"]["dof_address"]
    data.qpos[qadr:qadr+7] = matrices_to_pose(placement@pose_to_matrices(native["objects/plate/pose"][0]))
    data.qvel[vadr:vadr+3] = placement[:3, :3]@native["source/qvel"][0, source_vadr:source_vadr+3]
    # A free-joint angular velocity is in the body's local frame.
    data.qvel[vadr+3:vadr+6] = native["source/qvel"][0, source_vadr+3:source_vadr+6]
    initialize(model, data, robot, raw["robot_qpos"][0], manifest["reachy"]["mimics"], grip=2.)
    values = {name: float(initial_grips[side]) for name, side in zip(GRIPPERS, ("left", "right"))}
    def mimic_value(name):
        if name not in values:
            parent, mult, offset = manifest["reachy"]["mimics"][name]
            values[name] = mimic_value(parent)*mult+offset
        return values[name]
    for name in list(values)+list(manifest["reachy"]["mimics"]):
        data.qpos[model.joint(name).qposadr] = mimic_value(name)
    for name in GRIPPERS:
        data.ctrl[model.actuator(name).id] = values[name]
    data.time = float(raw["original_clock_s"][0])
    mujoco.mj_forward(model, data)
    return data


def timing_plan(clock, qpos, arm_ids, timestep, *, arm_speed_rad_s=None,
                base_axis_speed_m_s=.6, base_yaw_speed_rad_s=1.2, max_steps=200000,
                interval_grid_s=None):
    """Minimal interval-local slowdown under declared target-speed assumptions.

    Original source samples remain unchanged. Each source interval gets an
    integer number of physics steps; a strictly increasing piecewise-linear
    source-time map retains every original boundary. These are controller design
    limits, not certified hardware or measured-speed limits. No joint wrapping
    is applied to bounded arm coordinates. The planar yaw follows its continuous
    unwrapped path.
    """
    clock, qpos = np.asarray(clock, dtype=float), np.asarray(qpos, dtype=float)
    arm_ids = np.asarray(arm_ids, dtype=int)
    if (clock.ndim != 1 or len(clock) < 2 or qpos.ndim != 2 or len(qpos) != len(clock)
            or not np.isfinite(clock).all() or not np.isfinite(qpos).all() or np.any(np.diff(clock) <= 0)
            or timestep <= 0 or not np.allclose(np.linalg.norm(qpos[:, 2:4], axis=1), 1, atol=1e-6)):
        raise ValueError("Finite aligned source clock/configurations and a positive physics step required")
    delta = np.diff(clock)
    yaw = np.unwrap(np.arctan2(qpos[:, 3], qpos[:, 2]))
    required = delta.copy()
    enabled = arm_speed_rad_s is not None
    if enabled:
        if min(arm_speed_rad_s, base_axis_speed_m_s, base_yaw_speed_rad_s) <= 0:
            raise ValueError("Positive explicit target-speed limits required")
        required = np.maximum(required, np.abs(np.diff(qpos[:, arm_ids], axis=0)).max(1)/arm_speed_rad_s)
        required = np.maximum(required, np.abs(np.diff(qpos[:, :2], axis=0)).max(1)/base_axis_speed_m_s)
        required = np.maximum(required, np.abs(np.diff(yaw))/base_yaw_speed_rad_s)
    grid = timestep if interval_grid_s is None else float(interval_grid_s)
    multiplier = int(round(grid/timestep)) if np.isfinite(grid) else 0
    if multiplier < 1 or not np.isclose(multiplier*timestep, grid, atol=1e-12, rtol=0):
        raise ValueError("Interval duration grid must be a positive integer number of physics steps")
    steps = np.maximum(1, np.ceil(required/grid-1e-9).astype(int))*multiplier
    if not enabled and not np.allclose(steps*timestep, delta, atol=1e-10, rtol=0):
        raise ValueError("Unretimed source boundaries must coincide with physics steps")
    if int(steps.sum()) > max_steps:
        raise ValueError("Explicit per-attempt physical-step budget exceeded")
    endpoints = np.r_[0, np.cumsum(steps)]
    physical_clock = clock[0]+np.arange(endpoints[-1]+1)*timestep
    mapped_clock, targets, command_rows = [float(clock[0])], [], []
    for index, count in enumerate(steps):
        alpha = np.arange(1, count+1)/count
        q = qpos[index]+alpha[:, None]*(qpos[index+1]-qpos[index])
        angle = yaw[index]+alpha*(yaw[index+1]-yaw[index])
        q[:, 2], q[:, 3] = np.cos(angle), np.sin(angle)
        targets.append(q)
        mapped_clock.extend(clock[index]+alpha*delta[index])
        command_rows.extend([index+1]*count)
    return {"physical_clock": physical_clock, "source_clock_at_boundary": np.asarray(mapped_clock),
        "source_boundary_step_index": endpoints, "substeps_per_source_interval": steps,
        "target_qpos": np.concatenate(targets), "source_command_row": np.asarray(command_rows, dtype=int),
        "summary": {"enabled": enabled, "source_duration_s": float(clock[-1]-clock[0]),
            "derived_duration_s": float(physical_clock[-1]-physical_clock[0]), "physical_steps": int(steps.sum()),
            "interval_duration_grid_s": grid,
            "max_interval_time_factor": float(np.max(steps*timestep/delta)),
            "arm_target_speed_limit_rad_s": arm_speed_rad_s,
            "base_axis_target_speed_limit_m_s": base_axis_speed_m_s if enabled else None,
            "base_yaw_target_speed_limit_rad_s": base_yaw_speed_rad_s if enabled else None,
            "semantics": "Minimal interval-local speed-budget slowdown; unchanged source trajectory and explicit monotone time map",
            "hardware_limit_claim": False, "measured_speed_validation": False}}


def gripper_control_policy(original_targets, plan, policy=None):
    """Derive bounded motor targets; preserve recorded binary source intent."""
    targets = {side: np.array(values, dtype=float, copy=True) for side, values in original_targets.items()}
    report = {"method": "verified_binary_endpoint_mapping", "source_intent_preserved": True,
              "default_closed_target_rad": -.06, "open_target_rad": 2., "overrides": {}}
    if policy is None:
        return targets, report
    if policy.get("method") != "cad_contact_minus_margin" or not policy.get("evidence"):
        raise ValueError("An explicit geometry-derived gripper policy and retained evidence are required")
    margins = policy.get("closure_margin_rad", {})
    if not isinstance(margins, dict) or not margins or set(margins)-set(targets):
        raise ValueError("Explicit left/right closure margins are required")
    for side, margin in margins.items():
        margin = float(margin)
        contact = float(plan["pad_calibration"]["hands"][side]["nominal_gripper_rad"])
        target = contact-margin
        if not np.isfinite([margin, contact, target]).all() or not 0 < margin <= .1 or not 0 <= target < contact < 2:
            raise ValueError("Positive bounded closure below a verified contact aperture is required")
        values = targets[side]
        if not np.isin(values, [-.06, 2.]).all() or not np.any(values == -.06):
            raise ValueError("Only a verified binary grasping hand may receive a closure override")
        values[values == -.06] = target
        report["overrides"][side] = dict(contact_angle_rad=contact, closure_margin_rad=margin,
            motor_closed_target_rad=target, default_derived_closed_target_rad=-.06,
            semantics="Derived motor overdrive relative to CAD contact; not a measured aperture or contact-force guarantee")
    report.update(method=policy["method"], evidence=policy["evidence"])
    return targets, report


class InactiveHandContactAudit:
    """Measure unintended inactive-hand forces immediately after each mj_step."""
    def __init__(self, model, grasp_hands):
        self.model = model
        self.object_bodies = _body_members(model, "plate/")
        self.inactive = {side: {i for i in range(model.nbody)
                               if model.body(i).name.startswith(prefix+"_hand")}
                         for side, prefix in (("left", "l"), ("right", "r")) if side not in grasp_hands}
        if any(not ids for ids in self.inactive.values()):
            raise ValueError("Inactive hand must bind actual compiled robot bodies")
        self.steps, self.events = 0, []

    def sample(self, data):
        import mujoco
        self.steps += 1
        for index, contact in enumerate(data.contact[:data.ncon]):
            pair = {int(self.model.geom_bodyid[contact.geom1]), int(self.model.geom_bodyid[contact.geom2])}
            if not pair & self.object_bodies:
                continue
            for side, bodies in self.inactive.items():
                if pair & bodies:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, data, index, force)
                    if not np.isfinite(force).all():
                        raise ValueError("Nonfinite inactive-hand contact force")
                    if force[0] > 1e-6:
                        self.events.append(dict(time_s=float(data.time), side=side,
                            geom1=self.model.geom(contact.geom1).name, geom2=self.model.geom(contact.geom2).name,
                            normal_force_n=float(force[0]), penetration_m=float(max(0., -contact.dist))))

    def finish(self, expected_steps):
        return dict(passed=self.steps == expected_steps and not self.events,
            complete=self.steps == expected_steps, measured_steps=self.steps, expected_steps=expected_steps,
            inactive_hands=list(self.inactive), positive_contact_force_threshold_n=1e-6,
            positive_force_contacts=len(self.events), events=self.events,
            timing="Solved contacts immediately after each mj_step, before cache synchronization",
            scope="Measured additional guard; source intent unchanged, no inactive hand excluded from collision checks")


def bounded_arm_feedforward(model, names, reference, seconds, initial, policy=None):
    """Robot-only damping compensation with explicit position and slew bounds."""
    reference, seconds = np.asarray(reference, float), np.asarray(seconds, float)
    initial = np.asarray(initial, float)
    if (reference.ndim != 2 or reference.shape != (len(seconds), len(names))
            or len(seconds) < 2 or initial.shape != (len(names),)
            or not all(np.isfinite(x).all() for x in (reference, seconds, initial))
            or np.any(np.diff(seconds) <= 0)):
        raise ValueError("Finite aligned arm references and increasing physical timestamps required")
    if policy is None:
        return reference.copy(), dict(enabled=False, changed_groups=[])
    if policy.get("method") != "bounded_velocity" or not policy.get("evidence"):
        raise ValueError("Explicit bounded velocity feedforward and retained evidence required")
    scale = float(policy.get("scale", 1.))
    cap = float(policy.get("max_offset_rad", .2))
    speed = float(policy.get("max_command_speed_rad_s", 1.))
    margin = float(policy.get("joint_margin_rad", .025))
    taper = float(policy.get("boundary_taper_s", .1))
    if (not np.isfinite([scale, cap, speed, margin, taper]).all()
            or not 0 < scale <= 1 or not 0 < cap <= .2
            or not 0 < speed <= 1 or margin < .025 or taper <= 0):
        raise ValueError("Finite conservative feedforward limits required")
    from ..servo_feedforward import velocity_offsets
    ids = np.asarray([model.actuator(name).id for name in names], int)
    joints = np.asarray([model.joint(name).id for name in names], int)
    if (not np.all(model.jnt_limited[joints]) or np.any(model.jnt_type[joints] != 3)
            or not np.array_equal(model.actuator_trnid[ids, 0], joints)):
        raise ValueError("Named bounded hinge position servos required")
    kp, kv = model.actuator_gainprm[ids, 0], -model.actuator_biasprm[ids, 2]
    if (not np.isfinite([kp, kv]).all() or np.any(kp <= 0) or np.any(kv < 0)
            or not np.allclose(model.actuator_biasprm[ids, 1], -kp)):
        raise ValueError("Verified affine position-servo gains required")
    # The shared helper's first three columns are its untouched mobile base.
    padded = np.column_stack([np.zeros((len(seconds), 3)), reference])
    offsets = velocity_offsets(model, np.r_[np.repeat(ids[0], 3), ids], padded, seconds, scale)[:, 3:]
    phase = np.clip(np.minimum(seconds-seconds[0], seconds[-1]-seconds)/taper, 0., 1.)
    offsets *= (phase**3*(10.-15.*phase+6.*phase**2))[:, None]
    offsets = np.clip(offsets, -cap, cap)
    low, high = model.jnt_range[joints, 0]+margin, model.jnt_range[joints, 1]-margin
    limited = model.actuator_ctrllimited[ids].astype(bool)
    low = np.where(limited, np.maximum(low, model.actuator_ctrlrange[ids, 0]), low)
    high = np.where(limited, np.minimum(high, model.actuator_ctrlrange[ids, 1]), high)
    dt = np.r_[seconds[1]-seconds[0], np.diff(seconds)]
    baseline_rates = np.diff(np.vstack([initial, reference]), axis=0)/dt[:, None]
    if (np.any(low >= high) or np.any(initial < low) or np.any(initial > high)
            or np.any(reference < low) or np.any(reference > high)
            or np.max(np.abs(baseline_rates)) > speed+1e-8):
        raise ValueError("Reference/initial commands must already satisfy the declared margin and rate")
    bounded = np.clip(reference+offsets, low, high)
    commands, previous, slew_clamps = [], initial.copy(), 0
    for value, interval in zip(bounded, dt):
        command = np.clip(value, previous-speed*interval, previous+speed*interval)
        slew_clamps += int(np.count_nonzero(np.abs(command-value) > 1e-12))
        commands.append(command); previous = command
    commands = np.asarray(commands)
    if np.max(np.abs(commands-reference)) > cap+1e-8:
        raise ValueError("Rate-limited command exceeded declared compensation bound")
    return commands, dict(enabled=True, method="bounded_velocity", changed_groups=["left_arm", "right_arm"],
        arm_joint_names=list(names), kp=kp.tolist(), kv=kv.tolist(), damping_lag_s=(kv/kp).tolist(),
        scale=scale, max_offset_rad=cap, max_command_speed_rad_s=speed, joint_margin_rad=margin,
        boundary_taper_s=taper, position_clamps=int(np.count_nonzero(np.abs(bounded-reference-offsets) > 1e-12)),
        slew_clamps=slew_clamps, maximum_applied_offset_rad=float(np.max(np.abs(commands-reference))),
        maximum_command_rate_rad_s=float(np.max(np.abs(np.diff(np.vstack([initial, commands]), axis=0)/dt[:, None]))),
        reference_clock_unchanged=True, source_gripper_intent_unchanged=True, evidence=policy["evidence"],
        semantics="Applied position targets include bounded kv/kp times reference angular velocity; source poses, base/neck/gripper targets, model gains and all object states are unchanged")


def resumed_arm_feedforward(model, names, reference, seconds, initial, policy):
    """Bind compensation to the resumed reference after measured acquisition.

    The caller supplies only the remaining corrected reference. A constant
    elapsed-time offset from the bounded pause does not change its derivative.
    Boundary tapering starts at zero on the first resumed command.
    """
    if policy.get("activation") != "after_measured_acquisition":
        raise ValueError("Explicit post-acquisition activation is required")
    commands, report = bounded_arm_feedforward(model, names, reference, seconds, initial, policy)
    report.update(activation="after_measured_acquisition", entry_settle_close_compensation=False,
                  timing_semantics="Remaining corrected physical reference clock; inserted pause is a constant time offset",
                  first_resumed_offset_rad=(commands[0]-reference[0]).tolist())
    return commands, report


def rollout(root, replay, native_assets, ik_attempt, episode_id, output, *, penetration_limit_m=.002,
            timing=None, simulation_profile=None, gripper_policy=None, arm_feedforward_policy=None, acquisition_policy=None,
            post_acquisition_arm_feedforward_policy=None):
    """Replay all selected derived robot commands; save admissions and failures."""
    started = time.perf_counter()
    performance = {"capture_s": 0.}
    if post_acquisition_arm_feedforward_policy is not None:
        if (acquisition_policy is None or arm_feedforward_policy is not None
                or post_acquisition_arm_feedforward_policy.get("activation") != "after_measured_acquisition"):
            raise ValueError("Post-acquisition compensation requires an isolated measured rendezvous")
    import mujoco
    from ..robot import Robot, ARMS, NECK
    from ..physics import BASE, GRIPPERS, measured
    from ..physical_gates import GateObserver, verify_actuator_replay
    from ..dynamics_audit import bind_scene_assets
    from ..state_observations import StateObserver, StateObservationBuffer
    from ..state_recording import synchronize_observation
    from ..agent_dataset import write_archive
    from .bigym import normalize
    from .bigym_retarget import prepare
    root, replay, ik_attempt, output = map(Path, (root, replay, ik_attempt, output))
    output.mkdir(parents=True, exist_ok=True)
    if (output/"admission.json").exists():
        raise FileExistsError("Immutable physical attempt exists")
    plan = json.loads((ik_attempt/"kinematic-validation.json").read_text())
    if plan.get("kinematic_passed") is not True:
        raise ValueError("A complete passing geometric candidate is required")
    raw = np.load(ik_attempt/"raw-ik.npz", allow_pickle=False)
    placement = raw["placement"].copy()
    source = normalize(replay)
    grips, mapping = prepare(source)
    baseline_grips = {side: values.copy() for side, values in grips.items()}
    grips, gripper_policy_report = gripper_control_policy(grips, plan, gripper_policy)
    clock = raw["original_clock_s"]
    if not np.array_equal(clock, source["arrays"]["timestamp"]):
        raise ValueError("Derived/source clock mismatch")
    if any(len(raw[key]) != len(clock) for key in ("robot_qpos", "target_hand_matrix")):
        raise ValueError("Incomplete full-trajectory derived candidate")
    model, manifest = build_scene(replay, native_assets, output, placement)
    model, manifest = simulation_scene(model, manifest, output, simulation_profile)
    scene = Path(manifest["scene_path"])
    asset_binding = bind_scene_assets(scene.read_text(), scene.parent)
    _json(output/"scene-asset-binding.json", asset_binding)
    robot = Robot(output/"runtime-identity")
    timing = {} if timing is None else dict(timing)
    timing_result = timing_plan(clock, raw["robot_qpos"], robot.arm_ids, model.opt.timestep, **timing)
    physical_clock = timing_result["physical_clock"]
    acquisition = None
    if acquisition_policy is not None:
        from .bigym_acquisition import PlateRendezvous
        if arm_feedforward_policy is not None or gripper_policy_report["overrides"].get("left", {}).get("closure_margin_rad") != .05:
            raise ValueError("Acquisition comparison requires fixed .05 gripper margin and disabled arm feedforward")
        acquisition = PlateRendezvous(model, robot, raw, plan, manifest, timing_result, acquisition_policy, output/"acquisition")
    arm_targets, arm_feedforward_report = bounded_arm_feedforward(model, ARMS,
        timing_result["target_qpos"][:, robot.arm_ids], physical_clock[1:],
        raw["robot_qpos"][0, robot.arm_ids], arm_feedforward_policy)
    _json(output/"arm-feedforward.json", arm_feedforward_report)
    np.savez_compressed(output/"robot-timing-map.npz", original_source_clock=clock,
        **{k: v for k, v in timing_result.items() if k != "summary"})
    _json(output/"robot-timing-map.json", timing_result["summary"])
    data = initialize_scene(model, manifest, replay, raw, robot, {s: v[0] for s, v in grips.items()})
    initial = contact_metrics(model, data)
    admission = dict(initial, accepted=initial["max_robot_penetration_m"] <= penetration_limit_m,
        penetration_limit_m=penetration_limit_m, source_task_success=source["metadata"].get("source_task_success"),
        reset_semantics="First archived post-action source plate pose and velocity, then fully free dynamics; robot initialized from derived first IK row",
        source_ik_attempt=str(ik_attempt), source_ik_sha256=_sha(ik_attempt/"raw-ik.npz"),
        scene_sha256=manifest["scene_sha256"], physics_validated=False)
    _json(output/"admission.json", admission)
    np.savez_compressed(output/"initial-state.npz", timestamp=data.time, qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl)
    if not admission["accepted"]:
        result = dict(status="initial_collision_rejected", physics_validated=False, physical_steps=0,
                      scene_path=manifest["scene_path"], admission=admission)
        _json(output/"physical-validation.json", result)
        _json(output/"performance.json", dict(setup_s=time.perf_counter()-started, physical_steps=0))
        return result
    observer = StateObserver(model, task_objects={key: value["body"] for key, value in manifest["native"]["task_objects"].items()},
        robot_body_root="base_link", base_body="base_link", head_body="head",
        tcp_sites={"left": "l_arm_tip_tcp", "right": "r_arm_tip_tcp"},
        object_roles={"pickup": "plate", "receptacle": "rack_target"},
        model_identity={"scene_sha256": manifest["scene_sha256"], "source_urdf_sha256": manifest["reachy"]["source_urdf_sha256"]})
    # Closure intent is the verified original binary action, not inferred from
    # object motion or a convenient choice of which hand passed validation.
    grasp_intent = {side: np.asarray(source["arrays"]["source/action"][:, 13+i]) == 1
                    for i, side in enumerate(("left", "right"))}
    grasp_hands = tuple(side for side, values in grasp_intent.items() if values.any())
    inactive_audit = InactiveHandContactAudit(model, grasp_hands)
    gates = GateObserver(model, data, active_object_body="plate/", grasp_hands=grasp_hands,
        initial_self_clearance_m=robot.r.geometry(measured(model, data, robot))[0],
        scene_sha256=manifest["scene_sha256"])
    command_names = [model.joint(int(model.actuator_trnid[i, 0])).name for i in range(model.nu)]
    frames = StateObservationBuffer(observer)
    commands, metrics, tasks = [], [], []
    source_rows, source_times, source_step_indices, phase_codes = [], [], [], []
    original_arm_targets, issued_arm_targets, acquisition_offsets = [], [], []
    post_active_rows, post_age_rows = [], []
    post_commands = None
    post_report = dict(enabled=False, activation="after_measured_acquisition")
    post_start_index, post_start_time = None, None
    ref_index, inserted_steps = 0, 0
    previous_arm_command = raw["robot_qpos"][0, robot.arm_ids].copy()
    performance["setup_s"] = time.perf_counter()-started
    loop_started = time.perf_counter()
    error = None
    try:
        while ref_index < len(timing_result["target_qpos"]):
            if acquisition is not None and ref_index == acquisition.boundary and acquisition.machine.phase == "approach":
                acquisition.prepare(data)
            holding = acquisition is not None and acquisition.machine.phase in ("settle", "close")
            source_row = acquisition.row if holding else int(timing_result["source_command_row"][ref_index])
            q = acquisition.target_q.copy() if holding else (acquisition.targets[ref_index] if acquisition is not None else timing_result["target_qpos"][ref_index])
            phase = acquisition.machine.phase if acquisition is not None else "approach"
            if holding:
                arm_command = np.clip(q[robot.arm_ids], previous_arm_command-model.opt.timestep,
                                      previous_arm_command+model.opt.timestep)
                baseline_arm = raw["robot_qpos"][acquisition.row, robot.arm_ids]
            else:
                arm_command = q[robot.arm_ids] if acquisition is not None else arm_targets[ref_index]
                baseline_arm = timing_result["target_qpos"][ref_index, robot.arm_ids]
            acquisition_command = arm_command.copy()
            post_active = post_acquisition_arm_feedforward_policy is not None and phase == "carry"
            if post_active:
                if post_commands is None:
                    post_start_index, post_start_time = ref_index, float(data.time)
                    post_commands, post_report = resumed_arm_feedforward(model, ARMS,
                        acquisition.targets[ref_index:, robot.arm_ids], physical_clock[ref_index+1:],
                        previous_arm_command, post_acquisition_arm_feedforward_policy)
                    post_report.update(activation_actual_time_s=post_start_time,
                        activation_source_step_index=post_start_index, activation_source_row=source_row)
                    _json(output/"post-acquisition-arm-feedforward.json", post_report)
                arm_command = post_commands[ref_index-post_start_index].copy()
            previous_arm_command = arm_command.copy()
            values = dict(zip(ARMS, arm_command))
            values.update(zip(BASE, robot.base(q)))
            values.update(zip(NECK, q[robot.neck_ids]))
            values.update({name: grips[side][source_row] for name, side in zip(GRIPPERS, ("left", "right"))})
            if acquisition is not None and phase in ("approach", "settle"):
                values[GRIPPERS[0]] = 2.
            for name, value in values.items():
                data.ctrl[model.actuator(name).id] = value
            source_rows.append(source_row)
            source_times.append(float(clock[acquisition.row]) if holding else float(timing_result["source_clock_at_boundary"][ref_index]))
            source_step_indices.append(ref_index)
            phase_codes.append(acquisition.machine.CODES[phase] if acquisition is not None else 0)
            original_arm_targets.append(baseline_arm.copy())
            issued_arm_targets.append(arm_command.copy())
            acquisition_offsets.append(acquisition_command-baseline_arm if acquisition is not None else np.zeros(len(ARMS)))
            post_active_rows.append(post_active)
            post_age_rows.append(float(data.time)-post_start_time if post_active else 0.)
            # Observation/control are pre-action. Only robot actuators are written.
            synchronize_observation(model, data)
            capture_started = time.perf_counter()
            frames.capture(data); commands.append(data.ctrl.copy())
            performance["capture_s"] += time.perf_counter()-capture_started
            c = contact_metrics(model, data)
            metrics.append(c)
            tasks.append(task_success(model, data, manifest["native"]["target_sites"], placement, c))
            mujoco.mj_step(model, data)
            solved_contacts = gates.sample_contacts(data)
            inactive_audit.sample(data)
            pad_forces = acquisition.pad_forces(data) if holding else None
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise ValueError("Nonfinite integrated Reachy/plate state")
            if abs(data.time-(physical_clock[0]+len(frames)*model.opt.timestep)) > 1e-9:
                raise ValueError("Simulator clock diverged/reset during integration")
            # Solved forces belong to mj_step's contacts. Cache synchronization
            # is separate and may replace those contacts; never sample later.
            synchronize_observation(model, data)
            current_task = task_success(model, data, manifest["native"]["target_sites"], placement)
            gates.update(data, contact_sample=solved_contacts,
                grasp_intent={side: bool(values[source_row]) for side, values in grasp_intent.items()},
                task_satisfied=current_task["success"],
                self_clearance_m=robot.r.geometry(measured(model, data, robot))[0])
            if holding:
                inserted_steps += 1
                acquisition.update(data, pad_forces)
                if acquisition.machine.phase == "failed":
                    raise ValueError("Acquisition "+acquisition.machine.failure)
            else:
                ref_index += 1
    except Exception as exc:
        error = type(exc).__name__+": "+str(exc)
    performance["rollout_s"] = time.perf_counter()-loop_started
    synchronize_observation(model, data)
    terminal = observer.capture(data)
    terminal_contact = contact_metrics(model, data)
    final_task = task_success(model, data, manifest["native"]["target_sites"], placement, terminal_contact)
    peak = max([initial["max_robot_penetration_m"], terminal_contact["max_robot_penetration_m"]]+[c["max_robot_penetration_m"] for c in metrics])
    floor = final_task["plate_floor_contact"] or any(t["plate_floor_contact"] for t in tasks)
    expected_steps = len(physical_clock)-1+inserted_steps
    expected_final_time = physical_clock[-1]+inserted_steps*model.opt.timestep
    acquisition_report = acquisition.report() if acquisition is not None else {"enabled": False}
    _json(output/"post-acquisition-arm-feedforward.json", post_report)
    passed = bool(not error and len(frames) == expected_steps and final_task["success"] and not floor and peak <= penetration_limit_m)
    common = gates.finish(expected_steps=expected_steps, expected_final_time_s=expected_final_time)
    inactive_report = inactive_audit.finish(expected_steps=expected_steps)
    _json(output/"inactive-hand-contact-audit.json", inactive_report)
    report = {"status": "diagnostic_task_pass" if passed else "diagnostic_failed", "physics_validated": False,
        "diagnostic_only": True, "native_scene_task_pass": final_task["success"], "diagnostic_gate_passed": passed,
        "validation_scope": "Native-scene diagnostic plus separately reported measured common gates and independent actuator replay; promotion requires review",
        "missing_common_validation_gates": common["missing_fields"], "common_validation": common,
        "physical_steps": len(frames), "source_frames": len(clock), "source_task_success": source["metadata"].get("source_task_success"),
        "native_task_predicate_final": final_task, "max_robot_penetration_m": peak, "penetration_limit_m": penetration_limit_m,
        "plate_floor_contact_any": floor, "error": error, "admission": admission,
        "scene_path": manifest["scene_path"], "scene_sha256": manifest["scene_sha256"],
        "grasp_relocation": plan.get("grasp_relocation"), "pad_calibration": plan.get("pad_calibration"),
        "arm_feedforward": arm_feedforward_report, "acquisition_rendezvous": acquisition_report,
        "source_reference_steps_completed": ref_index, "expected_source_reference_steps": len(physical_clock)-1,
        "gripper_control_policy": gripper_policy_report, "inactive_hand_contact_audit": inactive_report,
        "post_acquisition_arm_feedforward": post_report,
        "producer_module_sha256": _sha(Path(__file__)),
        "simulation_assumptions": manifest["simulation_assumptions"], "original_source_clock_preserved": True,
        "simulation_profile": manifest.get("simulation_profile"),
        "source_contact_parameters_preserved": simulation_profile is None,
        "physical_clock_matches_source": not timing_result["summary"]["enabled"] and acquisition is None, "robot_timing": timing_result["summary"],
        "object_state_writes_after_reset": 0, "actuator_scope": "Reachy base, arms, neck and grippers only"}
    _json(output/"physical-validation.json", report)
    _json(output/"contact-history.json", {"rows": metrics, "terminal": terminal_contact, "task_rows": tasks})
    if len(frames) < 2:
        np.savez_compressed(output/"partial-final-state.npz", **terminal["arrays"])
        _json(output/"performance.json", dict(performance, total_s=time.perf_counter()-started, physical_steps=len(frames)))
        return report
    stage_started = time.perf_counter()
    frame_count = len(frames)
    record = frames.finish()
    del frames
    performance["stack_s"] = time.perf_counter()-stage_started
    arrays = record["arrays"]
    arrays.update({"command/joint_position": np.asarray(commands), "control_interval_s": np.full(frame_count, model.opt.timestep),
                   "reference/source_clock_s": np.asarray(source_times),
                   "source/native_timestamp": clock,
                   "source/action": source["arrays"]["source/action"],
                   "derived/source_command_row": np.asarray(source_rows),
                   "derived/source_step_index": np.asarray(source_step_indices),
                   "controller/acquisition_phase_code": np.asarray(phase_codes)})
    arrays["reference/arm_joint_position"] = np.asarray(original_arm_targets)
    arrays["derived/acquisition_arm_offset_rad"] = np.asarray(acquisition_offsets)
    arrays["derived/arm_feedforward_offset_rad"] = np.asarray(issued_arm_targets)-arrays["reference/arm_joint_position"]-arrays["derived/acquisition_arm_offset_rad"]
    arrays["controller/post_acquisition_feedforward_active"] = np.asarray(post_active_rows, dtype=bool)
    arrays["controller/post_acquisition_feedforward_age_s"] = np.asarray(post_age_rows)
    arrays["source/original_robot_qpos"] = raw["robot_qpos"]
    arrays["source/original_hand_targets"] = raw["target_hand_matrix"]
    np.savez_compressed(output/"actual-phase-source-clock-map.npz", timestamp=arrays["timestamp"],
        source_clock_s=np.asarray(source_times), source_command_row=np.asarray(source_rows),
        source_step_index=np.asarray(source_step_indices), acquisition_phase_code=np.asarray(phase_codes),
        terminal_timestamp=terminal["arrays"]["timestamp"], source_native_timestamp=clock)
    for side in ("left", "right"):
        arrays["source/"+side+"_pinch_pose"] = source["arrays"]["source/"+side+"_pinch_pose"]
        arrays["reference/"+side+"_gripper_control_target_rad"] = grips[side]
        arrays["reference/"+side+"_default_binary_gripper_target_rad"] = baseline_grips[side]
    for key, values in source["arrays"].items():
        if key.startswith("objects/"):
            arrays["source/"+key] = values
    arrays.update({"terminal/"+key: value for key, value in terminal["arrays"].items()})
    # Lossless stacked arrays own every numeric value. Drop duplicate per-step
    # lists before export/replay, rather than retaining two complete recordings.
    del commands, original_arm_targets, issued_arm_targets, acquisition_offsets, metrics, tasks
    del post_active_rows, post_age_rows, post_commands
    import gc
    gc.collect()
    performance["stacked_numeric_bytes"] = sum(value.nbytes for value in arrays.values())
    performance["stacked_physics_rows"] = frame_count
    performance["duplicate_frame_buffers_released"] = True
    # Preserve expensive states before the separately reconstructed replay. The
    # audit is permitted to fail; its failure must not erase the original run.
    stage_started = time.perf_counter()
    with (output/"physical-observations.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    performance["raw_npz_export_s"] = time.perf_counter()-stage_started
    stage_started = time.perf_counter()
    try:
        audit = verify_actuator_replay(scene, arrays, expected_scene_sha256=manifest["scene_sha256"],
            expected_assets=asset_binding, active_object_body="plate/", active_object_joint="plate/",
            object_pose_key="observation/objects/plate/pose",
            initial_state=np.load(output/"initial-state.npz", allow_pickle=False))
    except Exception as exc:
        audit = {"actuator_replay_pass": False, "scene_sha256": manifest["scene_sha256"],
                 "error": type(exc).__name__+": "+str(exc)}
    performance["actuator_audit_s"] = time.perf_counter()-stage_started
    _json(output/"actuator-replay-audit.json", audit)
    common = gates.finish(expected_steps=expected_steps, expected_final_time_s=expected_final_time,
                          actuator_replay=audit)
    report.update(common_validation=common, missing_common_validation_gates=common["missing_fields"])
    _json(output/"physical-validation.json", report)
    meta = dict(record["metadata"], source_sequence=source["metadata"]["source_sequence"], source_group=source["metadata"]["source_group"],
        source_urls=source["metadata"]["source_urls"], source_revision=source["metadata"]["source_revision"],
        source_metadata=source["metadata"], status=report["status"],
        physics_validated=False, diagnostic_only=True, validation_scope=report["validation_scope"],
        physical_validation=report, missing_fields=record["missing_fields"]+report["missing_common_validation_gates"],
        gripper_control_policy=gripper_policy_report, arm_feedforward=arm_feedforward_report, acquisition_rendezvous=acquisition_report,
        post_acquisition_arm_feedforward=post_report,
        producer_module_sha256=_sha(Path(__file__)),
        derived_fields={"command/joint_position": "Actually issued absolute position-servo commands derived from IK, not original native robot actions",
                        "reference/source_clock_s": "Monotone piecewise-linear map from physical time to original source time; source/native_timestamp remains unchanged",
                        "derived/source_command_row": "Original command-row index for each actually applied physical substep"},
        simulation_assumptions=manifest["simulation_assumptions"],
        simulation_profile=manifest.get("simulation_profile"),
        source_contact_parameters_preserved=simulation_profile is None, command_joint_names=command_names,
        command_joint_units=["m" if int(model.jnt_type[model.joint(name).id]) == 2 else "rad" for name in command_names],
        command_semantics={"command/joint_position": {"kind": "joint_position", "representation": "absolute", "timing": "pre_action", "joint_names": command_names}},
        command_profile={"required_fields": ["command/joint_position"]}, issued_control_modes=["joint_position"],
        action_semantics="Actual absolute Reachy position-servo controls, held for one "+str(model.opt.timestep)+" s physical step; timing map recorded separately",
        observation_timing="pre_action", terminal_storage="terminal/* is the actual final integrated state, not an extra command row",
        rgb_stored=False, source_ik_artifact=str(ik_attempt/"raw-ik.npz"))
    meta["control_labels"] = {"desired_joint_targets": "command/joint_position; exact applied position-servo controls",
        "desired_ee_targets": "not issued", "controller_state": "Recorded bounded acquisition rendezvous" if acquisition is not None else "stateless position-servo target selection",
        "substep_controls": "One mj_step per row; command/joint_position equals actual data.ctrl for that step"}
    meta["derived_fields"]["controller/acquisition_phase_code"] = "Declared bounded source-command clock rendezvous phase; code mapping in acquisition_rendezvous.phase_codes"
    meta["derived_fields"]["derived/acquisition_arm_offset_rad"] = "Robot-only measured closing-axis centering with bounded command settling; relative to original per-step IK reference; original rows unchanged"
    meta["derived_fields"]["reference/*_gripper_control_target_rad"] = "Derived per-source-row motor targets under the declared gripper policy; actual issued substeps are command/joint_position"
    meta["derived_fields"]["derived/arm_feedforward_offset_rad"] = "Actual issued arm position command minus unchanged interpolated arm reference, ordered by arm_feedforward.arm_joint_names when enabled or the recorded left/right arm joint groups"
    meta["derived_fields"]["controller/post_acquisition_feedforward_active"] = "True only after measured rendezvous success and reference resumption under the explicit post-acquisition policy; entry/settle/close remain false"
    meta["derived_fields"]["controller/post_acquisition_feedforward_age_s"] = "Actual elapsed simulation time since resumption, zero while inactive; determines the saved smooth boundary taper"
    meta["derived_fields"]["reference/*_default_binary_gripper_target_rad"] = "Retained baseline endpoint conversion of original binary source intent; not original source robot joint commands"
    meta["clock"] = dict(meta["clock"], sampling_rate_hz=float(1/model.opt.timestep),
                         origin="First selected native replay timestamp; original source file has no timestamp array")
    _json(output/"physical-observation-metadata.json", meta)
    stage_started = time.perf_counter()
    archive = write_archive(root, "bigym", episode_id, arrays, meta)
    performance["common_archive_export_s"] = time.perf_counter()-stage_started
    performance.update(total_s=time.perf_counter()-started, physical_steps=frame_count,
                       capture_included_in_rollout=True, timing_scope="Wall-clock measurements; no state/control changes")
    _json(output/"performance.json", performance)
    return dict(report, archive=str(archive))
