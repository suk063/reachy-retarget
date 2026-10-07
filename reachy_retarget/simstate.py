"""Decode recorded robosuite free-body poses without importing a simulator SDK."""

import xml.etree.ElementTree as ET
import numpy as np


ACTIVE_OBJECTS = {
    "PickPlaceCan": {"Can"}, "Lift": {"cube"}, "NutAssemblySquare": {"SquareNut"},
    "ToolHang": {"stand", "frame", "tool"},
    "TwoArmTransport": {"payload", "trash", "transport_start_bin", "transport_target_bin",
                         "transport_trash_bin", "transport_start_bin_lid"},
    "Stack_D0": {"cubeA", "cubeB"}, "Threading_D0": {"needle_obj", "tripod_obj"},
}


def free_object_poses(model_xml, states, env_name):
    """Read time+qpos+qvel records using the embedded XML's joint traversal.

    MuJoCo free joints are world-child bodies with xyz+wxyz qpos. No dynamic
    stepping, target assignment or inferred object observation ordering is used.
    Articulated objects and unresolved XML includes are deliberately unsupported.
    """
    if env_name not in ACTIVE_OBJECTS:
        raise ValueError(f"Unverified object schema: {env_name}")
    tree = ET.fromstring(model_xml)
    if tree.tag != "mujoco":
        raise ValueError("Recorded model must be a MuJoCo XML document")
    if any(tree.find(".//"+tag) is not None for tag in ("include", "replicate", "attach", "composite", "flexcomp", "frame")):
        raise ValueError("Unexpanded XML requires an explicit model adapter")
    if any(j.get("type", "hinge") != "hinge" for j in tree.findall(".//default/joint")):
        raise ValueError("Joint-type defaults require explicit resolution")
    if any(c.get("alignfree") == "true" for c in tree.findall("compiler")) or any(
        j.get("align") == "true" for j in tree.findall(".//freejoint")
    ):
        raise ValueError("Aligned free-joint frames require a compiled model adapter")
    world = tree.find("worldbody")
    if world is None:
        raise ValueError("Recorded model has no worldbody")
    nq = nv = 0
    bodies = {}

    def visit(body, parent):
        nonlocal nq, nv
        # The compiled order groups this body's joints before its children,
        # even when child XML elements precede a joint element in the source.
        joints = [node for node in body if node.tag in ("joint", "freejoint")]
        for node in joints:
            kind = "free" if node.tag == "freejoint" else node.get("type", "hinge")
            sizes = {"free": (7, 6), "ball": (4, 3), "hinge": (1, 1), "slide": (1, 1)}
            if kind not in sizes:
                raise ValueError(f"Unverified joint type: {kind}")
            qsize, vsize = sizes[kind]
            if kind == "free":
                if parent is not world:
                    raise ValueError("Free body is not a direct child of worldbody")
                if len(joints) != 1:
                    raise ValueError("A free body must have exactly one joint")
                name = body.get("name", "").removesuffix("_main").removesuffix("_root")
                if not name or name in bodies:
                    raise ValueError(f"Missing or ambiguous canonical free-body name: {name!r}")
                bodies[name] = {"qpos_address": nq, "body": body.get("name"), "joint": node.get("name")}
            nq += qsize
            nv += vsize
        for child in body.findall("body"):
            visit(child, body)
    for body in world.findall("body"):
        visit(body, world)
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] != 1+nq+nv or not np.isfinite(states).all():
        raise ValueError(f"Recorded state width must be time+nq+nv={1+nq+nv}; got {states.shape}")
    selected = ACTIVE_OBJECTS[env_name]
    if not selected <= bodies.keys():
        raise ValueError(f"Required object bodies missing: {selected-bodies.keys()}")
    arrays, metadata = {}, {}
    for name in sorted(selected):
        spec = bodies[name]
        start = 1+spec["qpos_address"]
        poses = states[:, start:start+7].copy()
        if not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1., atol=1e-5, rtol=0):
            raise ValueError(f"Non-unit recorded object quaternion: {name}")
        arrays[f"objects/{name}/pose"] = poses
        metadata[name] = dict(spec, pose_frame="world", pose_source="recorded simulator free-joint qpos",
                              geometry_source="embedded model_xml; asset files may require separate acquisition",
                              mass_friction="source model values retained; not independently calibrated")
    evidence = {"nq": nq, "nv": nv, "layout": "time, qpos, qvel", "quaternion": "wxyz",
                "inactive_free_bodies_omitted": sorted(bodies.keys()-selected),
                "references": ["https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/utils/binding_utils.py",
                               "https://github.com/ARISE-Initiative/robosuite/blob/v1.5.1/robosuite/utils/binding_utils.py",
                               "https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint"]}
    return arrays, metadata, evidence
