"""Contact-faithful MuJoCo conversion of the pinned Reachy URDF.

The object has a free joint and is assigned a pose only at episode reset.
No object tracking actuator, equality, mocap body or kinematic correction exists.
"""

import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from .robot import CONTROL, ARMS, NECK
from .store import sha256, json_write
from .episodes import matrices_to_pose

BASE = ("base_x", "base_y", "base_yaw")
GRIPPERS = ("l_hand_finger", "r_hand_finger")


def numbers(a):
    return " ".join(f"{x:.12g}" for x in np.asarray(a).ravel())


def origin(e):
    o = e.find("origin")
    T = np.eye(4)
    if o is not None:
        T[:3, 3] = np.fromstring(o.get("xyz", "0 0 0"), sep=" ")
        T[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(o.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return T


def attrs(T):
    p = matrices_to_pose(T)
    return dict(pos=numbers(p[:3]), quat=numbers(p[3:]))


def prepare_robot(root, *, include_neck=False, state_profile=False):
    include_neck = include_neck or state_profile
    folder = Path(root) / "data/assets" / ("reachy_mujoco_state" if state_profile else "reachy_mujoco")
    folder.mkdir(parents=True, exist_ok=True)
    u = ET.parse(CONTROL / "asset/reachy.urdf").getroot()
    links = {e.get("name"): e for e in u.findall("link")}
    joints = {e.get("name"): e for e in u.findall("joint")}
    children = {}
    for j in joints.values():
        children.setdefault(j.find("parent").get("link"), []).append(j)

    def fixed(name):
        m = joints[name].find("mimic")
        return (
            0.0
            if m is None
            else fixed(m.get("joint")) * float(m.get("multiplier", 1))
            + float(m.get("offset", 0))
        )

    dynamic = set(ARMS) | {n for n in joints if "hand_finger" in n}
    if include_neck:
        dynamic.update(NECK)
    mj = ET.Element("mujoco", model="reachy_state_replay")
    ET.SubElement(
        mj,
        "compiler",
        angle="radian",
        autolimits="true",
        fusestatic="false",
        balanceinertia="false",
    )
    ET.SubElement(
        mj,
        "option",
        timestep=".002",
        integrator="implicitfast",
        gravity="0 0 -9.81",
        cone="elliptic",
        impratio="10",
        iterations="100",
    )
    vis = ET.SubElement(mj, "visual")
    ET.SubElement(vis, "global", offwidth="960", offheight="720")
    default = ET.SubElement(mj, "default")
    ET.SubElement(
        default, "geom", friction="1 .01 .001", solref=".004 1", solimp=".95 .99 .001"
    )
    asset = ET.SubElement(mj, "asset")
    world = ET.SubElement(mj, "worldbody")
    act = ET.SubElement(mj, "actuator")
    eq = ET.SubElement(mj, "equality")
    contact = ET.SubElement(mj, "contact")
    ET.SubElement(world, "light", pos="0 -2 4", dir="0 0 -1", diffuse=".8 .8 .8")
    ET.SubElement(world, "light", pos="2 2 3", dir="-1 -1 -1", diffuse=".6 .6 .6")
    ET.SubElement(
        world, "geom", name="floor", type="plane", size="10 10 .1", rgba=".18 .20 .24 1"
    )
    mesh_cache = {}
    source_hashes = {}
    mimics = {}

    def geometry(body, element, name, index):
        g = element.find("geometry")
        spec = dict(name=f"{name}_collision_{index}", mass="0", rgba=".70 .76 .82 1")
        if g.find("mesh") is not None:
            tag = g.find("mesh")
            uri = tag.get("filename")
            relative = (
                uri.split("/share/")[-1]
                if "/share/" in uri
                else uri.removeprefix("package://")
            )
            p = CONTROL / "asset/packages" / relative
            if not p.exists():
                raise FileNotFoundError(p)
            scale = np.fromstring(tag.get("scale", "1 1 1"), sep=" ")
            key = (str(p), tuple(scale))
            if key not in mesh_cache:
                scene = trimesh.load(p, force="scene", process=False)
                parts = []
                for node in sorted(scene.graph.nodes_geometry):
                    transform, geom = scene.graph[node]
                    part = scene.geometry[geom].copy()
                    part.apply_transform(transform)
                    part.vertices *= scale
                    parts.append(part)
                combined = trimesh.util.concatenate(parts)
                mesh_name = "mesh_" + sha256(p)[:14] + "_" + str(len(mesh_cache))
                dest = folder / (mesh_name + ".stl")
                combined.export(dest)
                ET.SubElement(asset, "mesh", name=mesh_name, file=str(dest))
                mesh_cache[key] = mesh_name
                source_hashes[str(p)] = sha256(p)
            spec.update(type="mesh", mesh=mesh_cache[key])
        elif g.find("box") is not None:
            spec.update(
                type="box",
                size=numbers(np.fromstring(g.find("box").get("size"), sep=" ") / 2),
            )
        elif g.find("sphere") is not None:
            spec.update(type="sphere", size=g.find("sphere").get("radius"))
        elif g.find("cylinder") is not None:
            c = g.find("cylinder")
            spec.update(
                type="cylinder", size=f"{c.get('radius')} {float(c.get('length')) / 2}"
            )
        else:
            raise ValueError("Unknown geometry " + name)
        ET.SubElement(body, "geom", **spec, **attrs(origin(element)))

    def visit(parent, name, joint=None):
        T = np.eye(4) if joint is None else origin(joint)
        jn = joint.get("name") if joint is not None else None
        if jn and jn not in dynamic and joint.get("type") != "fixed":
            v = fixed(jn)
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            if joint.get("type") == "prismatic":
                T[:3, 3] += T[:3, :3] @ (axis * v)
            else:
                T[:3, :3] = T[:3, :3] @ Rotation.from_rotvec(axis * v).as_matrix()
        b = ET.SubElement(parent, "body", name=name, gravcomp="1", **attrs(T))
        if joint is None:
            for n, kind, axis in zip(
                BASE, ("slide", "slide", "hinge"), ("1 0 0", "0 1 0", "0 0 1")
            ):
                ET.SubElement(b, "joint", name=n, type=kind, axis=axis, limited="false")
                ET.SubElement(
                    act,
                    "position",
                    name=n,
                    joint=n,
                    kp="50000" if kind == "slide" else "3000",
                    dampratio="1",
                    forcerange="-2000 2000" if kind == "slide" else "-500 500",
                )
        elif jn in dynamic:
            lim = joint.find("limit")
            spec = {
                "name": jn,
                "type": "hinge",
                "axis": joint.find("axis").get("xyz"),
                "damping": ".01",
                "armature": ".005",
            }
            if lim is not None:
                spec["range"] = lim.get("lower") + " " + lim.get("upper")
            ET.SubElement(b, "joint", **spec)
            mimic = joint.find("mimic")
            if mimic is not None:
                multiplier = float(mimic.get("multiplier", 1))
                offset = float(mimic.get("offset", 0))
                parent_joint = mimic.get("joint")
                mimics[jn] = (parent_joint, multiplier, offset)
                ET.SubElement(
                    eq,
                    "joint",
                    joint1=jn,
                    joint2=parent_joint,
                    polycoef=numbers([offset, multiplier, 0, 0, 0]),
                    solref=".004 1",
                )
            else:
                grip = jn in GRIPPERS
                ET.SubElement(
                    act,
                    "position",
                    name=jn,
                    joint=jn,
                    kp="8" if grip else "180",
                    kv=".8" if grip else "24",
                    forcerange="-2 2" if grip else "-40 40",
                    ctrlrange=spec["range"],
                )
        inertial = links[name].find("inertial")
        if inertial is not None and float(inertial.find("mass").get("value")) > 0:
            it = origin(inertial)
            iv = inertial.find("inertia").attrib
            I = np.array(
                [
                    [float(iv["ixx"]), float(iv["ixy"]), float(iv["ixz"])],
                    [float(iv["ixy"]), float(iv["iyy"]), float(iv["iyz"])],
                    [float(iv["ixz"]), float(iv["iyz"]), float(iv["izz"])],
                ]
            )
            I = it[:3, :3] @ I @ it[:3, :3].T
            ET.SubElement(
                b,
                "inertial",
                pos=numbers(it[:3, 3]),
                mass=inertial.find("mass").get("value"),
                fullinertia=numbers(
                    [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
                ),
            )
        for i, e in enumerate(links[name].findall("collision")):
            geometry(b, e, name, i)
        if name in ("l_arm_tip", "r_arm_tip"):
            ET.SubElement(b, "site", name=name + "_tcp", size=".004", rgba="1 .2 0 1")
        if name.endswith(("hand_distal_link", "hand_distal_mimic_link")):
            ET.SubElement(
                b,
                "site",
                name=name + "_endpoint",
                pos="0 0 .0461",
                size=".002",
                rgba="0 1 0 1",
            )
        for child in children.get(name, []):
            visit(b, child.find("child").get("link"), child)

    visit(world, "base_link")
    camera_frames = {}
    if state_profile:
        body_elements = {b.get("name"): b for b in world.iter("body")}
        for name, frame in (("torso", "depth_cam_rgb_optical"),
                            ("left_head", "left_camera_optical"),
                            ("right_head", "right_camera_optical")):
            optical_to_mujoco = np.eye(4)
            optical_to_mujoco[:3, :3] = np.diag([1., -1., -1.])
            ET.SubElement(body_elements[frame], "camera", name=name, **attrs(optical_to_mujoco))
            camera_frames[name] = {"frame": frame, "pose_source": "URDF optical frame", "rgb_recorded": False,
                                   "calibration": "URDF nominal extrinsics; no fitted lens/mount calibration asserted"}
    exclusions = {
        tuple(sorted((j.find("parent").get("link"), j.find("child").get("link"))))
        for j in joints.values()
    }
    # Multi-axis joints in this URDF use intermediate massless/dummy links.
    # Exclude neighboring physical segments across those dummy links, just as
    # an ordinary parent-child hinge is excluded by MuJoCo.
    parents = {
        j.find("child").get("link"): j.find("parent").get("link")
        for j in joints.values()
    }
    collision_links = {n for n, e in links.items() if e.findall("collision")}
    for n in collision_links:
        p = parents.get(n)
        while p is not None and p not in collision_links:
            p = parents.get(p)
        if p is not None:
            exclusions.add(tuple(sorted((n, p))))
    srdf = CONTROL / "asset/reachy.srdf"
    if srdf.exists():
        exclusions |= {
            tuple(sorted((e.get("link1"), e.get("link2"))))
            for e in ET.parse(srdf).getroot().findall("disable_collisions")
            if e.get("reason") == "Adjacent"
        }
    for a, b in sorted(exclusions):
        ET.SubElement(contact, "exclude", body1=a, body2=b)
    manifest = {
        "recording_profile": "complete-state-no-rgb-v1" if state_profile else "legacy",
        "cameras": camera_frames,
        "neck_joints_enabled": include_neck,
        "source_urdf": str(CONTROL / "asset/reachy.urdf"),
        "source_urdf_sha256": sha256(CONTROL / "asset/reachy.urdf"),
        "meshes": source_hashes,
        "mimics": mimics,
        "adjacent_body_exclusions": sorted(exclusions),
        "physics_hz": 500,
        "control_hz": 100,
        "assumptions": [
            "Robot-only gravity compensation; object gravity remains enabled",
            "Planar ideal base servos; wheels locked",
            "Manufacturer collision meshes used as convex hulls",
            "Finger gains kp=8 kv=.8 torque limit=2 Nm; arm kp=180 kv=24 limit=40 Nm",
            "Unspecified robot/environment friction=1,.01,.001; source Can friction retained",
        ],
        "reference_implementation": "/home/sunghwan/workspace/reachy-agent/simulation/prepare.py",
    }
    (folder / "robot.xml").write_text(ET.tostring(mj, encoding="unicode"))
    json_write(folder / "manifest.json", manifest)
    return mj, manifest


def scene(root, source_xml, placement, initial_object):
    folder = Path(root) / "data/assets/reachy_mujoco"
    if (folder / "robot.xml").exists():
        mj = ET.parse(folder / "robot.xml").getroot()
        manifest = json.loads((folder / "manifest.json").read_text())
    else:
        mj, manifest = prepare_robot(root)
    world = mj.find("worldbody")
    asset = mj.find("asset")
    source = ET.fromstring(source_xml)
    for name in ("bin1", "bin2"):
        b = copy.deepcopy(source.find(f'.//body[@name="{name}"]'))
        T = np.eye(4)
        T[:3, 3] = np.fromstring(b.get("pos"), sep=" ")
        b.attrib.update(attrs(placement @ T))
        for i, g in enumerate(list(b)):
            if g.tag != "geom":
                b.remove(g)
                continue
            if g.get("contype") == "0" and not "leg" in g.get("name", ""):
                b.remove(g)
                continue
            g.attrib.pop("material", None)
            g.set("name", f"{name}_geom_{i}")
            g.set("rgba", ".35 .3 .2 1" if name == "bin1" else ".2 .4 .28 1")
        world.append(b)
    can = copy.deepcopy(source.find('.//body[@name="Can_main"]'))
    can.attrib.update(attrs(initial_object))
    for g in list(can):
        if g.tag == "geom" and g.get("contype") == "0":
            can.remove(g)
    can.find("geom").set("rgba", ".8 .04 .06 1")
    ET.SubElement(
        asset,
        "mesh",
        name="Can_can_mesh",
        file=str(Path(root) / "data/raw/robosuite_can_assets/meshes/can.msh"),
    )
    world.append(can)
    return ET.tostring(mj, encoding="unicode"), manifest


def initialize(model, data, robot, q, mimics, grip=2.0):
    import mujoco
    from collections.abc import Mapping

    values = dict(zip(ARMS, q[robot.arm_ids]))
    values.update(zip(BASE, robot.base(q)))
    grippers = dict(grip) if isinstance(grip, Mapping) else {g: grip for g in GRIPPERS}
    if set(grippers) != set(GRIPPERS) or not all(np.isfinite(v) for v in grippers.values()):
        raise ValueError('Explicit gripper positions must give both finite gripper joint values')
    values.update(grippers)
    neck_names = [n for n in NECK if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0]
    values.update({n: q[robot.r.q_indices[n]] for n in neck_names})

    def value(n):
        if n not in values:
            parent, m, o = mimics[n]
            values[n] = value(parent) * m + o
        return values[n]

    for n in list(values) + list(mimics):
        data.qpos[model.joint(n).qposadr] = value(n)
    for n in (*BASE, *ARMS, *GRIPPERS, *neck_names):
        data.ctrl[model.actuator(n).id] = values[n]
    mujoco.mj_forward(model, data)


def measured(model, data, robot):
    import mujoco
    neck = np.array([data.qpos[model.joint(n).qposadr[0]] if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0 else 0.
                     for n in NECK])
    return robot.pack(
        np.array([data.qpos[model.joint(n).qposadr[0]] for n in ARMS]),
        np.array([data.qpos[model.joint(n).qposadr[0]] for n in BASE]),
        neck=neck,
    )
