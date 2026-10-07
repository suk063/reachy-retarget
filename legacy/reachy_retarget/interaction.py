"""Name-independent contact seeds and interaction contracts.

Inputs are geometry, measured trajectories and signed closure, never task labels.
The seeds are proposals for forward-dynamics optimization, not grasp guarantees.
"""

from dataclasses import asdict, dataclass
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ContactPhase:
    start: int
    stop: int
    mode: str


def phases(closure):
    closure = np.asarray(closure)
    if closure.ndim != 1 or not len(closure) or not np.isin(closure, (-1, 1)).all():
        raise ValueError("Expected aligned signed open/close states")
    cuts = np.r_[0, np.flatnonzero(np.diff(closure)) + 1, len(closure)]
    return [ContactPhase(int(a), int(b), "grasp" if closure[a] > 0 else "free")
            for a, b in zip(cuts[:-1], cuts[1:])]


def contact_contract(closure):
    schedule = phases(closure)
    return {"schema": "reachy-interaction-v1", "mode": "grasp_carry",
            "phases": [asdict(p) for p in schedule],
            "terminal_contact": bool(closure[-1] > 0),
            "release_required": bool(any(p.mode == "grasp" for p in schedule) and closure[-1] < 0),
            "derived_from": "source gripper command, interpreted by the verified source adapter",
            "limitation": "one active rigid object and right gripper; other modes require explicit adapters"}


def contact_gates(mode, *, acquired, fraction, translation, rotation):
    """Mode-specific checks; never require lifting for a push or pull."""
    if mode == "grasp_carry":
        return {"bilateral_lifted_grasp": bool(acquired), "carry_contact_95pct": fraction >= .95,
                "grasp_translation_3mm": translation <= .003,
                "grasp_rotation_3deg": rotation <= np.deg2rad(3)}
    raise ValueError("No complete physical contact contract for mode: " + mode)


def add_search_sensors(xml, object_body):
    """Read-only sensors used by native sampled rollouts; no new constraints."""
    root = ET.fromstring(xml)
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "framepos", name="retarget_tcp_pos", objtype="site", objname="r_arm_tip_tcp")
    ET.SubElement(sensor, "framequat", name="retarget_tcp_quat", objtype="site", objname="r_arm_tip_tcp")
    for index, finger in enumerate(("r_hand_distal_link", "r_hand_distal_mimic_link")):
        ET.SubElement(sensor, "contact", name=f"retarget_contact_{index}",
                      body1=finger, body2=object_body, data="force", reduce="maxforce")
    ET.SubElement(sensor, "contact", name="retarget_self_contact", subtree1="base_link",
                  subtree2="base_link", data="force", reduce="maxforce")
    ET.SubElement(sensor, "contact", name="retarget_object_depth", body1=object_body,
                  data="dist", reduce="mindist")
    ET.SubElement(sensor, "contact", name="retarget_robot_depth", subtree1="base_link",
                  data="dist", reduce="mindist")
    return ET.tostring(root, encoding="unicode")


def _descendants(model, root):
    result = {int(root)}
    for body in range(int(root) + 1, model.nbody):
        if int(model.body_parentid[body]) in result:
            result.add(body)
    return result


def geometry_seed(model, data, body, source_local, object_rotation, jaw_local, *, rank=0):
    """Choose a nearby collision region and its narrow horizontal grasp axis.

    Axes come from each compiled primitive/mesh, not object names or task IDs.
    Object geometry and inertia are read only. Friction gives a conservative
    starting force, bounded by the same global range for every episode.
    """
    import mujoco

    bodies = _descendants(model, body)
    inverse = data.xmat[body].reshape(3, 3).T
    choices = []
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) not in bodies or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        center = inverse @ (data.geom_xpos[geom] - data.xpos[body])
        axes = inverse @ data.geom_xmat[geom].reshape(3, 3)
        kind = int(model.geom_type[geom])
        size = model.geom_size[geom]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            extents = 2 * size
        elif kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            extents = 2 * np.array([size[0], size[0], size[1] + (size[0] if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else 0)])
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            extents = np.full(3, 2 * size[0])
        elif kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(model.geom_dataid[geom])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices = model.mesh_vert[start:start + count]
            center += axes @ ((vertices.min(0) + vertices.max(0)) / 2)
            extents = np.ptp(vertices, axis=0)
        else:
            continue
        # Stable geometric tie breaking does not depend on body/geom names.
        choices.append((float(np.linalg.norm(center-source_local)), tuple(center), geom, center, axes, extents))
    if not choices:
        raise ValueError("No supported collision region for contact seeding")
    choices.sort(key=lambda x: (x[0], x[1]))
    _, _, geom, center, axes, extents = choices[int(rank) % len(choices)]
    world_axes = object_rotation @ axes
    horizontal = [i for i in range(3) if abs(world_axes[2, i]) < .5]
    base_rotation = Rotation.from_euler("Y", -90, degrees=True).as_matrix()
    jaw = base_rotation @ np.asarray(jaw_local)
    default_yaw = np.arctan2(jaw[1], jaw[0])
    options = []
    for axis in horizontal:
        yaw = (np.arctan2(world_axes[1, axis], world_axes[0, axis])-default_yaw+np.pi/2) % np.pi-np.pi/2
        options.append((round(float(extents[axis]), 5), abs(float(yaw)), float(yaw)))
    yaw = min(options)[2] if options else 0.
    # Rotationally symmetric cross sections admit the least-rotation approach.
    kind = int(model.geom_type[geom])
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE or (kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE)
                                              and abs(world_axes[2, 2]) > .9):
        yaw = 0.
    if kind == mujoco.mjtGeom.mjGEOM_MESH and len(horizontal) == 2:
        # A nearly circular polygonal cross section should not inherit the
        # arbitrary eigenvector orientation of a mesh compiler's inertia frame.
        mesh = int(model.geom_dataid[geom])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices = model.mesh_vert[start:start+count][:, horizontal]
        xy = vertices-(vertices.min(0)+vertices.max(0))/2
        radii = np.linalg.norm(xy, axis=1)
        boundary = radii > .95*max(float(radii.max()), 1e-12)
        angular_bins = np.unique(np.clip(np.floor((np.arctan2(xy[boundary, 1], xy[boundary, 0])+np.pi)/(2*np.pi)*32),0,31))
        if (len(angular_bins) >= 20
                and np.ptp(extents[horizontal]) < .03*np.mean(extents[horizontal])):
            yaw = 0.
    mass = float(sum(model.body_mass[b] for b in bodies))
    friction = max(.05, float(model.geom_friction[geom, 0]))
    force = float(np.clip(1.5 * mass * np.linalg.norm(model.opt.gravity) / (2 * friction), 1.5, 5.))
    return {"point": center, "yaw_deg": float(np.rad2deg(yaw)), "normal_force_n": force,
            "collision_geom": model.geom(geom).name, "region_count": len(choices),
            "mass_kg": mass, "sliding_friction": friction, "extents_m": extents.tolist(),
            "policy": "nearest collision region, narrow horizontal axis, mass/friction force seed"}


def supported_release(position, goal, support_force, weight, *, lifted, near_release):
    """Release only near the intended placement with actual upward support."""
    error = np.asarray(position)-np.asarray(goal)
    return bool(lifted and near_release and np.linalg.norm(error[:2]) < .015
                and abs(error[2]) < .008 and support_force >= max(.03, .5*weight))
