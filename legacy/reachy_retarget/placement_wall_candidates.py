"""Deterministic robot placement proposals from the unchanged task-bin walls.

These are isolated kinematic seeds, not admissions. The original Can geometry,
source arrays, target region and all physical thresholds remain unchanged.
"""
import itertools
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .feasible_geometry_search import placement_bank
from .feasible_maniskill import intent_parameters
from .physics import initialize


def nearest_walls(model, data, target):
    """Find the nearest thin compiled source wall on each planar axis."""
    target = np.asarray(target, float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError('Finite target XY required')
    walls = []
    for geom in range(model.ngeom):
        name = model.geom(geom).name or ''
        if not name.startswith('source_scene_geom_') or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        corners = np.array(list(itertools.product((-1., 1.), repeat=3)))*model.geom_size[geom]
        corners = corners @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        lower, upper = corners.min(0), corners.max(0)
        size = upper-lower
        if size[2] < .05:
            continue
        for axis in (0, 1):
            other = 1-axis
            if size[axis] > .025 or not lower[other] <= target[other] <= upper[other]:
                continue
            if upper[axis] < target[axis]:
                distance, inward = target[axis]-upper[axis], np.eye(2)[axis]
            elif lower[axis] > target[axis]:
                distance, inward = lower[axis]-target[axis], -np.eye(2)[axis]
            else:
                continue
            walls.append(dict(geom=name, axis=axis, distance=float(distance),
                              inward=inward.tolist(), aabb=[lower.tolist(), upper.tolist()]))
    result = []
    for axis in (0, 1):
        matches = [wall for wall in walls if wall['axis'] == axis]
        if not matches:
            raise ValueError('No qualifying source task wall on planar axis '+str(axis))
        result.append(min(matches, key=lambda wall: (wall['distance'], wall['geom'])))
    return result


def yaw_toward(palm_offset, pitch, direction):
    rotated = Rotation.from_euler('y', pitch, degrees=True).apply(palm_offset)
    if np.linalg.norm(rotated[:2]) < 1e-6:
        raise ValueError('Release palm projection cannot define a placement azimuth')
    yaw = np.rad2deg(np.arctan2(direction[1], direction[0])-np.arctan2(rotated[1], rotated[0]))
    return float((yaw+180.) % 360.-180.)


def bank(prepared, robot, *, base_margin_m=.75, minimum_fixture_gap_m=.003,
         pose_tolerance_constraints=True, pitch_angles_deg=(30., 40., 50.)):
    """Up to nine declared wall-facing proposals with mobile base/right-arm IK."""
    if prepared[5].get('task') != 'PickPlaceCan':
        raise ValueError('Wall placement requires the declared original Can task region')
    if not np.isfinite(base_margin_m) or not 0. <= base_margin_m <= 1.5:
        raise ValueError('Explicit finite bounded robot workspace margin required')
    pitches = np.asarray(pitch_angles_deg, float)
    if (pitches.ndim != 1 or not 1 <= len(pitches) <= 3 or not np.isfinite(pitches).all()
            or np.any(pitches < 0) or np.any(pitches > 90)
            or len(np.unique(pitches)) != len(pitches)):
        raise ValueError('One to three distinct finite pitch angles in [0, 90] required')
    base = placement_bank(prepared)[0]['parameters']
    target = np.asarray(base['placement_xy_world'])
    model, manifest, details = prepared[0], prepared[2], prepared[5]
    times, reference, commands, _, objects = prepared[7:12]
    first = int(np.flatnonzero(commands < 0)[0])
    release = int(np.flatnonzero((np.arange(len(times)) > first) & (commands >= 1.9999))[0])
    data = mujoco.MjData(model)
    object_addresses = [(int(model.joint(spec['joint']).qposadr[0]),
                         int(model.joint(spec['joint']).type[0])) for spec in manifest['objects'].values()
                        if 'joint' in spec]
    original_qpos = data.qpos.copy()
    _, closed = intent_parameters(details)
    ref = reference[release]
    initialize(model, data, robot, robot.pack(ref[3:], ref[:3]), manifest['mimics'],
               {'l_hand_finger': 2., 'r_hand_finger': closed})
    for address, kind in object_addresses:
        width = 7 if kind == int(mujoco.mjtJoint.mjJNT_FREE) else 1
        if not np.array_equal(data.qpos[address:address+width], original_qpos[address:address+width]):
            raise ValueError('Robot-only proposal query modified a task object')
    walls = nearest_walls(model, data, target)
    directions = [np.asarray(wall['inward']) for wall in walls]
    directions.append(sum(directions)/np.linalg.norm(sum(directions)))
    offset = data.xpos[model.body('r_hand_palm_link').id]-objects[release, :3, 3]
    contract = details['task_contract']; frame = np.asarray(details['world_placement'])
    corners = np.array(list(itertools.product(*zip(contract['bin_lower'], contract['bin_upper']))))
    corners = corners @ frame[:3, :3].T + frame[:3, 3]
    extent = np.vstack((reference[:, :2], corners[:, :2]))
    bounds = np.c_[extent.min(0)-base_margin_m, extent.max(0)+base_margin_m].tolist()
    bounds.append([float(reference[:, 2].min()-np.pi), float(reference[:, 2].max()+np.pi)])
    proposals = []
    for pitch in pitches:
        for direction_id, direction in enumerate(directions):
            angle = [0., pitch, yaw_toward(offset, pitch, direction)]
            parameters = dict(base, base_offset_xyyaw=[0., 0., 0.], rotation_xyz_deg=angle,
                mobile_ik=dict(base_bounds_xyyaw=bounds, minimum_fixture_gap_m=minimum_fixture_gap_m,
                               orientation_transport=True, pose_tolerance_constraints=pose_tolerance_constraints))
            index = len(proposals)
            cost = float(np.linalg.norm(target-objects[release, :2, 3])
                         + .1*Rotation.from_euler('xyz', angle, degrees=True).magnitude())
            proposals.append(dict(id=index, rank=[cost, index], parameters=parameters,
                                  wall_direction_id=direction_id, wall_inward=direction.tolist()))
    report = dict(nearest_task_walls=walls, placement_xy_world=target.tolist(),
                  release_frame=release, release_palm_minus_object_center_world=offset.tolist(),
                  base_bounds_xyyaw=bounds, base_margin_m=base_margin_m,
                  pitch_angles_deg=pitches.tolist(),
                  scope='Geometry-derived proposals only. Full inserted/interpolated IK and fixture checks required before physics; no object state assignment or source-array modification.')
    return sorted(proposals, key=lambda item: item['rank']), report
