"""Screen robot placement targets using the complete compiled hand geometry.

These are geometric proposals, never full robot admissions. All distance queries
use private, non-integrated MjData with unchanged object qpos. Only the hand's
derived geom transforms are placed at a requested Cartesian pose.
"""
from copy import deepcopy
import hashlib
import itertools
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .feasible_maniskill import intent_parameters
from .geometry_clearance import RobotFixtureClearance, _subtree
from .physics import initialize
from .store import json_write, sha256


def _pose(value):
    value = np.asarray(value, float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0., 0., 0., 1.], atol=1e-10, rtol=0)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-8, rtol=0)
            or np.linalg.det(value[:3, :3]) < .99999999):
        raise ValueError('A finite proper rigid Cartesian pose is required')
    return value


class CompiledHandFixtureQuery:
    """Exact MuJoCo geometry-distance queries at hypothetical hand poses.

    The geometry is MuJoCo's compiled collision representation, including mesh
    asset/geom rotations and finger articulations at each declared aperture.
    No model field, source array or caller-owned simulation data is modified.
    """
    def __init__(self, prepared, robot, *, minimum_m=.003,
                 hand_body='r_hand_palm_link', tcp_site='r_arm_tip_tcp'):
        self.model, _, self.manifest = prepared[:3]
        self.robot = robot
        self.data = mujoco.MjData(self.model)
        self.initial_qpos = self.data.qpos.copy()
        self.minimum = float(minimum_m)
        active = self.manifest['objects'][prepared[5]['object_id']]['body']
        full = RobotFixtureClearance(self.model, active_object_bodies=(active,), minimum_m=minimum_m)
        bodies = _subtree(self.model, hand_body)
        robot_bodies = _subtree(self.model, 'base_link')
        self.object_qpos_indices = []
        for joint in range(self.model.njnt):
            if int(self.model.jnt_bodyid[joint]) not in robot_bodies:
                address = int(self.model.jnt_qposadr[joint])
                kind = int(self.model.jnt_type[joint])
                width = 7 if kind == int(mujoco.mjtJoint.mjJNT_FREE) else 4 if kind == int(mujoco.mjtJoint.mjJNT_BALL) else 1
                self.object_qpos_indices.extend(range(address, address+width))
        self.geoms = np.array([g for g in range(self.model.ngeom)
                               if int(self.model.geom_bodyid[g]) in bodies], dtype=int)
        self.pairs = np.array([(g, h) for g, h in full.pairs if g in self.geoms], dtype=int)
        if not len(self.pairs):
            raise ValueError('No compiled physical hand/fixture pairs remain')
        self.site = self.model.site(tcp_site).id
        reference = np.asarray(prepared[8])[0]
        self.q = robot.pack(reference[3:], reference[:3])
        self.templates = {}
        self.radii = self.model.geom_rbound[self.pairs].sum(axis=1)
        unbounded = [int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD),
                     int(mujoco.mjtGeom.mjGEOM_SDF)]
        self.bounded = ~np.isin(self.model.geom_type[self.pairs], unbounded).any(axis=1)
        self.metadata = dict(hand_body=hand_body, tcp_site=tcp_site,
            hand_geoms=[self.model.geom(g).name for g in self.geoms],
            physical_pair_count=len(self.pairs), minimum_m=self.minimum,
            active_object_bodies=[active], fixture_state_assumption='Unchanged compiled scene reset',
            query_state='Private non-integrated MjData; hand geom transforms only; object qpos unchanged')
        signature = hashlib.sha256()
        fields = ['body_parentid', 'body_pos', 'body_quat', 'geom_type', 'geom_size', 'geom_bodyid',
                  'geom_pos', 'geom_quat', 'geom_contype', 'geom_conaffinity', 'geom_dataid',
                  'mesh_vert', 'mesh_face', 'mesh_vertadr', 'mesh_vertnum', 'mesh_faceadr', 'mesh_facenum',
                  'jnt_type', 'jnt_bodyid', 'jnt_axis', 'jnt_pos', 'qpos0', 'pair_geom1', 'pair_geom2',
                  'exclude_signature', 'site_pos', 'site_quat', 'site_bodyid']
        for field in fields:
            value = np.asarray(getattr(self.model, field))
            signature.update((field+str(value.dtype)+str(value.shape)).encode())
            signature.update(value.tobytes())
        self.metadata['compiled_geometry_sha256'] = signature.hexdigest()
        self.metadata['compiled_geometry_signature_fields'] = fields

    def _template(self, aperture):
        aperture = float(aperture)
        if not np.isfinite(aperture) or not -.06 <= aperture <= 2.:
            raise ValueError('A declared Reachy aperture in [-.06, 2] is required')
        if aperture not in self.templates:
            initialize(self.model, self.data, self.robot, self.q, self.manifest['mimics'],
                       {'l_hand_finger': 2., 'r_hand_finger': aperture})
            if (self.data.time != 0. or not np.array_equal(self.data.qpos[self.object_qpos_indices],
                                                        self.initial_qpos[self.object_qpos_indices])):
                raise ValueError('Isolated robot initialization modified an object or clock')
            site = np.eye(4)
            site[:3, :3] = self.data.site_xmat[self.site].reshape(3, 3)
            site[:3, 3] = self.data.site_xpos[self.site]
            xyz = (self.data.geom_xpos[self.geoms]-site[:3, 3]) @ site[:3, :3]
            rotations = np.einsum('ij,njk->nik', site[:3, :3].T,
                                 self.data.geom_xmat[self.geoms].reshape(-1, 3, 3))
            self.templates[aperture] = xyz.copy(), rotations.copy()
        return self.templates[aperture]

    def inspect(self, goal, apertures):
        goal = _pose(goal)
        apertures = np.asarray(apertures, float)
        if apertures.ndim != 1 or not len(apertures):
            raise ValueError('A nonempty finite aperture envelope is required')
        cap = self.minimum + 1e-6
        nearest, pair, selected_aperture = cap, None, None
        queries = 0
        for aperture in apertures:
            xyz, rotations = self._template(aperture)
            self.data.geom_xpos[self.geoms] = xyz @ goal[:3, :3].T + goal[:3, 3]
            self.data.geom_xmat[self.geoms] = np.einsum('ij,njk->nik', goal[:3, :3], rotations).reshape(-1, 9)
            centers = self.data.geom_xpos[self.pairs]
            bound = np.linalg.norm(centers[:, 0]-centers[:, 1], axis=1)-self.radii
            selected = ~self.bounded | (bound <= cap)
            for g, h in self.pairs[selected]:
                distance = float(mujoco.mj_geomDistance(self.model, self.data, g, h, cap, None))
                if not np.isfinite(distance):
                    raise ValueError('Nonfinite compiled hand/fixture distance')
                queries += 1
                if distance < nearest:
                    nearest, pair, selected_aperture = distance, [self.model.geom(g).name, self.model.geom(h).name], float(aperture)
        return dict(rigid_hand_screen_passed=nearest >= self.minimum,
                    minimum_signed_distance_or_lower_bound_m=nearest,
                    distance_is_lower_bound=pair is None, closest_geom_pair=pair,
                    worst_aperture_rad=selected_aperture, exact_distance_queries=queries)


def _release(prepared):
    times, _, commands, hands, objects = map(np.asarray, prepared[7:12])
    closed = np.flatnonzero(commands < 0)
    releases = np.flatnonzero((np.arange(len(commands)) > closed[0]) & (commands >= 1.9999)) if len(closed) else []
    if not len(releases) or len(times) != len(hands) or len(times) != len(objects):
        raise ValueError('Aligned complete source arrays and a recorded release are required')
    return int(releases[0])


def waypoints(prepared, parameters):
    """Match supported_placement's exact Cartesian proposal, without IK."""
    release = _release(prepared)
    anchor = _pose(prepared[10][release]).copy()
    objects = np.asarray(prepared[11])
    final_position = objects[-1, :3, 3].copy()
    xy = np.asarray(parameters.get('placement_xy_world', final_position[:2]), float)
    angles = np.asarray(parameters.get('rotation_xyz_deg', [0., 0., 0.]), float)
    descent = float(parameters.get('extra_descent_m', .008))
    if (xy.shape != (2,) or angles.shape != (3,) or not np.isfinite(xy).all()
            or not np.isfinite(angles).all() or np.any(abs(angles) > [90., 90., 180.])
            or not 0 <= descent <= .02):
        raise ValueError('Finite bounded placement parameters are required')
    final_position[:2] = xy
    contract = prepared[5]['task_contract']
    frame = _pose(prepared[5]['world_placement'])
    local = frame[:3, :3].T @ (final_position-frame[:3, 3])
    if not (np.all(local > contract['bin_lower']) and np.all(local < contract['bin_upper'])):
        raise ValueError('Placement must remain inside the original task region')
    delta = final_position-objects[release, :3, 3]
    if delta[2] > .02 or abs(delta[2]) > .3 or np.linalg.norm(delta[:2]) > .3:
        raise ValueError('Placement is outside the original bounded local policy')
    rotation = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    bottom = anchor.copy()
    bottom[:3, :3] = rotation @ anchor[:3, :3]
    bottom[:3, 3] = final_position + rotation @ (anchor[:3, 3]-objects[release, :3, 3])
    bottom[2, 3] -= descent
    traverse = bottom.copy()
    traverse[2, 3] = max(anchor[2, 3], bottom[2, 3])
    return np.stack([anchor, traverse, bottom])


def screen(prepared, query, parameters, *, segment_samples=21):
    """Screen sampled desired hand poses only; complete robot checks still required."""
    if type(segment_samples) is not int or not 2 <= segment_samples <= 101:
        raise ValueError('A bounded explicit Cartesian screening density is required')
    goals = waypoints(prepared, parameters)
    contact, closed = intent_parameters(prepared[5])
    opening = float(parameters.get('release_aperture_rad', 2.))
    if not 1.2 <= opening <= 2.:
        raise ValueError('A bounded release aperture is required')
    lower_apertures = np.linspace(closed, max(opening, contact), 5)
    bottom = query.inspect(goals[-1], lower_apertures)
    checks = [dict(phase='bottom', fraction=1., **bottom)]
    if bottom['rigid_hand_screen_passed']:
        for name, first, last, upper in [('traverse', goals[0], goals[1], 2.),
                                         ('lower', goals[1], goals[2], max(opening, contact))]:
            orientation = Slerp([0., 1.], Rotation.from_matrix([first[:3, :3], last[:3, :3]]))
            for fraction in np.linspace(0., 1., segment_samples):
                goal = first.copy()
                goal[:3, 3] = first[:3, 3]*(1-fraction)+last[:3, 3]*fraction
                goal[:3, :3] = orientation(fraction).as_matrix()
                found = query.inspect(goal, np.linspace(closed, upper, 5))
                checks.append(dict(phase=name, fraction=float(fraction), **found))
                if not found['rigid_hand_screen_passed']:
                    break
            if not checks[-1]['rigid_hand_screen_passed']:
                break
    worst = min(checks, key=lambda r: r['minimum_signed_distance_or_lower_bound_m'])
    return dict(rigid_hand_screen_passed=all(r['rigid_hand_screen_passed'] for r in checks),
                full_robot_admitted=False, physics_validated=False,
                requires_full_mobile_ik_and_fixture_admission=True,
                segment_samples=segment_samples, sampled_poses=len(checks), worst=worst)


def propose(prepared, robot, candidates, output, *, minimum_m=.003, xy_grid_count=5,
            interior_margin_m=.04, segment_samples=21, max_proposals=9):
    """Screen a bounded cell-interior XY bank for declared orientation candidates.

    Original XY and the logical cell center are always retained. Rotation is
    ranked before XY displacement. No orientation is introduced by this helper.
    """
    if (type(xy_grid_count) is not int or not 2 <= xy_grid_count <= 7
            or not np.isfinite(interior_margin_m) or not 0 < interior_margin_m <= .1
            or type(max_proposals) is not int or not 1 <= max_proposals <= 27):
        raise ValueError('A small declared geometric proposal budget is required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    originals = output/'original-inputs.npz'
    np.savez_compressed(originals, **dict(zip(['time_s', 'reference', 'gripper_intent', 'hand_targets', 'object_targets'], prepared[7:12])))
    candidates = deepcopy(list(candidates))
    json_write(output/'original-candidates.json', candidates)
    query = CompiledHandFixtureQuery(prepared, robot, minimum_m=minimum_m)
    contract = prepared[5]['task_contract']
    lower, upper = np.asarray(contract['bin_lower']), np.asarray(contract['bin_upper'])
    frame = _pose(prepared[5]['world_placement'])
    if np.any(upper[:2]-lower[:2] <= 2*interior_margin_m):
        raise ValueError('Declared interior margin empties the original task region')
    center = (lower+upper)/2
    points = [frame[:3, :3] @ center + frame[:3, 3]]
    for x, y in itertools.product(*(np.linspace(lower[i]+interior_margin_m, upper[i]-interior_margin_m, xy_grid_count) for i in (0, 1))):
        points.append(frame[:3, :3] @ [x, y, center[2]] + frame[:3, 3])
    events, proposals = [], []
    for candidate in candidates:
        original_xy = np.asarray(candidate['parameters']['placement_xy_world'])
        seen = set()
        for target in [original_xy]+[point[:2] for point in points]:
            key = tuple(np.round(target, 12))
            if key in seen:
                continue
            seen.add(key)
            params = deepcopy(candidate['parameters'])
            params['placement_xy_world'] = np.asarray(target).tolist()
            event = dict(id=len(events), parent_candidate_id=candidate['id'], parameters=params)
            try:
                event['screen'] = screen(prepared, query, params, segment_samples=segment_samples)
            except ValueError as error:
                # Bounded geometric scope errors are retained; unexpected errors propagate.
                event['screen'] = dict(rigid_hand_screen_passed=False, reason=str(error))
            events.append(event)
            if event['screen']['rigid_hand_screen_passed']:
                angle = float(Rotation.from_euler('xyz', params['rotation_xyz_deg'], degrees=True).magnitude())
                event['rank'] = [angle, float(np.linalg.norm(target-original_xy)), event['id']]
                proposals.append(event)
    proposals.sort(key=lambda r: r['rank'])
    selected = proposals[:max_proposals]
    report = dict(schema='compiled-hand-placement-proposals-v1', query=query.metadata,
        source_metadata=deepcopy(prepared[4]),
        frozen_parent_binding=deepcopy(prepared[5].get('frozen_plan_reuse')),
        scene_xml_sha256=hashlib.sha256(prepared[1].encode()).hexdigest() if isinstance(prepared[1], str) else None,
        scene_asset_hashes=deepcopy(prepared[2].get('scene_asset_hashes')),
        original_inputs_sha256=sha256(originals), original_candidates_sha256=sha256(output/'original-candidates.json'),
        module_sha256=sha256(output/Path(__file__).name), task_center_world=(frame[:3, :3] @ center+frame[:3, 3]).tolist(),
        xy_grid_count=xy_grid_count, interior_margin_m=interior_margin_m,
        original_candidate_count=len(candidates), screened_count=len(events), passed_count=len(proposals),
        selected_ids=[r['id'] for r in selected], candidates=events,
        physics_validated=False, full_robot_admitted=False,
        scope='Sampled rigid hand targets against all compiled physical fixtures. Original source arrays and object/model properties unchanged. A screen pass requires fresh complete robot IK, interpolated aperture/fixture checks and independent dynamics validation.')
    json_write(output/'result.json', report)
    return selected, report
