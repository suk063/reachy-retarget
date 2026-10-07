"""Read-only positive robot/fixture separation queries for kinematic admission.

Unlike a maximum penetration gate, a positive gap detects paths whose servo can
stall against a fixture at almost zero penetration. This supplements, rather
than changes, the independent physical gates. The caller must synchronize FK.
"""
import mujoco
import numpy as np
from copy import deepcopy
from pathlib import Path

from .store import json_write, sha256


def _subtree(model, root):
    body = model.body(root).id if isinstance(root, str) else int(root)
    if body <= 0 or body >= model.nbody:
        raise ValueError('A non-world body root is required')
    ids = {body}
    for child in range(body + 1, model.nbody):
        if int(model.body_parentid[child]) in ids:
            ids.add(child)
    return ids


class RobotFixtureClearance:
    """Bind physical collision pairs once, then inspect already-computed states.

    ``support_pairs`` names exact permitted support geom pairs, e.g. a drive
    wheel and floor. There is no blanket floor or hand exemption. Entire active
    object subtrees are excluded because their intended grasp is assessed by
    separate contact checks. Nonactive boxes remain fixtures.
    """
    def __init__(self, model, *, robot_body_root='base_link', active_object_bodies=(),
                 support_pairs=(), minimum_m=.003):
        if not np.isfinite(minimum_m) or not 0 < minimum_m <= .05:
            raise ValueError('A positive fixture clearance in (0, 50 mm] is required')
        self.model, self.minimum_m = model, float(minimum_m)
        robots = _subtree(model, robot_body_root)
        active = set()
        for body in active_object_bodies:
            active |= _subtree(model, body)
        if active & robots:
            raise ValueError('An active object cannot be part of the robot')
        support_pairs = tuple(tuple(pair) for pair in support_pairs)
        excluded = {tuple(sorted((model.geom(a).id, model.geom(b).id))) for a, b in support_pairs}
        paired = {tuple(sorted((int(a), int(b)))) for a, b in zip(model.pair_geom1, model.pair_geom2)}
        body_exclusions = set(map(int, model.exclude_signature))
        robot_geoms = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in robots]
        fixture_geoms = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) not in robots | active]
        candidates = []
        for a in robot_geoms:
            for b in fixture_geoms:
                key = tuple(sorted((a, b)))
                b1, b2 = sorted(map(int, model.geom_bodyid[[a, b]]))
                if key in excluded:
                    continue
                # Explicit pairs take precedence over dynamic-pair filtering.
                if key not in paired:
                    if not ((int(model.geom_contype[a]) & int(model.geom_conaffinity[b]))
                            or (int(model.geom_contype[b]) & int(model.geom_conaffinity[a]))):
                        continue
                    if ((b1 << 16) + b2 in body_exclusions
                            or model.body_weldid[b1] == model.body_weldid[b2]):
                        continue
                candidates.append((a, b))
        expected = {tuple(sorted((a, b))) for a in robot_geoms for b in fixture_geoms}
        if not excluded <= expected:
            raise ValueError('Support exemptions must be exact robot/fixture geom pairs')
        if not candidates:
            raise ValueError('No physical robot/fixture pairs remain to inspect')
        self.pairs = tuple(candidates)
        self._indices = np.asarray(candidates, dtype=int)
        self._radii = model.geom_rbound[self._indices].sum(axis=1)
        unbounded = (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD),
                     int(mujoco.mjtGeom.mjGEOM_SDF))
        self._bounded = ~np.isin(model.geom_type[self._indices], unbounded).any(axis=1)
        self.metadata = dict(minimum_m=self.minimum_m, collision_pair_count=len(candidates),
                             support_pairs=[list(pair) for pair in support_pairs],
                             active_object_body_ids=sorted(active), robot_body_ids=sorted(robots),
                             scope='All compiled physical robot/fixture collision pairs; selected active-object subtrees and only explicit support pairs excluded')

    def inspect(self, data):
        """Query signed distances without stepping, forwarding or changing data."""
        cap = self.minimum_m + 1e-6
        nearest, pair, segment = cap, None, None
        line = np.empty(6)
        centers = data.geom_xpos[self._indices]
        bound = np.linalg.norm(centers[:, 0]-centers[:, 1], axis=1)-self._radii
        selected = ~self._bounded | (bound <= cap)
        for a, b in self._indices[selected]:
            distance = float(mujoco.mj_geomDistance(self.model, data, a, b, cap, line))
            if not np.isfinite(distance):
                raise ValueError('Non-finite compiled-geometry distance')
            if distance < nearest:
                nearest, pair, segment = distance, [self.model.geom(a).name, self.model.geom(b).name], line.tolist()
        return dict(admitted=nearest >= self.minimum_m, minimum_required_m=self.minimum_m,
                    minimum_signed_distance_or_lower_bound_m=nearest,
                    distance_is_lower_bound=pair is None, closest_geom_pair=pair,
                    closest_segment_world=segment, inspected_pairs=len(self.pairs),
                    exact_distance_queries=int(selected.sum()),
                    certified_far_pairs=int((~selected).sum()))


def reachy_wheel_floor_pairs(model, floor_geom='floor'):
    """Return three explicit canonical wheel/floor support exemptions."""
    floor = model.geom(floor_geom).name
    return [(model.geom(f'drivewhl{i}_link_collision_0').name, floor) for i in (1, 2, 3)]


def closed_aperture_interval(details, explicit=None):
    """Bind static aperture checks to actual numeric targets, not contact intent."""
    candidate = details.get('effective_candidate', {})
    contact = float(details['pad_alignment']['inferred_contact_angle_rad'])
    if explicit is None:
        if candidate.get('force_feedback'):
            raise ValueError('Force-feedback clearance requires an explicit static closed_aperture_rad envelope')
        target = candidate.get('gripper_closed_target')
        if target is None:
            raise ValueError('A recorded numeric gripper_closed_target is required before clearance admission')
        interval = np.sort([float(target), contact])
    else:
        interval = np.asarray(explicit, float)
    if (interval.shape != (2,) or not np.isfinite(interval).all()
            or not -.06 <= interval[0] <= contact <= interval[1] <= 2.):
        raise ValueError('A valid declared aperture interval containing calibrated contact is required')
    if not candidate.get('force_feedback') and candidate.get('gripper_closed_target') is not None:
        if not interval[0] <= candidate['gripper_closed_target'] <= interval[1]:
            raise ValueError('The aperture envelope excludes the actual numeric command target')
    return interval


def prepare(prepared, robot, output, *, minimum_m=.003, support_pairs=(), closed_aperture_rad=None):
    """Save failures as well as a complete positive-clearance admission."""
    try:
        return _prepare(prepared, robot, output, minimum_m=minimum_m,
                        support_pairs=support_pairs, closed_aperture_rad=closed_aperture_rad)
    except ValueError as error:
        path = Path(output)/'result.json'
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            json_write(path, dict(admitted=False, physics_validated=False, reason=str(error)))
        raise


def _prepare(prepared, robot, output, *, minimum_m, support_pairs, closed_aperture_rad):
    """Inspect the complete current reference with explicit fixture assumptions.

    Check sampled aperture envelopes in each phase and across transitions.
    Fixtures default to reset, or an explicitly bound source forecast on
    isolated planning data. Neither proves actual future physical clearance.
    """
    from .physics import initialize
    from .object_scene import initialize_fixtures
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model, _, manifest = prepared[:3]
    active = manifest['objects'][prepared[5]['object_id']]['body']
    checker = RobotFixtureClearance(model, active_object_bodies=(active,),
                                    support_pairs=support_pairs, minimum_m=minimum_m)
    data = mujoco.MjData(model)
    initialize_fixtures(model, data, manifest)
    times, reference, commands = prepared[7], prepared[8], prepared[9]
    angle = float(prepared[5]['pad_alignment']['inferred_contact_angle_rad'])
    closed_interval = closed_aperture_interval(prepared[5], closed_aperture_rad)
    if (not np.isfinite(angle) or not -.06 <= angle <= 2.
            or len(times) != len(reference) or len(times) != len(commands)):
        raise ValueError('Aligned references and a valid calibrated aperture are required')
    gaps, apertures, query_counts = [], [], []
    worst = None
    failure = None
    fixture_query = None
    try:
        from .planning_fixtures import make_query
        fixture_query = make_query(prepared)
        if fixture_query is not None:
            data = fixture_query.data
        for index, row in enumerate(reference):
            command = float(commands[index]); closed = command < 0
            low, high = closed_interval if closed else (command, command)
            if index == 0 or (closed != (commands[index-1] < 0)):
                previous = 2. if index == 0 else float(commands[index-1]) if commands[index-1] >= 0 else closed_interval[0]
                low, high = min(low, previous), max(high, previous)
            frame = None
            q = robot.pack(row[3:], row[:3])
            count = 0
            for grip in np.unique(np.linspace(low, high, 5)):
                initialize(model, data, robot, q, manifest['mimics'],
                           {'l_hand_finger': 2., 'r_hand_finger': float(grip)})
                if fixture_query is not None:
                    fixture_query.synchronize(index)
                found = checker.inspect(data)
                count += found['exact_distance_queries']
                if frame is None or found['minimum_signed_distance_or_lower_bound_m'] < frame['minimum_signed_distance_or_lower_bound_m']:
                    frame = dict(found, gripper_angle_rad=float(grip), frame=index, time_s=float(times[index]))
            if worst is None or frame['minimum_signed_distance_or_lower_bound_m'] < worst['minimum_signed_distance_or_lower_bound_m']:
                worst = frame
            gaps.append(frame['minimum_signed_distance_or_lower_bound_m'])
            apertures.append([low, high]); query_counts.append(count)
    except Exception as error:
        failure = type(error).__name__ + ': ' + str(error)
    complete = len(gaps) == len(times) and failure is None
    checks = dict(complete=complete, positive_fixture_gap=bool(complete and min(gaps) >= minimum_m))
    artifact = output/'fixture-clearance.npz'
    np.savez_compressed(artifact, original_time_s=times, reference=reference,
                        original_object_goals=prepared[11], original_gripper_intent=commands,
                        signed_distance_or_lower_bound_m=gaps, aperture_envelopes_rad=apertures,
                        exact_distance_query_counts=query_counts)
    report = dict(checker.metadata, admitted=all(checks.values()), admission_checks=checks,
                  completed_frames=len(gaps), total_frames=len(times), exception=failure,
                  closed_aperture_rad=closed_interval.tolist(), inactive_left_gripper_rad=2.,
                  worst=worst, artifact_sha256=sha256(artifact), physics_validated=False,
                  fixture_pose_scope='Original compiled reset with source passive fixture initialization; no object trajectory or physics stepping',
                  aperture_scope='Five-point closed aperture envelopes and opening/closing transitions; continuous swept-volume clearance is not asserted')
    if fixture_query is not None:
        report['planning_fixture_forecast'] = fixture_query.save(output)
        report['fixture_pose_scope'] = fixture_query.metadata['scope']
    json_write(output/'result.json', report)
    if not report['admitted']:
        raise ValueError('Positive fixture clearance rejected; inspect ' + str(output/'result.json'))
    result = list(prepared)
    result[5] = deepcopy(prepared[5])
    result[5]['robot_fixture_clearance'] = report
    return tuple(result)
