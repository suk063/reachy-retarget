"""Mobile insertion retains exact original rows and the measured support guard."""
import json
import numpy as np
import pytest

from reachy_retarget import mobile_placement_ik as mobile
from reachy_retarget import mobile_fixture_ik
from reachy_retarget import supported_placement as placement
from test_supported_placement import prepared, Robot, exercise


def fake_solver(monkeypatch, gap=.004, base_distance=.01):
    class Solver:
        def __init__(self, prepared, robot, **kwargs):
            self.robot, self.records, self.minimum = robot, [], .003
            self.metadata = {'minimum_fixture_gap_m': .003, 'scope': 'test fixture'}

        def solve(self, goal, q, apertures, *, phase, weight):
            assert np.min(apertures) == .998 and np.max(apertures) >= 1.
            self.records.append({'phase': phase, 'weight': weight, 'ik_errors': [0., 0.],
                                 'fixture': {'admitted': True}})
            q = q.copy(); q[0] = base_distance * weight if phase == 'traverse' else base_distance
            targets = self.robot.fk(q); targets[1] = goal
            return self.robot.ik(targets, q)[0]

        def inspect(self, q, apertures):
            return {'minimum_signed_distance_or_lower_bound_m': gap}

        def reference_base(self, q):
            return self.robot.base(q)

    monkeypatch.setattr(mobile, 'Solver', Solver)


def source_and_robot(monkeypatch):
    source = prepared(monkeypatch)
    source[5]['pad_alignment'] = {'inferred_contact_angle_rad': 1.}
    source[5]['effective_candidate']['gripper_closed_target'] = .998
    robot = Robot(); robot.base = lambda q: q[:3].copy()
    return source, robot


def test_mobile_insert_keeps_original_rows_and_measured_opening(monkeypatch, tmp_path):
    fake_solver(monkeypatch)
    source, robot = source_and_robot(monkeypatch)
    result = placement.prepare(source, robot, tmp_path/'insert', mobile_ik={})
    policy = result[5]['robot_supported_placement']
    with np.load(policy['source_clock_artifact']) as saved:
        indices = saved['original_row_indices']
        np.testing.assert_array_equal(saved['source_reference_time_s'][indices], source[7])
    for i in (8, 9, 10, 11):
        np.testing.assert_array_equal(result[i][indices], source[i])
    assert np.max(result[8][:, 0]) > 0 and policy['admission_checks']['mobile_inserted_reference_speed']
    assert policy['admission_checks']['mobile_boundary_reference_speed']
    refused, _ = exercise(result, lambda step, control: 0.)
    assert refused.failed and not refused.opened
    opened, _ = exercise(result, lambda step, control: 1.)
    assert opened.opened and opened.opening_verified


def test_interval_timing_keeps_complete_source_and_physical_speed_checks(monkeypatch,tmp_path):
    fake_solver(monkeypatch)
    source,robot=source_and_robot(monkeypatch)
    result=placement.prepare(source,robot,tmp_path/'local-clock',mobile_ik={},interval_knot_timing=True)
    policy=result[5]['robot_supported_placement']
    assert policy['interval_knot_timing'] and policy['interval_clock_artifacts']
    assert policy['admission_checks']['mobile_inserted_reference_speed']
    assert policy['admission_checks']['mobile_boundary_reference_speed']
    with np.load(policy['source_clock_artifact']) as saved:
        indices=saved['original_row_indices']
        for i in (8,9,10,11):np.testing.assert_array_equal(result[i][indices],source[i])


def test_inserted_base_speed_cap_changes_clock_without_moving_source_or_ik_knots(monkeypatch,tmp_path):
    fake_solver(monkeypatch,base_distance=.6)
    source,robot=source_and_robot(monkeypatch)
    original=placement.prepare(source,robot,tmp_path/'ordinary-base',mobile_ik={},interval_knot_timing=True)
    slower=placement.prepare(source,robot,tmp_path/'slower-base',mobile_ik={},interval_knot_timing=True,
                              mobile_base_axis_speed_m_s=.1)
    old=original[5]['robot_supported_placement'];new=slower[5]['robot_supported_placement']
    assert new['inserted_duration_s']>old['inserted_duration_s']
    assert new['mobile_base_axis_speed_m_s']==.1
    assert max(new['mobile_ik']['inserted_speed_peak'][:2])<=.1+1e-7
    np.testing.assert_array_equal(np.load(tmp_path/'ordinary-base/ik-knots.npz')['q'],
                                  np.load(tmp_path/'slower-base/ik-knots.npz')['q'])
    with np.load(new['source_clock_artifact'])as saved:
        rows=saved['original_row_indices']
        for field in (8,9,10,11):np.testing.assert_array_equal(slower[field][rows],source[field])


def test_interpolated_fixture_failure_cannot_be_hidden_by_successful_knots(monkeypatch, tmp_path):
    fake_solver(monkeypatch, gap=.001)
    source, robot = source_and_robot(monkeypatch)
    with pytest.raises(ValueError, match='admission rejected'):
        placement.prepare(source, robot, tmp_path/'blocked', mobile_ik={})
    report = json.loads((tmp_path/'blocked/result.json').read_text())
    assert not report['admission_checks']['mobile_positive_fixture_clearance']
    assert (tmp_path/'blocked/mobile-fixture-gaps.npy').exists()
    assert (tmp_path/'blocked/mobile-ik-knots.json').exists()


def test_prescribed_base_and_mobile_base_cannot_compete(monkeypatch, tmp_path):
    source, robot = source_and_robot(monkeypatch)
    with pytest.raises(ValueError, match='no prescribed base offset'):
        placement.prepare(source, robot, tmp_path/'invalid', mobile_ik={}, base_offset_xyyaw=(.1, 0, 0))


def test_real_pose_solver_leaves_free_object_and_clock_untouched(monkeypatch):
    from test_mobile_fixture_ik import prepared as actual_fixture
    robot, source = actual_fixture(monkeypatch)
    monkeypatch.setattr(mobile, 'initialize', mobile_fixture_ik.initialize)
    solver = mobile.Solver(source, robot, base_bounds_xyyaw=[[-.2, .2], [-.2, .2], [-.2, .2]],
                           support_pairs=[])
    before = source[0].qpos0.copy()
    q = solver.solve(source[10][1], source[6], np.array([.998, 1.]), phase='lower', weight=.5)
    np.testing.assert_allclose(robot.fk(q)[1], source[10][1], atol=1e-5)
    np.testing.assert_array_equal(source[0].qpos0, before)


def test_planar_speed_disc_keeps_base_bounded_while_arm_reaches_target(monkeypatch):
    from test_mobile_fixture_ik import prepared as actual_fixture
    robot,source=actual_fixture(monkeypatch)
    monkeypatch.setattr(mobile,'initialize',mobile_fixture_ik.initialize)
    solver=mobile.Solver(source,robot,base_bounds_xyyaw=[[-.2,.2],[-.2,.2],[-.2,.2]],
                         support_pairs=[],pose_tolerance_constraints=True)
    before=source[0].qpos0.copy()
    q=solver.solve(source[10][3],source[6],[.998,1.],phase='lower',weight=.5,
                   planar_constraints=[[0.,0.,.001]])
    assert np.linalg.norm(robot.base(q)[:2])<=.001+1e-9
    assert np.linalg.norm(robot.fk(q)[1][:3,3]-source[10][3,:3,3])<=.002
    assert q[11]>.025
    np.testing.assert_array_equal(source[0].qpos0,before)
    np.testing.assert_array_equal(solver.data.qpos[1:], before[1:])
    assert solver.data.time == 0. and solver.inspect(q, [.998, 1.])['admitted']


def test_failed_knot_stops_before_invalid_interpolation_and_saves_numeric_state(monkeypatch, tmp_path):
    fake_solver(monkeypatch)
    previous = mobile.Solver.solve
    def failing(self, goal, q, apertures, **kwargs):
        q = previous(self, goal, q, apertures, **kwargs)
        self.records[-1].update(q=q.tolist(), ik_errors=[.005, .01])
        return q
    monkeypatch.setattr(mobile.Solver, 'solve', failing)
    source, robot = source_and_robot(monkeypatch)
    with pytest.raises(ValueError, match='IK rejected'):
        placement.prepare(source, robot, tmp_path/'failed-knot', mobile_ik={})
    report = json.loads((tmp_path/'failed-knot/result.json').read_text())
    assert not report['complete'] and not report['admitted']
    assert report['rejection_kind'] == 'mobile_knot_admission'
    assert report['original_source_rows'] == len(source[7])
    assert report['completed_knot_count'] == 2
    with np.load(tmp_path/'failed-knot/ik-knots.npz') as saved:
        np.testing.assert_array_equal(saved['q'][-1], report['failed_knot']['q'])
    with np.load(tmp_path/'failed-knot/original-and-proposal.npz') as saved:
        np.testing.assert_array_equal(saved['original_reference'], source[8])
        np.testing.assert_array_equal(saved['original_object_targets'], source[11])
    assert not (tmp_path/'failed-knot/placement-admission.npz').exists()


def test_planar_transport_exactly_follows_pure_world_yaw_and_xy_without_changing_joints():
    from scipy.spatial.transform import Rotation
    class PlanarRobot:
        def base(self, q):
            return np.r_[q[:2], np.arctan2(q[3], q[2])]
        def fk(self, q):
            yaw = self.base(q)[2]
            rotation = Rotation.from_euler('z', yaw).as_matrix()
            result = np.tile(np.eye(4), (2, 1, 1))
            result[1, :3, :3] = rotation @ Rotation.from_euler('xyz', q[4:7]).as_matrix()
            result[1, :3, 3] = np.r_[q[:2], 0.] + rotation @ np.array([.4, -.2, .9])
            return result
    robot = PlanarRobot()
    q = np.r_[.2, -.1, np.cos(.3), np.sin(.3), .1, -.2, .4]
    original = q.copy(); goal = robot.fk(q)[1]
    goal[:3, :3] = Rotation.from_euler('z', 2.3).as_matrix() @ goal[:3, :3]
    goal[:2, 3] += [.2, -.3]
    transported, proof = mobile.transport_planar_seed(robot, q, goal)
    np.testing.assert_allclose(robot.fk(transported)[1], goal, atol=1e-14)
    np.testing.assert_array_equal(q, original)
    np.testing.assert_array_equal(transported[4:], q[4:])
    assert proof['delta_yaw_rad'] == pytest.approx(2.3)


def test_hard_pose_box_finds_feasible_tradeoff_rejected_by_soft_objective(monkeypatch):
    from scipy.spatial.transform import Rotation
    from test_mobile_fixture_ik import prepared as actual_fixture
    robot, source = actual_fixture(monkeypatch)
    original_fk = robot.fk
    def coupled_fk(q):
        poses = original_fk(q)
        poses[1, :3, :3] = Rotation.from_euler('z', q[11]).as_matrix()
        return poses
    robot.fk = coupled_fk
    monkeypatch.setattr(mobile, 'initialize', mobile_fixture_ik.initialize)
    goal = source[10][0].copy()
    goal[:3, :3] = Rotation.from_euler('z', .0215).as_matrix()
    options = dict(base_bounds_xyyaw=[[-1e-10, 1e-10]] * 3, support_pairs=[])
    soft = mobile.Solver(source, robot, **options)
    soft.solve(goal, source[6], [.998, 1.], phase='traverse', weight=.5)
    assert soft.records[-1]['ik_errors'][0] > .002
    hard = mobile.Solver(source, robot, **options, pose_tolerance_constraints=True)
    hard.solve(goal, source[6], [.998, 1.], phase='traverse', weight=.5)
    assert np.all(np.asarray(hard.records[-1]['ik_errors']) <= [.002, .02])
    assert hard.records[-1]['fixture']['admitted']


def test_minimum_motion_keeps_nearby_feasible_state_instead_of_chasing_micrometers(monkeypatch):
    from test_mobile_fixture_ik import prepared as actual_fixture
    robot, source = actual_fixture(monkeypatch)
    monkeypatch.setattr(mobile, 'initialize', mobile_fixture_ik.initialize)
    goal = source[10][0].copy(); goal[0,3] += .0015
    options = dict(base_bounds_xyyaw=[[-1e-10,1e-10]]*3,support_pairs=[],pose_tolerance_constraints=True)
    ordinary = mobile.Solver(source,robot,**options)
    moved = ordinary.solve(goal,source[6],[.998,1.],phase='lower',weight=.5)
    quiet = mobile.Solver(source,robot,**options,minimum_motion=True)
    held = quiet.solve(goal,source[6],[.998,1.],phase='lower',weight=.5)
    assert np.linalg.norm(moved-source[6]) > .001
    assert np.linalg.norm(held-source[6]) < .0002
    assert np.all(np.asarray(quiet.records[-1]['ik_errors']) <= [.002,.02])
    assert quiet.records[-1]['fixture']['admitted']
    with pytest.raises(ValueError,match='requires hard pose'):
        mobile.Solver(source,robot,base_bounds_xyyaw=[[-.1,.1]]*3,support_pairs=[],minimum_motion=True)


def test_midpoint_pose_reserve_is_stricter_and_does_not_leak_to_next_solve(monkeypatch):
    from test_mobile_fixture_ik import prepared as actual_fixture
    robot, source = actual_fixture(monkeypatch)
    monkeypatch.setattr(mobile, 'initialize', mobile_fixture_ik.initialize)
    solver = mobile.Solver(source, robot, base_bounds_xyyaw=[[-1e-10,1e-10]]*3,
                          support_pairs=[], pose_tolerance_constraints=True, minimum_motion=True)
    goal = source[10][0].copy(); goal[0,3] += .00215
    original = source[6].copy()
    strict = solver.solve(goal, original, [.998,1.], phase='lower_midpoint', weight=.5,
                          position_interior_reserve_m=.0001)
    assert solver.records[-1]['ik_errors'][0] <= .0019
    assert solver.records[-1]['solve_position_tolerance_m'] == pytest.approx(.0019)
    assert np.linalg.norm(strict-original) > .00004
    solver.solve(goal, original, [.998,1.], phase='lower', weight=1.)
    assert .0019 < solver.records[-1]['ik_errors'][0] <= .002
    assert solver.records[-1]['position_interior_reserve_m'] == 0.
    assert solver.metadata['pose_tolerances_m_rad'] == [.002,.02]
    np.testing.assert_array_equal(source[6], original)
    with pytest.raises(ValueError, match='interior reserve'):
        solver.solve(goal, original, [.998,1.], phase='lower', weight=.5,
                     position_interior_reserve_m=.002)


def test_fixture_reserve_discovers_pairs_beyond_default_query_cap_and_does_not_leak(monkeypatch):
    from test_mobile_fixture_ik import prepared as actual_fixture
    robot, source = actual_fixture(monkeypatch, obstacle_x=.12305)
    monkeypatch.setattr(mobile, 'initialize', mobile_fixture_ik.initialize)
    solver = mobile.Solver(source, robot, base_bounds_xyyaw=[[-1e-10,1e-10]]*3,
                          support_pairs=[], pose_tolerance_constraints=True, minimum_motion=True)
    seed=source[6].copy(); goal=robot.fk(seed)[1]
    ordinary=solver.inspect(seed,[.998,1.])
    assert ordinary['distance_is_lower_bound'] and ordinary['admitted']
    strict=solver.solve(goal,seed,[.998,1.],phase='lower_midpoint',weight=.5,
                        fixture_interior_reserve_m=.0001)
    record=solver.records[-1]
    assert solver.pairs and record['fixture']['admitted']
    assert record['fixture']['minimum_required_m']==pytest.approx(.0031)
    assert strict[11] < -.00005
    solver.solve(goal,seed,[.998,1.],phase='lower',weight=1.)
    assert solver.records[-1]['solve_minimum_fixture_gap_m']==.003
    assert solver.records[-1]['fixture']['minimum_required_m']==.003
    assert solver.checker.minimum_m==.003
    assert solver.records[-1]['q'][11] > -.00001


def test_unexpected_solver_exception_is_not_a_geometric_rejection(monkeypatch, tmp_path):
    fake_solver(monkeypatch)
    def crash(*args, **kwargs):
        raise RuntimeError('unexpected backend failure')
    monkeypatch.setattr(mobile.Solver, 'solve', crash)
    source, robot = source_and_robot(monkeypatch)
    with pytest.raises(ValueError, match='IK rejected'):
        placement.prepare(source, robot, tmp_path/'unexpected', mobile_ik={})
    report = json.loads((tmp_path/'unexpected/result.json').read_text())
    assert 'unexpected backend failure' in report['exception']
    assert 'rejection_kind' not in report


def test_planar_solver_crosses_pi_without_bound_clipping_or_full_turn(monkeypatch):
    from scipy.spatial.transform import Rotation
    from test_mobile_fixture_ik import prepared as actual_fixture
    robot, source = actual_fixture(monkeypatch)
    monkeypatch.setattr(mobile, 'initialize', mobile_fixture_ik.initialize)
    # A planar orientation fixture with joint-independent yaw makes an
    # artificial wrap/clip impossible to hide through redundant wrist motion.
    def fk(q):
        poses = np.tile(np.eye(4), (2, 1, 1))
        poses[1, :3, :3] = Rotation.from_euler('z', robot.base(q)[2]).as_matrix()
        poses[1, :3, 3] = [q[0]+q[11], q[1], 1.]
        return poses
    robot.fk = fk
    solver = mobile.Solver(source, robot,
        base_bounds_xyyaw=[[-.2, .2], [-.2, .2], [2.8, 3.5]],
        initial_base_yaw=3.10, orientation_transport=True, support_pairs=[])
    q = robot.pack(source[8][0, 3:], [0., 0., 3.10])
    yaws = []
    for index, angle in enumerate([3.12, 3.15, 3.20, 3.25]):
        target = np.eye(4); target[2, 3] = 1.
        target[:3, :3] = Rotation.from_euler('z', angle).as_matrix()
        q = solver.solve(target, q, [.998, 1.], phase='traverse', weight=index/3)
        yaws.append(solver.reference_base(q)[2])
        assert solver.records[-1]['ik_errors'][1] < 1e-5
    np.testing.assert_allclose(yaws, [3.12, 3.15, 3.20, 3.25], atol=1e-5)
    assert np.max(np.abs(np.diff(yaws))) < .051
    assert mobile.continuous_yaw(-3.10, 3.15) > np.pi
    np.testing.assert_array_equal(solver.data.qpos[1:], source[0].qpos0[1:])
    assert solver.data.time == 0.
