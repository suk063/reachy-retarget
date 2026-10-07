import copy
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget import cylindrical_acquisition as acquisition


def calibration():
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('xyz', [.17, -.24, .53]).as_matrix()
    midpoint = np.array([.012, 0., -.04])
    patch = np.array([.002, -.001, .014])
    transform[:3, 3] = patch-transform[:3, :3]@midpoint
    details = dict(pad_alignment={'pad': {'points_tcp': [midpoint+[0, -.025, 0], midpoint+[0, .025, 0]],
                                         'midpoint_tcp': midpoint}},
                   cylindrical_grasp={'closing_axis_after_object': transform[:3, :3]@[0, 1, 0],
                       'longitudinal_axis_tcp': [0, 0, 1],
                       'longitudinal_axis_after_object': transform[:3, :3]@[0, 0, 1],
                       'desired_pad_midpoint_object_m': patch,
                       'shape_eligibility': {'axis_object': [0, 0, 1]}})
    return details, transform


def test_calibration_reconstruction_uses_geometry_not_stale_clock_indices():
    details, expected = calibration()
    details['pad_alignment']['retarget_anchor_frame'] = 90000
    np.testing.assert_allclose(acquisition.calibrated_object_tcp(details), expected, atol=1e-14)
    changed = copy.deepcopy(details)
    changed['cylindrical_grasp']['longitudinal_axis_after_object'] = changed['cylindrical_grasp']['closing_axis_after_object']
    with pytest.raises(ValueError, match='orthogonal'):
        acquisition.calibrated_object_tcp(changed)


def test_ramp_clock_roundoff_does_not_add_an_unrequested_interval():
    dt = float(np.median(np.diff(np.arange(3300)*.01)))
    assert acquisition._ramp_intervals(.1, dt) == 10
    assert acquisition._ramp_intervals(.10001, dt) == 11


def selection_fixture():
    details, _ = calibration()
    midpoint = np.asarray(details['pad_alignment']['pad']['midpoint_tcp'])
    patch = np.asarray(details['cylindrical_grasp']['desired_pad_midpoint_object_m'])
    hands = np.tile(np.eye(4), (8, 1, 1))
    hands[:, :3, 3] = patch-midpoint
    hands[:3, 0, 3] += .01
    objects = np.tile(np.eye(4), (8, 1, 1))
    commands = np.r_[2., np.full(6, -.06), 2.]
    return np.arange(8)*.1, commands, hands, objects, details


def test_first_supported_row_precedes_lift_and_requires_true_source_clock():
    time, commands, hands, objects, details = selection_fixture()
    first, chosen, release, _ = acquisition.select_index(time, time, commands, hands, objects, np.eye(4), details)
    assert (first, chosen, release) == (1, 3, 7)
    objects[3:, 2, 3] = .021
    with pytest.raises(ValueError, match='pre-lift'):
        acquisition.select_index(time, time, commands, hands, objects, np.eye(4), details)
    with pytest.raises(ValueError, match='source-reference clock'):
        acquisition.select_index(time, time[::-1], commands, hands, objects, np.eye(4), details)


class Robot:
    def __init__(self):
        self.active = np.arange(17)

    def pack(self, arms, base):
        return np.r_[base, arms]

    def fk(self, q):
        poses = np.tile(np.eye(4), (2, 1, 1))
        poses[1, :3, 3] = q[-6:-3]
        poses[1, :3, :3] = Rotation.from_rotvec(q[-3:]).as_matrix()
        return poses


def test_exit_consumes_every_original_row_and_exactly_rejoins_joint_suffix():
    robot = Robot()
    reference = np.zeros((20, 17)); reference[:, -1] = np.arange(20)*.01
    goals = np.asarray([robot.fk(row)[1] for row in reference])
    fixed = reference[4].copy(); fixed[-1] += .03; fixed[-4] += .002
    before = reference.copy()
    result = acquisition.ramps(reference, goals, 4, fixed, robot, 5)
    np.testing.assert_array_equal(reference, before)
    np.testing.assert_array_equal(result['entry_reference'][0], reference[4])
    np.testing.assert_array_equal(result['entry_reference'][-1], fixed)
    np.testing.assert_array_equal(result['exit_source_indices'], np.arange(5, 10))
    np.testing.assert_array_equal(result['exit_reference'][-1], reference[9])
    np.testing.assert_array_equal(result['exit_hand_goals'][-1], goals[9])
    np.testing.assert_array_equal(result['exit_reference'][:, :10], reference[5:10, :10])
    with pytest.raises(ValueError, match='subsequent'):
        acquisition.ramps(reference, goals, 17, fixed, robot, 5)


def measured_fixture(monkeypatch):
    xml = '''<mujoco><worldbody><body name="base_link"><geom name="base" size=".01" pos="0 0 -1"/>
      <site name="r_arm_tip_tcp" pos="0 0 .4"/>
      <body name="r_hand_distal_link" pos="-.05 0 .4"><joint name="r_hand_finger" ref="2"/>
      <geom name="finger0" size=".004"/></body>
      <body name="r_hand_distal_mimic_link" pos=".05 0 .4"><geom name="finger1" size=".004"/></body></body>
      <body name="Can" pos="0 0 .4"><freejoint name="object_free"/><geom name="Can_geom" type="cylinder" size=".025 .04"/></body>
      </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    monkeypatch.setattr(acquisition, 'pad_surfaces', lambda m, d: {'midpoint_tcp': np.array([.005, 0, 0])})
    fixed = np.eye(4); fixed[2, 3] = .4
    meta = dict(object_body='Can', fixed_hand_goal=fixed, axis_object=[0, 0, 1], patch_object_m=[0, 0, 0],
                calibrated_midpoint_tcp_m=[0, 0, 0], wall_interval_with_guard_m=[-.039, .039],
                tcp_position_tolerance_m=.001, tcp_rotation_tolerance_rad=.01,
                radial_tolerance_m=.001, axial_tolerance_m=.002, non_distal_clearance_m=.003,
                contact_angle_rad=1., minimum_pad_force_N=.5)
    faces = [np.array([[[-.01, y, -.01], [.01, y, -.01], [0, y, .01]]]) for y in [-.025, .025]]
    monkeypatch.setattr(acquisition, 'finite_pad_faces', lambda m, d: faces)
    plan = dict(metadata=meta, arrays={'calibrated_pad0_triangles_tcp': faces[0], 'calibrated_pad1_triangles_tcp': faces[1]})
    return model, data, plan


def test_guard_is_read_only_and_accounts_for_open_aperture_midpoint(monkeypatch):
    model, data, plan = measured_fixture(monkeypatch)
    q, v, control = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
    report = acquisition.guard(model, data, plan)
    assert report['alignment'] and not report['bilateral_force']
    assert not report['actual_midpoint_guard_applicable']
    assert report['actual_aperture_pad_midpoint_object_m'][0] == pytest.approx(.005)
    np.testing.assert_array_equal(data.qpos, q)
    np.testing.assert_array_equal(data.qvel, v)
    np.testing.assert_array_equal(data.ctrl, control)
    data.qpos[model.joint('r_hand_finger').qposadr] = 1.
    mujoco.mj_forward(model, data)
    closed = acquisition.guard(model, data, plan)
    assert closed['actual_midpoint_guard_applicable']
    assert not closed['checks']['actual_contact_midpoint']


def test_guard_uses_actual_object_pose_and_rejects_body_contact(monkeypatch):
    model, data, plan = measured_fixture(monkeypatch)
    address = model.joint('object_free').qposadr[0]
    data.qpos[address] = .005
    mujoco.mj_forward(model, data)
    report = acquisition.guard(model, data, plan)
    assert not report['checks']['calibrated_contact_radial']
    assert report['calibrated_contact_radial_error_m'] == pytest.approx(.005)
    monkeypatch.setattr(acquisition, '_object_clearance', lambda *args, **kwargs: dict(
        whole_robot_object_gap_m=-.013, non_distal_object_gap_m=-.013, closest_non_distal_object_pair=['palm', 'Can']))
    assert not acquisition.guard(model, data, plan)['checks']['non_distal_object_clearance']


def test_missing_retimed_source_clock_saves_rejection_and_preserves_active(tmp_path):
    robot = Robot(); active = robot.active.copy()
    reference = np.zeros((8, 17)); poses = np.tile(np.eye(4), (8, 1, 1))
    prepared = (None, '<mujoco/>', {}, {}, {}, dict(task='PickPlaceCan', cylindrical_grasp={}, robot_control_retiming={'enabled': True}),
                reference[0], np.arange(8)*.01, reference, np.r_[2., np.full(7, -.06)], poses, poses)
    with pytest.raises(ValueError, match='rejected'):
        acquisition.prepare(prepared, robot, tmp_path/'rejection')
    report = json.loads((tmp_path/'rejection/result.json').read_text())
    assert 'source-reference time map' in report['exception']
    assert not report['admitted'] and not report['physics_validated']
    assert (tmp_path/'rejection/acquisition-plan.npz').exists()
    np.testing.assert_array_equal(robot.active, active)


def test_binding_rejects_source_mutation_and_numeric_plan_mutation(tmp_path):
    reference = np.zeros((4, 17)); poses = np.tile(np.eye(4), (4, 1, 1))
    prepared = (None, '<mujoco/>', {}, {}, {}, {}, reference[0], np.arange(4)*.1, reference, np.full(4, 2.), poses, poses)
    artifact = tmp_path/'numeric.npz'; values = {'entry_reference': reference.copy()}; np.savez(artifact, **values)
    plan = dict(arrays=values, metadata=dict(input_binding=acquisition.binding(prepared), artifact=str(artifact),
                artifact_sha256=acquisition.sha256(artifact), plan_array_hashes={k: acquisition._array_hash(v) for k, v in values.items()}))
    assert acquisition.verify(prepared, plan)
    reference[0, 0] = .1
    with pytest.raises(ValueError, match='different'):
        acquisition.verify(prepared, plan)
    reference[0, 0] = 0.
    values['entry_reference'][0, 1] = .1
    with pytest.raises(ValueError, match='in-memory'):
        acquisition.verify(prepared, plan)
