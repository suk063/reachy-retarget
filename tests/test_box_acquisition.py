import json
import mujoco
import numpy as np
import pytest
from types import SimpleNamespace
from scipy.spatial.transform import Rotation

from reachy_retarget import box_acquisition as acquisition


def metadata():
    fixed = np.eye(4); fixed[2, 3] = .4
    return dict(object_body='needle', fixed_hand_goal=fixed, fixed_object_relative_tcp=np.eye(4),
                component={'center_object_m': [0, 0, 0], 'axes_object': np.eye(3), 'half_size_m': [.02]*3},
                face_axis=0, axis_object=[1, 0, 0], patch_object_m=[0, 0, 0], calibrated_midpoint_tcp_m=[0, 0, 0],
                calibrated_points_tcp_m=[[-.02, 0, 0], [.02, 0, 0]],
                tcp_position_tolerance_m=.001, tcp_rotation_tolerance_rad=.01,
                relative_rotation_tolerance_rad=.01, radial_tolerance_m=.002, axial_tolerance_m=.001,
                minimum_supported_fraction=.99, minimum_center_edge_margin_m=.01,
                non_distal_clearance_m=.003, contact_angle_rad=1., minimum_pad_force_N=.1)


def faces():
    return [np.array([[[x, -.01, -.01], [x, .01, -.01], [x, 0, .01]]]) for x in [-.02, .02]]


def test_exact_projected_clipping_measures_supported_fraction_without_full_pad_claim():
    triangle = np.array([[0., 0.], [2., 0.], [0., 2.]])
    assert acquisition.polygon_area(triangle) == pytest.approx(2.)
    assert acquisition.clipped_triangle_area(triangle, [1., 1.]) == pytest.approx(1.)
    assert acquisition.clipped_triangle_area(triangle+4., [1., 1.]) == 0.
    meta = metadata()
    supported = acquisition.projected_support(faces(), np.eye(4), meta['calibrated_points_tcp_m'], meta)
    assert all(pad['supported_fraction'] == 1. for pad in supported['pads'])
    assert supported['contact_center_edge_margin_m'] == pytest.approx(.02)
    moved = np.eye(4); moved[2, 3] = .015
    partial = acquisition.projected_support(faces(), moved, meta['calibrated_points_tcp_m'], meta)
    assert all(0 < pad['supported_fraction'] < 1 for pad in partial['pads'])
    assert partial['contact_center_edge_margin_m'] == pytest.approx(.005)


def test_read_only_guard_detects_object_rotation_with_unchanged_tcp_and_center(monkeypatch):
    xml = '''<mujoco><worldbody><body name="base_link"><geom size=".01" pos="0 0 -1"/>
      <site name="r_arm_tip_tcp" pos="0 0 .4"/>
      <body name="r_hand_distal_link" pos="-.05 0 .4"><joint name="r_hand_finger" ref="2"/>
      <geom size=".004"/></body><body name="r_hand_distal_mimic_link" pos=".05 0 .4"><geom size=".004"/></body></body>
      <body name="needle" pos="0 0 .4"><freejoint name="object_free"/><geom type="box" size=".02 .02 .02"/></body>
      </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    monkeypatch.setattr(acquisition, 'pad_surfaces', lambda m, d: {'midpoint_tcp': np.zeros(3)})
    plan = dict(metadata=metadata(), arrays={f'calibrated_pad{i}_triangles_tcp': face for i, face in enumerate(faces())})
    original = [data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()]
    valid = acquisition.guard(model, data, plan)
    assert valid['alignment'] and not valid['bilateral_force']
    for actual, expected in zip([data.qpos, data.qvel, data.ctrl], original):
        np.testing.assert_array_equal(actual, expected)
    address = model.joint('object_free').qposadr[0]
    xyzw = Rotation.from_rotvec([.03, 0, 0]).as_quat()
    data.qpos[address+3:address+7] = xyzw[[3, 0, 1, 2]]
    mujoco.mj_forward(model, data)
    rotated = acquisition.guard(model, data, plan)
    assert rotated['target_alignment'] and rotated['geometry_safe']
    assert rotated['checks']['calibrated_contact_axial'] and rotated['checks']['calibrated_contact_radial']
    assert not rotated['checks']['relative_rotation'] and not rotated['alignment']


def test_selection_uses_complete_clock_and_pre_lift_reset_patch():
    time = np.arange(8)*.1
    hand = np.tile(np.eye(4), (8, 1, 1)); hand[:, 1, 3] = .06; hand[:3, 2, 3] = .004
    objects = np.tile(np.eye(4), (8, 1, 1))
    commands = np.r_[2., np.full(6, -.06), 2.]
    details = dict(pad_alignment={'pad': {'midpoint_tcp': [0, 0, 0]}},
                   grasp_attachment={'face_axis': 0, 'shape_eligibility': {'axes_object': np.eye(3), 'center_object_m': [0, .06, 0]}})
    result = acquisition.select_index(time, time, commands, hand, objects, np.eye(4), details)
    assert result[:3] == (1, 3, 7)
    objects[3:, 2, 3] = .016
    with pytest.raises(ValueError, match='pre-lift'):
        acquisition.select_index(time, time, commands, hand, objects, np.eye(4), details)
    with pytest.raises(ValueError, match='clock'):
        acquisition.select_index(time, time[::-1], commands, hand, objects, np.eye(4), details)


def test_retimed_anchor_is_rejected_before_geometry_and_failure_is_preserved(tmp_path):
    class Robot:
        active = np.array([1, 2, 3])
    reference = np.zeros((8, 17)); poses = np.tile(np.eye(4), (8, 1, 1))
    details = dict(task='Threading_D0', grasp_attachment={'method': 'native_grasp_box_component'}, robot_control_retiming={'enabled': True})
    prepared = (None, '<mujoco/>', {}, {}, {}, details, reference[0], np.arange(8)*.01, reference,
                np.r_[2., np.full(7, -.06)], poses, poses)
    with pytest.raises(ValueError, match='rejected'):
        acquisition.prepare(prepared, Robot(), tmp_path/'rejected')
    report = json.loads((tmp_path/'rejected/result.json').read_text())
    assert not report['admitted'] and not report['physics_validated']
    assert 'remapped anchors' in report['exception']
    assert (tmp_path/'rejected/acquisition-plan.npz').exists()


@pytest.mark.parametrize('unreachable', [False, True])
def test_explicit_pose_tolerance_box_preserves_base_and_joint_bounds(unreachable):
    class Robot:
        arm_ids = np.arange(3, 17)
        r = SimpleNamespace(model=SimpleNamespace(lowerPositionLimit=np.full(17, -1.),
                                                 upperPositionLimit=np.full(17, 1.)))

        def fk(self, q):
            pose = np.eye(4)
            # Neither residual can be minimized independently; only a narrow
            # interval satisfies both. A different fixture makes it infeasible.
            pose[0, 3] = (.002 if unreachable else .0011)-.001*q[10]
            pose[:3, :3] = Rotation.from_rotvec([0., 0., .009+.008*q[10]]).as_matrix()
            return np.array([np.eye(4), pose])

    original = np.zeros(17); original[:10] = .23
    saved = original.copy()
    changed, report = acquisition.refine_fixed_patch(Robot(), original, np.eye(4), np.eye(4),
        np.zeros(3), np.zeros(3), np.array([1., 0., 0.]),
        position_m=.001, rotation_rad=.01, axial_m=.001, radial_m=.002)
    np.testing.assert_array_equal(original, saved)
    np.testing.assert_array_equal(changed[:10], saved[:10])
    assert np.all(abs(changed[10:]) <= .969+1e-12)
    assert report['base_and_inactive_arm_unchanged']
    if unreachable:
        assert not report['within_guard_bounds']
    else:
        assert report['within_guard_bounds']
        assert report['maximum_normalized_error'] == pytest.approx(.988888888889, abs=1e-7)
        assert report['normalized_errors_before'][0] > 1.
