import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget.grasp_roll import apply, rotate_about_pad_line


def test_roll_preserves_entire_contact_line_and_world_relative_motion():
    hands = np.repeat(np.eye(4)[None], 5, axis=0)
    hands[:, :3, :3] = Rotation.from_euler('zyx', np.arange(15).reshape(5, 3)*.03).as_matrix()
    hands[:, :3, 3] = np.arange(15).reshape(5, 3)*.02
    saved = hands.copy()
    midpoint = np.array([-.01, 0., -.05]); axis = np.array([0., 2., 0.])
    changed, attachment = rotate_about_pad_line(hands, midpoint, axis, np.pi/3)
    for distance in (-.08, -.02, 0., .02, .08):
        point = midpoint+distance*axis
        np.testing.assert_allclose(changed[:, :3, :3]@point+changed[:, :3, 3],
                                   hands[:, :3, :3]@point+hands[:, :3, 3], atol=1e-14)
    np.testing.assert_allclose(changed[4]@np.linalg.inv(changed[1]),
                               hands[4]@np.linalg.inv(hands[1]), atol=1e-14)
    np.testing.assert_allclose(changed, hands@attachment, atol=1e-14)
    np.testing.assert_array_equal(hands, saved)
    assert np.linalg.norm(changed[0, :3, 3]-hands[0, :3, 3]) > .04


def test_roll_invalidates_old_admission_without_mutating_source(tmp_path):
    hands = np.repeat(np.eye(4)[None], 3, axis=0)
    details = dict(pad_alignment=dict(pad=dict(points_tcp=[[0., -.02, -.05], [0., .02, -.05]],
        midpoint_tcp=[0., 0., -.05]), inferred_contact_angle_rad=.94),
        robot_initialization=dict(admitted=True, total_frames=3, completed_frames=3))
    source = (None, None, None, None, None, details, None, np.arange(3.), np.zeros((3, 17)),
              np.array([2., -.06, -.06]), hands, hands.copy())
    result = apply(source, tmp_path/'roll', angle_rad=.3)
    for i in (7, 8, 9, 11):
        assert result[i] is source[i]
    assert source[5]['robot_initialization']['admitted']
    assert 'robot_initialization' not in result[5]
    assert result[5]['requires_full_ik_and_collision_admission']
    report = json.loads((tmp_path/'roll/result.json').read_text())
    assert report['max_calibrated_pad_displacement_m'] < 1e-14
    assert report['invalidated_prior_admissions']['robot_initialization']['admitted']


@pytest.mark.parametrize('axis,angle', [([0., 0., 0.], 0.), ([0., 1., 0.], np.pi),
                                       ([0., 1., np.nan], .1)])
def test_invalid_roll_is_rejected(axis, angle):
    with pytest.raises(ValueError):
        rotate_about_pad_line(np.eye(4)[None], [0., 0., -.05], axis, angle)
