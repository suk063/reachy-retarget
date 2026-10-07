import numpy as np

from reachy_retarget.retarget import resample_object_track


def test_explicit_invalid_finite_pose_is_not_bridged():
    times = np.arange(4.)
    poses = np.tile([0., 0, 0, 1, 0, 0, 0], (4, 1))
    poses[:, 0] = times
    mask = np.array([True, False, True, True])
    target = np.arange(0, 3.01, .5)
    placement = np.eye(4)
    placement[2, 3] = .8
    values, valid = resample_object_track(times, poses, target, placement, mask)
    np.testing.assert_array_equal(valid, [True, False, False, False, True, True, True])
    np.testing.assert_allclose(values[valid, :3], np.c_[target[valid], np.zeros(valid.sum()), np.full(valid.sum(), .8)])
    assert np.isnan(values[~valid]).all()
    np.testing.assert_array_equal(mask, [True, False, True, True])


def test_nonunit_and_missing_quaternions_remain_missing():
    poses = np.tile([0., 0, 0, 1, 0, 0, 0], (3, 1))
    poses[1, 3:] = 0
    poses[2] = np.nan
    result, valid = resample_object_track(np.arange(3.), poses, np.arange(0, 2.1, .5), np.eye(4))
    np.testing.assert_array_equal(valid, [True, False, False, False, False])
    assert np.isnan(result[~valid]).all()
