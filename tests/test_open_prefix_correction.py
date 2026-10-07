import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget.open_prefix_correction import apply, base_track, blend, weights


def prepared():
    original = np.repeat(np.eye(4)[None], 11, axis=0)
    original[:, 0, 3] = np.arange(11)*.01
    changed = original.copy()
    changed[:, :3, :3] = Rotation.from_euler('y', .5).as_matrix()
    changed[:, 1, 3] += .03
    commands = np.r_[np.full(8, 2.), np.full(3, -.06)]
    return original, (None, None, None, None, None, {}, None, np.arange(11)*.1,
                       np.zeros((11, 17)), commands, changed, original.copy())


def test_initial_pose_base_and_closed_suffix_are_preserved(tmp_path):
    original, p = prepared()
    result = apply(p, original, tmp_path/'blend', complete_fraction=.75)
    np.testing.assert_array_equal(result[10][0], original[0])
    np.testing.assert_array_equal(result[10][8:], p[10][8:])
    assert 0. < result[10][3, 1, 3] < .03
    for index in (7, 8, 9, 11):
        assert result[index] is p[index]
    track = base_track(p, [-.05, .05, .2], complete_fraction=.75)
    np.testing.assert_array_equal(track[0], p[8][0, :3])
    np.testing.assert_array_equal(track[8:], np.tile([-.05, .05, .2], (3, 1)))
    assert result[5]['open_prefix_correction']['requires_full_ik_and_collision_admission']


def test_closed_prefix_or_mismatched_source_arrays_are_rejected():
    original, p = prepared()
    with pytest.raises(ValueError, match='explicitly open'):
        weights(p[7], np.full(11, -.06))
    with pytest.raises(ValueError, match='Aligned'):
        blend(original[:-1], p[10], np.ones(11))
    invalid = original.copy(); invalid[0, :3, :3] *= 2.
    with pytest.raises(ValueError, match='Rigid'):
        blend(invalid, p[10], np.ones(11))


def test_real_stack_528b8e2_clock_polynomial_roundoff_stays_in_range():
    # Recorded plan:736 rows, first close255 at2.5500000000000003s.
    # Completion2.0400000000000005 puts row204 at phase1-2e-16;
    # the unbounded polynomial previously returned1.0000000000000002.
    time = np.arange(736)*.01
    commands = np.full(736, 2.); commands[255:500] = -.06
    amount = weights(time, commands, complete_fraction=.8)
    assert np.all((amount >= 0.) & (amount <= 1.))
    assert amount[204] == 1.
    original = np.tile(np.eye(4), (736, 1, 1))
    corrected = original.copy(); corrected[:, 2, 3] = .04
    changed = blend(original, corrected, amount)
    np.testing.assert_array_equal(changed[204:], corrected[204:])
    np.testing.assert_array_equal(changed[0], original[0])
