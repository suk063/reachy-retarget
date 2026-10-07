import copy

import numpy as np
import pytest

from reachy_retarget.unloaded_return import propose


def plan():
    n = 151
    time = np.arange(n)*.01
    hands = np.tile(np.eye(4), (n, 1, 1))
    hands[:, 0, 3] = np.minimum(time, 1.)
    objects = np.tile(np.eye(4), (n, 1, 1))
    objects[:, 1, 3] = np.arange(n)  # Distinct rows expose crop/repeat mistakes.
    commands = np.full(n, 2.); commands[10:40] = -.06
    details = dict(evaluation_hold_s=.5, robot_initialization={'admitted': True})
    return (None, 'original scene', {}, {}, {}, details, None, time,
            np.arange(n*17).reshape(n, 17), commands, hands, objects)


def test_return_preserves_source_and_closed_path_and_invalidates_old_admission(tmp_path):
    original = plan()
    before = copy.deepcopy(original)
    changed = propose(original, tmp_path/'proposal', lift_m=.06)
    for i in (7, 8, 9, 11):
        assert changed[i] is original[i]
        np.testing.assert_array_equal(changed[i], before[i])
    np.testing.assert_array_equal(changed[10][:41], original[10][:41])
    np.testing.assert_array_equal(changed[10][100:], original[10][100:])
    np.testing.assert_array_equal(changed[10][:, :3, :3], original[10][:, :3, :3])
    assert changed[10][60, 2, 3] == pytest.approx(.06)
    assert 'robot_initialization' not in changed[5]
    assert original[5]['robot_initialization']['admitted'] is True
    report = changed[5]['robot_unloaded_return_proposal']
    assert report['joint_reference_valid'] is False
    assert report['physics_validated'] is False
    assert report['original_return_end_frame'] == 100
    assert min(report['changed_source_frames']) > 40
    with np.load(report['artifact']) as saved:
        np.testing.assert_array_equal(saved['original_object_goals'], original[11])
        np.testing.assert_array_equal(saved['original_hand_goals'], original[10])
        np.testing.assert_array_equal(saved['hand_goals'], changed[10])


def test_terminal_correction_is_explicit_and_hold_remains_stationary(tmp_path):
    original = plan()
    changed = propose(original, tmp_path/'proposal', lift_m=.04, terminal_lift_m=.01)
    np.testing.assert_allclose(changed[10][100:, 2, 3], .01)
    assert not changed[5]['robot_unloaded_return_proposal']['terminal_hand_target_preserved_exactly']
    np.testing.assert_array_equal(changed[10][:41], original[10][:41])


def test_explicit_hold_alignment_preserves_late_release_blend_without_inventing_hold(tmp_path):
    original = list(plan())
    original[10][100:106, 0, 3] = np.arange(6)*.01+1.
    original[10][106:, 0, 3] = 1.05
    with pytest.raises(ValueError, match='stationary declared'):
        propose(tuple(original), tmp_path/'strict')
    changed = propose(tuple(original), tmp_path/'aligned', lift_m=.01,
                      terminal_lift_m=.01, hold_alignment_tolerance_s=.05)
    report = changed[5]['robot_unloaded_return_proposal']
    assert report['declared_return_end_frame'] == 100
    assert report['original_return_end_frame'] == 105
    assert report['declared_evaluation_hold_s'] == .5
    assert report['observed_stationary_hold_s'] == pytest.approx(.45)
    np.testing.assert_array_equal(changed[10][:, 0, 3], original[10][:, 0, 3])
    np.testing.assert_array_equal(changed[7], original[7])
    np.testing.assert_array_equal(changed[11], original[11])
    with pytest.raises(ValueError, match='exceeds'):
        propose(tuple(original), tmp_path/'too_short', hold_alignment_tolerance_s=.04)


@pytest.mark.parametrize('change', ['second_grasp', 'ambiguous_release', 'retimed', 'moving_hold'])
def test_ambiguous_phase_or_stale_clock_rejected(tmp_path, change):
    original = list(plan())
    if change == 'second_grasp':
        original[9][45] = -.06
    elif change == 'ambiguous_release':
        original[9][45] = .5
    elif change == 'retimed':
        original[5]['robot_control_retiming'] = {}
    else:
        original[10][-1, 0, 3] += .01
    with pytest.raises(ValueError):
        propose(tuple(original), tmp_path/'proposal')
    assert not (tmp_path/'proposal').exists()
