"""A safe pair of knots can have an unsafe joint-space chord."""
import numpy as np
import pytest
from reachy_retarget.retimed_source_clearance import RepairRejected, repair_reference


def example():
    original = np.zeros((2, 17)); original[:, 0] = [-1., 1.]
    reference = np.zeros((3, 17)); reference[:, 0] = [-1., 0., 1.]
    seconds = np.array([0., .5, 1.])
    speed = np.ones(17); speed[0] = 2.
    return reference, seconds, seconds.copy(), np.array([0., 1.]), original, speed


def gap(row):
    return .002 + .004*row[0]**2 + row[1]


def test_only_new_unsafe_chord_sample_changes_and_original_knots_remain_exact():
    inputs = example(); before = inputs[0].copy(); visited = []
    def solve(i, row, lower, upper):
        visited.append((i, lower.copy(), upper.copy()))
        row[1] = .00101
        return row
    changed, proof = repair_reference(*inputs, gap, solve)
    assert [i for i, _, _ in visited] == [1]
    assert np.array_equal(inputs[0], before)
    assert np.array_equal(changed[[0, 2]], before[[0, 2]])
    assert np.array_equal(changed[:, 3:], before[:, 3:])
    assert np.all(proof['after_gaps_m'] >= .003)
    assert proof['status'] == 'admitted'
    assert proof['repaired_rows'][0]['before_gap_m'] == .002


def test_source_endpoint_failure_is_not_reclassified_as_interpolation():
    called = []
    with pytest.raises(RepairRejected, match='Original source knots') as error:
        repair_reference(*example(), lambda row: .002, lambda *args: called.append(args))
    assert not called
    assert error.value.evidence['status'] == 'original_knot_rejected'
    assert np.array_equal(error.value.evidence['corrected_reference'], example()[0])


def test_solver_cannot_change_inactive_arm_or_exceed_neighbor_speed_bounds():
    def inactive(i, row, lower, upper):
        row[1] = .00101; row[3] = .001
        return row
    with pytest.raises(RepairRejected, match='independent checks'):
        repair_reference(*example(), gap, inactive)
    inputs = list(example()); inputs[-1][1] = .001
    def fast(i, row, lower, upper):
        assert upper[1] == pytest.approx(.0005)
        row[1] = .00101
        return row
    with pytest.raises(RepairRejected, match='independent checks'):
        repair_reference(*inputs, gap, fast)


def test_partial_numeric_evidence_survives_solver_exception():
    def fail(*args):
        raise RuntimeError('bounded numerical solve failed')
    with pytest.raises(RepairRejected, match='bounded numerical solve failed') as error:
        repair_reference(*example(), gap, fail)
    assert error.value.evidence['status'] == 'solver_exception'
    assert np.isfinite(error.value.evidence['before_gaps_m']).all()


def test_saturated_neighbor_speed_roundoff_does_not_reject_valid_interval():
    inputs = list(example())
    inputs[0][:, 13] = [.37, .77, 1.17 + 5e-15]
    inputs[4][:, 13] = inputs[0][[0, 2], 13]
    inputs[-1][13] = .8
    def solve(i, row, lower, upper):
        row[1] = .00101
        row[13] = (lower[6]+upper[6])*.5
        return row
    changed, proof = repair_reference(*inputs, gap, solve)
    assert proof['status'] == 'admitted'
    assert np.max(np.abs(np.diff(changed[:, 13]))/.5) <= .8+1e-7


def test_unbound_reference_or_cropped_clock_rejected_before_geometry():
    inputs = list(example()); inputs[0] = inputs[0].copy(); inputs[0][1, 2] = .01
    with pytest.raises(ValueError, match='bound linear retiming'):
        repair_reference(*inputs, gap, lambda *args: None)
    inputs = list(example()); inputs[2] = np.array([.1, .5, .9])
    with pytest.raises(ValueError):
        repair_reference(*inputs, gap, lambda *args: None)


def test_planar_norm_constraint_cannot_be_replaced_by_separate_axis_limits():
    inputs=list(example());inputs[0][:,:2]=[[-.5,-.5],[0.,0.],[.5,.5]]
    inputs[4][:,:2]=inputs[0][[0,2],:2];inputs[-1][:2]=2.
    def fixture(row):return .002+.008*row[0]**2+row[10]
    def bad(i,row,lower,upper,*,planar_constraints):
        np.testing.assert_allclose(planar_constraints[:,:2],[[-.5,-.5],[.5,.5]])
        row[0]=.001;row[10]=.00101
        assert np.all(row[np.r_[0:3,10:17]]<=upper)
        return row
    with pytest.raises(RepairRejected,match='independent checks'):
        repair_reference(*inputs,fixture,bad,planar_speed=np.sqrt(2))
    def good(i,row,lower,upper,*,planar_constraints):
        row[10]=.00101
        return row
    changed,proof=repair_reference(*inputs,fixture,good,planar_speed=np.sqrt(2))
    np.testing.assert_array_equal(changed[:,:2],inputs[0][:,:2])
    assert proof['status']=='admitted' and proof['planar_speed_peak_m_s']<=np.sqrt(2)
