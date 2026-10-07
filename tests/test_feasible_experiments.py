"""Bounded gripper experiments preserve source intent and physical references."""
from dataclasses import asdict
import numpy as np
import pytest
from reachy_retarget.dynamics import Candidate
from reachy_retarget.feasible_experiments import variant


def prepared():
    c=Candidate(force_feedback=False,integral_compensation=False)
    details={'effective_candidate':asdict(c),'pad_alignment':{'inferred_contact_angle_rad':.8}}
    values=[object() for _ in range(12)];values[5]=details
    values[7]=np.array([0.,.01,.02]);values[9]=np.array([2.,-.06,-.06])
    return c,tuple(values)


def test_calibrated_positive_closed_target_preserves_contact_intent_and_all_frozen_factors():
    candidate,original=prepared();changed,result=variant(candidate,original,'contact_005')
    assert changed.gripper_closed_target==pytest.approx(.75)
    assert candidate.gripper_closed_target==-.06
    np.testing.assert_array_equal(result[9],[2.,-.06,-.06])
    for i in range(12):
        if i!=5:assert result[i] is original[i]
    assert 'experiment_variant' not in original[5]


def test_unverified_aperture_is_rejected_instead_of_using_guessed_angle():
    candidate,original=prepared();del original[5]['pad_alignment']
    with pytest.raises(ValueError,match='verified pad'):variant(candidate,original,'contact_002')


def test_force_variant_keeps_mechanical_endpoint_and_records_bounded_policy():
    candidate,original=prepared();changed,result=variant(candidate,original,'force_1n')
    assert changed.force_feedback and changed.balanced_force and changed.soft_force_feedback
    assert changed.grasp_force==1. and changed.gripper_closed_target==-.06
    assert result[5]['effective_candidate']['grasp_force']==1.


def test_preclose_uses_calibrated_opening_and_preserves_source_timeline():
    candidate,original=prepared();changed,result=variant(candidate,original,'preforce2_velocity')
    assert changed.force_preclose_target==pytest.approx(.82)
    assert changed.arm_velocity_feedforward==1.
    assert result[7] is original[7] and result[9] is original[9]
