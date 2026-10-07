"""Cached geometry cannot silently change aperture or discard source phases."""
from copy import deepcopy
import numpy as np
import pytest
from reachy_retarget.feasible_placement_search import validate_parent, chunk_placements
from cluster.local_spool_experiment import runner_module


def parent():
    data = [None]*12
    data[7] = np.array([0., .01, .02])
    data[5] = dict(task='PickPlaceCan', experiment_variant='contact_0035',
        robot_initialization=dict(admitted=True, completed_frames=3, total_frames=3,
                                  admission_checks={'ik': True, 'environment': True}))
    return data


def test_parent_requires_complete_original_path_and_identical_aperture():
    p = parent()
    validate_parent(p, 'contact_0035')
    with pytest.raises(ValueError, match='aperture'):
        validate_parent(p, 'contact_005')
    for key in ('robot_control_retiming', 'robot_supported_placement',
                'cylindrical_acquisition', 'box_acquisition', 'planning_fixture_forecast'):
        modified = deepcopy(p)
        modified[5][key] = {'present': True}
        with pytest.raises(ValueError):
            validate_parent(modified, 'contact_0035')
    p[5]['robot_initialization']['completed_frames'] = 2
    with pytest.raises(ValueError, match='complete'):
        validate_parent(p, 'contact_0035')


def test_chunks_cover_bank_once_in_physics_independent_rank_order():
    bank = [dict(id=i, rank=[45-i]) for i in range(45)]
    chunks = [chunk_placements(bank, i, 3) for i in range(3)]
    assert sorted(p['id'] for chunk in chunks for p in chunk) == list(range(45))
    assert all([p['rank'] for p in chunk] == sorted(p['rank'] for p in chunk) for chunk in chunks)
    with pytest.raises(ValueError):
        chunk_placements(bank, 3, 3)
    assert runner_module({'operation': 'feasible_placement_search'}) == 'reachy_retarget.feasible_placement_search'
def test_adaptive_chord_repair_preserves_bank_identity_and_records_selected_policy():
    from reachy_retarget.feasible_placement_search import refinement_bank
    import pytest
    original = [dict(id=7, rank=[.2, .01, 7], parameters={'extra_descent_m': .002})]
    assert refinement_bank(original, 0) is original
    repaired = refinement_bank(original, 3)
    assert repaired[0]['id'] == 7 and repaired[0]['rank'] == original[0]['rank']
    assert repaired[0]['parameters'] == {'extra_descent_m': .002, 'adaptive_midpoints': 3}
    assert 'adaptive_midpoints' not in original[0]['parameters']
    local = refinement_bank(original, 3, midpoint_minimum_motion=True, interval_knot_timing=True)
    assert local[0]['parameters']['midpoint_minimum_motion'] is True
    assert local[0]['parameters']['interval_knot_timing'] is True
    interior = refinement_bank(original, 3, midpoint_position_reserve_m=.0001,
                               midpoint_fixture_reserve_m=.0001)
    assert interior[0]['id'] == original[0]['id'] and interior[0]['rank'] == original[0]['rank']
    assert interior[0]['parameters']['midpoint_position_reserve_m'] == .0001
    assert interior[0]['parameters']['midpoint_fixture_reserve_m'] == .0001
    quarters=refinement_bank(original,3,midpoint_quarter_probes=True)
    assert quarters[0]['parameters']['midpoint_quarter_probes'] is True
    for rounds, value in ((0,True),(3,1),(3,'yes')):
        with pytest.raises(ValueError,match='booleans'):
            refinement_bank(original,rounds,midpoint_quarter_probes=value)
    for key in ('midpoint_position_reserve_m', 'midpoint_fixture_reserve_m'):
        for value in (-.0001, .002, float('nan'), float('inf'), True, '0.0001'):
            with pytest.raises(ValueError, match='finite reserve'):
                refinement_bank(original, 3, **{key: value})
        with pytest.raises(ValueError, match='nonzero'):
            refinement_bank(original, 0, **{key: .0001})
    with pytest.raises(ValueError, match='nonzero'):
        refinement_bank(original, 0, midpoint_minimum_motion=True)
    for value in (True, -1, 4, 1.5):
        with pytest.raises(ValueError, match='integer'):
            refinement_bank(original, value)
