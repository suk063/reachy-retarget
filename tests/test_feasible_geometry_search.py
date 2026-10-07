import json
from pathlib import Path

import numpy as np
import pytest

from reachy_retarget import feasible_geometry_search as search


def fixture():
    n = 30
    hands = np.tile(np.eye(4), (n, 1, 1)); hands[13, 0, 3] = 2.
    objects = hands.copy(); objects[:, :3, 3] = [1., -.3, .85]
    reference = np.zeros((n, 17)); reference[19, 1] = -.1
    commands = np.r_[np.full(7, 2.), np.full(13, -.06), np.full(10, 2.)]
    frame = np.eye(4); frame[:2, :2] = [[0., -1.], [1., 0.]]
    details = dict(task='PickPlaceCan', world_placement=frame.tolist(),
                   pad_alignment=dict(inferred_contact_angle_rad=1.1, pad=dict(
                       points_tcp=[[0., -.02, .05], [0., .02, .05]], midpoint_tcp=[0., 0., .05])),
                   task_contract=dict(bin_lower=[-.4, -1.1, .8], bin_upper=[-.2, -.9, 1.1]))
    return (None, 'original scene', {}, {}, {}, details, None, np.arange(n)*.01,
            reference, commands, hands, objects)


def test_bank_chunks_are_complete_disjoint_deterministic_and_ranked():
    chunks = [search.chunk_candidates(i, 3) for i in range(3)]
    assert [len(c) for c in chunks] == [18]*3
    assert sorted(c['id'] for chunk in chunks for c in chunk) == list(range(54))
    assert all(chunk == sorted(chunk, key=lambda c: c['rank']) for chunk in chunks)
    assert search.describe_bank() == search.describe_bank()
    with pytest.raises(ValueError):
        search.chunk_candidates(3, 3)


def test_sparse_screen_retains_transitions_extrema_and_labels_incomplete_scope():
    original = fixture(); indices = search.screen_indices(original)
    assert {0, 6, 7, 13, 19, 20, 29} <= set(indices)
    sampled = search._sample(original, indices)
    assert sampled[5]['geometry_search_screening_only']
    assert 'geometry_search_screening_only' not in original[5]
    for i in range(7, 12):
        np.testing.assert_array_equal(sampled[i], original[i][indices])
    assert sampled[3] is original[3]


def test_placement_bank_uses_original_verified_region_and_finite_rank():
    original = fixture(); bank = search.placement_bank(original)
    assert len(bank) == 45
    assert bank == sorted(bank, key=lambda c: c['rank'])
    frame = np.asarray(original[5]['world_placement'])
    for item in bank:
        xyz = np.r_[item['parameters']['placement_xy_world'], .9]
        local = frame[:3, :3].T @ (xyz-frame[:3, 3])
        assert np.all(local > original[5]['task_contract']['bin_lower'])
        assert np.all(local < original[5]['task_contract']['bin_upper'])
        assert np.isfinite(item['rank']).all()


def test_select_requires_complete_chunks_and_ignores_physics_outcomes():
    a = dict(selection_rank=[0., 0, 3, .5, 1], physics_validated=False)
    b = dict(selection_rank=[.1, 0, 3, .1, 1], physics_validated=True)
    rows = [dict(chunk_index=0, selected=b), dict(chunk_index=1, selected=a)]
    assert search.select(rows, expected_chunks=2) is a
    with pytest.raises(ValueError, match='Complete'):
        search.select(rows[:1], expected_chunks=2)
    rows[0]['search_complete'] = False
    with pytest.raises(ValueError, match='Incomplete'):
        search.select(rows, expected_chunks=2)


def test_roll_selection_requires_every_declared_angle_and_chunk():
    angles = np.deg2rad([-15., 15.]).tolist()
    rows = [dict(chunk_index=i, grasp_roll=dict(angle_rad=a), selected=None)
            for a in angles for i in range(3)]
    assert search.select(rows, expected_chunks=3, expected_rolls=angles) is None
    with pytest.raises(ValueError, match='Complete'):
        search.select(rows[:-1], expected_chunks=3, expected_rolls=angles)


@pytest.mark.parametrize('solver_error', [False, True])
@pytest.mark.parametrize('rolled', [False, True])
def test_runner_retains_earlier_rejection_and_caches_base_before_resampling(monkeypatch, tmp_path, solver_error, rolled):
    from reachy_retarget import feasible_base, frozen_plan, grasp_approach, feasible_timing, supported_placement
    from reachy_retarget import feasible_experiments, robot
    from reachy_retarget.dynamics import Candidate
    original = fixture(); workspace = tmp_path/'workspace'; workspace.mkdir()
    (workspace/'source.h5').write_bytes(b'preserved source')
    monkeypatch.setattr(robot, 'Robot', lambda _: object())
    monkeypatch.setattr(frozen_plan, 'load', lambda *args: (Candidate(), original))
    monkeypatch.setattr(search, 'chunk_candidates', lambda *args: [
        dict(id=0, seed='source', base_offset_xyyaw=[0., 0., 0.], rank=[0., 0, 0]),
        dict(id=1, seed='source', base_offset_xyyaw=[.1, 0., 0.], rank=[.1, 0, 1])])
    calls = []
    def base(prepared, rob, output, track, **kwargs):
        Path(output).mkdir(parents=True)
        calls.append(('base', len(prepared[7])))
        if 'base-00' in str(output):
            if solver_error:
                raise RuntimeError('solver interrupted')
            (Path(output)/'result.json').write_text(json.dumps(dict(admitted=False, exception=None)))
            raise ValueError('collision rejected')
        return prepared
    monkeypatch.setattr(feasible_base, 'prepare', base)
    def cache(prepared, candidate, source, rob, output):
        calls.append(('cache', len(prepared[7]))); Path(output).mkdir(); return Path(output)
    monkeypatch.setattr(frozen_plan, 'save_admitted', cache)
    monkeypatch.setattr(grasp_approach, 'prepare', lambda p, *args, **kwargs: p)
    monkeypatch.setattr(feasible_timing, 'prepare', lambda p, *args, **kwargs: p)
    monkeypatch.setattr(feasible_experiments, 'variant', lambda c, p, name: (c, p))
    monkeypatch.setattr(search, 'placement_bank', lambda _: [dict(id=0, rank=[.1, 0], parameters={})])
    monkeypatch.setattr(search, '_screen_placement', lambda *args: dict(passed=True, exception=None))
    monkeypatch.setattr(supported_placement, 'prepare', lambda p, *args, **kwargs: p)
    job = dict(workspace='workspace', source='source.h5', evaluation_split='geometry_optimization',
               geometry_search=dict(chunk_index=0, chunk_count=1))
    if rolled:
        job['grasp_roll'] = dict(angle_rad=float(np.deg2rad(15.)))
    result = search.run(tmp_path, job)
    assert result['status'] == ('search_incomplete' if solver_error else 'kinematic_candidate')
    assert result['candidates'][0]['status'] == ('error' if solver_error else 'rejected')
    assert result['search_complete'] is not solver_error
    if solver_error:
        with pytest.raises(ValueError, match='Incomplete'):
            search.select([result], expected_chunks=1)
    assert result['candidates'][1]['status'] == 'admitted'
    assert not result['physics_tested'] and not result['physics_validated']
    assert calls[-2:] == [('cache', len(original[7])), ('cache', len(original[7]))]
    assert result['selected']['admitted_base_plan'] != result['selected']['admitted_plan']
    saved = np.load(workspace/'geometry-search/source-reference.npz')
    np.testing.assert_array_equal(saved['hand_goals'], original[10])
    np.testing.assert_array_equal(saved['object_goals'], original[11])
    assert (workspace/'source.h5').read_bytes() == b'preserved source'
    if rolled:
        assert result['selected']['selection_rank'][0] == np.deg2rad(15.)
        assert result['selected']['grasp_roll'] == job['grasp_roll']
        assert 'grasp_roll' not in original[5]
        with np.load(workspace/'geometry-search/grasp-roll/pad-line-roll.npz') as rolled_artifact:
            np.testing.assert_array_equal(rolled_artifact['original_hand_goals'], original[10])
            assert not np.array_equal(rolled_artifact['hand_goals'], original[10])
