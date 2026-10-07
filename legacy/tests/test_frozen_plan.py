import pytest
from reachy_retarget.frozen_plan import validate_alignment, validate_cache_admission


def test_modified_attachment_requires_completed_postmodification_admission():
    details={'pad_alignment':{},'alignment_variant':'constant_tcp_attachment',
             'grasp_attachment':{'requires_full_ik_and_collision_admission':True}}
    with pytest.raises(ValueError,match='subsequent'):validate_alignment(details)
    details['robot_initialization']={'admitted':True,'admission_checks':{'ik':True,'collision':True},
                                      'completed_frames':726,'total_frames':726}
    validate_alignment(details)
    details['robot_initialization']['completed_frames']=725
    with pytest.raises(ValueError,match='subsequent'):validate_alignment(details)


def test_unrecognized_alignment_is_never_reused():
    with pytest.raises(ValueError,match='Unsupported'):
        validate_alignment({'pad_alignment':{},'alignment_variant':'unverified'})


def test_empty_admission_is_not_a_verified_attachment():
    with pytest.raises(ValueError, match="subsequent"):
        validate_alignment({'pad_alignment': {}, 'alignment_variant': 'constant_tcp_attachment',
            'grasp_attachment': {'requires_full_ik_and_collision_admission': True},
            'robot_initialization': {'admitted': True, 'admission_checks': {},
                                     'completed_frames': 0, 'total_frames': 0}})


def test_cache_requires_current_path_admission_after_inserted_placement():
    from copy import deepcopy
    original = {'admitted': True, 'admission_checks': {'complete': True, 'collision': True},
                'completed_frames': 100, 'total_frames': 100}
    details = {'robot_initialization': original, 'robot_control_retiming': {'duration_s': 2.}}
    with pytest.raises(ValueError, match='current reference'):
        validate_cache_admission(details, 200)
    details['robot_supported_placement'] = dict(deepcopy(original), completed_frames=200, total_frames=200)
    validate_cache_admission(details, 200)
    assert details['robot_initialization']['completed_frames'] == 100
    details['robot_supported_placement']['completed_frames'] = 199
    with pytest.raises(ValueError, match='current reference'):
        validate_cache_admission(details, 200)
    details['robot_supported_placement']['completed_frames'] = 200
    details['robot_grasp_approach'] = dict(deepcopy(original), admitted=False)
    with pytest.raises(ValueError, match='robot_grasp_approach'):
        validate_cache_admission(details, 200)


def test_admitted_cache_roundtrip_preserves_reset_and_references(monkeypatch, tmp_path):
    import json
    import numpy as np
    from reachy_retarget import feasible_maniskill, frozen_plan
    from reachy_retarget.dynamics import Candidate
    from reachy_retarget.episodes import write_episode
    from reachy_retarget.store import Store
    from test_feasible_maniskill import fixture, Robot
    original = fixture(monkeypatch)
    original[5].update(pad_alignment={}, alignment_variant='pad_translation')
    robot = Robot()
    prepared = feasible_maniskill.prepare(original, robot, tmp_path/'admission', [.2, .3, 0])
    monkeypatch.setattr(frozen_plan, 'initialize', feasible_maniskill.initialize)
    source = write_episode(Store(tmp_path/'input'), 'test-source', 'full-episode',
                           {'time_s': original[7], 'objects/cube/pose': np.tile([8,0,1,1,0,0,0], (3,1))},
                           {'objects': {'cube': {'body':'cube'}}})
    folder = frozen_plan.save_admitted(prepared, Candidate(), source, robot, tmp_path/'frozen')
    destination = tmp_path/'load'; destination.mkdir()
    _, loaded = frozen_plan.load(folder, source, robot, destination)
    for index in (6,7,8,9,10,11):
        np.testing.assert_array_equal(loaded[index], prepared[index])
    report = json.loads((folder/'result.json').read_text())
    assert not report['physics_tested'] and not report['physics_validated']
    assert report['status'] == 'kinematic_candidate'
    source.write_bytes(b'tampered source')
    with pytest.raises(ValueError, match='different source'):
        frozen_plan.load(folder, source, robot, destination)
