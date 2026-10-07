"""Command pauses must preserve the original denominator and measured guards."""
import json

import h5py
import numpy as np
import pytest

from reachy_retarget.acquisition_clock import AcquisitionClock
from reachy_retarget.dynamics_audit import _load_acquisition
from reachy_retarget.store import sha256


def recording(tmp_path, fail=False):
    rows = 12
    metadata = dict(schema='cylindrical-acquisition-v1',first_close_index=2, acquisition_index=4, timestep_s=.01,
                    timeout_s=.2, stable_s=.1)
    artifact = tmp_path/'acquisition-plan.npz'
    original_times = np.arange(rows)*.01
    original_intent = np.where(np.arange(rows) < 2, 2., -.06)
    original_hands = np.tile(np.eye(4), (rows, 1, 1))
    np.savez(artifact, entry_reference=np.zeros((3, 17)), exit_source_indices=np.array([5, 6]),
             original_time_s=original_times, original_gripper_intent=original_intent,
             original_reference=np.zeros((rows, 17)), original_hand_goals=original_hands,
             original_object_goals=original_hands)
    metadata.update(artifact=str(artifact), artifact_sha256=sha256(artifact))
    clock = AcquisitionClock(source_rows=rows, first_close_index=2, acquisition_index=4,
        entry_rows=3, exit_source_indices=[5, 6], wait_limit_s=.2)
    frames, guards = [], []
    for frame in clock.frames():
        frames.append(frame)
        guard = {}
        if frame.phase in ('settle', 'close'):
            guard = dict(alignment=True, bilateral_force=not fail)
            clock.observe(**guard)
        guards.append(json.dumps(guard))
    indices = np.array([f.reference_index for f in frames])
    times = np.arange(rows)*.01
    intent = np.where(np.arange(rows) < 2, 2., -.06)
    with h5py.File(tmp_path/'plan.h5', 'w') as f:
        f['time_s'] = times; f['original_gripper_intent'] = intent; f['gripper'] = intent
        f['reference'] = np.zeros((rows, 17))
        from reachy_retarget.episodes import matrices_to_pose
        f['hand_goals'] = matrices_to_pose(original_hands)
        f['object_goals'] = matrices_to_pose(original_hands)
    with h5py.File(tmp_path/'replay.h5', 'w') as f:
        f['source/reference_index'] = indices
        f['source/reference_time_s'] = times[indices]
        f['source/original_gripper_intent'] = intent[indices]
        f['command/effective_gripper_intent'] = [2. if row.phase in ('delayed_open', 'entry', 'settle') else intent[row.reference_index] for row in frames]
        f.create_dataset('acquisition/phase', data=[row.phase for row in frames], dtype=h5py.string_dtype())
        f.create_dataset('acquisition/guard_json', data=guards, dtype=h5py.string_dtype())
        f['metrics/bilateral_contact'] = np.ones(len(frames))
    report = dict(plan={'cylindrical_acquisition': metadata}, acquisition_execution=clock.report(),
                  gates={'rollout_complete': not fail, 'measured_acquisition_before_carry': not fail})
    return report, len(frames)


def test_full_and_failed_prefix_are_auditable_without_false_success(tmp_path):
    report, count = recording(tmp_path)
    proof = _load_acquisition(tmp_path, report, count)
    assert proof['complete'] and proof['rows'] == 12 and count > 12
    report, count = recording(tmp_path, fail=True)
    proof = _load_acquisition(tmp_path, report, count)
    assert not proof['complete'] and proof['rows'] == 12
    report['gates']['rollout_complete'] = True
    with pytest.raises(ValueError, match='coverage'):
        _load_acquisition(tmp_path, report, count)


def test_box_geometry_uses_the_same_complete_source_clock_audit(tmp_path):
    report,count=recording(tmp_path)
    metadata=report['plan'].pop('cylindrical_acquisition')
    metadata['schema']='box-acquisition-v1'
    report['plan']['box_acquisition']=metadata
    assert _load_acquisition(tmp_path,report,count)['complete']
    with h5py.File(tmp_path/'replay.h5','r+') as f:
        f['source/reference_index'][1]=3
    with pytest.raises(ValueError,match='skips'):
        _load_acquisition(tmp_path,report,count)


@pytest.mark.parametrize('mutation,reason', [('skip', 'skips'), ('phase', 'bypasses'),
    ('close_early', 'Effective gripper'), ('intent', 'original intent'), ('time', 'source clock')])
def test_tampered_acquisition_mapping_or_early_closure_is_rejected(tmp_path, mutation, reason):
    report, count = recording(tmp_path)
    with h5py.File(tmp_path/'replay.h5', 'r+') as f:
        if mutation == 'skip': f['source/reference_index'][1] = 3
        if mutation == 'phase': f['acquisition/phase'][7] = 'close'
        if mutation == 'close_early': f['command/effective_gripper_intent'][2] = -.06
        if mutation == 'intent': f['source/original_gripper_intent'][0] = -.06
        if mutation == 'time': f['source/reference_time_s'][2] = 7.
    with pytest.raises(ValueError, match=reason):
        _load_acquisition(tmp_path, report, count)


@pytest.mark.parametrize('phase', ['source', 'exit'])
def test_other_phase_intent_cannot_silently_change(tmp_path, phase):
    report,count=recording(tmp_path)
    with h5py.File(tmp_path/'replay.h5','r+') as f:
        i=f['acquisition/phase'].asstr()[()].tolist().index(phase)
        f['command/effective_gripper_intent'][i]=-.06 if phase=='source' else 2.
    with pytest.raises(ValueError,match='Effective gripper'):
        _load_acquisition(tmp_path,report,count)


@pytest.mark.parametrize('field', ['time_s','original_gripper_intent','reference','hand_goals','object_goals'])
def test_prepared_plan_denominator_and_originals_are_hash_bound(tmp_path,field):
    report,count=recording(tmp_path)
    with h5py.File(tmp_path/'plan.h5','r+') as f:
        values=f[field][()]; values.flat[-1]+=.1; f[field][:]=values
    with pytest.raises(ValueError,match='bound acquisition originals'):
        _load_acquisition(tmp_path,report,count)


def test_guard_cannot_count_twice_as_much_stability_as_physical_time(tmp_path):
    report,count=recording(tmp_path)
    report['plan']['cylindrical_acquisition']['timestep_s']=.02
    with pytest.raises(ValueError,match='100 Hz'):
        _load_acquisition(tmp_path,report,count)
