"""Execute existing source episodes through baseline IK and measured dynamics.

No source acquisition, timing search, geometry scaling or synthesized observations
is performed here. Every stage is journalled independently so a late failure does
not erase an earlier result. Each job requires an isolated, unused workspace.
"""
import json
from pathlib import Path
import traceback

import h5py
import numpy as np

from .agent_dataset import write_archive, assess_state_observations
from .cluster_pipeline import atomic, identifier, under
from .episodes import read_episode
from .store import Store, sha256


def _retarget(store, row):
    from .retarget import one
    return one(store, row)


def _paired_plans(store, row, output):
    from .pad_alignment import paired_plans
    from .robot import Robot
    return paired_plans(store, row, Robot(store.root), output, state_profile=True)


def _rollout(store, row, label, candidate, plan):
    from .dynamics import rollout
    return rollout(store, row, label, candidate, prepared_plan=plan, state_profile=True)


def _audit(attempt):
    from .dynamics_audit import verify
    return verify(attempt)


def _task_objects():
    from .task_contracts import TASK_OBJECTS
    return TASK_OBJECTS


def _arrays(path):
    values = {}
    with h5py.File(path, 'r') as handle:
        def read(name, value):
            if isinstance(value, h5py.Dataset):
                values[name] = value.asstr()[()] if value.dtype.kind in 'OUS' else value[()]
        handle.visititems(read)
        metadata = json.loads(handle.attrs.get('metadata_json', '{}'))
    return values, metadata


def _kinematic_archive(root, job, source, metadata, motion, report):
    """The baseline motion is a derived target, never a measured observation."""
    values, _ = _arrays(motion)
    original, _ = read_episode(source)
    arrays = {'timestamp': values.pop('time_s'), 'source/time_s': original['time_s']}
    for name, value in values.items():
        if name.endswith('/rgb'):
            continue
        if name.startswith('robot/'):
            key = 'derived/' + name.removeprefix('robot/')
        elif name.startswith('target/'):
            key = 'reference/target/' + name.removeprefix('target/')
        elif name.startswith('objects/'):
            key = 'reference/' + name
        elif name.startswith('derived/'):
            key = name
        else:
            key = 'derived/motion/' + name
        arrays[key] = value
    from .state_recording import _source_provenance
    provenance = _source_provenance(metadata, job.get('source_provenance'))
    object_names = {name.split('/')[1] for name in original
                    if name.startswith('objects/') and name.endswith('/pose')}
    declared = metadata.get('objects', {})
    objects = {name: dict(declared.get(name, {}), state_semantics='source reference; not measured Reachy rollout')
               for name in sorted(object_names)}
    meta = dict(provenance, source_sequence=metadata['source_sequence'],
                source_group=metadata['source_group'], source_episode_id=metadata['episode_id'],
                source_metadata=metadata, objects=objects, robot_type='Reachy2',
                source_normalized_hdf5_sha256=sha256(source), motion_hdf5_sha256=sha256(motion),
                source_normalized_path=str(source), motion_path=str(motion),
                status=report.get('status', 'partial_kinematic_artifact'), kinematic_report=report,
                physics_validated=False,
                missing_fields=['measured Reachy state', 'applied_controls', 'physics_state', 'controller_state_json'],
                derived_fields={'derived/*': 'Existing retarget.one IK and gripper/neck mapping; not observations',
                                'reference/*': 'Existing baseline reference trajectories under the recorded rigid placement',
                                'timestamp': 'Existing baseline 100 Hz resampling and documented global velocity dilation',
                                'source/time_s': 'Unchanged original source timestamps'},
                simulation_assumptions=list(metadata.get('simulation_assumptions', [])) + [
                    'Kinematic output does not establish collision/contact or physical task success.',
                    'No new optimization or timing adjustment beyond the existing retarget.one baseline.',
                    'Existing baseline world placement and time dilation are retained in kinematic_report.'])
    return write_archive(root, job['dataset'], job['id'] + '-ik', arrays, meta)


def run(root, job):
    """Run one isolated legacy job and retain each completed or failed stage."""
    root = Path(root)
    identifier(job['id']); identifier(job['dataset'])
    workspace = under(root, job['workspace'])
    source = under(workspace, job['source'])
    output = workspace / 'runs/legacy-pipeline' / job['id']
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', dataset=job['dataset'], source=str(source),
                  archives={}, stages={}, missing=[], retargeted=False,
                  kinematic_report=None, physics_report=None, physics_validated=False,
                  actuator_replay_pass=False, audit=None, diagnostic_baseline=False)

    def save():
        atomic(output / 'result.json', result)

    def failed(stage, exc):
        value = dict(status='failed', error=type(exc).__name__ + ': ' + str(exc),
                     traceback=traceback.format_exc())
        result['stages'][stage] = value
        result['missing'].append(stage + ': ' + value['error'])
        atomic(output / (stage + '-failed.json'), value)
        save()

    def finish(status):
        result['status'] = status
        result['pipeline_report'] = str(output / 'result.json')
        save()
        return result

    save()
    try:
        _, metadata = read_episode(source)
        if metadata.get('hdf5_sha256') and metadata['hdf5_sha256'] != sha256(source):
            raise ValueError('Normalized source checksum changed')
        row = dict(id=identifier(metadata['episode_id']), source_id=identifier(metadata['source_id']),
                   source_sequence=metadata['source_sequence'], split=metadata['split'],
                   path=job['source'], source_provenance=job.get('source_provenance'))
        if not metadata.get('source_group'):
            raise ValueError('Original source_group is required')
        motion_dir = workspace / 'data/retargeted' / row['source_id'] / row['id']
        if motion_dir.exists():
            raise FileExistsError('Isolated workspace must not contain a preexisting retarget: ' + str(motion_dir))
        store = Store(workspace)
        result['stages']['source'] = dict(status='verified', sha256=sha256(source),
                                        source_group=metadata['source_group'])
        save()
    except Exception as exc:
        failed('source', exc)
        return finish('source_failed')

    try:
        result['kinematic_report'] = _retarget(store, row)
        result['retargeted'] = True
        result['stages']['ik'] = dict(status='completed', result=result['kinematic_report']['status'])
        save()
    except Exception as exc:
        failed('ik', exc)
    motion = motion_dir / 'motion.h5'
    if motion.exists():
        result['motion_artifact'] = str(motion)
        try:
            report = result['kinematic_report'] or {'status': 'partial_kinematic_artifact'}
            result['archives']['kinematic'] = str(_kinematic_archive(root, job, source, metadata, motion, report))
            result['stages']['kinematic_export'] = {'status': 'completed'}
            save()
        except Exception as exc:
            failed('kinematic_export', exc)
    if not result['retargeted']:
        return finish('ik_failed')

    task = metadata.get('env_args', {}).get('env_name')
    if task not in _task_objects():
        blocker = ('No verified source scene, contact/task predicate and measured dynamics adapter for '
                   + str(task or metadata.get('source_format', 'unknown source')))
        result['stages']['physics'] = {'status': 'blocked_unsupported_task', 'reason': blocker}
        result['missing'].append(blocker)
        return finish('retargeted_physics_blocked')
    plans = workspace / 'plans' / job['id']
    try:
        plans.mkdir(parents=True, exist_ok=False)
        candidate, before, after = _paired_plans(store, row, plans)
        selected = after if after is not None else before
        result['diagnostic_baseline'] = after is None
        result['alignment_variant'] = 'uncorrected_diagnostic_baseline' if after is None else 'pad_translation'
        rejection = plans / 'alignment-rejected.json'
        result['stages']['pad_alignment'] = dict(status='rejected' if after is None else 'completed',
            artifacts=str(plans), rejection=json.loads(rejection.read_text()) if rejection.exists() else None)
        if after is None:
            result['missing'].append('Pad alignment rejected; uncorrected baseline is diagnostic only')
        save()
    except Exception as exc:
        failed('pad_alignment', exc)
        return finish('retargeted_plan_failed')

    label = job['id'] + ('-baseline' if result['diagnostic_baseline'] else '-aligned')
    attempt = workspace / 'runs/dynamics' / label / row['id']
    result['physics_attempt'] = str(attempt)
    try:
        physics = _rollout(store, row, label, candidate, selected)
        result['physics_report'] = physics
        result['stages']['physics'] = dict(status='completed', result=physics['status'])
        save()
    except Exception as exc:
        failed('physics', exc)
        # Recover a completed stage report if the simulator failed later.
        if (attempt / 'result.json').exists():
            result['physics_report'] = json.loads((attempt / 'result.json').read_text())
        physics = result['physics_report'] or {}

    if (attempt / 'replay.h5').exists():
        try:
            result['audit'] = _audit(attempt)
            result['actuator_replay_pass'] = bool(result['audit']['actuator_replay_pass'])
            result['stages']['actuator_audit'] = {'status': 'completed'}
            atomic(output / 'actuator-audit.json', result['audit'])
            save()
        except Exception as exc:
            failed('actuator_audit', exc)
    else:
        result['stages']['actuator_audit'] = {'status': 'blocked_no_replay'}
        result['missing'].append('Actuator replay artifact unavailable')
    result['physics_task_pass'] = bool(physics.get('physics_validated', False))
    result['physics_validated'] = result['physics_task_pass'] and result['actuator_replay_pass']

    recording = physics.get('state_recording', {})
    recorded = recording.get('archive') or recording.get('path')
    if not recorded and (attempt / 'state-attempt.hdf5').exists():
        recorded = str(attempt / 'state-attempt.hdf5')
    if recorded:
        result['state_recording_artifact'] = recorded
        try:
            arrays, archive_meta = _arrays(recorded)
            if any(name.endswith('/rgb') for name in arrays):
                raise ValueError('State-only legacy execution unexpectedly recorded RGB')
            assessment = assess_state_observations({'arrays': arrays, 'metadata': archive_meta})
            result['state_assessment'] = assessment
            archive_meta.update(diagnostic_baseline=result['diagnostic_baseline'],
                                alignment_variant=result['alignment_variant'],
                                physics_task_pass=result['physics_task_pass'],
                                physics_validated=result['physics_validated'],
                                actuator_replay_audit=result['audit'])
            result['archives']['physics'] = str(write_archive(root, job['dataset'], job['id'] + '-physics', arrays, archive_meta))
            result['stages']['physics_export'] = {'status': 'completed'}
            save()
        except Exception as exc:
            failed('physics_export', exc)
    else:
        result['missing'].append('Measured state recording unavailable')
        result['stages']['physics_export'] = {'status': 'blocked_no_recording'}
    return finish('physical_pass' if result['physics_validated'] else
                  'physical_rollout_recorded' if 'physics' in result['archives'] else 'retargeted_physics_incomplete')
