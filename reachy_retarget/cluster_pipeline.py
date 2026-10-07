"""Explicit cluster acquisition, adapter execution, and durable agent decisions.

Importing this module never performs network access or starts a simulator.
All transfers and mutations require a queued, immutable task. The persistent
worker is a queue consumer, not an autonomous substitute for an image agent.
"""
import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import traceback
import uuid
import zlib

DATASETS = ('behavior', 'momagen', 'm3bench', 'mello')
RESERVE = 50_000_000_000
PROTOCOL = 'reachy-retarget-agent-v1'


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('w') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('Unsafe identifier')
    return value


def under(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Expected a safe relative path')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path escapes dataset root')
    return path


def adapter(dataset):
    if dataset not in DATASETS:
        raise ValueError('Unknown native adapter: ' + dataset)
    return importlib.import_module('reachy_retarget.adapters.' + dataset)


def enqueue(root, job):
    """Create once; never overwrite an existing job or its attempts."""
    root = Path(root)
    task = identifier(job['id'])
    folder = root / 'queue/tasks' / task
    folder.parent.mkdir(parents=True, exist_ok=True)
    if folder.exists():
        if json.loads((folder / 'job.json').read_text()) != job:
            raise ValueError('Task ID already names a different immutable job')
        return folder
    # Keep unpublished jobs outside the consumer's */job.json scan. Even a
    # hidden directory matches pathlib.glob; renaming it after discovery made
    # workers open a vanished claim.lock and restart during batch submission.
    staging = root / 'queue/staging'
    staging.mkdir(parents=True, exist_ok=True)
    temp = staging / (task + '-' + uuid.uuid4().hex)
    temp.mkdir()
    atomic(temp / 'job.json', job)
    try:
        os.rename(temp, folder)
    except OSError:
        if folder.exists() and json.loads((folder / 'job.json').read_text()) == job:
            shutil.rmtree(temp)
        else:
            raise
    return folder


def _verify(path, spec):
    expected = spec.get('bytes', spec.get('size'))
    if expected is not None and path.stat().st_size != expected:
        raise ValueError('Source size does not match pinned manifest')
    sha = digest(path)
    if spec.get('sha256') and sha != spec['sha256']:
        raise ValueError('Source SHA256 does not match manifest')
    blob = spec.get('git_blob_sha1', spec.get('git_blob'))
    if blob:
        h = hashlib.sha1(('blob ' + str(path.stat().st_size) + '\0').encode())
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        if h.hexdigest() != blob:
            raise ValueError('Source Git blob SHA1 does not match pinned commit')
    return sha


def fetch(root, dataset, spec):
    """Explicit bounded transfer; cluster-wide lock keeps the 50 GB reserve.

    Download serialization is intentional: process IDs are not shared between
    hosts. Local-PID reservations cannot safely enforce a shared CephFS budget.
    Computation and offline adapters run concurrently across all workers.
    """
    import requests
    root = Path(root)
    path = under(root / 'data/raw' / identifier(dataset), spec['path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = spec.get('bytes', spec.get('size'))
    if not isinstance(expected, int) or expected < 1:
        raise ValueError('Explicit positive payload size is required before transfer')
    lockpath = root / 'queue/transfer.lock'
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            sha = _verify(path, spec)
        else:
            if shutil.disk_usage(root).free - expected < RESERVE:
                raise RuntimeError('50 decimal GB reserve prevents this transfer')
            temp = path.with_name(path.name + '.part-' + uuid.uuid4().hex)
            headers = {'Accept-Encoding': 'identity'}
            byte_range = spec.get('range')
            if byte_range:
                offset, length = byte_range['offset'], byte_range['length']
                headers['Range'] = f'bytes={offset}-{offset + length - 1}'
            with requests.get(spec['url'], headers=headers, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                if byte_range:
                    if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-{offset+length-1}/'):
                        raise ValueError('Server ignored or changed exact byte-range request')
                    payload = response.raw.read(length + 1)
                    if len(payload) != length:
                        raise ValueError('Compressed member length mismatch')
                    if byte_range.get('compression') == 8:
                        dec = zlib.decompressobj(-15)
                        payload = dec.decompress(payload, expected + 1)
                        if len(payload) > expected or not dec.eof:
                            raise ValueError('Oversized/incomplete deflate member')
                    elif byte_range.get('compression') != 0:
                        raise ValueError('Unsupported ZIP member compression')
                    temp.write_bytes(payload)
                else:
                    total = 0
                    with temp.open('xb') as output:
                        for chunk in response.iter_content(1024 * 1024):
                            total += len(chunk)
                            if total > expected or shutil.disk_usage(root).free - len(chunk) < RESERVE:
                                raise RuntimeError('Transfer exceeded size or disk reserve; partial retained')
                            output.write(chunk)
            sha = _verify(temp, spec)
            os.rename(temp, path)
        receipt = dict(spec, sha256=sha, downloaded_bytes=path.stat().st_size,
                       retargeted=False, physics_validated=False)
        atomic(path.with_name(path.name + '.receipt.json'), receipt)
    return path


def request_agent(folder, dataset, artifacts, summary, allowed):
    request_path = folder / 'agent/request.json'
    if request_path.exists():
        return json.loads(request_path.read_text())
    evidence = {str(path): digest(path) for path in artifacts}
    value = dict(protocol=PROTOCOL, task_id=folder.name, request_id=uuid.uuid4().hex,
                 dataset=dataset, stage='adapter_review', evidence_sha256=evidence,
                 summary=summary, allowed_decisions=allowed,
                 instructions='Review source fields, missing information, object scale and grasp assumptions. '
                 'Select an allowed bounded action and record evidence-based observations. '
                 'No arbitrary shell command, invented geometry or automatic success is accepted.')
    atomic(request_path, value)
    return value


def validate_response(request, response):
    for key in ('protocol', 'task_id', 'request_id'):
        if response.get(key) != request.get(key):
            raise ValueError('Stale/mismatched agent response: ' + key)
    if response.get('decision') not in request['allowed_decisions']:
        raise ValueError('Decision is not permitted for this adapter state')
    if response.get('evidence_sha256') != request['evidence_sha256']:
        raise ValueError('Agent did not acknowledge exact evidence hashes')
    if not isinstance(response.get('observation'), str) or not response['observation'].strip():
        raise ValueError('Evidence-based agent observation is required')
    if not response.get('reviewer'):
        raise ValueError('Reviewer identity is required')
    for path, sha in request['evidence_sha256'].items():
        if digest(path) != sha:
            raise ValueError('Evidence changed since the request was issued')


def _archive_source(record, root, dataset, task_id):
    """Resume only when an immutable archive describes the same source record."""
    from .agent_dataset import export_normalized, inspect_archive
    archive = root / 'datasets' / identifier(dataset) / 'episodes' / (identifier(task_id) + '.hdf5')
    if archive.exists():
        inspect_archive(archive)
        previous = json.loads(archive.with_suffix('.json').read_text())['source_metadata']
        expected = dict(record['metadata'], status=record['status'],
                        missing_fields=record.get('missing_fields', []))
        if previous != expected:
            raise ValueError('Immutable normalized source differs on resume; enqueue a new versioned task')
    else:
        archive = export_normalized(record, root, dataset, task_id)
    return archive


def _native(folder, root, job, attempt):
    dataset = job['dataset']
    module = adapter(dataset)
    description = module.describe()
    atomic(attempt / 'description.json', description)
    paths = [fetch(root, dataset, spec) for spec in job['fetch']]
    source = paths[job.get('source_index', 0)]
    if job.get('source_directory'):
        source = under(root / 'data/raw' / dataset, job['source_directory'])
    kwargs = {'episode': job['episode']} if job.get('episode') else {}
    inspected = module.inspect(source)
    atomic(attempt / 'inspection.json', inspected)
    record = module.normalize(source, **kwargs)
    # Always save the explicit native result before attempting canonical export.
    atomic(attempt / 'normalization.json', {k:v for k,v in record.items() if k != 'arrays'})
    archive = _archive_source(record, root, dataset, job['id'])
    artifacts = [archive, archive.with_suffix('.json'), attempt / 'inspection.json']
    if job.get('planning_policy') is not None:
        # Bind task-hand semantics, constraints and seed to the reviewed job,
        # rather than acknowledging only the unchanged normalized source.
        artifacts.append(folder / 'job.json')
    can_ik = (dataset == 'momagen' and 'source/robot_root_pose' in record['arrays']
              and all('hand/' + s + '_pose' in record['arrays'] for s in ('left','right')))
    request = request_agent(folder, dataset, artifacts,
        dict(status=record['status'], missing_fields=record.get('missing_fields', []),
             proposed_method=('declared_task_hands_mobile_ik' if job.get('planning_policy')
                              else 'source_base_and_bimanual_ik') if can_ik else 'native_reconstruction_required',
             planning_policy=job.get('planning_policy'),
             physics_validated=False),
        ['run_kinematic', 'defer', 'abort'] if can_ik else ['defer', 'abort'])
    response_path = folder / 'agent/response.json'
    result = dict(status='waiting_for_agent', dataset=dataset, archive=str(archive),
                  request=str(folder / 'agent/request.json'), normalized=True,
                  retargeted=False, physics_validated=False, native_status=record['status'])
    if not response_path.exists():
        return result
    try:
        response = json.loads(response_path.read_text())
        validate_response(request, response)
    except (ValueError, KeyError, TypeError) as error:
        result.update(response_sha256=digest(response_path), rejection=str(error))
        atomic(attempt / 'response-rejected.json', result)
        return result
    atomic(attempt / 'agent-accepted.json', response)
    if response['decision'] == 'run_kinematic':
        from .mobile_retarget import retarget_momagen
        result.update(retarget_momagen(record, root, dataset, job['id'], attempt,
                                     planning_policy=job.get('planning_policy')))
    else:
        result.update(status='blocked_prerequisites' if response['decision']=='defer' else 'aborted',
                      reason=response['observation'])
    return result


def execute(root, folder, attempt):
    root, folder, attempt = Path(root), Path(folder), Path(attempt)
    with (folder / 'execution.lock').open('a') as lock:
        # The executor owns this independently of the queue-consumer parent.
        # A surviving child retains the lock if its worker is interrupted.
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = folder / 'result.json'
        if existing.exists():
            prior = json.loads(existing.read_text())
            if prior.get('status') != 'waiting_for_agent':
                return prior
        return _execute_locked(root, folder, attempt)


def _execute_locked(root, folder, attempt):
    job = json.loads((folder / 'job.json').read_text())
    try:
        if job['operation'] == 'native_pipeline':
            result = _native(folder, root, job, attempt)
        elif job['operation'] == 'export_normalized':
            from .agent_dataset import export_normalized
            source = under(root, job['source'])
            path = export_normalized(source, root, job['dataset'], job['id'])
            result = dict(status='archived', archive=str(path), retargeted=False, physics_validated=False)
        elif job['operation'] in ('mobile_pilot', 'normalize_mobile'):
            from .mobile_pilot import discover, fetch as fetch_mobile, inspect
            from .adapters.native_mobile import normalize
            if job['dataset'] not in ('robocasa', 'bigym'):
                raise ValueError('Unsupported mobile pilot')
            task_root = root / 'native' / job['dataset']
            if job['operation'] == 'mobile_pilot':
                # This legacy fetcher has a process-local disk ledger. Hold the
                # cluster transfer lock as well to enforce the shared reserve.
                lockpath = root / 'queue/transfer.lock'
                lockpath.parent.mkdir(parents=True, exist_ok=True)
                with lockpath.open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    discover(task_root, job['dataset'])
                    fetch_mobile(task_root, job['dataset'])
            inspected = inspect(task_root, job['dataset'])
            atomic(attempt / 'inspection.json', inspected)
            record = normalize(task_root, job['dataset'])
            atomic(attempt / 'normalization.json', {k:v for k,v in record.items() if k != 'arrays'})
            archive = _archive_source(record, root, job['dataset'], job['id'])
            request = request_agent(folder, job['dataset'],
                [archive, archive.with_suffix('.json'), attempt / 'inspection.json'],
                dict(status=record['status'], missing_fields=record['missing_fields'],
                     proposed_method='pinned_native_replay_before_retargeting', physics_validated=False),
                ['defer', 'abort'])
            result = dict(status='waiting_for_agent', archive=str(archive), inspection=inspected,
                          normalized=True, native_status=record['status'],
                          missing_fields=record['missing_fields'],
                          retargeted=False, physics_validated=False)
            response_path = folder / 'agent/response.json'
            if response_path.exists():
                try:
                    response = json.loads(response_path.read_text())
                    validate_response(request, response)
                except (ValueError, KeyError, TypeError) as error:
                    result.update(response_sha256=digest(response_path), rejection=str(error))
                    atomic(attempt / 'response-rejected.json', result)
                else:
                    atomic(attempt / 'agent-accepted.json', response)
                    result.update(status='blocked_prerequisites' if response['decision']=='defer' else 'aborted',
                                  reason=response['observation'])
        elif job['operation'] == 'full_state_rollout':
            result = full_state_rollout(root, job)
        elif job['operation'] == 'legacy_pipeline':
            from .legacy_pipeline import run
            result = run(root, job)
        elif job['operation'] in ('feasible_experiment','feasible_geometry_experiment'):
            from .feasible_experiments import run
            result = run(root, job)
        elif job['operation'] == 'feasible_geometry_search':
            from .feasible_geometry_search import run
            result = run(root, job)
        elif job['operation'] == 'feasible_placement_search':
            from .feasible_placement_search import run
            result = run(root, job)
        elif job['operation'] == 'bigym_physics_diagnostic':
            from .native_replay.bigym_physics import rollout
            result = rollout(root, under(root,job['source_replay']), under(root,job['native_assets']),
                             under(root,job['ik_attempt']), job['id'], attempt,timing=job.get('timing'),
                             simulation_profile=job.get('simulation_profile'))
            result.update(diagnostic_only=True, physics_validated=False,
                          validation_scope='Native scene task/contact diagnostic; common dynamics gates and source reproduction remain unresolved')
        elif job['operation'] == 'bigym_native_replay':
            import subprocess
            from .native_replay.bigym import normalize
            output=attempt/'native-replay'
            command=[str(under(root,job['native_python'])),'-m','reachy_retarget.native_replay.bigym',
                     '--pilot-root',str(under(root,job['pilot_root'])),
                     '--source-root',str(under(root,job['native_source_root'])),
                     '--output-dir',str(output)]
            with (attempt/'native-replay.log').open('w') as log:
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
            record=normalize(output)
            clock=record['arrays'].get('timestamp',[])
            archive=(_archive_source(record,root,'bigym',job['id']) if len(clock)>=2 else None)
            result=dict(status=record['status'],source_task_success=record['metadata']['source_task_success'],
                        native_returncode=completed.returncode,source_replay=str(output),
                        source_error=record['metadata'].get('source_error'),
                        archive=None if archive is None else str(archive),
                        normalized=archive is not None,retargeted=False,physics_validated=False,
                        missing_fields=record['missing_fields'])
        elif job['operation'] == 'reconstruct_robocasa':
            from .native_replay.robocasa import reconstruct
            record = reconstruct(root / 'native/robocasa')
            atomic(attempt / 'reconstruction.json', {k:v for k,v in record.items() if k != 'arrays'})
            archive = _archive_source(record, root, 'robocasa', job['id'])
            result = dict(status=record['status'], archive=str(archive), normalized=True,
                          missing_fields=record['missing_fields'], retargeted=False,
                          physics_validated=False, source_reconstruction='verified_fk_only')
        elif job['operation'] == 'source_pose_feasible':
            from .feasible_pose import retarget
            grips=None
            if job['dataset']=='bigym':
                from .native_replay.bigym import normalize
                from .native_replay.bigym_retarget import prepare, BASE_KEY, HAND_KEYS
                record=normalize(under(root,job['source_replay']))
                grips,mapping=prepare(record)
                base_key,hand_keys=BASE_KEY,HAND_KEYS
                mapping['base']='Derived bounded mobile robot placement relative to unchanged source chassis trajectory'
                mapping['hands']={side:dict(source=key,mapping='Native approach/closing axes to calibrated Reachy pad midpoint; constant tool attachment') for side,key in hand_keys.items()}
            elif job['dataset']=='robocasa':
                from .native_replay.robocasa import reconstruct
                record=reconstruct(root/'native/robocasa')
                base_key='source/mobile_base_body_pose';hand_keys={'right':'source/right_tcp_pose'}
                mapping=dict(clock='Unchanged source simulator timestamps',base='Derived bounded mobile robot placement relative to recorded chassis',
                             hands='Verified native grip axes to calibrated Reachy pad midpoint',gripper='No target invented; original source actions preserved')
            else:raise ValueError('Unsupported geometric pose adapter')
            result=retarget(record,root,job['dataset'],job['id'],attempt,clock_key='timestamp',base_key=base_key,
                            hand_keys=hand_keys,gripper_targets=grips,mapping_metadata=mapping,**job.get('solver_settings',{}))
        elif job['operation'] == 'source_pose_ik':
            from .pose_retarget import retarget
            if job['dataset'] == 'bigym':
                from .native_replay.bigym import normalize
                from .native_replay.bigym_retarget import retarget as retarget_bigym
                record = normalize(under(root,job['source_replay']))
                result = retarget_bigym(record,root,'bigym',job['id'],attempt)
            elif job['dataset'] == 'robocasa':
                from .native_replay.robocasa import reconstruct
                record = reconstruct(root/'native/robocasa')
                base_key = 'source/mobile_base_body_pose'
                hand_keys = {'right':'source/right_tcp_pose'}
                mapping = dict(clock='Unchanged measured source simulator clock including gaps',
                               base='Recorded mobile chassis body XY/yaw; no source torso height substituted',
                               tcp='Verified right grip site for both position and orientation',
                               gripper='Original source actions retained in source archive; no inferred target in this diagnostic IK')
            elif job['dataset'] == 'd4rl_kitchen':
                from .adapters.d4rl_kitchen import normalize
                if job.get('nominal_timing') is not True:
                    raise ValueError('D4RL IK requires an explicit nominal-time reference policy')
                record = normalize(root/'native/d4rl_kitchen',job['variant'],episode=job['episode'],nominal_timing=True)
                base_key = 'source/base_pose'
                hand_keys = {'right':'source/right_tcp_pose_estimate'}
                mapping = dict(clock='Derived nominal 0.08 s interval from pinned 0.002 s timestep and frame_skip=40; not measured',
                               base='Fixed Panda mounting body XY/yaw; source mounting height retained in hand references',
                               tcp='FK estimate from noisy recorded configuration; exact source qpos is unavailable',
                               gripper='No source finger target synthesized from noisy observations',
                               workspace='No vertical workspace translation; unreachable source heights remain failures')
            else:
                raise ValueError('Unsupported source pose IK adapter')
            if job['dataset'] != 'bigym':
                result = retarget(record,root,job['dataset'],job['id'],attempt,clock_key='timestamp',
                                  base_key=base_key,hand_keys=hand_keys,mapping_metadata=mapping)
        else:
            raise ValueError('Unknown queued operation')
    except Exception as error:
        result = dict(status='failed', error=type(error).__name__ + ': ' + str(error),
                      traceback=traceback.format_exc(), retargeted=False, physics_validated=False)
    result.update(task_id=folder.name, finished_unix=time.time(), attempt=str(attempt))
    atomic(attempt / 'result.json', result)
    atomic(folder / 'result.json', result)
    print(json.dumps(result, indent=2))
    return result


def full_state_rollout(root, job):
    """Execute a frozen-source, translation-only pad plan with complete states."""
    import h5py
    from .store import Store
    from .robot import Robot
    from .episodes import read_episode
    from .pad_alignment import paired_plans
    from .dynamics import rollout
    from .dynamics_audit import verify
    from .agent_dataset import write_archive, assess_state_observations
    workspace = under(root, job['workspace'])
    store = Store(workspace)
    source = under(workspace, job['source'])
    _, metadata = read_episode(source)
    row = dict(id=metadata['episode_id'], source_id=metadata['source_id'],
               source_sequence=metadata['source_sequence'], path=job['source'],
               source_provenance=job.get('source_provenance'))
    plans = workspace / 'plans' / job['id']
    plans.mkdir(parents=True, exist_ok=False)
    candidate, before, after = paired_plans(store, row, Robot(workspace), plans, state_profile=True)
    if after is None:
        raise ValueError('Pad alignment rejected; failure evidence retained with the plan')
    report = rollout(store, row, job['id'], candidate, prepared_plan=after, state_profile=True)
    attempt_dir = workspace / 'runs/dynamics' / job['id'] / row['id']
    audit = verify(attempt_dir)
    recorded = Path(report['state_recording']['archive'])
    arrays = {}
    with h5py.File(recorded) as f:
        archive_meta = json.loads(f.attrs['metadata_json'])
        def read(name, value):
            if isinstance(value, h5py.Dataset):
                arrays[name] = value.asstr()[()] if value.dtype.kind in 'OUS' else value[()]
        f.visititems(read)
    assessment = assess_state_observations({'arrays':arrays,'metadata':archive_meta})
    destination = write_archive(root, job['dataset'], job['id'], arrays, archive_meta)
    return dict(status='physical_rollout_recorded', archive=str(destination),
                physics_status=report['status'], physics_validated=report['physics_validated'],
                actuator_replay_pass=audit['actuator_replay_pass'],
                state_assessment=assessment, report=str(attempt_dir/'result.json'))


def status(root):
    root = Path(root)
    workers = [json.loads(p.read_text()) for p in (root/'workers').glob('*.json')]
    fresh = [v for v in workers if time.time()-v['time'] < 60]
    jobs = []
    for p in sorted((root/'queue/tasks').glob('*/job.json')):
        result = p.parent/'result.json'
        jobs.append(dict(task=p.parent.name, **(json.loads(result.read_text()) if result.exists() else {'status':'pending'})))
    return dict(workers_live=len(fresh), physical_hosts=len({v.get('node') for v in fresh}),
                free_bytes=shutil.disk_usage(root).free, jobs=jobs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('enqueue','execute','status','describe'))
    parser.add_argument('--root',type=Path,default=Path('/mnt/reachy-retarget'))
    parser.add_argument('--job',type=Path)
    parser.add_argument('--task',type=Path)
    parser.add_argument('--attempt',type=Path)
    parser.add_argument('--dataset',choices=DATASETS)
    args=parser.parse_args()
    if args.operation=='enqueue':
        print(enqueue(args.root,json.loads(args.job.read_text())))
    elif args.operation=='execute':
        execute(args.root,args.task,args.attempt)
    elif args.operation=='describe':
        print(json.dumps(adapter(args.dataset).describe(),indent=2))
    else:
        print(json.dumps(status(args.root),indent=2))


if __name__=='__main__':
    main()
