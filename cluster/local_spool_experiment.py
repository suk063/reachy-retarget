"""Run one frozen experiment in pod-local scratch while shared storage is read-only.

This launcher copies and verifies numeric inputs and pinned Python sources. It
does not enqueue work, fetch datasets, edit a release or publish output. Scene
and asset paths are retained, with externally referenced assets hash-verified by
the unchanged frozen-plan loader and actuator replay auditor.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def within(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('A safe relative input path is required')
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('An input escaped its declared root')
    return path


def verify_published_manifest(proof, release_sha256):
    """Reproduce cluster/publish.py's composite identity, including its JSON separators."""
    for field, identity in (('source_files','source_sha256'),('control_files','control_sha256')):
        entries=proof.get(field)
        if not isinstance(entries,dict) or not entries:
            raise ValueError('Missing published file manifest: '+field)
        for name,checksum in entries.items():
            path=Path(name)
            if path.is_absolute() or '..' in path.parts or not re.fullmatch('[0-9a-f]{64}',checksum):
                raise ValueError('Invalid published file manifest entry: '+name)
        composite=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()
        if composite != proof.get(identity):
            raise ValueError('Composite published manifest mismatch: '+identity)
    if proof['source_sha256'] != release_sha256:
        raise ValueError('Published source identity differs from requested release')


def bind_job_release(job, release_sha256):
    """An explicit conflicting immutable pin is never silently rebound."""
    original=job.get('runtime_source_sha256')
    if original not in (None,'PENDING_NEXT_PUBLISHED_RELEASE',release_sha256):
        raise ValueError('Job runtime pin conflicts with explicit requested release')
    result=deepcopy(job)
    result['runtime_source_sha256']=release_sha256
    return result,dict(original_pin=original,bound_pin=release_sha256,
                      explicit_binding=True,previously_unbound=original!=release_sha256)


def runner_module(job):
    operation = job.get('operation', 'feasible_geometry_experiment')
    modules = {'legacy_pipeline': 'reachy_retarget.legacy_pipeline',
               'feasible_experiment': 'reachy_retarget.feasible_experiments',
               'feasible_geometry_experiment': 'reachy_retarget.feasible_experiments',
               'feasible_geometry_search': 'reachy_retarget.feasible_geometry_search',
               'feasible_placement_search': 'reachy_retarget.feasible_placement_search'}
    if operation not in modules:
        raise ValueError('Unsupported frozen local-spool operation: '+str(operation))
    return modules[operation]


def stage_inputs(pvc, spool, job, copy_verified):
    """Stage either an untouched source or an exact frozen experiment parent."""
    fresh = job.get('operation') == 'legacy_pipeline'
    if fresh and job.get('parent_attempt'):
        raise ValueError('Fresh-source baseline cannot inherit a frozen attempt')
    if not fresh and not job.get('parent_attempt'):
        raise ValueError('Local spool requires a frozen parent attempt')
    workspace = within(spool, job['workspace'])
    workspace.mkdir(parents=True)
    parent = within(pvc, job['parent_workspace'])
    source = Path(job['source'])
    for relative in (source, source.with_suffix('.json')):
        expected = job.get('source_normalized_sha256') if relative == source else None
        copy_verified(within(parent, relative), workspace/relative, expected)
    (workspace/'data/raw').symlink_to(parent/'data/raw', target_is_directory=True)
    record = dict(source_sha256=digest(workspace/source), parent_workspace=str(parent), job=job,
                  input_kind='untouched_normalized_source' if fresh else 'frozen_experiment_parent')
    if not fresh:
        motion = Path('data/retargeted')/source.parent.name/source.stem
        for name in ('motion.h5', 'validation.json'):
            copy_verified(within(parent, motion/name), workspace/motion/name)
        parent_attempt = within(pvc, job['parent_attempt'])
        parent_hashes = job.get('parent_attempt_files_sha256')
        if parent_hashes is not None and (not isinstance(parent_hashes,dict)
                or set(parent_hashes) != {'scene.xml','plan.h5','result.json'}
                or any(not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value)
                       for value in parent_hashes.values())):
            raise ValueError('Frozen parent requires exact scene, plan and report checksums')
        for name in ('scene.xml', 'plan.h5', 'result.json'):
            copy_verified(parent_attempt/name, workspace/'frozen-plan'/name,
                          None if parent_hashes is None else parent_hashes[name])
        write_json(workspace/'frozen-plan/binding.json', dict(parent_attempt=str(parent_attempt),
            files={name:digest(workspace/'frozen-plan'/name) for name in ('scene.xml', 'plan.h5', 'result.json')}))
        record['motion_sha256'] = digest(workspace/motion/'motion.h5')
    write_json(workspace/'input-manifest.json', record)
    return workspace


def prohibit_shared_writes(root, rejected, *, writable_spool=None):
    """Guard Python file/mutation APIs; native libraries receive local outputs."""
    root = root.resolve()
    writable_spool = Path(writable_spool).resolve() if writable_spool is not None else None

    def check(path):
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        resolved = Path(os.fsdecode(path)).resolve()
        if resolved.is_relative_to(root) and not (
                writable_spool is not None and resolved.is_relative_to(writable_spool)):
            rejected.append(str(resolved))
            raise PermissionError('Local-spool execution forbids shared writes: '+str(resolved))

    def audit(event, args):
        if event == 'open':
            path, mode, flags = args
            if ((isinstance(mode, str) and any(c in mode for c in 'wax+'))
                    or (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))):
                check(path)
        elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.chown', 'os.utime', 'os.truncate'):
            check(args[0])
        elif event in ('os.rename', 'os.link'):
            check(args[0]); check(args[1])
        elif event == 'os.symlink':
            check(args[1])
    sys.addaudithook(audit)


def validate_spool_location(pvc, spool, job):
    """An explicit shared fallback may write only its separate output subtree."""
    pvc, spool = Path(pvc).resolve(), Path(spool).resolve()
    storage = job.get('spool_storage', 'pod_local')
    if storage not in ('pod_local', 'shared_pvc'):
        raise ValueError('Unknown spool storage')
    if storage == 'shared_pvc':
        allowed = pvc/'shared-spool'
        if spool == allowed or not spool.is_relative_to(allowed):
            raise ValueError('Shared spool must be below the dedicated shared-spool root')
    elif spool.is_relative_to(pvc):
        raise ValueError('Spool must be outside shared storage')
    return storage


def run(pvc, spool, job, release_sha256, *, runtime_root=None):
    pvc, spool = Path(pvc).resolve(), Path(spool).resolve()
    runtime_root = Path(runtime_root).resolve() if runtime_root is not None else pvc
    publication_scope = 'shared_pvc' if runtime_root == pvc else 'pod_local_unpublished'
    if not re.fullmatch('[0-9a-f]{64}', release_sha256):
        raise ValueError('An explicit published source SHA256 is required')
    job,binding=bind_job_release(job,release_sha256)
    if job.get('operation') != 'legacy_pipeline' and not job.get('parent_attempt'):
        raise ValueError('Local spool requires a frozen parent attempt; unstaged native planning is unsupported')
    if job.get('operation') == 'legacy_pipeline' and job.get('parent_attempt'):
        raise ValueError('Fresh-source baseline cannot inherit a frozen attempt')
    module_name = runner_module(job)
    if any(name=='reachy_retarget' or name.startswith('reachy_retarget.') for name in sys.modules):
        raise ValueError('Run local spool in a fresh process to exclude pre-imported unpinned source modules')
    storage = validate_spool_location(pvc, spool, job)
    spool.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(spool.parent).free < 50_000_000_000:
        raise OSError('Spool disk free space is below the 50 decimal GB reserve')
    spool.mkdir(exist_ok=False)
    os.chdir(spool)
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = os.environ['OMP_NUM_THREADS'] = '1'
    rejected = []
    prohibit_shared_writes(pvc, rejected,
                           writable_spool=spool if storage == 'shared_pvc' else None)
    copies = []
    source_modules = {}
    original_pin = binding['original_pin']
    started = time.time()
    write_json(spool/'job.json', job)
    write_json(spool/'job-release-binding.json',binding)
    launcher=Path(__file__).resolve()
    if launcher.is_file():
        shutil.copyfile(launcher,spool/'launcher.py')
    write_json(spool/'progress.json', dict(stage='staging', started_unix_s=started))

    def copy_verified(original, target, expected=None):
        original, target = Path(original), Path(target)
        before = digest(original)
        if expected is not None and before != expected:
            raise ValueError('Pinned input mismatch: '+str(original))
        target.parent.mkdir(parents=True, exist_ok=True)
        with original.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst)
            dst.flush(); os.fsync(dst.fileno())
        if digest(target) != before or digest(original) != before:
            raise ValueError('Input changed while staging: '+str(original))
        entry=dict(source=str(original), local=str(target.relative_to(spool)),
                   bytes=target.stat().st_size, sha256=before)
        copies.append(entry)
        return entry

    result = None
    try:
        proof_path = runtime_root/'provenance'/(release_sha256+'.json')
        copy_verified(proof_path, spool/'runtime-provenance.json')
        proof = json.loads((spool/'runtime-provenance.json').read_text())
        verify_published_manifest(proof,release_sha256)
        if publication_scope == 'pod_local_unpublished' and proof.get('publication_scope') != publication_scope:
            raise ValueError('Local runtime must explicitly disclose that shared publication is pending')
        release = runtime_root/'releases'/release_sha256
        controller = pvc/'references/reachy-control'/proof['control_sha256']
        for relative, expected in proof['source_files'].items():
            if relative.startswith('reachy_retarget/') and relative.endswith('.py'):
                entry=copy_verified(within(release,relative),spool/'source'/relative,expected)
                source_modules[entry['local']]=expected
        for relative, expected in proof['control_files'].items():
            if relative.endswith('.py'):
                entry=copy_verified(within(controller,relative),spool/'controller'/relative,expected)
                source_modules[entry['local']]=expected
            elif digest(within(controller,relative)) != expected:
                raise ValueError('Controller asset does not match published proof: '+relative)
        (spool/'controller/asset').symlink_to(controller/'asset',target_is_directory=True)
        stage_inputs(pvc,spool,job,copy_verified)
        write_json(spool/'input-binding.json',dict(source_release_sha256=release_sha256,
            original_job_release_sha256=original_pin,control_sha256=proof['control_sha256'],
            release_binding=binding,composite_manifest_hashes_verified=True,
            published_composite_manifest_hashes_verified=publication_scope=='shared_pvc',
            runtime_publication_scope=publication_scope,
            copies=copies,source_module_sha256=source_modules,
            spool_storage=storage,
            shared_write_policy=('Only the exclusive shared output subtree is writable; source and other shared paths remain denied'
                                 if storage == 'shared_pvc' else
                                 'No shared output paths; Python write APIs denied; native simulation outputs explicitly local')))
        os.environ['REACHY_RETARGET_CONTROL'] = str(spool/'controller')
        os.environ['REACHY_RETARGET_BACKEND'] = 'reference'
        sys.path.insert(0,str(spool/'source'))
        write_json(spool/'progress.json',dict(stage='running',started_unix_s=started,
            source_release_sha256=release_sha256,job=job['id'],input_bytes=sum(x['bytes'] for x in copies)))
        print(json.dumps(dict(stage='running',spool=str(spool),job=job['id'])),flush=True)
        from importlib import import_module
        result = import_module(module_name).run(spool,job)
        if job.get('operation') == 'legacy_pipeline':
            # Preserve the baseline's complete result and kinematic evidence.
            # Only its measured physics archive enters the canonical state index.
            result = dict(result, source_episode=Path(job['source']).stem,
                          archive=result.get('archives', {}).get('physics'),
                          actuator_audit=result.get('audit'),
                          canonical_archive_scope='Measured physics only; kinematic reference archive retained in full artifact')
        write_json(spool/'result.json',result)
        return result
    except BaseException as error:
        write_json(spool/'launcher-failure.json',dict(error=type(error).__name__+': '+str(error),
            traceback=traceback.format_exc(),copied_inputs=copies,rejected_shared_writes=rejected))
        raise
    finally:
        module_checks=[]
        for name,module in tuple(sys.modules.items()):
            filename=getattr(module,'__file__',None)
            if not filename:continue
            path=Path(filename).resolve()
            if not path.is_relative_to(spool):continue
            relative=str(path.relative_to(spool))
            if relative in source_modules:
                actual=digest(path)
                module_checks.append(dict(module=name,path=relative,sha256=actual,
                                          matches_pin=actual==source_modules[relative]))
        files=[];links=[]
        for directory,dirs,names in os.walk(spool,followlinks=False):
            for name in dirs+names:
                path=Path(directory)/name
                if path.is_symlink():links.append(dict(path=str(path.relative_to(spool)),target=str(path.resolve())))
            for name in names:
                path=Path(directory)/name
                if path.is_symlink() or name=='portable-artifact-manifest.json':continue
                files.append(dict(path=str(path.relative_to(spool)),bytes=path.stat().st_size,sha256=digest(path)))
        write_json(spool/'portable-artifact-manifest.json',dict(schema='local-spool-v1',
            job=job['id'],source_release_sha256=release_sha256,spool=str(spool),
            spool_storage=storage,
            runtime_publication_scope=publication_scope,
            source_job_release_sha256=original_pin,status=result.get('status') if result else 'launcher_failed',
            files=files,external_read_only_links=links,imported_source_modules=module_checks,
            all_imported_source_modules_match_pin=all(x['matches_pin'] for x in module_checks),
            copied_input_bytes=sum(x['bytes'] for x in copies),artifact_bytes=sum(x['bytes'] for x in files),
            rejected_shared_write_attempts=rejected,started_unix_s=started,finished_unix_s=time.time(),
            portability='Archive includes local generated files and Python sources. Original scene paths and external source asset references are retained byte-for-byte; restore original PVC references or use separately verified mount mapping to replay.'))
        print(json.dumps(dict(stage='finished',spool=str(spool),
            status=result.get('status') if result else 'launcher_failed',
            artifact_bytes=sum(x['bytes'] for x in files))),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pvc',default='/mnt/reachy-retarget')
    parser.add_argument('--spool',required=True)
    parser.add_argument('--job',required=True)
    parser.add_argument('--release',required=True)
    parser.add_argument('--runtime-root')
    args=parser.parse_args()
    run(args.pvc,args.spool,json.loads(Path(args.job).read_text()),args.release,runtime_root=args.runtime_root)
