"""The local launcher fails closed on shared writes and unbound releases."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from cluster.local_spool_experiment import bind_job_release, run, runner_module, stage_inputs, validate_spool_location, verify_published_manifest, within


def test_shared_fallback_needs_explicit_separate_output_root(tmp_path):
    shared = tmp_path/'shared'
    spool = shared/'shared-spool/batch/job/artifact'
    with pytest.raises(ValueError, match='outside shared'):
        validate_spool_location(shared, spool, {})
    assert validate_spool_location(shared, spool, {'spool_storage': 'shared_pvc'}) == 'shared_pvc'
    for path in (shared, shared/'source', shared/'shared-spool'):
        with pytest.raises(ValueError, match='dedicated'):
            validate_spool_location(shared, path, {'spool_storage': 'shared_pvc'})


def test_shared_fallback_cannot_overwrite_inputs_via_output_symlinks(tmp_path):
    shared = tmp_path/'shared';shared.mkdir()
    (shared/'original').write_text('original')
    spool = shared/'shared-spool/job/artifact';spool.mkdir(parents=True)
    script = '''
from pathlib import Path
from cluster.local_spool_experiment import prohibit_shared_writes
shared=Path(SHARED);spool=Path(SPOOL);denied=[]
prohibit_shared_writes(shared,denied,writable_spool=spool)
(spool/'result').write_text('allowed')
(spool/'alias').symlink_to(shared/'original')
for path in (shared/'original',spool/'alias',shared/'shared-spool/other'):
 try:path.write_text('forbidden')
 except PermissionError:pass
 else:raise AssertionError('Shared input/other output became writable')
assert (shared/'original').read_text()=='original'
assert len(denied)==3
'''.replace('SHARED',repr(str(shared))).replace('SPOOL',repr(str(spool)))
    subprocess.run([sys.executable, '-c', script], check=True)
    assert (spool/'result').read_text() == 'allowed'


def test_input_paths_cannot_escape_declared_root_through_symlink(tmp_path):
    parent=tmp_path/'inputs';parent.mkdir()
    outside=tmp_path/'outside';outside.mkdir()
    (parent/'link').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='escaped'):
        within(parent,'link/file.h5')
    with pytest.raises(ValueError,match='safe relative'):
        within(parent,'../file.h5')


def test_pending_release_is_rejected_before_any_local_spool_is_created(tmp_path):
    with pytest.raises(ValueError,match='explicit published'):
        run(tmp_path/'shared',tmp_path/'scratch',{},'PENDING_NEXT_PUBLISHED_RELEASE')
    assert not (tmp_path/'scratch').exists()


def test_python_guard_blocks_shared_write_and_symlink_alias_but_allows_reads(tmp_path):
    # Audit hooks are process-wide and irrevocable, so isolate this one.
    shared=tmp_path/'shared';shared.mkdir();(shared/'existing').write_text('original')
    script='''
import json
from pathlib import Path
from cluster.local_spool_experiment import prohibit_shared_writes
root=Path(ROOT);denied=[]
(root/'alias').symlink_to(root/'shared',target_is_directory=True)
prohibit_shared_writes(root/'shared',denied)
assert (root/'shared/existing').read_text()=='original'
(root/'scratch.txt').write_text('allowed')
for path in [root/'shared/existing',root/'alias/existing']:
 try:path.write_text('overwritten')
 except PermissionError:pass
 else:raise AssertionError('A shared write was permitted')
assert (root/'shared/existing').read_text()=='original'
print(json.dumps(denied))
'''.replace('ROOT',repr(str(tmp_path)))
    process=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[1],
                           text=True,capture_output=True,check=True)
    assert len(json.loads(process.stdout))==2
    assert (tmp_path/'scratch.txt').read_text()=='allowed'


def test_composite_release_and_controller_identity_match_exact_publisher_encoding():
    sources={'reachy_retarget/x.py':'1'*64,'README.md':'2'*64}
    control={'control/reachy.py':'3'*64}
    key=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
    proof=dict(source_files=sources,source_sha256=key(sources),
               control_files=control,control_sha256=key(control))
    verify_published_manifest(proof,proof['source_sha256'])
    proof['control_files']['control/reachy.py']='4'*64
    with pytest.raises(ValueError,match='Composite published manifest mismatch'):
        verify_published_manifest(proof,proof['source_sha256'])


def test_conflicting_job_pin_cannot_be_overridden(tmp_path):
    with pytest.raises(ValueError,match='conflicts'):
        run(tmp_path/'shared',tmp_path/'scratch',{'runtime_source_sha256':'1'*64},'2'*64)
    assert not (tmp_path/'scratch').exists()


def test_pending_job_requires_recorded_explicit_binding_without_mutating_original():
    job={'runtime_source_sha256':'PENDING_NEXT_PUBLISHED_RELEASE'}
    changed,binding=bind_job_release(job,'2'*64)
    assert changed['runtime_source_sha256']=='2'*64
    assert job['runtime_source_sha256']=='PENDING_NEXT_PUBLISHED_RELEASE'
    assert binding==dict(original_pin='PENDING_NEXT_PUBLISHED_RELEASE',bound_pin='2'*64,
                         explicit_binding=True,previously_unbound=True)


def test_no_frozen_parent_fails_before_staging_or_runtime_import(tmp_path):
    with pytest.raises(ValueError,match='frozen parent attempt'):
        run(tmp_path/'shared',tmp_path/'scratch',{},'2'*64)
    assert not (tmp_path/'scratch').exists()


def test_geometry_only_jobs_cannot_dispatch_to_physical_runner_and_unknown_ops_fail_before_staging(tmp_path):
    assert runner_module({'operation':'legacy_pipeline'}) == 'reachy_retarget.legacy_pipeline'
    assert runner_module({'operation':'feasible_geometry_search'}) == 'reachy_retarget.feasible_geometry_search'
    assert runner_module({'operation':'feasible_geometry_experiment'}) == 'reachy_retarget.feasible_experiments'
    assert runner_module({'operation':'feasible_experiment'}) == 'reachy_retarget.feasible_experiments'
    with pytest.raises(ValueError,match='Unsupported frozen'):
        run(tmp_path/'shared',tmp_path/'scratch',{'operation':'native_replay','parent_attempt':'parent'},'2'*64)
    assert not (tmp_path/'scratch').exists()


def test_fresh_source_staging_needs_no_old_motion_and_preserves_hash_bound_inputs(tmp_path):
    import shutil
    pvc, spool = tmp_path/'shared', tmp_path/'scratch'
    source = Path('data/normalized/example/episode.h5')
    parent = pvc/'imports/fresh'
    (parent/source).parent.mkdir(parents=True)
    (parent/source).write_bytes(b'unchanged original normalized source')
    (parent/source.with_suffix('.json')).write_text('{"objects":{"Can":{}}}')
    (parent/'data/raw').mkdir()
    sha = hashlib.sha256((parent/source).read_bytes()).hexdigest()
    copied = []
    def copy(original, target, expected=None):
        raw = original.read_bytes()
        if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Pinned input mismatch')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        copied.append(original)
    job = dict(operation='legacy_pipeline', workspace='workspaces/new', parent_workspace='imports/fresh',
               source=str(source), source_normalized_sha256=sha)
    workspace = stage_inputs(pvc, spool, job, copy)
    assert len(copied) == 2
    assert (workspace/source).read_bytes() == (parent/source).read_bytes()
    assert (workspace/'data/raw').resolve() == parent/'data/raw'
    assert not (workspace/'data/retargeted').exists()
    assert not (workspace/'frozen-plan').exists()
    record = json.loads((workspace/'input-manifest.json').read_text())
    assert record['source_sha256'] == sha and record['input_kind'] == 'untouched_normalized_source'
    with pytest.raises(ValueError, match='Pinned input mismatch'):
        stage_inputs(pvc, tmp_path/'other', dict(job, source_normalized_sha256='0'*64), copy)
    with pytest.raises(ValueError, match='cannot inherit'):
        run(pvc, tmp_path/'invalid', dict(job, parent_attempt='old'), '2'*64)
    assert not (tmp_path/'invalid').exists()


def test_cached_parent_plan_must_match_declared_publication_hashes(tmp_path):
    pvc = tmp_path/'shared'
    parent = pvc/'parent'
    source = Path('data/normalized/example/episode.h5')
    motion = Path('data/retargeted/example/episode')
    files = {source: b'source', source.with_suffix('.json'): b'{}',
             motion/'motion.h5': b'motion', motion/'validation.json': b'{}',
             Path('admitted/scene.xml'): b'<mujoco/>', Path('admitted/plan.h5'): b'admitted plan',
             Path('admitted/result.json'): b'{}'}
    for name, raw in files.items():
        path = parent/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (parent/'data/raw').mkdir()
    hashes = {name: hashlib.sha256((parent/'admitted'/name).read_bytes()).hexdigest()
              for name in ('scene.xml', 'plan.h5', 'result.json')}
    job = dict(operation='feasible_placement_search', workspace='workspaces/test',
               parent_workspace='parent', parent_attempt='parent/admitted', source=str(source),
               parent_attempt_files_sha256=hashes)
    def copy(original, target, expected=None):
        raw = original.read_bytes()
        if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Pinned parent bytes changed')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    staged = stage_inputs(pvc, tmp_path/'valid', job, copy)
    assert (staged/'frozen-plan/plan.h5').read_bytes() == b'admitted plan'
    (parent/'admitted/plan.h5').write_bytes(b'different plan')
    with pytest.raises(ValueError, match='Pinned parent bytes changed'):
        stage_inputs(pvc, tmp_path/'changed', job, copy)
