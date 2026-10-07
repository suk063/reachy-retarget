"""Stage a predeclared held-out Can cohort from an already acquired raw file.

This command creates source inputs and a pending manifest, never downloads or
launches simulations. Existing source bytes and previous evaluation cohorts stay
unchanged. Execution requires a separately published explicit runtime pin.
"""
import argparse,json,hashlib,subprocess,textwrap
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--pod',required=True)
    args=parser.parse_args()
    source_id='hf__robomimic__robomimic_datasets'
    parent=json.load(open('configs/can-finite-pad-heldout-v1.json'))['jobs'][0]
    used=set()
    for path in Path('configs').glob('*.json'):
     try:v=json.loads(path.read_text())
     except (ValueError,OSError):continue
     if isinstance(v,dict):
      used.update(Path(j['source']).stem for j in v.get('jobs',[]) if j.get('source'))
    policy=dict(label='can-heldout-cohort-v2',selection='Consecutive previously unused source demos25through48, chosen before new physical outcomes',source_id=source_id,source_sequences=[f'v1.5/can/ph/demo_{i}' for i in range(25,49)],geometry_policy=dict(cylindrical_grasp=dict(end_margin_m=.001,orientation_policy='jaw_level',azimuth_policy='source'),base_placement_bank='can-geometry-bank-v1;54base candidates with existing45placementbank;rank beforephysicaloutcomes',full_original_admission_gates=True),physical_policies=[dict(name='primary',variant='contact_0035',source_prefix_velocity_feedforward=1.0),dict(name='feedforward_ablation',variant='contact_0035',source_prefix_velocity_feedforward=0.0)],object_and_contact_model='Original unchanged model only',evaluation='Report each frozen policy separately over24unique source episodes, plus separate geometry-admitted denominator. No outcome-driven retuning within this cohort; failed attempts retained. Successful variants do not add demonstrations.',excluded_source_episode_ids=sorted(used))
    policy_path=Path('configs/can-heldout-cohort-v2-policy.json');assert not policy_path.exists();policy_path.write_text(json.dumps(policy,indent=2)+'\n')
    runtime=json.loads(args.runtime.read_text())
    v=dict(policy=policy,runtime=runtime,source_provenance=parent['source_provenance'],policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest())
    code=r'''
    import json,sys,shutil,hashlib,h5py,os
    from pathlib import Path
    v=json.load(sys.stdin);r=Path('/mnt/reachy-retarget');policy=v['policy'];label=policy['label'];base=r/'imports'/label
    assert not base.exists(),'Immutable source cohort already exists'
    assert shutil.disk_usage(r).free-500_000_000>=50_000_000_000
    sys.path.insert(0,v['runtime']['source'])
    from reachy_retarget.normalize import robomimic
    from reachy_retarget.store import Store,sha256,json_write
    raw=r/'imports/additional-independent-v1'/v['source_provenance']['provenance'][0]['path'];assert sha256(raw)==v['source_provenance']['provenance'][0]['sha256']
    with h5py.File(raw,'r') as f:assert all('demo_'+str(i) in f['data'] for i in range(25,49))
    base.mkdir();link=base/v['source_provenance']['provenance'][0]['path'];link.parent.mkdir(parents=True);link.symlink_to(raw)
    for name in ('robosuite_assets','robosuite_can_assets'):
        assets=r/'imports/legacy-all-v1/data/raw'/name
        if assets.exists():(base/'data/raw'/name).symlink_to(assets,target_is_directory=True)
    json_write(base/'selection-policy.json',dict(policy,local_policy_sha256=v['policy_sha256'],normalization_runtime=v['runtime']))
    store=Store(base);paths=robomimic(store,policy['source_id'],49);jobs=[]
    for row in store.rows('episodes'):
     if row['source_sequence'] not in policy['source_sequences']:continue
     assert row['id'] not in policy['excluded_source_episode_ids'],'Held-out source already used'
     meta=json.loads((base/row['path']).with_suffix('.json').read_text());assert meta['objects'] and meta['env_args']['env_name']=='PickPlaceCan'
     job_id=label+'-baseline-'+row['id']
     jobs.append(dict(id=job_id,operation='legacy_pipeline',dataset='robomimic',parent_workspace='imports/'+label,workspace='workspaces/'+job_id,source=row['path'],source_normalized_sha256=sha256(base/row['path']),source_provenance=v['source_provenance'],evaluation_split='frozen_policy_heldout_v2',source_group=meta['source_group'],source_episode_id=row['id'],runtime_source_sha256='PENDING_NEXT_PUBLISHED_RELEASE',scope='Full-source diagnostic baseline and fresh frozen numeric parent; not the held-out corrected-policy outcome'))
    assert len(jobs)==24
    jobs.sort(key=lambda j:int(j['source_group'].rsplit('_',1)[1]))
    manifest=dict(label=label+'-baseline',jobs=jobs,source_policy=policy,source_policy_sha256=v['policy_sha256'],normalization_runtime_sha256=v['runtime']['source_sha256'],normalized_sources=len(paths),scope='No source download or objectless data.49normalizations include25existing source identities for reproducible adapter output; only24previously unused trajectories are execution jobs. Original raw source retained unchanged.')
    json_write(base/'manifest.json',manifest)
    assert sha256(raw)==v['source_provenance']['provenance'][0]['sha256']
    print(json.dumps(manifest))
    '''
    proc=subprocess.run(['kubectl','-n','erl-ucsd','exec','-i',args.pod,'--',runtime['python'],'-c',textwrap.dedent(code)],input=json.dumps(v),text=True,capture_output=True,timeout=600)
    Path('/tmp/seed-heldout-can-v2.stderr').write_text(proc.stderr)
    if proc.returncode:raise RuntimeError(proc.stderr)
    manifest=json.loads(proc.stdout);Path('configs/can-heldout-cohort-v2-baseline.pending.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(jobs=len(manifest['jobs']),first=manifest['jobs'][0]['source_group'],last=manifest['jobs'][-1]['source_group'],new_downloads=False)))

if __name__ == '__main__':
    main()
