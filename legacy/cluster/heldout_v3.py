"""Predeclared frozen Can generalization policy v3: staging, candidates and selection.

Every stage writes an immutable batch config or selection record. Candidate
generation and selection read only kinematic admission receipts; physical
outcomes are never read before the physics batch is generated. The policy file
(configs/can-heldout-cohort-v3-policy.json) is bound by SHA-256 into every job.

Subcommands (run from the repository root):
  stage          normalize the untouched v3 cohort on the PVC and write its baseline batch
  materialize    create portable frozen parents from completed baseline archives
  source-jobs    derive mobile source-path admission jobs for one declared tier
  select-source  choose the first admitted source path by declared rank and materialize it
  placement-jobs placement bank search on the selected source path for one declared tier
  select-placement  choose the minimum-rank admitted placement after all chunks complete
  physics-jobs   generate the predeclared physical policy arms for selected candidates
  evidence       list every attempt and score each predeclared policy over the full denominator
"""
import argparse
import collections
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import textwrap

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO/'configs/can-heldout-cohort-v3-policy.json'
NAMESPACE = 'erl-ucsd'
PVC = '/mnt/reachy-retarget'
RUNS = REPO/'runs/local-spool-pool'
COHORTS = {
    'v3': dict(label='can-heldout-cohort-v3', split='frozen_policy_heldout_v3',
               parents='configs/can-heldout-cohort-v3-materialized-parents.json'),
    'v2': dict(label='can-v3policy-rescore-v2', split='frozen_policy_v3_rescore_of_development_v2_cohort',
               parents='configs/can-heldout-cohort-v2-materialized-parents.json'),
    'smoke': dict(label='can-v3policy-smoke', split='development_mechanics_smoke_test_before_v3_freeze',
                  parents='configs/can-heldout-cohort-v2-materialized-parents.json'),
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy():
    return json.loads(POLICY.read_text()), sha256_file(POLICY)


def write_config(path, value):
    path = REPO/path
    if path.exists():
        raise FileExistsError('Immutable stage record already exists: '+str(path))
    path.write_text(json.dumps(value, indent=2)+'\n')
    print(path, len(value.get('jobs', value.get('sources', []))))


def pods():
    out = subprocess.check_output(['kubectl', '-n', NAMESPACE, 'get', 'pods', '-l',
                                   'app.kubernetes.io/name=reachy-retarget-worker', '-o', 'json'])
    return sorted(p['metadata']['name'] for p in json.loads(out)['items']
                  if p['status'].get('phase') == 'Running')


def pod_python(code, value, *, pod=None, timeout=1800):
    pod = pod or pods()[0]
    process = subprocess.run(['kubectl', '-n', NAMESPACE, 'exec', '-i', pod, '--',
                              PVC+'/runtime/run-python', '-c', textwrap.dedent(code)],
                             input=json.dumps(value), text=True, capture_output=True, timeout=timeout)
    if process.returncode:
        raise RuntimeError(process.stderr[-6000:])
    return json.loads(process.stdout.strip().splitlines()[-1])


def receipts(label):
    batch = RUNS/label
    manifest = json.loads((batch/'manifest.json').read_text())
    for job in manifest['jobs']:
        path = batch/job['id']/'backup.json'
        receipt = json.loads(path.read_text()) if path.exists() else None
        yield job, receipt


def common_job_fields(parent, policy_sha, cohort, runtime):
    return dict(dataset='robomimic', parent_task=parent['baseline_job'],
                parent_workspace=parent['parent_workspace'], source=parent['source'],
                source_normalized_sha256=parent['source_normalized_sha256'],
                source_provenance=parent['source_provenance'],
                evaluation_split=COHORTS[cohort]['split'], runtime_source_sha256=runtime,
                variant='contact_0035', frozen_policy_sha256=policy_sha,
                source_parent_materialization=dict(manifest=parent['materialization_manifest'],
                    sha256=parent['materialization_manifest_sha256'],
                    original_archive_sha256=parent['archive_sha256']))


# ---------------------------------------------------------------- staging

STAGE_CODE = r'''
import json,sys,shutil,hashlib,h5py,os
from pathlib import Path
v=json.load(sys.stdin);r=Path('/mnt/reachy-retarget');policy=v['policy'];label=policy['label'];base=r/'imports'/label
assert not base.exists(),'Immutable source cohort already exists'
assert shutil.disk_usage(r).free-2_000_000_000>=50_000_000_000
sys.path.insert(0,'/mnt/reachy-retarget/releases/'+v['runtime'])
from reachy_retarget.normalize import robomimic
from reachy_retarget.store import Store,sha256,json_write
prov=v['source_provenance']['provenance'][0]
raw=r/'imports/additional-independent-v1'/prov['path'];assert sha256(raw)==prov['sha256']
numbers=[int(s.rsplit('_',1)[1]) for s in policy['cohort']['source_sequences']]
with h5py.File(raw,'r') as f:assert all('demo_'+str(i) in f['data'] for i in numbers)
base.mkdir();link=base/prov['path'];link.parent.mkdir(parents=True);link.symlink_to(raw)
for name in ('robosuite_assets','robosuite_can_assets'):
    assets=r/'imports/legacy-all-v1/data/raw'/name
    if assets.exists():(base/'data/raw'/name).symlink_to(assets,target_is_directory=True)
json_write(base/'selection-policy.json',dict(policy=policy,local_policy_sha256=v['policy_sha256'],normalization_runtime=v['runtime']))
store=Store(base);paths=robomimic(store,policy['source_id'],max(numbers)+1);jobs=[]
excluded=set(policy['cohort']['excluded_source_episode_ids'])
for row in store.rows('episodes'):
    if row['source_sequence'] not in policy['cohort']['source_sequences']:
        assert row['id'] in excluded,'Normalized non-cohort demonstration must be a known earlier identity: '+row['id']
        continue
    assert row['id'] not in excluded,'Held-out source already used'
    meta=json.loads((base/row['path']).with_suffix('.json').read_text());assert meta['objects'] and meta['env_args']['env_name']=='PickPlaceCan'
    job_id=label+'-baseline-'+row['id']
    jobs.append(dict(id=job_id,operation='legacy_pipeline',dataset='robomimic',parent_workspace='imports/'+label,
        workspace='workspaces/'+job_id,source=row['path'],source_normalized_sha256=sha256(base/row['path']),
        source_provenance=v['source_provenance'],evaluation_split='frozen_policy_heldout_v3',
        source_group=meta['source_group'],source_episode_id=row['id'],runtime_source_sha256=v['runtime'],
        frozen_policy_sha256=v['policy_sha256'],
        scope='Full-source retarget baseline and fresh frozen numeric parent for the v3 frozen policy; not itself a policy outcome'))
assert len(jobs)==len(numbers)
jobs.sort(key=lambda j:int(j['source_group'].rsplit('_',1)[1]))
manifest=dict(label=label+'-baseline',jobs=jobs,source_policy_sha256=v['policy_sha256'],normalization_runtime_sha256=v['runtime'],
    normalized_sources=len(paths),scope='No download. Earlier identities are re-normalized only to reproduce adapter output; only the declared cohort becomes jobs.')
json_write(base/'manifest.json',manifest)
assert sha256(raw)==prov['sha256']
print(json.dumps(manifest))
'''


def stage(args):
    policy, policy_sha = load_policy()
    template = json.loads((REPO/'configs/can-heldout-cohort-v2-baseline-rest.json').read_text())['jobs'][0]
    manifest = pod_python(STAGE_CODE, dict(policy=policy, policy_sha256=policy_sha, runtime=args.runtime,
                                           source_provenance=template['source_provenance']), timeout=3600)
    write_config('configs/can-heldout-cohort-v3-baseline.json', manifest)


# ------------------------------------------------------------ materialization

MATERIALIZE_CODE = r'''
import json,sys
from concurrent.futures import ThreadPoolExecutor
v=json.load(sys.stdin)
sys.path.insert(0,'/mnt/reachy-retarget/releases/'+v['runtime'])
from reachy_retarget.frozen_parent import materialize
from reachy_retarget.store import sha256
def one(item):
    try:
        report=materialize(item['archive'],item['output'],archive_sha256=item['archive_sha256'],
            original_spool_root=item['original_spool_root'],workspace_relative=item['workspace_relative'],
            source_relative=item['source_relative'],attempt_relative=item['attempt_relative'])
        return dict(item,ready=bool(report.get('ready')),manifest_sha256=sha256(item['output']+'/materialization.json'))
    except Exception as error:
        return dict(item,ready=False,error=type(error).__name__+': '+str(error)[:2000])
with ThreadPoolExecutor(4) as pool:results=list(pool.map(one,v['items']))
print(json.dumps(results))
'''


def materialize(args):
    policy, policy_sha = load_policy()
    items, failures = [], []
    for job, receipt in receipts('can-heldout-cohort-v3-baseline'):
        sid = job['source_episode_id']
        status = (receipt or {}).get('result', {}) or {}
        if receipt is None or receipt['manifest']['status'] != 'physical_rollout_recorded':
            failures.append(dict(source_episode_id=sid, source_group=job['source_group'], baseline_job=job['id'],
                                 ready=False, reason='baseline_not_recorded',
                                 baseline_status=None if receipt is None else receipt['manifest']['status']))
            continue
        items.append(dict(source_episode_id=sid, source_group=job['source_group'], baseline_job=job['id'],
            source=job['source'], source_normalized_sha256=job['source_normalized_sha256'],
            archive=receipt['archive'], archive_sha256=receipt['sha256'],
            original_spool_root=receipt['original_spool'], workspace_relative=job['workspace'],
            source_relative=job['source'],
            attempt_relative=f"{job['workspace']}/runs/dynamics/{job['id']}-aligned/{sid}",
            output=f"{PVC}/materialized-parents/{job['id']}-v1"))
    results = pod_python(MATERIALIZE_CODE, dict(runtime=args.runtime, items=items), timeout=3600)
    sources = []
    for item in results:
        rel = item['output'][len(PVC)+1:]
        sources.append(dict(item, parent_workspace=rel+'/workspace',
            parent_attempt=f"{rel}/workspace/runs/dynamics/{item['baseline_job']}-aligned/{item['source_episode_id']}",
            materialization_manifest=item['output']+'/materialization.json',
            materialization_manifest_sha256=item.get('manifest_sha256')))
    write_config('configs/can-heldout-cohort-v3-materialized-parents.json', dict(
        schema='materialized-frozen-parent-cohort-v1', source_cohort='can-heldout-cohort-v3',
        runtime_source_sha256=args.runtime, frozen_policy_sha256=policy_sha,
        expected_sources=len(items)+len(failures), ready_sources=sum(s['ready'] for s in sources),
        scope='Portable numeric parents only; baseline outcomes are not policy outcomes.',
        sources=sources+failures))


def parents(cohort):
    record = json.loads((REPO/COHORTS[cohort]['parents']).read_text())
    template = json.loads((REPO/'configs/can-heldout-cohort-v2-baseline-rest.json').read_text())['jobs'][0]
    result = []
    for item in record['sources']:
        if not item.get('ready'):
            continue
        item = dict(item)
        item.setdefault('source_provenance', template['source_provenance'])
        if 'source_normalized_sha256' not in item:
            for config in ('can-heldout-cohort-v2-baseline-rest.json', 'can-heldout-cohort-v2-baseline.pending.json'):
                for job in json.loads((REPO/'configs'/config).read_text())['jobs']:
                    if job['source_episode_id'] == item['source_episode_id']:
                        item['source_normalized_sha256'] = job['source_normalized_sha256']
        item.setdefault('materialization_manifest_sha256', item.get('manifest_sha256'))
        result.append(item)
    return result


# ---------------------------------------------------------- source admission

DERIVE_CODE = r'''
import json,sys,math
import numpy as np,h5py
v=json.load(sys.stdin)
sys.path.insert(0,'/mnt/reachy-retarget/releases/'+v['runtime'])
import mujoco
from reachy_retarget.store import sha256
rule=v['rule'];out=[]
for p in v['parents']:
    attempt='/mnt/reachy-retarget/'+p['parent_attempt']
    try:
        files={n:sha256(attempt+'/'+n) for n in ('scene.xml','plan.h5','result.json')}
        with h5py.File(attempt+'/plan.h5','r') as f:
            times=f['time_s'][()];ref=f['reference'][()];grip=f['gripper'][()];obj=f['object_goals'][()]
        closed=np.flatnonzero(grip<0)
        if not len(closed):raise ValueError('No closed source interval')
        first=int(closed[0]);z0=float(obj[0,2])
        lifted=np.flatnonzero((np.arange(len(times))>=first)&(obj[:,2]-z0>=rule['lift_threshold_m']))
        anchor=int(lifted[0]) if len(lifted) else first
        model=mujoco.MjModel.from_xml_path(attempt+'/scene.xml');data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        final=obj[-1,:3];floors=[]
        for g in range(model.ngeom):
            name=model.geom(g).name or ''
            if not name.startswith('source_scene_geom_') or model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_BOX:continue
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):continue
            R=data.geom_xmat[g].reshape(3,3);size=model.geom_size[g]
            if abs(R[2,2])<.999999 or size[2]>=min(size[:2]):continue
            top=float(data.geom_xpos[g][2]+size[2]);c=data.geom_xpos[g]
            local=R.T@(final-c)
            if abs(local[0])<=size[0] and abs(local[1])<=size[1] and top<=final[2]:
                floors.append((top,name))
        if not floors:raise ValueError('No support floor below final source object position')
        floor=max(floors)[1]
        out.append(dict(source_episode_id=p['source_episode_id'],ready=True,files=files,frames=int(len(times)),
            duration_s=float(times[-1]-times[0]),first_closed=first,anchor_index=anchor,
            anchor_reference=ref[anchor].tolist(),reference_base_min=ref[:,:3].min(0).tolist(),
            reference_base_max=ref[:,:3].max(0).tolist(),object_anchor_pose=obj[anchor].tolist(),
            support_floor_geom=floor,support_floor_candidates=sorted(floors)))
    except Exception as error:
        out.append(dict(source_episode_id=p['source_episode_id'],ready=False,error=type(error).__name__+': '+str(error)[:1000]))
print(json.dumps(out))
'''


def derive(cohort, runtime, rule, selected=None):
    items = [p for p in parents(cohort) if selected is None or p['source_episode_id'] in selected]
    rows = pod_python(DERIVE_CODE, dict(runtime=runtime, rule=rule,
                                        parents=[dict(source_episode_id=p['source_episode_id'],
                                                      parent_attempt=p['parent_attempt']) for p in items]))
    by_id = {r['source_episode_id']: r for r in rows}
    return [(p, by_id[p['source_episode_id']]) for p in items]


def mobile_seed(derived, offset_deg, perturbation_m, rule):
    """Transport the parent anchor seed around the source object; bounds contain both."""
    anchor = list(map(float, derived['anchor_reference']))
    theta = math.radians(offset_deg)
    pivot = derived['object_anchor_pose'][:2]
    dx, dy = anchor[0]-pivot[0], anchor[1]-pivot[1]
    moved = [pivot[0]+math.cos(theta)*dx-math.sin(theta)*dy,
             pivot[1]+math.sin(theta)*dx+math.cos(theta)*dy, anchor[2]+theta]
    margins = rule['base_bound_margins_xyyaw']
    low = [min(derived['reference_base_min'][i], moved[i])-margins[i] for i in range(3)]
    high = [max(derived['reference_base_max'][i], moved[i])+margins[i] for i in range(3)]
    seed = [moved[0]+perturbation_m, moved[1], moved[2]]+anchor[3:]
    return dict(anchor_index=int(derived['anchor_index']), anchor_reference=seed,
                base_bounds_xyyaw=[[low[i], high[i]] for i in range(3)],
                minimum_fixture_gap_m=.003, pose_tolerance_constraints=True, stop_on_first_invalid=True), \
        dict(pivot_world_xy=pivot, planar_yaw_transport_rad=theta, requested_grasp_rotation_deg=offset_deg,
             seed_base_x_perturbation_m=perturbation_m, original_anchor_reference=anchor)


def source_candidates(policy):
    rank = 0
    for tier in policy['source_path']['tiers']:
        for grasp in tier['grasps']:
            for seed, perturbation in enumerate(policy['source_path']['seed_base_x_perturbations_m']):
                yield rank, tier['name'], grasp, seed, perturbation
                rank += 1


def admitted_paths(cohort):
    """Admission receipts across all source tiers, keyed by source."""
    label = COHORTS[cohort]['label']
    found = collections.defaultdict(list)
    for path in sorted(RUNS.glob(label+'-source-*/manifest.json')):
        for job, receipt in receipts(path.parent.name):
            result = (receipt or {}).get('result') or {}
            found[job['source_episode_id_v3']].append(dict(job=job, batch=path.parent.name,
                status=result.get('status') if receipt else 'not_completed', receipt=receipt))
    return found


def source_jobs(args):
    policy, policy_sha = load_policy()
    label = COHORTS[args.cohort]['label']
    tiers = [t['name'] for t in policy['source_path']['tiers']]
    tier_index = tiers.index(args.tier)
    previous = admitted_paths(args.cohort)
    eligible = set()
    for p in parents(args.cohort):
        sid = p['source_episode_id']
        attempts = previous.get(sid, [])
        if any(a['status'] == 'not_completed' for a in attempts):
            raise RuntimeError('Earlier tier still incomplete for '+sid)
        ran = {a['job']['source_tier'] for a in attempts}
        if any(t not in ran for t in tiers[:tier_index]) and tier_index:
            continue  # an earlier tier has not been evaluated for this source
        if any(a['status'] == 'kinematic_candidate' for a in attempts):
            continue
        if args.tier in ran:
            continue
        if args.only and sid not in args.only:
            continue
        eligible.add(sid)
    rows = derive(args.cohort, args.runtime, policy['source_path']['anchor_rule'], eligible)
    jobs, skipped = [], []
    for parent, derived in rows:
        if not derived['ready']:
            skipped.append(dict(source_episode_id=parent['source_episode_id'], reason=derived['error']))
            continue
        for rank, tier, grasp, seed, perturbation in source_candidates(policy):
            if tier != args.tier:
                continue
            mobile, seed_record = mobile_seed(derived, grasp.get('azimuth_offset_deg', 0.), perturbation,
                                              policy['source_path']['anchor_rule'])
            sid = parent['source_episode_id']
            job_id = f"{label}-source-{sid}-{grasp['name']}-s{seed}"
            job = dict(id=job_id, workspace='workspaces/'+job_id, operation='feasible_geometry_experiment',
                       parent_attempt=parent['parent_attempt'],
                       **common_job_fields(parent, policy_sha, args.cohort, args.runtime))
            job.update(parent_attempt_files_sha256=derived['files'],
                       cylindrical_grasp={k: v for k, v in grasp.items() if k != 'name'},
                       grasp_approach=dict(policy['source_path']['grasp_approach']),
                       approach_before_base_admission=True, approach_reuse_admitted_reference=True,
                       geometry_clearance=dict(minimum_m=.003), geometry_seed_rank=seed,
                       mobile_fixture_ik=mobile, admission_only=True,
                       diagnostic_seed=dict(seed_record, source=sid, anchor_rule=policy['source_path']['anchor_rule'],
                                            first_closed_frame=derived['first_closed'], frames=derived['frames'],
                                            reference_base_min=derived['reference_base_min'],
                                            reference_base_max=derived['reference_base_max'],
                                            object_anchor_pose=derived['object_anchor_pose'],
                                            support_floor_geom=derived['support_floor_geom'],
                                            parent_files=derived['files']),
                       source_tier=tier, source_candidate_rank=rank, source_grasp_name=grasp['name'],
                       source_episode_id_v3=sid,
                       scope='Frozen v3 policy source-path kinematic admission; no physics')
            jobs.append(job)
    write_config(f'configs/{label}-source-{args.tier}.json', dict(label=f'{label}-source-{args.tier}', jobs=jobs,
        frozen_policy_sha256=policy_sha, skipped_sources=skipped,
        scope='Frozen v3 source-path admission tier '+args.tier+'; selection by declared rank only'))


MATERIALIZE_GEOMETRY_CODE = r'''
import json,sys
v=json.load(sys.stdin)
sys.path.insert(0,'/mnt/reachy-retarget/releases/'+v['runtime'])
from cluster.materialize_admitted_parent import materialize
out=[]
for item in v['items']:
    try:out.append(dict(item,report=materialize(**item['request']),ready=True))
    except Exception as error:out.append(dict(item,ready=False,error=type(error).__name__+': '+str(error)[:1000]))
print(json.dumps(out))
'''


def admitted_files(receipt, job):
    """The admitted plan files are bound by the job artifact manifest."""
    member = f"workspaces/{job['id']}/plans/{job['id']}/admitted-plan"
    files = {}
    for entry in receipt['manifest']['files']:
        for name in ('scene.xml', 'plan.h5', 'result.json'):
            if entry['path'] == member+'/'+name:
                files[name] = entry['sha256']
    if set(files) != {'scene.xml', 'plan.h5', 'result.json'}:
        raise ValueError('Admitted plan files absent from manifest: '+job['id'])
    return member, files


def select_source(args):
    policy, policy_sha = load_policy()
    label = COHORTS[args.cohort]['label']
    tiers = [t['name'] for t in policy['source_path']['tiers']]
    found = admitted_paths(args.cohort)
    selections, items = [], []
    for parent in parents(args.cohort):
        sid = parent['source_episode_id']
        attempts = sorted(found.get(sid, []), key=lambda a: a['job']['source_candidate_rank'])
        if any(a['status'] == 'not_completed' for a in attempts):
            raise RuntimeError('Incomplete source admission for '+sid)
        admitted = [a for a in attempts if a['status'] == 'kinematic_candidate']
        ran = {a['job']['source_tier'] for a in attempts}
        entry = dict(source_episode_id=sid, source_group=parent['source_group'],
                     evaluated_tiers=[t for t in tiers if t in ran],
                     attempts=[dict(job=a['job']['id'], rank=a['job']['source_candidate_rank'], status=a['status'])
                               for a in attempts])
        usable = admitted
        if not usable:
            entry.update(selected=None, outcome='no_admitted_source_path' if set(tiers) <= ran
                         else 'pending_next_tier')
            selections.append(entry)
            continue
        best = usable[0]
        member, files = admitted_files(best['receipt'], best['job'])
        destination = f"{PVC}/materialized-geometry-parents/{best['job']['id']}"
        entry.update(selected=dict(job=best['job']['id'], rank=best['job']['source_candidate_rank'],
                                   batch=best['batch'], parent_attempt=destination[len(PVC)+1:],
                                   files=files, archive=best['receipt']['archive'],
                                   archive_sha256=best['receipt']['sha256']), outcome='source_path_admitted')
        selections.append(entry)
        items.append(dict(source_episode_id=sid, request=dict(archive=best['receipt']['archive'],
            archive_sha256=best['receipt']['sha256'], member_directory=member, files=files,
            destination=destination)))
    results = pod_python(MATERIALIZE_GEOMETRY_CODE, dict(runtime=args.runtime, items=items)) if items else []
    status = {r['source_episode_id']: r for r in results}
    for entry in selections:
        if entry.get('selected'):
            result = status[entry['source_episode_id']]
            entry['selected']['materialized'] = result['ready']
            if not result['ready']:
                entry['selected']['materialization_error'] = result['error']
    write_config(f'configs/{label}-source-selection{args.suffix}.json', dict(
        schema='frozen-v3-source-selection', frozen_policy_sha256=policy_sha, cohort=args.cohort,
        rule='Minimum declared source candidate rank among kinematic_candidate receipts; physics never read',
        sources=selections))


# ------------------------------------------------------------------ placement

def placement_jobs(args):
    policy, policy_sha = load_policy()
    label = COHORTS[args.cohort]['label']
    selection = json.loads((REPO/f'configs/{label}-source-selection{args.selection_suffix}.json').read_text())
    tier = next(t for t in policy['placement']['tiers'] if t['name'] == args.tier)
    by_id = {p['source_episode_id']: p for p in parents(args.cohort)}
    skip = set()
    if args.after:
        done = json.loads((REPO/f'configs/{label}-placement-selection{args.after}.json').read_text())
        skip = {s['source_episode_id'] for s in done['sources'] if s.get('selected')}
    source_configs = {}
    for path in RUNS.glob(label+'-source-*/manifest.json'):
        for job in json.loads(path.read_text())['jobs']:
            source_configs[job['id']] = job
    jobs = []
    for entry in selection['sources']:
        selected = entry.get('selected')
        if not selected or not selected.get('materialized') or entry['source_episode_id'] in skip:
            continue
        if args.only and entry['source_episode_id'] not in args.only:
            continue
        parent = by_id[entry['source_episode_id']]
        geometry = source_configs[selected['job']]
        for chunk in range(policy['placement']['chunk_count']):
            job_id = f"{label}-place{args.selection_suffix}-{tier['name']}-{entry['source_episode_id']}-chunk{chunk}"
            job = dict(id=job_id, workspace='workspaces/'+job_id, operation='feasible_placement_search',
                       parent_attempt=selected['parent_attempt'],
                       **common_job_fields(parent, policy_sha, args.cohort, args.runtime))
            search = dict(bank=policy['placement']['bank'], chunk_index=chunk,
                          chunk_count=policy['placement']['chunk_count'],
                          adaptive_midpoints=policy['placement']['adaptive_midpoints'],
                          hand_settings=dict(policy['placement']['hand_settings']),
                          refinement_options=dict(policy['placement']['refinement_options']),
                          parameter_overrides=dict(policy['placement']['parameter_overrides']))
            if tier.get('palm_yaw_offset_deg'):
                search['palm_yaw_offset_deg'] = tier['palm_yaw_offset_deg']
            job.update(parent_attempt_files_sha256=selected['files'], cylindrical_grasp=geometry['cylindrical_grasp'],
                       geometry_clearance=dict(minimum_m=.003), parent_geometry_job=selected['job'],
                       placement_search=search, retiming_settings=dict(policy['timing']['retiming_settings']),
                       retimed_source_clearance=dict(policy['timing']['retimed_source_clearance']),
                       carried_object_clearance=dict(support_geoms=[geometry['diagnostic_seed']['support_floor_geom']],
                                                     minimum_m=policy['placement']['carried_object_clearance_m']),
                       placement_tier=tier['name'], placement_tier_rank=policy['placement']['tiers'].index(tier),
                       source_episode_id_v3=entry['source_episode_id'],
                       scope='Frozen v3 placement bank search; kinematic only')
            jobs.append(job)
    write_config(f'configs/{label}-place{args.selection_suffix}-{tier["name"]}.json', dict(
        label=f'{label}-place{args.selection_suffix}-{tier["name"]}', jobs=jobs, frozen_policy_sha256=policy_sha,
        scope='Frozen v3 placement tier '+tier['name']))


def select_placement(args):
    policy, policy_sha = load_policy()
    label = COHORTS[args.cohort]['label']
    tiers = [t['name'] for t in policy['placement']['tiers']]
    results = collections.defaultdict(list)
    for path in sorted(RUNS.glob(f'{label}-place{args.selection_suffix}-*/manifest.json')):
        for job, receipt in receipts(path.parent.name):
            result = (receipt or {}).get('result') or {}
            results[job['source_episode_id_v3']].append(dict(job=job, receipt=receipt,
                status=result.get('status') if receipt else 'not_completed', result=result))
    selection = json.loads((REPO/f'configs/{label}-source-selection{args.selection_suffix}.json').read_text())
    sources = []
    for entry in selection['sources']:
        sid = entry['source_episode_id']
        attempts = results.get(sid, [])
        record = dict(source_episode_id=sid, source_group=entry['source_group'],
                      source_path=entry.get('selected'), attempts=[dict(job=a['job']['id'], status=a['status'],
                      tier=a['job']['placement_tier']) for a in attempts])
        if not entry.get('selected'):
            record.update(selected=None, outcome=entry['outcome'])
            sources.append(record)
            continue
        if any(a['status'] == 'not_completed' for a in attempts):
            raise RuntimeError('Incomplete placement search for '+sid)
        incomplete = [a for a in attempts if not a['result'].get('search_complete', False)
                      and a['status'] != 'kinematic_candidate']
        ranked = []
        for a in attempts:
            if a['status'] == 'kinematic_candidate':
                chosen = a['result']['selected']
                ranked.append(((a['job']['placement_tier_rank'], chosen['selection_rank']), a, chosen))
        ran = {a['job']['placement_tier'] for a in attempts}
        if ranked:
            ranked.sort(key=lambda r: r[0])
            # The lowest tier with any admission wins; all chunks of that tier completed.
            key, a, chosen = ranked[0]
            tier_attempts = [b for b in attempts if b['job']['placement_tier'] == a['job']['placement_tier']]
            if len(tier_attempts) != policy['placement']['chunk_count']:
                raise RuntimeError('Missing placement chunks for '+sid)
            record.update(selected=dict(job=a['job']['id'], tier=a['job']['placement_tier'],
                selection_rank=chosen['selection_rank'], placement_parameters=chosen['placement_parameters'],
                evidence=[dict(path=str(RUNS/b['job']['id'].rsplit('-chunk', 1)[0]), job=b['job']['id'],
                               sha256=b['receipt']['sha256']) for b in tier_attempts]),
                outcome='placement_admitted', incomplete_chunks=[b['job']['id'] for b in incomplete])
        else:
            record.update(selected=None, outcome='no_admitted_placement' if set(tiers) <= ran
                          else 'pending_next_placement_tier',
                          incomplete_chunks=[b['job']['id'] for b in incomplete])
        sources.append(record)
    write_config(f'configs/{label}-placement-selection{args.selection_suffix}{args.suffix}.json', dict(
        schema='frozen-v3-placement-selection', frozen_policy_sha256=policy_sha,
        rule='Lowest placement tier with an admission, then minimum declared bank rank across all chunks; physics never read',
        sources=sources))


# -------------------------------------------------------------------- physics

def physics_jobs(args):
    policy, policy_sha = load_policy()
    label = COHORTS[args.cohort]['label']
    selection = json.loads((REPO/f'configs/{label}-placement-selection{args.selection}.json').read_text())
    source_configs = {}
    for path in RUNS.glob(label+'-source-*/manifest.json'):
        for job in json.loads(path.read_text())['jobs']:
            source_configs[job['id']] = job
    jobs = []
    for entry in selection['sources']:
        if not entry.get('selected'):
            continue
        geometry = source_configs[entry['source_path']['job']]
        for arm in policy['physical_policies']:
            job = copy.deepcopy(geometry)
            for key in ('admission_only', 'scope', 'source_tier', 'source_candidate_rank'):
                job.pop(key, None)
            job_id = f"{label}-physics{args.selection}-{entry['source_episode_id']}-{arm['name']}"
            parameters = copy.deepcopy(entry['selected']['placement_parameters'])
            job.update(id=job_id, workspace='workspaces/'+job_id, runtime_source_sha256=args.runtime,
                       retime=True, retiming_settings=dict(policy['timing']['retiming_settings']),
                       retimed_source_clearance=dict(policy['timing']['retimed_source_clearance']),
                       supported_placement=parameters,
                       carried_object_clearance=dict(support_geoms=[geometry['diagnostic_seed']['support_floor_geom']],
                                                     minimum_m=policy['placement']['carried_object_clearance_m']),
                       source_prefix_velocity_feedforward=arm['source_prefix_velocity_feedforward'],
                       frozen_physical_policy=arm['name'], source_path_job=geometry['id'],
                       placement_selection=dict(job=entry['selected']['job'], tier=entry['selected']['tier'],
                                                rank=entry['selected']['selection_rank']),
                       scope='Frozen v3 policy physical evaluation of the first admitted candidate by declared rank')
            jobs.append(job)
    write_config(f'configs/{label}-physics{args.selection}.json', dict(
        label=f'{label}-physics{args.selection}', jobs=jobs, frozen_policy_sha256=policy_sha,
        scope='One physical attempt per predeclared policy arm per placement-admitted source'))


# ------------------------------------------------------------------- evidence

def attempt_record(job, receipt, batch):
    if receipt is None:
        return dict(job=job['id'], batch=batch, status='not_completed')
    result = receipt.get('result') or {}
    report = result.get('physics_report') or {}
    audit = result.get('actuator_audit') or {}
    record = dict(job=job['id'], batch=batch, status=result.get('status') or receipt['manifest']['status'],
                  operator_archive=dict(path=receipt['archive'], sha256=receipt['sha256'], bytes=receipt['bytes']))
    if report:
        record.update(physics_validated=bool(result.get('physics_validated')),
            all_physics_gates=bool(report.get('gates')) and all(report['gates'].values()),
            failure_reasons=report.get('failure_reasons'),
            actuator_replay_pass=audit.get('actuator_replay_pass'),
            independent_validation_pass=audit.get('independent_validation_pass'),
            source_sequence=report.get('source_sequence'),
            metrics={k: report.get(k) for k in ('max_grasp_drift_m', 'max_grasp_drift_deg', 'final_stable_s',
                     'max_hand_object_penetration_m', 'bilateral_carry_fraction', 'peak_base_velocity') if k in report})
    elif result.get('reason'):
        record['reason'] = str(result['reason'])[:300]
    if result.get('selected'):
        record['selected_rank'] = result['selected'].get('selection_rank')
    return record


def evidence(args):
    policy, policy_sha = load_policy()
    label = COHORTS[args.cohort]['label']
    batches = sorted(p.parent.name for p in RUNS.glob(label+'-*/manifest.json'))
    attempts, by_source = [], collections.defaultdict(list)
    for batch in batches:
        for job, receipt in receipts(batch):
            record = attempt_record(job, receipt, batch)
            sid = job.get('source_episode_id_v3') or job.get('source_episode_id') or Path(job['source']).stem
            record['source_episode_id'] = sid
            for key in ('source_tier', 'source_grasp_name', 'geometry_seed_rank', 'placement_tier',
                        'frozen_physical_policy', 'source_prefix_velocity_feedforward'):
                if key in job:
                    record[key] = job[key]
            attempts.append(record)
            by_source[sid].append(record)
    cohort_ids = [s for s in args.sources] if args.sources else None
    placement = json.loads((REPO/f'configs/{label}-placement-selection{args.selection}.json').read_text())
    arms = [a['name'] for a in policy['physical_policies']]
    sources, stages = [], {arm: collections.Counter() for arm in arms}
    materialized = {}
    if args.cohort == 'v3':
        for item in json.loads((REPO/COHORTS['v3']['parents']).read_text())['sources']:
            materialized[item['source_episode_id']] = item
    selection_by = {s['source_episode_id']: s for s in placement['sources']}
    universe = cohort_ids or sorted(set(selection_by) | set(materialized))
    for sid in universe:
        entry = selection_by.get(sid)
        row = dict(source_episode_id=sid, source_group=(entry or materialized.get(sid, {})).get('source_group'))
        if entry is None:
            parent = materialized.get(sid, {})
            row['stage'] = 'baseline_or_materialization_failed'
            row['detail'] = parent.get('reason') or parent.get('error')
            for arm in arms:
                row[arm] = 'fail:'+row['stage']
                stages[arm][row['stage']] += 1
            sources.append(row)
            continue
        row['source_path'] = (entry.get('source_path') or {}).get('job')
        row['placement'] = entry.get('selected')
        for arm in arms:
            if not entry.get('selected'):
                stage = entry['outcome']
                row[arm] = 'fail:'+stage
            else:
                physics = [a for a in by_source[sid] if a.get('frozen_physical_policy') == arm
                           and '-physics'+args.selection+'-' in a['job']]
                if len(physics) != 1:
                    stage = 'physics_missing'
                elif physics[0]['status'] == 'physical_pass' and physics[0].get('actuator_replay_pass') \
                        and physics[0].get('independent_validation_pass') is not False:
                    stage = 'pass'
                else:
                    stage = 'physics_'+str(physics[0]['status'])
                    row[arm+'_failure_reasons'] = physics[0].get('failure_reasons') or physics[0].get('reason')
                row[arm] = stage
            stages[arm][stage] += 1
        sources.append(row)
    denominator = len(universe)
    value = dict(schema='frozen-can-generalization-evidence-v3', cohort=args.cohort, label=label,
        policy=str(POLICY.relative_to(REPO)), frozen_policy_sha256=policy_sha,
        evaluation_split=COHORTS[args.cohort]['split'], denominator=denominator,
        scores={arm: dict(passed=stages[arm]['pass'], denominator=denominator,
                          rate=round(stages[arm]['pass']/denominator, 4) if denominator else None,
                          stage_breakdown=dict(stages[arm])) for arm in arms},
        arm_union_note='Arms are separate predeclared policies; their union is not a frozen rate.',
        batches=batches, attempt_count=len(attempts),
        attempt_status_counts=dict(collections.Counter(a['status'] for a in attempts)),
        sources=sources, attempts=attempts)
    out = REPO/args.output
    out.write_text(json.dumps(value, indent=2)+'\n')
    print(out, json.dumps(value['scores']))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('stage', 'materialize', 'source-jobs', 'select-source', 'placement-jobs',
                 'select-placement', 'physics-jobs', 'evidence'):
        p = sub.add_parser(name)
        p.add_argument('--runtime', required=name not in ('select-placement', 'evidence'))
        p.add_argument('--cohort', choices=sorted(COHORTS), default='v3')
        p.add_argument('--only', nargs='*', help='Restrict to these source ids (smoke tests only)')
        if name == 'source-jobs':
            p.add_argument('--tier', required=True)
        if name == 'select-source':
            p.add_argument('--suffix', default='')
        if name == 'placement-jobs':
            p.add_argument('--tier', required=True)
            p.add_argument('--selection-suffix', default='')
            p.add_argument('--after', help='Skip sources already admitted in this placement selection suffix')
        if name == 'select-placement':
            p.add_argument('--selection-suffix', default='')
            p.add_argument('--suffix', default='')
        if name in ('physics-jobs', 'evidence'):
            p.add_argument('--selection', default='')
        if name == 'evidence':
            p.add_argument('--output', required=True)
            p.add_argument('--sources', nargs='*')
    args = parser.parse_args()
    {'stage': stage, 'materialize': materialize, 'source-jobs': source_jobs, 'select-source': select_source,
     'placement-jobs': placement_jobs, 'select-placement': select_placement,
     'physics-jobs': physics_jobs, 'evidence': evidence}[args.command](args)


if __name__ == '__main__':
    main()
