"""Read actual pool, queue and archive metadata without modifying cluster data."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import subprocess

REMOTE = r'''
import json,shutil,time
from pathlib import Path
root=Path('/mnt/reachy-retarget')
runtime=json.loads((root/'runtime.json').read_text())
workers=[json.loads(p.read_text()) for p in (root/'workers').glob('*.json')]
workers=[w for w in workers if time.time()-w['time']<60]
jobs=[]
for path in sorted((root/'queue/tasks').glob('*/job.json')):
    job=json.loads(path.read_text());result=path.parent/'result.json'
    value=json.loads(result.read_text()) if result.exists() else {'status':'pending'}
    claims=[]
    for claim_path in sorted((path.parent/'attempts').glob('*/claim.json')):
        claim=json.loads(claim_path.read_text()); finished=claim_path.parent/'result.json'
        final=json.loads(finished.read_text()) if finished.exists() else {}
        claims.append(dict(worker=claim['worker'],start_unix=claim['time'],
                           finished_unix=final.get('finished_unix'),status=final.get('status')))
    jobs.append({'job':job,'result':value,'agent_reviewed':(path.parent/'agent/response.json').exists(),
                 'attempt_count':len(claims),'attempt_claims':claims})
archives=[]
for path in sorted((root/'datasets').glob('*/episodes/*.json')):
    m=json.loads(path.read_text());fields=m.get('fields',{})
    archives.append({'path':str(path.with_suffix('.hdf5')),'sha256':m.get('hdf5_sha256'),
                     'dataset':m['dataset_id'],'episode':m['episode_id'],
                     'source_group':m['source_group'],'source_sequence':m['source_sequence'],
                     'status':m.get('status'),'rows':m.get('row_count'),'timed':m.get('timed'),
                     'objects':list(m.get('objects',{})),
                     'object_pose_fields':[n for n in fields if n.startswith('observation/objects/') and n.endswith('/pose')],
                     'rgb_fields':[n for n in fields if n.endswith('/rgb')],
                     'physics_validated':m.get('physics_validated',False),
                     'missing_fields':m.get('missing_fields',[])})
print(json.dumps({'runtime':runtime,'workers_live':len(workers),
                  'physical_hosts':len({w.get('node') for w in workers}),
                  'working_tasks':[w['task'] for w in workers if w['status']=='working'],
                  'free_bytes':shutil.disk_usage(root).free,'jobs':jobs,'archives':archives,
                  'initial_source_tasks':[j['id'] for j in json.loads((root/'queue/initial-jobs.json').read_text())]}))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pod',required=True)
    parser.add_argument('--namespace',default='erl-ucsd')
    parser.add_argument('--output',type=Path,default=Path('docs/cluster-run-evidence.json'))
    args=parser.parse_args()
    kube=['kubectl','-n',args.namespace]
    def get(*params):
        return json.loads(subprocess.check_output(kube+['get',*params,'-o','json'],text=True))
    deployment=get('deployment','reachy-retarget-worker')
    pvc=get('pvc','reachy-retarget-data')
    pods=get('pods','-l','app.kubernetes.io/name=reachy-retarget-worker')['items']
    result=json.loads(subprocess.check_output(kube+['exec',args.pod,'--','python','-c',REMOTE],text=True))
    result.update(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  namespace=args.namespace,context=subprocess.check_output(['kubectl','config','current-context'],text=True).strip(),
                  deployment={'name':deployment['metadata']['name'],'created':deployment['metadata']['creationTimestamp'],
                              'desired':deployment['spec']['replicas'],'ready':deployment['status'].get('readyReplicas',0)},
                  pvc={'name':pvc['metadata']['name'],'phase':pvc['status']['phase'],
                       'capacity':pvc['status'].get('capacity'), 'access_modes':pvc['spec']['accessModes']},
                  pods={'running':sum(p['status']['phase']=='Running' for p in pods),
                        'distinct_hosts':len({p['spec'].get('nodeName') for p in pods if p['status']['phase']=='Running'})})
    job_counts=Counter(j['result']['status'] for j in result['jobs'])
    archives=result['archives']
    tasks={j['job']['id']:j for j in result['jobs']}
    successors={}
    for task,item in tasks.items():
        parent=item['job'].get('supersedes')
        if parent:
            if parent in successors:raise ValueError('Ambiguous replacement lineage: '+parent)
            successors[parent]=task
    selected=[]
    for task in result['initial_source_tasks']:
        visited=set()
        while task in successors:
            if task in visited:raise ValueError('Cyclic replacement lineage')
            visited.add(task);task=successors[task]
        item=tasks[task];state=item['result']
        selected.append({'task':task,'dataset':item['job']['dataset'],'status':state['status'],
                         'archive':state.get('archive'),'retarget_archive':state.get('retarget_archive'),
                         'retargeted':state.get('retargeted',False),
                         'physics_validated':state.get('physics_validated',False)})
    result['current_source_tasks']=selected
    # Initial source publication and subsequent retarget/reconstruction stages
    # are separate immutable jobs, not necessarily replacement attempts.
    result['executed_pipeline_tasks']=[
        dict(task=item['job']['id'],dataset=item['job']['dataset'],operation=item['job']['operation'],
             status=item['result']['status'],retargeted=item['result'].get('retargeted',False),
             physics_validated=item['result'].get('physics_validated',False))
        for item in result['jobs'] if item['job']['operation'] in
        ('legacy_pipeline','full_state_rollout','reconstruct_robocasa','source_pose_ik')]
    result['counts']={'task_statuses':dict(job_counts),'common_archives':len(archives),
                      'dataset_families':len({a['dataset'] for a in archives}),
                      'current_source_tasks':len(selected),
                      'current_source_statuses':dict(Counter(v['status'] for v in selected)),
                      'distinct_recorded_source_groups':len({a['source_group'] for a in archives}),
                      'archives_with_rgb':sum(bool(a['rgb_fields']) for a in archives),
                      'physics_validated_archives':sum(a['physics_validated'] for a in archives),
                      'note':'Recorded source-group count is provenance grouping, not proof of statistical independence. Retries and derived views do not add source groups. current_source_tasks follows initial publication/review jobs; executed_pipeline_tasks lists later compute stages separately.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('verified_utc','deployment','pvc','workers_live','physical_hosts','working_tasks','counts')},indent=2))


if __name__=='__main__':main()
