"""Predeclare a contact-control bank and publish useful parallel experiments."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pod',required=True)
    parser.add_argument('--label',default='feasible-contact-v1')
    parser.add_argument('--bank',choices=['contact','tracking'],default='contact')
    parser.add_argument('--retime',action='store_true')
    parser.add_argument('--include-maniskill',action='store_true')
    parser.add_argument('--reuse-plans',action='store_true')
    parser.add_argument('--contact-profile',choices=['global-contact-4ms'])
    args=parser.parse_args()
    from reachy_retarget.cluster_pipeline import identifier
    identifier(args.label)
    evidence=json.loads(Path('docs/cluster-run-evidence.json').read_text())
    groups=defaultdict(list);baseline=[]
    for item in evidence['jobs']:
        job,result=item['job'],item['result']
        if not job['id'].startswith('legacy-all-v1-'):continue
        physics=result.get('physics_report') or {}
        if not physics:continue
        task=physics['plan']['task']
        baseline.append(dict(episode=job['source_episode_id'],task=task,
                             status=physics['status'],source_group=physics['source_sequence']))
        if job['dataset']!='maniskill' and not result['diagnostic_baseline']:
            groups[task].append(job)
    from reachy_retarget.feasible_experiments import VARIANTS, TRACKING_VARIANTS
    bank = VARIANTS if args.bank == 'contact' else TRACKING_VARIANTS
    jobs=[];held_out=[]
    for task,rows in sorted(groups.items()):
        for i,parent in enumerate(sorted(rows,key=lambda x:x['id'])):
            if i>=2:
                held_out.append(parent['id']);continue
            for name in bank:
                task_id=f'{args.label}-{parent["source_episode_id"]}-{name}'
                jobs.append(dict(id=task_id,operation='feasible_experiment',dataset=parent['dataset'],
                    parent_task=parent['id'],parent_workspace=parent['workspace'],workspace='workspaces/'+task_id,
                    source=parent['source'],source_provenance=parent['source_provenance'],
                    variant=name,evaluation_split='development',task=task))
    if args.include_maniskill:
        parent=next(j['job'] for j in evidence['jobs'] if j['job']['id']=='legacy-all-v1-8a7847a26e8fbb00783e4114')
        for x in (.25,.30):
            for name in bank:
                task_id=f'{args.label}-{parent["source_episode_id"]}-base{int(x*100)}-{name}'
                jobs.append(dict(id=task_id,operation='feasible_experiment',dataset='maniskill',
                    parent_task=parent['id'],parent_workspace=parent['workspace'],workspace='workspaces/'+task_id,
                    source=parent['source'],source_provenance=parent['source_provenance'],
                    base_xyyaw=[x,-.653,3.141592653589793],variant=name,evaluation_split='development',task='PickCube-v1'))
    if args.retime:
        for job in jobs:job['retime']=True
    if args.reuse_plans:
        parents={item['job']['id']:item['result'] for item in evidence['jobs']}
        for job in jobs:
            job['parent_attempt']=str(Path(parents[job['parent_task']]['physics_attempt']).relative_to('/mnt/reachy-retarget'))
    if args.contact_profile:
        for job in jobs:job['contact_profile']=args.contact_profile
    manifest=dict(label=args.label,bank=args.bank,jobs=jobs,held_out=held_out,baseline=baseline,
                  denominator='All20 original verified dynamics inputs remain listed; no failed input removed from reporting.',
                  policy='First two pad-aligned non-ManiSkill inputs per task develop a fixed6variant control bank; later inputs are not used for selection. All were previously inspected; not an unseen-corpus claim.')
    output=Path('configs')/(args.label+'.json');output.write_text(json.dumps(manifest,indent=2)+'\n')
    remote=r'''
import concurrent.futures,json,sys
from pathlib import Path
r=Path('/mnt/reachy-retarget');runtime=json.loads((r/'runtime.json').read_text());sys.path.insert(0,runtime['source'])
from reachy_retarget.cluster_pipeline import enqueue,atomic
from reachy_retarget.feasible_experiments import stage
m=json.load(sys.stdin)
receipt=r/'experiments'/m['label']/'manifest.json'
if receipt.exists():
    if json.loads(receipt.read_text())!=m:raise ValueError('Immutable batch ID conflict')
else:atomic(receipt,m)
jobs=m['jobs']
def prepare(job):
    w=r/job['workspace']
    if not w.exists():stage(r,job)
    elif json.loads((w/'input-manifest.json').read_text())['job']!=job:raise ValueError('Workspace conflict')
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(prepare,jobs))
for job in jobs:enqueue(r,job)
print(json.dumps({'queued':len(jobs),'development_sources':len({j['parent_task'] for j in jobs}),'held_out':len(m['held_out'])}))
'''
    subprocess.run(['kubectl','-n','erl-ucsd','exec','-i',args.pod,'--','/mnt/reachy-retarget/runtime/run-python','-c',remote],input=json.dumps(manifest),text=True,check=True)


if __name__=='__main__':main()
