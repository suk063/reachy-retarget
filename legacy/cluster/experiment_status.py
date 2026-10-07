"""Collect one experiment batch without rescanning unrelated historical archives."""
import argparse
import collections
import datetime
import json
from pathlib import Path
import subprocess

REMOTE = r'''
import concurrent.futures,json,sys,time
from pathlib import Path
r=Path('/mnt/reachy-retarget'); label=sys.argv[1]
m=json.loads((r/'experiments'/label/'manifest.json').read_text())
def read(job):
 folder=r/'queue/tasks'/job['id']; p=folder/'result.json'
 errors=[]
 def record(path,missing):
  try:return json.loads(path.read_text())
  except FileNotFoundError:return missing
  except (OSError,ValueError) as e:
   errors.append({'path':str(path),'error':str(e)})
   return {'status':'unreadable_artifact','path':str(path),'error':str(e)}
 result=record(p,{'status':'pending'})
 claims=[]
 for p in (folder/'attempts').glob('*/claim.json'):
  c=record(p,{}); end=p.parent/'result.json'
  if 'time' not in c or 'worker' not in c:continue
  c['finished_unix']=record(end,{}).get('finished_unix')
  claims.append(c)
 return {'job':job,'result':result,'claims':claims,'artifact_read_errors':errors}
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool: jobs=list(pool.map(read,m['jobs']))
print(json.dumps({'manifest':m,'jobs':jobs,'collected_unix':time.time()}))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pod',required=True);p.add_argument('--label',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    v=json.loads(subprocess.check_output(['kubectl','-n','erl-ucsd','exec',a.pod,'--','python','-c',REMOTE,a.label],text=True))
    claims=[c for j in v['jobs'] for c in j['claims']]
    events=[]
    for c in claims:
        events.append((c['time'],1))
        if c['finished_unix']:events.append((c['finished_unix'],-1))
    n=peak=0
    for _,d in sorted(events):n+=d;peak=max(peak,n)
    v['summary']=dict(jobs=len(v['jobs']),claimed=len(claims),workers_used=len({c['worker'] for c in claims}),
        peak_overlapping_claims=peak,statuses=dict(collections.Counter(j['result']['status'] for j in v['jobs'])),
        peak_overlapping_claims_is_upper_bound=any(not c.get('finished_unix') for c in claims),
        open_or_interrupted_claims=sum(not c.get('finished_unix') for c in claims),
        artifact_read_errors=sum(len(j['artifact_read_errors']) for j in v['jobs']),
        physical_passes=sum(j['result'].get('physics_validated',False) for j in v['jobs']),
        alternate_model_passes=sum(j['result'].get('alternate_model_physics_validated',False) for j in v['jobs']),
        unique_sources=len({j['job']['parent_task'] for j in v['jobs']}))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(v,indent=2)+'\n')
    print(json.dumps(v['summary'],indent=2))


if __name__=='__main__':main()
