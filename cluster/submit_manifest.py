"""Stage and enqueue an explicit immutable job list on existing workers."""
import argparse
import json
from pathlib import Path
import subprocess

REMOTE=r'''
import concurrent.futures,json,sys,time
from pathlib import Path
r=Path('/mnt/reachy-retarget');runtime=json.loads((r/'runtime.json').read_text());sys.path.insert(0,runtime['source'])
from reachy_retarget.cluster_pipeline import atomic,enqueue,identifier
from reachy_retarget.feasible_experiments import stage
m=json.load(sys.stdin);identifier(m['label']);jobs=m['jobs']
if len({j['id'] for j in jobs})!=len(jobs):raise ValueError('Duplicate immutable job ID')
receipt=r/'experiments'/m['label']/'manifest.json'
if receipt.exists():
 if json.loads(receipt.read_text())!=m:raise ValueError('Immutable manifest conflict')
else:atomic(receipt,m)
def prepare(job):
 try:
  if job['operation'] in ('feasible_experiment','feasible_geometry_experiment','feasible_geometry_search'):
   w=r/job['workspace']
   if not w.exists():stage(r,job)
   elif json.loads((w/'input-manifest.json').read_text())['job']!=job:raise ValueError('Immutable workspace conflict')
  # Publish each complete input immediately; workers need not wait for the
  # slowest workspace in a large batch. enqueue atomically publishes job.json.
  enqueue(r,job)
  return {'id':job['id'],'queued':True,'time_unix':time.time()}
 except Exception as error:
  return {'id':job['id'],'queued':False,'error':type(error).__name__+': '+str(error)}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(prepare,jobs))
summary={'label':m['label'],'queued':sum(v['queued'] for v in results),'failed':sum(not v['queued'] for v in results)}
atomic(receipt.parent/('submission-'+str(time.time_ns())+'.json'),dict(summary,results=results))
print(json.dumps(summary))
if summary['failed']:raise SystemExit(1)
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pod',required=True);p.add_argument('manifest',type=Path);a=p.parse_args()
    subprocess.run(['kubectl','-n','erl-ucsd','exec','-i',a.pod,'--','/mnt/reachy-retarget/runtime/run-python','-c',REMOTE],
                   input=a.manifest.read_text(),text=True,check=True)


if __name__=='__main__':main()
