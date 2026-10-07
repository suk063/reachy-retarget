"""Refresh idle worker containers once, preserving pod UIDs and the shared PVC.

Explicit maintenance command: active children and the access pod are skipped.
The mounted ConfigMap must already contain this exact service implementation.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess

REMOTE=r'''
import hashlib,json,os,signal,socket,sys,time
from pathlib import Path
expected=sys.argv[1]; service=Path('/service/worker_service.py')
heartbeat=Path('/mnt/reachy-retarget/workers')/(os.environ.get('POD_NAME',socket.gethostname())+'.json')
if heartbeat.exists() and json.loads(heartbeat.read_text()).get('service_revision')==expected:
 print('Service already updated; container untouched',flush=True);sys.exit(0)
deadline=time.time()+45
while hashlib.sha256(service.read_bytes()).hexdigest()!=expected:
 if time.time()>deadline:raise RuntimeError('Mounted service not yet updated; container untouched')
 time.sleep(1)
if b'/service/worker_service.py' not in Path('/proc/1/cmdline').read_bytes():raise RuntimeError('Unexpected PID1')
children=Path('/proc/1/task/1/children').read_text().strip()
if children:raise RuntimeError('Active worker children; container untouched: '+children)
print('Verified idle; refreshing container with service '+expected,flush=True)
# Python installs a SIGINT handler. Linux PID-namespace init ignores signals
# without a registered handler, including an in-namespace SIGKILL request.
os.kill(1,signal.SIGINT)
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exclude-pod',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();k=['kubectl','-n','erl-ucsd']
    pods=json.loads(subprocess.check_output(k+['get','pods','-l','app.kubernetes.io/name=reachy-retarget-worker','-o','json']))['items']
    sha=hashlib.sha256(Path('cluster/worker_service.py').read_bytes()).hexdigest()
    def run(pod):
        name=pod['metadata']['name']
        before=dict(pod=name,uid=pod['metadata']['uid'],node=pod['spec']['nodeName'],
                    restarts=pod['status']['containerStatuses'][0]['restartCount'])
        if name==a.exclude_pod:return dict(before,skipped='access pod retained')
        r=subprocess.run(k+['exec',name,'--','python','-c',REMOTE,sha],text=True,capture_output=True)
        return dict(before,returncode=r.returncode,output=r.stdout.strip(),diagnostic=r.stderr.strip(),
                    refresh_requested='Verified idle; refreshing container' in r.stdout)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:results=list(pool.map(run,pods))
    value=dict(service_sha256=sha,pods=results,source='Explicit one-time idle service refresh; no pod/deployment/PVC deleted')
    a.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'requested':sum(r.get('refresh_requested',False) for r in results),'skipped_or_rejected':[r for r in results if not r.get('refresh_requested')]}))


if __name__=='__main__':main()
