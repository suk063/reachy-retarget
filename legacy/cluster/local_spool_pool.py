"""Use persistent pods for immutable experiments without shared output writes.

Each pod runs one detached process at a time on its local disk. Completed and
failed artifacts are copied to the operator's disk, verified against every
manifest hash, and retained separately from published PVC datasets. An
interrupted control connection does not terminate or silently duplicate work.
"""
import argparse
import base64
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath
import queue
import re
import shutil
import subprocess
import tarfile
import threading
import time

from cluster.local_spool_experiment import digest, write_json


BOOTSTRAP = r'''
import base64,fcntl,hashlib,io,json,os,shutil,subprocess,sys,tarfile,time
from pathlib import Path,PurePosixPath
v=json.load(sys.stdin);base=Path(v['base']);base.mkdir(parents=True,exist_ok=True)
if shutil.disk_usage(base).free < 50_000_000_000:raise OSError('50 GB local disk reserve')
job=v['job'];outer=base/job['id'];outer.mkdir(exist_ok=True)
spool=outer/'artifact';status=outer/'launch.json'
if status.exists():
 old=json.loads(status.read_text())
 if old['job']!=job or old['launcher_sha256']!=v['launcher_sha256']:raise ValueError('Immutable local launch conflict')
 print(json.dumps(old));raise SystemExit(0)
slot=int(v.get('slot',0))
lock=Path('/tmp/reachy-retarget-pod-execution.lock'+('' if slot==0 else '.'+str(slot))).open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:print(json.dumps(dict(pod_busy=True)));raise SystemExit(0)
runtime_root=None
if v.get('runtime_bundle'):
 bundle=v['runtime_bundle'];content=base64.b64decode(bundle['base64']);pin=job['runtime_source_sha256']
 if bundle['source_sha256']!=pin or hashlib.sha256(content).hexdigest()!=bundle['sha256']:raise ValueError('Local runtime bundle mismatch')
 runtime_root=Path('/tmp/reachy-retarget-runtimes');runtime_root.mkdir(exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(content),mode='r:gz') as archive:
  for member in archive:
   name=PurePosixPath(member.name)
   if not member.isfile() or name.is_absolute() or '..' in name.parts or not (str(name).startswith('releases/'+pin+'/') or str(name)=='provenance/'+pin+'.json'):raise ValueError('Unsafe local runtime member')
   path=runtime_root/str(name)
   if not path.resolve().is_relative_to(runtime_root.resolve()):raise ValueError('Local runtime path escaped root')
   value=archive.extractfile(member).read();path.parent.mkdir(parents=True,exist_ok=True)
   if path.exists():
    if path.read_bytes()!=value:raise ValueError('Immutable local runtime conflict')
   else:
    with path.open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
if hashlib.sha256(v['launcher'].encode()).hexdigest()!=v['launcher_sha256']:raise ValueError('Launcher hash mismatch')
def save(path,content):
 with path.open('x') as f:f.write(content);f.flush();os.fsync(f.fileno())
save(outer/'launcher.py',v['launcher']);save(outer/'job.json',json.dumps(job,sort_keys=True))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
log=(outer/'stdout.log').open('xb')
command=['/mnt/reachy-retarget/runtime/run-python',str(outer/'launcher.py'),'--spool',str(spool),'--job',str(outer/'job.json'),'--release',job['runtime_source_sha256']]
if runtime_root is not None:command+=['--runtime-root',str(runtime_root)]
child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=str(outer),start_new_session=True,pass_fds=(lock.fileno(),))
record=dict(job=job,launcher_sha256=v['launcher_sha256'],pid=child.pid,spool=str(spool),outer=str(outer),started_unix_s=time.time(),pid_start_ticks=(Path('/proc')/str(child.pid)/'stat').read_text().split()[21])
save(status,json.dumps(record,sort_keys=True));print(json.dumps(record))
'''

POLL = r'''
import json,os,sys
from pathlib import Path
v=json.load(sys.stdin);outer=Path(v['outer']);spool=Path(v['spool']);done=spool/'portable-artifact-manifest.json'
alive=False
try:
 proc=Path('/proc')/str(v['pid']);state=(proc/'stat').read_text().split()
 alive=state[2]!='Z' and (v.get('pid_start_ticks')==state[21] if v.get('pid_start_ticks') else v['spool'].encode() in (proc/'cmdline').read_bytes())
except FileNotFoundError:pass
result=dict(alive=alive,complete=done.exists())
if done.exists():result['manifest']=json.loads(done.read_text())
if (spool/'progress.json').exists():result['progress']=json.loads((spool/'progress.json').read_text())
if not alive and not done.exists():result['log_tail']=(outer/'stdout.log').read_text(errors='replace')[-6000:]
print(json.dumps(result))
'''


PVC_BACKUP = r'''
import hashlib,json,os,shutil,subprocess,sys,tarfile,time
from pathlib import Path,PurePosixPath
v=json.load(sys.stdin);target=Path(v['path']);spool=Path(v['spool'])
if shutil.disk_usage('/mnt/reachy-retarget').free-v['reserve_bytes']<50_000_000_000:raise OSError('50 decimal GB shared-storage reserve')
target.parent.mkdir(parents=True,exist_ok=True)
reused=target.exists()
part=target if reused else target.with_name('.'+target.name+'.partial-'+str(time.time_ns()))
try:
 if not reused:
  with part.open('xb') as handle:
   subprocess.run(['tar','-czf','-','-C',str(spool),'.'],stdout=handle,check=True,stderr=subprocess.PIPE,timeout=600,env=dict(os.environ,COPYFILE_DISABLE='1'))
   handle.flush();os.fsync(handle.fileno())
 # An archive left by an interrupted orchestrator is reused only after every
 # regular file again matches the immutable artifact manifest.
 result=verify_archive(part,v['manifest'])
 sha=hashlib.sha256();size=0
 with part.open('rb') as handle:
  for block in iter(lambda:handle.read(8*1024*1024),b''):sha.update(block);size+=len(block)
 if not reused:os.link(part,target)
finally:
 if not reused:part.unlink(missing_ok=True)
fd=os.open(target.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
log=Path(v['outer'])/'stdout.log'
print(json.dumps(dict(path=str(target),sha256=sha.hexdigest(),bytes=size,result=result,reused_verified_archive=reused,
 stdout_log=log.read_text(errors='replace')[-2_000_000:] if log.exists() else None)))
'''

PVC_READBACK = r'''
import hashlib,json,sys
v=json.load(sys.stdin);sha=hashlib.sha256();size=0
with open(v['path'],'rb') as handle:
 for block in iter(lambda:handle.read(8*1024*1024),b''):sha.update(block);size+=len(block)
print(json.dumps(dict(sha256=sha.hexdigest(),bytes=size)))
'''


def safe_name(value):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,199}', value):
        raise ValueError('Unsafe immutable job or batch name')
    return value


def verify_archive(path, manifest):
    """Verify regular bytes without extracting external symlinks on the host."""
    expected = {item['path']: item for item in manifest['files']}
    if len(expected) != len(manifest['files']):
        raise ValueError('Duplicate artifact manifest path')
    seen = set()
    result = None
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            name = str(PurePosixPath(member.name))
            if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
                raise ValueError('Unsafe artifact member')
            if not member.isfile():
                continue
            if name in seen:
                raise ValueError('Duplicate archive file')
            seen.add(name)
            stream = archive.extractfile(member)
            if name == 'portable-artifact-manifest.json':
                if json.load(stream) != manifest:
                    raise ValueError('Artifact manifest changed during backup')
                continue
            if name not in expected:
                raise ValueError('Unbound archive file: '+name)
            sha = hashlib.sha256()
            value = bytearray() if name == 'result.json' else None
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                sha.update(block)
                if value is not None:
                    value.extend(block)
            if member.size != expected[name]['bytes'] or sha.hexdigest() != expected[name]['sha256']:
                raise ValueError('Artifact checksum mismatch: '+name)
            if value is not None:
                result = json.loads(value)
    if seen != set(expected) | {'portable-artifact-manifest.json'}:
        raise ValueError('Incomplete artifact backup')
    return result


# Captured at import so a later edit of this file cannot change the remote
# verifier sent by an already running batch process.
VERIFY_ARCHIVE_SOURCE = inspect.getsource(verify_archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--namespace', default='erl-ucsd')
    parser.add_argument('--exclude-pod', action='append', default=[])
    parser.add_argument('--pods', type=int, default=None, help='Use at most this many ready pods (default: all)')
    parser.add_argument('--poll-seconds', type=float, default=8)
    parser.add_argument('--runtime-bundle', type=Path)
    parser.add_argument('--lease-dir', type=Path,
                        default=Path(__file__).resolve().parents[1]/'runs/local-spool-pool/.pod-leases',
                        help='Operator-side per-pod leases shared by concurrent batch processes')
    parser.add_argument('--slots-per-pod', type=int, default=2,
                        help='Concurrent single-threaded jobs per pod (pods have a 2-CPU limit)')
    parser.add_argument('--resume-only', action='store_true',
                        help='Only finish and back up jobs that already have a retained launch; launch nothing new')
    parser.add_argument('--archive-storage', choices=('pvc', 'local'), default='pvc',
                        help='Keep verified operator archives on the shared PVC (default) or on this disk')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    safe_name(manifest['label'])
    jobs = manifest['jobs']
    for job in jobs:
        safe_name(job['id'])
        if not re.fullmatch('[0-9a-f]{64}', job.get('runtime_source_sha256', '')):
            raise ValueError('Each job needs an immutable published runtime pin')
    if len({job['id'] for job in jobs}) != len(jobs):
        raise ValueError('Duplicate job IDs')
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output/'manifest.json'
    if snapshot.exists() and json.loads(snapshot.read_text()) != manifest:
        raise ValueError('Immutable local batch conflict')
    write_json(snapshot, manifest)
    launcher_path = args.output/'launcher.py'
    launcher = (launcher_path if launcher_path.exists() else Path(__file__).with_name('local_spool_experiment.py')).read_text()
    launcher_hash = hashlib.sha256(launcher.encode()).hexdigest()
    if not launcher_path.exists():
        launcher_path.write_text(launcher)
    runtime_bundle = None
    if args.runtime_bundle:
        receipt = json.loads(args.runtime_bundle.with_suffix(args.runtime_bundle.suffix+'.json').read_text())
        content = args.runtime_bundle.read_bytes()
        if hashlib.sha256(content).hexdigest() != receipt['bundle_sha256']:
            raise ValueError('Local runtime bundle checksum mismatch')
        if not any(job['runtime_source_sha256'] == receipt['source_sha256'] for job in jobs):
            raise ValueError('No job explicitly binds the supplied local runtime')
        runtime_bundle = dict(source_sha256=receipt['source_sha256'], sha256=receipt['bundle_sha256'],
                              base64=base64.b64encode(content).decode())
        write_json(args.output/'local-runtime-receipt.json', receipt)
    kubectl = ['kubectl', '-n', args.namespace]
    inventory = json.loads(subprocess.check_output(kubectl+['get', 'pods', '-l',
        'app.kubernetes.io/name=reachy-retarget-worker', '-o', 'json']))['items']
    pods = [item['metadata']['name'] for item in inventory
            if item['status']['phase'] == 'Running'
            and any(c.get('ready') for c in item['status'].get('containerStatuses', []))
            and item['metadata']['name'] not in args.exclude_pod][:args.pods or None]
    if not pods:
        raise ValueError('No ready unreserved persistent pods')
    write_json(args.output/'pod-inventory.json', inventory)
    previous = json.loads((args.output/'status.json').read_text()) if (args.output/'status.json').exists() else {}
    work = queue.Queue()
    assigned = {pod: queue.Queue() for pod in pods}
    unavailable = []
    for job in jobs:
        destination = args.output/job['id']
        if (destination/'backup.json').exists():
            continue
        if (destination/'launch.json').exists():
            launch = json.loads((destination/'launch.json').read_text())
            pod = launch.get('pod') or previous.get('states', {}).get(job['id'], {}).get('pod')
            if pod not in assigned:
                unavailable.append(dict(job=job['id'], original_pod=pod,
                    reason='Retained scratch launch needs its original pod; no duplicate execution'))
            else:
                assigned[pod].put(job)
        elif not args.resume_only:
            work.put(job)
    reporting_lock = threading.Lock()
    transfer_lock = threading.Lock()
    backup_slots = threading.Semaphore(8)
    reserved_bytes = 0
    states = previous.get('states', {})
    started = previous.get('started_unix_s', time.time())

    def report(job, **values):
        with reporting_lock:
            states[job['id']] = dict(states.get(job['id'], {}), **values)
            write_json(args.output/'status.json', dict(label=manifest['label'],
                started_unix_s=started, updated_unix_s=time.time(), pods=pods,
                total_jobs=len(jobs), states=states, durable_pvc_export=False))
            print(json.dumps(dict(job=job['id'], **values)), flush=True)

    def remote(pod, code, value, timeout=90):
        process = subprocess.run(kubectl+['exec', '-i', pod, '--', 'python3', '-c', code],
            input=json.dumps(value), text=True, capture_output=True, timeout=timeout, check=True)
        return json.loads(process.stdout)

    def pvc_backup(pod, job, launch, binding, destination):
        # The archive is written and verified by the executing pod directly
        # on shared storage, then re-read on a different pod; only receipts
        # and the log are kept on the operator disk.
        path = str(PurePosixPath('/mnt/reachy-retarget/operator-backups/local-spool-pool')
                   / manifest['label'] / job['id'] / 'artifact.tar.gz')
        code = ('import hashlib,json,tarfile\nfrom pathlib import PurePosixPath\n'
                + VERIFY_ARCHIVE_SOURCE + PVC_BACKUP)
        with backup_slots:
            written = remote(pod, code, dict(path=path, spool=launch['spool'], outer=launch['outer'],
                manifest=binding, reserve_bytes=2*binding['artifact_bytes']+10_000_000), timeout=900)
        verifier = next(p for p in pods if p != pod) if len(pods) > 1 else None
        if verifier is None:
            raise ValueError('Cross-pod PVC readback requires a second pod')
        readback = remote(verifier, PVC_READBACK, dict(path=path), timeout=600)
        if readback != dict(sha256=written['sha256'], bytes=written['bytes']):
            raise ValueError('Cross-pod PVC archive readback mismatch')
        if written['stdout_log'] is not None:
            (destination/'stdout.log').write_text(written['stdout_log'])
        receipt = dict(archive=path, archive_storage='shared_pvc', bytes=written['bytes'],
            sha256=written['sha256'], all_file_hashes_verified=True, original_spool=launch['spool'],
            pod=pod, cross_pod_readback_pod=verifier, result=written['result'], manifest=binding,
            reused_verified_pvc_archive=written['reused_verified_archive'],
            durable_pvc_export=False, completed_unix_s=time.time())
        write_json(destination/'backup.json', receipt)
        return receipt

    def backup(pod, job, launch, binding, destination):
        nonlocal reserved_bytes
        if args.archive_storage == 'pvc':
            return pvc_backup(pod, job, launch, binding, destination)
        required = binding['artifact_bytes'] + 10_000_000
        with backup_slots:
            with transfer_lock:
                if shutil.disk_usage(args.output).free - reserved_bytes - required < 50_000_000_000:
                    raise OSError('Backup would cross the 50 decimal GB disk reserve')
                reserved_bytes += required
            try:
                part = destination/'artifact.tar.gz.partial'
                archive = destination/'artifact.tar.gz'
                with part.open('wb') as handle:
                    process = subprocess.run(kubectl+['exec', pod, '--', 'tar', '-czf', '-',
                        '-C', launch['spool'], '.'], stdout=handle, stderr=subprocess.PIPE, timeout=600)
                    handle.flush()
                    import os
                    os.fsync(handle.fileno())
                if process.returncode:
                    raise RuntimeError(process.stderr.decode(errors='replace'))
                result = verify_archive(part, binding)
                part.replace(archive)
                log = subprocess.check_output(kubectl+['exec', pod, '--', 'cat', launch['outer']+'/stdout.log'])
                (destination/'stdout.log').write_bytes(log)
                receipt = dict(archive=str(archive.resolve()), bytes=archive.stat().st_size,
                    sha256=digest(archive), all_file_hashes_verified=True, original_spool=launch['spool'],
                    pod=pod, result=result, manifest=binding, durable_pvc_export=False,
                    completed_unix_s=time.time())
                write_json(destination/'backup.json', receipt)
                return receipt
            finally:
                with transfer_lock:
                    reserved_bytes -= required

    args.lease_dir.mkdir(parents=True, exist_ok=True)
    held_leases = []

    def acquire(pod, slot):
        # Concurrent batch processes share the persistent pods. A pod is
        # leased for one launched job at a time, so a new batch can start on
        # whichever pods free up instead of fixing its pod set at start.
        handle = (args.lease_dir/(pod+('.lock' if slot == 0 else f'.slot{slot}.lock'))).open('a+')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return None
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(dict(label=manifest['label'], pid=__import__('os').getpid(),
                                     leased_unix_s=time.time()))+'\n')
        handle.flush()
        return handle

    def worker(pod, slot):
        while True:
            own_queue = assigned[pod] if not assigned[pod].empty() else work
            if own_queue.empty():
                return
            lease = acquire(pod, slot)
            if lease is None:
                time.sleep(args.poll_seconds)
                continue
            try:
                job = own_queue.get_nowait()
            except queue.Empty:
                lease.close()
                return
            destination = args.output/job['id']
            destination.mkdir(exist_ok=True)
            keep_lease = False
            try:
                report(job, stage='launching', pod=pod)
                if (destination/'launch.json').exists():
                    launch = json.loads((destination/'launch.json').read_text())
                    if launch['job'] != job or launch['launcher_sha256'] != launcher_hash:
                        raise ValueError('Retained launch identity does not match this immutable job')
                else:
                    base = ('/mnt/reachy-retarget/shared-spool/' if job.get('spool_storage') == 'shared_pvc'
                            else '/tmp/reachy-retarget-pool/')
                    launch = remote(pod, BOOTSTRAP, dict(base=base+manifest['label'], slot=slot,
                        job=job, launcher=launcher, launcher_sha256=launcher_hash,
                        runtime_bundle=(runtime_bundle if runtime_bundle and
                            job['runtime_source_sha256']==runtime_bundle['source_sha256'] else None)))
                    if launch.get('pod_busy'):
                        # Another operator or unleased process owns this pod.
                        report(job, stage='queued', pod=None, deferred_from_busy_pod=pod)
                        own_queue.put(job)
                        time.sleep(max(args.poll_seconds, 60))
                        continue
                launch['pod'] = pod
                write_json(destination/'launch.json', launch)
                report(job, stage='running', pod=pod, slot=slot, spool=launch['spool'])
                while True:
                    polled = remote(pod, POLL, launch)
                    if polled['complete']:
                        break
                    if not polled['alive']:
                        write_json(destination/'incomplete-process.json', polled)
                        raise RuntimeError('Scratch process ended without an artifact manifest')
                    time.sleep(args.poll_seconds)
                write_json(destination/'artifact-manifest.json', polled['manifest'])
                report(job, stage='backing_up', pod=pod, outcome=polled['manifest']['status'])
                receipt = backup(pod, job, launch, polled['manifest'], destination)
                report(job, stage='backed_up', pod=pod, outcome=polled['manifest']['status'],
                       bytes=receipt['bytes'], sha256=receipt['sha256'])
            except Exception as error:
                detail = dict(error=repr(error), pod=pod, job=job,
                    stdout=getattr(error, 'stdout', None), stderr=getattr(error, 'stderr', None))
                write_json(destination/'orchestration-error.json', detail)
                report(job, stage='orchestration_error', pod=pod, error=repr(error))
                # A lost connection may leave an active process. Reserve this
                # pod rather than running another experiment on top of it.
                keep_lease = True
                return
            finally:
                own_queue.task_done()
                if keep_lease:
                    held_leases.append(lease)
                else:
                    lease.close()

    if not 1 <= args.slots_per_pod <= 4:
        raise ValueError('Between one and four slots per pod')
    with ThreadPoolExecutor(max_workers=len(pods)*args.slots_per_pod) as pool:
        for future in as_completed([pool.submit(worker, pod, slot) for slot in range(args.slots_per_pod)
                                    for pod in pods]):
            future.result()
    remaining = [] if not args.resume_only else [job['id'] for job in jobs
        if not (args.output/job['id']/'launch.json').exists()]
    while not work.empty():
        remaining.append(work.get_nowait()['id'])
    for pod, assigned_queue in assigned.items():
        while not assigned_queue.empty():
            unavailable.append(dict(job=assigned_queue.get_nowait()['id'], original_pod=pod,
                reason='Original pod stopped accepting work after an orchestration error'))
    write_json(args.output/'completion.json', dict(total_jobs=len(jobs), states=states,
        unlaunched_jobs=remaining, unavailable_original_pods=unavailable,
        started_unix_s=started, finished_unix_s=time.time(),
        durable_pvc_export=False))


if __name__ == '__main__':
    main()
