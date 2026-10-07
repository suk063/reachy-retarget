"""Persistent queue consumer. Only explicit queued tasks can transfer data."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import time
import uuid

ROOT = Path(os.environ.get('REACHY_RETARGET_ROOT', '/mnt/reachy-retarget'))
WORKER = os.environ.get('POD_NAME', socket.gethostname())
SERVICE_REVISION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('w') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_json(path):
    """Bounded retry; persistent corruption must remain visible and isolated."""
    for trial in range(3):
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError('A JSON object is required: '+str(path))
            return value
        except (OSError, ValueError):
            if trial == 2:
                raise
            time.sleep(.05)


def defer_invalid(folder, error, deferred):
    """Never replace invalid records or restart the entire consumer for them."""
    deferred[folder.name] = time.monotonic()+60.
    print(json.dumps(dict(event='queue_record_unreadable',worker=WORKER,
                          task=folder.name,error=str(error),retry_after_s=60.)),flush=True)


def heartbeat(status, **extra):
    try:
        atomic(ROOT / 'workers' / (WORKER + '.json'), dict(worker=WORKER,
               node=os.environ.get('NODE_NAME'), time=time.time(), status=status,
               service_revision=SERVICE_REVISION, **extra))
        return True
    except OSError as error:
        print(json.dumps(dict(event='heartbeat_storage_error',worker=WORKER,error=str(error))),flush=True)
        return False


def pending_jobs(queue, completed):
    # Skip immutable terminal IDs before traversing each directory on CephFS.
    return [folder/'job.json' for folder in queue.iterdir()
            if not folder.name.startswith('.') and folder.name not in completed]


def claim_runtime(root, job):
    """Read activation at claim time, with an optional immutable source pin."""
    runtime=read_json(root/'runtime.json')
    pinned=job.get('runtime_source_sha256')
    if pinned:
        if not re.fullmatch('[0-9a-f]{64}',pinned):raise ValueError('Invalid source release pin')
        source=root/'releases'/pinned
        proof=read_json(root/'provenance'/f'{pinned}.json')
        if proof['source_sha256']!=pinned or not source.is_dir():raise ValueError('Missing pinned release')
        if proof['control_sha256']!=runtime['control_sha256']:raise ValueError('Pinned controller reference differs')
        runtime.update(source=str(source),source_sha256=pinned)
    return runtime


def serve():
    queue = ROOT / 'queue' / 'tasks'
    queue.mkdir(parents=True, exist_ok=True)
    heartbeat('waiting_for_runtime')
    completed=set(); deferred={}
    while True:
        if not heartbeat('checking_storage'):
            # A successful close alone does not guarantee persisted Ceph data.
            # Do not claim more work until a flushed publication succeeds.
            time.sleep(5)
            continue
        pointer = ROOT / 'runtime.json'
        if not pointer.exists():
            heartbeat('waiting_for_runtime')
            time.sleep(5)
            continue
        try:
            runtime = read_json(pointer)
        except (OSError, ValueError) as error:
            print(json.dumps(dict(event='runtime_unreadable',error=str(error))),flush=True)
            time.sleep(5)
            continue
        try:
            jobs = pending_jobs(queue,completed)
        except OSError as error:
            print(json.dumps(dict(event='queue_scan_storage_error',error=str(error))),flush=True)
            time.sleep(5)
            continue
        random.shuffle(jobs)
        handled = False
        last_scan_heartbeat=0.
        for job in jobs:
            if time.monotonic()-last_scan_heartbeat>=10.:
                heartbeat('scanning',source_revision=runtime.get('source_sha256'),deferred_tasks=len(deferred))
                last_scan_heartbeat=time.monotonic()
            folder = job.parent
            if deferred.get(folder.name,0.) > time.monotonic():
                continue
            try:
                claim = (folder / 'claim.lock').open('a')
            except FileNotFoundError:
                # A stale scan must not terminate a persistent worker.
                continue
            except OSError as error:
                defer_invalid(folder,error,deferred)
                continue
            with claim as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                except OSError as error:
                    defer_invalid(folder,error,deferred)
                    continue
                result_path = folder / 'result.json'
                try:
                    result = read_json(result_path) if result_path.exists() else None
                except (OSError, ValueError, KeyError) as error:
                    defer_invalid(folder,error,deferred)
                    continue
                if result is not None:
                    if result.get('status') != 'waiting_for_agent':
                        completed.add(folder.name)
                        continue
                    response = folder / 'agent' / 'response.json'
                    if not response.exists():
                        continue
                    response_hash = hashlib.sha256(response.read_bytes()).hexdigest()
                    if response_hash == result.get('response_sha256'):
                        continue
                try:
                    job_spec = read_json(job)
                    selected_runtime = claim_runtime(ROOT,job_spec)
                except (OSError, ValueError, KeyError) as error:
                    defer_invalid(folder,error,deferred)
                    continue
                attempt = folder / 'attempts' / (str(time.time_ns()) + '-' + WORKER)
                runtime=selected_runtime
                try:
                    attempt.mkdir(parents=True)
                    atomic(attempt / 'runtime.json', runtime)
                    atomic(attempt / 'claim.json', dict(worker=WORKER, time=time.time()))
                except OSError as error:
                    defer_invalid(folder,error,deferred)
                    continue
                env = dict(os.environ, PYTHONPATH=runtime['source'],
                           REACHY_RETARGET_CONTROL=runtime['control'],
                           REACHY_RETARGET_BACKEND='reference',
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
                command = [runtime['python'], '-m', 'reachy_retarget.cluster_pipeline',
                           'execute', '--root', str(ROOT), '--task', str(folder),
                           '--attempt', str(attempt)]
                try:
                    with (attempt / 'stdout.log').open('w') as log:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                        while child.poll() is None:
                            heartbeat('working', task=folder.name, attempt=str(attempt), pid=child.pid)
                            time.sleep(5)
                        if child.returncode != 0:
                            value = dict(status='failed', returncode=child.returncode,
                                         log=str(attempt / 'stdout.log'), worker=WORKER)
                            atomic(attempt / 'result.json', value)
                            atomic(result_path, value)
                except OSError as error:
                    # Storage may fail after the claim but before opening the
                    # child log. Preserve that claim and defer only this task.
                    defer_invalid(folder,error,deferred)
                    continue
                handled = True
                try:
                    if result_path.exists() and read_json(result_path).get('status')!='waiting_for_agent':
                        completed.add(folder.name)
                except (OSError, ValueError) as error:
                    defer_invalid(folder,error,deferred)
        heartbeat('idle', source_revision=runtime.get('source_sha256'))
        time.sleep(2 if handled else 8)


if __name__ == '__main__':
    serve()
