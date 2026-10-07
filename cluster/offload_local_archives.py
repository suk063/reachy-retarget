"""Free operator disk by keeping verified scratch archives only on the shared PVC.

A local ``artifact.tar.gz`` is removed only after a pod other than the one that
wrote the PVC copy re-reads that copy and reproduces the exact operator backup
SHA-256. Archives whose published PVC bytes differ from the operator backup
(for example a pod-local re-tar) are first uploaded byte-for-byte under
``operator-backups``. Each removal leaves an ``archive-offload.json`` receipt
next to the retained backup, manifest and logs; no failed attempt is dropped.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path, PurePosixPath
import subprocess
import time

from cluster.local_spool_experiment import digest, write_json
from cluster.local_spool_pool import safe_name

ROOT = PurePosixPath('/mnt/reachy-retarget')

HASH = r'''
import hashlib,json,os,sys
v=json.load(sys.stdin);out={}
for p in v['paths']:
 try:
  sha=hashlib.sha256();n=0
  with open(p,'rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):sha.update(b);n+=len(b)
  out[p]=dict(sha256=sha.hexdigest(),bytes=n)
 except FileNotFoundError:out[p]=None
print(json.dumps(out))
'''

UPLOAD = r'''
import hashlib,json,os,shutil,sys,time
from pathlib import Path
v=json.loads(sys.argv[1]);target=Path(v['path'])
if target.exists():
 sha=hashlib.sha256()
 with target.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):sha.update(b)
 if sha.hexdigest()!=v['sha256']:raise ValueError('Immutable operator backup conflict')
 sys.stdin.buffer.read();print(json.dumps(dict(path=str(target),reused=True)));raise SystemExit(0)
if shutil.disk_usage('/mnt/reachy-retarget').free-v['bytes']<50_000_000_000:raise OSError('50 decimal GB shared-storage reserve')
target.parent.mkdir(parents=True,exist_ok=True)
part=target.with_name('.'+target.name+'.partial-'+str(time.time_ns()));sha=hashlib.sha256();n=0
with part.open('xb') as f:
 for b in iter(lambda:sys.stdin.buffer.read(8*1024*1024),b''):f.write(b);sha.update(b);n+=len(b)
 f.flush();os.fsync(f.fileno())
if sha.hexdigest()!=v['sha256'] or n!=v['bytes']:
 part.unlink();raise ValueError('Upload checksum mismatch')
os.link(part,target);part.unlink()
fd=os.open(target.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(path=str(target),reused=False)))
'''


def ready_pods(namespace):
    items = json.loads(subprocess.check_output(['kubectl', '-n', namespace, 'get', 'pods', '-l',
        'app.kubernetes.io/name=reachy-retarget-worker', '-o', 'json']))['items']
    return [i['metadata']['name'] for i in items if i['status']['phase'] == 'Running'
            and any(c.get('ready') for c in i['status'].get('containerStatuses', []))]


def candidates(roots):
    for root in roots:
        for backup_path in sorted(root.glob('*/*/backup.json')):
            folder = backup_path.parent
            archive = folder/'artifact.tar.gz'
            if not archive.exists() or (folder/'archive-offload.json').exists():
                continue
            batch, job = safe_name(folder.parent.name), safe_name(folder.name)
            backup = json.loads(backup_path.read_text())
            publication = folder/'publication.json'
            readback = folder/'common-files-readback.json'
            published = json.loads(publication.read_text()) if publication.exists() else None
            if published and published.get('published_archive_sha256',
                    published['operator_archive_sha256']) == backup['sha256'] and readback.exists():
                remote = str(ROOT/'node-local-exports'/batch/job/'artifact.tar.gz')
                mode = 'published_identical_bytes'
            else:
                remote = str(ROOT/'operator-backups'/root.name/batch/job/'artifact.tar.gz')
                mode = 'operator_backup_upload'
            yield dict(folder=folder, archive=archive, batch=batch, job=job, sha256=backup['sha256'],
                       bytes=backup['bytes'], remote=remote, mode=mode,
                       writer_pod=(published or {}).get('readback_pod'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', type=Path, nargs='+')
    parser.add_argument('--namespace', default='erl-ucsd')
    parser.add_argument('--parallel', type=int, default=16)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    pods = ready_pods(args.namespace)
    if len(pods) < 2:
        raise ValueError('Cross-pod verification requires two ready pods')
    items = list(candidates(args.roots))
    summary = dict(total=len(items), total_bytes=sum(i['bytes'] for i in items),
                   modes={m: sum(i['mode'] == m for i in items) for m in
                          ('published_identical_bytes', 'operator_backup_upload')})
    print(json.dumps(summary), flush=True)
    if args.dry_run:
        return
    kubectl = ['kubectl', '-n', args.namespace, 'exec', '-i']

    def offload(index, item):
        if digest(item['archive']) != item['sha256']:
            raise ValueError('Local archive differs from its backup receipt: '+str(item['archive']))
        upload_pod = pods[index % len(pods)]
        if item['mode'] == 'operator_backup_upload':
            with item['archive'].open('rb') as stream:
                subprocess.run(kubectl+[upload_pod, '--', 'python3', '-c', UPLOAD,
                    json.dumps(dict(path=item['remote'], sha256=item['sha256'], bytes=item['bytes']))],
                    stdin=stream, check=True, capture_output=True, timeout=1800)
        verifier = next(p for p in pods[index % len(pods)+1:]+pods
                        if p not in (upload_pod, item['writer_pod']))
        out = subprocess.run(kubectl+[verifier, '--', 'python3', '-c', HASH],
            input=json.dumps(dict(paths=[item['remote']])), text=True,
            capture_output=True, check=True, timeout=900)
        found = json.loads(out.stdout)[item['remote']]
        if not found or found['sha256'] != item['sha256'] or found['bytes'] != item['bytes']:
            raise ValueError('PVC copy does not reproduce the operator archive: '+item['remote'])
        receipt = dict(schema='operator-archive-offload-v1', batch=item['batch'], job=item['job'],
            local_archive_removed=str(item['archive']), pvc_archive=item['remote'], mode=item['mode'],
            sha256=item['sha256'], bytes=item['bytes'], local_rehash_matched=True,
            cross_pod_verifier=verifier, upload_pod=upload_pod if item['mode'] == 'operator_backup_upload' else None,
            verified_unix_s=time.time())
        write_json(item['folder']/'archive-offload.json', receipt)
        item['archive'].unlink()
        return receipt

    done, failures = 0, []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(offload, n, item): item for n, item in enumerate(items)}
        for future in as_completed(futures):
            try:
                future.result()
                done += 1
                if done % 100 == 0:
                    print(json.dumps(dict(offloaded=done)), flush=True)
            except Exception as error:
                failures.append(dict(job=futures[future]['job'], batch=futures[future]['batch'],
                                     error=repr(error), stderr=getattr(error, 'stderr', None)))
    print(json.dumps(dict(offloaded=done, failures=failures), indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
