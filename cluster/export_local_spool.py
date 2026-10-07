"""Durably publish verified scratch archives and their state-only episodes.

Original bytes, failures and original absolute-path provenance are retained.
Publication adds an explicit original-spool mapping; it does not rewrite a scene,
recorded trajectory or source artifact to conceal its execution location.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import time

from cluster.local_spool_experiment import write_json
from cluster.local_spool_pool import safe_name, verify_archive


def common_entries(manifest, result):
    """Require every writer-contract file to be bound by the original spool."""
    if not result or not result.get('archive'):
        return []
    relative = Path(result['archive']).relative_to(manifest['spool'])
    if (len(relative.parts) != 4 or relative.parts[0] != 'datasets'
            or relative.parts[2] != 'episodes' or '..' in relative.parts
            or relative.suffix not in ('.h5', '.hdf5')):
        raise ValueError('Unexpected state episode path')
    expected = {item['path']: item for item in manifest['files']}
    paths = [relative, relative.with_suffix('.json'), relative.parent.parent/'metadata.json']
    entries = []
    for role, path in zip(('episode', 'episode_sidecar', 'dataset_metadata'), paths):
        if str(path) not in expected:
            raise ValueError('Missing hash-bound common file: '+str(path))
        entries.append(dict(expected[str(path)], role=role, relative_path=str(path)))
    return entries


def file_sha256(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def ensure_common_files(root, target, manifest, result):
    """Repair indexes from the retained archive without replacing any bytes."""
    entries = common_entries(manifest, result)

    def check(path, entry):
        if not path.is_file() or path.stat().st_size != entry['bytes'] or file_sha256(path) != entry['sha256']:
            raise ValueError('Common file checksum conflict: '+str(path))

    missing = {entry['relative_path']: entry for entry in entries
               if not (target/entry['relative_path']).exists()}
    if missing:
        if shutil.disk_usage(root).free - sum(entry['bytes'] for entry in missing.values()) < 50_000_000_000:
            raise OSError('50 decimal GB shared-storage reserve')
        with tarfile.open(target/'artifact.tar.gz', 'r:gz') as source:
            for member in source:
                name = str(PurePosixPath(member.name))
                if name not in missing:
                    continue
                if not member.isfile():
                    raise ValueError('Common file is not a regular archive member')
                output = target/name
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name('.'+output.name+'.extract-'+str(time.time_ns()))
                try:
                    with source.extractfile(member) as src, temporary.open('xb') as dst:
                        shutil.copyfileobj(src, dst)
                        dst.flush()
                        os.fsync(dst.fileno())
                    check(temporary, missing[name])
                    try:
                        os.link(temporary, output)
                    except FileExistsError:
                        check(output, missing[name])
                    fd = os.open(output.parent, os.O_RDONLY)
                    os.fsync(fd)
                    os.close(fd)
                finally:
                    temporary.unlink(missing_ok=True)
                del missing[name]
        if missing:
            raise ValueError('Missing common files in retained archive: '+str(sorted(missing)))
    for entry in entries:
        check(target/entry['relative_path'], entry)
    if entries:
        sidecar = json.loads((target/entries[1]['relative_path']).read_text())
        dataset = json.loads((target/entries[2]['relative_path']).read_text())
        dataset_id = Path(entries[0]['relative_path']).parts[1]
        if sidecar.get('hdf5_sha256') != entries[0]['sha256']:
            raise ValueError('Episode sidecar does not bind the HDF5 checksum')
        if sidecar.get('dataset_id') != dataset_id or dataset.get('dataset_id') != dataset_id:
            raise ValueError('Common metadata dataset identity mismatch')
    published = []
    for entry in entries:
        canonical = root/entry['relative_path']
        canonical.parent.mkdir(parents=True, exist_ok=True)
        try:
            canonical.symlink_to(target/entry['relative_path'])
        except FileExistsError:
            pass
        # Shared dataset metadata, including an existing regular file, must
        # match exactly. Concurrent identical publications may reuse its link.
        check(canonical, entry)
        fd = os.open(canonical.parent, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
        published.append(dict(entry, path=str(target/entry['relative_path']), canonical_path=str(canonical)))
    # Persist new directory entries as well as the files and their direct
    # parents. An interrupted repair can then safely resume from its tar.
    directories = {root, target}
    for entry in published:
        for key in ('path', 'canonical_path'):
            parent = Path(entry[key]).parent
            while parent != root:
                directories.add(parent)
                parent = parent.parent
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        fd = os.open(directory, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    receipt = dict(schema='node-local-common-files-v1',
                   publication_sha256=file_sha256(target/'publication.json'),
                   artifact_manifest_sha256=file_sha256(target/'artifact-manifest.json'),
                   result_sha256=file_sha256(target/'result.json'),
                   files=published, complete=True)
    path = target/'common-files-publication.json'
    if path.exists():
        if json.loads(path.read_text()) != receipt:
            raise ValueError('Immutable common-files receipt conflict')
    else:
        temporary = path.with_name('.'+path.name+'.staging-'+str(time.time_ns()))
        try:
            with temporary.open('x') as stream:
                json.dump(receipt, stream, indent=2)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        fd = os.open(target, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    return dict(receipt, receipt_path=str(path), receipt_sha256=file_sha256(path))


REMOTE = r'''
import fcntl,hashlib,json,os,shutil,subprocess,sys,tarfile,time
from pathlib import Path,PurePosixPath
''' + '\n'.join(inspect.getsource(fn) for fn in
                 (verify_archive, common_entries, file_sha256, ensure_common_files)) + r'''
v=json.loads(sys.argv[1]);root=Path('/mnt/reachy-retarget');target=root/'node-local-exports'/v['batch']/v['job']
target.parent.mkdir(parents=True,exist_ok=True)
lock=(target.parent/('.'+v['job']+'.publication.lock')).open('a+')
fcntl.flock(lock,fcntl.LOCK_EX)
def finish(receipt):
 common=ensure_common_files(root,target,v['manifest'],json.loads((target/'result.json').read_text()))
 print(json.dumps(dict(receipt,common_files=common)))
if target.exists():
 receipt=json.loads((target/'publication.json').read_text())
 if receipt['operator_archive_sha256']!=v['sha256']:raise ValueError('Immutable exported archive conflict')
 if json.loads((target/'artifact-manifest.json').read_text())!=v['manifest']:raise ValueError('Retained artifact manifest conflict')
 if file_sha256(target/'artifact.tar.gz')!=receipt.get('published_archive_sha256',receipt['operator_archive_sha256']):raise ValueError('Retained archive checksum mismatch')
 verified_result=verify_archive(target/'artifact.tar.gz',v['manifest'])
 if verified_result!=json.loads((target/'result.json').read_text()):raise ValueError('Retained result differs from bound archive')
 # Consume the stream so a retry is well-defined for the sending process.
 if v.get('input_mode') not in ('pod_local','shared_pvc','repair_existing','pvc_backup'):
  sha=hashlib.sha256()
  for block in iter(lambda:sys.stdin.buffer.read(8*1024*1024),b''):sha.update(block)
  if sha.hexdigest()!=v['sha256']:raise ValueError('Retry stream checksum mismatch')
 finish(receipt);raise SystemExit(0)
if v.get('input_mode')=='repair_existing':raise FileNotFoundError('Expected retained publication is absent')
if shutil.disk_usage(root).free-v['bytes']-v['manifest']['artifact_bytes']<50_000_000_000:raise OSError('50 decimal GB shared-storage reserve')
stage=target.parent/('.staging-'+v['job']+'-'+str(time.time_ns()));stage.mkdir(parents=True)
def write(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
try:
 archive=stage/'artifact.tar.gz';sha=hashlib.sha256();size=0
 if v.get('input_mode') in ('pod_local','shared_pvc'):
  source=Path(v['manifest']['spool']).resolve()
  if v['input_mode']=='shared_pvc':
   expected=root.resolve()/'shared-spool'/v['batch']/v['job']/'artifact'
   if source!=expected or not source.is_dir():raise ValueError('Original shared spool binding is unavailable')
  elif source.is_relative_to(root) or not source.is_dir():raise ValueError('Original node-local spool is unavailable')
  subprocess.run(['tar','-czf',str(archive),'-C',str(source),'.'],check=True,capture_output=True,env=dict(os.environ,COPYFILE_DISABLE='1'))
  with archive.open('rb') as f:
   os.fsync(f.fileno())
   for block in iter(lambda:f.read(8*1024*1024),b''):sha.update(block);size+=len(block)
 else:
  stream=open(v['archive_path'],'rb') if v.get('input_mode')=='pvc_backup' else sys.stdin.buffer
  with archive.open('xb') as f:
   for block in iter(lambda:stream.read(8*1024*1024),b''):
    f.write(block);sha.update(block);size+=len(block)
   f.flush();os.fsync(f.fileno())
  if size!=v['bytes'] or sha.hexdigest()!=v['sha256']:raise ValueError('Archive transport checksum mismatch')
 published_sha256=sha.hexdigest()
 result=verify_archive(archive,v['manifest'])
 entries=common_entries(v['manifest'],result)
 write(stage/'artifact-manifest.json',v['manifest'])
 write(stage/'result.json',result)
 episode=None
 if entries:
  entry=entries[0]
  episode=dict(path=str(target/entry['relative_path']),relative_path=entry['relative_path'],sha256=entry['sha256'],bytes=entry['bytes'])
 receipt=dict(schema='node-local-publication-v1',job=v['job'],batch=v['batch'],
  path=str(target),operator_archive_sha256=v['sha256'],archive_bytes=size,
  published_archive_sha256=published_sha256,input_mode=v.get('input_mode','operator_upload'),
  all_regular_file_hashes_verified=True,all_file_writes_fsynced=True,
  source_release_sha256=v['manifest']['source_release_sha256'],
  runtime_publication_scope=v['manifest'].get('runtime_publication_scope','shared_pvc'),
  status=(result or {}).get('status',v['manifest']['status']),episode=episode,
  original_spool=v['manifest']['spool'],archive_layout='Contents of original spool, with original bytes and external read-only references preserved',
  replay_path_policy='Restore original spool layout or apply an explicit verified path mapping; no archived paths were rewritten',
  published_unix_s=time.time(),cross_pod_readback_verified=False)
 write(stage/'publication.json',receipt)
 fd=os.open(stage,os.O_RDONLY);os.fsync(fd);os.close(fd)
 stage.rename(target)
 fd=os.open(target.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
 finish(receipt)
except BaseException as error:
 if stage.exists():
  try:write(stage/'publication-failure.json',dict(error=repr(error)))
  except OSError:pass
 raise
'''

READBACK = r'''
import hashlib,json,sys
from pathlib import Path
v=json.loads(sys.argv[1]);root=Path(v['path']);values={}
common=v.get('common_files')
if not common or common.get('schema')!='node-local-common-files-v1' or not common.get('complete'):raise ValueError('Complete common-files receipt required')
checks=[(root/'artifact.tar.gz',v.get('published_archive_sha256',v['operator_archive_sha256'])),
 (root/'publication.json',common['publication_sha256']),
 (root/'artifact-manifest.json',common['artifact_manifest_sha256']),
 (root/'result.json',common['result_sha256']),
 (Path(common['receipt_path']),common['receipt_sha256'])]
for entry in common['files']:
 checks.extend([(Path(entry['path']),entry['sha256']),(Path(entry['canonical_path']),entry['sha256'])])
for p,expected in checks:
 sha=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):sha.update(block)
 if sha.hexdigest()!=expected:raise ValueError('Cross-pod readback mismatch: '+str(p))
 values[str(p)]=sha.hexdigest()
print(json.dumps(dict(passed=True,files=values,common_files_receipt_sha256=common['receipt_sha256'],canonical_common_files_verified=len(common['files']))))
'''


def publication_summary(batch, manifest, results, errors, backup_jobs):
    expected = [safe_name(job['id']) for job in manifest['jobs']]
    if len(set(expected)) != len(expected):
        raise ValueError('Duplicate job IDs in batch manifest')
    completed = [result['job'] for result in results]
    missing = sorted(set(expected)-set(completed))
    unexpected = sorted((set(completed) | set(backup_jobs))-set(expected))
    return dict(batch=batch, exported=len(results), expected_jobs=sorted(expected),
                missing_jobs=missing, unexpected_jobs=unexpected,
                missing_backups=sorted(set(expected)-set(backup_jobs)),
                results=results, failures=errors,
                complete=not errors and not missing and not unexpected
                         and set(backup_jobs)==set(expected)
                         and len(completed)==len(set(completed)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch_output', type=Path)
    parser.add_argument('--pod', required=True)
    parser.add_argument('--readback-pod', required=True)
    parser.add_argument('--parallel', type=int, default=4)
    parser.add_argument('--upload', action='store_true', help='Use the verified operator backup if the original pod-local spool is unavailable')
    args = parser.parse_args()
    if args.pod == args.readback_pod:
        raise ValueError('Cross-pod readback must use a distinct pod')
    batch = safe_name(args.batch_output.name)
    backups = sorted(args.batch_output.glob('*/backup.json'))
    manifest = json.loads((args.batch_output/'manifest.json').read_text())
    expected = {safe_name(job['id']) for job in manifest['jobs']}
    if len(expected) != len(manifest['jobs']):
        raise ValueError('Duplicate job IDs in batch manifest')

    def publish(path):
        job = safe_name(path.parent.name)
        if job not in expected:
            raise ValueError('Backup job is absent from expected batch manifest')
        backup = json.loads(path.read_text())
        proof = path.parent/'publication.json'
        retained = json.loads(proof.read_text()) if proof.exists() else None
        if retained and retained['operator_archive_sha256'] != backup['sha256']:
            raise ValueError('Local immutable publication conflicts with backup')
        shared_spool=(backup['manifest']['spool']==
            str(PurePosixPath('/mnt/reachy-retarget/shared-spool')/batch/job/'artifact'))
        pvc_backup = backup.get('archive_storage') == 'shared_pvc'
        source_pod = args.pod if retained or args.upload or shared_spool or pvc_backup else backup['pod']
        readback_pod = args.readback_pod if args.readback_pod != source_pod else args.pod
        if source_pod == readback_pod:
            raise ValueError('A distinct readback pod is required')
        k = ['kubectl', '-n', 'erl-ucsd', 'exec', '-i', source_pod, '--', 'python3', '-c', REMOTE]
        mode = ('repair_existing' if retained else 'pvc_backup' if pvc_backup
                else 'operator_upload' if args.upload
                else 'shared_pvc' if shared_spool else 'pod_local')
        meta = dict(batch=batch, job=job, sha256=backup['sha256'], bytes=backup['bytes'],
                    manifest=backup['manifest'], input_mode=mode)
        if pvc_backup:
            meta['archive_path'] = backup['archive']
        if mode == 'operator_upload':
            with Path(backup['archive']).open('rb') as stream:
                process = subprocess.run(k+[json.dumps(meta)], stdin=stream, text=True,
                                         capture_output=True, check=True, timeout=1200)
        else:
            process = subprocess.run(k+[json.dumps(meta)], stdin=subprocess.DEVNULL, text=True,
                                     capture_output=True, check=True, timeout=1200)
        receipt = json.loads(process.stdout)
        common = receipt.pop('common_files')
        if not proof.exists():
            write_json(proof, receipt)
        # Keep historical publication.json unchanged, even if it contains an
        # older partial readback claim. Add an explicitly bound upgrade receipt.
        common_proof = path.parent/'common-files-publication.json'
        if common_proof.exists() and json.loads(common_proof.read_text()) != common:
            raise ValueError('Local immutable common-files receipt conflict')
        if not common_proof.exists():
            write_json(common_proof, common)
        receipt['common_files'] = common
        process = subprocess.run(['kubectl', '-n', 'erl-ucsd', 'exec', readback_pod,
            '--', 'python3', '-c', READBACK, json.dumps(receipt)],
            text=True, capture_output=True, check=True, timeout=300)
        readback = json.loads(process.stdout)
        receipt.update(cross_pod_readback=readback,
                       cross_pod_readback_verified=readback['passed'], readback_pod=readback_pod)
        write_json(path.parent/'common-files-readback.json', dict(readback, readback_pod=readback_pod))
        print(json.dumps(dict(job=job, published=True, status=receipt['status'],
                              episode=receipt['episode'], common_files_complete=True,
                              cross_pod_readback=True)), flush=True)
        return receipt

    results, errors = [], []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(publish, path): path for path in backups}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                item = dict(job=futures[future].parent.name, error=repr(error),
                            stderr=getattr(error, 'stderr', None))
                errors.append(item)
                print(json.dumps(item), flush=True)
    summary = publication_summary(batch, manifest, results, errors, [p.parent.name for p in backups])
    write_json(args.batch_output/'publication-summary.json', summary)
    if not summary['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
