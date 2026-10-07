import hashlib
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from cluster.export_local_spool import REMOTE
from cluster.local_spool_pool import PVC_BACKUP, PVC_READBACK, verify_archive


def fixture(tmp_path):
    spool = tmp_path/'spool'
    files = {'datasets/example/episodes/job.hdf5': b'state-only bytes',
             'result.json': json.dumps({'status': 'physical_fail',
                 'archive': str(spool/'datasets/example/episodes/job.hdf5')}).encode()}
    files['datasets/example/episodes/job.json'] = json.dumps(dict(dataset_id='example',
        hdf5_sha256=hashlib.sha256(files['datasets/example/episodes/job.hdf5']).hexdigest())).encode()
    files['datasets/example/metadata.json'] = json.dumps(dict(dataset_id='example')).encode()
    manifest = dict(spool=str(spool), status='physical_fail', source_release_sha256='a'*64,
        artifact_bytes=sum(map(len, files.values())),
        files=[dict(path=p, bytes=len(v), sha256=hashlib.sha256(v).hexdigest()) for p, v in files.items()])
    for name, value in dict(files, **{'portable-artifact-manifest.json': json.dumps(manifest).encode()}).items():
        path = spool/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    return spool, files, manifest


def run(code, value, *argv):
    process = subprocess.run([sys.executable, "-c", code, *argv], input=value,
                             capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


def test_pvc_backup_writes_verified_archive_and_export_reads_it_without_stdin(tmp_path):
    spool, files, manifest = fixture(tmp_path)
    (tmp_path/'outer').mkdir()
    (tmp_path/'outer/stdout.log').write_text('log')
    target = tmp_path/'pvc/operator-backups/batch/job/artifact.tar.gz'
    code = ('import hashlib,json,tarfile\nfrom pathlib import PurePosixPath\n'
            + inspect.getsource(verify_archive) + PVC_BACKUP)
    code = code.replace("'/mnt/reachy-retarget'", repr(str(tmp_path))).replace('50_000_000_000', '0')
    written = run(code, json.dumps(dict(path=str(target), spool=str(spool), outer=str(tmp_path/'outer'),
                                        manifest=manifest, reserve_bytes=0)))
    assert written['result']['status'] == 'physical_fail' and written['stdout_log'] == 'log'
    assert run(PVC_READBACK, json.dumps(dict(path=str(target)))) == dict(sha256=written['sha256'], bytes=written['bytes'])
    assert not [p for p in target.parent.iterdir() if p.name.startswith('.')]
    root = tmp_path/'shared'
    root.mkdir()
    export = REMOTE.replace("Path('/mnt/reachy-retarget')", 'Path('+repr(str(root))+')').replace('50_000_000_000', '0')
    meta = dict(batch='batch', job='job', sha256=written['sha256'], bytes=written['bytes'], manifest=manifest,
                input_mode='pvc_backup', archive_path=str(target))
    receipt = run(export, '', json.dumps(meta))
    assert receipt['input_mode'] == 'pvc_backup'
    assert Path(receipt['path'], 'artifact.tar.gz').read_bytes() == target.read_bytes()
    assert Path(receipt['episode']['path']).read_bytes() == files['datasets/example/episodes/job.hdf5']


def test_pvc_backup_reuses_an_interrupted_archive_only_after_manifest_verification(tmp_path):
    spool, files, manifest = fixture(tmp_path)
    (tmp_path/'outer').mkdir()
    target = tmp_path/'pvc/job/artifact.tar.gz'
    code = ('import hashlib,json,tarfile\nfrom pathlib import PurePosixPath\n'
            + inspect.getsource(verify_archive) + PVC_BACKUP)
    code = code.replace("'/mnt/reachy-retarget'", repr(str(tmp_path))).replace('50_000_000_000', '0')
    value = json.dumps(dict(path=str(target), spool=str(spool), outer=str(tmp_path/'outer'), manifest=manifest, reserve_bytes=0))
    first = run(code, value)
    second = run(code, value)
    assert not first['reused_verified_archive'] and second['reused_verified_archive']
    assert second['sha256'] == first['sha256']
    (spool/'result.json').write_text('{}')
    target.unlink()
    with tarfile.open(target, 'w:gz') as archive:
        archive.add(spool, arcname='.')
    process = subprocess.run([sys.executable, '-c', code], input=value, capture_output=True, text=True)
    assert process.returncode != 0 and 'checksum mismatch' in process.stderr
