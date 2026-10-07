"""Offline checks for lossless node-local archive backup validation."""
import hashlib
import io
import json
import tarfile

import pytest

from cluster.local_spool_pool import safe_name, verify_archive


def archive(tmp_path, entries, manifest):
    path = tmp_path/'artifact.tar.gz'
    with tarfile.open(path, 'w:gz') as tar:
        for name, content in dict(entries, **{'portable-artifact-manifest.json':
                json.dumps(manifest).encode()}).items():
            member = tarfile.TarInfo('./'+name)
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
        link = tarfile.TarInfo('./raw-source')
        link.type = tarfile.SYMTYPE
        link.linkname = '/mnt/reachy-retarget/preserved/source'
        tar.addfile(link)
    return path


def test_backup_verifies_payload_without_extracting_external_links(tmp_path):
    payload = b'{"status":"physical_fail"}'
    manifest = {'files': [{'path': 'result.json', 'bytes': len(payload),
        'sha256': hashlib.sha256(payload).hexdigest()}]}
    path = archive(tmp_path, {'result.json': payload}, manifest)
    assert verify_archive(path, manifest) == {'status': 'physical_fail'}
    assert not (tmp_path/'raw-source').exists()
    assert not (tmp_path/'result.json').exists()


def test_backup_rejects_missing_and_corrupt_files(tmp_path):
    manifest = {'files': [{'path': 'result.json', 'bytes': 2,
        'sha256': hashlib.sha256(b'{}').hexdigest()}]}
    with pytest.raises(ValueError, match='checksum'):
        verify_archive(archive(tmp_path, {'result.json': b'[]'}, manifest), manifest)
    with pytest.raises(ValueError, match='Incomplete'):
        verify_archive(archive(tmp_path, {}, manifest), manifest)


def test_backup_rejects_path_traversal_and_duplicate_jobs_names():
    for value in ('../../a', '/root', 'abc/def', ''):
        with pytest.raises(ValueError, match='Unsafe'):
            safe_name(value)
    assert safe_name('can-regression-01') == 'can-regression-01'


def test_bootstrap_reports_a_busy_pod_instead_of_failing(tmp_path):
    import fcntl
    import subprocess
    import sys
    from pathlib import Path
    from cluster.local_spool_pool import BOOTSTRAP
    lock_path = tmp_path/'execution.lock'
    code = (BOOTSTRAP.replace("'/tmp/reachy-retarget-pod-execution.lock'", repr(str(lock_path)))
            .replace('50_000_000_000', '0'))
    value = json.dumps(dict(base=str(tmp_path/'spool'), job=dict(id='job', runtime_source_sha256='a'*64),
                            launcher='', launcher_sha256=hashlib.sha256(b'').hexdigest()))
    with lock_path.open('a+') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process = subprocess.run([sys.executable, '-c', code], input=value, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == {'pod_busy': True}
    assert not (Path(tmp_path)/'spool/job/launch.json').exists()


def test_a_second_execution_slot_is_not_blocked_by_the_first(tmp_path):
    import fcntl
    import subprocess
    import sys
    from cluster.local_spool_pool import BOOTSTRAP
    lock_path = tmp_path/'execution.lock'
    code = (BOOTSTRAP.replace("'/tmp/reachy-retarget-pod-execution.lock'", repr(str(lock_path)))
            .replace('50_000_000_000', '0'))
    value = dict(base=str(tmp_path/'spool'), job=dict(id='job', runtime_source_sha256='a'*64),
                 launcher='', launcher_sha256=hashlib.sha256(b'').hexdigest())
    with lock_path.open('a+') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        busy = subprocess.run([sys.executable, '-c', code], input=json.dumps(dict(value, slot=0)),
                              capture_output=True, text=True)
        free = subprocess.run([sys.executable, '-c', code], input=json.dumps(dict(value, slot=1)),
                              capture_output=True, text=True)
    assert json.loads(busy.stdout) == {'pod_busy': True}
    # Slot 1 takes its own lock and proceeds to launch (the runtime is absent here).
    assert 'pod_busy' not in free.stdout and (tmp_path/'execution.lock.1').exists()
