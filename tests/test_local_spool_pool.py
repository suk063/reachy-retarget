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
