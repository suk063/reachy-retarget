import hashlib
import json
import tarfile

from cluster.local_runtime_bundle import build
from cluster.local_spool_experiment import verify_published_manifest


def test_local_runtime_is_byte_bound_and_keeps_controller_identity(tmp_path, monkeypatch):
    repo = tmp_path/'repo'
    (repo/'reachy_retarget').mkdir(parents=True)
    (repo/'reachy_retarget/example.py').write_text('VALUE = 1\n')
    (repo/'pyproject.toml').write_text('[project]\nname="example"\n')
    checksum = hashlib.sha256(b'old source').hexdigest()
    source = {'reachy_retarget/example.py': checksum}
    control = {'control/reachy.py': hashlib.sha256(b'preserved controller').hexdigest()}
    identity = lambda values: hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    original = dict(source_files=source, source_sha256=identity(source),
        control_files=control, control_sha256=identity(control))
    monkeypatch.setattr('cluster.local_runtime_bundle.subprocess.check_output', lambda *a, **kw: 'main\n')
    path = tmp_path/'bundle.tar.gz'
    result = build(repo, original, path)
    proof = result['proof']
    verify_published_manifest(proof, result['source_sha256'])
    assert proof['publication_scope'] == 'pod_local_unpublished'
    assert proof['control_sha256'] == original['control_sha256']
    assert result['source_sha256'] != original['source_sha256']
    # An edit after snapshot construction cannot alter the bound bundle bytes.
    (repo/'reachy_retarget/example.py').write_text('VALUE = 2\n')
    with tarfile.open(path) as archive:
        name = 'releases/'+result['source_sha256']+'/reachy_retarget/example.py'
        content = archive.extractfile(name).read()
    assert content == b'VALUE = 1\n'
    assert hashlib.sha256(content).hexdigest() == proof['source_files']['reachy_retarget/example.py']
