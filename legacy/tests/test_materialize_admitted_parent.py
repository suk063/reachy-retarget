"""Archived parents must be explicitly expanded without changing their bytes."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile
import pytest
from cluster.materialize_admitted_parent import materialize


def test_materialization_preserves_every_bound_file_and_rejects_replacement(tmp_path,monkeypatch):
    monkeypatch.setattr('cluster.materialize_admitted_parent.shutil.disk_usage',
                        lambda path:SimpleNamespace(free=100_000_000_000))
    files={'scene.xml':b'<mujoco/>','plan.h5':b'original numeric plan','result.json':b'{}'}
    archive=tmp_path/'artifact.tar.gz'
    with tarfile.open(archive,'w:gz') as handle:
        for name,raw in files.items():
            info=tarfile.TarInfo('./work/plan/'+name);info.size=len(raw)
            handle.addfile(info,io.BytesIO(raw))
    hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
    args=dict(archive=archive,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
              member_directory='work/plan',files=hashes,destination=tmp_path/'materialized')
    report=materialize(**args)
    assert materialize(**args)==report
    for name,raw in files.items():
        assert (args['destination']/name).read_bytes()==raw
    assert json.loads((args['destination']/'binding.json').read_text())['files']==hashes
    (args['destination']/'plan.h5').write_bytes(b'changed')
    with pytest.raises(ValueError,match='conflict'):
        materialize(**args)
    with pytest.raises(ValueError,match='absent'):
        materialize(**dict(args,member_directory='not-present',destination=tmp_path/'other'))
