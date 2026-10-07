import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from cluster.export_local_spool import READBACK, REMOTE, publication_summary


def test_durable_export_preserves_bytes_and_reuses_existing_episode_index(tmp_path):
    spool=str(tmp_path/'original-immutable-spool')
    files={'datasets/example/episodes/job.hdf5':b'preserved state-only bytes',
           'result.json':json.dumps({'status':'physical_fail','archive':spool+'/datasets/example/episodes/job.hdf5'}).encode()}
    files['datasets/example/episodes/job.json']=json.dumps(dict(dataset_id='example',hdf5_sha256=hashlib.sha256(files['datasets/example/episodes/job.hdf5']).hexdigest())).encode()
    files['datasets/example/metadata.json']=json.dumps(dict(dataset_id='example')).encode()
    manifest=dict(spool=spool,status='physical_fail',source_release_sha256='a'*64,
        artifact_bytes=sum(map(len,files.values())),files=[dict(path=p,bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for p,v in files.items()])
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name,content in dict(files,**{'portable-artifact-manifest.json':json.dumps(manifest).encode()}).items():
            member=tarfile.TarInfo('./'+name);member.size=len(content);archive.addfile(member,io.BytesIO(content))
    content=buffer.getvalue();root=tmp_path/'shared';root.mkdir()
    meta=dict(batch='test-batch',job='test-job',sha256=hashlib.sha256(content).hexdigest(),bytes=len(content),manifest=manifest)
    code=REMOTE.replace("Path('/mnt/reachy-retarget')",'Path('+repr(str(root))+')')
    # The test does not depend on the host's free disk; production retains its reserve.
    code=code.replace('50_000_000_000','0')
    for _ in range(2):
        result=subprocess.run([sys.executable,'-c',code,json.dumps(meta)],input=content,capture_output=True,check=True)
        receipt=json.loads(result.stdout)
        assert receipt['status']=='physical_fail'
        assert receipt['all_regular_file_hashes_verified']
    index=root/'datasets/example/episodes/job.hdf5'
    assert index.is_symlink()
    assert index.read_bytes()==files['datasets/example/episodes/job.hdf5']
    publication=Path(receipt['path'])
    assert (publication/'artifact.tar.gz').read_bytes()==content
    assert len([p for p in publication.parent.iterdir() if p.is_dir()])==1
    assert len(receipt['common_files']['files'])==3
    for entry in receipt['common_files']['files']:
        assert Path(entry['canonical_path']).read_bytes()==files[entry['relative_path']]
    # Repacking on the source node avoids retransferring the same bytes through
    # the operator. The compressed representation may differ; every file must
    # still match the original immutable manifest.
    source=Path(spool);source.mkdir()
    for name,value in dict(files,**{'portable-artifact-manifest.json':json.dumps(manifest).encode()}).items():
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
    meta.update(batch='direct-batch',input_mode='pod_local')
    # Keep this fixture's direct export out of the earlier common episode index.
    (root/'datasets/example/episodes/job.hdf5').unlink()
    result=subprocess.run([sys.executable,'-c',code,json.dumps(meta)],input=b'',capture_output=True,check=True)
    direct=json.loads(result.stdout)
    assert direct['input_mode']=='pod_local'
    assert direct['all_regular_file_hashes_verified']
    assert Path(direct['episode']['path']).read_bytes()==files['datasets/example/episodes/job.hdf5']


def fixture_archive(tmp_path, *, omit=None, bad_sidecar=False, genuine=False, spool_path=None):
    spool=tmp_path/'original-spool' if spool_path is None else Path(spool_path)
    relative='datasets/example/episodes/job.hdf5'
    if genuine:
        import numpy as np
        from reachy_retarget.agent_dataset import write_archive
        write_archive(spool, 'example', 'job', {'timestamp':np.array([0.,.01])},
                      dict(source_sequence='recorded-episode', source_group='original-source',
                           objects={'box':{}}, source_urls=[], source_revision=None,
                           missing_fields=[], derived_fields={}, simulation_assumptions=[]))
        files={str(p.relative_to(spool)):p.read_bytes() for p in spool.rglob('*')
               if p.is_file() and p.suffix in ('.json','.hdf5')}
    else:
        files={relative:b'preserved measured states'}
        digest=hashlib.sha256(files[relative]).hexdigest()
        files['datasets/example/episodes/job.json']=json.dumps(
            dict(dataset_id='example',hdf5_sha256='f'*64 if bad_sidecar else digest)).encode()
        files['datasets/example/metadata.json']=b'{"dataset_id":"example"}'
    files['result.json']=json.dumps(dict(status='physical_fail', archive=str(spool/relative))).encode()
    if omit:
        del files[omit]
    manifest=dict(spool=str(spool), status='physical_fail',source_release_sha256='a'*64,
                  artifact_bytes=sum(map(len,files.values())),
                  files=[dict(path=p,bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for p,v in files.items()])
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name,value in dict(files,**{'portable-artifact-manifest.json':json.dumps(manifest).encode()}).items():
            member=tarfile.TarInfo('./'+name)
            member.size=len(value)
            archive.addfile(member,io.BytesIO(value))
    content=buffer.getvalue()
    root=tmp_path/'shared';root.mkdir()
    meta=dict(batch='batch',job='job',sha256=hashlib.sha256(content).hexdigest(),bytes=len(content),manifest=manifest)
    code=REMOTE.replace("Path('/mnt/reachy-retarget')",'Path('+repr(str(root))+')').replace('50_000_000_000','0')
    return root,meta,content,code,files


def run_remote(code,meta,content=b''):
    return subprocess.run([sys.executable,'-c',code,json.dumps(meta)],input=content,capture_output=True)


def test_shared_spool_publishes_without_operator_upload_and_preserves_exact_files(tmp_path):
    spool=tmp_path/'shared/shared-spool/batch/job/artifact'
    root,meta,content,code,files=fixture_archive(tmp_path,spool_path=spool)
    spool.mkdir(parents=True)
    for name,value in dict(files,**{'portable-artifact-manifest.json':json.dumps(meta['manifest']).encode()}).items():
        path=spool/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
    meta['input_mode']='shared_pvc'
    for _ in range(2):
        result=run_remote(code,meta)
        assert result.returncode==0,result.stderr.decode()
        receipt=json.loads(result.stdout)
        assert receipt['input_mode']=='shared_pvc' and receipt['all_regular_file_hashes_verified']
        assert receipt['original_spool']==str(spool)
        for entry in receipt['common_files']['files']:
            assert Path(entry['canonical_path']).read_bytes()==files[entry['relative_path']]


@pytest.mark.parametrize('escape',['different_job','symlink'])
def test_shared_spool_rejects_an_unbound_location(tmp_path,escape):
    spool=tmp_path/'shared/shared-spool/batch/job/artifact'
    root,meta,content,code,files=fixture_archive(tmp_path,spool_path=spool)
    spool.parent.mkdir(parents=True)
    if escape=='symlink':
        outside=tmp_path/'outside';outside.mkdir();spool.symlink_to(outside,target_is_directory=True)
    else:
        spool.mkdir();meta['job']='another-job'
    meta['input_mode']='shared_pvc'
    result=run_remote(code,meta)
    assert result.returncode!=0
    assert b'shared spool binding' in result.stderr


def test_repairs_legacy_hdf5_only_publication_without_upload_or_receipt_change(tmp_path):
    root,meta,content,code,files=fixture_archive(tmp_path,genuine=True)
    # Recreate exactly the old publisher's immutable receipt + tar + HDF5 only.
    target=root/'node-local-exports/batch/job';target.mkdir(parents=True)
    (target/'artifact.tar.gz').write_bytes(content)
    (target/'artifact-manifest.json').write_text(json.dumps(meta['manifest']))
    (target/'result.json').write_bytes(files['result.json'])
    relative='datasets/example/episodes/job.hdf5'
    episode=target/relative;episode.parent.mkdir(parents=True);episode.write_bytes(files[relative])
    index=root/relative;index.parent.mkdir(parents=True);index.symlink_to(episode)
    receipt=dict(schema='node-local-publication-v1',job='job',batch='batch',path=str(target),
                 operator_archive_sha256=meta['sha256'],published_archive_sha256=meta['sha256'],
                 status='physical_fail',cross_pod_readback_verified=True,
                 episode=dict(path=str(episode),relative_path=relative,sha256=hashlib.sha256(files[relative]).hexdigest()))
    original=json.dumps(receipt,indent=3).encode()
    (target/'publication.json').write_bytes(original)
    before=(target/'artifact.tar.gz').stat().st_mtime_ns
    meta['input_mode']='repair_existing'
    outputs=[]
    for _ in range(2):
        result=run_remote(code,meta)
        assert result.returncode==0,result.stderr.decode()
        outputs.append(json.loads(result.stdout))
        assert (target/'publication.json').read_bytes()==original
        assert (target/'artifact.tar.gz').stat().st_mtime_ns==before
    assert outputs[0]['common_files']==outputs[1]['common_files']
    from reachy_retarget.agent_dataset import inspect_archive
    assert inspect_archive(index)['rows']==2
    result=subprocess.run([sys.executable,'-c',READBACK,json.dumps(outputs[-1])],capture_output=True,check=True)
    report=json.loads(result.stdout)
    assert report['canonical_common_files_verified']==3
    assert all(str(root/p) in report['files'] for p in (relative,'datasets/example/episodes/job.json','datasets/example/metadata.json'))
    # Receipt alone must never skip checking or repairing lost canonical links.
    (root/'datasets/example/episodes/job.json').unlink()
    (target/'datasets/example/episodes/job.json').unlink()
    repaired=run_remote(code,meta)
    assert repaired.returncode==0,repaired.stderr.decode()
    assert inspect_archive(index)['rows']==2
    assert json.loads(repaired.stdout)['common_files']==outputs[-1]['common_files']


@pytest.mark.parametrize('missing',['datasets/example/episodes/job.json','datasets/example/metadata.json'])
def test_missing_contract_member_rejects_incomplete_export(tmp_path,missing):
    root,meta,content,code,_=fixture_archive(tmp_path,omit=missing)
    result=run_remote(code,meta,content)
    assert result.returncode!=0
    assert b'Missing hash-bound common file' in result.stderr
    assert not (root/'datasets/example/episodes/job.hdf5').exists()


def test_sidecar_must_bind_exact_hdf5(tmp_path):
    root,meta,content,code,_=fixture_archive(tmp_path,bad_sidecar=True)
    result=run_remote(code,meta,content)
    assert result.returncode!=0
    assert b'sidecar does not bind' in result.stderr
    assert not (root/'datasets/example/episodes/job.hdf5').exists()


@pytest.mark.parametrize('same_bytes',[False,True])
def test_existing_dataset_metadata_requires_identical_bytes(tmp_path,same_bytes):
    root,meta,content,code,files=fixture_archive(tmp_path)
    existing=root/'datasets/example/metadata.json';existing.parent.mkdir(parents=True)
    value=files['datasets/example/metadata.json'] if same_bytes else b'{"dataset_id": "example"}'
    existing.write_bytes(value)
    result=run_remote(code,meta,content)
    assert (result.returncode==0)==same_bytes
    assert existing.read_bytes()==value
    if not same_bytes:
        assert b'Common file checksum conflict' in result.stderr


def test_readback_checks_canonical_sidecar_not_only_extracted_copy(tmp_path):
    root,meta,content,code,_=fixture_archive(tmp_path)
    result=run_remote(code,meta,content)
    assert result.returncode==0,result.stderr.decode()
    receipt=json.loads(result.stdout)
    path=root/'datasets/example/episodes/job.json';path.unlink();path.write_bytes(b'wrong')
    result=subprocess.run([sys.executable,'-c',READBACK,json.dumps(receipt)],capture_output=True)
    assert result.returncode!=0
    assert b'Cross-pod readback mismatch' in result.stderr
    assert str(path).encode() in result.stderr


def test_completion_uses_expected_job_ids_not_available_backup_count():
    manifest={'jobs':[{'id':'one'},{'id':'two'}]}
    early=publication_summary('batch',manifest,[{'job':'one'}],[],['one'])
    assert not early['complete']
    assert early['missing_jobs']==early['missing_backups']==['two']
    assert not publication_summary('batch',manifest,[{'job':'one'},{'job':'other'}],[],['one','other'])['complete']
    assert not publication_summary('batch',manifest,[{'job':'one'},{'job':'two'},{'job':'two'}],[],['one','two'])['complete']
    assert publication_summary('batch',manifest,[{'job':'one'},{'job':'two'}],[],['one','two'])['complete']


def test_cached_local_v1_proof_still_repairs_remotely_and_verifies_all_files(tmp_path,monkeypatch):
    from cluster import export_local_spool
    root,meta,content,code,files=fixture_archive(tmp_path)
    initial=run_remote(code,meta,content)
    assert initial.returncode==0,initial.stderr.decode()
    receipt=json.loads(initial.stdout)
    receipt.pop('common_files')
    target=Path(receipt['path'])
    (target/'common-files-publication.json').unlink()
    for relative in ('datasets/example/episodes/job.json','datasets/example/metadata.json'):
        (root/relative).unlink()
        (target/relative).unlink()
    # A previous exporter even claimed readback: this must not bypass repair.
    receipt['cross_pod_readback_verified']=True
    local=tmp_path/'batch';job=local/'job';job.mkdir(parents=True)
    (local/'manifest.json').write_text(json.dumps({'jobs':[{'id':'job'}]}))
    original=json.dumps(receipt,indent=3).encode()
    (job/'publication.json').write_bytes(original)
    (job/'backup.json').write_text(json.dumps(dict(sha256=meta['sha256'],bytes=len(content),
        manifest=meta['manifest'],pod='original-pod-no-longer-present',archive='/must/not/reupload')))
    real_run=subprocess.run
    calls=[]

    def simulated_kubectl(command,**kwargs):
        calls.append(command)
        source=command[command.index('-c')+1]
        payload=command[-1]
        if source==export_local_spool.REMOTE:
            assert json.loads(payload)['input_mode']=='repair_existing'
            assert kwargs['stdin']==subprocess.DEVNULL
            assert 'available-pod' in command
            source=code
        return real_run([sys.executable,'-c',source,payload],**kwargs)

    monkeypatch.setattr(export_local_spool.subprocess,'run',simulated_kubectl)
    monkeypatch.setattr(sys,'argv',['export-local-spool',str(local),'--pod','available-pod',
                                 '--readback-pod','other-pod'])
    export_local_spool.main()
    assert len(calls)==2
    assert (job/'publication.json').read_bytes()==original
    assert json.loads((job/'common-files-readback.json').read_text())['canonical_common_files_verified']==3
    assert json.loads((local/'publication-summary.json').read_text())['complete']
    for relative in ('datasets/example/episodes/job.json','datasets/example/metadata.json'):
        assert (root/relative).read_bytes()==files[relative]
