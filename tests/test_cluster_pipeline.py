import json
from pathlib import Path
from unittest.mock import patch
import pytest
from reachy_retarget.cluster_pipeline import enqueue, under, validate_response, fetch, digest


def test_queue_jobs_are_immutable_and_idempotent(tmp_path):
    job={'id':'a','operation':'native_pipeline','dataset':'momagen'}
    path=enqueue(tmp_path,job)
    assert enqueue(tmp_path,job)==path
    with pytest.raises(ValueError):enqueue(tmp_path,dict(job,dataset='behavior'))
    assert json.loads((path/'job.json').read_text())==job


def test_paths_cannot_escape_shared_root(tmp_path):
    for value in ('../other','/etc/passwd'):
        with pytest.raises(ValueError):under(tmp_path,value)
    (tmp_path/'link').symlink_to(tmp_path.parent)
    with pytest.raises(ValueError):under(tmp_path,'link/escape')


def test_unpublished_jobs_are_invisible_to_existing_workers(tmp_path, monkeypatch):
    from reachy_retarget import cluster_pipeline as pipeline
    rename = pipeline.os.rename
    observed = []
    def publish(source, destination):
        # Reproduce the old consumer's glob at exactly the pre-rename point.
        observed.extend((tmp_path / 'queue/tasks').glob('*/job.json'))
        assert (source / 'job.json').exists()
        rename(source, destination)
    monkeypatch.setattr(pipeline.os, 'rename', publish)
    enqueue(tmp_path, {'id': 'new-job', 'operation': 'export_normalized'})
    assert observed == []
    assert len(list((tmp_path / 'queue/tasks').glob('*/job.json'))) == 1


def test_agent_decision_requires_exact_unchanged_evidence(tmp_path):
    path=tmp_path/'evidence.json';path.write_text('{}')
    request={'protocol':'v1','task_id':'a','request_id':'b','allowed_decisions':['defer'],
             'evidence_sha256':{str(path):digest(path)}}
    response={**request,'decision':'defer','observation':'Missing native object geometry','reviewer':'agent'}
    validate_response(request,response)
    for update in ({'request_id':'old'},{'decision':'execute'},{'observation':''},{'evidence_sha256':{}}):
        with pytest.raises(ValueError):validate_response(request,dict(response,**update))
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError):validate_response(request,response)


def test_disk_reserve_stops_transfer_before_network(tmp_path):
    spec={'path':'test.bin','bytes':100,'url':'https://invalid.test/test.bin'}
    from collections import namedtuple
    Usage=namedtuple('Usage','total used free')
    with patch('shutil.disk_usage',return_value=Usage(100,90,10)),patch('requests.get') as get:
        with pytest.raises(RuntimeError,match='50 decimal GB'):fetch(tmp_path,'momagen',spec)
        get.assert_not_called()


def test_cached_source_still_requires_hash_verification(tmp_path):
    path=tmp_path/'data/raw/momagen/test.bin';path.parent.mkdir(parents=True);path.write_bytes(b'123')
    spec={'path':'test.bin','bytes':3,'url':'https://invalid.test/test.bin','sha256':'0'*64}
    with patch('requests.get') as get:
        with pytest.raises(ValueError,match='SHA256'):fetch(tmp_path,'momagen',spec)
        get.assert_not_called()


def test_unknown_size_never_starts_transfer(tmp_path):
    with patch('requests.get') as get:
        with pytest.raises(ValueError,match='positive'):fetch(tmp_path,'momagen',{'path':'x','url':'https://invalid.test'})
        get.assert_not_called()


def test_agent_resume_reuses_immutable_archive(tmp_path,monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from reachy_retarget import cluster_pipeline as pipeline
    pose=np.array([[0,0,0,1,0,0,0],[0,0,.1,1,0,0,0]],dtype=float)
    record={'arrays':{'time_s':np.array([0.,.1]),'objects/can/pose':pose},
            'metadata':{'objects':{'can':{}},'source_sequence':'trial','source_group':'parent',
                        'source_urls':['https://example.test/pinned'],'source_revision':'fixed',
                        'derived_fields':{},'simulation_assumptions':[]},
            'status':'partial','missing_fields':['hand_pose']}
    mod=SimpleNamespace(describe=lambda:{'dataset':'behavior'},inspect=lambda p:{'source':'fixed'},normalize=lambda p:record)
    monkeypatch.setattr(pipeline,'adapter',lambda dataset:mod)
    monkeypatch.setattr(pipeline,'fetch',lambda *args:tmp_path/'raw')
    job={'id':'sample','dataset':'behavior','operation':'native_pipeline','fetch':[{}]}
    folder=enqueue(tmp_path,job);attempt=folder/'attempts/first'
    result=pipeline.execute(tmp_path,folder,attempt)
    assert result['status']=='waiting_for_agent'
    request=json.loads((folder/'agent/request.json').read_text())
    response={**request,'decision':'defer','reviewer':'test-agent','observation':'Missing scene geometry'}
    pipeline.atomic(folder/'agent/response.json',response)
    resumed=pipeline.execute(tmp_path,folder,folder/'attempts/second')
    assert resumed['status']=='blocked_prerequisites'
    assert len(list((tmp_path/'datasets/behavior/episodes').glob('*.hdf5')))==1


def test_mobile_offline_normalization_requires_evidence_review_without_refetch(tmp_path,monkeypatch):
    import numpy as np
    from reachy_retarget import cluster_pipeline as pipeline, mobile_pilot
    from reachy_retarget.adapters import native_mobile
    def no_network(*args,**kwargs):
        raise AssertionError('Offline normalization must not discover or fetch')
    monkeypatch.setattr(mobile_pilot,'discover',no_network)
    monkeypatch.setattr(mobile_pilot,'fetch',no_network)
    monkeypatch.setattr(mobile_pilot,'inspect',lambda *args:{'measured_object_poses':False})
    record={'arrays':{'source/frame_index':np.arange(3),'source/action':np.zeros((3,15))},
            'metadata':{'objects':{'plate':{'decoded_pose_available':False}},
                        'source_sequence':'trial','source_group':'bigym/trial',
                        'source_urls':['https://example.test/pinned'],'source_revision':'fixed',
                        'derived_fields':{},'simulation_assumptions':[]},
            'status':'blocked_source_replay_and_timestamp','missing_fields':['timestamp','object_poses']}
    monkeypatch.setattr(native_mobile,'normalize',lambda *args:record)
    folder=enqueue(tmp_path,{'id':'bigym-offline','operation':'normalize_mobile','dataset':'bigym'})
    result=pipeline.execute(tmp_path,folder,folder/'attempts/first')
    assert result['status']=='waiting_for_agent'
    request=json.loads((folder/'agent/request.json').read_text())
    assert request['allowed_decisions']==['defer','abort']
    pipeline.atomic(folder/'agent/response.json',{**request,'decision':'defer','reviewer':'test-agent',
                                                'observation':'Measured object state requires native replay'})
    resumed=pipeline.execute(tmp_path,folder,folder/'attempts/reviewed')
    assert resumed['status']=='blocked_prerequisites'
    assert resumed['normalized'] and not resumed['physics_validated']
    assert len(list((tmp_path/'datasets/bigym/episodes').glob('*.hdf5')))==1


def test_legacy_export_route_keeps_source_offline(tmp_path,monkeypatch):
    from reachy_retarget import cluster_pipeline as pipeline, agent_dataset
    source=tmp_path/'original.h5';source.write_bytes(b'untouched')
    archive=tmp_path/'exported.hdf5'
    def export(path,root,dataset,episode):
        assert path==source and path.read_bytes()==b'untouched'
        assert dataset=='legacy' and episode=='legacy-example'
        return archive
    monkeypatch.setattr(agent_dataset,'export_normalized',export)
    folder=enqueue(tmp_path,dict(id='legacy-example',dataset='legacy',operation='export_normalized',source='original.h5'))
    result=pipeline.execute(tmp_path,folder,folder/'attempts/first')
    assert result['status']=='archived' and result['archive']==str(archive)
