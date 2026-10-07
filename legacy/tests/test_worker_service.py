import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('retarget_worker',Path(__file__).parents[1]/'cluster/worker_service.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


def test_terminal_cache_does_not_hide_a_new_job_or_pending_agent_review(tmp_path):
    for name in ('finished','waiting-review','new','.staging'):(tmp_path/name).mkdir()
    assert {p.parent.name for p in worker.pending_jobs(tmp_path,{'finished'})}=={'waiting-review','new'}


def test_claim_reads_current_activation_and_respects_immutable_source_pin(tmp_path):
    pin='a'*64;(tmp_path/'releases'/pin).mkdir(parents=True);(tmp_path/'provenance').mkdir()
    (tmp_path/'provenance'/f'{pin}.json').write_text(json.dumps({'source_sha256':pin,'control_sha256':'control'}))
    point=tmp_path/'runtime.json';point.write_text(json.dumps({'source':'old','source_sha256':'old','control_sha256':'control'}))
    assert worker.claim_runtime(tmp_path,{'runtime_source_sha256':pin})['source']==str(tmp_path/'releases'/pin)
    point.write_text(json.dumps({'source':'new','source_sha256':'new','control_sha256':'control'}))
    assert worker.claim_runtime(tmp_path,{})['source']=='new'
    with pytest.raises(ValueError):worker.claim_runtime(tmp_path,{'runtime_source_sha256':'../escape'})


@pytest.mark.parametrize('failure_mode', ['corrupt_result', 'log_denied'])
def test_storage_failure_does_not_stop_an_independent_job(tmp_path,monkeypatch,failure_mode):
    queue=tmp_path/'queue/tasks';bad=queue/'bad';good=queue/'good'
    bad.mkdir(parents=True);good.mkdir()
    if failure_mode == 'corrupt_result':
        (bad/'result.json').write_bytes(b'\0'*32)
    else:
        (bad/'job.json').write_text(json.dumps({'id':'bad'}))
        original_open=Path.open
        def denied_log(path,*args,**kwargs):
            if path.name=='stdout.log' and bad in path.parents:
                raise PermissionError('Shared storage rejected log creation')
            return original_open(path,*args,**kwargs)
        monkeypatch.setattr(Path,'open',denied_log)
    (good/'job.json').write_text(json.dumps({'id':'good'}))
    (tmp_path/'runtime.json').write_text(json.dumps(dict(source='release',python='python',control='control')))
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    monkeypatch.setattr(worker.time,'sleep',lambda duration:None)
    monkeypatch.setattr(worker.random,'shuffle',lambda jobs:jobs.sort(key=lambda p:p.parent.name))
    launched=[]
    class Child:
        returncode=0
        def __init__(self,command,**kwargs):
            launched.append(command)
            (good/'result.json').write_text(json.dumps({'status':'physical_fail'}))
        def poll(self):return 0
    monkeypatch.setattr(worker.subprocess,'Popen',Child)
    class CompletedIteration(Exception):pass
    def heartbeat(status,**kwargs):
        if status=='idle':raise CompletedIteration
        return True
    monkeypatch.setattr(worker,'heartbeat',heartbeat)
    with pytest.raises(CompletedIteration):worker.serve()
    assert len(launched)==1 and str(good) in launched[0]
    if failure_mode == 'corrupt_result':
        assert (bad/'result.json').read_bytes()==b'\0'*32
        assert not (bad/'attempts').exists()
    else:
        assert list((bad/'attempts').glob('*/claim.json'))
        assert not (bad/'result.json').exists()


def test_failed_flush_never_publishes_over_an_existing_record(tmp_path,monkeypatch):
    target=tmp_path/'result.json';target.write_text('{"status":"retained"}')
    def failed(fd):raise OSError('storage flush failed')
    monkeypatch.setattr(worker.os,'fsync',failed)
    with pytest.raises(OSError,match='storage flush'):worker.atomic(target,{'status':'replacement'})
    assert json.loads(target.read_text())=={'status':'retained'}
