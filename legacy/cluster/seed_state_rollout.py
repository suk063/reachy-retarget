"""Copy existing immutable local inputs and queue one complete state recording."""
import argparse
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import tarfile
import tempfile

REPO=Path(__file__).resolve().parents[1]
EPISODE='1d0214a54bfcfdfd95faa196'
SOURCE='hf__robomimic__robomimic_datasets'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pod',required=True)
    parser.add_argument('--task',default='full-state-can-v3')
    parser.add_argument('--supersedes')
    args=parser.parse_args()
    task=args.task;workspace='workspaces/'+task
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',task):
        raise ValueError('Unsafe task identifier')
    local=REPO/'runs/object-matrix'
    metadata=json.loads((local/'data/normalized'/SOURCE/(EPISODE+'.json')).read_text())
    with sqlite3.connect('file:'+str(local/'catalog/ledger.sqlite')+'?mode=ro',uri=True) as conn:
        conn.row_factory=sqlite3.Row
        source=dict(conn.execute('SELECT * FROM sources WHERE id=?',(SOURCE,)).fetchone())
        files=[]
        for raw in metadata['provenance']:
            relative=str(Path(raw['path']).relative_to(Path('data/raw')/SOURCE))
            item=dict(conn.execute('SELECT * FROM files WHERE source_id=? AND path=?',(SOURCE,relative)).fetchone())
            if item['status']!='downloaded' or item['sha256']!=raw['sha256']:
                raise ValueError('Source catalog and normalized provenance disagree')
            files.append(dict(raw,url=item['url'],bytes=item['bytes']))
    provenance=dict(source_urls=[item['url'] for item in files],
                    source_revision=json.loads(source['details'])['revision'],provenance=files)
    selected=[]
    for relative in [Path('data/normalized')/SOURCE/(EPISODE+'.h5'),Path('data/normalized')/SOURCE/(EPISODE+'.json')]:
        selected.append((local/relative,workspace+'/'+str(relative)))
    for relative in (Path('data/retargeted')/SOURCE/EPISODE,Path('data/raw/robosuite_assets'),Path('data/raw/robosuite_can_assets')):
        selected.extend((p,workspace+'/'+str(p.relative_to(local))) for p in (local/relative).rglob('*') if p.is_file())
    job=dict(id=task,operation='full_state_rollout',workspace=workspace,dataset='robomimic',
             source=f'data/normalized/{SOURCE}/{EPISODE}.h5',
             source_provenance=provenance,
             scope='complete original frozen source interval; minimal pad translation; full state, no RGB; task objects only')
    if args.supersedes:job['supersedes']=args.supersedes
    with tempfile.TemporaryDirectory() as tmp:
        package=Path(tmp)/'inputs.tar'
        with tarfile.open(package,'w') as tar:
            for p,name in selected:tar.add(p,arcname=name,recursive=False)
            spec=Path(tmp)/'job.json';spec.write_text(json.dumps(job,indent=2));tar.add(spec,arcname='queue/full-state-job.json')
        with package.open('rb') as data:subprocess.run(['kubectl','exec','-i','-n','erl-ucsd',args.pod,'--','tar','-xf','-','-C','/mnt/reachy-retarget'],stdin=data,check=True)
    code="import json,sys;from pathlib import Path;r=Path('/mnt/reachy-retarget');v=json.loads((r/'runtime.json').read_text());sys.path.insert(0,v['source']);from reachy_retarget.cluster_pipeline import enqueue;print(enqueue(r,json.loads((r/'queue/full-state-job.json').read_text())))"
    subprocess.run(['kubectl','exec','-n','erl-ucsd',args.pod,'--','/mnt/reachy-retarget/runtime/run-python','-c',code],check=True)
    print('Copied input bytes',sum(p.stat().st_size for p,_ in selected))


if __name__=='__main__':main()
