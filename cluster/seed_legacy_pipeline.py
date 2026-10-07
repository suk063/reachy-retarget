"""Stage every existing object episode for isolated parallel IK and dynamics.

Source datasets and sibling repositories are read-only. Shared imported assets
are content-verified; each episode gets its own controller, scene and outputs.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import tarfile
import tempfile

REPO=Path(__file__).resolve().parents[1]
ALIASES={'hf__robomimic__robomimic_datasets':'robomimic',
         'hf__amandlek__mimicgen_datasets':'mimicgen',
         'hf__haosulab__ManiSkill_Demonstrations':'maniskill','humoto':'humoto'}
ASSETS=('robosuite_assets','robosuite_can_assets','maniskill_panda_assets','maniskill_scene_evidence')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pod',required=True)
    parser.add_argument('--namespace',default='erl-ucsd')
    parser.add_argument('--label',default='legacy-all-v1')
    args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',args.label):raise ValueError('Unsafe label')
    local=REPO/'runs/object-matrix'
    prefix='imports/'+args.label
    snapshots={}
    for directory in [local/'data/normalized',*[local/'data/raw'/name for name in ASSETS]]:
        for path in directory.rglob('*'):
            if path.is_file():snapshots[str(path.relative_to(local))]=path.read_bytes()
    inventory={name:{'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
               for name,content in snapshots.items()}
    jobs=[]
    with sqlite3.connect('file:'+str(local/'catalog/ledger.sqlite')+'?mode=ro',uri=True) as conn:
        conn.row_factory=sqlite3.Row
        sources={r['id']:dict(r) for r in conn.execute('SELECT * FROM sources')}
        for row in conn.execute('SELECT * FROM episodes ORDER BY source_id,id'):
            row=dict(row);meta=json.loads(snapshots[str(Path(row['path']).with_suffix('.json'))])
            sid=row['source_id'];details=json.loads(sources[sid]['details'])
            references=[]
            for raw in meta['provenance']:
                raw_relative=Path(raw['path']).relative_to(Path('data/raw'))
                raw_source=raw_relative.parts[0]
                relative=str(Path(*raw_relative.parts[1:]))
                record=conn.execute('SELECT * FROM files WHERE source_id=? AND path=?',(raw_source,relative)).fetchone()
                if record is None or record['status']!='downloaded' or record['sha256']!=raw['sha256']:
                    raise ValueError('Unverified source acquisition provenance: '+raw['path'])
                references.append(dict(raw,url=record['url'],bytes=record['bytes'],source_id=raw_source))
            task=args.label+'-'+row['id']
            job=dict(id=task,operation='legacy_pipeline',dataset=ALIASES[sid],workspace='workspaces/'+task,
                     source=row['path'],source_episode_id=row['id'],source_split=row['split'],
                     source_provenance=dict(source_urls=[r['url'] for r in references],
                                            source_revision=details.get('revision'),provenance=references),
                     input_import=prefix,input_sha256=inventory[row['path']]['sha256'],
                     scope='rerun the existing documented kinematic baseline; task-specific physical validation; task objects only; no RGB')
            jobs.append(job)
    with tempfile.TemporaryDirectory(prefix='reachy-legacy-stage-') as temp:
        package=Path(temp)/'inputs.tar'
        with tarfile.open(package,'w') as tar:
            for name,content in snapshots.items():
                info=tarfile.TarInfo(prefix+'/'+name);info.size=len(content);tar.addfile(info,io.BytesIO(content))
            for name,value in (('manifest.json',inventory),('jobs.json',jobs)):
                content=json.dumps(value,indent=2).encode();info=tarfile.TarInfo(prefix+'/'+name);info.size=len(content);tar.addfile(info,io.BytesIO(content))
        check="import shutil; assert shutil.disk_usage('/mnt/reachy-retarget').free - %d >= 50000000000" % sum(len(v) for v in snapshots.values())
        kube=['kubectl','-n',args.namespace,'exec',args.pod,'--']
        subprocess.run(kube+['python','-c',check],check=True)
        with package.open('rb') as stream:
            subprocess.run(['kubectl','-n',args.namespace,'exec','-i',args.pod,'--','tar','--skip-old-files','-xf','-','-C','/mnt/reachy-retarget'],stdin=stream,check=True)
    remote=r'''
import hashlib,json,os,shutil,sys
from pathlib import Path
r=Path('/mnt/reachy-retarget');base=r/sys.argv[1]
runtime=json.loads((r/'runtime.json').read_text());sys.path.insert(0,runtime['source'])
from reachy_retarget.cluster_pipeline import enqueue,atomic
manifest=json.loads((base/'manifest.json').read_text())
for name,expected in manifest.items():
    path=base/name
    if path.stat().st_size!=expected['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']:
        raise RuntimeError('Imported payload checksum mismatch: '+name)
jobs=json.loads((base/'jobs.json').read_text())
for job in jobs:
    workspace=r/job['workspace']
    if not workspace.exists():
        workspace.mkdir(parents=True)
        source=Path(job['source'])
        for relative in (source,source.with_suffix('.json')):
            target=workspace/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/relative,target)
        raw=workspace/'data/raw';raw.mkdir(parents=True,exist_ok=True)
        for directory in (base/'data/raw').iterdir():(raw/directory.name).symlink_to(directory,target_is_directory=True)
        atomic(workspace/'input-manifest.json',dict(import_root=str(base),source_sha256=job['input_sha256']))
    recorded=json.loads((workspace/'input-manifest.json').read_text())
    if recorded['source_sha256']!=job['input_sha256']:raise RuntimeError('Workspace already has different source')
    enqueue(r,job)
print(json.dumps({'queued':len(jobs),'datasets':sorted({j['dataset'] for j in jobs}),'label':sys.argv[1]}))
'''
    subprocess.run(kube+['/mnt/reachy-retarget/runtime/run-python','-c',remote,prefix],check=True)


if __name__=='__main__':main()
