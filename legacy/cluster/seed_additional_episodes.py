"""Stage 50 additional independent episodes from three already acquired files.

Source files are copied with verified hashes, never downloaded or modified.
Normalization and per-episode IK/physics execute in the existing cluster pool.
The 50 episodes are a new evaluation batch, not additional controller trials.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile


REMOTE = r'''
import concurrent.futures,hashlib,json,shutil,sys
from pathlib import Path
r=Path('/mnt/reachy-retarget');label=sys.argv[1];base=r/'imports'/label
runtime=json.loads((r/'runtime.json').read_text());sys.path.insert(0,runtime['source'])
from reachy_retarget.cluster_pipeline import atomic,enqueue
from reachy_retarget.normalize import robomimic
from reachy_retarget.store import Store,sha256
inventory=json.loads((base/'input-files.json').read_text())
for entry in inventory:
 p=base/entry['path']
 if p.stat().st_size!=entry['bytes'] or sha256(p)!=entry['sha256']:
  raise ValueError('Transferred source does not match its acquisition hash')
for directory in (r/'imports/legacy-all-v1/data/raw').iterdir():
 link=base/'data/raw'/directory.name
 if not link.exists():link.symlink_to(directory,target_is_directory=True)
store=Store(base)
if not (base/'normalized.json').exists():
 for source in sorted({e['source_id'] for e in inventory}):
  paths=robomimic(store,source,25)
  print(json.dumps({'source':source,'normalized':len(paths)}),flush=True)
 atomic(base/'normalized.json',{'sources':[e['source_id'] for e in inventory]})
existing={p.stem for p in (r/'imports/legacy-all-v1/data/normalized').glob('*/*.h5')}
counts={'PickPlaceCan':22,'Lift':21,'Stack_D0':7};selected={k:[] for k in counts}
for row in store.rows('episodes'):
 meta=json.loads((base/row['path']).with_suffix('.json').read_text())
 task=meta.get('env_args',{}).get('env_name')
 if task in selected and row['id'] not in existing:selected[task].append((row,meta))
jobs=[]
for task,count in counts.items():
 entries=sorted(selected[task],key=lambda item:int(item[1]['source_sequence'].rsplit('_',1)[1]))[:count]
 if len(entries)!=count:raise ValueError('Insufficient new independent episodes: '+task)
 for row,meta in entries:
  source=next(e for e in inventory if e['source_id']==row['source_id'] and e['sha256']==meta['provenance'][0]['sha256'])
  task_id=label+'-'+row['id']
  jobs.append(dict(id=task_id,parent_task=task_id,operation='legacy_pipeline',dataset=source['dataset'],
   workspace='workspaces/'+task_id,source=row['path'],source_episode_id=row['id'],
   source_provenance=source['source_provenance'],runtime_source_sha256=runtime['source_sha256'],
   evaluation_split='new_independent_evaluation',source_group=meta['source_group'],
   scope='Frozen full-source baseline, actual task-object physics and independent actuator replay; no RGB'))
manifest=dict(label=label,jobs=jobs,policy='50 previously unexecuted source episodes, selected by source sequence before outcomes:22Can,21Lift,7Stack. Raw source files and all failures retained; normalization is not physical success.')
receipt=r/'experiments'/label/'manifest.json'
if receipt.exists() and json.loads(receipt.read_text())!=manifest:raise ValueError('Immutable batch conflict')
atomic(receipt,manifest)
def stage(job):
 w=r/job['workspace']
 if not w.exists():
  w.mkdir(parents=True);source=Path(job['source'])
  for rel in (source,source.with_suffix('.json')):
   target=w/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/rel,target)
  (w/'data/raw').symlink_to(base/'data/raw',target_is_directory=True)
  atomic(w/'input-manifest.json',dict(import_root=str(base),source_sha256=sha256(w/source)))
 elif json.loads((w/'input-manifest.json').read_text())['source_sha256']!=sha256(base/job['source']):
  raise ValueError('Existing workspace source mismatch')
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(stage,jobs))
for job in jobs:enqueue(r,job)
print(json.dumps({'label':label,'queued':len(jobs),'counts':counts,'manifest':manifest}),flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pod', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    label = 'additional-independent-v1'
    parents = json.loads((repo/'configs/feasible-contact-v1.json').read_text())['jobs']
    inputs = []
    for sid in ('1d0214a54bfcfdfd95faa196', '0cacb4e6f5eb9ea88fc84d7c', 'a3cc117492cbe040d6a22525'):
        parent = next(j for j in parents if sid in j['id'])
        source = parent['source_provenance']['provenance'][0]
        path = repo/'runs/object-matrix'/source['path']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != source['sha256']:
            raise ValueError('Original acquired source hash mismatch: '+str(path))
        inputs.append(dict(path=source['path'], sha256=digest, bytes=path.stat().st_size,
                           source_id=source['source_id'], dataset=parent['dataset'],
                           source_provenance=parent['source_provenance']))
    kube = ['kubectl', '-n', 'erl-ucsd', 'exec', args.pod, '--']
    check = 'import shutil; assert shutil.disk_usage("/mnt/reachy-retarget").free-%d>=50000000000' % sum(e['bytes'] for e in inputs)
    subprocess.run(kube+['python', '-c', check], check=True)
    with tempfile.TemporaryDirectory(prefix='reachy-new-sources-') as temporary:
        package = Path(temporary)/'source-files.tar.gz'
        with tarfile.open(package, 'w:gz', compresslevel=1) as archive:
            for entry in inputs:
                archive.add(repo/'runs/object-matrix'/entry['path'],
                            arcname='imports/'+label+'/'+entry['path'], recursive=False)
            content = (json.dumps(inputs, indent=2)+'\n').encode()
            info = tarfile.TarInfo('imports/'+label+'/input-files.json'); info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        with package.open('rb') as stream:
            subprocess.run(['kubectl', '-n', 'erl-ucsd', 'exec', '-i', args.pod, '--',
                            'tar', '--skip-old-files', '-xzf', '-', '-C', '/mnt/reachy-retarget'],
                           stdin=stream, check=True)
    subprocess.run(kube+['/mnt/reachy-retarget/runtime/run-python', '-c', REMOTE, label], check=True)


if __name__ == '__main__':
    main()
