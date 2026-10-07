"""Queue explicit, bounded native pilots and copied legacy normalized episodes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reachy_retarget.adapters import behavior,momagen,m3bench,mello


def range_spec(value,url=None,path=None):
    return dict(url=url or value['url'],path=path or value['path'],bytes=value['bytes'],sha256=value['sha256'],
                archive_member=value['path'],range=dict(offset=value['offset'],length=value['compressed_bytes'],compression=value['compression']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pod',required=True)
    args=parser.parse_args()
    jobs=[]
    for spec in behavior.describe()['fetch']:
        jobs.append(dict(id='behavior-episode-00021890',operation='native_pipeline',dataset='behavior',fetch=[spec]))
    for spec in momagen.describe()['fetch']:
        jobs.append(dict(id='momagen-'+Path(spec['path']).stem,operation='native_pipeline',dataset='momagen',fetch=[spec],episode=spec['enriched_episode']))
    m=m3bench.describe();prefix=m3bench.PILOT_SEQUENCE
    members=[range_spec(x,m['pilot']['url']) for x in m['pilot']['members']]
    members.append(range_spec(m['fk_asset'],path=prefix+'/robot.urdf'))
    jobs.append(dict(id='m3bench-pick-book-28-23',operation='native_pipeline',dataset='m3bench',fetch=members,source_directory=prefix))
    jobs.append(dict(id='mello-subject1-box-walk1',operation='native_pipeline',dataset='mello',fetch=[mello.describe()['pilot']]))
    for source in ('robocasa','bigym'):
        jobs.append(dict(id=source+'-mobile-pilot',operation='mobile_pilot',dataset=source))
    aliases={'hf__robomimic__robomimic_datasets':'robomimic','hf__amandlek__mimicgen_datasets':'mimicgen',
             'hf__haosulab__ManiSkill_Demonstrations':'maniskill','humoto':'humoto'}
    legacy=[]
    for path in sorted((ROOT/'runs/object-matrix/data/normalized').glob('*/*.h5')):
        rel='imports/legacy/'+str(path.relative_to(ROOT/'runs/object-matrix/data/normalized'))
        legacy.extend([(path,rel),(path.with_suffix('.json'),str(Path(rel).with_suffix('.json')))])
        jobs.append(dict(id='legacy-'+path.stem,operation='export_normalized',dataset=aliases[path.parent.name],source=rel))
    with tempfile.TemporaryDirectory() as tmp:
        package=Path(tmp)/'inputs.tar'
        with tarfile.open(package,'w') as tar:
            for path,relative in legacy:
                if path.exists():tar.add(path,arcname=relative,recursive=False)
            plan=Path(tmp)/'initial-jobs.json';plan.write_text(json.dumps(jobs,indent=2))
            tar.add(plan,arcname='queue/initial-jobs.json')
        with package.open('rb') as data:
            subprocess.run(['kubectl','exec','-i','-n','erl-ucsd',args.pod,'--','tar','-xf','-','-C','/mnt/reachy-retarget'],stdin=data,check=True)
        code="import json,sys;from pathlib import Path;r=Path('/mnt/reachy-retarget');v=json.loads((r/'runtime.json').read_text());sys.path.insert(0,v['source']);from reachy_retarget.cluster_pipeline import enqueue; jobs=json.loads((r/'queue/initial-jobs.json').read_text());[enqueue(r,j) for j in jobs];print('queued',len(jobs))"
        subprocess.run(['kubectl','exec','-n','erl-ucsd',args.pod,'--','/mnt/reachy-retarget/runtime/run-python','-c',code],check=True)
    print(json.dumps({'jobs':len(jobs),'native_datasets':6,'legacy_datasets':4,'legacy_payload_bytes':sum(p.stat().st_size for p,_ in legacy if p.exists())}))


if __name__=='__main__':main()
