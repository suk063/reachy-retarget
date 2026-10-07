"""Publish an immutable source snapshot to the existing persistent worker pool.

Does not deploy/delete pods, download datasets, update sibling repositories or
switch running attempts to new code. Each new claim captures runtime.json.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

REPO=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace',default='erl-ucsd')
    parser.add_argument('--pod')
    parser.add_argument('--control',type=Path,default=REPO.parent/'reachy-control')
    parser.add_argument('--activate',action='store_true')
    args=parser.parse_args()
    k=['kubectl','-n',args.namespace]
    if args.pod is None:
        pods=json.loads(subprocess.check_output(k+['get','pods','-l','app.kubernetes.io/name=reachy-retarget-worker','-o','json']))['items']
        args.pod=next(p['metadata']['name'] for p in pods if p['status']['phase']=='Running')
    sources={}
    for directory in ('reachy_retarget','tests','configs','docs','cluster'):
        for path in (REPO/directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc',):
                sources[str(path.relative_to(REPO))]=path
    sources['pyproject.toml']=REPO/'pyproject.toml'
    sources['README.md']=REPO/'README.md'
    control={}
    for directory in ('control','asset','util'):
        for path in (args.control/directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                control[str(path.relative_to(args.control))]=path
    # Hash and upload the same bytes, even when another agent edits a file
    # while this publication is being assembled.
    sources={name:path.read_bytes() for name,path in sources.items()}
    control={name:path.read_bytes() for name,path in control.items()}
    hashes=lambda entries:{name:hashlib.sha256(content).hexdigest() for name,content in sorted(entries.items())}
    src_hashes,ctl_hashes=hashes(sources),hashes(control)
    key=lambda values:hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()
    src_key,ctl_key=key(src_hashes),key(ctl_hashes)
    base='/mnt/reachy-retarget'
    release=f'{base}/releases/{src_key}'
    reference=f'{base}/references/reachy-control/{ctl_key}'
    # Avoid retransmitting the same large, read-only mesh reference on every
    # short agent iteration. Check exact bytes remotely before omitting it.
    probe="""import hashlib,json,sys
from pathlib import Path
v=json.load(sys.stdin);root=Path(v['root']);valid=True
for name,expected in v['files'].items():
 p=root/name
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
  valid=False;break
print(json.dumps({'complete':valid}))
"""
    checked=subprocess.run(k+['exec','-i',args.pod,'--','python','-c',probe],
                           input=json.dumps({'root':reference,'files':ctl_hashes}),text=True,
                           capture_output=True,check=True)
    control_present=json.loads(checked.stdout)['complete']
    provenance={'source_sha256':src_key,'control_sha256':ctl_key,'source_files':src_hashes,'control_files':ctl_hashes,
        'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'git_branch':subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
        'includes_uncommitted_changes':True,'sibling_modified':False}
    agent_lock=json.loads((REPO/'configs/reachy-agent-main-lock.json').read_text())
    agent_revision=agent_lock['revision']
    agent_repo=REPO.parent/'reachy-agent'
    agent_names=subprocess.check_output(['git','-C',str(agent_repo),'ls-tree','-r','--name-only',agent_revision],text=True).splitlines()
    agent_names=[n for n in agent_names if n.endswith('.py') and n.startswith(('robot/','simulation/','policy/'))]
    runtime={'python':f'{base}/runtime/run-python','source':release,'control':reference,'source_sha256':src_key,'control_sha256':ctl_key,
             'reachy_agent_revision':agent_revision,'reachy_agent_source':f'{base}/references/reachy-agent/{agent_revision}'}
    with tempfile.TemporaryDirectory(prefix='reachy-retarget-publish-') as tmp:
        package=Path(tmp)/'source.tar.gz'
        with tarfile.open(package,'w:gz',compresslevel=1) as tar:
            bundles=[(f'releases/{src_key}',sources)]
            if not control_present:bundles.append((f'references/reachy-control/{ctl_key}',control))
            for prefix,entries in bundles:
                for name,content in entries.items():
                    info=tarfile.TarInfo(f'{prefix}/{name}')
                    info.size=len(content)
                    tar.addfile(info,io.BytesIO(content))
            for name in agent_names:
                content=subprocess.check_output(['git','-C',str(agent_repo),'show',agent_revision+':'+name])
                info=tarfile.TarInfo(f'references/reachy-agent/{agent_revision}/{name}')
                info.size=len(content)
                tar.addfile(info,io.BytesIO(content))
            p=Path(tmp)/'provenance.json';p.write_text(json.dumps(provenance,indent=2))
            tar.add(p,arcname=f'provenance/{src_key}.json')
        with package.open('rb') as data:
            subprocess.run(k+['exec','-i',args.pod,'--','tar','--skip-old-files','-xzf','-','-C',base],stdin=data,check=True)
        # Buffered extraction is not durable publication. Verify the exact
        # captured bytes and flush every source/controller file before the
        # runtime pointer can select this release.
        verify="""import hashlib,json,os,sys
from pathlib import Path
v=json.load(sys.stdin)
for group in v['groups']:
 root=Path(group['root'])
 for name,expected in group['files'].items():
  path=root/name;sha=hashlib.sha256()
  with path.open('rb') as handle:
   os.fsync(handle.fileno())
   for block in iter(lambda:handle.read(8*1024*1024),b''):sha.update(block)
  if sha.hexdigest()!=expected:raise ValueError('Published file checksum mismatch: '+str(path))
proof=Path(v['proof']);value=json.loads(proof.read_text())
if value!=v['expected_proof']:raise ValueError('Published provenance differs from captured snapshot')
with proof.open('rb') as handle:os.fsync(handle.fileno())
print(json.dumps({'verified':True,'source_sha256':value['source_sha256']}))
"""
        verified=subprocess.run(k+['exec','-i',args.pod,'--','python','-c',verify],
            input=json.dumps(dict(groups=[dict(root=release,files=src_hashes),dict(root=reference,files=ctl_hashes)],
                proof=f'{base}/provenance/{src_key}.json',expected_proof=provenance)),
            text=True,capture_output=True,check=True)
        if not json.loads(verified.stdout)['verified']:
            raise ValueError('Runtime publication was not verified')
        if args.activate:
            code="import json,sys,os;from pathlib import Path;r=json.load(sys.stdin);p=Path('/mnt/reachy-retarget/runtime.json');t=p.with_suffix('.tmp');f=t.open('w');f.write(json.dumps(r,indent=2));f.flush();os.fsync(f.fileno());f.close();os.replace(t,p)"
            subprocess.run(k+['exec','-i',args.pod,'--','python','-c',code],input=json.dumps(runtime),text=True,check=True)
    print(json.dumps(runtime,indent=2))


if __name__=='__main__':main()
