"""Explicitly build one immutable runtime; reuse and verify an existing bundle.

Packages are installed on node-local disk, never once per task on CephFS.
This command does not restart workers or modify an existing environment bundle.
"""
import argparse
import json
from pathlib import Path
import subprocess

REMOTE = r'''
import fcntl,hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
root=Path('/mnt/reachy-retarget')
bundle=root/'runtime/py312-bundle.tar.gz'
receipt=root/'provenance/runtime-bundle.sha256'
bundle.parent.mkdir(parents=True,exist_ok=True)
build_lock=(bundle.parent/'bootstrap.lock').open('a')
fcntl.flock(build_lock,fcntl.LOCK_EX)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
requirements=json.load(sys.stdin)['requirements']
if bundle.exists():
    if not receipt.exists() or receipt.read_text().split()[0]!=sha(bundle):
        raise RuntimeError('Existing runtime has no matching checksum; refusing to overwrite')
    print('Verified existing immutable runtime:',bundle)
    sys.exit(0)
bundle.parent.mkdir(parents=True,exist_ok=True)
receipt.parent.mkdir(parents=True,exist_ok=True)
if shutil.disk_usage(root).free<50_000_000_000:
    raise RuntimeError('Shared storage is below the 50 decimal GB reserve')
with tempfile.TemporaryDirectory(prefix='reachy-runtime-build-') as temp:
    temp=Path(temp);venv=temp/'reachy-retarget-venv'
    lock=temp/'requirements.txt';lock.write_text(requirements)
    subprocess.run([sys.executable,'-m','venv',str(venv)],check=True)
    python=str(venv/'bin/python')
    log=root/'provenance/runtime-build.log'
    with log.open('x') as stream:
        subprocess.run([python,'-m','pip','install','--disable-pip-version-check','-r',str(lock)],
                       stdout=stream,stderr=subprocess.STDOUT,check=True)
    freeze=subprocess.check_output([python,'-m','pip','freeze'],text=True)
    (root/'provenance/runtime-pip-freeze.txt').write_text(freeze)
    subprocess.run([python,'-c','import numpy, scipy, h5py, pinocchio, mujoco; print(numpy.__version__,mujoco.__version__)'],check=True)
    packed=temp/'bundle.tar.gz'
    subprocess.run(['tar','-czf',str(packed),'-C',str(temp),'reachy-retarget-venv'],check=True)
    if shutil.disk_usage(root).free-packed.stat().st_size<50_000_000_000:
        raise RuntimeError('Runtime bundle would cross the shared disk reserve')
    partial=bundle.with_suffix('.gz.part')
    with packed.open('rb') as src,partial.open('xb') as dst:shutil.copyfileobj(src,dst)
    expected=sha(packed)
    if sha(partial)!=expected:raise RuntimeError('Bundle copy checksum mismatch')
    receipt.write_text(expected+'  '+str(bundle)+'\n')
    os.rename(partial,bundle)
    print('Published immutable runtime:',bundle)
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pod',required=True)
    parser.add_argument('--namespace',default='erl-ucsd')
    args=parser.parse_args()
    requirements=Path(__file__).with_name('runtime-requirements.txt').read_text()
    subprocess.run(['kubectl','exec','-i','-n',args.namespace,args.pod,'--','python','-c',REMOTE],
                   input=json.dumps({'requirements':requirements}),text=True,check=True)


if __name__=='__main__':
    main()
