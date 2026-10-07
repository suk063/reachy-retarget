"""Extend the canonical common-state inspection with newly published batches.

Read-only on the PVC. For every state episode published by a batch whose
publication summary is complete and cross-pod verified, a pod re-hashes the
canonical index file, re-runs the measured-state / issued-control contract
check from the pinned release and confirms that no RGB arrays are stored.
Failed and alternative-model attempts are inspected the same way; this audit
never turns a physical failure into a success.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]

REMOTE = r'''
import hashlib,json,sys
v=json.load(sys.stdin);sys.path.insert(0,v['release'])
import h5py,numpy as np
from reachy_retarget.agent_dataset import assess_state_observations
out=[]
for item in v['items']:
 row=dict(item)
 try:
  sha=hashlib.sha256()
  with open(item['path'],'rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):sha.update(b)
  row['rehashed_sha256']=sha.hexdigest()
  rgb=[]
  with h5py.File(item['path'],'r') as h:
   def visit(name,obj):
    if isinstance(obj,h5py.Dataset):
     low=name.lower()
     if any(k in low for k in ('rgb','image','depth_image')) or (obj.dtype==np.uint8 and obj.ndim>=3 and obj.shape[-1] in (3,4)):rgb.append(name)
   h.visititems(visit)
  a=assess_state_observations(item['path'])
  row.update(rows=a['rows'],rgb_stored=bool(rgb),rgb_datasets=rgb,state_assessment=a,
   canonical_inspection_passed=bool(row['rehashed_sha256']==item['sha256'] and a['state_observations_complete']
    and a['complete_recorded_command_profile'] and not rgb))
 except Exception as e:
  row.update(canonical_inspection_passed=False,error=repr(e))
 out.append(row)
print(json.dumps(out))
'''


def published_episodes(batch):
    if not (batch/'publication-summary.json').exists():
        return None, []
    summary = json.loads((batch/'publication-summary.json').read_text())
    if not summary.get('complete'):
        return None, []
    rows = []
    for receipt in sorted(batch.glob('*/common-files-publication.json')):
        common = json.loads(receipt.read_text())
        readback = receipt.parent/'common-files-readback.json'
        status = json.loads((receipt.parent/'publication.json').read_text())['status']
        for entry in common['files']:
            if entry['role'] == 'episode':
                rows.append(dict(batch=batch.name, job=receipt.parent.name, status=status,
                                 path=entry['canonical_path'], sha256=entry['sha256'],
                                 cross_pod_readback=readback.exists()))
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pod', required=True)
    parser.add_argument('--release', required=True, help='Published runtime source SHA-256 providing the contract checker')
    parser.add_argument('batches', type=Path, nargs='+')
    args = parser.parse_args()
    previous = json.loads(args.previous.read_text())
    seen = {row['path'] for row in previous['rows']}
    known_batches = {b['batch'] for b in previous['batches']}
    batches, items = [], []
    for batch in args.batches:
        summary, rows = published_episodes(batch)
        if summary is None or batch.name in known_batches:
            continue
        batches.append(dict(batch=batch.name, expected=len(summary['expected_jobs']), published=summary['exported'],
                            batch_complete=summary['complete'],
                            cross_pod_verified=all(r['cross_pod_readback'] for r in rows)))
        items.extend(r for r in rows if r['path'] not in seen)
    checked = []
    for start in range(0, len(items), 40):
        process = subprocess.run(['kubectl', '-n', 'erl-ucsd', 'exec', '-i', args.pod, '--',
            '/mnt/reachy-retarget/runtime/run-python', '-c', REMOTE],
            input=json.dumps(dict(release='/mnt/reachy-retarget/releases/'+args.release, items=items[start:start+40])),
            text=True, capture_output=True, check=True, timeout=1800)
        checked.extend(json.loads(process.stdout.strip().splitlines()[-1]))
    for row in checked:
        row['pod'] = args.pod
    attempts = previous['attempts_published'] + sum(b['published'] for b in batches)
    rows = previous['rows'] + checked
    value = dict(schema=args.output.stem, previous_evidence=str(args.previous),
        new_archives_inspected=len(checked), recorded_unix=time.time(), runtime_source_sha256=args.release,
        batches=previous['batches'] + batches, attempts_published=attempts, state_archives=len(rows),
        inspection_passed=sum(bool(r.get('canonical_inspection_passed')) for r in rows),
        complete=all(r.get('canonical_inspection_passed') for r in rows) and all(b['batch_complete'] for b in batches),
        rows=rows, scope=previous['scope'])
    args.output.write_text(json.dumps(value, indent=1)+'\n')
    print(json.dumps({k: value[k] for k in ('new_archives_inspected', 'attempts_published', 'state_archives',
                                            'inspection_passed', 'complete')}))
    failed = [r for r in checked if not r.get('canonical_inspection_passed')]
    for row in failed[:10]:
        print('FAILED', row['job'], row.get('error') or {k: row['state_assessment'].get(k) for k in
              ('state_observations_complete', 'complete_recorded_command_profile', 'missing_fields')}, row.get('rgb_datasets'))


if __name__ == '__main__':
    main()
