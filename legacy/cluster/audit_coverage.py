"""Separate historical catalog claims from present files and executed pipelines.

Read-only inventory; no discovery, transfer, source removal or robot access.
Presence and byte counts do not replace acquisition checksum verification.
"""
import argparse
from collections import Counter, defaultdict
import datetime
import json
from pathlib import Path
import sqlite3
import subprocess

REPO = Path(__file__).resolve().parents[1]
ALIASES = {
    'hf__robomimic__robomimic_datasets': 'robomimic',
    'hf__amandlek__mimicgen_datasets': 'mimicgen',
    'hf__haosulab__ManiSkill_Demonstrations': 'maniskill',
    'robocasa_native_pilot': 'robocasa', 'bigym_native_pilot': 'bigym',
}
EXCLUDED = {'cmu', 'lafan1', 'amass', 'hf__glannuzel__reachy2_pick_place'}
REMOTE = r'''
import json
from pathlib import Path
r=Path('/mnt/reachy-retarget');out=[]
folders=list((r/'data/raw').glob('*'))
for native in (r/'native').glob('*'):
    folders.extend((native/'data/raw').glob('*'))
    if (native/'raw').is_dir(): folders.append(native/'raw')
    for workspace in (native/'workspaces').glob('*'):
        folders.extend((workspace/'data/raw').glob('*'))
for folder in folders:
    files=[p for p in folder.rglob('*') if p.is_file() and '.part' not in p.name]
    dataset=folder.parent.name if folder.name=='raw' else folder.name
    out.append({'source':dataset,'path':str(folder),'files':len(files),
                'bytes':sum(p.stat().st_size for p in files)})
print(json.dumps(out))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pod', required=True)
    parser.add_argument('--evidence', type=Path, default=REPO/'docs/cluster-run-evidence.json')
    parser.add_argument('--output', type=Path, default=REPO/'docs/datasets/current-coverage.json')
    args = parser.parse_args()
    catalog = json.loads((REPO/'catalog/datasets.json').read_text())
    seeds = json.loads((REPO/'catalog/seed_status.json').read_text())
    evidence = json.loads(args.evidence.read_text())
    family = {release: seed['id'] for seed in seeds for release in seed['releases']}
    family.update(ALIASES)
    def canonical(source):
        return family.get(source, source)
    present = defaultdict(list)
    # Inspect existing stores only; opening Store itself would create a ledger.
    for database in sorted((REPO/'runs').glob('*/catalog/ledger.sqlite')):
        store = database.parent.parent
        connection = sqlite3.connect('file:'+str(database.resolve())+'?mode=ro', uri=True)
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT source_id,path,bytes,sha256 FROM files WHERE status='downloaded'").fetchall()
        grouped = defaultdict(list)
        for row in rows:
            path = store/'data/raw'/row['source_id']/row['path']
            if path.is_file():
                grouped[row['source_id']].append((path, row))
        connection.close()
        for source, files in grouped.items():
            present[canonical(source)].append(dict(location='local', store=str(store.relative_to(REPO)),
                source=source, files=len(files), bytes=sum(p.stat().st_size for p,_ in files),
                recorded_bytes_match=all(p.stat().st_size==row['bytes'] for p,row in files)))
    remote = json.loads(subprocess.check_output(['kubectl','-n','erl-ucsd','exec',args.pod,
                                                '--','python','-c',REMOTE], text=True))
    for item in remote:
        present[canonical(item['source'])].append(dict(item, location='cluster'))
    jobs = defaultdict(list)
    for item in evidence['jobs']:
        job, result = item['job'], item['result']
        jobs[canonical(job['dataset'])].append(dict(task=job['id'], operation=job['operation'],
            status=result['status'], retargeted=result.get('retargeted',False),
            physics_tested=bool((result.get('physics_report') or {}).get('physics_tested')),
            physics_validated=result.get('physics_validated',False), attempts=item['attempt_count']))
    profiles = []
    for entry in catalog:
        source = entry['id']; name = canonical(source)
        excluded = source in EXCLUDED or entry['category']=='search_exclusion' or entry['status']=='unsuitable'
        state = ('excluded' if excluded else 'pipeline_tasks_recorded' if jobs[name] else
                 'present_not_queued' if present[name] else 'not_present_in_checked_stores')
        profiles.append(dict(source=source, family=name, category=entry['category'],
            url=entry['url'], historical_catalog_status=entry['status'],
            historical_catalog_bytes=entry.get('downloaded_bytes',0), current_status=state,
            exclusion_reason=entry['reason'] if excluded else None))
    names = set(jobs)|set(present)|{s['id'] for s in seeds}
    families = {name: dict(present=present[name], tasks=jobs[name],
        status_counts=dict(Counter(job['status'] for job in jobs[name])),
        catalog_profiles=[p['source'] for p in profiles if p['family']==name]) for name in sorted(names)}
    output = dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        execution_evidence_utc=evidence['verified_utc'], catalog_profiles=len(profiles),
        historical_seed_families=len(seeds), profile_status_counts=dict(Counter(p['current_status'] for p in profiles)),
        families=families, profiles=profiles, limitations=[
            'Catalog entries include versions, mirrors, assets and exclusions; profile count is not a dataset or demo count.',
            'Historical acquired flags do not establish current file presence.',
            'Present file counts can overlap across stores and raw views; do not sum them as independent demonstrations.',
            'This inventory checks presence/size, not every payload hash; acquisition receipts retain pinned hashes.',
            'Direct agent workspaces and transfers in progress may precede queue/common-archive registration.',
            'A family-level pipeline does not imply every release or episode in that family was processed.',
            'Not present in the checked stores is not evidence that upstream access is unavailable.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    print(json.dumps({key:output[key] for key in ('verified_utc','catalog_profiles','historical_seed_families','profile_status_counts')}))


if __name__ == '__main__':
    main()
