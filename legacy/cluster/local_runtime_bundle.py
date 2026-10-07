"""Freeze tested local Python sources for pod-local execution during PVC outage.

The bundle is explicitly unpublished to shared storage. It reuses the exact
controller manifest from a verified published runtime; every worker separately
verifies those original shared controller assets before simulation.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from cluster.local_spool_experiment import verify_published_manifest, write_json


def build(repo, base_proof, output):
    repo, output = Path(repo), Path(output)
    verify_published_manifest(base_proof, base_proof['source_sha256'])
    entries = {}
    for directory in ('reachy_retarget', 'tests', 'cluster'):
        for path in (repo/directory).rglob('*.py'):
            if '__pycache__' not in path.parts:
                entries[str(path.relative_to(repo))] = path.read_bytes()
    entries['pyproject.toml'] = (repo/'pyproject.toml').read_bytes()
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in sorted(entries.items())}
    pin = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    proof = dict(source_sha256=pin, source_files=hashes,
        control_sha256=base_proof['control_sha256'], control_files=base_proof['control_files'],
        publication_scope='pod_local_unpublished', source_scope='Python package, tests, cluster launchers and pyproject.toml',
        git_revision=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        git_branch=subprocess.check_output(['git', '-C', str(repo), 'branch', '--show-current'], text=True).strip(),
        includes_uncommitted_changes=True, sibling_modified=False,
        controller_parent_release=base_proof['source_sha256'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, 'x:gz', compresslevel=1) as archive:
        files = {f'releases/{pin}/{name}': value for name, value in entries.items()}
        files[f'provenance/{pin}.json'] = json.dumps(proof, indent=2).encode()
        for name, content in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    receipt = dict(source_sha256=pin, bundle_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        bytes=output.stat().st_size, publication_scope='pod_local_unpublished', proof=proof)
    write_json(output.with_suffix(output.suffix+'.json'), receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-proof', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(Path(__file__).resolve().parents[1], json.loads(args.base_proof.read_text()), args.output)
    print(json.dumps({key: value for key, value in result.items() if key != 'proof'}, indent=2))
