"""Expose three exact admitted-plan files from a durably published archive.

No scene paths, numerical data, source files or archived metadata are rewritten.
The caller must check the parent with the ordinary frozen-plan loader before use.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile


def materialize(archive, archive_sha256, member_directory, files, destination):
    from cluster.local_spool_experiment import digest, write_json
    archive, destination = Path(archive), Path(destination)
    member_directory = PurePosixPath(member_directory)
    if member_directory.is_absolute() or '..' in member_directory.parts:
        raise ValueError('A safe relative archived directory is required')
    if (set(files) != {'scene.xml','plan.h5','result.json'}
            or any(not re.fullmatch('[0-9a-f]{64}',value) for value in files.values())):
        raise ValueError('Exact scene, plan and report hashes are required')
    if digest(archive) != archive_sha256:
        raise ValueError('Published parent archive checksum differs')
    selected = {}
    expected = {str(member_directory/name):name for name in files}
    with tarfile.open(archive) as handle:
        for member in handle:
            name = str(PurePosixPath(member.name))
            if name not in expected:
                continue
            leaf = expected[name]
            if not member.isfile() or leaf in selected:
                raise ValueError('Parent files must be unique regular archive members')
            raw = handle.extractfile(member).read()
            if hashlib.sha256(raw).hexdigest() != files[leaf]:
                raise ValueError('Archived admitted file checksum differs: '+leaf)
            selected[leaf] = raw
    if set(selected) != set(files):
        raise ValueError('Admitted files are absent from the published archive')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free-sum(map(len,selected.values())) < 50_000_000_000:
        raise ValueError('Materialization would cross the 50 decimal GB reserve')
    destination.mkdir(exist_ok=True)
    for name,raw in selected.items():
        target = destination/name
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError('Immutable materialized parent conflict: '+name)
        else:
            with target.open('xb') as output:
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
    report = dict(schema='admitted-parent-materialization-v1', archive=str(archive),
        archive_sha256=archive_sha256, archived_directory=str(member_directory),
        files_sha256=files, destination=str(destination), original_bytes_preserved=True,
        scene_paths_rewritten=False, physics_validated=False)
    binding = dict(parent_attempt=str(destination), files=files)
    for name,value in [('binding.json',binding),('materialization.json',report)]:
        target=destination/name
        if target.exists():
            if json.loads(target.read_text()) != value:
                raise ValueError('Immutable parent manifest conflict')
        else:
            write_json(target,value)
    fd=os.open(destination,os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
    return report


if __name__ == '__main__':
    print(json.dumps(materialize(**json.load(sys.stdin))))
