"""Explicit selective ParaHome acquisition and task-object source normalization.

Network access is confined to fetch(). The existing human.parahome decoder is
reused through a documented local archive view; original acquired members remain
immutable. No image, SMPL-X model, mass, contact label, or object pose is invented.
"""
import fcntl
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import struct
import time
import zipfile
import zlib

from .cluster_pipeline import atomic, under, RESERVE
from .episodes import read_episode
from .store import Store, sha256, json_write

REVISION = '535dada556536a54d8b3a5185f2a153e6e3ccbca'
REPOSITORY = 'https://github.com/snuvclab/ParaHome'
# Reviewed release naming: README's cabinet is beneath the sink; joint_info
# stores its doors under sink. Other aliases reconcile annotated furniture names
# with object_in_scene and scan directory names; all articulated parts remain.
ANNOTATION_ALIASES = {'table': 'diningtable', 'trashcan': 'trashbin',
                      'cabinet': 'sink', 'freezer': 'refrigerator'}
FILES = {
    'seq': ('10MYSSM2H7f6g2n9nnXta48qmAhZ7r4yd', 2233478273, 'f5ba43903299e197af323c5c85822b3d59057e32c222aa1a28a061638336cd0e'),
    'scan': ('1-OuWvVFOFCEhut7J2t1kNbr5jv78QNFP', 135179489, 'cd88fdec27ab317a61e621c424ea6dfabd62d0e0462c1d0114abc931500501b0'),
    'metadata': ('1jPRCsotiep0nElHgyLQNjlkHsWgHbjhi', 2449, 'f7cea6a6f4aaf5580f3586deb8b1017b6b7d7e4f3b1314679bf94cea2a5b7140'),
    'joint_info': ('15fGnZn8o4I2bzQtQF-9MliwxKc2IUdzI', 2610, '253f69215639e93025ea8b3c66fa293c39a68855fd8fb6229adfd2b61e667fba'),
}


def describe():
    return {'dataset': 'parahome', 'source_revision': REVISION, 'source_urls': [REPOSITORY],
            'license': 'CC-BY-NC-SA-4.0', 'fetch_is_explicit': True,
            'archives': {name: {'url': _url(name), 'bytes': row[1], 'historical_sha256': row[2]}
                         for name, row in FILES.items()},
            'selection': 'Complete selected recordings, task objects from official annotation-to-item mapping; simplified meshes only',
            'physics_blockers': ['No verified task dynamics adapter', 'Object mass/friction/contact are not recorded'],
            'source_caveat': 'Publisher states that some objects retrieved from the cabinet were filled manually and may penetrate.'}


def _url(name):
    return 'https://drive.usercontent.google.com/download?id=' + FILES[name][0] + '&export=download&confirm=t'


def _put(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Acquired source changed: ' + str(path))
    else:
        partial = path.with_suffix(path.suffix + '.part')
        with partial.open('xb') as stream:
            stream.write(content)
        partial.replace(path)


def _range(root, archive, offset, length):
    """Bounded exact read; callers hold the cluster-wide transfer lock."""
    import requests
    total = FILES[archive][1]
    if offset < 0 or length < 1 or offset + length > total or length > 128_000_000:
        raise ValueError('Invalid or oversized archive range')
    if shutil.disk_usage(root).free - 2 * length < RESERVE:
        raise RuntimeError('50 decimal GB disk reserve prevents transfer')
    for attempt in range(3):
        try:
            with requests.get(_url(archive), headers={'Range': f'bytes={offset}-{offset+length-1}',
                               'Accept-Encoding': 'identity'}, stream=True, timeout=(20, 60)) as response:
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{offset+length-1}/{total}':
                    raise ValueError('Publisher ignored or changed exact byte range')
                content = response.raw.read(length + 1)
                if len(content) != length:
                    raise ValueError('Range payload length changed')
                return content
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2)


def _directory(root, archive, folder):
    path = folder / (archive + '-index.json')
    if path.exists():
        return json.loads(path.read_text())
    total = FILES[archive][1]
    tail = _range(root, archive, total - 65536, 65536)
    offset = tail.rfind(b'PK\x05\x06')
    if offset < 0:
        raise ValueError('ZIP end record missing')
    _, disk, central_disk, disk_count, count, size, start, comment = struct.unpack('<4s4H2IH', tail[offset:offset+22])
    if disk or central_disk or disk_count != count or start == 0xffffffff:
        raise ValueError('Unverified split or ZIP64 archive')
    central = _range(root, archive, start, size)
    entries, cursor = [], 0
    while cursor < len(central):
        h = struct.unpack('<4s6H3I5H2I', central[cursor:cursor+46])
        if h[0] != b'PK\x01\x02' or h[3] & 1 or h[4] not in (0, 8):
            raise ValueError('Unverified encrypted/compressed ZIP member')
        n, extra, note = h[10:13]
        name = central[cursor+46:cursor+46+n].decode('utf-8')
        under(folder / 'members', name)
        entries.append(dict(name=name, bytes=h[9], compressed_bytes=h[8], crc32=h[7],
                            header_offset=h[16], compression=h[4]))
        cursor += 46 + n + extra + note
    if len(entries) != count:
        raise ValueError('ZIP directory entry count mismatch')
    _put(folder / (archive + '-central.bin'), central)
    atomic(path, entries)
    return entries


def _members(root, archive, folder, entries, label):
    """Read a contiguous source region, verify every selected member's CRC."""
    receipt_path = folder / (label + '-receipt.json')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        for row in receipt['members']:
            if sha256(under(folder / 'members', row['name'])) != row['sha256']:
                raise ValueError('Acquired member checksum changed')
        return receipt
    if not entries:
        raise ValueError('No selected source members')
    start = min(row['header_offset'] for row in entries)
    end = min(FILES[archive][1], max(row['header_offset'] + row['compressed_bytes'] + 1024 for row in entries))
    if shutil.disk_usage(root).free - sum(row['bytes'] for row in entries) - (end-start) < RESERVE:
        raise RuntimeError('50 decimal GB reserve prevents member extraction')
    content = _range(root, archive, start, end-start)
    _put(folder / (label + '.zip-range'), content)
    records = []
    for row in entries:
        pos = row['header_offset'] - start
        h = struct.unpack('<4s5H3I2H', content[pos:pos+30])
        if h[0] != b'PK\x03\x04':
            raise ValueError('ZIP member header mismatch')
        n, extra = h[-2:]
        if content[pos+30:pos+30+n].decode('utf-8') != row['name']:
            raise ValueError('ZIP member filename mismatch')
        encoded = content[pos+30+n+extra:pos+30+n+extra+row['compressed_bytes']]
        if row['compression'] == 8:
            decoder = zlib.decompressobj(-15)
            decoded = decoder.decompress(encoded, row['bytes'] + 1)
            if not decoder.eof:
                raise ValueError('Oversized/incomplete ZIP member')
        elif row['compression'] == 0:
            decoded = encoded
        else:
            raise ValueError('Unsupported ZIP compression')
        if len(decoded) != row['bytes'] or zlib.crc32(decoded) != row['crc32']:
            raise ValueError('ZIP member length/CRC mismatch')
        _put(under(folder / 'members', row['name']), decoded)
        records.append(dict(row, sha256=hashlib.sha256(decoded).hexdigest()))
    receipt = dict(url=_url(archive), source_archive_bytes=FILES[archive][1],
                   historical_whole_archive_sha256=FILES[archive][2], whole_archive_sha256_verified=False,
                   range=[start, end-1], range_sha256=hashlib.sha256(content).hexdigest(), members=records)
    atomic(receipt_path, receipt)
    return receipt


def task_objects(annotations, mapping, in_scene):
    """Exact annotation mapping plus whitespace/underscore spelling equivalence."""
    canonical = lambda name: re.sub(r'[ _-]', '', name).casefold()
    names = ([name for name, present in in_scene.items() if present is True]
             if isinstance(in_scene, dict) else list(in_scene))
    by_name = {canonical(name): name for name in names}
    if len(by_name) != len(names):
        raise ValueError('Ambiguous scene-object names')
    selected, unknown, absent = set(), [], set()
    for interval, label in annotations.items():
        begin, end = map(int, interval.split())
        if begin < 0 or end < begin or not isinstance(label, str):
            raise ValueError('Invalid original text annotation interval')
        if label not in mapping:
            unknown.append(label)
            continue
        for name in mapping[label]:
            key = canonical(name)
            match = by_name.get(ANNOTATION_ALIASES.get(key, key))
            if match is None:
                absent.add(name)
            else:
                selected.add(match)
    if not selected:
        raise ValueError('No verified task objects in the selected recording')
    return dict(selected=sorted(selected), unmatched_annotations=sorted(set(unknown)),
                annotation_items_not_in_scene=sorted(absent),
                naming_aliases=ANNOTATION_ALIASES,
                rule='Official annot2item exact labels, spelling normalization and explicit reviewed release-name aliases')


def fetch(root, sequences=('s1', 's2', 's3')):
    """Explicit public range acquisition under the shared cluster disk reserve."""
    import requests
    root = Path(root)
    folder = root / 'native/parahome'
    folder.mkdir(parents=True, exist_ok=True)
    for seq in sequences:
        if not re.fullmatch(r's[1-9][0-9]*', seq):
            raise ValueError('Unsafe sequence identifier')
    lockpath = root / 'queue/transfer.lock'
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lockpath.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            mapping_path = folder / 'annot2item.json'
            if not mapping_path.exists():
                if shutil.disk_usage(root).free - 100_000 < RESERVE:
                    raise RuntimeError('50 decimal GB reserve prevents metadata transfer')
                url = f'https://raw.githubusercontent.com/snuvclab/ParaHome/{REVISION}/data/annot2item.json'
                response = requests.get(url, timeout=(20, 60)); response.raise_for_status()
                if len(response.content) > 100_000:
                    raise ValueError('Oversized annotation metadata')
                _put(mapping_path, response.content)
            mapping = json.loads(mapping_path.read_text())
            for name, suffix in (('metadata', '.json'), ('joint_info', '.pkl')):
                path = folder / (name + suffix)
                if not path.exists():
                    _put(path, _range(root, name, 0, FILES[name][1]))
                if sha256(path) != FILES[name][2]:
                    raise ValueError('Publisher metadata hash changed')
            seq_entries = _directory(root, 'seq', folder)
            scan_entries = _directory(root, 'scan', folder)
            selections = {}
            for seq in sequences:
                rows = [row for row in seq_entries if row['name'].startswith('seq/' + seq + '/') and row['bytes']]
                if any(not row['name'].endswith(('.json', '.pkl')) for row in rows):
                    raise ValueError('Selected sequence contains unreviewed non-state files')
                _members(root, 'seq', folder, rows, seq)
                seqdir = folder / 'members/seq' / seq
                selection = task_objects(json.loads((seqdir / 'text_annotation.json').read_text()), mapping,
                                         json.loads((seqdir / 'object_in_scene.json').read_text()))
                selections[seq] = selection
                for object_name in selection['selected']:
                    meshes = [row for row in scan_entries if row['name'].startswith('scan/' + object_name + '/simplified/')
                              and row['name'].endswith('.obj')]
                    for mesh in meshes:
                        label = 'mesh-' + hashlib.sha256(mesh['name'].encode()).hexdigest()[:16]
                        _members(root, 'scan', folder, [mesh], label)
            atomic(folder / 'selection.json', selections)
            return {'status': 'selected_source_acquired', 'sequences': selections,
                    'source_revision': REVISION, 'physics_validated': False}
    except Exception as exc:
        atomic(folder / 'fetch-failure.json', {'error': type(exc).__name__ + ': ' + str(exc),
                                              'sequences': list(sequences), 'partial_sources_preserved': True})
        raise


def normalize(root):
    """Return normalized records and local paths; never fetch on normalization."""
    from .human import parahome
    root = Path(root)
    folder = root / 'native/parahome'
    selection = json.loads((folder / 'selection.json').read_text())
    outputs = []
    for seq, scope in selection.items():
        workspace = folder / 'workspaces' / seq
        native = workspace / 'data/raw/parahome'
        native.mkdir(parents=True, exist_ok=True)
        receipts = [json.loads(path.read_text()) for path in folder.glob('*-receipt.json')]
        expected = {row['name']: row['sha256'] for receipt in receipts for row in receipt['members']}
        source_dir = folder / 'members/seq' / seq
        inputs = list(source_dir.glob('*'))
        meshes = [path for name in scope['selected'] for path in (folder / 'members/scan' / name / 'simplified').glob('*.obj')]
        for path in inputs + meshes:
            member = str(path.relative_to(folder / 'members'))
            if member not in expected or sha256(path) != expected[member]:
                raise ValueError('Selected input checksum changed: ' + member)
        in_scene = json.loads((source_dir / 'object_in_scene.json').read_text())
        selected_scene = ({name: in_scene[name] for name in scope['selected']} if isinstance(in_scene, dict)
                          else scope['selected'])
        # This derived view filters declared object scope before the existing
        # decoder writes pose channels. Original downloaded JSON stays untouched.
        for archive_name, paths in (('seq.zip', inputs), ('scan.zip', meshes)):
            content = io.BytesIO()
            with zipfile.ZipFile(content, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(paths):
                    name = str(path.relative_to(folder / 'members'))
                    data = json.dumps(selected_scene).encode() if path.name == 'object_in_scene.json' else path.read_bytes()
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, data)
            _put(native / archive_name, content.getvalue())
        _put(native / 'joint_info.pkl', (folder / 'joint_info.pkl').read_bytes())
        paths = list((workspace / 'data/normalized/parahome').glob('*.h5'))
        if not paths:
            paths = parahome(Store(workspace), 'parahome', 1)
        if len(paths) != 1:
            raise ValueError('Expected exactly one complete recording per workspace')
        path = paths[0]
        arrays, metadata = read_episode(path)
        provenance = [{'path': str(member), 'sha256': sha256(member),
                       'archive_member': str(member.relative_to(folder / 'members'))}
                      for member in inputs + meshes]
        for name in ('annot2item.json', 'metadata.json', 'joint_info.pkl'):
            provenance.append({'path': str(folder / name), 'sha256': sha256(folder / name)})
        participants = json.loads((folder / 'metadata.json').read_text())
        participant = [name for name, sequences in participants.items() if seq in sequences]
        metadata.update(source_revision=REVISION,
                        source_urls=[REPOSITORY, _url('seq'), _url('scan'), _url('metadata'), _url('joint_info')],
                        provenance=provenance, license='CC-BY-NC-SA-4.0', task_object_selection=scope,
                        source_participants=participant,
                        archive_view={'derivation': 'Selected full sequence members; object_in_scene filtered by official task annotations; selected simplified OBJ only',
                                      'original_members_preserved': True, 'seq_view_sha256': sha256(native / 'seq.zip')},
                        source_integrity='Selected bytes verified by current ZIP CRC32 and local SHA256; historical whole-archive SHA256 not reverified',
                        object_coverage={'selected_parent_objects': scope['selected'], 'decoded_part_names': list(metadata['objects']),
                                         'unmatched_annotations': scope['unmatched_annotations'],
                                         'annotation_items_not_in_scene': scope['annotation_items_not_in_scene']},
                        simulation_assumptions=['30 Hz source camera-synchronized timeline, no cropping or retiming.',
                            'Some cabinet-object positions were manually filled by publisher; physical alignment is not guaranteed.',
                            'No task-success physics contract, object mass/friction, or contact measurements.'],
                        derived_fields={'time_s': 'Source frame index divided by documented 30 Hz; full recording retained',
                                        'hand/*_pose': 'Existing human.parahome anatomical wrist mapping; not Reachy pad poses'},
                        physics_validated=False)
        for spec in metadata['objects'].values():
            spec['articulation_metadata'] = str((native / 'joint_info.pkl').relative_to(workspace))
        json_write(path.with_suffix('.json'), metadata)
        outputs.append({'arrays': arrays, 'metadata': metadata, 'status': 'normalized_source_reference',
                        'missing_fields': list(metadata.get('missing', [])) + ['verified task dynamics adapter'],
                        'normalized_path': str(path), 'workspace': str(workspace)})
    return outputs
