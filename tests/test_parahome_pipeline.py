import hashlib
import io
import json
import pickle
import zipfile

import numpy as np
import pytest

from reachy_retarget import parahome_pipeline as pipeline


def test_annotation_scope_excludes_background_and_absent_objects():
    result = pipeline.task_objects({'0 2': 'Carry cup', '3 5': 'Open cabinet'},
        {'Carry cup': ['cup', 'table', 'pan'], 'Open cabinet': ['cabinet']},
        {'cup': True, 'diningtable': True, 'sink': True, 'chair': True, 'pan': False})
    assert result['selected'] == ['cup', 'diningtable', 'sink']
    assert result['annotation_items_not_in_scene'] == ['pan']
    with pytest.raises(ValueError, match='No verified task objects'):
        pipeline.task_objects({'0 2': 'unknown'}, {}, {'cup': True})


def fixture_source(root):
    folder = root / 'native/parahome'
    source = folder / 'members/seq/s1'
    source.mkdir(parents=True)
    matrices = np.tile(np.eye(4), (3, 1, 1))
    transforms = {i: {'cup_base': matrices[i], 'chair_base': matrices[i]} for i in range(3)}
    values = {'joint_positions': np.ones((3, 73, 3)),
              'body_joint_orientations': np.tile([1., 0., 0., 0., 1., 0.], (3, 23, 1)),
              'body_global_transform': matrices, 'object_transformations': transforms,
              'joint_states': {}, 'hand_joint_orientations': np.ones((3, 40, 6))}
    for name, value in values.items():
        (source / (name + '.pkl')).write_bytes(pickle.dumps(value, protocol=4))
    (source / 'object_in_scene.json').write_text(json.dumps({'cup': True, 'chair': True}))
    (source / 'text_annotation.json').write_text(json.dumps({'0 2': 'Move cup'}))
    mesh = folder / 'members/scan/cup/simplified/base.obj'
    mesh.parent.mkdir(parents=True)
    mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
    rows = [{'name': str(path.relative_to(folder / 'members')), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in [*source.iterdir(), mesh]]
    (folder / 's1-receipt.json').write_text(json.dumps({'members': rows}))
    (folder / 'annot2item.json').write_text(json.dumps({'Move cup': ['cup']}))
    (folder / 'metadata.json').write_text(json.dumps({'p1': ['s1']}))
    (folder / 'joint_info.pkl').write_bytes(pickle.dumps({}))
    selection = pipeline.task_objects({'0 2': 'Move cup'}, {'Move cup': ['cup']}, {'cup': True, 'chair': True})
    (folder / 'selection.json').write_text(json.dumps({'s1': selection}))
    return folder


def test_existing_decoder_normalizes_full_clock_and_only_task_object(tmp_path):
    folder = fixture_source(tmp_path)
    records = pipeline.normalize(tmp_path)
    assert len(records) == 1
    record = records[0]
    assert set(record['metadata']['objects']) == {'cup_base'}
    assert record['metadata']['source_group'] == 'parahome/seq/s1'
    assert record['metadata']['source_participants'] == ['p1']
    np.testing.assert_array_equal(record['arrays']['time_s'], np.arange(3) / 30)
    assert not any('chair' in key for key in record['arrays'])
    assert record['metadata']['objects']['cup_base']['mesh_sha256']
    assert record['metadata']['source_revision'] == pipeline.REVISION
    assert (folder / 'members/seq/s1/object_in_scene.json').read_text() == json.dumps({'cup': True, 'chair': True})
    assert pipeline.normalize(tmp_path)[0]['metadata']['source_group'] == 'parahome/seq/s1'


def test_normalization_rejects_changed_source_member(tmp_path):
    folder = fixture_source(tmp_path)
    (folder / 'members/seq/s1/joint_states.pkl').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum changed'):
        pipeline.normalize(tmp_path)


def test_zip_directory_and_member_ranges_are_validated_offline(tmp_path, monkeypatch):
    content = io.BytesIO(b'prefix' * 12000)
    content.seek(0, 2)
    with zipfile.ZipFile(content, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('seq/s1/test.json', b'{"observed": true}')
    raw = content.getvalue()
    folder = tmp_path / 'native/parahome'
    folder.mkdir(parents=True)
    monkeypatch.setitem(pipeline.FILES, 'seq', ('fixture', len(raw), 'historical-whole-hash'))
    monkeypatch.setattr(pipeline, '_range', lambda root, archive, offset, length: raw[offset:offset+length])
    index = pipeline._directory(tmp_path, 'seq', folder)
    assert len(index) == 1 and index[0]['name'] == 'seq/s1/test.json'
    receipt = pipeline._members(tmp_path, 'seq', folder, index, 's1')
    assert (folder / 'members/seq/s1/test.json').read_bytes() == b'{"observed": true}'
    assert receipt['whole_archive_sha256_verified'] is False
    (folder / 'members/seq/s1/test.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum changed'):
        pipeline._members(tmp_path, 'seq', folder, index, 's1')
