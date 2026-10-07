"""Materialize a portable frozen parent from an immutable local-spool archive.

Only scene asset paths and their bindings change. Original bytes remain in the
retained archive and original-inputs tree. No SDK, transfers or import-time I/O.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import xml.etree.ElementTree as ET

import h5py
import mujoco
import numpy as np

from .store import sha256
from .dynamics_audit import bind_scene_assets


def _relative(value):
    path = PurePosixPath(str(value))
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('An explicit safe relative archive path is required')
    return path


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    _sync_dir(path.parent)


def _json(path, value):
    _write(path, (json.dumps(value, indent=2, allow_nan=False)+'\n').encode())


def _reserve(path, count, minimum):
    if shutil.disk_usage(path).free-count < minimum:
        raise ValueError('Materialization would cross the declared free-space reserve')


def model_equivalence(original, relocated):
    """Compare compiled model contents except external resource path storage."""
    omitted = {'paths', 'mesh_pathadr', 'tex_pathadr', 'hfield_pathadr', 'npaths', 'nbuffer'}
    fields = []
    digest = hashlib.sha256()
    for label, left, right in (('model', original, relocated), ('option', original.opt, relocated.opt),
                                ('statistic', original.stat, relocated.stat)):
        for name in sorted(dir(left)):
            if name.startswith('_') or name in omitted:
                continue
            value = getattr(left, name)
            other = getattr(right, name)
            if isinstance(value, np.ndarray):
                if (value.dtype != other.dtype or value.shape != other.shape
                        or value.tobytes() != other.tobytes()):
                    raise ValueError('Compiled model changed beyond asset paths: '+label+'.'+name)
                raw = value.tobytes()
            elif (label != 'model' or name.startswith('n')) and isinstance(value, (int, float, np.number)):
                if value != other:
                    raise ValueError('Compiled scalar changed beyond asset paths: '+label+'.'+name)
                raw = repr(value).encode()
            elif label == 'model' and name == 'names':
                if value != other:
                    raise ValueError('Compiled names changed during asset relocation')
                raw = bytes(value)
            else:
                continue
            field = label+'.'+name
            fields.append(field)
            digest.update(field.encode()+b'\0'+raw)
    if not fields:
        raise ValueError('No numeric model comparison was performed')
    return dict(equal=True, fields=fields, physical_model_sha256=digest.hexdigest(),
                excluded_path_storage=sorted(omitted), comparator='Exact compiled array bytes and scalar values')


def replay_equivalence(original, relocated, recording, *, maximum_steps=500):
    """Integrate real recorded controls on both models after one initial reset."""
    if original.na or relocated.na or maximum_steps < 1:
        raise ValueError('A positive replay budget and stateless actuators are required')
    with h5py.File(recording, 'r') as handle:
        times = np.asarray(handle['time_s'])
        controls = np.asarray(handle['simulation/actuator_control'])
        if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
                or times[0] <= 0 or np.any(np.diff(times) <= 0)
                or controls.shape != (len(times), original.nu) or not np.isfinite(controls).all()):
            raise ValueError('Recorded real controls and increasing timestamps are required')
        dt = float(original.opt.timestep)
        steps = np.rint(np.diff(np.r_[0., times])/dt).astype(int)
        if np.any(steps < 1) or not np.allclose(np.cumsum(steps)*dt, times, atol=1e-8, rtol=0):
            raise ValueError('Recorded clock does not match the unchanged physics timestep')
        data = [mujoco.MjData(model) for model in (original, relocated)]
        for model, item in zip((original, relocated), data):
            for name in ('qpos', 'qvel', 'ctrl'):
                values = np.asarray(handle['initial/'+name])
                if values.shape != getattr(item, name).shape or not np.isfinite(values).all():
                    raise ValueError('Recorded initial state is incomplete')
                getattr(item, name)[:] = values
            mujoco.mj_forward(model, item)
        count = rows = 0
        stored_error = 0.
        for index, length in enumerate(steps):
            if count >= maximum_steps:
                break
            applied = min(int(length), maximum_steps-count)
            for item in data:
                item.ctrl[:] = controls[index]
            for _ in range(applied):
                for model, item in zip((original, relocated), data):
                    mujoco.mj_step(model, item)
                count += 1
                for name in ('qpos', 'qvel', 'ctrl'):
                    if not np.array_equal(getattr(data[0], name), getattr(data[1], name)):
                        raise ValueError('Relocated model actuator replay differs: '+name)
            for model, item in zip((original, relocated), data):
                mujoco.mj_forward(model, item)
            if applied == length:
                rows += 1
                for name in ('qpos', 'qvel'):
                    recorded = handle['simulation/'+name][index]
                    stored_error = max(stored_error, float(np.max(np.abs(recorded-getattr(data[0], name)), initial=0.)))
        if not count or stored_error > 1e-7:
            raise ValueError('Original-model prefix no longer reproduces recorded states')
    return dict(equal=True, physics_steps=count, complete_recorded_rows=rows,
                recorded_prefix_max_state_error=stored_error, relocated_max_state_difference=0.,
                duration_s=float(data[0].time), object_state_assignment='Initial reset only; real robot actuator commands thereafter',
                scope='Short relocation equivalence audit; retained original full replay audit remains separate')


def _verify_existing(output, request):
    report = json.loads((output/'materialization.json').read_text())
    if report['request'] != request:
        raise ValueError('Existing immutable parent belongs to a different materialization request')
    for relative, expected in report['files'].items():
        if sha256(output/relative) != expected:
            raise ValueError('Published parent checksum changed: '+relative)
    return report


def materialize(archive_path, output, *, archive_sha256, original_spool_root,
                workspace_relative, source_relative, attempt_relative,
                shared_roots=('/mnt/reachy-retarget',), minimum_free_bytes=50_000_000_000,
                replay_steps=500):
    """Create a ready parent only after hashes, compilation and control parity.

    ``attempt_relative`` and ``workspace_relative`` are relative to the tar's
    original spool root; ``source_relative`` is relative to the workspace.
    A completed output is verified idempotently. Incomplete attempts are retained
    and require a fresh output label, never overwritten or silently resumed.
    """
    archive_path, output = Path(archive_path).resolve(), Path(output).resolve()
    workspace = _relative(workspace_relative)
    source = _relative(source_relative)
    attempt = _relative(attempt_relative)
    try:
        attempt_in_workspace = attempt.relative_to(workspace)
    except ValueError as error:
        raise ValueError('Frozen attempt must belong to the declared workspace') from error
    origin = Path(original_spool_root)
    if not origin.is_absolute():
        raise ValueError('Original absolute spool root is required for archived asset resolution')
    request = dict(archive_sha256=archive_sha256, original_spool_root=str(origin),
                   workspace_relative=str(workspace), source_relative=str(source),
                   attempt_relative=str(attempt), replay_steps=int(replay_steps))
    if sha256(archive_path) != archive_sha256:
        raise ValueError('Input archive checksum mismatch')
    if output.exists():
        if not (output/'materialization.json').is_file():
            raise ValueError('Incomplete materialization retained; choose a fresh output label')
        return _verify_existing(output, request)
    output.parent.mkdir(parents=True, exist_ok=True)
    _reserve(output.parent, archive_path.stat().st_size*2, minimum_free_bytes)
    output.mkdir()
    try:
        return _materialize(archive_path, output, request, workspace, source, attempt,
                            attempt_in_workspace, origin, shared_roots, minimum_free_bytes)
    except Exception as error:
        _json(output/'failure.json', dict(ready=False, request=request, reason=type(error).__name__+': '+str(error)))
        raise


def _materialize(archive_path, output, request, workspace, source, attempt,
                 attempt_in_workspace, origin, shared_roots, minimum):
    retained = output/'original-artifact.tar.gz'
    with archive_path.open('rb') as src, retained.open('xb') as dst:
        shutil.copyfileobj(src, dst)
        dst.flush(); os.fsync(dst.fileno())
    if sha256(retained) != request['archive_sha256']:
        raise ValueError('Retained original archive checksum mismatch')
    parent = output/'workspace'
    target_attempt = parent/str(attempt_in_workspace)
    original_inputs = output/'original-inputs'
    with tarfile.open(retained) as tar:
        members = {}
        for member in tar:
            if member.isdir() and member.name in ('.', './'):
                continue
            key = str(_relative(member.name))
            if key in members:
                raise ValueError('Ambiguous duplicate archive member: '+key)
            members[key] = member
        def read(relative):
            member = members.get(str(relative))
            if member is None or not member.isfile():
                raise ValueError('Required regular archived input is missing: '+str(relative))
            return tar.extractfile(member).read()
        motion = PurePosixPath('data/retargeted')/source.parent.name/source.stem
        selected = [workspace/source, workspace/source.with_suffix('.json'),
                    workspace/motion/'motion.h5', workspace/motion/'validation.json']
        selected += [attempt/name for name in ('scene.xml', 'plan.h5', 'result.json', 'replay.h5', 'actuator-replay-audit.json')]
        inputs = {str(path): read(path) for path in selected}
        for relative, raw in inputs.items():
            _reserve(output, len(raw)*2, minimum)
            _write(original_inputs/relative, raw)
        original_report = json.loads(inputs[str(attempt/'result.json')])
        original_xml = inputs[str(attempt/'scene.xml')].decode()
        if (hashlib.sha256(inputs[str(workspace/source)]).hexdigest() != original_report['source_hdf5_sha256']
                or hashlib.sha256(inputs[str(attempt/'scene.xml')]).hexdigest() != original_report['scene_sha256']):
            raise ValueError('Frozen report does not bind its normalized source and scene')
        audit = json.loads(inputs[str(attempt/'actuator-replay-audit.json')])
        if audit.get('actuator_replay_pass') is not True or audit.get('scene_sha256') != original_report['scene_sha256']:
            raise ValueError('Parent requires a complete hash-bound original actuator replay audit')
        if original_report['plan'].get('alignment_variant') != 'pad_translation':
            raise ValueError('This materializer currently supports original pad-translation baselines only')
        tree = ET.fromstring(original_xml)
        if tree.findall('.//include') or tree.findall('.//attach') or tree.findall('./asset/model'):
            raise ValueError('External MJCF inclusions must be expanded before materialization')
        if len(tree.findall('compiler')) > 1:
            raise ValueError('Ambiguous compiler asset directories')
        compiler = tree.find('compiler')
        options = {} if compiler is None else dict(compiler.attrib)
        if options.get('strippath', 'false').lower() == 'true':
            raise ValueError('strippath scenes require an explicit separate relocation policy')
        expected = original_report['physics']['scene_asset_hashes']
        resolved = {key: {} for key in ('meshes', 'textures', 'hfields')}
        vfs, relocations = {}, []
        for tag, category in (('mesh', 'meshes'), ('texture', 'textures'), ('hfield', 'hfields')):
            for index, node in enumerate(tree.findall('./asset/'+tag)):
                directory = options.get('texturedir' if tag == 'texture' else 'meshdir', options.get('assetdir', ''))
                for attribute, filename in list(node.attrib.items()):
                    if attribute != 'file' and not (tag == 'texture' and attribute in
                            {'fileleft','fileright','filefront','fileback','fileup','filedown'}):
                        continue
                    loader_path = Path(filename)
                    if not loader_path.is_absolute():
                        loader_path = Path(directory)/loader_path
                    old_path = loader_path if loader_path.is_absolute() else origin/str(attempt)/loader_path
                    old_path = Path(os.path.normpath(old_path))
                    old_name = str(old_path)
                    expected_hash = expected[category].get(old_name)
                    if not expected_hash:
                        raise ValueError('Scene path is absent from original asset binding: '+old_name)
                    if old_path.is_relative_to(origin):
                        raw = read(PurePosixPath(str(old_path.relative_to(origin))))
                        location = 'original_archive'
                    elif any(old_path.resolve().is_relative_to(Path(root).resolve()) for root in shared_roots):
                        raw = old_path.read_bytes()
                        location = 'explicit_shared_source_root'
                    else:
                        raise ValueError('Asset is neither archived nor in an explicit shared source root: '+old_name)
                    if hashlib.sha256(raw).hexdigest() != expected_hash:
                        raise ValueError('Original asset checksum mismatch: '+old_name)
                    destination = output/'assets'/expected_hash/old_path.name
                    if not destination.exists():
                        _reserve(output, len(raw), minimum); _write(destination, raw)
                    elif destination.read_bytes() != raw:
                        raise ValueError('Content-addressed asset collision')
                    vfs_key = str(loader_path)
                    if vfs_key in vfs and vfs[vfs_key] != raw:
                        raise ValueError('Ambiguous virtual asset path')
                    vfs[vfs_key] = raw
                    node.set(attribute, str(destination))
                    resolved[category][old_name] = expected_hash
                    relocations.append(dict(category=category, element=f'asset/{tag}[{index}]', attribute=attribute,
                                            original_attribute=filename, original_path=old_name,
                                            materialized_path=str(destination), sha256=expected_hash,
                                            bytes=len(raw), verified_source=location))
        if resolved != expected:
            raise ValueError('Asset inventory differs from original hash binding')
    # Normalized data, original source metadata and joint/control references
    # remain byte-identical. Only the scene and current scene binding change.
    for relative in (source, source.with_suffix('.json'), motion/'motion.h5', motion/'validation.json'):
        _write(parent/str(relative), inputs[str(workspace/relative)])
    (parent/'data/raw').mkdir(parents=True, exist_ok=True)
    for name in ('plan.h5', 'replay.h5'):
        _write(target_attempt/name, inputs[str(attempt/name)])
    xml = ET.tostring(tree, encoding='unicode')
    _write(target_attempt/'scene.xml', xml.encode())
    original_model = mujoco.MjModel.from_xml_string(original_xml, assets=vfs)
    relocated_model = mujoco.MjModel.from_xml_path(str(target_attempt/'scene.xml'))
    model_proof = model_equivalence(original_model, relocated_model)
    replay_proof = replay_equivalence(original_model, relocated_model, target_attempt/'replay.h5',
                                      maximum_steps=request['replay_steps'])
    report = deepcopy(original_report)
    report['scene_sha256'] = sha256(target_attempt/'scene.xml')
    report['physics']['scene_asset_hashes'] = bind_scene_assets(xml, target_attempt)
    report['parent_materialization'] = dict(original_archive_sha256=request['archive_sha256'],
        original_result_sha256=hashlib.sha256(inputs[str(attempt/'result.json')]).hexdigest(),
        original_scene_sha256=original_report['scene_sha256'], manifest=str(output/'materialization.json'),
        scope='Asset-path relocation only; original complete validation result retained with compiled-model and recorded-control-prefix equivalence')
    _json(target_attempt/'result.json', report)
    _json(target_attempt/'binding.json', dict(parent_attempt=str(target_attempt),
        files={name:sha256(target_attempt/name) for name in ('scene.xml','plan.h5','result.json')}))
    files = {str(path.relative_to(output)):sha256(path) for path in sorted(output.rglob('*')) if path.is_file()}
    result = dict(schema='portable-frozen-parent-v1', ready=True, request=request,
        parent_workspace=str(parent), parent_attempt=str(target_attempt), source=str(source),
        original_actuator_audit=dict(path=str(original_inputs/str(attempt/'actuator-replay-audit.json')),
                                    sha256=hashlib.sha256(inputs[str(attempt/'actuator-replay-audit.json')]).hexdigest()),
        asset_relocations=relocations, model_equivalence=model_proof, actuator_prefix_equivalence=replay_proof,
        original_bytes_retained=True, original_physical_success=original_report.get('success'),
        raw_source_policy='No raw downloads copied or claimed; normalized data and provenance are exact originals. Frozen follow-ups require no raw decoding.',
        files=files)
    # Presence of this final, fsynced receipt is the publication boundary.
    for directory in sorted((p for p in output.rglob('*') if p.is_dir()), key=lambda p:len(p.parts), reverse=True):
        _sync_dir(directory)
    _json(output/'materialization.json', result)
    _sync_dir(output); _sync_dir(output.parent)
    return result
