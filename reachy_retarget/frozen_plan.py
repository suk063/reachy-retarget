"""Reuse an immutable aligned plan without repeated mesh conversion and IK."""
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import h5py
import mujoco
import numpy as np
from .cluster_pipeline import atomic, digest
from .dynamics import Candidate
from .dynamics_audit import _verify_assets, bind_scene_assets
from .episodes import read_episode, pose_to_matrices, matrices_to_pose
from .physics import initialize
from .object_scene import initialize_fixtures


def validate_alignment(details):
    if 'pad_alignment' not in details:
        raise ValueError('Reuse requires a calibrated pad plan')
    variant=details.get('alignment_variant')
    if variant=='pad_translation':
        return
    if variant=='constant_tcp_attachment':
        admission=details.get('robot_initialization',{})
        checks=admission.get('admission_checks',{})
        if (details.get('grasp_attachment') and admission.get('admitted') is True
                and checks and all(value is True for value in checks.values())
                and admission.get('completed_frames',0)>0
                and admission.get('completed_frames',0)==admission.get('total_frames',-1)):
            return
        raise ValueError('Corrected attachment must have complete subsequent IK/collision admission')
    raise ValueError('Unsupported frozen pad alignment variant')


def validate_cache_admission(details, frame_count):
    """Require complete base ancestry and a check of the current robot path."""
    admission = details.get('robot_initialization', {})
    checks = admission.get('admission_checks', {})
    if (admission.get('admitted') is not True or not checks
            or not all(value is True for value in checks.values())
            or admission.get('completed_frames', 0) <= 0
            or admission.get('completed_frames') != admission.get('total_frames')):
        raise ValueError('Cache requires complete explicit robot configuration admission')
    latest = admission
    for key in ('robot_grasp_approach', 'robot_supported_placement', 'robot_fixture_clearance'):
        if key in details:
            latest = details[key]
            checks = latest.get('admission_checks', {})
            if (latest.get('admitted') is not True or not checks
                    or not all(value is True for value in checks.values())):
                raise ValueError('Cannot cache a rejected '+key)
    if (latest.get('completed_frames') != frame_count
            or latest.get('total_frames') != frame_count):
        raise ValueError('Cache requires complete admission of the current reference frames')
    if details.get('robot_control_retiming') and 'robot_supported_placement' not in details:
        raise ValueError('Retimed cache requires subsequent full-path admission')


def save_admitted(prepared, candidate, source_path, robot, output):
    """Freeze a full admitted plan without stepping physics or inventing state observations."""
    details = deepcopy(prepared[5])
    validate_alignment(details)
    validate_cache_admission(details, len(prepared[7]))
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    model, xml, manifest = prepared[:3]
    manifest = deepcopy(manifest)
    manifest['scene_asset_hashes'] = bind_scene_assets(xml, output)
    (output/'scene.xml').write_text(xml)
    data = mujoco.MjData(model)
    initialize(model, data, robot, prepared[6], manifest['mimics'], 2.)
    initialize_fixtures(model, data, manifest)
    with h5py.File(output/'plan.h5', 'w') as handle:
        for name, value in (('time_s', prepared[7]), ('reference', prepared[8]),
                            ('gripper', prepared[9]), ('hand_goals', matrices_to_pose(prepared[10])),
                            ('object_goals', matrices_to_pose(prepared[11]))):
            handle[name] = value
        for name in ('qpos', 'qvel', 'ctrl'):
            handle['initial/'+name] = getattr(data, name)
    _, metadata = read_episode(source_path)
    details['effective_candidate'] = asdict(candidate)
    report = dict(status='kinematic_candidate', physics_tested=False, physics_validated=False,
                  success=False, source_episode=metadata['episode_id'], source_id=metadata['source_id'],
                  source_hdf5_sha256=digest(source_path), scene_sha256=digest(output/'scene.xml'),
                  plan=details, physics=manifest,
                  scope='Complete admitted robot reference and reproducible reset only; no physics steps or measured observations')
    atomic(output/'result.json', report)
    atomic(output/'binding.json', dict(parent_attempt=str(output),
        files={name:digest(output/name) for name in ('scene.xml', 'plan.h5', 'result.json')}))
    return output


def load(folder, source_path, robot, output):
    folder=Path(folder);binding=json.loads((folder/'binding.json').read_text())
    for name,checksum in binding['files'].items():
        if digest(folder/name)!=checksum:raise ValueError('Frozen plan checksum mismatch: '+name)
    report=json.loads((folder/'result.json').read_text())
    if report['source_hdf5_sha256']!=digest(source_path):
        raise ValueError('Frozen plan belongs to a different source')
    if digest(folder/'scene.xml')!=report['scene_sha256']:
        raise ValueError('Frozen scene checksum mismatch')
    details=report['plan']
    validate_alignment(details)
    _verify_assets(folder/'scene.xml',report['physics'])
    source,metadata=read_episode(source_path)
    xml=(folder/'scene.xml').read_text();model=mujoco.MjModel.from_xml_string(xml)
    with h5py.File(folder/'plan.h5') as f:
        seconds=f['time_s'][()];reference=f['reference'][()];commands=f['gripper'][()]
        goals=pose_to_matrices(f['hand_goals'][()]);objects=pose_to_matrices(f['object_goals'][()])
        initial={k:f['initial/'+k][()] for k in ('qpos','qvel','ctrl')}
    q0=robot.pack(reference[0,3:],reference[0,:3])
    data=mujoco.MjData(model);initialize(model,data,robot,q0,report['physics']['mimics'],2.)
    initialize_fixtures(model,data,report['physics'])
    for key,value in initial.items():
        if not np.allclose(getattr(data,key),value,rtol=0,atol=1e-12):
            raise ValueError('Frozen initial state cannot be reproduced: '+key)
    details=dict(details,frozen_plan_reuse=binding)
    atomic(Path(output)/'frozen-plan.json',binding)
    return Candidate(**details['effective_candidate']), (model,xml,report['physics'],source,metadata,details,
            q0,seconds,reference,commands,goals,objects)
