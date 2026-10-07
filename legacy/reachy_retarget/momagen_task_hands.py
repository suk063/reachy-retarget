"""Explicit task-hand constraints from a checksum-bound source task spec.

This is an opt-in kinematic policy. An inactive arm may stay at Reachy neutral;
its original source target remains in the archive. Nothing here admits source
assets, contact calibration, fixtures or physical success.
"""
import hashlib
import json
import numpy as np


def validate(policy, metadata):
    if policy.get('kind') != 'declared_task_hands_mobile_v1':
        raise ValueError('Unknown task-hand planning policy')
    text = policy['source_task_config_text']
    if hashlib.sha256(text.encode()).hexdigest() != policy['source_task_config_sha256']:
        raise ValueError('Source task config checksum mismatch')
    config = json.loads(text)
    name = metadata['source_sequence'].split('/')[0]
    revision = metadata['source_revision']
    if config.get('name') != name or not revision or policy['source_revision'] != revision:
        raise ValueError('Source task identity or revision mismatch')
    expected = (f'https://github.com/ChengshuLi/MoMaGen/blob/{revision}/'
                f'momagen/datasets/base_configs/{name}.json')
    if policy['source_task_config_url'] != expected:
        raise ValueError('Task config URL does not bind the source revision')
    active = set()
    witnesses = []
    phases = config['task']['task_spec']
    if not phases:
        raise ValueError('Empty task spec')
    for phase, specification in phases.items():
        for hand, side in enumerate(('left', 'right')):
            subtasks = specification['arm_'+side]
            if not subtasks:
                raise ValueError('Missing hand subtasks')
            for subtask, values in subtasks.items():
                if values.get('arm') != side or not {'object_ref', 'attached_obj'} <= values.keys():
                    raise ValueError('Unverified task-hand role')
                if values['object_ref'] is not None or values['attached_obj'] is not None:
                    active.add(hand)
                    witnesses.append(dict(phase=phase, subtask=subtask, hand=side,
                        object_ref=values['object_ref'], attached_obj=values['attached_obj']))
    if not active:
        raise ValueError('No declared manipulation hand')
    bounds = np.asarray(policy['base_xy_bounds'], float)
    yaw = policy['seed_yaw_deg']
    if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1]) or isinstance(yaw, bool)
            or not isinstance(yaw, (int, float)) or not np.isfinite(yaw) or not -180 <= yaw <= 180):
        raise ValueError('Explicit finite base bounds and yaw seed are required')
    return dict(kind=policy['kind'], active_hands=sorted(active),
        source_task_config_sha256=policy['source_task_config_sha256'],
        source_task_config_url=expected, source_revision=revision, role_witnesses=witnesses,
        base_xy_bounds=bounds.tolist(), seed_yaw_deg=float(yaw),
        scope='Union of declared task-active hands over the complete source interval; inactive arms stay at Reachy neutral. Original targets, clock and object arrays remain source references. No fixture or physics admission.')


def initialize(robot, targets, policy):
    """Seed the free planar base under the first declared active hand."""
    active = tuple(policy['active_hands'])
    robot.active = np.r_[np.arange(3), *(robot.arm_v[7*h:7*(h+1)] for h in active)]
    q = robot.q.copy()
    home = robot.fk(q)
    yaw = np.deg2rad(policy['seed_yaw_deg'])
    c, s = np.cos(yaw), np.sin(yaw)
    q[:2] = targets[0, active[0], :2, 3] - np.array([[c, -s], [s, c]]) @ home[active[0], :2, 3]
    bounds = np.asarray(policy['base_xy_bounds'])
    q[:2] = np.clip(q[:2], bounds[:, 0], bounds[:, 1])
    q[2:4] = [c, s]
    return q
