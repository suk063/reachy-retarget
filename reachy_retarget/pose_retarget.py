"""Conservative IK for documented source TCP/base reference trajectories.

Preserves the source clock and one shared rigid world placement. This produces
derived targets, never measured Reachy state or contact/physics validation.
"""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .agent_dataset import write_archive
from .cluster_pipeline import atomic
from .episodes import matrices_to_pose, pose_to_matrices
from .mobile_retarget import _save_raw_ik


def references(arrays, home, base_key, hand_keys):
    """One planar world gauge; no translation of hands relative to objects."""
    if not hand_keys or set(hand_keys) - {'left', 'right'}:
        raise ValueError('Explicit left/right source TCP channels are required')
    base = pose_to_matrices(arrays[base_key])
    count = len(base)
    yaw = np.arctan2(base[0, 1, 0], base[0, 0, 0])
    placement = np.eye(4)
    placement[:3, :3] = Rotation.from_euler('z', -yaw).as_matrix()
    placement[:2, 3] = -(placement[:3, :3] @ base[0, :3, 3])[:2]
    base = placement @ base
    targets = np.tile(home, (count, 1, 1, 1))
    attachments = np.tile(np.eye(3), (2, 1, 1))
    active = []
    for side, key in hand_keys.items():
        hand = ('left', 'right').index(side)
        values = np.asarray(arrays[key])
        if values.shape != (count, 7) or not np.isfinite(values).all():
            raise ValueError('Source TCP rows must be finite and aligned with the base')
        if not np.allclose(np.linalg.norm(values[:, 3:], axis=1), 1, atol=1e-5, rtol=0):
            raise ValueError('Source TCP quaternion is not unit length')
        targets[:, hand] = placement @ pose_to_matrices(values)
        attachments[hand] = targets[0, hand, :3, :3].T @ home[hand, :3, :3]
        targets[:, hand, :3, :3] = targets[:, hand, :3, :3] @ attachments[hand]
        active.append(hand)
    return base, targets, placement, attachments, tuple(sorted(active))


def retarget(record, root, dataset, episode_id, attempt, *, clock_key, base_key,
             hand_keys, gripper_targets=None, mapping_metadata=None):
    from .robot import Robot, ARMS
    arrays, metadata = record['arrays'], record['metadata']
    attempt = Path(attempt)
    clock = np.asarray(arrays[clock_key])
    if clock.ndim != 1 or len(clock) < 2 or not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0):
        raise ValueError('Explicit finite increasing source or documented nominal clock required')
    base_pose = np.asarray(arrays[base_key])
    if (base_pose.shape != (len(clock), 7) or not np.isfinite(base_pose).all()
            or not np.allclose(np.linalg.norm(base_pose[:, 3:], axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError('Explicit finite world base pose with unit quaternion required')
    if not any(key.startswith('objects/') and key.endswith('/pose') for key in arrays):
        raise ValueError('Task-object reference poses are required')
    robot = Robot(attempt/'runtime-identity')
    base, targets, placement, attachments, active = references(arrays, robot.fk(robot.q), base_key, hand_keys)
    robot.active = np.concatenate([robot.arm_v[7*h:7*(h+1)] for h in active])
    q, errors, achieved = [], [], []
    last = robot.q.copy()
    clock_semantics = (mapping_metadata or {}).get('clock', 'Selected input clock retained; measured timing is not inferred')
    try:
        for index, target in enumerate(targets):
            yaw = np.arctan2(base[index, 1, 0], base[index, 0, 0])
            last[:4] = [base[index, 0, 3], base[index, 1, 3], np.cos(yaw), np.sin(yaw)]
            last, pe, re = robot.ik(target, last, active_hands=active, iterations=60)
            q.append(last.copy()); errors.append([pe, re]); achieved.append(robot.fk(last))
    except Exception as exc:
        _save_raw_ik(attempt, clock, q, errors, achieved, targets, base, placement,
                     attachments, metadata, failure={'type':type(exc).__name__, 'message':str(exc)},
                     clock_semantics=clock_semantics)
        raise
    q, errors, achieved = np.asarray(q), np.asarray(errors), np.asarray(achieved)
    raw = _save_raw_ik(attempt, clock, q, errors, achieved, targets, base, placement,
                      attachments, metadata, clock_semantics=clock_semantics)
    passed = bool(np.isfinite(q).all() and np.isfinite(errors).all()
                  and np.all(errors[:, 0] <= .02) and np.all(errors[:, 1] <= .15))
    report = dict(status='kinematic_candidate' if passed else 'kinematic_failed',
                  retargeted=True, kinematic_passed=passed, physics_validated=False,
                  frames=len(clock), duration_s=float(clock[-1]-clock[0]), active_hands=list(hand_keys),
                  max_position_error_m=float(errors[:, 0].max()), max_rotation_error_rad=float(errors[:, 1].max()),
                  position_threshold_m=.02, rotation_threshold_rad=.15,
                  raw_ik_artifact=str(raw), placement=placement.tolist(),
                  tool_rotation_attachments=attachments.tolist(), source_mapping=mapping_metadata or {})
    output = {'timestamp':clock, 'derived/robot_qpos':q, 'derived/joint_names':np.asarray(ARMS),
              'derived/robot_joint_position':q[:, robot.arm_ids], 'derived/ik_error_m_rad':errors,
              'derived/target_hand_pose':matrices_to_pose(targets),
              'derived/achieved_hand_pose':matrices_to_pose(achieved),
              'reference/source_base_pose':matrices_to_pose(base)}
    if gripper_targets is not None:
        for side, values in gripper_targets.items():
            values = np.asarray(values)
            if side not in hand_keys or values.shape != clock.shape or not np.isfinite(values).all():
                raise ValueError('Documented gripper targets must match an active hand and source clock')
            output['derived/'+side+'_gripper_joint_target_rad'] = values
    for key, values in arrays.items():
        if key.startswith('objects/'):
            if key.endswith(('/pose','_pose')):
                values = np.asarray(values)
                if values.shape != (len(clock), 7) or not np.isfinite(values).all():
                    raise ValueError('Selected task-object poses must align with the original source clock')
                output['reference/'+key] = matrices_to_pose(placement @ pose_to_matrices(values))
            else:
                # Non-pose source channels may include world-frame twists.
                # Retain them in their original frame rather than silently
                # attaching the transformed reference coordinate convention.
                output['source/'+key] = values
    meta = dict(source_sequence=metadata['source_sequence'], source_group=metadata['source_group'],
                source_urls=metadata.get('source_urls',[]), source_revision=metadata.get('source_revision'),
                source_metadata=metadata, source_mapping=mapping_metadata or {}, robot_identity=robot.identity,
                objects=metadata['objects'], status=report['status'], kinematic_validation=report,
                derived_fields={'derived/*':'Arm IK targets; never actual observations or issued controls',
                                'timestamp':'Unchanged selected input clock; origin/assumptions retained in source_mapping',
                                'reference/*':'Single rigid source-world gauge without trajectory/shape scaling'},
                missing_fields=list(record.get('missing_fields',[]))+[
                    'measured Reachy observations', 'Reachy applied controls', 'contact/physics validation',
                    'verified source TCP to Reachy gripper pad calibration', 'mapped Reachy neck target'],
                simulation_assumptions=list(metadata.get('simulation_assumptions',[]))+[
                    'Only source base XY/yaw is used; vertical/tilt base DOFs are not available on Reachy.',
                    'Constant initial tool rotation aligns with Reachy home; this is a heuristic, not pad calibration.',
                    'Inactive hand target arrays are placeholders at home; active_hands declares evaluated hands.',
                    'No retiming, object scaling, reference translation relative to objects or physical integration.'])
    archive = write_archive(root,dataset,episode_id+'-ik',output,meta)
    atomic(attempt/'kinematic-validation.json',report)
    return dict(report,retarget_archive=str(archive))
