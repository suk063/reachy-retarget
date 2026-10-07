"""Conservative bimanual/mobile IK candidate, separate from physical success.

The default preserves the relative source base path. An explicit task-spec
policy can instead solve the base and only declared manipulation hands while
keeping inactive arms at Reachy neutral. Source clocks, object trajectories and
both original hand targets remain intact; task-only IK is not fixture admission.
"""
from pathlib import Path
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from .episodes import matrices_to_pose, pose_to_matrices
from .agent_dataset import write_archive
from .cluster_pipeline import atomic


def _save_raw_ik(attempt, clock, q, errors, achieved, targets, base, placement,
                 attachments, metadata, *, failure=None, clock_semantics=None):
    """Keep expensive IK results before any pose conversion or common export."""
    attempt = Path(attempt)
    attempt.mkdir(parents=True, exist_ok=True)
    path = attempt / 'raw-ik.npz'
    partial = attempt / 'raw-ik.npz.part'
    if path.exists() or partial.exists():
        raise FileExistsError('An immutable raw IK artifact already exists: ' + str(path))
    with partial.open('xb') as stream:
        np.savez_compressed(stream, original_clock_s=np.asarray(clock),
                            robot_qpos=np.asarray(q), ik_error_m_rad=np.asarray(errors),
                            achieved_hand_matrix=np.asarray(achieved),
                            target_hand_matrix=np.asarray(targets),
                            source_base_reference_matrix=np.asarray(base),
                            placement=np.asarray(placement),
                            tool_rotation_attachments=np.asarray(attachments))
    partial.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic(attempt / 'raw-ik.json', {
        'schema': 'reachy-raw-ik-attempt-v1', 'path': path.name, 'sha256': digest,
        'requested_frames': len(clock), 'completed_frames': len(q),
        'all_frames_computed': len(q) == len(clock), 'source_sequence': metadata.get('source_sequence'),
        'source_group': metadata.get('source_group'), 'source_revision': metadata.get('source_revision'),
        'source_urls': metadata.get('source_urls', []), 'failure': failure,
        'physics_validated': False, 'state_semantics': 'derived IK configurations, not physical observations',
        'clock_semantics': clock_semantics or 'original source clock retained without retiming',
    })
    return path


def reference_targets(arrays, home):
    root = pose_to_matrices(arrays['source/robot_root_pose'])
    yaw0 = np.arctan2(root[0,1,0], root[0,0,0])
    placement = np.eye(4)
    placement[:3,:3] = Rotation.from_euler('z', -yaw0).as_matrix()
    placement[:2,3] = -(placement[:3,:3] @ root[0,:3,3])[:2]
    base = placement @ root
    poses = np.stack([pose_to_matrices(arrays['hand/'+s+'_pose']) for s in ('left','right')],axis=1)
    targets = placement @ poses
    attachments = []
    for h in range(2):
        attachment = targets[0,h,:3,:3].T @ home[h,:3,:3]
        targets[:,h,:3,:3] = targets[:,h,:3,:3] @ attachment
        attachments.append(attachment.tolist())
    return base, targets, placement, attachments


def retarget_momagen(record, root, dataset, episode_id, attempt, *, planning_policy=None):
    from .robot import Robot, ARMS
    a, metadata = record['arrays'], record['metadata']
    clock = np.asarray(a['time_s'])
    if clock.ndim != 1 or len(clock) < 2 or not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0):
        raise ValueError('A finite increasing original source clock is required')
    robot = Robot(Path(attempt) / 'runtime-identity')
    base, targets, placement, attachments = reference_targets(a, robot.fk(robot.q))
    task_policy = None
    if planning_policy is not None:
        from . import momagen_task_hands
        task_policy = momagen_task_hands.validate(planning_policy, metadata)
        atomic(Path(attempt) / 'task-hand-policy.json', dict(task_policy,
            source_task_config_text=planning_policy['source_task_config_text']))
    # Default comparison retains the observed horizontal source path. The
    # optional task policy explicitly records a derived mobile base instead.
    robot.active = robot.arm_v.copy()
    q, errors, achieved = [], [], []
    last = robot.q.copy()
    if task_policy is not None:
        last = momagen_task_hands.initialize(robot, targets, task_policy)
    try:
        for i, target in enumerate(targets):
            if task_policy is None:
                yaw = np.arctan2(base[i,1,0],base[i,0,0])
                last[:4] = [base[i,0,3],base[i,1,3],np.cos(yaw),np.sin(yaw)]
                last, pe, re = robot.ik(target,last,active_hands=(0,1),iterations=60)
            else:
                last, pe, re = robot.ik(target, last,
                    active_hands=tuple(task_policy['active_hands']), iterations=100,
                    base_xy_bounds=task_policy['base_xy_bounds'])
            frame_achieved = robot.fk(last)
            q.append(last.copy()); errors.append([pe,re]); achieved.append(frame_achieved)
    except Exception as exc:
        _save_raw_ik(attempt, clock, q, errors, achieved, targets, base, placement,
                     attachments, metadata, failure={'type': type(exc).__name__, 'message': str(exc)})
        raise
    q, errors, achieved = np.asarray(q), np.asarray(errors), np.asarray(achieved)
    raw_ik = _save_raw_ik(attempt, clock, q, errors, achieved, targets, base, placement,
                          attachments, metadata)
    finite = bool(np.isfinite(q).all() and np.isfinite(errors).all())
    tolerance = (.02, .15) if task_policy is None else (.002, .02)
    within = finite and bool((errors[:,0] <= tolerance[0]).all() and (errors[:,1] <= tolerance[1]).all())
    planar_base = np.broadcast_to(np.eye(4), base.shape).copy()
    planar_base[:,:2,3] = q[:,:2]
    # Explicit (N, 1) preserves the single-axis batch shape across SciPy APIs.
    planar_base[:,:3,:3] = Rotation.from_euler('z',np.arctan2(q[:,3],q[:,2])[:,None]).as_matrix()
    out = {'timestamp':clock,'derived/robot_qpos':q,
           'derived/ik_error_m_rad':errors,
           'derived/target_hand_pose':matrices_to_pose(targets),
           'derived/achieved_hand_pose':matrices_to_pose(achieved),
           'derived/robot_joint_position':q[:,robot.arm_ids],
           'derived/joint_names':np.asarray(ARMS),
           'derived/base_pose':matrices_to_pose(planar_base),
           'reference/source_base_pose':matrices_to_pose(base),
           'source/gripper_action':a['source/gripper_action']}
    # Verified MoMaGen smooth parallel-finger command semantics: -1 closed,
    # +1 open. This is a derived desired motor target, never observed contact.
    g = np.asarray(a['source/gripper_action'])
    if g.shape != (len(clock),2) or not np.isfinite(g).all() or np.any(np.abs(g)>1+1e-5):
        raise ValueError('Unverified source gripper commands')
    out['derived/gripper_joint_target_rad'] = -.06 + 2.06 * (np.clip(g,-1,1)+1)/2
    for name,value in a.items():
        if name.startswith('objects/') and name.endswith('/pose'):
            out['reference/'+name] = matrices_to_pose(placement @ pose_to_matrices(value))
    report = dict(status='kinematic_candidate' if within else 'kinematic_failed',
                  frames=len(clock),duration_s=float(clock[-1]-clock[0]),
                  max_position_error_m=float(errors[:,0].max()),
                  max_rotation_error_rad=float(errors[:,1].max()),
                  p95_position_error_m=float(np.quantile(errors[:,0],.95)),
                  position_threshold_m=tolerance[0],rotation_threshold_rad=tolerance[1],
                  retargeted=True,kinematic_passed=within,physics_validated=False,
                  physics_status='blocked_source_geometry_and_task_contract',
                  object_state_updated_during_physics=False,
                  raw_ik_artifact=str(raw_ik),
                  placement=placement.tolist(),tool_rotation_attachments=attachments)
    if task_policy is not None:
        report['task_hand_planning'] = task_policy
        report['fixture_clearance_checked'] = False
        report['source_tool_frame_calibration_verified'] = False
    derived = {'timestamp':'Unchanged original source clock',
                    'derived/robot_qpos':'Reachy arm-only IK with source planar base path fixed',
                    'derived/gripper_joint_target_rad':'source [-1,1] continuous aperture -> Reachy [-0.06,2.0] rad; desired target, no measured contact',
                    'reference/objects':'single rigid world-frame gauge; no scaling or physical replay'}
    if task_policy is not None:
        derived['derived/robot_qpos'] = 'Task-active arm and free planar base IK; inactive arms and neck remain Reachy neutral. Not measured robot state.'
    output_metadata = dict(source_sequence=metadata['source_sequence'],
        source_group=metadata['source_group'],source_urls=metadata.get('source_urls',[]),
        source_revision=metadata.get('source_revision'),objects=metadata.get('objects',{}),
        source_metadata=metadata,source_record_status=record.get('status'),
        object_state_semantics='source task-object reference poses; no Reachy observation or physical rollout',
        derived_fields=derived,missing_fields=list(record.get('missing_fields',[]))+
        ['action','applied_controls','physics_state','controller_state_json'],
        simulation_assumptions=list(metadata.get('simulation_assumptions',[]))+[
            'Kinematic only. No contact, collision, dynamic success, or trained-policy eligibility is asserted.',
            'Initial source base yaw defines a coordinate gauge, not verified source heading semantics.',
            'Source base Z/roll/pitch are unavailable Reachy planar DOFs; source hand heights remain unchanged.',
            'Only constant hand tool rotations are aligned; no position correction or temporal dilation.'],
        status=report['status'],kinematic_validation=report,robot_identity=robot.identity)
    output = write_archive(root,dataset,episode_id+'-ik-'+Path(attempt).name.split('-')[0],out,output_metadata)
    atomic(Path(attempt)/'kinematic-validation.json',report)
    return dict(report,retarget_archive=str(output))
