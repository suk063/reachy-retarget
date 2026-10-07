"""Immutable minimal-change experiments on existing full-source retargets.

Control, geometric and timing corrections are explicit immutable job inputs.
Alternative global contact models are reported separately from source-fidelity
passes. Source contact intent is separate from numerical gripper motor targets.
"""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil

import h5py
import numpy as np

from .cluster_pipeline import atomic, digest, identifier, under


VARIANTS = {
    'contact_002': {'closure_offset_rad': .02},
    'contact_0035': {'closure_offset_rad': .035},
    'contact_005': {'closure_offset_rad': .05},
    'contact_010': {'closure_offset_rad': .10},
    'contact_020': {'closure_offset_rad': .20},
    'force_1n': {'force_feedback': True, 'soft_force_feedback': True, 'balanced_force': True, 'grasp_force': 1.},
    'force_2n': {'force_feedback': True, 'soft_force_feedback': True, 'balanced_force': True, 'grasp_force': 2.},
}

TRACKING_VARIANTS = {
    'contact005_integral': {'closure_offset_rad': .05, 'integral_compensation': True},
    'contact010_integral': {'closure_offset_rad': .10, 'integral_compensation': True},
    'contact005_velocity': {'closure_offset_rad': .05, 'integral_compensation': True, 'arm_velocity_feedforward': 1.},
    'contact010_velocity': {'closure_offset_rad': .10, 'integral_compensation': True, 'arm_velocity_feedforward': 1.},
    'preforce1_integral': {'preclose_offset_rad': .02, 'force_feedback': True, 'soft_force_feedback': True, 'balanced_force': True, 'grasp_force': 1., 'integral_compensation': True},
    'preforce2_velocity': {'preclose_offset_rad': .02, 'force_feedback': True, 'soft_force_feedback': True, 'balanced_force': True, 'grasp_force': 2., 'integral_compensation': True, 'arm_velocity_feedforward': 1.},
}


def variant(candidate, prepared, name):
    bank = {**VARIANTS, **TRACKING_VARIANTS}
    if name not in bank:
        raise ValueError('Unknown predeclared experiment')
    settings = dict(bank[name])
    details = dict(prepared[5])
    if 'closure_offset_rad' in settings:
        calibration = details.get('pad_alignment')
        if not calibration:
            raise ValueError('Contact-aperture experiment requires verified pad alignment')
        offset = settings.pop('closure_offset_rad')
        angle = float(calibration['inferred_contact_angle_rad'])
        settings['gripper_closed_target'] = float(np.clip(angle-offset, -.06, 2.))
    if 'preclose_offset_rad' in settings:
        calibration = details.get('pad_alignment')
        if not calibration:
            raise ValueError('Preclosure requires verified pad alignment')
        offset = settings.pop('preclose_offset_rad')
        settings['force_preclose_target'] = float(np.clip(calibration['inferred_contact_angle_rad']+offset, -.06, 2.))
    changed = replace(candidate, **settings)
    details.update(effective_candidate=asdict(changed), experiment_variant=name,
                   experiment_settings=bank[name],
                   closed_command_semantics='Original negative values are contact intent; actual motor target is recorded separately',
                   frozen_factors=['object geometry/mass/friction', 'source clock', 'arm/base reference', 'initial physical state', 'validation gates'])
    if details.get('robot_control_retiming'):
        details['frozen_factors'].remove('source clock')
        details['frozen_factors'].append('inherited derived robot clock; complete original source clock map retained')
    result = list(prepared); result[5] = details
    return changed, tuple(result)


def stage(root, job):
    """Copy the frozen numeric input; assets remain read-only shared references."""
    root = Path(root)
    parent = under(root,job['parent_workspace']); workspace = under(root,job['workspace'])
    workspace.mkdir(parents=True,exist_ok=False)
    source = Path(job['source'])
    def copy_verified(original, target):
        expected=digest(original)
        shutil.copy2(original,target)
        with target.open('rb') as handle:
            os.fsync(handle.fileno())
        if digest(target)!=expected:
            raise ValueError('Staged input checksum differs from preserved parent: '+str(target))
    for relative in (source,source.with_suffix('.json')):
        target = workspace/relative;target.parent.mkdir(parents=True,exist_ok=True)
        copy_verified(under(parent,relative),target)
    motion = Path('data/retargeted')/source.parent.name/source.stem
    target = workspace/motion;target.mkdir(parents=True)
    for name in ('motion.h5','validation.json'):
        copy_verified(under(parent,motion/name),target/name)
    (workspace/'data/raw').symlink_to(parent/'data/raw',target_is_directory=True)
    if job.get('parent_attempt'):
        attempt=under(root,job['parent_attempt']);frozen=workspace/'frozen-plan';frozen.mkdir()
        for name in ('scene.xml','plan.h5','result.json'):
            copy_verified(attempt/name,frozen/name)
        atomic(frozen/'binding.json',dict(parent_attempt=str(attempt),
            files={name:digest(frozen/name) for name in ('scene.xml','plan.h5','result.json')}))
    atomic(workspace/'input-manifest.json',dict(source_sha256=digest(workspace/source),
        motion_sha256=digest(target/'motion.h5'),parent_workspace=str(parent),job=job))
    return workspace


def run(root,job):
    """Keep a geometric admission rejection distinct from infrastructure failure."""
    try:
        return _run(root,job)
    except ValueError as error:
        workspace=under(root,job['workspace']);plans=workspace/'plans'/job['id']
        rejected=[]
        for path in (plans/'robot-initialization/result.json',plans/'retiming/result.json',
                     plans/'grasp-approach/result.json',plans/'grasp-closure/result.json',
                     plans/'grasp-region/result.json',plans/'grasp-roll/result.json',
                     plans/'cylindrical-grasp/result.json',
                     plans/'cylindrical-acquisition/result.json',
                     plans/'box-acquisition/result.json',
                     plans/'planning-fixtures/result.json',
                     plans/'fixture-clearance/result.json',
                     plans/'grasp-reset-alignment/result.json',plans/'supported-placement/result.json'):
            if path.exists():
                report=json.loads(path.read_text())
                if report.get('admitted') is False or report.get('duration_ratio',0)>3:
                    rejected.append(dict(path=str(path),report=report))
        if not rejected:raise
        result=dict(status='plan_rejected',physics_validated=False,physics_tested=False,
                    reason=str(error),admission_failures=rejected,variant=job['variant'],
                    evaluation_split=job['evaluation_split'],parent_task=job['parent_task'])
        atomic(workspace/'experiment-result.json',result)
        return result


def _run(root,job):
    from .store import Store
    from .robot import Robot
    from .episodes import read_episode
    from .pad_alignment import paired_plans
    from .dynamics import rollout
    from .dynamics_audit import verify
    from .agent_dataset import write_archive,assess_state_observations
    root=Path(root);identifier(job['id']);workspace=under(root,job['workspace'])
    source=under(workspace,job['source']);_,meta=read_episode(source)
    manifest=json.loads((workspace/'input-manifest.json').read_text())
    if manifest['source_sha256']!=digest(source):raise ValueError('Frozen input changed')
    row=dict(id=meta['episode_id'],source_id=meta['source_id'],source_sequence=meta['source_sequence'],
             path=job['source'],source_provenance=job.get('source_provenance'))
    store=Store(workspace);plans=workspace/'plans'/job['id'];plans.mkdir(parents=True,exist_ok=False)
    robot=Robot(workspace)
    if job.get('parent_attempt'):
        from . import frozen_plan
        candidate,after=frozen_plan.load(workspace/'frozen-plan',source,robot,plans)
    else:
        candidate,before,after=paired_plans(store,row,robot,plans,state_profile=True)
    if after is None:
        return dict(status='plan_rejected',reason='Translation-only pad calibration failed; source retained',
                    plans=str(plans),physics_validated=False,source_episode=row['id'])
    prepared=after
    acquisition_keys=[key for key in ('cylindrical_acquisition','box_acquisition') if key in job]
    if len(acquisition_keys)>1:
        raise ValueError('An experiment must select exactly one acquisition geometry adapter')
    acquisition_key=acquisition_keys[0] if acquisition_keys else None
    mobile_parameters=job.get('mobile_fixture_ik')
    if mobile_parameters is not None:
        if not isinstance(mobile_parameters,dict) or any(key in job for key in
                ('base_xyyaw','base_offset_xyyaw','planning_fixtures','grasp_region','open_prefix_correction')):
            raise ValueError('Mobile fixture IK requires one explicit base policy without fixed-base, fixture forecast or grasp-region overrides')
    has_base_admission = mobile_parameters is not None or 'base_xyyaw' in job or 'base_offset_xyyaw' in job
    original_hand_targets=np.array(after[10],copy=True)
    for key,setting in (('simulation_profile','contact_profile'),('solver_sensitivity','solver_profile')):
        inherited=prepared[5].get(key,{})
        if inherited.get('source_fidelity') is False and job.get(setting)!=inherited.get('profile'):
            raise ValueError('Inherited alternative model must be explicitly declared: '+str(inherited.get('profile')))
    extra_sources={Path(__file__).name:Path(__file__).read_text()}
    if job.get('parent_attempt'):
        extra_sources['frozen_plan.py']=Path(frozen_plan.__file__).read_text()
    if 'planning_fixtures' in job:
        if 'base_xyyaw' not in job and 'base_offset_xyyaw' not in job:
            raise ValueError('Fixture forecasts require fresh complete base/IK admission')
        if job.get('supported_placement'):
            raise ValueError('Fixture forecasts do not support inserted placement clocks')
        from . import planning_fixtures
        prepared=planning_fixtures.bind(prepared,source,plans/'planning-fixtures',**job['planning_fixtures'])
        extra_sources['planning_fixtures.py']=Path(planning_fixtures.__file__).read_text()
    if 'cylindrical_grasp' in job:
        if any(job.get(key) for key in ('grasp_attachment','grasp_region','grasp_reset_alignment','grasp_roll')):
            raise ValueError('Cylindrical pad alignment requires its own explicit attachment comparison')
        if not has_base_admission:
            raise ValueError('Cylindrical pad alignment requires fresh complete base/IK admission')
        from . import cylindrical_grasp
        try:
            prepared=cylindrical_grasp.apply(prepared,robot,plans/'cylindrical-grasp',**job['cylindrical_grasp'])
        except ValueError as error:
            atomic(plans/'cylindrical-grasp/result.json',dict(admitted=False,physics_validated=False,
                   reason=str(error),stage='original_mesh_and_finite_pad_attachment'))
            raise
        extra_sources['cylindrical_grasp.py']=Path(cylindrical_grasp.__file__).read_text()
    if job.get('grasp_attachment'):
        from . import grasp_attachment
        if 'base_xyyaw' not in job and 'base_offset_xyyaw' not in job:
            raise ValueError('Attachment correction requires explicit full base/IK admission')
        prepared=grasp_attachment.apply(prepared,robot,plans/'grasp-attachment',**job['grasp_attachment'])
        extra_sources['grasp_attachment.py']=Path(grasp_attachment.__file__).read_text()
    if job.get('grasp_region'):
        if job.get('grasp_attachment') or 'base_offset_xyyaw' not in job:
            raise ValueError('Component grasp requires a separate explicit base-offset admission')
        from . import grasp_region
        prepared=grasp_region.apply(prepared,robot,plans/'grasp-region',**job['grasp_region'])
        extra_sources['grasp_region.py']=Path(grasp_region.__file__).read_text()
    if job.get('grasp_reset_alignment'):
        from . import grasp_reset_alignment
        if 'base_xyyaw' not in job and 'base_offset_xyyaw' not in job:
            raise ValueError('Reset alignment requires subsequent complete IK/collision admission')
        prepared=grasp_reset_alignment.apply(prepared,plans/'grasp-reset-alignment',**job['grasp_reset_alignment'])
        extra_sources['grasp_reset_alignment.py']=Path(grasp_reset_alignment.__file__).read_text()
    if job.get('grasp_roll'):
        if 'base_xyyaw' not in job and 'base_offset_xyyaw' not in job:
            raise ValueError('Pad-line roll requires subsequent complete base/IK admission')
        from . import grasp_roll
        prepared=grasp_roll.apply(prepared,plans/'grasp-roll',**job['grasp_roll'])
        extra_sources['grasp_roll.py']=Path(grasp_roll.__file__).read_text()
    if job.get('open_prefix_correction'):
        if 'base_offset_xyyaw' not in job or job.get('base_reference_policy','preserve_trajectory')!='preserve_trajectory':
            raise ValueError('Open-prefix correction requires an explicit smoothly introduced base offset')
        from . import open_prefix_correction
        prepared=open_prefix_correction.apply(prepared,original_hand_targets,plans/'open-prefix-correction',
                                               **job['open_prefix_correction'])
        extra_sources['open_prefix_correction.py']=Path(open_prefix_correction.__file__).read_text()
    reuse_approach=job.get('approach_reuse_admitted_reference',False)
    if (type(reuse_approach) is not bool or
            (reuse_approach and (not job.get('approach_before_base_admission') or mobile_parameters is None))):
        raise ValueError('Admitted approach reuse requires a pre-proposal and complete mobile admission')
    if job.get('approach_before_base_admission'):
        if not job.get('grasp_approach') or not has_base_admission:
            raise ValueError('Pre-admission approach proposal requires a complete base/IK and approach validation')
        from . import grasp_approach
        prepared=grasp_approach.propose(prepared,plans/'grasp-approach-proposal',**job['grasp_approach'])
        extra_sources['grasp_approach.py']=Path(grasp_approach.__file__).read_text()
    if job.get('unloaded_return'):
        if 'base_xyyaw' not in job and 'base_offset_xyyaw' not in job:
            raise ValueError('Return correction requires subsequent complete base/IK admission')
        from . import unloaded_return
        prepared=unloaded_return.propose(prepared,plans/'unloaded-return-proposal',**job['unloaded_return'])
        extra_sources['unloaded_return.py']=Path(unloaded_return.__file__).read_text()
    # Admission must inspect the declared controller aperture, not an unrelated
    # mechanical endpoint inherited before contact calibration.
    candidate,prepared=variant(candidate,prepared,job['variant'])
    if mobile_parameters is not None:
        from . import mobile_fixture_ik
        prepared=mobile_fixture_ik.prepare(prepared,robot,plans/'robot-initialization',**mobile_parameters)
        extra_sources['mobile_fixture_ik.py']=Path(mobile_fixture_ik.__file__).read_text()
    if 'base_xyyaw' in job:
        from . import feasible_maniskill
        if (job.get('grasp_attachment') or 'cylindrical_grasp' in job) and prepared[5].get('task') != 'PickCube-v1':
            from . import feasible_base
            prepared=feasible_base.prepare(prepared,robot,plans/'robot-initialization',job['base_xyyaw'],
                                           right_arm_seed=job.get('right_arm_seed'))
            extra_sources['feasible_base.py']=Path(feasible_base.__file__).read_text()
        else:
            prepared=feasible_maniskill.prepare(prepared,robot,plans/'robot-initialization',job['base_xyyaw'])
        extra_sources['feasible_maniskill.py']=Path(feasible_maniskill.__file__).read_text()
    if 'base_offset_xyyaw' in job:
        from . import feasible_base, feasible_maniskill
        offset=np.asarray(job['base_offset_xyyaw'],dtype=float)
        if offset.shape!=(3,) or not np.isfinite(offset).all():raise ValueError('Finite three-component base offset required')
        base_solver = grasp_region.admit if job.get('grasp_region') else feasible_base.prepare
        base_policy=job.get('base_reference_policy','preserve_trajectory')
        if base_policy not in ('preserve_trajectory','stationary_initial'):
            raise ValueError('Unknown explicit base reference policy')
        base_reference=(prepared[8][0,:3] if base_policy=='stationary_initial' else prepared[8][:,:3])+offset
        if job.get('open_prefix_correction'):
            base_reference=open_prefix_correction.base_track(prepared,offset,**job['open_prefix_correction'])
        prepared=base_solver(prepared,robot,plans/'robot-initialization',base_reference,
                                       right_arm_seed=job.get('right_arm_seed'))
        extra_sources['feasible_base.py']=Path(feasible_base.__file__).read_text()
        extra_sources['feasible_maniskill.py']=Path(feasible_maniskill.__file__).read_text()
    if job.get('grasp_approach'):
        from . import grasp_approach
        approach_options=dict(job['grasp_approach'])
        if reuse_approach:
            approach_options['preserve_admitted_reference']=True
        prepared=grasp_approach.prepare(prepared,robot,plans/'grasp-approach',**approach_options)
        extra_sources['grasp_approach.py']=Path(grasp_approach.__file__).read_text()
    if (mobile_parameters is not None or 'cylindrical_grasp' in job or job.get('open_prefix_correction')
            or job.get('approach_before_base_admission') or job.get('unloaded_return')):
        prepared[5]['joint_reference_valid']=True
        prepared[5]['requires_full_ik_and_collision_admission']=False
    if job.get('servo_joint_margin_rad'):
        margin=float(job['servo_joint_margin_rad'])
        if not .025 <= margin <= .1:raise ValueError('Servo target margin must be between 25 and 100 mrad')
        candidate=replace(candidate,servo_joint_margin_rad=margin)
        prepared[5]['effective_candidate']=asdict(candidate)
    if has_base_admission or job.get('grasp_approach'):
        prepared[5]['frozen_factors']=[s for s in prepared[5]['frozen_factors'] if s not in ('arm/base reference','initial physical state')]
        prepared[5]['frozen_factors'].append('moving object initial physical state')
    if job.get('retime'):
        from . import feasible_timing
        prepared=feasible_timing.prepare(prepared,plans/'retiming',**job.get('retiming_settings',{}))
        extra_sources['feasible_timing.py']=Path(feasible_timing.__file__).read_text()
    if 'retimed_source_clearance' in job:
        from . import retimed_source_clearance
        prepared=retimed_source_clearance.prepare(prepared,robot,plans/'retimed-source-clearance',
                                                **job['retimed_source_clearance'])
        extra_sources['retimed_source_clearance.py']=Path(retimed_source_clearance.__file__).read_text()
    if job.get('grasp_closure'):
        from . import grasp_closure
        prepared=grasp_closure.apply(prepared,plans/'grasp-closure',**job['grasp_closure'])
        extra_sources['grasp_closure.py']=Path(grasp_closure.__file__).read_text()
    if job.get('supported_placement'):
        from . import supported_placement
        prepared=supported_placement.prepare(prepared,robot,plans/'supported-placement',**job['supported_placement'])
        candidate=replace(candidate,contact_release=False,calibrate_attachment=False,arm_velocity_feedforward=0.)
        prepared[5]['effective_candidate']=asdict(candidate)
        extra_sources['supported_placement.py']=Path(supported_placement.__file__).read_text()
    if 'source_prefix_velocity_feedforward' in job:
        gain=float(job['source_prefix_velocity_feedforward'])
        if not np.isfinite(gain) or not 0. <= gain <= 1.:
            raise ValueError('Source-prefix velocity feedforward must be between zero and one')
        if not prepared[5].get('robot_supported_placement'):
            raise ValueError('Source-prefix compensation requires an explicit placement phase map')
        candidate=replace(candidate,source_prefix_velocity_feedforward=gain)
        prepared[5]['effective_candidate']=asdict(candidate)
    if 'source_prefix_base_velocity_feedforward' in job:
        gain=float(job['source_prefix_base_velocity_feedforward'])
        if not np.isfinite(gain) or not 0. <= gain <= 1.:
            raise ValueError('Source-prefix base compensation must be between zero and one')
        if not prepared[5].get('robot_supported_placement'):
            raise ValueError('Source-prefix base compensation requires an explicit placement phase map')
        candidate=replace(candidate,source_prefix_base_velocity_feedforward=gain)
        prepared[5]['effective_candidate']=asdict(candidate)
    if 'base_velocity_feedforward' in job:
        gain=float(job['base_velocity_feedforward'])
        if not np.isfinite(gain) or not 0. <= gain <= 1.:
            raise ValueError('Whole-source base compensation must be between zero and one')
        if job.get('supported_placement') or acquisition_key is not None:
            raise ValueError('Whole-source base compensation cannot cover inserted feedback phases')
        candidate=replace(candidate,base_velocity_feedforward=gain)
        prepared[5]['effective_candidate']=asdict(candidate)
    if 'post_acquisition_base_velocity_feedforward' in job:
        gain=float(job['post_acquisition_base_velocity_feedforward'])
        if not np.isfinite(gain) or not 0. <= gain <= 1.:
            raise ValueError('Post-acquisition base compensation must be between zero and one')
        if (acquisition_key is None or job.get('supported_placement')
                or candidate.base_velocity_feedforward or candidate.source_prefix_base_velocity_feedforward):
            raise ValueError('Post-acquisition base compensation requires an isolated acquisition clock')
        candidate=replace(candidate,post_acquisition_base_velocity_feedforward=gain)
        prepared[5]['effective_candidate']=asdict(candidate)
    if job.get('terminal_hold'):
        from . import terminal_hold
        prepared=terminal_hold.apply(prepared,plans/'terminal-hold',**job['terminal_hold'])
        extra_sources['terminal_hold.py']=Path(terminal_hold.__file__).read_text()
    if job.get('geometry_clearance'):
        from . import geometry_clearance
        settings=dict(job['geometry_clearance'])
        settings.setdefault('support_pairs',geometry_clearance.reachy_wheel_floor_pairs(prepared[0]))
        prepared=geometry_clearance.prepare(prepared,robot,plans/'fixture-clearance',**settings)
        extra_sources['geometry_clearance.py']=Path(geometry_clearance.__file__).read_text()
    if 'carried_object_clearance' in job:
        from . import carried_object_clearance
        prepared[5]['carried_object_clearance']=carried_object_clearance.prepare(
            prepared,plans/'carried-object-clearance',**job['carried_object_clearance'])
        extra_sources['carried_object_clearance.py']=Path(carried_object_clearance.__file__).read_text()
    if job.get('admission_only'):
        if acquisition_key is not None:
            raise ValueError('Acquisition waits require their own measured execution; do not freeze as static admission')
        if job.get('contact_profile') or job.get('solver_profile'):
            raise ValueError('Geometry admission must precede numerical sensitivity profiles')
        from .frozen_plan import save_admitted
        frozen = save_admitted(prepared, candidate, source, robot, plans/'admitted-plan')
        result = dict(status='kinematic_candidate', physics_tested=False, physics_validated=False,
                      source_episode=row['id'], admitted_plan=str(frozen),
                      evaluation_split=job['evaluation_split'], parent_task=job['parent_task'],
                      variant=job['variant'], admission=prepared[5]['robot_initialization'])
        atomic(workspace/'experiment-result.json', result)
        return result
    acquisition_plan = None
    if acquisition_key is not None:
        from . import acquisition_policy, acquisition_clock, acquisition_rendezvous
        schema=next(schema for schema,key in acquisition_policy.SCHEMAS.items() if key==acquisition_key)
        geometry_adapter=acquisition_policy.adapter({'schema':schema})
        if job.get('contact_profile') or job.get('solver_profile'):
            raise ValueError('Initial acquisition evaluation requires the unchanged original physical model')
        source_clock = None
        timing = prepared[5].get('robot_control_retiming')
        if timing:
            if digest(Path(timing['source_clock_artifact'])) != timing['artifact_sha256']:
                raise ValueError('Acquisition source clock checksum differs')
            with np.load(timing['source_clock_artifact'], allow_pickle=False) as clocks:
                source_clock = clocks['source_reference_time_s'].copy()
        prepared, acquisition_plan = geometry_adapter.prepare(prepared, robot,
            plans/acquisition_key.replace('_','-'), source_reference_time_s=source_clock, **job[acquisition_key])
        for module in (geometry_adapter, acquisition_policy, acquisition_clock, acquisition_rendezvous):
            extra_sources[Path(module.__file__).name] = Path(module.__file__).read_text()
    alternate=bool(job.get('contact_profile') or job.get('solver_profile'))
    profile=job.get('contact_profile') or job.get('solver_profile') or 'source-fidelity'
    if job.get('contact_profile') and job.get('solver_profile'):
        raise ValueError('Keep contact compliance and friction solver sensitivities separate')
    if job.get('contact_profile'):
        from . import contact_profile
        if job['contact_profile']!=contact_profile.NAME:raise ValueError('Unknown contact profile')
        prepared=contact_profile.prepare(prepared,plans/'contact-profile',job.get('physics_timestep_s',.001))
        extra_sources['contact_profile.py']=Path(contact_profile.__file__).read_text()
    if job.get('solver_profile'):
        from . import solver_profile
        prepared=solver_profile.prepare(prepared,plans/'solver-profile',job['solver_profile'])
        extra_sources['solver_profile.py']=Path(solver_profile.__file__).read_text()
    report=rollout(store,row,job['id'],candidate,prepared_plan=prepared,state_profile=True,
                   extra_sources=extra_sources,acquisition_plan=acquisition_plan)
    attempt=workspace/'runs/dynamics'/job['id']/row['id']
    audit=verify(attempt) if (attempt/'replay.h5').exists() else {'actuator_replay_pass':False}
    audit_pass=bool(audit.get('independent_validation_pass',audit['actuator_replay_pass']))
    model_pass=bool(report['physics_validated'] and audit_pass)
    passed=bool(model_pass and not alternate)
    status=(('alternate_model_pass' if alternate else 'physical_pass') if model_pass
            else 'independent_audit_failed' if report['physics_validated'] and not audit_pass else report['status'])
    result=dict(status=status,physics_validated=passed,
                alternate_model_physics_validated=bool(model_pass and alternate),simulation_profile=profile,
                physics_report=report,actuator_audit=audit,source_episode=row['id'],variant=job['variant'],
                evaluation_split=job['evaluation_split'],physics_attempt=str(attempt),archive=None)
    recorded=report.get('state_recording',{});path=recorded.get('archive')
    if path:
        arrays={}
        with h5py.File(path) as f:
            m=json.loads(f.attrs['metadata_json'])
            def read(name,value):
                if isinstance(value,h5py.Dataset):arrays[name]=value.asstr()[()] if value.dtype.kind in 'OUS' else value[()]
            f.visititems(read)
        m.update(physics_validated=passed,actuator_replay_audit=audit,evaluation_split=job['evaluation_split'],
                 experiment_variant=job['variant'],parent_task=job['parent_task'],
                 status=result['status'],
                 alternate_model_physics_validated=bool(model_pass and alternate),simulation_profile=profile)
        if report['plan'].get('robot_terminal_hold'):
            m['robot_terminal_hold']=report['plan']['robot_terminal_hold']
        if report['plan'].get('robot_unloaded_return_proposal'):
            m['robot_unloaded_return_proposal']=report['plan']['robot_unloaded_return_proposal']
        if report['plan'].get('cylindrical_grasp'):
            m['cylindrical_grasp']=report['plan']['cylindrical_grasp']
        timing=report['plan'].get('robot_control_retiming')
        reference_indices = np.arange(len(arrays['timestamp']))
        if report.get('acquisition_execution'):
            from .acquisition_policy import from_details
            recorded_acquisition_key, acquisition_metadata = from_details(report['plan'])
            with h5py.File(attempt/'replay.h5') as replay:
                reference_indices = replay['source/reference_index'][()]
                for key in ('reference_index', 'reference_time_s', 'original_gripper_intent'):
                    arrays['reference/'+key] = replay['source/'+key][()]
                arrays['reference/effective_gripper_intent'] = replay['command/effective_gripper_intent'][()]
                arrays['reference/acquisition_phase'] = replay['acquisition/phase'].asstr()[()]
                arrays['reference/acquisition_guard_json'] = replay['acquisition/guard_json'].asstr()[()]
                arrays['reference/acquisition_guard_time_s'] = replay['time_s'][()]
            m['acquisition_execution'] = report['acquisition_execution']
            m[recorded_acquisition_key] = acquisition_metadata
            m.setdefault('derived_fields',{})['reference/reference_index'] = 'Executed index in the immutable complete prepared plan; repeated only by admitted acquisition entry and measured waits'
            m['derived_fields']['reference/reference_time_s'] = 'Prepared robot reference clock before acquisition insertion; original source clock is separately retained in reference/source_time_s'
            m['derived_fields']['reference/acquisition_guard_json'] = 'Read-only measured guard at the END of its control interval, timed by reference/acquisition_guard_time_s; empty outside settle/close. Canonical timestamp is the pre-action boundary.'
            if not timing:
                with np.load(acquisition_metadata['artifact'], allow_pickle=False) as saved:
                    arrays['reference/source_time_s'] = saved['original_source_reference_time_s'][reference_indices]
                m['derived_fields']['reference/source_time_s'] = 'Executed source-reference clock from the hash-bound acquisition originals; repeats preserve every original source row'
        if timing:
            with np.load(timing['source_clock_artifact']) as clocks:
                arrays['reference/source_time_s']=clocks['source_reference_time_s'][reference_indices]
            m['robot_control_retiming']=timing
            m.setdefault('derived_fields',{})['reference/source_time_s']='Monotonic map from measured control clock to the original complete frozen retarget clock; not a new source observation'
        result['state_assessment']=assess_state_observations({'arrays':arrays,'metadata':m})
        result['archive']=str(write_archive(root,job['dataset'],job['id'],arrays,m))
    else:
        result['partial_state_artifact']=recorded.get('path')
    atomic(workspace/'experiment-result.json',result)
    return result
