"""Search inserted placement from a checksum-bound admitted source path.

The source mobile IK is reused exactly. Sparse screens are rejection heuristics;
only complete subsequent admission can produce a candidate, never a physics pass.
"""
from dataclasses import replace
from pathlib import Path
import json
import math

from .feasible_geometry_search import placement_bank, _screen_placement
from .frozen_plan import validate_cache_admission
from .store import json_write, sha256


def validate_parent(prepared, variant):
    details = prepared[5]
    validate_cache_admission(details, len(prepared[7]))
    if details.get('task') != 'PickPlaceCan':
        raise ValueError('The declared placement bank requires the Can task contract')
    if details.get('experiment_variant') != variant:
        raise ValueError('Cached aperture admission must match the declared variant')
    if any(details.get(key) for key in ('robot_control_retiming', 'robot_supported_placement',
                                       'cylindrical_acquisition', 'box_acquisition', 'planning_fixture_forecast')):
        raise ValueError('Placement search requires an admitted complete original-clock source path')


def chunk_placements(bank, index, count):
    if type(index) is not int or type(count) is not int or not 0 <= index < count <= 45:
        raise ValueError('Placement chunk must be in [0, count), count <= 45')
    return sorted((p for p in bank if p['id'] % count == index), key=lambda p: p['rank'])


def refinement_bank(bank, rounds, *, midpoint_minimum_motion=False, interval_knot_timing=False,
                    midpoint_position_reserve_m=0., midpoint_fixture_reserve_m=0.,
                    midpoint_quarter_probes=False):
    """Bind the optional chord repair to saved parameters without reranking."""
    if type(rounds) is not int or not 0 <= rounds <= 3:
        raise ValueError('Adaptive midpoint rounds must be an integer in [0, 3]')
    if (type(midpoint_minimum_motion) is not bool or type(interval_knot_timing) is not bool
            or type(midpoint_quarter_probes) is not bool
            or (not rounds and (midpoint_minimum_motion or interval_knot_timing or midpoint_quarter_probes))):
        raise ValueError('Refinement options require explicit booleans and nonzero rounds')
    reserves = dict(midpoint_position_reserve_m=midpoint_position_reserve_m,
                    midpoint_fixture_reserve_m=midpoint_fixture_reserve_m)
    for name, value in reserves.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= .001
                or (value and not rounds)):
            raise ValueError(name+' requires a finite reserve in [0, .001] and nonzero rounds')
    if not rounds:
        return bank
    settings = dict(adaptive_midpoints=rounds)
    if midpoint_minimum_motion:
        settings['midpoint_minimum_motion'] = True
    if interval_knot_timing:
        settings['interval_knot_timing'] = True
    if midpoint_quarter_probes:
        settings['midpoint_quarter_probes'] = True
    settings.update((name, value) for name, value in reserves.items() if value)
    return [dict(item, parameters=dict(item['parameters'], **settings))
            for item in bank]


def run(root, job):
    from .cluster_pipeline import under
    from .robot import Robot
    from . import frozen_plan, feasible_timing, supported_placement, geometry_clearance
    root = Path(root)
    workspace = under(root, job['workspace'])
    output = workspace/'placement-search'
    output.mkdir(parents=True, exist_ok=False)
    (output/Path(__file__).name).write_text(Path(__file__).read_text())
    source = under(workspace, job['source'])
    robot = Robot(workspace)
    candidate, prepared = frozen_plan.load(workspace/'frozen-plan', source, robot, output)
    validate_parent(prepared, job['variant'])
    config = job['placement_search']
    index, count = config['chunk_index'], config['chunk_count']
    bank_name = config.get('bank', 'original')
    bank_metadata = None
    if bank_name == 'original':
        proposals = placement_bank(prepared)
    elif bank_name in ('wall_mobile', 'hand_mobile'):
        from . import placement_wall_candidates
        proposals, bank_metadata = placement_wall_candidates.bank(
            prepared, robot, **config.get('wall_settings', {}))
        (output/'placement_wall_candidates.py').write_text(Path(placement_wall_candidates.__file__).read_text())
        if bank_name == 'hand_mobile':
            from . import placement_hand_targets
            proposals, hand_report = placement_hand_targets.propose(
                prepared, robot, proposals, output/'hand-target-screen', **config.get('hand_settings', {}))
            bank_metadata = dict(wall_directions=bank_metadata,
                                 hand_target_screen=str(output/'hand-target-screen/result.json'),
                                 sampled_hand_passes=hand_report['passed_count'],
                                 selected_proposals=hand_report['selected_ids'])
    else:
        raise ValueError('Unknown declared Can placement bank')
    bank = chunk_placements(refinement_bank(proposals, config.get('adaptive_midpoints', 0),
        **config.get('refinement_options', {})), index, count)
    report = dict(policy='admitted-source-can-placement-bank-v1', bank=bank_name,
        bank_metadata=bank_metadata, source_episode=source.stem,
        source_sha256=sha256(source), chunk_index=index, chunk_count=count,
        evaluation_split=job['evaluation_split'], status='no_admitted_candidate',
        search_complete=True, physics_tested=False, physics_validated=False,
        candidates=[], selected=None, parent_binding=prepared[5]['frozen_plan_reuse'],
        scope='Cached original source path; unchanged aperture. Original placement rank before physical outcomes; sampled rejection is not proof of infeasibility')
    def save():
        report['search_complete'] = not any(p['status'] == 'error' for p in report['candidates'])
        if not report['search_complete']:
            report['status'] = 'search_incomplete'
        json_write(output/'result.json', report)
    save()
    timed = feasible_timing.prepare(prepared, output/'retiming',
                                    **job.get('retiming_settings', {'arm_speed_rad_s': .8}))
    if 'retimed_source_clearance' in job:
        from . import retimed_source_clearance
        repair_path=output/'retimed-source-clearance'
        (output/'retimed_source_clearance.py').write_text(Path(retimed_source_clearance.__file__).read_text())
        try:
            timed=retimed_source_clearance.prepare(timed,robot,repair_path,**job['retimed_source_clearance'])
        except ValueError as error:
            path=repair_path/'result.json'
            details=json.loads(path.read_text()) if path.exists() else {}
            bounded_rejection=str(details.get('exception','')).startswith('RepairRejected:')
            report.update(status='no_admitted_candidate' if bounded_rejection else 'search_incomplete',
                search_complete=bounded_rejection, source_repair_rejected=True,
                source_repair_report=str(path), source_repair_reason=str(error),
                candidates=[dict(id=p['id'],rank=p['rank'],status='not_attempted_source_repair_rejected') for p in bank])
            json_write(output/'result.json',report)
            return report
    for number, item in enumerate(bank):
        folder = output/f"placement-{item['id']:02d}"
        entry = dict(id=item['id'], rank=item['rank'], parameters=item['parameters'],
                     status='running', path=str(folder))
        report['candidates'].append(entry)
        stage = 'screen'
        try:
            if bank_name == 'original':
                screened = _screen_placement(timed, robot, item['parameters'], folder/stage)
                if screened.get('exception'):
                    raise RuntimeError(screened['exception'])
                if not screened['passed']:
                    entry['status'] = 'screen_rejected'
                    save()
                    continue
            else:
                entry['screen_scope'] = 'Fixed-base sparse heuristic is inapplicable; use full mobile placement admission'
            stage = 'full'
            ready = supported_placement.prepare(timed, robot, folder/stage, **item['parameters'])
            stage = 'fixture-clearance'
            ready = geometry_clearance.prepare(ready, robot, folder/stage,
                support_pairs=geometry_clearance.reachy_wheel_floor_pairs(ready[0]),
                **job.get('geometry_clearance', {'minimum_m': .003}))
            if 'carried_object_clearance' in job:
                from . import carried_object_clearance
                stage='carried-object-clearance'
                ready[5]['carried_object_clearance']=carried_object_clearance.prepare(
                    ready,folder/stage,**job['carried_object_clearance'])
            control = replace(candidate, contact_release=False, calibrate_attachment=False,
                              arm_velocity_feedforward=0.)
            cache = frozen_plan.save_admitted(ready, control, source, robot, folder/'admitted-plan')
            entry['status'] = 'admitted'
            report.update(status='kinematic_candidate', selected=dict(selection_rank=item['rank'],
                placement_parameters=item['parameters'], admitted_plan=str(cache),
                source_episode=source.stem, physics_validated=False))
            for remaining in bank[number+1:]:
                report['candidates'].append(dict(id=remaining['id'], rank=remaining['rank'],
                    status='not_attempted_higher_declared_rank'))
            save()
            return report
        except ValueError as error:
            path = folder/stage/'result.json'
            details = json.loads(path.read_text()) if path.exists() else {}
            expected_knot_rejection = (details.get('rejection_kind') == 'mobile_knot_admission'
                                       and details.get('admitted') is False
                                       and bool(details.get('failed_knot')))
            entry.update(status='error' if not details or (details.get('exception') and not expected_knot_rejection) else 'rejected',
                         failed_stage=stage, reason=str(error), failure_report=str(path))
        except Exception as error:
            entry.update(status='error', failed_stage=stage, reason=type(error).__name__+': '+str(error))
        save()
    return report
