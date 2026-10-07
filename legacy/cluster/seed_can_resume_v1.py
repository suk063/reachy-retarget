"""Derive the can-resume-v1 development batch from two retained physical failures.

3873 failed only ``actual_base_speed_limits`` (peak base x 0.658 > 0.611 m/s)
under both feedforward settings, so only the source planar speed cap and the
inserted mobile base axis cap are varied. d755 kept its grasp through carry but
touched a fixture about 20 mm above the bin floor during lowering (measured
hand-frame translation jump at t~15 s), the same failure class recovered for a2
and 710 by small placement-X corrections; only the release XY target and the
arm feedforward gain are varied. Sources, objects, contact model, clock and all
physical gates are unchanged. Every job is adaptive development, not a frozen
policy evaluation.
"""
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LABEL = 'can-resume-v1'


def main():
    jobs = []
    for parent in json.loads((REPO/'configs/can-3873-quarter-physics-v1.json').read_text())['jobs']:
        ff = parent['source_prefix_velocity_feedforward']
        for planar in (.35, .30, .25):
            for axis in (.30, .25, .20):
                job = copy.deepcopy(parent)
                job['id'] = f'{LABEL}-3873-planar{int(planar*100)}-axis{int(axis*100)}-ff{int(ff)}'
                job['retiming_settings']['planar_speed_m_s'] = planar
                job['supported_placement']['mobile_base_axis_speed_m_s'] = axis
                job['comparison_parent_job'] = parent['id']
                job['comparison_scope'] = ('Base speed only failure of parent (peak vx 0.658 > 0.611 m/s). Varies only the '
                    'source planar reference cap and the inserted mobile base axis cap; all gates unchanged. Adaptive development.')
                jobs.append(job)
    parents = {p['source_prefix_velocity_feedforward']: p for p in
               json.loads((REPO/'configs/can-d755-mobile-physics-v1.json').read_text())['jobs']}
    parent = parents[1.0]
    x0, y0 = parent['supported_placement']['placement_xy_world']
    for ff in (1.0, .5):
        for dx in (0, 4, 8, 12, 16):
            for dy in (-6, 0, 6):
                if ff == 1.0 and dx == 0 and dy == 0:
                    continue
                job = copy.deepcopy(parent)
                job['id'] = f'{LABEL}-d755-dx{dx}-dy{dy:+d}-ff{int(ff*10):02d}'.replace('+', 'p').replace('-dy-', '-dym')
                job['supported_placement']['placement_xy_world'] = [round(x0+dx/1000, 6), round(y0+dy/1000, 6)]
                job['source_prefix_velocity_feedforward'] = ff
                job['comparison_parent_job'] = parent['id']
                job['comparison_scope'] = ('Parent kept bilateral carry but met a fixture ~20 mm above the bin floor during '
                    'lowering. Bounded release XY grid (mm offsets) and feedforward gain only; original source, object, '
                    'contact model and gates unchanged. Adaptive development.')
                jobs.append(job)
    for job in jobs:
        job['workspace'] = 'workspaces/'+job['id']
    out = REPO/'configs'/f'{LABEL}.json'
    out.write_text(json.dumps(dict(label=LABEL, jobs=jobs), indent=2)+'\n')
    print(out, len(jobs))


if __name__ == '__main__':
    main()
