"""Derive the can-710-resume-v1 development batch.

The 710 FF0 rollout kept a clean source/traverse grasp but its carried Can met
the native bin partition during lowering. A measured-attachment CAD screen
derived a 9 mm +X placement shift, whose lower knot then failed IK at the
existing 2 mm / 20 mrad tolerance near the wrist range. This batch keeps the
original source, object, contact model, tolerances and gates, and varies only
the release XY target around that derived shift and the transported palm yaw
of the placement. Carried-object clearance stays enabled.
"""
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LABEL = 'can-710-resume-v1'


def main():
    parent = json.loads((REPO/'docs/710-carried-xy-admission-v1.json').read_text())['job']
    quarter = {j['source_prefix_velocity_feedforward']: j for j in
               json.loads((REPO/'configs/can-710-quarter-physics-v1.json').read_text())['jobs']}[0.0]
    x0, y0 = quarter['supported_placement']['placement_xy_world']
    yaw0 = parent['supported_placement']['rotation_xyz_deg'][2]
    jobs = []
    for dx in (6, 9, 12):
        for dy in (-6, -3, 0, 3):
            for dyaw in (-10, 0, 10):
                job = copy.deepcopy(parent)
                job.pop('admission_only', None)
                job['id'] = f'{LABEL}-dx{dx}-dy{dy}-yaw{dyaw}'.replace('-dy-', '-dym').replace('-yaw-', '-yawm')
                job['workspace'] = 'workspaces/'+job['id']
                sp = job['supported_placement']
                sp['placement_xy_world'] = [round(x0+dx/1000, 6), round(y0+dy/1000, 6)]
                sp['rotation_xyz_deg'] = [sp['rotation_xyz_deg'][0], sp['rotation_xyz_deg'][1], yaw0+dyaw]
                job['comparison_parent_job'] = quarter['id']
                job['comparison_scope'] = ('Bounded release XY (mm) and placement palm-yaw (deg) grid around the '
                    'CAD/attachment-derived 9 mm +X shift whose lower knot failed IK. Original source, object, '
                    'contact model, IK tolerances and physical gates unchanged. Adaptive development.')
                jobs.append(job)
    out = REPO/'configs'/f'{LABEL}.json'
    out.write_text(json.dumps(dict(label=LABEL, jobs=jobs), indent=2)+'\n')
    print(out, len(jobs))


if __name__ == '__main__':
    main()
