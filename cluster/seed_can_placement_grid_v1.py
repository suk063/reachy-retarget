"""Derive can-placement-grid-v1 from the retained lower-knot census.

For each unsolved Can source with an admitted source path, the two distinct
placement postures whose lowering progressed furthest before the unchanged
2 mm / 20 mrad knot check failed are re-tried with a bounded release XY grid
(+/-8 mm) and transported palm yaw (+/-12, +/-24 deg). Each job is the source-path
admission job plus the retiming/repair used by its placement search, so an
admitted placement continues directly into the unchanged original-model
physics and replay gates. d755 was recovered by the same class of release-XY
correction, and 710 by a -10 deg palm-yaw change at a
+12 mm X shift. Adaptive development only; selection never uses physical outcomes.
"""
import argparse
import copy
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LABEL = 'can-placement-grid-v1'


def signed(value):
    return ('p' if value > 0 else 'm' if value < 0 else '')+str(abs(value))


def configs():
    jobs = {}
    for path in sorted(glob.glob(str(REPO/'configs/*.json'))):
        try:
            value = json.loads(Path(path).read_text())
        except ValueError:
            continue
        for job in value.get('jobs', []) if isinstance(value, dict) else []:
            jobs.setdefault(job.get('id'), job)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default=LABEL)
    parser.add_argument('--runtime', help='Pin every job to this published runtime source SHA-256')
    parser.add_argument('--skip-launched', type=Path,
                        help='Omit grid points already launched with the same runtime pin in this batch output')
    args = parser.parse_args()
    label = args.label
    launched = set()
    if args.skip_launched:
        for path in args.skip_launched.glob('*/launch.json'):
            job = json.loads(path.read_text())['job']
            if args.runtime is None or job['runtime_source_sha256'] == args.runtime:
                launched.add(job['id'][len(LABEL)+1:])
    census = json.loads((REPO/'docs/can-placement-lower-knot-census-v1.json').read_text())
    jobs_by_id = configs()
    jobs = []
    for source, ranked in sorted(census['ranked_lower_candidates'].items()):
        chosen, seen = [], set()
        for item in sorted(ranked, key=lambda r: -r['weight']):
            key = (item['parent_geometry_job'], tuple(round(x, 3) for x in item['parameters']['rotation_xyz_deg']))
            if key in seen:
                continue
            seen.add(key)
            chosen.append(item)
            if len(chosen) == 2:
                break
        for rank, item in enumerate(chosen):
            parent = jobs_by_id[item['parent_geometry_job']]
            search = jobs_by_id[item['job']]
            x0, y0 = item['parameters']['placement_xy_world']
            yaw0 = item['parameters']['rotation_xyz_deg'][2]
            for dx in (-8, 0, 8):
                for dy in (-8, 0, 8):
                    for dyaw in (-24, -12, 0, 12, 24):
                        if dx == dy == dyaw == 0:
                            continue
                        job = copy.deepcopy(parent)
                        job.pop('admission_only', None)
                        suffix = f'{source[:4]}-c{rank}-x{signed(dx)}-y{signed(dy)}-r{signed(dyaw)}'
                        if suffix in launched:
                            continue
                        job['id'] = f'{label}-{source[:4]}-c{rank}-x{signed(dx)}-y{signed(dy)}-r{signed(dyaw)}'
                        job['workspace'] = 'workspaces/'+job['id']
                        if args.runtime:
                            if job['runtime_source_sha256'] != args.runtime:
                                job['inherited_runtime_source_sha256'] = job['runtime_source_sha256']
                            job['runtime_source_sha256'] = args.runtime
                        job['retime'] = True
                        job['retiming_settings'] = search.get('retiming_settings', {'arm_speed_rad_s': .8})
                        if 'retimed_source_clearance' in search:
                            job['retimed_source_clearance'] = search['retimed_source_clearance']
                        sp = copy.deepcopy(item['parameters'])
                        sp['placement_xy_world'] = [round(x0+dx/1000, 6), round(y0+dy/1000, 6)]
                        sp['rotation_xyz_deg'] = [sp['rotation_xyz_deg'][0], sp['rotation_xyz_deg'][1], yaw0+dyaw]
                        if sp.get('adaptive_midpoints'):
                            sp['midpoint_quarter_probes'] = True
                        job['supported_placement'] = sp
                        job['source_prefix_velocity_feedforward'] = 1.0
                        job['comparison_parent_job'] = item['job']
                        job['comparison_scope'] = ('Placement candidate %d of %s failed lower at weight %.3f; bounded '
                            'release XY / palm-yaw grid. Source, object, contact model, IK tolerances and physical gates '
                            'unchanged. Adaptive development.' % (item['candidate'], item['job'], item['weight']))
                        jobs.append(job)
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate job IDs')
    out = REPO/'configs'/f'{label}.json'
    out.write_text(json.dumps(dict(label=label, jobs=jobs), indent=2)+'\n')
    print(out, len(jobs))


if __name__ == '__main__':
    main()
