"""Summarize a local-spool batch from its retained backup receipts."""
import argparse
import collections
import json
from pathlib import Path

METRICS = ('max_grasp_drift_m', 'max_grasp_drift_deg', 'bilateral_carry_fraction', 'final_stable_s',
           'max_object_environment_penetration_m', 'peak_base_velocity')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.batch/'manifest.json').read_text())
    status = json.loads((args.batch/'status.json').read_text()) if (args.batch/'status.json').exists() else {}
    counts = collections.Counter()
    for job in manifest['jobs']:
        path = args.batch/job['id']/'backup.json'
        if not path.exists():
            stage = status.get('states', {}).get(job['id'], {}).get('stage', 'queued')
            counts[stage] += 1
            continue
        result = json.loads(path.read_text())['result'] or {}
        report = result.get('physics_report') or {}
        counts[result.get('status')] += 1
        failed = report.get('failure_reasons')
        detail = {k: (round(report[k], 4) if isinstance(report.get(k), float) else
                      [round(v, 3) for v in report[k][:3]] if isinstance(report.get(k), list) else report.get(k))
                  for k in METRICS if k in report}
        reason = str(result.get('reason') or result.get('error') or '')[:120] if not report else ''
        print(f"{job['id']:<48} {str(result.get('status')):<22} {failed if failed is not None else reason} {detail if detail else ''}")
    print(dict(counts))


if __name__ == '__main__':
    main()
