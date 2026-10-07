"""Write batch recovery evidence and extend the native Can source identity index.

Every retained attempt in the given batches is listed, including rejections
and failures. A source is added to the identity index only from attempts whose
original-model physics, exact actuator replay and independent validation all
passed; repeated passing variants of one native demonstration count once.
"""
import argparse
import collections
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
METRICS = ('max_grasp_drift_m', 'max_grasp_drift_deg', 'bilateral_carry_fraction', 'final_stable_s',
           'max_hand_object_penetration_m', 'max_object_environment_penetration_m', 'peak_base_velocity',
           'min_joint_margin_rad', 'supported_release_time_s')


def attempts(batch):
    manifest = json.loads((batch/'manifest.json').read_text())
    for job in manifest['jobs']:
        folder = batch/job['id']
        backup = folder/'backup.json'
        if not backup.exists():
            yield dict(job=job['id'], batch=str(batch), status='not_completed', parameters=job); continue
        receipt = json.loads(backup.read_text())
        result = receipt['result'] or {}
        report = result.get('physics_report') or {}
        audit = result.get('actuator_audit') or {}
        publication = folder/'publication.json'
        published = json.loads(publication.read_text()) if publication.exists() else None
        readback = folder/'common-files-readback.json'
        yield dict(job=job['id'], batch=str(batch), status=result.get('status') or receipt['manifest']['status'],
            source_episode=job['source'].split('/')[-1].split('.')[0],
            source_id=report.get('source_id'), source_sequence=report.get('source_sequence'),
            source_hdf5_sha256=report.get('source_hdf5_sha256'),
            evaluation_split=job.get('evaluation_split'), physics_validated=bool(result.get('physics_validated')),
            all_physics_gates=bool(report.get('gates')) and all(report['gates'].values()),
            failure_reasons=report.get('failure_reasons'), reason=None if report else result.get('reason'),
            metrics={k: report.get(k) for k in METRICS if k in report},
            actuator_replay_pass=audit.get('actuator_replay_pass'),
            independent_validation_pass=audit.get('independent_validation_pass'),
            object_welds_actuators=audit.get('object_welds_actuators'),
            object_state_assignments=audit.get('object_state_assignments'),
            varied=dict(retiming_settings=job.get('retiming_settings'),
                        source_prefix_velocity_feedforward=job.get('source_prefix_velocity_feedforward'),
                        placement_xy_world=(job.get('supported_placement') or {}).get('placement_xy_world'),
                        rotation_xyz_deg=(job.get('supported_placement') or {}).get('rotation_xyz_deg'),
                        mobile_base_axis_speed_m_s=(job.get('supported_placement') or {}).get('mobile_base_axis_speed_m_s')),
            comparison_parent_job=job.get('comparison_parent_job'),
            operator_archive=dict(path=receipt['archive'], sha256=receipt['sha256'], bytes=receipt['bytes'],
                                  storage=receipt.get('archive_storage', 'operator_disk')),
            published=dict(path=published['path'], archive_sha256=published['published_archive_sha256'],
                           episode=published.get('episode'),
                           cross_pod_readback_verified=readback.exists()) if published else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batches', type=Path, nargs='+')
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--identity-in', type=Path, required=True)
    parser.add_argument('--identity-out', type=Path, required=True)
    parser.add_argument('--scope', required=True)
    args = parser.parse_args()
    rows = [row for batch in args.batches for row in attempts(batch)]
    passing = [r for r in rows if r['physics_validated'] and r['all_physics_gates']
               and r['actuator_replay_pass'] and r['independent_validation_pass']
               and r['object_welds_actuators'] is False]
    by_source = collections.defaultdict(list)
    for row in passing:
        by_source[row['source_episode']].append(row)
    identity = json.loads(args.identity_in.read_text())
    known = {tuple(map(tuple, s['native_identity'])) for s in identity['sources']}
    known |= {((m['metadata']['source_id'], m['metadata']['source_sequence']),)
              for m in identity.get('manual_canonical_archives', [])}
    known_episodes = {s['source_episode'] for s in identity['sources']}
    added = []
    for source, items in sorted(by_source.items()):
        native = ((items[0]['source_id'], items[0]['source_sequence']),)
        if native in known or source in known_episodes:
            continue
        identity['sources'].append(dict(source_episode=source, native_identity=[list(native[0])], evidence=[
            dict(source_episode=source, source_id=r['source_id'], source_sequence=r['source_sequence'],
                 source_hdf5_sha256=r['source_hdf5_sha256'], evaluation_split=r['evaluation_split'],
                 archive=(r['published'] or {}).get('episode', {}).get('path'),
                 evidence=str(Path(r['batch'])/r['job']/'backup.json'),
                 job=r['job'], all_physics_gates=True, actuator_replay_pass=True,
                 independent_validation_pass=True) for r in items]))
        known.add(native)
        added.append(source)
    previous = identity['total_unique_native_demonstrations']
    identity['schema'] = 'can-original-success-source-identity-v5'
    identity['root_and_legacy_sources'] = len(identity['sources'])
    identity['source_id_sequence_groups'] = len({tuple(map(tuple, s['native_identity'])) for s in identity['sources']})
    identity['total_unique_native_demonstrations'] = previous+len(added)
    identity['previous_version'] = str(args.identity_in)
    identity['added_sources'] = added
    args.identity_out.write_text(json.dumps(identity, indent=1)+'\n')
    counts = collections.Counter(r['status'] for r in rows)
    evidence = dict(schema='can-recovery-batch-evidence-v1', batches=[str(b) for b in args.batches],
        scope=args.scope, attempts=len(rows), status_counts=dict(counts),
        passing_attempts=len(passing), new_unique_sources=added,
        passing_sources={s: len(v) for s, v in by_source.items()},
        identity_index=str(args.identity_out), total_unique_native_demonstrations=identity['total_unique_native_demonstrations'],
        attempts_detail=rows)
    args.evidence.write_text(json.dumps(evidence, indent=1)+'\n')
    print(json.dumps({k: evidence[k] for k in ('attempts', 'status_counts', 'passing_attempts', 'new_unique_sources',
                                               'passing_sources', 'total_unique_native_demonstrations')}, indent=1))


if __name__ == '__main__':
    main()
