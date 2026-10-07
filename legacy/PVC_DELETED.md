# Deleted cluster PVC artifacts (2026-10-07)

PVC reachy-retarget-data (/mnt/reachy-retarget, CephFS) legacy results deleted at the user's request when the project restarted as v2.
Kept: workers/ (heartbeats), runtime/ + runtime.json (old worker bootstrap), queue/ directory. New outputs live under /mnt/reachy-retarget/v2/.

## Top-level and first-level entries of at least 1 GB (du -h -d 1)

```
258M	data
12G	datasets
2.0G datasets/bigym
1.5G datasets/maniskill
1.4G datasets/mimicgen
6.4G datasets/robomimic
42M	diagnostics
5.9M	experiments
162M	imports
16M	materialized-geometry-parents
667M	materialized-parents
1.8G	native
1.7G native/parahome
37G	node-local-exports
2.0G node-local-exports/can-710-resume-v1
1.9G node-local-exports/can-finite-pad-matched-v2
1.1G node-local-exports/can-mobile-midpoint-physics-v2
1.6G node-local-exports/can-mobile-source-speed-v1
1.2G node-local-exports/can-prefix-base-tracking-v1
5.1G node-local-exports/can-resume-v1
1.2G node-local-exports/can-retimed-source-physics-v1
2.2G node-local-exports/fixture-facet-recovery-v1
4.1G node-local-exports/local-feasible-recovery-v1
9.5G	operator-backups
9.5G operator-backups/local-spool-pool
3.1M	provenance
1.5G	queue
1.5G queue/tasks
31M	references
7.2G	releases
1.0K	runtime-verification
2.1G	shared-spool
1.1G shared-spool/can-placement-grid-v2
27G	workspaces
```
