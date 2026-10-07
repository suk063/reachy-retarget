# Shared storage interruption during feasible-trajectory experiments

## Observed evidence

At approximately 10:43 UTC on 2026-10-07, new queue JSON files became unreadable and persistent consumers restarted. An audit inspected 3,107 public queue records across 1,560 task directories and found 11 records whose complete contents were NUL bytes: nine job descriptions and two terminal records. Their original bytes and checksums are retained under `/mnt/reachy-retarget/diagnostics/queue-corruption-recovery-v1`. The local canonical job descriptions were restored; affected tasks are explicitly quarantined as `infrastructure_corruption`, requiring fresh immutable retries. Corrupt historical attempt records also remain visible in evidence collection.

Later, new file creation and some lock opens returned `EACCES`; worker logs also reported `ENOSPC`. The owning UID/GID and directory permissions remained 0/0 and 0755, with write access reported by `os.access`. The PVC quota is 1 TiB, its reported available capacity was 1,059,783,180,288 bytes, and `ceph.quota.max_files` is zero (no configured file-count quota). This does not establish backing-pool health. The storage class points to `rook` / `nautilusfs` / `nautilusfs-data0`; the current Kubernetes identity cannot read the Ceph cluster health resource.

Ceph documents that subvolume `df` figures can report the subvolume quota rather than overall filesystem capacity: [quota reporting](https://docs.ceph.com/en/latest/cephfs/quota/). It also documents that a full backing store can discard buffered data after a successful close, while `fsync` detects persistence failures: [full filesystem handling](https://docs.ceph.com/en/quincy/cephfs/full/). This is consistent with the observations; the exact backing-store fault remains unconfirmed without administrative health information. No global storage settings, permissions, quotas, or source data were modified.

## Changes and recovery state

* Queue/result JSON and runtime activation now flush and fsync temporary files before publication.
* Staged input copies are flushed and checksum-compared against their preserved parents.
* Workers retry malformed reads briefly, then defer only the affected task. Filesystem access failures and heartbeat failures no longer terminate the whole consumer.
* A flushed heartbeat is required before claiming new work. Long queue scans report progress so readiness does not misclassify a healthy scan as a dead consumer.
* Evidence collection preserves unreadable-artifact diagnostics. When interrupted claim end times are unknown, reported peak overlap is explicitly an upper bound.
* One already stopped worker pod was replaced to restore maintenance access; the existing deployment and PVC were retained. All other worker pods were retained. Ephemeral debugging containers were unavailable under the current RBAC policy.

The interrupted prefix-controller comparison has five physical outcomes (three passing regression trials and two physical failures) and 19 infrastructure failures. The stationary-base bank has 156 genuine geometric rejections, 25 infrastructure failures, 11 quarantined corrupt-record tasks, and 24 unpublished/pending tasks at the recovery snapshot. Fresh retry manifests contain 19 controller jobs and 60 geometric jobs. These retries do not add independent source demonstrations.

New shared-storage submissions are withheld until durable writes work. The node-local scratch execution path has completed its first full Can IK, physics, independent actuator replay and common-archive run using checksum-verified read-only shared inputs and the identical v23 simulation. Its 107 MB artifact was fully backed up and every regular-file checksum verified. All 30 imported source modules match the pinned release and no guarded shared-write attempt occurred. The physical outcome remains a failure (10.084 mm translation drift), not an infrastructure failure. BiGym separately completed all 2,256 IK frames on node-local disk; physical admission is still under review.

`cluster/local_spool_pool.py` now distributes a declared 92-job recovery batch across 48 existing pods, while two pods remain assigned to BiGym and grasp-position investigation. It preserves detached local execution across control-connection interruptions and copies complete successes, failures and rejections back to operator disk. Every regular file is hash-verified without extracting external asset symlinks. Both the node-local and backup disks keep a 50 decimal GB free-space reserve. These outputs have not been exported to the requested PVC path; that final publication remains pending durable storage recovery. A subsequent create/fsync probe still returned EACCES even though all 50 worker containers were Ready. Pod readiness is therefore not used as evidence of shared-storage recovery.

## Verified recovery and final publication

At approximately 11:24 UTC, independent create/fsync/readback probes succeeded
on two pods. This establishes recovery of the tested operations, not a diagnosis
of the backing-store incident. Immutable v25 publication then verified and
fsynced every captured source and controller file before activation.

All five node-local batches have now been durably exported: 356 attempts,
including 35 complete physical state archives, geometric rejections and six
preserved launcher failures. Those six floating-point endpoint failures were
retried in fresh workspaces after correction; all six then correctly rejected
geometry. They are not additional demonstrations. Every regular artifact file
was verified, and published archives and HDF5 files were independently read and
hashed from another pod. See [the publication audit](node-local-publication-audit-v1.json).
The complete cross-pod receipts are also preserved on the PVC at
`/mnt/reachy-retarget/diagnostics/node-local-publication-audit-v1`.

Archives reside in `/mnt/reachy-retarget/node-local-exports/<batch>/<job>`;
complete state-only HDF5 records are indexed under
`/mnt/reachy-retarget/datasets/<dataset>/episodes`. No recorded arrays or original
path provenance were rewritten. Operator backups remain retained. Repacking
the finished node-local spool directly to the shared volume removed an observed
approximately 1 MB/s operator-uplink bottleneck without rerunning simulation.
