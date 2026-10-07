# Persistent cluster retargeting and state-only records

## Executed evidence: 2026-10-07

[`cluster-run-evidence.json`](cluster-run-evidence.json) records 50 ready workers
on 50 distinct hosts, 1 TiB shared storage, and 40 current source selections
across ten dataset families in the initial acquisition batch. A later snapshot
has 124 immutable archives including retries, derived IK candidates, source FK
and recorded rollouts. They represent 46 recorded source groups, not 124
independent demonstrations. All archives omit RGB. The evidence timestamp is
authoritative; additional source adapters can publish after that snapshot.
The evidence's `current_source_tasks` follows explicit replacement links so
historical all-scene coverage metadata and failed infrastructure attempts are
not mistaken for the current object scope. No source file or failed attempt was
deleted to produce this view.

The initial source publication outcomes are 29 legacy archives, five native
records with reconstruction prerequisites, and six executed MoMaGen IK failures.
The later `legacy-all-v1` batch **actually reran IK for all 29 legacy sources**:
robomimic 15, MimicGen 6, ManiSkill 5 and HUMOTO 3. It ran on 29 different hosts,
with a peak of 23 concurrent claimed attempts (including runtime startup, not a
CPU-utilization measurement). Twenty executed MuJoCo validation and all failed
at least one physical gate; nine lack a verified task-specific physics adapter.
Fifteen full common state archives were published. Five early ManiSkill failures
have one measured row and a terminal state preserved in `state-attempt.hdf5`;
the sequence writer rejected these single-row publications, retaining the
original failed attempts. See [distribution evidence](cluster-distribution-evidence.json).

Three full ParaHome recordings and the five source-pose IK pilots (RoboCasa,
BiGym, and one segment from each D4RL kitchen variant) also completed as
retained kinematic failures. D4RL uses explicitly derived nominal timing and
noisy source configuration; its high fixed mount is not moved to make IK pass.
BiGym retains native source failure and a proven left-hand reach-bound violation.
These diagnostics do not establish successful retargeting.

An idle worker takes the next eligible episode from the shared queue. Fifty
ready workers does not mean fifty runnable tasks or fifty concurrent simulations.
Native reconstruction, shared transfer serialization and agent review can limit
the ready backlog. Historical catalog flags are not treated as current files;
[`current-coverage.json`](datasets/current-coverage.json) audits all 333 catalog
profiles against existing stores and queue evidence, including absent sources.

The MoMaGen references preserve their original clock, planar base path, task
object trajectories and constant tool rotations. None stays within both the
2 cm / 0.15 rad whole-episode thresholds; maximum position errors range from
0.444 to 1.460 m. These results expose the limits of this conservative baseline;
they do not establish successful mobile retargeting. Each attempt retains the
full raw IK arrays, their checksum and the rejected common-format candidate.

The complete-state can recording is
`datasets/robomimic/episodes/full-state-can-v3.hdf5`: 761 rows at measured 100 Hz,
30 scalar joints, 22 actuator commands, every 500 Hz physics substep, and one
terminal boundary. Its task-object channels contain only the can and the two
task bins. The [independent recording audit](full-state-recording-audit.json)
passes 40 numerical/scope checks and 12 metadata checks. Recorded actuator replay
passes; the physical task fails six gates. Recording completeness is not physical
success. V2 and V3 remain two attempts of the same source demonstration.

## Deployed pool

The deployed pool uses Kubernetes context `nautilus`, namespace `erl-ucsd`,
Deployment `reachy-retarget-worker`, 50 replicas, and required hostname
anti-affinity. It uses 50 distinct existing cluster hosts, not fabricated Node
objects. Each worker requests 10 mCPU / 128 MiB while idle, with limits of
2 CPU / 4 GiB. No GPU is reserved by this idle pool.

PVC `reachy-retarget-data` is 1 TiB, ReadWriteMany, `rook-cephfs`, mounted by
all workers at `/mnt/reachy-retarget`. The PVC and task records survive worker
replacement. Dataset transfers stop before free space falls below 50 decimal GB.
A shared filesystem lock serializes transfers; computation remains parallel.

**Lifetime constraint:** Nautilus deletes ordinary Deployments after two weeks.
This pool reuses workers for all tasks during that period. It does not bypass
cleanup by using infinite Jobs or recreating workloads automatically. See the
[official long-idle policy](https://nrp.ai/documentation/userdocs/running/long-idle/).

## Files and execution

```text
/mnt/reachy-retarget/
  runtime.json                      # selected immutable source/runtime reference
  runtime/                          # one verified Python bundle and local-cache launcher
  releases/<source-hash>/            # exact source snapshot, including uncommitted main changes
  references/reachy-control/<hash>/  # copied offline control/model utilities; no SDK
  references/reachy-agent/<commit>/  # committed schema/controller/simulation reference source
  provenance/                       # file hashes, dependency lock, runtime build log
  data/raw/<dataset>/                # pinned source payloads and acquisition receipts
  datasets/<dataset>/episodes/       # common immutable HDF5 + hash-bound JSON sidecars
  queue/tasks/<task-id>/
    job.json                        # immutable explicit acquisition/processing request
    agent/request.json              # source/model evidence and bounded proposed operation
    agent/response.json             # agent decision acknowledging exact evidence hashes
    attempts/<attempt-id>/          # runtime snapshot, log, accepted/rejected response, result
    result.json                     # latest task state; all older attempts remain
  workers/<pod-name>.json            # persistent heartbeat and current task
  workspaces/<task-id>/              # isolated scene, inputs, rollout and validation artifacts
```

`cluster/worker_service.py` stays running. It claims work using filesystem locks
and launches the existing local interpreter; it does not create/delete Kubernetes
Jobs. The executor has its own lock so a surviving child cannot overlap a new
worker. A crashed process releases its lock. Terminal failures require an
explicit new task ID; retries retain their parent and original source ancestry.

Queue publication stages new jobs under `queue/staging`, outside the worker's
scan, then atomically renames a complete job into `queue/tasks`. This fixes the
observed race where a worker found a hidden temporary directory immediately
before its rename and restarted with `FileNotFoundError`. The producer fix is
active without recreating pods; the ConfigMap also filters hidden directories
and tolerates vanished scan entries for subsequent worker starts.

The Python environment is built on node-local disk once, archived to the PVC,
checksum-verified, and unpacked locally by each worker. This avoids package
installation and heavy metadata traffic on CephFS for every task.

Source adapters run offline. Transfers occur only in explicit queued fetch
operations. Payload length and publisher SHA256 or Git blob hash are checked;
archive ranges must return the exact HTTP Content-Range and decoded checksum.

## Agent loop

The protocol follows the durable mailbox design in pinned
`reachy-agent/simulation/agent_loop.py`. A worker writes source inspection,
normalization, hashes, missing fields and bounded options, then waits. The agent
reviews these artifacts and supplies a response. Stale IDs, changed evidence,
unknown actions and empty observations are rejected. There is no automatic
success label or arbitrary agent-provided shell command.

These are **50 compute workers**, not 50 separately funded LLM services. This
session's agent supplies actual reviews; the persistent mailbox also supports a
separately configured agent service later. Work waiting for an agent stays saved.

Stages are acquisition, native inspection, normalization, agent review,
dataset-specific retargeting/reconstruction, actuator-only physical validation,
and common-format export. Missing prerequisites stop the relevant stage without
pretending an inspected or kinematic sequence passed physics.

| Dataset | Implemented native path | Remaining boundary |
|---|---|---|
| [BEHAVIOR](datasets/behavior-momagen-integration.md) | Pinned raw HDF5, task object identities, T+1 states/T actions, source replay callback | Compatible RTX/OmniGibson replay and task asset permissions |
| [MoMaGen](datasets/behavior-momagen-integration.md) | Six actual enriched source references; both hands, planar base reference, task object poses and gripper intent; bounded Reachy IK | Faithful permitted object geometry, task semantics, grasp dynamics |
| [M3Bench](datasets/m3bench-mello-integration.md) | Four exact ZIP member ranges, untimed named joint plan, independently checked URDF FK | Reviewed timing/gripper plan and continuous task-object physics |
| [MeLLO](datasets/m3bench-mello-integration.md) | Actual Vicon markers, masks, measured rate, derived rigid task-object marker frame | Mesh-frame and human-hand calibration, load feasibility |
| [RoboCasa](datasets/robocasa-replay.md) | 232 verified FK rows, exact clock, source robot/base/TCP and task fixtures, all 12 source action channels preserved | Asset-faithful source dynamics, Reachy tool mapping and physical validation |
| [BiGym](datasets/bigym-replay.md) | 2,253 native replay rows at actual 500 Hz; source joints/base/hands/cameras and four task objects; exact parity with uninstrumented replay | Source task fails; recorded/public source-version gap remains; Reachy contact validation pending |
| [ParaHome](datasets/parahome-integration.md) | Three full recordings, 12,995 frames, task-only objects; three fresh IK attempts retained | All three baseline IK candidates fail; source physical parameters and task contracts missing |
| [D4RL kitchen](datasets/d4rl-kitchen-integration.md) | Three pinned raw variants and official assets; untimed observed configurations plus explicit nominal-time FK references | Source configuration is noisy; exact simulator state and measured timestamps unavailable |
| Existing robomimic, MimicGen, ManiSkill, HUMOTO | 29 fresh IK attempts; 20 actual measured dynamics attempts; common archives and early failure artifacts retained | This run has zero all-gate physics passes; nine source tasks need physics adapters |

Full source files, alternate groups and failed attempts remain preserved. Native
and processed views of one source recording share ancestry and do not increase
the independent demonstration count.

## Recording requested control information

The reference is the remote `reachy-agent/main` commit pinned in
[`reachy-agent-main-lock.json`](../configs/reachy-agent-main-lock.json). The shared
archive uses a **state-only** profile: no RGB images. It does not pretend to be
compatible with reachy-agent's RGB policy loader.

New complete-state rollouts explicitly enable all three neck joints. Legacy
frozen-neck scenes remain distinguishable and are not marked complete.
`state_observations.py` records:

- Ordered scalar joint positions/velocities for both seven-joint arms, neck,
  grippers, base and selected task articulations; explicit group/address/unit maps.
- Both TCPs and head poses in world and moving-base frames, their world/base-axis
  and relative-base twists, base world pose, local planar velocity and full twist.
- Explicitly selected task-object root/part poses and articulated qpos/qvel.
  Direct body/object streams exclude unrelated scene props.
- Robot camera optical poses, full MuJoCo replay state, ordered actuators and
  their current controls/forces/velocities. Full replay state remains a compact
  simulator snapshot rather than expanding every background object's pose.

Body-origin velocities use `mjOBJ_XBODY`; `mjOBJ_BODY` would return the inertial
COM frame and is incorrect for these saved poses. TCP velocity uses `mjOBJ_SITE`.

`state_recording.py` adds synchronized pre-integration observations, actual
position-servo commands, controller/feedback memory, every physics-substep's
control input, exact interval duration and the terminal measured boundary.
It stores failures as well as successful attempts. Object state is never written
while validating dynamics. Recorded command modes are declared explicitly; a
position servo does not imply that a Cartesian or velocity command was issued.

[`control-views.md`](control-views.md) describes absolute/delta pose and joint
views, true measured velocities, and command conversions. Observed next states
are never relabeled as issued commands. SE(3) composition, quaternion order,
moving frames and irregular time intervals are explicit.

## Reuse and inspect

```sh
kubectl -n erl-ucsd get deployment reachy-retarget-worker
kubectl -n erl-ucsd get pods -l app.kubernetes.io/name=reachy-retarget-worker -o wide
kubectl -n erl-ucsd exec -it <worker-pod> -- bash
```

Inside a worker, `/mnt/reachy-retarget/runtime/run-python` is the cached Python
launcher. Read `runtime.json` to add the selected source directory to PYTHONPATH,
then run `python -m reachy_retarget.cluster_pipeline status`. Queue new immutable
JSON jobs through `cluster_pipeline enqueue --job FILE`; reuse the same pool.

`cluster/publish.py` uploads a new immutable source snapshot. It can activate it
without rebuilding pods. `cluster/seed.py` explicitly queues bounded source
pilots and existing normalized inputs. `cluster/seed_state_rollout.py` queues the
complete-state can validation using the original frozen trajectory and minimal
pad translation. All commands preserve source repositories and prior attempts.

`cluster/bootstrap_runtime.py --pod POD` verifies and reuses an existing runtime;
it only builds a missing bundle. The tested Linux dependencies are pinned in
`cluster/runtime-requirements.txt`. `cluster/collect_evidence.py --pod POD`
reads actual pool, queue and archive metadata into a local evidence report.
`normalize_mobile` jobs reuse already acquired RoboCasa/BiGym pilots without
discovering or downloading again, and still require an evidence-bound agent
decision before reporting their remaining reconstruction prerequisites.
