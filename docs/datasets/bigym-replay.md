# BiGym native numeric replay

`reachy_retarget.native_replay.bigym` consumes the already acquired `MovePlate`
lightweight pilot and explicitly installed native source. It does not download
anything, invoke `DemoStore`, import a robot SDK, or request RGB rendering.
Its output is reconstructed **source H1 state**, not a Reachy trajectory or a
Reachy dynamics success.

## Verified source contract

The acquired recording UUID is `09431449a6cf4931b5b85fa5b1c7426a`. It contains
2,253 actions, seed `783198766`, and package versions MuJoCo `3.1.5` / BiGym
`4.0.0`. Its SHA-256 is
`f6dccbaafcc6b5b648a2ced99ba5bed644c285084775c9316bb89a91dbfa4d18`.
The original raw member and acquisition ledger remain unchanged.

The native replay recipe uses official historical revision
[`52070fa73e8dd88a3d76eed4be575a6d01497931`](https://github.com/NeuracoreAI/bigym/tree/52070fa73e8dd88a3d76eed4be575a6d01497931),
whose setup pins MuJoCo `3.1.5`, dm-control `1.0.19`, NumPy `1.26.*`, the
Gymnasium `0.29.2` fork, and Mojo `0.1.1`. This source identifies itself as
BiGym `4.1.0`; even the first public commit reports `4.1.0`. The recording's
exact `4.0.0` source revision is unavailable in the inspected public history.
The mismatch is retained as a missing field, regardless of replay outcome.
The earlier discovery pin, `14beb30318ad14c5d6723175c2ee2281129792af`, remains
acquisition provenance; it is not silently substituted for the historical
runtime source.

[Official DemoPlayer](https://github.com/NeuracoreAI/bigym/blob/52070fa73e8dd88a3d76eed4be575a6d01497931/demonstrations/demo_player.py)
defaults native demonstrations to `CONTROL_FREQUENCY_MAX`, which the
[environment](https://github.com/NeuracoreAI/bigym/blob/52070fa73e8dd88a3d76eed4be575a6d01497931/bigym/bigym_env.py)
defines as **500 Hz** with a 0.002-second physics step. The 50 Hz README example
requests decimation; it is not the untouched lightweight recording frequency.
This replay consumes every action once, in order, with no decimation, clipping,
interpolation, artificial slowdown, or time inferred from waypoint count.
Stored timestamps are actual `mjData.time` values produced by native replay.
They are reconstructed simulation time, not timestamps found in the original
lightweight tensor.

The [action implementation](https://github.com/NeuracoreAI/bigym/blob/52070fa73e8dd88a3d76eed4be575a6d01497931/bigym/action_modes.py)
and [floating base](https://github.com/NeuracoreAI/bigym/blob/52070fa73e8dd88a3d76eed4be575a6d01497931/bigym/robots/floating_base.py)
define a mixed 15-component action:

| Components | Meaning |
| --- | --- |
| 0–2 | Pelvis x/y/yaw position-target **increments**, relative to previous actuator targets |
| 3–12 | Ten absolute arm joint position targets, in compiled source actuator order |
| 13–14 | Left/right normalized gripper commands mapped by the native gripper controller |

The compiled names, order, ranges, and types are saved. Calling the complete
vector "absolute joint positions" would mislabel the base and grippers.
The recorded reset-position zeros are retained as original metadata. Official
replay uses `env.reset(seed=...)` and the source robot reset configuration;
it does not install the zero placeholder as a measured joint state.

The loop follows official fast stepping and reward/success evaluation. It does
not additionally call historical `env.fail`: upstream
[fix #49](https://github.com/NeuracoreAI/bigym/commit/8a7a83ce4bf9ec5c41b43eb3e65740a6d6db6eb7)
documents a floating-base fail-check replay divergence. Original termination
and truncation tensors remain separately labeled recorded values.

## Numeric outputs and task scope

Each post-action row contains source qpos/qvel, all scalar robot joints and
velocities, actuator controls/forces, pelvis pose/twist, both true source wrist
site poses/twists and native gripper pinch-site poses, robot body poses, camera optical poses, and an opaque MuJoCo
integration state for replay. Camera pose does not require an image. The H1
model has no articulated neck; a head camera transform does not become a neck
joint or desired head command. Wrist sites are not assumed to be Reachy pad
contact frames. Native floating-base and animated-leg behavior remain explicit
simulation assumptions, not mobile locomotion validation.

Only the task's **plate, starting rack, target rack, and supporting table** get
named object pose/twist channels. Other scene contents can remain in the
original compiled model and full physics-state vector for reproducibility;
they do not become named object features. Source object's initial placement is
performed by the official reset. The replay helper never writes, welds,
teleports, or rescales a moving object during stepping.

`replay(pilot_root, output_dir, source_root, max_steps=None)` creates a new,
immutable attempt directory with:

- `states.npz`: sampled states and original consumed actions;
- `reset-state.npz`: separate initial state, not a fabricated action row;
- `source-model.mjb` and `source-model.xml`: compiled model and source MJCF;
- `source-provenance.json`: acquisition records and all source-file hashes;
- `metadata.json`: package versions, schema, source semantics, results, errors,
  explicit missing fields, and output checksums.

Failed attempts and successfully sampled prefixes are saved. `normalize(path)`
verifies checksums and returns the common `{arrays, metadata, status,
missing_fields}` record. The common archive exporter retains source identity
and aliases task object poses without relabeling H1 state as Reachy state.
The source reward criterion is separate from successful source replay,
retargeting, and Reachy physical validation.

## Cluster recipe

Use the existing persistent worker; no task-specific pod is required. The
isolated workspace is
`/mnt/reachy-retarget/workspaces/bigym-native-20261007-01`. Its acquisition
manifest pins these official archives and verifies their hashes:

| Source | Revision | Compressed bytes |
| --- | --- | ---: |
| BiGym | `52070fa73e8dd88a3d76eed4be575a6d01497931` | 59,220,703 |
| Mojo | `ccec1deaf9bde9fa7a2ac051c9376ad5a81a3aad` | 544,255 |
| Gymnasium fork | `2cbc4d34c3124ef4921977fe4ed1e4e532f33ed3` | 122,169,427 |

Dependencies use a dedicated node-local environment with a persistent packed
runtime, served by the workspace's `run-python` launcher; the shared runtime is
unchanged. A first install directly onto Ceph was interrupted after slow small
file writes; its partial environment and log remain as evidence. The successful
install records are `install-local-command.json`, `pip-install-local-report.json`,
`install-local.log`, and `requirements-frozen.txt`. All transfers reserve at least
50 decimal GB. Source code is Apache-2.0; the original asset attribution file
must also remain because individual assets use CC0, CC BY, and CC BY-NC licenses.

```sh
PYTHONPATH=/path/to/current/reachy-retarget \
  /mnt/reachy-retarget/workspaces/bigym-native-20261007-01/run-python \
  -m reachy_retarget.native_replay.bigym \
  --pilot-root /mnt/reachy-retarget/native/bigym \
  --source-root /mnt/reachy-retarget/workspaces/bigym-native-20261007-01/bigym \
  --output-dir /mnt/reachy-retarget/workspaces/bigym-native-20261007-01/new-attempt
```

Offline tests verify source hashes/configuration, mixed action semantics,
post-step kinematic cache accuracy, unchanged physical state and solver
warm-start under observation, task-only object selection, and durable failed
attempts. Full native replay outcomes must be read from the actual attempt
report; passing those tests alone is not a successful native task.

## Executed pilot result, 2026-10-07

`attempt-01` completed **all 2,253 actions** without a replay exception and
saved 2,253 post-action frames from 0.002 to 4.506 seconds. The full operation,
including model construction, observation extraction, and artifact writing,
took 37.14 seconds. It contains 30 scalar source joints: three base coordinates,
five joints per arm, eight mechanism joints per gripper, and one additional
floating-base mechanism coordinate. Both wrist sites, both native pinch sites,
three camera optical poses, and the four task objects are present. No RGB arrays
or inferred neck channels are present.

**The native task success criterion is false.** The plate remains near the
starting rack; it does not contact the target rack at the end. The base moves
0.4475 m. The original file's 251 termination flags do not establish success,
because termination can also denote source failure.

An independent loop using only official `env.step(action, fast=True)` and
`env.reward` reproduced the same task failure. All 2,253 qpos and qvel rows
matched the instrumented replay exactly: both maximum absolute differences
were **0.0**. The baseline stepping loop took 1.55 seconds. This verifies that
the added observer did not degrade this native trajectory. Original measured
state is absent, so this comparison cannot establish bit-exact reproduction
of the unavailable recorded BiGym 4.0.0 implementation.

The common archive is:
`/mnt/reachy-retarget/datasets/bigym/episodes/bigym-native-replay-v1.hdf5`.
It preserves the native failure and version mismatch and remains
`physics_validated=false`, `retargeted=false`, and `policy_ready=false`.
The state record can support a separate Reachy IK attempt; source success,
Reachy kinematic feasibility, and Reachy physical task success remain separate.

The independent evidence files are `native-baseline.npz`,
`native-baseline-comparison.json`, and `common-archive-inspection.json` in the
workspace. The persistent runtime bundle has SHA-256
`a1daaaf6885f9531e594022ec8e7b4a0cc68fd2b8064b4489faa4f3eb745a4c4`
and size 146,117,978 bytes. Its launcher verifies the checksum before unpacking
on another persistent worker; it does not download or install packages.

## Conservative Reachy IK stage

`reachy_retarget.native_replay.bigym_retarget.retarget(record, root, dataset,
episode_id, attempt)` delegates to the common `pose_retarget` solver. Its input
is `bigym.normalize(native_attempt_path)`, so source H1 observations remain
separate from the derived Reachy candidate. Both native gripper **pinch sites**
are used as position references, not the source wrist frames. The shared solver
applies one initial world XY/yaw gauge and one constant orientation attachment
per hand. It performs no source-height correction, hand-position shift,
trajectory resampling, base optimization, or object-trajectory optimization.
Mapping the native pinch origin to Reachy `arm_tip` remains an explicit
uncalibrated tool-frame assumption; this is not verified gripper-pad contact.

The wrapper verifies the compiled source action names, order, ranges, and
actuator identity before converting gripper commands. The pinned native
`GripperConfig` defaults to discrete mode; native `0` commands actuator `0`
(open), and native `1` commands actuator `255` (closed). This polarity was also
checked against the actual native replay: the left pad-origin distance was
0.097845 m at the final stable open command and 0.014676 m at the final stable
closed command. These are pad-origin distances, not a calibrated aperture or
evidence of holding the plate. The right gripper remains open in this recording.

Derived desired Reachy targets follow the existing repository convention:
`0 -> 2.0 rad` open and `1 -> -0.06 rad` closed. Raw source commands and event
times stay unchanged. Nonbinary, reordered, unknown-version, or partial replay
inputs are rejected instead of receiving a guessed conversion. The targets are
not measured Reachy joints or issued actuator commands. H1 head-camera data
remains source information; no Reachy neck command is inferred.

Before the full solve, `geometric-preflight.npz` and
`geometric-preflight.json` record every frame's required shoulder-to-tool
distance and an upper bound from the pinned Reachy chain's link lengths. The
Reachy shoulder height is 1.166 m, and each shoulder-to-`arm_tip` maximum
geometric reach is 0.66 m. In this pilot, the left target reaches 0.694960 m at
frame 779 (native time 1.56 s), with 456 frames outside that reach sphere. This
proves at least 0.034960 m of position error at the worst frame for the chosen
fixed base path, even before orientation, joint-limit, or collision constraints.
The right target's maximum distance is 0.599784 m; being inside the sphere does
not prove feasibility.

The full 2,253-frame shared-engine IK attempt still runs, bounded to 60 iterations
per frame, and retains its raw configurations, achieved/target poses, errors,
original clock, placement, and attachments. The reach-bound failure is not
hidden by dropping frames or changing objects. The common output remains a
derived candidate or a failed kinematic attempt, with native task failure and
the exact-source-version gap preserved. No physical Reachy success is asserted.
Eight adapter tests verify source polarity and provenance, reject unverified
commands, check geometric lower bounds, and ensure the entire trajectory is
delegated even when preflight proves a tracking failure.
