# BiGym native-scene Reachy diagnostics

`native_replay.bigym_physics` replaces only the native H1 robot with Reachy.
The plate, both racks, table and floor retain the original seeded task geometry.
The moving plate has its original free joint, is initialized once from the first
saved native state, and subsequently moves only through physics. It has no
tracking actuator, mocap target, equality weld or state correction.

The first physical diagnostic used an **unreproduced source task**, not
a verified successful original demonstration. That acquired
lightweight source contains only 2,253 actions, termination and truncation flags,
and an empty reward array. It has no original tool, object or joint-state
observations with which to validate the reconstructed trajectory. The first
original termination flag is at frame 2002; termination alone does not identify
success. Original metadata says BiGym 4.0.0; the earliest available public code
used for replay reports 4.1.0. The replayed plate stays near the starting rack
and the native task predicate is false. These limitations must remain attached
to every derived artifact, even when Reachy IK succeeds.

The original archive member has no resampled frequency directory. The original
[BiGym paper, section 3.3](https://chernyadev.github.io/bigym/static/paper/bigym_paper.pdf)
states that raw demonstrations were captured at 500 Hz, with optional later
downsampling. Both pinned
[DemoPlayer](https://github.com/NeuracoreAI/bigym/blob/52070fa73e8dd88a3d76eed4be575a6d01497931/demonstrations/demo_player.py)
and the environment use a native 500 Hz default. Replaying all rows once takes
4.506 simulated seconds. The exported clock is actual reconstructed simulator
time under that documented collection protocol; the file has no individual
measured timestamp channel. Current upstream's
[floating-base replay fix](https://github.com/NeuracoreAI/bigym/pull/49)
addresses extra `physics.forward()` calls during failure checks. The retained
instrumented and uninstrumented native replays already match exactly, so that
specific known divergence does not explain this pilot's failure. Recovering
recorded 4.0 code/assets or finding another independently reproducible original
episode is preferable to optimizing Reachy around a failed source reconstruction.

Three further independent MovePlate UUIDs were explicitly acquired (967,656
bytes total), with separate provenance manifests and the shared 50 GB reserve.
Two successfully replayed under the same pinned native environment and unchanged
500 Hz protocol. The version gap remains; native task reproduction is now a
positive, independently checked prerequisite for those recordings.

| Original UUID | Action rows | Native replay result | Evaluation role |
| --- | ---: | --- | --- |
| `09431449a6cf4931b5b85fa5b1c7426a` | 2253 | Completed; task false | Retained failure diagnostic |
| `c795f62efb4f4d20ab899af96d766a2f` | 2256 | Completed; task true | Development reference |
| `103d65f03899407ebc37ffed9a859f6d` | 2320 | Completed; task true | Held-out reference; no physical tuning |
| `d47012004888461fa639a9504e1ce902` | 3332 | Rejected at first action | Original shoulder-roll target exceeds public version's lower bound by 0.00627 rad; no clipping |

The c795 source plate moves from `[0.74017,0.45870,1.09997]` to
`[0.65718,-0.21128,1.06591]` m and satisfies the native target-rack predicate.
These are four distinct recordings, not alternative versions of one source.

## Preserving native physical assets

`export_assets(pilot_root, replay, source_root, output)` runs explicitly in the
isolated original MuJoCo 3.1.5 environment. It checks the reconstructed seeded
XML against the retained native XML byte for byte, verifies source provenance,
and saves official dm_control virtual assets with checksums. The current bundle
contains 72 assets, 19,366,467 bytes. Export retains at least 50 decimal GB free.
It does not fetch network content.

`export_inertials` and `export_mesh_geometry` capture original compiled physical
properties from the archived MuJoCo 3.1.5 model. Newer MuJoCo changes the mesh
principal-axis/inertia calculation. The replacement scene therefore installs
the original explicit mass, principal inertia, inertial position and quaternion
for every task body. It compares physical mesh vertices in their owning body
frames, rather than comparing compiler-dependent mesh bounding boxes.

The actual scene compiled successfully with 37 position coordinates, 36 velocity
coordinates and 22 robot actuators. Original mass/inertia and material/contact
parameters matched; 60 original task mesh vertex sets matched within
`8.1e-8` m. The original H1 body and H1-only controls, equalities, tendons and
collision exclusions are removed. Remaining H1 assets are unused resources,
not physical bodies. One common planar gauge is applied to all native bodies.
Reachy is the pinned collision-mesh robot with enabled neck state and separate
base, arm, neck and gripper position servos. Wheel traction is not simulated.

## Admission, execution and recording

`rollout(root, replay, native_assets, ik_attempt, episode_id, output,
penetration_limit_m=0.002)` requires a complete passing geometric candidate.
It checks source-clock identity and original 2 ms intervals. It then checks real
initial collisions using the full native scene before any integration. The four
initial BiGym jaw-symmetry candidates passed; their only initial robot contacts
were wheels touching the floor at numerical zero depth. Every admission and
rejection is retained separately.

The rollout initializes source plate pose and velocity once, initializes the
derived robot state, and writes only robot actuator targets thereafter. Native
gripper binary commands use the verified source-to-Reachy conversion. All
2,252 intervals between the 2,253 source state boundaries are preserved. Numeric
robot state includes both arms, both TCPs, neck/head, base, actual actuator
controls, full replay state and task-related object subtrees. No images are
generated. `terminal/*` retains the last integrated boundary separately from
the 2,252 pre-action command rows; no terminal command is invented.

The source
[MovePlate predicate](https://github.com/NeuracoreAI/bigym/blob/52070fa73e8dd88a3d76eed4be575a6d01497931/bigym/envs/move_plates.py)
requires plate position within 5 cm of a target rack slot, orientation within
20 degrees of its prescribed upright direction, target-rack contact, no table
contact, and no gripper-pad contact. The prescribed world direction follows the
same gauge transform. The Reachy adaptation uses its actual distal collision
bodies for the pad-release condition. Initial admission rejects robot penetration
above the declared 2 mm threshold; plate-floor collision is checked during the
rollout. `native_scene_task_pass` records the adapted task predicate.
`physics_validated` remains false pending review of the separately recorded
common gates and actuator replay, even if the native predicate passes.
Source replay success, Reachy IK success and native-scene diagnostic outcome are
separate fields. Failed attempts retain complete available recordings.

The first complete diagnostic integrated 2,252 steps, recorded 349 pad/plate
contact frames, and ended with the task false. Maximum robot penetration was
0.244 mm and no plate-floor contact occurred. All observed states and original
source data were retained. Its outcome is not promoted to physical success.

Before physical integration, derived arm speeds reached 25–44 rad/s in the four
IK candidates. Small geometric residuals therefore give no assurance of servo
tracking or stable grasping. No physical-success claim is made by this document.

## Explicit local robot retiming

`rollout(..., timing={"arm_speed_rad_s": 0.8,
"base_axis_speed_m_s": 0.6, "base_yaw_speed_rad_s": 1.2})` opts into a
separate derived time map. Each original interval gets the minimum integer
number of 2 ms physics steps needed for the declared target-speed budgets.
The bounded arm coordinates interpolate directly; planar yaw follows its
continuous unwrapped path. Every original boundary remains represented.
No source timestamp, source object pose, shape, mass or original action changes.
Binary gripper intent follows its original row through the saved map.

These budgets are controller design assumptions. The pinned URDF's generic
100 rad/s arm velocity entries are not treated as validated hardware capability.
Actual measured speed gates remain a separate requirement. This policy aims to
leave headroom below the existing common 1 rad/s measured-arm gate.

For the successful c795 source's best geometric candidate (`left=+1,right=-1`),
the reference duration of 4.510 s becomes 20.466 s with 10,233 actual physics
steps. 813 of 2,255 source intervals need no slowdown; the fastest single
interval requires 34 physics steps. This is not a uniform global time scale.
`robot-timing-map.npz` retains the original source clock, monotone source time at
every physical boundary, source-boundary indices, per-interval substep counts,
derived robot targets and original command-row index. Every applied substep
control and measured state is recorded; no frames are dropped to improve runtime.

The previous 190-second wall-time diagnostic spent most time in scalar state
pose conversion, not MuJoCo physics. Batched conversion improved capture by
12.56 times with exact equality across all 63 fields on six recorded states;
see [the bound performance evidence](../state-observer-performance.json).

## Measured common gates and independent replay

Every physical step collects solved contact forces immediately after `mj_step`,
before synchronizing position and velocity caches. The synchronized measured
robot state then supplies arm/base velocity, joint-limit margin and the existing
robot geometry checker's positive self-clearance. Original binary gripper
actions establish grasp intent: c795 uses the left hand, while its right hand
never receives closure intent. Both hands still retain their full trajectory,
state and collision checks.

The shared `physical_gates` collector measures lifted bilateral acquisition,
bilateral contact throughout carry, object-to-TCP grasp drift, hand/object and
environment penetration, and the native task's final continuous one-second
stable interval. Acceleration is recorded diagnostically; the common protocol
does not invent an acceleration acceptance threshold. Exact thresholds and
failed gates are saved in `physical-validation.json/common_validation`.
No terminal settling duration is appended by this adapter currently.

`scene-asset-binding.json` hashes every referenced scene resource before
integration. The independent audit constructs another simulator from that
scene, verifies all resource hashes, resets once from the recorded integration
state, and replays only the issued robot actuator controls. It compares every
pre-action state and the terminal state, including the free plate pose. It
rejects moving-object actuators, welds, mocap, external forces and other
assistance. `actuator-replay-audit.json` records numerical errors or an explicit
failure. Raw observations are retained before auditing, so a failed audit does
not discard the original physical attempt.

The diagnostic label remains attached during this integration stage. Complete
common-gate results, native source reproduction and task success are separate
fields; failure in any required measured gate prevents a physical-success
claim. A passing target-speed plan is never substituted for measured velocities.

The replay API was also checked against the retained first 2,252-step diagnostic:
all joint positions, velocities and timestamps reproduced exactly; maximum plate
pose error was `1.12e-15`. That check bound assets after the original run and is
explicitly a numerical regression check, not retroactive provenance admission.
The successful c795 candidate's 2,256 planned configurations have minimum
self-clearance 71.70 mm and joint-limit margin 31 mrad. These planned checks are
separate from the measured physical gates, which can still fail under servo
tracking, contact and object dynamics.

## Development result and isolated follow-ups

The c795 timed run completed all 10,233 physical steps and satisfied the native
goal predicate. Its independent actuator replay reproduced joint positions,
velocities and timestamps exactly (plate-pose error below `1.3e-15`). It is
**not a common physical pass**: translation/rotation drift reached 34.74 mm and
26.16 degrees, final stable duration was zero, and initial plate/start-rack
penetration reached 4.143 mm. Bilateral carry contact was 99.78%, measured arm
speed stayed below 0.767 rad/s, and joint/self-clearance margins passed.
An independent replay found drift before any target-rack contact, with roughly
47–50 N on both pads; placement alone does not explain it.

The unchanged native 3.1.5 source's saved configurations already have 4.735 mm
start-rack penetration at 0.096 s, before grasping. This inherited contact-model
issue is retained as a failed strict gate. An explicitly labeled alternative
`simulation_profile="global-contact-4ms"` uses the shared global compliance
profile and a 1 ms numerical step. It preserves original scene XML separately
and verifies that object geometry, inertia, friction and initial state remain
unchanged. It is not promoted as a pass under original source contact parameters.

The first compliance comparison removed the 2 mm penetration violation but
increased grasp drift to 66.77 mm / 51.55 degrees and failed the task's final
orientation predicate. Its 1 ms duration rounding also changed the derived
duration to 19.651 s. Further comparisons set `timing.interval_grid_s=0.002`
to retain exactly the same 20.466 s interval schedule across numerical steps;
the first comparison remains labeled with its timing difference.

The aperture-only follow-up derives its constant pad attachment from the
unchanged native plate: the source closing-axis line at the first closed and
15 mm lifted frame (764) intersects the original convex collider across
7.90126 mm. Inverting Reachy's CAD aperture on its opening branch gives
0.222199 rad and a pad midpoint `[-0.010196,0,-0.04325928]` m in TCP coordinates.
This shifts the robot TCP calibration by 3.31087 mm while preserving the source
pinch center, clock and object references. The right-hand calibration stays
unchanged. The measured earlier left gripper aperture was approximately
0.21961 rad, supporting the geometric scale but not proving stability.

The solver accepts `pad_calibration_gap_m={"left":0.007901259983523621}` with
an explicit `pad_calibration_evidence` dictionary. Original and altered
calibrations, raw IK and every failed physical attempt are retained. Aperture
correction alone, contact-profile correction alone and their combination are
separate experiments; none inherits a physical-success label from task success.

All these target rollouts cover the complete **available reconstructed pose
interval**, starting at the first saved post-action state at 0.002 s. They do
not execute or claim reconstruction of the absent initial pre-action 2 ms.
`performance.json` measures setup, rollout (including capture), stacking,
lossless NPZ export, actuator audit and common-archive export without changing
control or recording cadence.

The development side-rim repositioning candidate now has a complete 2,256-frame
IK solution retained in the verified shared recovery package, with both source hands constrained.
Its source-clock planning base bounds are 1.2 m/s per axis and 2.4 rad/s yaw;
execution uses a separate interval-local map capped at 0.8 rad/s for arms,
0.6 m/s per base axis and 1.2 rad/s yaw. The available reconstructed source pose
interval is 4.510 s; the derived physical duration is 17.506 s. No uniform slowdown
or source timestamp rewrite is used. The derived IK archive and raw solver evidence are durably exported with immutable path mappings.

A conservative static check over the old `-0.06 .. 2` rad left-gripper command
envelope found 4.158 mm intersection between the two distal finger collision
meshes at the negative endpoint. This rejected check is retained. The right
hand was held at its verified open 2 rad command and remained included in all
fixture pairs; no hand collision pairs were removed. Both actual candidate
controller envelopes, using contact angle 0.2456865 rad minus 0.02 or 0.05 rad,
passed all source rows with at least 3 mm fixture separation, 28.04 mm minimum
arm-sphere self-clearance, and zero sampled robot mesh self-penetration. These
are finite configuration/aperture samples, not continuous swept-volume or
actual servo-state guarantees. Only the three exact wheel/floor support pairs
are exempted; the moving plate is assessed by separate physical contact gates.

`rollout(..., gripper_policy=...)` optionally accepts
`{"method": "cad_contact_minus_margin", "closure_margin_rad": {"left": 0.02},
"evidence": {...}}`. It derives a positive closed motor target from the declared
CAD contact angle while preserving source binary grasp/release intent and the
open endpoint. It retains the default endpoint conversion separately from the
actual policy targets, and saves every issued physical control. The margin is a
controller setting, not a contact-force guarantee. The source object geometry,
mass, friction and reference trajectory remain unchanged.

The producer additionally audits positive forces between the task plate and
any source-declared inactive hand immediately after every physics step, before
cache synchronization. All events and timestamps are retained. Passing the
native task predicate alone still does not promote a run: complete measured
common gates, actuator replay, the inactive-hand audit, source/protocol review
and durable artifact availability must be checked independently. Diagnostic
producer metadata always reports `physics_validated=false` until that review.

Four initial policy rollouts were interrupted by a container OOM at
2026-10-07 11:30:16 UTC because they were incorrectly colocated in one 4 GiB
container. All four exited 137; none produced a surviving complete physical
result. The previously verified 69-file IK/geometry backup was preserved
(SHA256 `7dcd59f90217088dcc913085f7dc21400b39ffaf923353238f05e0f6ae55aa09`).
The earlier passing policy-envelope checks had only ephemeral files and must
be recomputed before admitting a retry. Recovery assigns one rollout to each
of four distinct containers, holds the shared per-pod execution lock, verifies
the restored files and producer overlay, and records container memory throughout.
The immutable shared recovery package and explicit interruption records are
under `/mnt/reachy-retarget/workspaces/bigym-side-baseplan-oom-recovery-20261007-01`.

The four isolated retries are now complete and durably hash-verified on shared
storage, including their common HDF5 archives and JSON sidecars. Both closure
margins under both contact profiles failed lifted acquisition; all actual
actuator replays and inactive-right-hand contact guards passed. Original-contact
runs retain the source rack penetration failure; the alternative contact profile
passes the contact-depth gates but still does not lift. Detailed hashes, gates,
resource peaks and retained interruption records are in
[bigym-policy-recovery-v1-evidence.json](bigym-policy-recovery-v1-evidence.json).
The largest isolated container peak was 4.289 GB of 4.295 GB, with no OOM event;
future runs discard expendable cached pages of already durable local archives
before starting and continue measuring memory. No observations are decimated.

Independent replay identified 480 steps of bilateral pad force but no lift.
At first closure, measured TCP error relative to the actual interpolated command
was 31.88 mm. Offline FK decomposition attributed a 32.50 mm correction to the
arm configuration, versus 1.86 mm for base translation and 1.03 mm for base yaw;
these separate finite corrections are not an additive decomposition. Indexed
source versus interpolated-command mismatch was only 0.384 mm. The FK calculation
reproduced saved TCP observations within 4.35e-13 m.

`arm_feedforward_policy={"method":"bounded_velocity", "scale":1,
"max_offset_rad":0.2, "max_command_speed_rad_s":1,
"joint_margin_rad":0.025, "boundary_taper_s":0.1, "evidence":{...}}`
uses the existing `servo_feedforward.velocity_offsets` calculation, based on each
actual actuator's `kv/kp`. Both arm commands are clipped to position limits with
a margin and to the declared command slew rate; start/end compensation tapers
smoothly. Base, neck, gripper intent, reference poses, timestamps, model gains
and object physics are unchanged. All issued commands, unchanged interpolated
arm references and applied compensation offsets are recorded at every physics
step. This is a controller experiment; reference-path static admission does not
certify the changed servo commands or resulting motion. Full measured gates and
independent actuator replay remain mandatory.

The bounded-feedforward pair is complete, exported and independently hash-checked
across pods. It remains a failed acquisition experiment. The first-closure
reference error worsened from 31.88 to 38.55 mm; the closure-to-anchor median
improved from 32.72 to 28.05 mm. Later reference tracking improved, including
9.75 to 3.69 mm at source frame 764, but neither run lifted the plate. The
alternative-contact run also exceeded measured arm and base velocity gates
(1.083 rad/s arm; 0.6153 m/s base X), despite bounded command slew. Both exact
actuator replays and inactive-hand audits passed. The optional policy therefore
remains disabled by default. Complete results and artifact hashes are in
[bigym-feedforward-v1-evidence.json](bigym-feedforward-v1-evidence.json).
