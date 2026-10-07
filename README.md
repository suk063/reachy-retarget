# Reachy retarget

State-first acquisition and representative Reachy 2 retargeting for manipulation
with time-aligned object state. Sibling robot projects are read-only
references. No robot SDK, hardware connection, policy training or robot deployment is
part of this project.

The [persistent cluster pipeline](docs/cluster-retargeting.md) uses 50 existing
hosts and shared storage at `/mnt/reachy-retarget`. All source families use the
[same state-only archive](docs/reachy-agent-dataset-format.md), referenced against
the pinned remote `reachy-agent/main`. Direct object-pose channels include only
task objects and their relevant supports, receptacles and articulated parts.
New complete-state MuJoCo rollouts record both arms, neck, grippers, base,
actual controls and controller history without RGB. Source-only archives remain
explicitly incomplete; the writer never invents unavailable Reachy observations.
[Absolute/delta control views](docs/control-views.md) preserve the distinction
between observed motion and commands actually issued.

Current active work: [maximize full-interval physical success on feasible trajectories](docs/feasible-success-goal.md),
using all 50 persistent workers and preserving original objects and failed attempts.
The following paragraphs preserve the earlier experimental milestones.

Earlier direction: [restore a controlled gripper-pad alignment baseline](docs/contact-alignment-reset.md).
The [translation-only comparison](docs/pad-alignment-validation.md) is implemented:
two of six examples recover bilateral lifted grasps, but none passes all physical
gates. Existing exports remain preserved; correction is an explicit experiment.
The broad common trajectory optimizer remains experimental. Its full-interval
run passed 1/20; the historical per-task 6/20 below uses different source windows
and must not be presented as a comparable success-rate improvement or regression.

Read [the executed dynamics validation](docs/dynamics-validation.md) for historical
physical results and [the expanded mobile dataset survey](docs/datasets/README.md)
for additional sources. Six of 20 MuJoCo trials pass every physical gate:
Can 1/3, Lift 3/3, Stack 1/3, PickCube 1/5, Square 0/3, Threading 0/3.
All 20 saved trajectories reproduce from actuator commands alone. The six
Lift/Stack trials explicitly start at the stationary source frame at 0.25 s;
full-interval failures and original source data remain preserved. Four passes
are development episodes and two are additional episodes with frozen settings.
This does not establish broad object or mobile-navigation generalization.

The local matrix contains 29 source episodes; nine have no complete dynamics
adapter. Its [earlier kinematic audit](docs/coverage-audit.md) records 25 kinematic
passes. [REPORT.md](REPORT.md) preserves a historical acquisition snapshot whose
payloads are not all present here. Downloaded, kinematically feasible and
successful contact simulation are separate outcomes.

Objectless selections are excluded from normalization, retargeting and known
source acquisition. The previously downloaded robot-only Reachy selection,
including its raw files and generated trajectories, was removed at the user's
request. Removal manifests preserve URLs, revisions and checksums. A robot
dataset with object state remains eligible; a family name alone is not grounds
for deletion. A missing adapter also does not prove that upstream data lack
objects.

## Environments

Acquisition and normalization run in this project's `.venv`:

```bash
cd /path/to/reachy-retarget
python -m venv --system-site-packages .venv
.venv/bin/pip install -e .
.venv/bin/python -m pytest -q
```

The original pinned Linux setup dispatches robot conversion and MuJoCo replay to
`/home/sunghwan/workspace/reachy-control/.venv/bin/python`. That environment has
the Pinocchio ABI expected by the native controller. `robot.py` checks the exact
SHA256 identities of the controller, native library, URDF and collision model
against `catalog/controller_identity.json`; it refuses to silently use modified
files. That runtime uses Pinocchio 2.7, NumPy 1.26 and MuJoCo 3.8. The
workspace's system Pinocchio 4 runtime is not ABI-compatible with this controller.
The dispatch neither installs into nor edits the sibling projects.

The current desktop renderer uses `MUJOCO_GL=glfw`; a compatible headless MuJoCo
installation may use a different backend. Do not assume this machine's EGL
installation works without testing it.

## Commands

### Acquire and normalize the object-inclusive matrix

Use a fresh local root and the read-only sibling reference model. Downloads are
explicit; the lock records source revisions and acquired checksums. The 50 GB
free-space reserve applies to every transfer.

```bash
export REACHY_RETARGET_CONTROL="$(cd ../reachy-control && pwd)"
export REACHY_RETARGET_BACKEND=reference
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/reachy-retarget --root runs/object-matrix-reproduction discover --locked configs/object-matrix-lock.jsonl.gz
.venv/bin/reachy-retarget --root runs/object-matrix-reproduction fetch --transport http
.venv/bin/reachy-retarget --root runs/object-matrix-reproduction normalize --source hf__robomimic__robomimic_datasets --source hf__amandlek__mimicgen_datasets --source humoto --limit 3
.venv/bin/reachy-retarget --root runs/object-matrix-reproduction normalize --source hf__haosulab__ManiSkill_Demonstrations --limit 5
.venv/bin/reachy-retarget --root runs/object-matrix-reproduction retarget --limit 100
```

The current local reference environment uses Python 3.12, Pinocchio 4.1,
NumPy 2.5.3 and MuJoCo 3.15. Its identities are distinct from the pinned Linux
controller. The lock includes Panda FK evidence and Can simulation assets as
well as four demonstration sources; asset bundles are not extra datasets.
Normalization limits are per source file for robomimic/MimicGen. Task file names
participate in episode identity, so Stack and Threading do not overwrite one
another. Object validity masks survive resampling, with no interpolation across
missing observations. See the [ManiSkill adapter notes](docs/maniskill-adapter.md)
for its measured Panda state, independent FK check and scene placement.

### Frozen object dynamics validation

After explicit acquisition and normalization, run the saved per-task settings:

```bash
.venv/bin/python -m reachy_retarget.dynamics_benchmark --root runs/object-matrix-reproduction --config configs/dynamics-validation-v2.json --label frozen_v2
.venv/bin/python -m reachy_retarget.dynamics_audit runs/object-matrix-reproduction/runs/dynamics/frozen_v2/6ad5cc23ec6d1c207da3e4b2
```

Every attempt retains the actual free-object outcome, actuator commands, source
identity, exact scene/asset hashes, gates, failures and an independent replay
audit. Use a new label for each run. See [the implemented method](docs/dynamics-method.md)
and [source scene conversion](docs/object-scenes.md) for force feedback, grasp
calibration, controlled placement, source windows and simulator assumptions.
The method performs bounded simulation-guided tuning; it is not MPC.

The viewer accepts these generic multi-object physical recordings directly:

```bash
.venv/bin/python -m reachy_retarget.viewer runs/object-matrix-reproduction/runs/dynamics/frozen_v2/6ad5cc23ec6d1c207da3e4b2/replay.h5 --port 8082
```

Reachy's source visual meshes/materials and the tripod display repair are
retained. The scene's original materials and available textures are preserved;
primitive texture-projection limitations are reported. Recorded-state playback
is separate from the actuator-driven validation.

### Local mjviser preview

Install `.[preview]` into this project's virtual environment for browser playback.
`python -m reachy_retarget.viewer /absolute/path/to/motion.h5 --port 8080`
serves a looped replay at `http://127.0.0.1:8080`, with play/pause,
speed, frame, robot-visual and path controls. `--check` verifies every frame and compares
MuJoCo hand positions against the stored retargeted FK without starting a server.
The viewer preserves the `reachy-control` URDF visual meshes and their embedded
materials through GLB rendering, driven by MuJoCo body transforms. Collision
geometry is hidden by default (group 3). Recorded neck motion is included;
unavailable gripper channels stay at model defaults. This is kinematic playback,
not physical validation.

The local `viewer/reachy.visual.urdf` resolves the original ROS mesh paths and
extends the `back_bar_inner` visual to the torso attachment, matching
`reachy-control/util/visualization.py`. The left/right inner supports also extend
along their existing axes to the actual torso visual mesh: all four corners of
each upper end intersect the shell, with 2 mm minimum display overlap. Their
lower endpoints stay fixed. The remaining source gap is about 23 mm at each
front support center. Source URDF, joints, collision geometry,
inertias and sibling repositories are unchanged. The visual manifest records
source revision, source/GLB checksums and every display correction. Source DAE
materials are retained; no replacement robot texture is invented.

For robomimic Can trajectories, the viewer also replays `objects/Can/pose` and
the source bins, using the retargeted scene's shared rigid placement. Source
Can, wood and metal textures must already be fetched under
`data/raw/robosuite_can_assets/`; viewer imports and playback never download.
Box visuals use face-local UV meshes because mjviser omits 2D textures on
primitive boxes; this display approximation is recorded separately. Colliders
remain unchanged. Recorded object poses are assigned only for playback, with
no physics steps and no inference of grasp/contact success. Unsupported object
assets and missing Can poses raise errors rather than silently dropping objects
or filling missing tracks.

For an explicitly separate local conversion when the pinned Linux native runtime
is unavailable, set `REACHY_RETARGET_CONTROL` to the existing read-only
`reachy-control` checkout and `REACHY_RETARGET_BACKEND=reference`. This selects
`control.reachy.Reachy`, the Python reference model, in the current Python
environment. Use a separate `--root`, such as `runs/local-preview`, to keep its
outputs apart from the pinned native run. Its model/source hashes are written to
`catalog/controller_identity.reference.json`; outputs identify their backend.
The default remains the pinned native runtime and its original identity check.
A local kinematic pass is not the independent native tracking audit or a contact
simulation pass.

The object sample uses robomimic revision
`74fa018461f479cd9fd15b924a16103012096203`, `v1.5/can/ph/demo_0`, and robosuite
`v1.5.1` assets (all URLs and acquired checksums in the local ledger):

```bash
.venv/bin/python -m reachy_retarget.viewer runs/local-preview-v2/data/retargeted/hf__robomimic__robomimic_datasets/6ad5cc23ec6d1c207da3e4b2/motion.h5 --port 8081
```

This local Can sample is 586 frames / 5.85 seconds and passed the reference
kinematic check. It is separate from the historical native contact benchmark
in `REPORT.md`. Viewer FK/object pose checks, visual/scene manifests and the
adapted URDF live beside each motion under `viewer/`.

The corrected Can output is in `runs/local-preview-v2`; the first conversion is
preserved in `runs/local-preview`. The documented binary robomimic OSC gripper
action now produces `derived/gripper/right_position`: -1/open maps to 2.0 rad,
and +1/close maps to -0.06 rad, following the existing contact benchmark's command
mapping. Resampling uses zero-order hold on the globally dilated source clock,
so closing never starts before its source command. For this sample the switch
times are 2.60 s (close) and 5.20 s (open). Finger mimic joints follow the target
in playback. These are derived Reachy commands, not measured Reachy finger
positions or evidence of physical grasp success. Unknown action schemas are
rejected instead of guessed; no inactive-hand command is fabricated.

For contact-aware playback, run the dynamic contact validator and open its
`replay.h5`. The viewer then uses complete measured MuJoCo positions, velocities
and actuator controls, including finger mimic deflections under load. It checks
the recorded model checksum and verifies that display additions preserve the
physical bodies, joints and inertias. The moving Can remains a free dynamic body
throughout validation; object assignments occur only in the completed replay.

```bash
# Keep the environment exports above. Use a fresh label for each attempt.
.venv/bin/python -m reachy_retarget.contact --root runs/local-preview-v2 --limit 1 --label can_contact_v4 --time-scale 10 --depth 0.035 --no-render
.venv/bin/python -m reachy_retarget.viewer runs/local-preview-v2/runs/contact/can_contact_v3/demo_0/replay.h5 --port 8081
```

The existing `can_contact_v3/demo_0` reference-backend trial passed the recorded
criteria: 0.1198 m lift, 8.97 s final stable placement after release, bilateral
distal-finger contact during lift, and maximum hand/Can penetration of 0.891 mm
(limit 1 mm). No self/environment penetration above their configured thresholds
or actual velocity-limit violation was observed. MuJoCo uses compliant contact;
this is a bounded-penetration result, not a claim of exactly zero penetration.
The source clock is slowed by 10 with a final 3 s hold (61.5 s replay). The
grasp-depth offset is 35 mm; source object dimensions are preserved.

The local controller expects targets in its torso camera frame. The adapter now
converts world targets to that frame, preserving the base-frame contract for
older controllers without a camera-frame API. The failed `can_contact_v1` and
successful `can_contact_v2` remain saved. Version 3 additionally aligns logged
body poses with interval-end joint states and records complete replay state.
Each attempt retains code snapshots, model/source hashes, runtime versions,
failure reasons and contact metrics. This single development sample does not
replace the historical ten-trial benchmark or establish general success rates.

The improved local grasp is `speed4_grasp15_soft/demo_0`. It uses a 25 mm
insertion offset, a 15 mm TCP-X grasp-height adjustment toward the Can's center,
and 15 degrees of wrist tilt. The attachment changes smoothly during the last
0.5 source seconds before closing, preserving the original approach. Closing
ramps over 0.3 simulation seconds; source opening time is unchanged. No carry
height offset is applied. The complete physical scene has the same SHA-256 as
`can_contact_v3`: robot/object geometry, mass, inertia, friction and contact
settings are unchanged. Only robot target poses, their timing and gripper
commands change. The adapter also supplies the camera-frame controller with
the inactive hand's body-attached target Jacobian and the active hand's world
trajectory velocity, preventing base motion from being counted again as target
motion.

```bash
# Use a fresh label to reproduce; all unsuccessful attempts remain saved.
.venv/bin/python -m reachy_retarget.contact --root runs/local-preview-v2 --limit 1 --label grasp_reproduction --time-scale 4 --depth 0.025 --grasp-height 0.015 --tilt 15 --close-duration 0.3 --no-render
.venv/bin/python -m reachy_retarget.viewer runs/local-preview-v2/runs/contact/speed4_grasp15_soft/demo_0/replay.h5 --port 8081
```

The motion is 2.5 times faster than the previous time-scale-10 run (26.4 seconds
including the final hold, versus 61.5). This run passes the physical criteria
and a new grasp-stability gate: TCP-relative object drift below 3 mm / 3 degrees,
bilateral distal-finger contact in at least 95% of measured carry intervals, and
an observed release. Drift is measured from the first bilateral grasp above
3 cm lift until the release command; it is not the Can's world displacement.
Measured drift fell from 6.67 mm / 10.41 degrees to 0.47 mm / 1.81 degrees, with
bilateral contact in all measured carry intervals. Maximum hand/Can penetration
is 0.858 mm; lift is 14.49 cm; final stable placement is 5.15 seconds. The earlier
baseline did not use the new stability gate. `runs/local-preview-v2/runs/contact/
grasp-comparison.json` retains comparisons and failure reasons for all local
attempts; they are variations of one demonstration, not independent examples.

### Collection and validation

```bash
# Expand official registries and publicly accessible repositories.
.venv/bin/reachy-retarget discover --group all
# Or replay this run's fixed, reviewed state-only selection:
.venv/bin/reachy-retarget discover --locked configs/collection-lock.jsonl.gz

# Resume the durable queue. The official Xet batch path handles small HF files;
# HTTP Range handles other files and selected HDF5/archive/Parquet state reads.
.venv/bin/reachy-retarget fetch
.venv/bin/reachy-retarget fetch --source parahome --transport http

# Up to five sequences per selected source file; objectless episodes are rejected.
.venv/bin/reachy-retarget normalize --limit 5
.venv/bin/reachy-retarget retarget --limit 5

# Checksums, masks, transforms, source splits, batch loader and independent
# reachy-control/rl_tracking audit. Failed conversions remain failed.
.venv/bin/reachy-retarget validate

# Re-run all fixed ten dynamic Can trials, then the normal validation suite.
# Existing attempts are protected; use a fresh label in a copied config when
# comparing another controller or grasp/time adjustment.
.venv/bin/reachy-retarget --config configs/default.json validate --physics

# Refresh measured sequence counts and regenerate English reports.
.venv/bin/reachy-retarget report --refresh-inventory
```

`--root` and `--config` precede the command. Repeated `--source` selects multiple
sources. `fetch --limit` limits an invocation, not the collection's total storage
budget. Source selection, exact revisions and excluded routes live in the
ledger/evidence directory. The immutable collection lock can recreate this
specific selection without treating the latest upstream revision as identical.

## Storage and acquisition

- `catalog/`: SQLite ledger, source manifests, source-code/documentation evidence,
  versions, licenses, checksums, source identity and sequence inventories.
- `data/raw/<source>/`: acquired original states or explicitly marked state-only
  projections of mixed archives. Selected projections include an extraction
  manifest; their local hash does **not** verify the full remote archive.
- `data/blobs/`: SHA256-addressed original payloads, hardlinked into `raw` for
  byte-level mirror deduplication.
- `data/assets/`: linked object meshes and the converted Reachy collision model.
- `data/normalized/`: common HDF5 episodes with JSON sidecars.
- `data/retargeted/`: Reachy HDF5 trajectories, validation reasons and passing
  tracking NPZs.
- `runs/contact/<label>/`: actual simulated state, targets, actuator commands,
  contact metrics, per-trial results and MP4 replays, including failed trials.

Downloads resume `.part` files with validated Content-Range; published LFS
SHA256 values and known lengths are checked. Corrupt payloads are quarantined.
HDF5 projections resume at completed dataset boundaries; archive projections
resume completed members. Three attempts bound each transfer. HTTP 429 preserves
the queue and pauses the provider; public Xet read tokens are ephemeral and never
written to the catalog. No account registration or application is submitted.

Before each transfer, reconstruction batch or archive expansion, free space must
remain at least **50 decimal GB**. `storage_wait` leaves the queue intact. This is
a free-space floor, not a total collection cap. No video-only source is eligible;
image/video fields inside mixed sources are omitted where range extraction is
supported. Source-specific failures and partial extraction scopes are retained.

## Canonical episode contract

Each episode is `episode.h5` plus `episode.json`. Units are **m, rad, s** and every
normalized pose is `[x, y, z, qw, qx, qy, qz]`. Data is right-handed. Source frames,
source units, exact joint order, conventions, source file hashes, timing, missing
labels and transformations are recorded in the sidecar. Unknown semantics remain
unknown; the raw arrays are retained instead of assigning guessed labels.

Typical channels:

```text
time_s                         T
robot/joint_names              J
robot/joint_position            T,J
hand/left_pose                 T,7  (only when observed/derivable)
hand/right_pose                T,7
human/joint_names               J   (when available)
human/joint_pose                T,J,7
objects/<id>/pose               T,7
objects/<id>/valid              T   (missing tracks are NaN + false)
source/action                  T,A (original command semantics retained)
source/...                     original states, fingers, articulation, metadata
```

Objects link to meshes by source IDs and hashes. Shape scale is retained; glTF
node scale is baked into its local mesh once and recorded. Canonical ParaHome
keeps articulated parts and the original articulation metadata. Human fingers
remain in the source representation. A thumb/index aperture heuristic, when
available, is a separate `derived/gripper/...` output, never a contact label or
an observed actuator command. Native Reachy SDK joint angles are converted from
documented degrees; raw native gripper values are preserved without inventing
an unverified hand-motor calibration.

The five exercised adapter cases are Reachy LeRobot v2, Reachy LeRobot v3,
robomimic HDF5, ParaHome joint/object archives and the public HUMOTO GLB skeleton
release. Other acquired schemas remain raw until their own adapter is verified.
ParaHome's camera-synchronized release is 30 Hz; original hand/body rotations and
object validity masks are preserved. HUMOTO glTF uses meters and Y-up; a shared
rigid basis transform maps it to Z-up. robomimic Can uses the world-object half
of its 14-value object observation, confirmed against the source environment.

## Retargeting and splits

A common scene rotation/translation moves the human/robot, both hands and every
object together. Constant wrist-to-tool attachments and any global time dilation
are recorded. Object size never changes. Some native recordings exceed the pinned controller wrist limits. Their observed
FK goals are re-solved with bounded arm/base IK; original joint arrays and the
number of out-of-margin source frames remain recorded. Discontinuous motions, inaccessible
hand goals, joint margin violations and self-collision failures receive diagnostic
HDF5 output, not passing tracking examples.

The tracking export matches `reachy-nominal-v2`: 100 Hz, 601 samples per six-second
window, two world-frame hand poses, 14 arm joints, planar base witness, family 6
(`cooperation`) and actual source duration. The base is reset to zero by a shared
rigid transform at each window start. Short tails hold their last state and are
explicitly marked. Each window carries `parent_source`; splits are assigned to
original source groups **before** windowing. Exact normalized duplicates share
their source group and split. Witness FK is used for the feasible training target;
the requested source-derived target and its residual remain separate in HDF5.

The existing tracking environment fixes the neck. Derived neck rotations and
residuals remain in HDF5 and are not silently treated as validated by the arm/base
NPZ audit. The native controller's independently implemented tracking audit checks
FK, limits and clearance; it does not establish physical contact feasibility.

## Manipulation loader

```python
from pathlib import Path
from reachy_retarget.episodes import StateDataset

paths = list(Path('data/normalized/hf__robomimic__robomimic_datasets').rglob('*.h5'))
dataset = StateDataset(paths,
    ['hand/right_pose', 'objects/Can/pose', 'source/action', 'contact/observed'],
    window=16)
sample = dataset[0]
# sample['data'] contains available channels;
# sample['available']['contact/observed'] is false when the source lacks contact.
```

The source episode preserves original commands. `motion.h5` contains requested
Reachy goals and kinematic states. `replay.h5` separately stores `target/`,
`command/` and `simulation/`; only the latter contains actual contact outcomes.
This prevents using a commanded object trajectory as simulated ground truth.

## Physical benchmark

The fixed set is robomimic Can proficient-human `demo_0` through `demo_9`.
`configs/default.json` records the final tested conversion: time scale 10,
heading 0, grasp depth 0.035 m. All demonstrations use the same conversion
parameters. Source grasp, transport and release timing determine the targets.
Native controller code is unchanged; world targets are transformed into the
**measured current base frame** at every control step.

The Can is a free rigid body with the source collision mesh, mass, inertia and
friction. It is initialized once at reset. There is no weld, mocap body, repeated
object-state assignment or object-target forcing. Physics runs at 500 Hz and
control at 100 Hz. Robot-only gravity compensation, planar servo assumptions,
collision exclusions for adjacent rigid segments and extra friction assumptions
are recorded in every result.

The task requires actual grasp contact, at least 8 cm lift and final placement in
the target bin, followed by an open gripper and no hand contact for at least one
continuous second. Linear/angular stability thresholds are 0.02 m/s and 0.2 rad/s.
Joint/base speed, clearance, joint margin and robot/environment collision checks
are additional independent pass conditions. Calibration and these trials form a
development feasibility benchmark, not a held-out success-rate estimate.

## Scope and licensing

The catalog includes successful, partial, gated, failed and unsuitable routes.
Public mirrors retain their upstream corpus family and license provenance. A
missing license field is not a grant of rights. Gated GRAB/InterCap/ARCTIC/AMASS
releases and application-only portions are recorded without submitting forms.
LAFAN1 originals are retained under its source terms and are not converted by
default. Raw/reformatted/retargeted versions of one motion are not claimed as
independent demonstrations. Full-dataset conversion, policy training and real
robot execution remain follow-up work.

## Long acquisition runs

The provider may return HTTP 429 while enumerating or acquiring hundreds of
thousands of small files. Discovery checkpoints each committed page and pins the
revision. Fetch preserves partial payloads and the retry deadline in
`catalog/network_backoff.json`; it does not bypass provider limits.

A local queue-draining process can keep this collection running through those
cooldowns:

```bash
.venv/bin/python -u -m reachy_retarget.collect --root "$PWD" > runs/collector.log 2>&1
```

Inspect `runs/collector-status.json` and the durable ledger for current state.
Create `runs/collector.stop` to request a graceful stop between batches; remove
that file before a later restart. This is an immediate local process, not a cloud
service. It never retries a failed payload indefinitely: individual transfers
have bounded retries, and exhausted failures require an explicit `fetch` retry.
It stops for a storage decision before spending the 50 GB reserve. A pending
queue or provider cooldown must not be reported as complete acquisition.

Selected official RLDS shards require the optional dependencies installed with
`pip install -e '.[rlds]'`. TFRecord records are checked with CRC32C and decoded
without writing pixel arrays. Flat TFDS shapes are preserved with the official
`features.json`; missing coordinate units or frame rates are not guessed. NPZ
frames nested in archives are also projected to omit embedded RGB/depth arrays.
Original full-file identities remain in the ledger when only a state projection
is kept. For CALVIN's debug release, the verified mixed archive was replaced by a
state-only ZIP, reducing retained data from about 1.3 GB to 4.2 MB.

`catalog/object_inventory.json` lists source-scoped IDs with observed pose and
geometry links separately from the canonical representatives. It currently
checks TACO, ParaHome, HODome and HO-3D/YCB asset links; it does not turn every downloaded mesh into an
observed physical object. Language Table's documented 2D effector-only states
are excluded from 3D retarget counts. QUT's official bucket spelling is
`qut_dexterous_manpulation`, as verified against the actual public listing.

JSON episode sidecars are authoritative for later provenance clarifications;
the HDF5 `metadata_json` attribute is the creation-time snapshot. Numeric HDF5
content is immutable after creation. Native Reachy examples lacking measured
base pose are explicitly marked missing: stationary-base FK is a declared
reference, not an observation of the world's base trajectory.

Official gzip-compressed subject archives can use the `#state-tgz` projection.
They are decompressed as a stream; images are discarded without being written.
Compressed streams cannot seek directly to a decompressed member on restart, so
resumption replays the input prefix and skips output members already committed
in the ZIP checkpoint. The transfer is not falsely described as byte-resumed.

The MANUS-derived DexYCB mirror contains isolated 3D hand mesh samples. These are
retained as geometry/annotation samples, not counted as temporal policy episodes
or as full DexYCB object trajectories. Official DexYCB calibration, BOP labels,
object models and subject streams are tracked separately under `dexycb`.
