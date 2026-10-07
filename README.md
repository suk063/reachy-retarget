# Reachy retarget

State-first acquisition and representative Reachy 2 retargeting for manipulation
and arm/base tracking policies. The project is located at
`/home/sunghwan/workspace/reachy-retarget`. Sibling robot projects are read-only
references. No robot SDK, hardware connection, policy training or deployment is
part of this project.

Read [REPORT.md](REPORT.md) for measured results and outstanding source-specific
limits. A downloaded motion, a kinematically feasible motion, and a successful
contact simulation are three different outcomes.

## Environments

Acquisition and normalization run in this project's `.venv`:

```bash
cd /home/sunghwan/workspace/reachy-retarget
python -m venv --system-site-packages .venv
.venv/bin/pip install -e .
.venv/bin/python -m pytest -q
```

Robot conversion and MuJoCo replay dispatch to the existing
`/home/sunghwan/workspace/reachy-control/.venv/bin/python`. That environment has
the Pinocchio ABI expected by the native controller. `robot.py` checks the exact
SHA256 identities of the controller, native library, URDF and collision model
against `catalog/controller_identity.json`; it refuses to silently use modified
files. The current runtime uses Pinocchio 2.7, NumPy 1.26 and MuJoCo 3.8. The
workspace's system Pinocchio 4 runtime is not ABI-compatible with this controller.
The dispatch neither installs into nor edits the sibling projects.

The current desktop renderer uses `MUJOCO_GL=glfw`; a compatible headless MuJoCo
installation may use a different backend. Do not assume this machine's EGL
installation works without testing it.

## Commands

```bash
# Expand official registries and publicly accessible repositories.
.venv/bin/reachy-retarget discover --group all
# Or replay this run's fixed, reviewed state-only selection:
.venv/bin/reachy-retarget discover --locked configs/collection-lock.jsonl.gz

# Resume the durable queue. The official Xet batch path handles small HF files;
# HTTP Range handles other files and selected HDF5/archive/Parquet state reads.
.venv/bin/reachy-retarget fetch
.venv/bin/reachy-retarget fetch --source parahome --transport http

# Five sequences per representative source; Can normalizes fixed demo_0..9.
.venv/bin/reachy-retarget normalize --limit 5
.venv/bin/reachy-retarget retarget --limit 5

# Checksums, masks, transforms, source splits, batch loader and independent
# reachy-control/rl_tracking audit. Failed conversions remain failed.
.venv/bin/reachy-retarget validate

# Re-run all fixed ten dynamic Can trials, then the normal validation suite.
# This replaces the files in the configured label; use a new label in a copied
# config when comparing another controller or grasp/time adjustment.
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
