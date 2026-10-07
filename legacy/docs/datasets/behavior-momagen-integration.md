# BEHAVIOR and MoMaGen integration audit

Checked on 2026-10-06. This closes the release/schema investigation for these two
sources; it does not claim that all their tasks are retargeted or pass Reachy
physics. Data acquisition belongs on the cluster under `/mnt/reachy-retarget`.
No raw dataset was downloaded into the local repository for this audit. HTTP
range reads inspected HDF5 metadata and small pose/gripper probes. Source-code
and documentation responses have checksums in the adjacent evidence JSON.

## Corrected finding: MoMaGen includes source demonstrations

The previous survey did not verify downloadable MoMaGen demonstrations. The
[pinned public repository](https://github.com/ChengshuLi/MoMaGen/tree/5da621667669b15b97d7a220d086283b3c8ad0e5/momagen/datasets)
contains six processed source HDF5s and six corresponding raw HDF5s. They are
ordinary Git blobs, not unresolved LFS pointers. Raw and processed views count
as the same source recordings. The files contain **27 native groups, of which
six have enriched `datagen_info`**. The remaining groups retain source states;
they are neither silently discarded nor classified as robot-only or successful.

| Processed file | Bytes | Enriched group | Reference frames | Recorded active gripper |
| --- | ---: | --- | ---: | --- |
| `r1_picking_up_trash.hdf5` | 7,632,853 | `demo_0` | 2,483 | right |
| `r1_bringing_water.hdf5` | 15,012,078 | `demo_0` | 3,324 | right |
| `r1_tidy_table.hdf5` | 25,037,752 | `demo_1` | 1,273 | left |
| `r1_pick_cup.hdf5` | 29,948,093 | `demo_14` | 909 | left |
| `r1_dishes_away.hdf5` | 88,217,546 | `demo_2` | 2,788 | both |
| `r1_clean_pan.hdf5` | 103,232,338 | `demo_4` | 1,751 | both |

The six processed files total 269,080,660 bytes. All inspected configurations
record at 30 Hz and explicitly use **assisted grasping**. The source extractor
reads achieved world EEF transforms, not action targets. Its `[N,8,4]` EEF array
stacks left then right. It also records world base transforms, named object
transforms, and left/right gripper commands. `robot_r1` and `torso_link4` occur in
some object-pose groups; the adapter preserves them as robot references rather
than counting them as scene objects. [Source interface](https://github.com/ChengshuLi/MoMaGen/blob/5da621667669b15b97d7a220d086283b3c8ad0e5/momagen/env_interfaces/omnigibson.py)

The source controller is `MultiFingerGripperController` in smooth mode with
default limits. The command convention is -1 closed and +1 open. Intermediate
commands are source position targets; these are not measured gaps or contact
forces. R1's source finger positions range from 0 closed to 0.05 m open per
finger. The Reachy angle/gap conversion must be a separately recorded derived
mapping. [Controller](https://github.com/StanfordVL/BEHAVIOR-1K/blob/2ca5503895b2c81e02226dbe11c3fee1a68b5d6c/OmniGibson/omnigibson/controllers/multi_finger_gripper_controller.py),
[R1 model](https://github.com/StanfordVL/BEHAVIOR-1K/blob/2ca5503895b2c81e02226dbe11c3fee1a68b5d6c/OmniGibson/omnigibson/robots/r1.py)

The repository code is MIT with NVIDIA-licensed portions. No separate
source-demonstration-specific license was found. Simulator assets retain the
BEHAVIOR asset terms. MoMaGen pins BEHAVIOR commit
`2ca5503895b2c81e02226dbe11c3fee1a68b5d6c` and robomimic commit
`34460828098dc8ca0694f330cf48daeb96a4ef3a`; individual recorded scene versions
can differ and remain in each output's provenance. [License](https://github.com/ChengshuLi/MoMaGen/blob/5da621667669b15b97d7a220d086283b3c8ad0e5/LICENSE),
[installation](https://chengshuli.github.io/MoMaGen/installation/)

## BEHAVIOR: accessible raw state needs source-runtime replay

The [2026 raw release](https://huggingface.co/datasets/behavior-1k/2026-challenge-rawdata)
is publicly listed with 20,000 teleoperation episodes; the processed LeRobot
release describes the same recordings across 100 tasks. These release totals
are not acquisition counts. The pinned pilot is
`task-0002/episode_00021890.hdf5`, 667,124 bytes, revision
`f9d2901112993d75684223820eec70722806c2c4`, SHA-256
`b01e7b8183e2c738edf1f5e65bba53163fee30713970845631401e20e6cd1a75`.
It belongs to `putting_away_Halloween_decorations` and actually contains 149
serialized state frames of width 701 and 148 action vectors of width 23 at
30 Hz. The adapter preserves this T+1/T distinction. Metadata counters differ
from these shapes and must not replace the observed lengths.

The pilot's recorded versions are OmniGibson `3.7.0-alpha`, BDDL `3.7.0a0`,
commit `c1736c23fe8083bb210da158008ad0bfa1638423`, assets `1.2.0rc21`.
The inspected current replay code is commit
`a8247a8cc1633fe1ca0cc66aa07243d46c64f155`. Treat compatibility across these
versions as a check, not an assumption. State vector offsets depend on scene,
object registry, and simulator version. No fixed-offset decoder is used.
[HDF5 writer](https://github.com/StanfordVL/BEHAVIOR-1K/blob/a8247a8cc1633fe1ca0cc66aa07243d46c64f155/OmniGibson/omnigibson/envs/hdf5_data_wrapper.py)

The native scene JSON contains task object identities, model/category/scale,
asset hashes, and serialized state. It does not by itself supply full temporal
object trajectories. These identities are exported with
`decoded_pose_available: false` while raw state remains archived. Source
collection and evaluation use assisted grasping, confirmed both by the pilot
configuration and a maintainer. The robot-only example configuration is not the
collection configuration. [Maintainer confirmation](https://github.com/StanfordVL/BEHAVIOR-1K/issues/2245#issuecomment-4597125008)

Assets are USD geometry/material/physics plus scene JSON. Many are encrypted
third-party assets from ShapeNet and TurboSquid. Public raw HDF5 access does
not establish permission to redistribute decrypted assets. The simulator code
is MIT; checked raw-release metadata did not provide an explicit data license.
[Asset format and terms](https://behavior.stanford.edu/getting_started/important_concepts.html)

## Implemented adapter boundary

`reachy_retarget.adapters.{behavior,momagen}` each exposes:

- `describe()`: pinned acquisition members, stages, dependencies, agent-review
  checkpoints and missing-field/physics limitations; no network side effects.
- `inspect(path)`: local native shapes, exact file hashes, recorded versions,
  grasp mode, source identity and enriched/unannotated group distinctions.
- `normalize(path, episode=None)`: `{arrays, metadata, status, missing_fields}`.
  MoMaGen emits validated recorded world transforms as xyz+wxyz. BEHAVIOR
  preserves raw state/actions and returns `blocked_native_state_decode`.

Both retain source URLs, file hashes, parent recording lineage, observed vs
converted fields, assistance and simulation assumptions. Original actions and
states retain their original frame counts. Missing robot joints, cameras,
contact forces and object dynamics are never filled with zeros to imply a
measurement. Canonical training/Reachy export is the common cluster writer;
adapters do not create separate per-dataset training formats.

MoMaGen normalizes all six enriched source references without a simulator
installation. BEHAVIOR supplies `capture_replay_frame(env, object_names)`, a
read-only callback for measured source world base/hand/object pose and object
joint positions. It neither imports a simulator nor advances or writes state.
The callback is intended for the official wrapper's
`post_state_update_callback`. Source state replay is reference extraction;
it must never run inside target physical validation.

The explicit BEHAVIOR native replay recipe in `describe()['native_replay']`
uses `OmniGibson/scripts/learning/replay_obs.py --data_folder ... --demo_id
21890 --output_format hdf5`. The script expects the input under
`2026-challenge-rawdata/task-0002/episode_00021890.hdf5` and also needs compatible
OmniGibson/Isaac Sim, a GPU/driver, asset packages and challenge task-instance
metadata. Its standard export does not automatically include our object-pose
callback; a source worker must connect it and verify every restored frame.
[Official replay implementation](https://github.com/StanfordVL/BEHAVIOR-1K/blob/a8247a8cc1633fe1ca0cc66aa07243d46c64f155/OmniGibson/scripts/learning/replay_obs.py)

## Retargeting and agent review

The first candidate preserves recorded timing, navigation, EEF orientation,
and gripper schedule. Apply only a measured Reachy pad-frame translation before
introducing trajectory optimization. Review each concrete failure: unresolved
object identity, fixture articulation, out-of-reach mobile pose, opposing
bimanual contacts or contact drift. Record any derived frame transform and
parameter change with its parent attempt. Dataset-specific adapters are allowed;
the episode format and success evidence remain common.

The source scene mechanics and separately permitted assets must be reproduced before claiming
asset-faithful MuJoCo validation. The native BEHAVIOR asset terms below block
automatic conversion of its encrypted assets without separate rights. Do not replace source objects with convenient
primitives and call it the same task. Particle/material-state tasks and missing
articulated-link trajectories remain explicit representation gaps. All actuator
rollouts keep failed attempts; source assistance and task success labels never
promote an episode to Reachy physics-validated status.

Offline adapter tests cover T/T+1 retention, left/right pose convention, robot
vs object roles, matrix validity, missing frequency, raw attempts, source replay
read-only behavior, and pinned fetch plans. Real metadata inspection is distinct
from a full native replay or target dynamics test.

## Task-only pose coverage and robot state audit

The requested pose features include **task-related objects only**. Scene registry
names are provenance, not a requirement to store poses for every wall, lamp or
piece of background furniture. Fixed world ground stays in scene geometry and
is excluded from episode object-pose features. MoMaGen selects recorded
`datagen_info` task objects, including tracked supports/targets, and checks the
pinned `task_spec.object_ref` / `attached_obj` references independently. Robot
links remain robot references. [Task references](https://github.com/ChengshuLi/MoMaGen/tree/5da621667669b15b97d7a220d086283b3c8ad0e5/momagen/datasets/base_configs)

| MoMaGen reference | Selected object pose features | Named manipulation references covered |
| --- | --- | --- |
| picking up trash | can, trash bin (2) | 2 / 2 |
| bringing water | bottle, refrigerator (2) | 2 / 2 |
| tidy table | teacup, sink (2) | 2 / 2 |
| pick cup | coffee cup, breakfast table (2) | 1 / 1 |
| dishes away | three plates, bar, shelf (5) | 3 / 3 |
| clean pan | frying pan, brush, sink (3) | 2 / 2 |

All six enriched references cover the named manipulation objects in their
pinned motion task specifications. This does not prove every semantic task state
is represented. For example, `clean_pan` also has a BDDL `dust` system; a rigid
object root pose cannot encode cleanliness or particle state. Additional BDDL
roles are listed under `other_source_task_scope_names_for_semantic_review`.
The code's current dishes-away recorder names a countertop while the published
recording contains a bar; retain the actual recorded identity and report this
revision difference, rather than renaming one asset into the other.

`object_coverage` reports selected task names, missing selected poses, source
role evidence, and excluded fixed ground. It does not make all-scene pose
coverage a normalization requirement. For the BEHAVIOR pilot, eight task pose
features are selected from nine nonrobot BDDL roles after excluding fixed ground;
none is decoded yet. Raw simulator state and task roles remain preserved.

`state_control_coverage` also prevents EEF poses from being mistaken for complete
robot joint state. MoMaGen currently decodes both world EEF poses, world base pose,
and both source gripper commands. Named arm joint position/velocity, neck
position/velocity/control, and named base velocity/control are not decoded.
Whole source state/action arrays remain available for exact later decoding.
BEHAVIOR has no decoded base/EEF/joint trajectories until replay. Neither adapter
labels source references as measured Reachy state or Reachy controls.

## Concrete native BEHAVIOR runtime recipe

The machine-readable recipe is
[`configs/native-replay-behavior.json`](../../configs/native-replay-behavior.json).
It is a prerequisite/command plan, not a claim that the native runtime has run.
No additional worker or dataset transfer is launched by this file.

An official image exists despite the installation page's stale statement that
Docker installation is temporarily unavailable. The official CI publishes
`stanfordvl/behavior`; Docker Hub manifest and image configuration were checked
without pulling image layers. Use the immutable image
`docker.io/stanfordvl/behavior@sha256:3dda0241dae8c6afc5dc3b5d29f9d6e52035d96cc68e8117db37723e7fbb674e`
(`3.9.3-post1`, linux/amd64, 14,605,247,329 compressed layer bytes). Its OCI
revision label matches Git commit `bd049de3119acdcdf2334fe9e1ebe060fa20c108`.
The source installs Python 3.11, Isaac Sim 5.1.0 and CUDA 12.8. The plain `v3.9.3`
Git tag failed resolution during this check, while `v3.9.3-post1` resolved; do
not assume Docker tags and Git tags are interchangeable.
[Official image workflow](https://github.com/StanfordVL/BEHAVIOR-1K/blob/a8247a8cc1633fe1ca0cc66aa07243d46c64f155/.github/workflows/build-push-containers.yml),
[registry metadata](https://hub.docker.com/v2/repositories/stanfordvl/behavior/tags/3.9.3-post1),
[pinned installation script](https://github.com/StanfordVL/BEHAVIOR-1K/blob/bd049de3119acdcdf2334fe9e1ebe060fa20c108/setup.sh)

The native runtime needs a supported RTX GPU even when headless. Isaac Sim 5.1
lists RTX 4080 / 16 GB VRAM and 32 GB host RAM as its minimum; A100 and H100 lack
supported RT cores. A CPU-only worker cannot execute this replay. The older
BEHAVIOR installation page's RTX 2070 / 8 GB figures are weaker than the pinned
Isaac runtime's requirements. Validate the actual worker with NVIDIA's
compatibility checker before replay; simply seeing a CUDA device is insufficient.
[Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)

The three official prerequisite archives are pinned to Hugging Face asset
revision `9f0d57d465726976ed98138d3f8b8ca3e2186775`:

| Archive | Compressed bytes | Uncompressed bytes |
| --- | ---: | ---: |
| `behavior-1k-assets-3.9.0.zip` | 31,457,673,073 | 34,742,171,984 |
| `omnigibson-robot-assets-3.8.2.zip` | 641,004,353 | 2,927,238,404 |
| `2026-challenge-task-instances.zip` | 108,543,852 | 577,219,383 |

Sizes came from publisher LFS metadata and remote ZIP central directories;
no members were extracted locally. Keeping archives plus extracted files
requires 70,453,851,049 bytes, before image/cache/output overhead. Maintaining
this project's 50 GB reserve therefore requires more than 120,453,851,049 free
bytes before this complete asset fetch/extraction. There is no verified
per-scene dependency selector here yet; these three archives are substantially
smaller than acquiring the entire raw demonstration release.
[Asset archive listing](https://huggingface.co/datasets/behavior-1k/zipped-datasets/tree/9f0d57d465726976ed98138d3f8b8ca3e2186775)

The BEHAVIOR Data Bundle terms restrict the artwork/assets to noncommercial
academic use within OmniGibson and prohibit extracting or redistributing them.
The official downloader explicitly requests acceptance before obtaining the
asset key. This is a concrete restriction on the earlier proposed automatic
MuJoCo asset conversion: that conversion requires separate rights. Native
Reachy validation inside OmniGibson, or separately licensed corresponding assets,
would be different implementation routes; neither is implemented by this
recipe. Numeric source-reference extraction and asset export must not be
conflated. [Pinned asset terms and downloader](https://github.com/StanfordVL/BEHAVIOR-1K/blob/bd049de3119acdcdf2334fe9e1ebe060fa20c108/OmniGibson/omnigibson/utils/asset_utils.py)

After the GPU, terms and pinned assets are available, the persistent worker can
use the image's `/entrypoint.sh` with `sleep infinity` and later `kubectl exec`.
The replay command is recorded as an argument array in the config. It points
at the existing pilot through the layout expected by the official script;
source raw files remain immutable. Standard replay outputs do not automatically
include our task-object callback or named joint-state decoder. Those missing
exports and recorded-3.7-to-replay-3.9 compatibility remain explicit readiness
gates, not hidden behind a successful container launch.

The storage reference sibling `reachy-agent` was checked read-only with
`git ls-remote origin refs/heads/main`: both remote main and local HEAD remained
`eb39c2e0f133e8955acfb949513512eaae193db1` during this audit.
