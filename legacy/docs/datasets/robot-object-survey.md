# Robot–object datasets and simulation benchmarks

Access date: **2026-10-06**. This is an online source and schema investigation,
not a claim that the datasets below were downloaded, converted, or physically
validated with Reachy. No demonstration payloads were downloaded for this
survey. Public project documentation, repository source, GitHub metadata and
Hugging Face file metadata were inspected. The initial local catalog snapshot
contained 333 source/mirror entries; that number is not a count of independent
datasets or demonstrations and predates the object-inclusive scope cleanup.

## Findings that change the collection strategy

1. **Use native simulation state, not just policy-training exports.** robomimic,
   MimicGen, ManiSkill, LIBERO and RoboCasa already occur in the catalog. Their
   native releases can contain object state that an eight-dimensional robot
   observation in a LeRobot mirror omits. A source family and a particular
   mirror must receive separate eligibility decisions.
2. **The smallest verified second-family sample is ManiSkill PickCube teleop:**
   383,621 bytes plus 2,702 bytes of JSON. Native `env_states` tracks actors and
   articulations. robomimic Lift and MimicGen Stack are also modest downloads.
3. **RoboCasa365's official LeRobot export retains simulator extras.** The
   `extras` directory contains per-episode compressed MJCF and raw states;
   inspecting only Parquet feature names gives an incomplete answer.
4. **An object-centric task name does not establish tracked object state.**
   DROID, Bridge, FMB, standard ACT/ALOHA recordings and several LeRobot mirrors
   primarily expose robot state and images. They are useful manipulation data,
   but not ready for object-state-preserving contact retargeting.
5. **A simulator with object meshes is not automatically an object trajectory
   dataset.** RoboTwin, VLABench and BiGym are valuable generation candidates,
   but their inspected default recordings do not establish complete measured
   object trajectories without reconstruction or additional logging.
6. **Source task success is not Reachy grasp success.** In particular, RLBench's
   documented grasp mechanism reparents grasped objects. Such trajectories
   can be motion references but cannot certify a frictional Reachy grasp.

These are engineering conclusions from the per-source evidence below. For a
new Reachy physical benchmark, preserve object geometry, mass and friction;
change the Reachy grasp and timing; then run the object freely under contact.

## Coverage and priority

`A` means native state and scene support a concrete object-aware adapter;
`B` means promising but reconstruction or schema work remains; `C` means the
inspected public representation lacks the required temporal object state;
`D` means an object/goal asset benchmark rather than a demonstration corpus.
None of these grades means a Reachy physical pass.

| Family | Initial catalog coverage | Grade | Useful new coverage | Main qualification |
| --- | --- | --- | --- | --- |
| robomimic beyond Can | Already cataloged and partly acquired | A | Cube lift, nut insertion, tool hanging, bimanual transport | Reuse native HDF5/MJCF; task-specific object schema |
| MimicGen | Already cataloged and partly acquired | A | Stacking, threading, coffee, multi-piece assembly | Generated trajectories share human seed lineage |
| ManiSkill 3 | Already cataloged and selected scope acquired | A | Pick/stack, peg insertion, articulated objects, dual robot | SAPIEN to MuJoCo asset/dynamics conversion required |
| RoboCasa / RoboCasa365 | Already cataloged; unified mirror acquired | A/B | Kitchen fixtures, appliance interaction, long sequences | Official `extras` needed; mirrors may omit them |
| LIBERO | Already cataloged; robot-only mirrors acquired | A/B | Object, spatial and goal variation; long tasks | Native HDF5 `states` and MJCF needed |
| CALVIN | Already cataloged; debug state projection acquired | A/B | Drawer, slider, switch, block manipulation | Scene observation semantics and PyBullet assets |
| D4RL Kitchen / Minari | Already cataloged; selected scope acquired | A/B | Articulated kitchen objects and sequential goals | Old/new environment revisions are not identical |
| FurnitureBench | Already cataloged; mirrors acquired | B/C | Real assembly, insertion, screw/leg alignment | Environment has part poses; released files must be checked |
| DROID | Already cataloged; OXE projection acquired | C | Real household diversity and long-tail objects | No object 6D tracks in documented RLDS schema |
| OXE / BridgeData V2 | Already cataloged in many forms | C or source-specific | Cross-embodiment real manipulation | Mixture is not one uniform object schema |
| FMB | Already cataloged; LeRobot projection acquired | C | Real regrasp, fixture use, force/torque, insertion | CAD and object IDs are not object pose tracks |
| ACT / ALOHA / Mobile ALOHA | Already cataloged and mirrored | C/B | Bimanual transfer, insertion, mobile manipulation | Default ACT simulation writer omits object state |
| DexMimicGen | **Not cataloged** | A, license unresolved | Bimanual threading, assembly, drawer/box cleanup | Actual state available; conflicting published licenses |
| BiGym | **Not cataloged** | B | Mobile bimanual cabinet, dishes, kitchen cleanup | Lightweight actions/seed require replay and fresh logs |
| RLBench | **Not cataloged** | B | Many rigid/articulated tasks and variations | CoppeliaSim conversion; artificial grasp parenting |
| RoboTwin 2.0 | **Not cataloged** | B | Bimanual handover, tools, many object embodiments | Default object trajectory logging not verified |
| VLABench | **Not cataloged** | B/C | Language reasoning, object selection, long sequences | Entity IDs and initial scene do not track moving objects |
| COLOSSEUM V1 / V2 | **Not cataloged** | B | Perturbation/generalization and dual-arm tasks | V1 is RLBench-based; V2 is ManiSkill-based |
| Open6DOR / Open6DOR V2 | **Not cataloged** | D | Object orientation and goal-pose variation | Task/goal benchmark, not full executed motion corpus |

## Verified small download candidates

Metadata was read with the public Hugging Face dataset API using `?blobs=true`.
Sizes below are complete remote file sizes, not measured transfer volumes or
state-only projection sizes. SHA-256 values are **publisher LFS metadata**;
payload verification remains necessary at fetch time. MB uses decimal units.

| Candidate | File | Bytes | Dataset revision |
| --- | --- | ---: | --- |
| robomimic Lift PH | `v1.5/lift/ph/low_dim_v15.hdf5` | 21,084,088 | `74fa018461f479cd9fd15b924a16103012096203` |
| robomimic Square PH | `v1.5/square/ph/low_dim_v15.hdf5` | 51,145,256 | same robomimic revision |
| robomimic Tool Hang PH | `v1.5/tool_hang/ph/low_dim_v15.hdf5` | 198,601,752 | same robomimic revision |
| robomimic Transport PH | `v1.5/transport/ph/low_dim_v15.hdf5` | 303,187,328 | same robomimic revision |
| MimicGen Stack source | `source/stack.hdf5` | 11,753,280 | `33016f8a62c02334f929f2913af8fdd2a8a129e1` |
| MimicGen Threading source | `source/threading.hdf5` | 18,553,470 | same MimicGen revision |
| MimicGen Coffee source | `source/coffee.hdf5` | 20,019,192 | same MimicGen revision |
| ManiSkill PickCube teleop | `demos/PickCube-v1/teleop/trajectory.h5` | 383,621 | `d674485bbffdd533914e52d272fdda34c0515608` |
| ManiSkill PickCube metadata | `demos/PickCube-v1/teleop/trajectory.json` | 2,702 | same ManiSkill revision |
| ManiSkill StackCube motion planning | `demos/StackCube-v1/motionplanning/trajectory.h5` | 38,745,164 | same ManiSkill revision |

Exact publisher hashes:

```text
robomimic Lift       2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540
robomimic Square     45d8cabb6d57a4c03e839aa5e4b3e58fb60fe8bd20e1951fec563a2abfd14951
robomimic Tool Hang  c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4
robomimic Transport  260515618f4c8b660e54171497ecceb08c7c6463691a9ce862b158bcacd950b4
MimicGen Stack      00331a889de2fa610be9b5c4a0e066f4d387298e0b117181930f166c44076418
MimicGen Threading  4f7d90c819c966e0d3e3b5f1046efbafec3793cc2074d8dd220e9d2355feff74
MimicGen Coffee     d36b22adf480432910bf789bb87f9798c9d785da53619742b9576049c441bc8b
ManiSkill PickCube  0e6285c5e7a9293fbb1c10590a1be4fc115792cb816e00b37e9914369a0c3c94
ManiSkill StackCube 61d07df7b52f303d2088e977df5f863be7d2274348ff05791f3a815359815523
```

Sources: [robomimic pinned files](https://huggingface.co/datasets/robomimic/robomimic_datasets/tree/74fa018461f479cd9fd15b924a16103012096203/v1.5),
[MimicGen pinned source files](https://huggingface.co/datasets/amandlek/mimicgen_datasets/tree/33016f8a62c02334f929f2913af8fdd2a8a129e1/source),
[ManiSkill pinned teleop files](https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations/tree/d674485bbffdd533914e52d272fdda34c0515608/demos/PickCube-v1/teleop).

Select one complete episode from each family before expanding. Keep failed
retargeting and physical attempts. Do not count image/low-dimensional exports,
PH/MimicGen seed reuse, or different control-mode renderings as new independent
human demonstrations. Asset downloads are additional to the listed sizes.

## Per-family evidence and implementation implications

### 1. robomimic beyond Can

The official distribution covers Lift, Can, Square, Transport and Tool Hang,
with proficient-human, multi-human and machine-generated variants where
available. Native HDF5 retains actions, simulator state and environment/model
metadata. Low-dimensional observations are task-specific; the Can 14-value
object interpretation must not be copied to Square or Tool Hang. Match
robosuite `v1.5.1` assets and controller semantics for the current release.
The original `offline_study` generation and modern v1.5 conversion are related
versions, not new demonstrations. The public HF card declares MIT and is
ungated. The low-dimensional files above are an inexpensive path to cube,
nut/peg, tool and two-arm coverage. Source velocities/actions should guide
timing, while object trajectories remain references during free simulation.
See [official dataset documentation](https://github.com/ARISE-Initiative/robomimic/blob/master/docs/datasets/robomimic_v0.1.md)
and [dataset contents](https://robomimic.github.io/docs/tutorials/dataset_contents.html).

### 2. MimicGen

MimicGen extends robosuite manipulation with object-relative subtask generation:
stacking, threading, nut assembly, coffee, cleanup and multi-piece assembly.
The official workflow prepares source demonstrations with extra datagen
information. Native simulator states/models allow object-body poses to be
reconstructed rather than inferred from images. Human source data, generated
rollouts and their source-demo lineage must remain distinct; generated
variations should be grouped with their seeds for leakage-sensitive splits.
The public dataset card at the pinned revision declares CC BY 4.0, whereas
the code has the NVIDIA source license. Geometry and robot tool transforms
must be resolved per task. Source Stack is the cheapest useful rigid-object
extension here. See [official quickstart](https://github.com/NVlabs/mimicgen/blob/main/docs/tutorials/getting_started.md),
[code and license](https://github.com/NVlabs/mimicgen), and
[data release](https://huggingface.co/datasets/amandlek/mimicgen_datasets).

### 3. ManiSkill 3

Each trajectory has actions and `env_states` with one more state than actions;
actor and articulation arrays are stored as a nested dictionary. The matching
JSON identifies environment, reset arguments, control mode and source type.
This supports explicit object motion, including articulated parts. CPU/GPU
replay is not automatically deterministic, and even stored state may omit
details needed for exact continuation. Use the original native state as the
reference, not an action-replayed approximation silently substituted for it.
The public demonstration card is Apache-2.0; imported asset families still
need their own attribution. PickCube teleop is exceptionally small. StackCube,
PegInsertion and TwoRobotPickCube add qualitatively different contacts.
Physics transfer requires preserving SAPIEN shapes, scale, poses and material
assumptions in MuJoCo. Quaternion/body ordering must come from the pinned
state implementation. See [format and replay caveats](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html)
and [demonstration release](https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations).

### 4. RoboCasa / RoboCasa365

The current official release separates human pretraining, synthetic pretraining
and held-out target data. Tasks cover objects plus articulated kitchen fixtures,
appliances and longer skill sequences. RoboCasa365's LeRobot representation
has `extras/dataset_meta.json` and per-episode `ep_meta.json`, `model.xml.gz`
and `states.npz`. These are the route to reconstructing actual object and
fixture state; ordinary robot Parquet alone is insufficient. Downloads can be
selected by task, such as `PickPlaceCounterToCabinet`. Full transfer size was
not verified, so begin by enumerating a single episode's extras and assets.
Code is MIT; assets/datasets are declared CC BY 4.0. The inspected code revision
is `456174f62b89b8fca99eaaf33949c29fec9cfc2a`. Historical v0.2 HDF5 and v1.x
LeRobot data require different adapters. See [official structure](https://robocasa.ai/docs/build/html/datasets/using_datasets.html),
[coverage/splits](https://robocasa.ai/docs/build/html/datasets/datasets_overview.html),
and [release/license](https://github.com/robocasa/robocasa).

### 5. LIBERO

LIBERO has object, spatial, goal and long-task suites with controlled transfer
conditions. Its native creation script stores `states`, actions, robot state,
per-demo `model_file`, and BDDL task information. Those state/model pairs are
valuable for explicit object reconstruction. The cataloged LeRobot variants
mostly expose robot pose/gripper features, so their acquisition does not
establish possession of native object tracks. Use native task HDF5 and matching
assets/BDDL, and preserve initial warm-up trimming and state/action alignment.
The repository is MIT. A task-specific download is preferable to the complete
suite; exact current bytes were not verified. LIBERO-90/10, no-op-filtered
exports and image/subtask conversions overlap; task partition and original
episode IDs must survive conversion. See [official benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO)
and [native dataset writer](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/scripts/create_dataset.py).

### 6. CALVIN

CALVIN records `robot_obs` and `scene_obs`, with explicit scene variables for
slider, drawer, button, switch and block poses. This is object-inclusive play
data, despite not being a retargeting dataset. It introduces articulation and
multi-stage sequences beyond isolated pick/place. Official release sizes are
166 GB (D→D), 517 GB (ABC→D), 656 GB (ABCD→D), and 1.3 GB for debugging. The
initial repo catalog already records a 4.18 MB state-only debug projection;
reuse that scope before fetching more. Decode the published object ordering,
units and orientation convention and preserve scene configuration. PyBullet
assets and constraints need a separate MuJoCo conversion. Repository code is
MIT; a separate dataset license was not established by the inspected dataset
README. Splits overlap in environments and source play, so do not add their
counts. See [official dataset schema and downloads](https://github.com/mees/calvin/blob/main/dataset/README.md)
and [source repository](https://github.com/mees/calvin).

### 7. D4RL Kitchen and Minari

Kitchen adds articulated burners, microwave/cabinet controls and kettle-style
manipulation. Original observations encode robot and scene configuration, but
reconstructing Cartesian object poses requires the exact kitchen model and
observation-index mapping. The catalog already has a selected kitchen payload
of 39,675,763 bytes; that does not establish an adapter or physical pass.
D4RL now points users to Gymnasium-Robotics and Minari, whose revised
environments/data must not silently replace historical trajectories. D4RL
declares CC BY 4.0 data unless otherwise stated, Apache-2.0 code. Treat complete,
partial and mixed data as different source selections with potentially shared
underlying trajectories. Inspect terminal/time-out segmentation and full state
availability before using a clipped offline-RL transition collection as
temporal demonstrations. See [official migration and licensing](https://github.com/Farama-Foundation/D4RL)
and [kitchen implementation](https://github.com/Farama-Foundation/D4RL/tree/master/d4rl/kitchen).

### 8. FurnitureBench

FurnitureBench provides real assembly of standardized printable parts, including
long insertion/alignment sequences. The real environment API exposes
`parts_poses`, but the documented demonstration-file schema lists images and
robot state without promising those part tracks in every released pickle.
Therefore AprilTag-equipped objects and CAD availability must not be mistaken
for downloaded temporal object poses. The initial catalog's LeRobot mirrors
are also robot-state focused. Inspect a native episode before upgrading this
source to object-inclusive eligibility. Public Google Drive exposes individual
pickles and per-furniture archives; documented raw totals are about 1.18 TB.
Code is MIT; separate data terms were not established in the inspected dataset
page. World coordinates use a base AprilTag, so calibration/occlusion validity
is necessary. See [dataset schema/size](https://clvrai.github.io/furniture-bench/docs/tutorials/dataset.html),
[environment API](https://clvrai.github.io/furniture-bench/docs/tutorials/furniture_bench.html),
and [base frame](https://clvrai.github.io/furniture-bench/docs/getting_started/installing_furniture_bench.html).

### 9. DROID

DROID is real robot-object manipulation, but its documented RLDS schema has
robot joints, Cartesian/gripper states, images, actions and language, not
per-object 6D tracks or collision meshes. Raw stereo and improved camera
calibration can support a future reconstruction pipeline; its output would
be derived geometry/pose estimates, with uncertainty, not source ground truth.
The official debugging subset is 100 episodes / approximately 2 GB; the full
RLDS release is 1.7 TB and stereo raw release 8.7 TB. Public bucket access is
documented; separate dataset license terms were not verified on the inspected
pages. The current catalog's OXE projection does not solve missing object
state. Keep this as a perception/manipulation-training candidate, outside the
strict object-state retarget set until reconstructed. See [official schema](https://droid-dataset.github.io/droid/the-droid-dataset)
and [calibration updates](https://droid-dataset.github.io/).

### 10. Open X-Embodiment and BridgeData V2

OXE aggregates heterogeneous source datasets; it is not an independent corpus
to add on top of DROID, Bridge, FMB or ManiSkill counts. Licensing and semantic
conventions remain source-specific. In particular, standard Bridge robot-state,
action, RGB and language data does not by itself furnish object 6D tracks and
matching meshes. Such data is useful for real-world task coverage and future
vision-derived object estimates, but should not pass an object-state gate just
because language mentions a cup or drawer. Select a native source and inspect
its `features.json`, action semantics, timing and source IDs. RLDS/LeRobot
transformations can drop fields; preserve original IDs across versions. Small
episode streaming is possible where provider/shard layout permits, but the
inspected pages did not establish a universal small download size or license.
See [official OXE registry](https://github.com/google-deepmind/open_x_embodiment)
and [BridgeData V2](https://rail-berkeley.github.io/bridgedata/).

### 11. Functional Manipulation Benchmark (FMB)

FMB offers grasp/reorient/regrasp/insert sequences on printable parts and
fixtures. Its native schema documents TCP pose/velocity, joint state,
end-effector-frame force/torque, binary gripper, primitive labels, RGB-D and
object IDs/attributes. **Object IDs, dimensions and coarse initial orientation
are not temporal object pose tracks.** CAD is useful for reconstruction but
does not establish measured contacts. TCP pose is in robot base coordinates;
force/torque is in the tool frame. Public native downloads are large: the
single-object archive is labeled 545 GB and three assembly archives 86, 77
and 70 GB. Data is explicitly CC BY 4.0. Prefer a state projection with a
verified object-pose extension if available; do not fetch the complete archive
for current contact validation. See [official task coverage](https://functional-manipulation-benchmark.github.io/)
and [schema, downloads and license](https://functional-manipulation-benchmark.github.io/dataset/index.html).

### 12. ACT / ALOHA / Mobile ALOHA

These families are relevant to Reachy's two arms and to handover/insertion,
but a full object-inclusive trajectory must be checked separately from arm
logs. The inspected ACT simulation recorder internally reads the initial
`env_state` to synchronize two simulation passes, then writes robot `qpos`,
`qvel`, images and action arrays; it does not serialize the moving object's
complete trajectory. Reconstructing a simulation rollout and recording object
state is possible, but must be described as a new simulation-derived sample.
Real ALOHA object tracking would require perception/calibration work. Existing
LeRobot variants in the catalog do not supply this missing track. Code/data
licenses and download sizes must be resolved for the exact selected release;
no universal ALOHA-family grant is assumed. See [ACT source](https://github.com/tonyzhaozh/act)
and [the actual episode writer](https://github.com/tonyzhaozh/act/blob/main/record_sim_episodes.py).

### 13. DexMimicGen — new family

This is a strong bimanual extension: threading, multi-piece assembly, can
sorting and drawer/box cleanup. Its official playback reads stored flattened
MuJoCo states and model metadata, with an explicitly separate action-playback
mode. Thus native data can supply object trajectories and both hand targets;
dexterous-hand contacts still require redesign for Reachy's gripper.
At HF revision `181967e10c6277a653e7c9761f3978312af3646a`, the smallest listed
HDF5 is `generated/two_arm_threading.hdf5`, 3,913,396,274 bytes, publisher
SHA-256 `83ffc06f7bf2d11e83295252d7ec2b1ae50e93c96b12d0739078a371f3875b08`.
Range-based state extraction is preferable to whole image-bearing files.
There is a material license conflict: the code README says datasets are
CC BY 4.0; the HF card says CC BY-NC-SA 4.0; its bundled LICENSE is NVIDIA
research/noncommercial. Record the conflict rather than assigning permissive
terms. See [official code/release](https://github.com/NVlabs/dexmimicgen),
[state playback](https://github.com/NVlabs/dexmimicgen/blob/main/scripts/playback_datasets.py),
[data card](https://huggingface.co/datasets/MimicGen/dexmimicgen_datasets), and
[bundled license](https://huggingface.co/datasets/MimicGen/dexmimicgen_datasets/blob/main/LICENSE).

### 14. BiGym — new family

BiGym provides human-driven mobile bimanual household tasks with MuJoCo assets,
making cabinet, dish and kitchen-cleanup tasks relevant to Reachy. Its
lightweight demonstration representation stores action/seed metadata and can
recreate fuller observations. That is not evidence that the distributed
lightweight files contain complete object `qpos` at every frame. Replay with
the pinned version, preserve replay failures, then explicitly log all moving
objects before retargeting. Official docs warn that not every demo succeeds
on replay. The inspected revision is
`14beb30318ad14c5d6723175c2ee2281129792af`. Code is Apache-2.0; assets have
mixed CC0/CC BY/CC BY-NC attribution. DemoStore downloads via releases, so
calling it must occur only inside explicit fetch, never on import/tests.
Exact task-level archive sizes and dataset-specific license were not verified.
See [official repository](https://github.com/NeuracoreAI/bigym),
[demo format](https://github.com/NeuracoreAI/bigym/blob/master/demonstrations/demo.py),
and [download behavior](https://github.com/NeuracoreAI/bigym/blob/master/demonstrations/demo_store.py).

### 15. RLBench — new family

RLBench offers diverse object and articulation tasks in CoppeliaSim. Optional
`task_low_dim_state` includes object poses and joint values, with ordering
defined by the initial scene objects. Configure and retain it explicitly;
ordinary image/proprioception demonstrations do not guarantee it is enabled.
Scene assets need conversion and mapping to named bodies. More critically,
RLBench documents that grasped objects become children of the gripper, keeping
their relative pose fixed. Treat these as kinematic references and rebuild
free-object contact validation in MuJoCo. Source success cannot measure Reachy
slip. The current LICENSE limits software use to noncommercial/internal or
academic research; exact dataset redistribution terms remain unverified.
Generate a small object task with logged states rather than downloading a
large image archive. See [task-state code](https://github.com/stepjam/RLBench/blob/master/rlbench/backend/task.py),
[grasp mechanism](https://github.com/stepjam/RLBench/blob/master/tutorials/complex_task.md),
and [license](https://github.com/stepjam/RLBench/blob/master/LICENSE).

### 16. RoboTwin 2.0 — new family

RoboTwin adds broad bimanual manipulation, multiple robot embodiments,
articulated assets and annotated grasp/functional points. Those points are
especially useful for selecting a more stable Reachy grasp without modifying
the object. However, the inspected default observation writer records cameras,
robot joints/end poses and optional point clouds, not all object trajectories.
The Actor API exposes world pose and articulation state, so add explicit
logging or verify a richer native file before admission. Code and HF card
declare MIT; imported assets still need their own provenance.
HF revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186` is ungated and organized
by task/robot. `dataset/click_bell/franka_clean_50.zip` is 73,473,679 bytes;
its internal schema was not downloaded/validated. Avoid equating randomized,
cross-robot or reformatted versions with independent human demos. See
[official release](https://github.com/RoboTwin-Platform/RoboTwin),
[pose/contact-point API](https://robotwin-platform.github.io/doc/usage/API.html),
[asset transforms](https://robotwin-platform.github.io/doc/usage/object_marking/model_data_info.html),
and [dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0).

### 17. VLABench — new family

VLABench expands task/language reasoning and scene/object variety. The official
writer stores robot-frame waypoints, entity names, target entities, instruction
and an initial `episode_config`. Its inspected environment observation includes
robot state, cameras and point clouds; these fields do not establish per-frame
moving-object tracks. Native MuJoCo scenes make additional logging feasible.
Do not treat an initial scene snapshot as the object's state throughout grasp.
The inspected code revision is `cf588fe60c0c7282174fe979f5913170cfe69017` (MIT).
HF primitive data revision `765ca79fcb078f3b272d49580069ce17ef107134` is public,
MIT-tagged and split into approximately 10.7 GB compressed pieces. The final
1.286 GB piece is not an independently extractable small dataset. Generate one
task with object logging or seek an explicitly state-complete export.
See [official repository](https://github.com/OpenMOSS/VLABench),
[writer](https://github.com/OpenMOSS/VLABench/blob/main/scripts/trajectory_generation.py),
and [primitive data](https://huggingface.co/datasets/VLABench/vlabench_primitive_ft_dataset).

### 18. COLOSSEUM V1 / V2 — new benchmark family

The original COLOSSEUM adds controlled perturbations to RLBench, inheriting its
state/scene and artificial-grasp caveats. V2 is a different ManiSkill-based
implementation with 28 tasks across single-arm and bimanual manipulation:
handover, pot/tray lifting, threading, cabinet/drawer interaction and tool use.
Its public repository provides HDF5/JSON dataset links and motion-planning
generation. Start with one generated trajectory, retain actor/articulation
state, and verify native file schema before claiming completeness. V2 code
shows Apache-2.0 plus third-party license files; data-specific terms and remote
sizes were not verified. For this project, keep object dimensions fixed during
retargeting; size perturbations are new declared scenarios, not a method for
making a failing grasp pass. V1/V2 are related benchmarks, not interchangeable
releases. See [V1 design](https://robot-colosseum.readthedocs.io/en/latest/overview.html),
[V2 task coverage](https://colosseum-v2.github.io/), and
[V2 code/data links](https://github.com/jstmn/ColosseumV2).

### 19. Open6DOR and Open6DOR V2 — new supporting benchmark

Open6DOR supplies diverse object assets and language-conditioned target
position/orientation tasks. This helps test whether a chosen Reachy grasp can
reach a desired object orientation, but it does not provide complete robot–
object temporal demonstrations by itself. Use it as a scenario/goal generator
and evaluate newly planned free-object rollouts separately. The original code
is Apache-2.0; imported Objaverse/YCB assets require original licenses and scale
metadata. The original README's robot-interaction command is explicitly not
available, so do not describe it as a ready dynamic benchmark. Open6DOR V2
changes/filter tasks and is related, not an independent collection of human
demonstrations. Exact archive sizes were not verified. See
[official original benchmark](https://github.com/Selina2023/Open6DOR) and
[V2 release description](https://github.com/qizekun/SoFar).

## Recommended experiment matrix

The following is a proposed sequence of **new checks**, not a completed pass
table. Select at least two source episodes per cell before estimating any
generalization rate; choose hold-out seeds before grasp tuning.

| Stage | Source/task | Contact challenge | Required evidence |
| --- | --- | --- | --- |
| 1 | robomimic Lift + Can | Box versus cylinder grasp and release | Full object pose, unchanged source colliders, slip/drift |
| 1 | MimicGen Stack | Placement on a movable support | Both object bodies tracked, post-release stability |
| 1 | ManiSkill PickCube teleop | Independent simulator/source family | Actor pose/scale alignment and time/action conventions |
| 2 | robomimic Square / MimicGen Threading | Tight insertion and orientation | Grasp-relative error, contact normals, insertion success |
| 2 | CALVIN drawer/slider | Articulated environment interaction | Joint axis/limits, handle pose, joint progress |
| 2 | robomimic Transport | Two-arm transfer/carry | Both hands, independent contacts, object-relative drift |
| 3 | RoboCasa native extras | Clutter and long sequences | Scene reconstruction, fixture state, phase success |
| 3 | DexMimicGen / BiGym / RoboTwin | Bimanual/mobile diversity | Licensing resolved; complete object logs and asset mapping |

Report accessible, acquired, schema-inspected, normalized, kinematic-pass and
physical-pass separately for every episode. A replay of saved object poses is
useful visualization, but a physical pass requires state to evolve from
contact and gravity after reset. No object weld, repeated state assignment,
teleportation or success-only reporting is admissible. Record source URLs,
exact revisions, fetched payload hashes, object scales, frame transforms,
derived labels, missing states, controller changes and simulation assumptions.
