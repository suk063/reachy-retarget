# Mobile human-object data for Reachy retargeting

Checked: **2026-10-06**. This extends the earlier
[19-release human-object survey](human-object-survey.md) with mobile carrying,
long interactions, and physically informative data. It also rechecks existing
families that are frequently renamed or confused with new captures. Findings
come from primary project pages, papers, author code, and public file metadata.
The [evidence ledger](mobile-methods-evidence.json) records selected responses,
hashes, revisions, and failed checks. No motion/image payload was acquired or
retargeted by this research task.

**Recommended order:** use existing HUMOTO/ParaHome and rigid OMOMO examples to
extend local validation; investigate MeLLO for measured forces during locomotion;
inspect FORCE for carrying/pushing with resistance variation; seek HIMO access
for explicit multi-object interactions. KIT's object-aware recordings are another
strong option after account and terms checks. These choices address different
gaps and are not interchangeable benchmarks.

## Inclusion criteria and status vocabulary

For direct retargeting, require time-aligned human hand/body motion and a moving
object's pose in a shared metric frame, plus matching geometry. World-space
locomotion and continuous approach/carry/place intervals are additionally needed
for a mobile-manipulation claim. A scene mesh, action caption, camera trajectory,
or object category by itself does not meet this contract.

In the tables, **public files** means a release/file listing was inspected;
**application** means a form/account is required; **paper only** means no usable
payload route was verified. None means downloaded, adapter-supported,
Reachy-feasible, or physics-validated. Missing mass, friction, contact forces, or
collision geometry remain missing even when an animation looks plausible.

| Family newly investigated here | Mobile/task value | Moving-object + human evidence | Access status on checked date | Direct-retargeting assessment |
|---|---|---|---|---|
| MeLLO | Walking while carrying, pushing, pulling; large objects | Human/object markers, force/torque and object inertial measurements | Public repository; small per-trial files | Strong physics evidence; rigid-pose/mesh adapter work remains |
| FORCE | Carry/push/pull with varying resistance | SMPL motion, object pose, scanned meshes | Public files; dataset license not found in inspected tree | Strong candidate after terms, total mass and frame checks |
| KIT Bimanual + Extended | Multi-step household manipulation | Body/object motion, object models; richer sensors by capture | Account required; individual downloads | Strong object-aware subset; verify actual base travel |
| HIMO | One person interacting with two/three objects | Full body, object rotations/translations, meshes | Application form; custom research terms | Strong multi-object candidate; mobility must be measured per sequence |
| InterVLA | Long room-scale first-person tasks | Full-body and object motion described in paper | Project code link unavailable; download not verified | Good future mobile source, presently blocked on release |
| iReplica: EgoHOI + H-Contact | Long indoor interactions, doors/furniture | Wearable body capture, contacts, scene/object representations | Project download routes; custom research terms | Reconstruction study; distinguish inferred object motion from measured truth |
| IMHD² | Dynamic manipulation and whole-body actions | Fitted SMPL-H/object motion, object IMU, scanned meshes | Public HF listing; custom research terms | Useful dynamic stress cases, with discontinuous annotation intervals |
| ACE-Data-0 | Planned room-scale chained tasks and tactile interaction | Card describes body/hands, object pose/mesh, some tactile streams | Card says payload not yet released; storage metadata conflicts | Watchlist pending shard and schema verification |
| Nymeria/NymeriaPlus | Extensive locomotion and scene context | Body/world motion, **static** object geometry/boxes | Public release and downloader; dataset-specific terms | Exclude from direct moving-object track benchmark |
| Aria Everyday Activities | Shared-world multi-person daily activity | Device/world motion, gaze, point clouds | Public explorer/download tools | No verified body plus dynamic object state contract |
| HD-EPIC | Long kitchen tasks, object itineraries, navigation | Camera poses; pickup/place object locations | Public release, CC BY-NC 4.0 | Endpoint/context data, not continuous object SE(3) truth |
| EgoMAN | Derived egocentric hand trajectories and task context | Camera-relative hand paths from existing datasets | Preparation code; processed release stated unavailable | Derivative and object-state incomplete; not a new capture family |

The twelve rows are **research profiles, not twelve independently usable new
datasets**. KIT versions, Nymeria versions, and iReplica's named collections
require lineage tracking. EgoMAN is deliberately included as a negative/derived
case. Existing catalog families are discussed separately below.

## New candidate details

### MeLLO: strongest new lead for locomotion plus measured interaction

The Utah Manipulation and Locomotion of Large Objects library covers a box,
luggage, briefcase, walker, shopping cart, wheelbarrow, and door. Human/object
motion capture is accompanied by interaction-force and inertial measurements.
Walking while carrying or pushing is central to the collection, rather than an
incidental movement between tabletop actions. This makes it particularly useful
for checking whether contact timing and load transfer remain plausible during
transport. Large or assisted objects will often exceed Reachy's fixed grasp or
payload capability; such cases should remain explicit exclusions.
[Author paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11939698/)

The inspected [repository](https://github.com/nluttmer1/MeLLO-Data-Library) contains
per-subject/object/task directories, cut Vicon data, processing/transform scripts,
and AddBiomechanics/OpenSim outputs. Its
[license file](https://github.com/nluttmer1/MeLLO-Data-Library/blob/ea82019e6093b85311b2c5e4865760a5d0945ce6/LICENSE.txt)
identifies CC BY 4.0. A single tracked CSV is listed at 506,651 bytes:
[`Subject 1 - Box Walk 1.csv`](https://github.com/nluttmer1/MeLLO-Data-Library/blob/ea82019e6093b85311b2c5e4865760a5d0945ce6/Data%20Library/Subject%201/Box/Walk/Cut%20VICON%20Data/Subject%201%20-%20Box%20Walk%201.csv).
Some C3D entries in the Git tree are small Git-LFS pointer files, not the true
payload size. No OBJ/PLY/STL object meshes were found in the inspected tree.

**Adapter gaps:** recover rigid object transforms from calibrated marker groups;
verify sample clocks, units, gravity direction, and load-cell extrinsics; retain
measured versus inverse-kinematics-derived channels separately. Exact wrist
orientation, finger articulation, complete collision meshes, and object inertial
parameters were not established by this inspection. Do not pair a cut trial with
an unaligned whole-session force stream. Whole sessions, cut trials, and OpenSim
derivatives must share one capture lineage.

### FORCE: resistance-aware carrying, with important release discrepancies

The paper describes human carrying, pushing, and pulling across eight objects
with removable weights. Human motion combines inertial tracking and RGB-D
fitting; object geometry is scanned. Its force encoding is **derived** from
motion and physical assumptions, not a recorded per-contact force stream.
Crucially, resistance labels refer to added weights and exclude each object's
own mass. The paper's dataset section gives 450 sequences; the landing page's
older counts differ. [Paper](https://arxiv.org/html/2403.11237v2),
[project](https://virtualhumans.mpi-inf.mpg.de/force/)

Public [release code and data](https://github.com/xz6014/FORCE_dataset) now exist
despite the landing page's forthcoming notice. At inspected revision
`695c8b8616f0915dd61658862751e40ec658b81c`, the tree contains 398 motion NPZs,
six PLY object meshes, and annotations for weights, hands, objects, and sequences.
No license file was present in that tree. The
[visualizer](https://github.com/xz6014/FORCE_dataset/blob/695c8b8616f0915dd61658862751e40ec658b81c/visualize_force_dataset.py)
reads `smpl_pose`, `smpl_trans`, `obj_pose`, and `obj_trans`; object orientation is
converted from axis-angle. It centers the mesh at the bounding-box midpoint
before applying the object transform. Omitting this transform changes the grasp
surface's location.

**Adapter gaps:** confirm data-use terms, native frame rate, units/world axes,
per-object baseline mass plus added load, and collision suitability. Preserve the
physical properties of the chosen trial; do not convert added weight into total
mass or copy the method's shape augmentation into an unchanged-object benchmark.
SMPL body motion does not by itself establish measured finger poses. Public
sequence counts and mesh coverage should be reconciled before reporting coverage.

### KIT Bimanual and Extended KIT Bimanual: useful object-aware subsets

The original dataset combines bimanual household actions with tracked body and
object motion and richer modalities such as hand sensing. The extension adds
90 longer annotated sequences from six individuals. These are versions of one
family; an extended archive must not be summed blindly with every original or
processed copy. [Original paper](https://h2t.iar.kit.edu/pdf/KrebsMeixner2021.pdf),
[extended release](https://motion-database.humanoids.kit.edu/details/datasets/3521/?listpage=1)

The official FAQ documents 100 Hz MMM root/joint/object trajectories, raw C3D,
and Blender/Simox object models. Files can be acquired per motion or object after
free account registration. Its programmatic API is currently disabled. Additional
force/IMU channels depend on the experiment. Importantly, the database's imported
CMU recordings lack object information; they must not enter an object-aware
retargeting collection merely because the parent database has object models.
[KIT FAQ](https://motion-database.humanoids.kit.edu/faq/)

**Adapter gaps:** identify the exact object-aware recording, read its terms and
MMM units/transforms, match the object model, and preserve simultaneous streams.
The FAQ's GPL statement concerns MMM software, not a blanket dataset license.
Long sequences do not prove room-scale navigation; compute root travel before
labeling a trial mobile. An account was not created and no protected file was
requested in this survey.

### HIMO: explicit multi-object trajectories

HIMO is designed around full-body interaction with multiple objects. Its project
reports approximately 3.3 thousand sequences and 4.08 million frames. The official
release separates two-object and three-object processed data and requires a
download application. [Project](https://lvxintao.github.io/himo/),
[repository](https://github.com/LvXinTao/HIMO_dataset)

Documented folders include skeleton joints, SMPL-X parameters, per-object poses,
OBJ meshes, text, and temporal segments. The inspected
[skeleton viewer](https://github.com/LvXinTao/HIMO_dataset/blob/c6d6de8f3d94603c9bbacf967d622149c0667958/visualize/skel_viewer.py)
uses object-keyed `transl` and `rot` arrays, with rotations represented as matrices.
Its 30 fps playback setting is evidence about the viewer, not sufficient proof
of every raw stream's native rate. The
[license](https://github.com/LvXinTao/HIMO_dataset/blob/c6d6de8f3d94603c9bbacf967d622149c0667958/LICENSE)
contains both CC BY-NC-SA annotation language and additional restrictive research
terms; preserve the actual agreement rather than assigning a simple permissive
tag.

**Adapter gaps:** verify coordinate axes, units, wrist-orientation extraction,
object ordering and mesh origins from a licensed sample. Keep object IDs stable
across segments. Do not count whole sequences and their two-/three-object
processed windows as independent captures. Measure locomotion per sequence;
multi-object interaction is already valuable without a mobile claim.

### InterVLA: very relevant mobile evidence, release unavailable at inspection

The ICCV 2025 paper describes 519 long scripts segmented into 3,906 clips, with
11.4 hours of recording in an 8.5 by 5.4 meter area. It includes walking and
navigation, synchronized 30 Hz streams, full-body motion, scanned objects, and
object translation/rotation. Its six-dimensional rotation representation is a
rotation encoding, not six additional pose coordinates. Multiple views and
short clips are not independent demonstrations.
[Publisher paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_Perceiving_and_Acting_in_First-Person_A_Dataset_and_Benchmark_for_ICCV_2025_paper.pdf)

The [project page](https://liangxuy.github.io/InterVLA/) still labels code as
forthcoming; the linked repository returned unavailable during the web check.
An API retry later hit a GitHub rate limit, which is recorded separately and
does not independently establish absence. No usable archive, release license,
small-sample route, or exact adapter schema was verified. Hand articulation and
dynamic material parameters also remain unverified. Treat this as a release
watchlist item, not an acquired or presently executable benchmark.

### iReplica's EgoHOI and H-Contact: long interactions with inferred object motion

iReplica combines wearable inertial body capture, egocentric observations,
contact information, and scanned environments. Its project exposes EgoHOI and
H-Contact dataset routes under a custom noncommercial scientific-research
agreement. This is valuable for mobile human-scene interaction, but the method
infers object motion from human contacts and scene constraints; inferred motion
must not be relabeled as externally measured object truth.
[Project](https://virtualhumans.mpi-inf.mpg.de/ireplica/),
[dataset terms and routes](https://virtualhumans.mpi-inf.mpg.de/ireplica/datasets.html)

The [official code](https://github.com/vguzov/ireplica) accepts body motion,
contact predictions, object initial locations, and object point clouds. Its
[example configuration](https://github.com/vguzov/ireplica/blob/cedb6829e09783eb3db00de7cbbd72b5d3311e12/configs/example_single_action.toml)
distinguishes floor-constrained, hinged, and free motion. An optional endpoint
correction can snap a reconstructed object to a final pose. Such a corrected
reference may be useful for reconstruction evaluation, but is not an unassisted
physical rollout. Point clouds also do not guarantee closed collision meshes.

**Adapter gaps:** label observation, inference, and correction provenance per
channel; preserve scene coordinates and articulation; verify dataset size and
sample access before acquisition. Do not confuse this EgoHOI collection with
later projects using the same name for models or HOT3D-derived data.

### IMHD²: dynamic object motion with a small optional sample route

The author repository documents fitted SMPL-H motion, PHOSA-based object motion,
Polycam object scans, object-mounted IMUs, and multi-view observations. Coordinates
are shared within a calibrated recording, but recording dates can use different
world frames. Ground-truth filenames encode retained intervals, and poor-quality
intervals may be omitted. A sample can therefore be unsuitable for grasp
initialization even though a middle motion segment is annotated.
[Official repository](https://github.com/AfterJourney00/IMHD-Dataset)

The public [HF tree](https://huggingface.co/datasets/AfterJourney/IMHD-Dataset/tree/main)
was verified; no full archive is needed for a pilot. Metadata lists a pan sequence
at 2,771,997 bytes and a simplified pan mesh at 502,658 bytes:

- `ground_truth/20231015/20231015_dujsh_pan/freestyle1/gt_0_285_-1.pkl`
- `object_templates/pan/pan_simplified_transformed.obj`

The motion's published LFS SHA-256 is
`2118a9cb8aee156d306b925f1b41b23a98cb56274cb3e5c57f06a01fbdce6beb`.
These are metadata checks, not payload verification. The
[custom license](https://github.com/AfterJourney00/IMHD-Dataset/blob/master/LICENSE)
restricts use to stated noncommercial purposes and limits modification and
redistribution. Review those actual terms before deriving or distributing data.

**Adapter gaps:** exact array schema, sample rate, mesh/pose alignment, wrist
frames, and initial grasp coverage require inspection of an authorized sample.
Do not interpolate across missing intervals or unpickle an unreviewed external
artifact in the main application process. A simplified mesh is a separate asset;
record which geometry defines the fixed benchmark before optimization.

### ACE-Data-0: promising room-scale design, release status must be resolved

The author card describes room-scale and tabletop capture, long chained tasks,
human body/hands, scanned object geometry, object poses, and a tactile subset.
It also describes a mostly objectless human-scene subset, which should remain
outside this project's manipulation scope. The card claims common world/time
alignment but does not establish that every modality exists in every take.
It explicitly says dataset files are not yet published, while the HF interface
shows substantial storage usage. Storage size alone is not proof of an available
licensed sample. The stated agreement is a custom noncommercial research license
with access and redistribution restrictions.
[Author dataset card](https://huggingface.co/datasets/ACERobotics/ACE-Data-0)

**Next check:** resolve release status using the actual shard tree and access
response, then inspect one object-aware take. Verify frame masks, filtered-video
timestamps, calibration, and precise joint/object schemas before implementing an
adapter. Do not infer articulated joint state, deformable geometry, or liquid
dynamics from a rigid object pose field. No payload acquisition is recommended
solely from the headline scale.

## Mobile/context resources that fail the direct moving-object gate

### Nymeria and NymeriaPlus

NymeriaPlus enriches existing Nymeria captures with improved body motion,
scene/object annotations, geometry reconstructions, and extra wearable streams.
However, its paper explicitly states that the Boxy annotation system does not
support dynamic-object annotation. The object boxes and reconstructions are
static scene context, not time-varying manipulated-object poses. Thus it is
strong for locomotion and environment-aware motion, but cannot supply the
required dynamic object trajectory by itself.
[NymeriaPlus paper, section 3.3](https://arxiv.org/html/2603.18496v1)

The [official repository](https://github.com/facebookresearch/nymeria_dataset)
documents selective artifact downloads, current NymeriaPlus support, and legacy
Nymeria support. Treat original and improved body fits as versions of the same
captures. Inspect the separate
[NymeriaPlus data terms](https://github.com/facebookresearch/nymeria_dataset/blob/main/NYMERIAPLUS_DATASET_LICENSE);
the code license and paper license do not replace them. Static objects may later
help create a collision environment, but adding a guessed moving-object track
would be a new reconstruction with its own validation burden.

### Aria Everyday Activities

AEA offers multi-person egocentric activities localized in a shared world, along
with device trajectories, gaze, and point clouds. Its official data description
does not establish a complete human-body plus moving-object pose stream suitable
for direct contact retargeting. The explorer permits individual recording and
modality selection, so a full video download is unnecessary for investigating
camera-motion or scene context.
[Official data documentation](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_activities_dataset),
[example recording](https://explorer.projectaria.com/aea/loc3_script4_seq7_rec1)

**Use:** navigation/perception research, subject to its own data terms. **Do not
use:** camera motion as a surrogate for wrist motion or dynamic object pose.
Its relationship to the earlier Aria Pilot release must be retained rather than
counting the relabeled release as independent capture.

### HD-EPIC

HD-EPIC's real kitchen reconstructions and object itineraries are useful for
long tasks and movement between work areas. Its detailed annotation page defines
object tracks using pickup/place times and endpoint boxes, then lifts locations
to 3D. It does not establish continuous rigid-object orientation throughout the
interaction. The January 2026 intermediate release adds per-frame device/world
poses and gaze, not a replacement object SE(3) stream. The site identifies the
data as CC BY-NC 4.0 and permits selective modality downloads.
[Detailed release page](https://hd-epic.github.io/site/)

**Use:** task phases, endpoint goals, kitchen layout, and navigation context.
**Gap:** human hand/body state, dense dynamic object poses, matching per-object
collision assets, and physical parameters. Do not convert two endpoint locations
into an asserted measured carry trajectory.

### EgoMAN and ambiguous “EgoMo” names

EgoMAN prepares hand trajectories from EgoExo4D, Nymeria, and HOT3D, with
camera-relative pose conventions and additional task annotations. Its official
repository states that processed datasets and weights are not released because
of legal/licensing considerations. Preparation code is not a new captured data
source. Its hand-trajectory representation also does not independently provide
the moving-object states needed here. [EgoMAN repository](https://github.com/facebookresearch/egoman)

The search did not establish a unique public human-object dataset whose canonical
name is exactly “EgoMo.” That term should not silently be mapped to OMOMO,
EgoMAN, EgoHOI, or a similarly named robotics sensor project. Resolve a concrete
paper/project URL before adding a source ID. Existing video-only or body-only
corpora remain excluded unless an explicitly documented object-aware annotation
release satisfies the data contract.

## Existing families: mobility, aliases, and adapter scope

| Existing family | Relevant extension | Qualification for this project |
|---|---|---|
| OMOMO / FullBodyManipulation | Full-body carrying and relocation with tracked objects | One capture family; large-object feasibility and dataset/model terms still matter |
| HUMOTO | Natural body motion with varied object interactions | Existing normalization path is a practical starting point; validate each object and capture |
| ParaHome | Household movement with body/hands and objects | Preserve world coordinates and long task boundaries; supported parsing does not prove dynamics |
| ARCTIC | Bimanual manipulation of articulated objects | Keep root and articulation state; not inherently a room-navigation benchmark |
| CORE4D | Cooperative carrying and transfer | Already in earlier survey; omitted partner forces invalidate single-robot physics interpretation |
| Aria Digital Twin | Tracked interactions in a mapped environment | Already in earlier survey; distinguish per-object motion from scene geometry |
| BEHAVE / InterCap / GRAB / TACO | More object shapes and body/contact patterns | Existing sources; retain source-specific frame, contact, and mesh semantics |
| InterAct / InterMimic / SPIDER outputs | Processed or physically adapted HOI | Keep original capture lineage; crops, policies, and robot embodiments do not create independent human demos |

OMOMO's primary project describes approximately ten hours of body/object motion
for fifteen objects. Later papers use **FullBodyManipulation** for this family;
for example, DecHOI names FullBodyManipulation and cites the OMOMO source. Treat
that name as an alias, not another acquisition target. The official code is MIT,
but separate source-data and SMPL-H/SMPL-X model terms still apply.
[OMOMO project](https://lijiaman.github.io/projects/omomo/),
[official data/code route](https://github.com/lijiaman/omomo_release),
[DecHOI's dataset attribution](https://openaccess.thecvf.com/content/CVPR2026/papers/Jung_Decoupled_Generative_Modeling_for_Human-Object_Interaction_Synthesis_CVPR_2026_paper.pdf)

ARCTIC supplies articulated-object interaction with hands/body and multiple
cameras. Registration and dataset terms apply. Preserve object articulation,
object-part identity, and the source transformation convention. Its images from
several views must not be counted as separate physical demonstrations. A derived
release retaining only the rigid bottom of an object represents a narrower task.
[ARCTIC project](https://arctic.is.tue.mpg.de/),
[official repository](https://github.com/zc-alexfan/arctic)

At inspection, this repository's explicit human normalization dispatch covered
HUMOTO and ParaHome, alongside robot-data paths in
[`normalize.py`](../../reachy_retarget/normalize.py). No adapter support is
claimed here for MeLLO, FORCE, KIT, HIMO, or the other new profiles. Concurrent
implementation and evaluation results must be reported from their own artifacts.

## Adapter and evaluation gap map

| Required capability | Best sources to investigate | Required work before a valid experiment |
|---|---|---|
| Existing fixed-object grasp dynamics | Current robot-object samples, HUMOTO/ParaHome where supported | Compare actuator-produced object motion against a frozen model and declared reference |
| Human root travel with carrying | OMOMO, MeLLO, FORCE | Derive a reachable base/arm plan; implement dynamic mobile base before making mobile claims |
| Two/three independently tracked objects | HIMO; object-aware current HUMOTO scenarios | Preserve all object IDs, meshes, transforms, supports, and phase associations |
| Measured interaction-force comparison | MeLLO | Align force sensor frame and clock; separate sensor readings from derived human dynamics |
| Load sensitivity | FORCE | Identify total object mass; preserve the selected trial's material/inertial model |
| Long room-scale tasks | ParaHome, future InterVLA/ACE release | Keep continuous world/time and approach/carry/place; avoid independent clip resets |
| Articulated manipulation | ARCTIC; suitable iReplica/KIT records | Add object joints and limits, plus task semantics; do not flatten to a rigid mesh |
| High-acceleration object tracking | IMHD² | Preserve valid intervals, fitted-pose provenance and IMU alignment; enforce feasibility |
| Scene-aware path proposals | NymeriaPlus, AEA, HD-EPIC | Use only as context unless a separate verified dynamic object source is added |

For a new adapter, the smallest useful acquisition is **one continuous
demonstration plus its geometry/calibration/terms**, not an arbitrary handful of
frames. Prefer a full approach-grasp-carry-place sequence over the smallest file
if the small file begins mid-grasp. Read metadata and declared sizes first;
stop new transfers before free disk space falls below 50 decimal GB. Do not
fetch raw multiview video when documented state/mesh files suffice.

Persist source URL, immutable revision when available, expected and actual
checksums, bytes, license text/version, native timestamps, coordinate transforms,
missing channels, and any derived labels. Identify parent capture and original
frame range separately from normalized/resampled episode IDs. Dataset-level
statistics should count parent captures and report valid retargetable subsets.

The physical protocol should retain unchanged object geometry, mass, inertia,
and friction; drive Reachy through its simulated actuators; and integrate every
moving object freely after initialization. Failed grasp, slip, unsupported
payload, missing data, and unsupported articulation are separate outcomes.
See the [dynamics-aware methods report](dynamics-aware-retargeting-methods.md)
for an implementation plan and evidence requirements. Neither more source
datasets nor prettier playback establishes physical success.
