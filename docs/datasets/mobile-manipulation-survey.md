# Mobile manipulation data and simulation resources

Access date: **2026-10-06**. This survey covers **37 resource profiles**, not 37 independent demonstration datasets. It extends [the robot-object survey](robot-object-survey.md). The accompanying [machine-readable evidence](mobile-manipulation-survey-evidence.json) records source URLs, inspected revisions, public listing sizes, publisher checksums, access limitations and lineage.

The investigation read public documentation, source code, schemas, license text and repository metadata. It did not download demonstration payloads, accept dataset gates, install simulators, import robot SDKs or access hardware. Existing local acquisition, retargeting and MuJoCo validation belong to the repository's separate run records. A source being public, listed below or recommended does not mean it has passed Reachy physics validation.

## What to use first

RoboCasa365 has the most direct path to a mobile Reachy experiment because its native dataset preserves a MuJoCo model and simulator states. BiGym adds household manipulation with manageable demonstration downloads, although its floating base needs adaptation. ReplicaCAD contributes compact physical object assets for newly generated tests. BEHAVIOR offers much broader activity coverage, with substantial simulator and grasp-model differences.

| Priority | Resource | Evidence useful for Reachy | Work still required |
| --- | --- | --- | --- |
| 1 | [RoboCasa365](https://robocasa.ai/docs/build/html/datasets/using_datasets.html) | Native MuJoCo episode model and states; base, arm and gripper modalities; 220 of 365 tasks require mobility | Retain native extras, resolve assets and controller frames, replan Reachy base and grasp, validate free-object dynamics |
| 2 | [BiGym](https://github.com/NeuracoreAI/bigym) | Native MuJoCo kitchen/object models and human demonstration actions; 123 MB demo release | Reconstruct full object state by pinned replay; adapt floating base to the target base; preserve replay failures |
| 3 | [ReplicaCAD Interactive](https://huggingface.co/datasets/ai-habitat/ReplicaCAD_dataset) | Object collision geometry and physical parameters, articulated furniture and receptacles | Generate new Reachy trajectories; this is an asset package, not retargeting data |
| 4 | [BEHAVIOR / OmniGibson](https://behavior.stanford.edu/challenge/dataset.html) | Broad mobile household tasks and serialized simulator states; a raw episode as small as 667 KB is listed | Decode OmniGibson state and legally usable assets; replace assisted grasp; particle/material tasks exceed a rigid-object adapter |
| 5 | [MoMaGen](https://momagen.github.io/) | Released generator plans base/torso and end-effector motion under reachability and visibility constraints | Audit source-demo access, grasp mechanism and recorder; port planning principles to Reachy instead of assuming source success transfers |
| 6 | [M3Bench](https://huggingface.co/datasets/M3Bench/M3Bench) | Explicit mobile base/arm trajectories, scene/robot URDF and initial object configuration | Large archives; per-frame object movement is not established; separately validate friction and task completion |
| 7 | [MoMaRT](https://robomimic.github.io/docs/datasets/momart.html) | Long-horizon mobile household demos with native PyBullet world snapshots | Historical dependencies; current wrapper rejects use; recover exact body mapping and export assets |
| Real data | [Galaxea](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset), [AIRoA MoMa 5k](https://huggingface.co/datasets/airoa-org/airoa-moma-5k), [Mobile ALOHA](https://github.com/MarkFzp/mobile-aloha) | Broad real tasks; Galaxea explicitly lists chassis position/velocity/IMU; AIRoA has wrist wrench history; Mobile ALOHA has base velocity | Access terms, frame verification and object geometry/pose reconstruction; commands and estimated geometry are not measured object ground truth |

The ranking is an engineering judgment about accessible state, geometry and contact modeling, not a ranking of policy quality or dataset scale. Smaller, explicit simulator episodes can answer a Reachy contact question sooner than a large real-video archive.

## What qualifies as mobile manipulation

A mobile robot appearing in a video is insufficient. A selected sequence should contain a task-relevant object interaction, meaningful base displacement or a navigation-to-manipulation transition, and enough timing/frame information to connect base, torso, end-effector and object state. Report separately whether the base label is a measured pose, measured velocity, commanded displacement, reconstructed odometry, simulator state or a human-derived proxy.

| Field | Evidence to retain | Common mistake |
| --- | --- | --- |
| Base | SE(2)/SE(3), coordinate frame, origin, units, timestamps and whether observed or commanded | Treating a commanded twist as a measured global pose |
| Arms and gripper | Joint ordering, joint versus end-effector action, quaternion convention, gripper width/angle and open/close direction | Assuming all gripper scalars mean the same opening or force |
| Object | Identity, world pose, velocity, articulation, collision mesh, mass, inertia, friction and provenance | Treating an end-effector field, 2D box or static scene placement as a moving object's ground truth |
| Transition | Navigate, approach, establish contact, lift/carry, place, release; phase labels with provenance | Validating arm IK while ignoring base motion and carry stability |
| Physics | Free-body evolution and measured contacts in the target simulator | Copying source object poses, attaching the object to a hand or accepting source assisted-grasp success |

The repository should exclude navigation-only, reach-only, stand-only and other robot-only sequences. Mixed resources can still be useful when their object-interaction subset is selected explicitly. Task generators, scene assets and policy weights stay distinct from demonstration collections.

## Release and access findings that change implementation choices

1. **RoboCasa native extras matter.** The native layout includes `extras/dataset_meta.json`, per-episode `ep_meta.json`, `model.xml.gz` and `states.npz`. A video/LeRobot mirror can omit these. Compile the actual saved model and map its joints rather than guessing object offsets. [Native layout](https://robocasa.ai/docs/build/html/datasets/using_datasets.html).
2. **BEHAVIOR source success uses assisted grasp.** A maintainer confirmed assisted grasping for collection and evaluation. It supplies rich motion references, but does not establish a force-closure grasp for Reachy. The 2026 updates also corrected base velocity into the robot-local frame and later fixed joint velocities; state vectors from different revisions cannot be assumed equivalent. [Maintainer confirmation](https://github.com/StanfordVL/BEHAVIOR-1K/issues/2245#issuecomment-4597125008), [data corrections](https://behavior.stanford.edu/challenge/updates.html).
3. **Habitat interaction abstractions need replacement.** `MagicGraspAction` and `SuctionGraspAction`, and Galactic's kinematic grasp approximation, cannot serve as finger-friction validation. Their scenes, object goals and navigation transitions remain useful. [Habitat actions](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/tasks/rearrange/actions/actions.py), [Galactic](https://github.com/facebookresearch/galactic).
4. **MoMaGen is released code.** The old RSS landing page still says code is coming soon, while the current project links a public repository and installation documentation. The inspected code revision is `5da621667669b15b97d7a220d086283b3c8ad0e5`. Downloadable source demonstrations and their license remain separate unverified dependencies. [Current project](https://momagen.github.io/), [installation](https://chengshuli.github.io/MoMaGen/installation/), [older page](https://momagen-rss.github.io/).
5. **AIRoA MoMa 5k is a new release with different terms.** It reports 1,184,259 primitive episodes grouped into 425,518 short-horizon executions, rather than that many independent full household tasks. Its custom terms must not be replaced with the preliminary release's CC license. Wrist wrench padding/freshness and action-versus-state semantics require care. [Publisher schema and terms](https://huggingface.co/datasets/airoa-org/airoa-moma-5k).
6. **Mobile ALOHA records velocity; Galaxea records more base telemetry.** The inspected Mobile ALOHA writer saves two measured base-velocity values and leaves its T265 field commented out. Galaxea's card lists chassis positions, velocities and IMU as observations, plus a separate commanded twist. Neither schema supplies general all-object 6D trajectories. [Mobile ALOHA recorder](https://github.com/MarkFzp/mobile-aloha/blob/main/aloha_scripts/record_episodes.py), [Galaxea schema](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset).
7. **RoboMIND 2.0 release claims need payload verification.** Its mobile and digital-twin subsets are described publicly, and the project points to ModelScope. The inspected HF 2.0 location did not expose a usable payload listing; that observation does not prove that the ModelScope release is absent. The older gated HF 1.0 collection does not verify the newer mobile subset. [RoboMIND project](https://x-humanoid-robomind.github.io/).
8. **Small examples can still omit crucial state.** TidyBot++ has a mobile MuJoCo cube-pick example, but the inspected recorder/converter saves selected observations/actions rather than the original cube trajectory. The source warns that randomized reset makes open-loop replay unreliable. Use it as a generator with full logging, or label reconstructed source poses as estimates. [Official example and code](https://github.com/jimmyyhwu/tidybot2).

## Bounded acquisition options and measured listing sizes

All byte counts below come from publisher file listings or documentation. Payloads were not fetched by this survey. A listed LFS SHA-256 is a publisher-declared checksum, not a checksum verified against downloaded bytes. Metadata access to a gated repository does not authorize or establish payload access. Stop any future transfers before disk free space falls below 50 decimal GB, including space needed for decompression and derived files.

| Resource | Smallest useful starting point found | Size | Access and interpretation |
| --- | --- | --- | --- |
| BEHAVIOR 2026 raw | `task-0002/episode_00021890.hdf5` | 667,124 bytes | Ungated listing; serialized state inspection only until exact scene/assets and simulator version are resolved |
| BiGym v0.9.0 | `demonstrations.zip` | 123,362,114 bytes | Public release; `DemoStore.get_demos()` can auto-download the whole archive, so acquisition must remain explicit |
| PARTNR | `v0_0/ci.json.gz` | 2,185 bytes | Two compact task definitions, not two motion demonstrations |
| M3Bench | `robot_urdf.zip` | 56,672,394 bytes | Robot assets only; place trajectories add 1,983,364,796 bytes, pick trajectories add 2,811,609,913 bytes, scenes are separate |
| Galaxea | `Put_The_Items_Into_The_Storage_Box_20250929_002_007.tar.gz` | 501,166,913 bytes | Smallest listed task archive; contact-sharing gate and CC-BY-NC-SA terms |
| MoMaRT | Registry sample HDF5s | Approximately 0.6–1.1 GB each | Historical environment dependency; inspect schema before attempting broad acquisition |
| ReplicaCAD Interactive | Complete interactive asset package | Card: 132 MB; current listing/page about 157 MB | Compact physics assets; version/count differences must be pinned |
| RoboCasa365 | One native task package including extras | Not independently measured | Prefer selective official download, with assets resolved from the episode rather than the entire kitchen corpus |

Sources: [BEHAVIOR raw listing](https://huggingface.co/datasets/behavior-1k/2026-challenge-rawdata/tree/main), [BiGym release](https://github.com/NeuracoreAI/bigym_data/releases/tag/v0.9.0), [PARTNR files](https://huggingface.co/datasets/ai-habitat/partnr_episodes/tree/main), [M3Bench files](https://huggingface.co/datasets/M3Bench/M3Bench/tree/main), [Galaxea files](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset/tree/main), [MoMaRT registry](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/__init__.py), [ReplicaCAD card](https://huggingface.co/datasets/ai-habitat/ReplicaCAD_dataset).

For the 667,124-byte BEHAVIOR example, the raw repository revision is `f9d2901112993d75684223820eec70722806c2c4` and publisher LFS SHA-256 is `b01e7b8183e2c738edf1f5e65bba53163fee30713970845631401e20e6cd1a75`. This makes a reproducible future acquisition possible without implying this episode was downloaded or is physically suitable.

## Detailed resource profiles

The profile priority is a triage label: A favors simulator state and assets usable for initial experiments; B requires a generator or simulator conversion; C needs substantial reconstruction or release verification. The specific recommendation takes precedence over the label. A profile can describe a dataset, a benchmark, assets or a system; its resource type is explicit.

### Native simulation data and generators

#### RoboCasa365 and earlier RoboCasa

**Resource:** simulation demonstrations + tasks + assets. **Priority:** A1. **Embodiment:** Panda arm, Omron mobile base, parallel gripper.

- **Coverage:** 365 tasks; paper separates 220 requiring mobility and 145 stationary. Cabinets, appliances, dishes, food, containers; 30k human pretraining demonstrations plus generated data.
- **Base and hands:** Named base position/rotation in modality metadata; exact world/base frame must follow episode configuration. Native state enables authoritative reconstruction. Gripper qpos, closure action, arm end-effector pose/action; base motion and control-mode actions.
- **Objects and assets:** Native extras contain raw MuJoCo states plus episode MJCF. Map joints/bodies from compiled model, never fixed tail offsets. model.xml.gz and ep_meta.json identify fixtures, objects, layouts/styles; resolve referenced mesh/texture assets from pinned RoboCasa.
- **Access and license:** Public official downloader supports task/split/source selection. Download a native task package retaining extras. Exact package bytes were not verified here. Code MIT; data/assets CC-BY-4.0 with upstream asset notices.
- **Limits and lineage:** Version 1.0.1 raises evaluation horizons 1.5x. A LeRobot mirror lacking extras cannot support object replay. Check arm/controller timing and mobile-base constraints when replacing Panda with Reachy. RoboCasa versions, alternate cameras, LeRobot conversions and MimicGen expansions are related releases; record human seed lineage.
- **Next useful check:** First choice: one counter-to-cabinet or counter-to-counter mobile episode, its extras, and only referenced assets; verify base/object poses against simulator FK before Reachy physics.

Inspected revision: code `456174f62b89b8fca99eaaf33949c29fec9cfc2a`.

Primary sources: [Native dataset layout](https://robocasa.ai/docs/build/html/datasets/using_datasets.html), [RoboCasa365 paper](https://robocasa.ai/assets/robocasa365_iclr26.pdf), [Dataset registry](https://github.com/robocasa/robocasa/blob/main/robocasa/utils/dataset_registry.py), [Repository and license](https://github.com/robocasa/robocasa).

#### BiGym

**Resource:** simulation demonstrations + benchmark. **Priority:** A2. **Embodiment:** Bimanual floating-base humanoid with parallel grippers.

- **Coverage:** 40 tasks including dishwashing, cupboards, plates and kitchen cleanup; exclude target-reaching-only tasks for this repository.
- **Base and hands:** Floating-base control is explicit; it is not a wheeled nonholonomic base. Recover full pose from pinned MuJoCo model/state during replay. Bimanual joint-position demonstrations; gripper actions, arm proprioception and optional RGB/depth observations.
- **Objects and assets:** Lightweight demonstrations carry actions and reset seed, not a verified per-frame all-object state table. Replay and log MuJoCo bodies/joints. MJCF robot, household props, kitchen assets and collisions in source tree.
- **Access and license:** Public v0.9.0 demonstrations.zip: 123, 362, 114 bytes. DemoStore.get_demos can auto-download the entire ZIP; use explicit fetch and pin release. Code Apache-2.0; asset notices include CC0, CC-BY-4.0 and CC-BY-NC-4.0. Demo-specific grant should be checked alongside release.
- **Limits and lineage:** Upstream warns that some demonstrations do not succeed on replay. Preserve failed playback. Floating base may prescribe motion unrealistic for Reachy wheels. Former chernyadev repository redirects to NeuracoreAI; cached observation/frequency variants are the same underlying demos.
- **Next useful check:** Choose MovePlate or a cupboard manipulation demo, replay its native actions, export complete object state, then attempt Reachy contact without copying object states during validation.

Inspected revision: code `14beb30318ad14c5d6723175c2ee2281129792af`.

Primary sources: [Official repository](https://github.com/NeuracoreAI/bigym), [Demo loader](https://github.com/NeuracoreAI/bigym/blob/master/demonstrations/demo_store.py), [Demo schema](https://github.com/NeuracoreAI/bigym/blob/master/demonstrations/demo.py), [Release v0.9.0](https://github.com/NeuracoreAI/bigym_data/releases/tag/v0.9.0).

#### BEHAVIOR-1K / OmniGibson 2025 and 2026 releases

**Resource:** simulation demonstrations + tasks + assets. **Priority:** B1. **Embodiment:** 2026 R1Pro bimanual holonomic mobile manipulator.

- **Coverage:** 2026: 20k demos, 100 full-length tasks, seven challenge scenes; benchmark overall has 1, 000 activities. 2025 raw release is distinct version lineage, not automatically independent.
- **Base and hands:** 2026 LeRobot observation.state[0:3] is robot-local base velocity, not global SE(2); raw simulator state supports world pose recovery. 2026 state width 61, action width 23, arm/gripper/trunk positions and velocities; three RGB/depth cameras.
- **Objects and assets:** Raw HDF5 state/state_size plus initial scene information and add/remove transitions. Variable-length serialized state cannot be read using MuJoCo qpos offsets. USD geometry/material/physics and scene JSON; particle/material/semantic states exceed rigid-body MJCF. Many assets are encrypted under third-party licenses.
- **Access and license:** 2026 raw listing: 1, 440, 087, 423, 774 bytes, 20, 000 HDF5s. Smallest listed HDF5 667, 124 bytes. LeRobot documentation: 3.27 TB. Public metadata and raw listing accessible. 2026 LeRobot card says MIT; raw repository has no card license in API. Code MIT does not license encrypted ShapeNet/TurboSquid assets for arbitrary redistribution.
- **Limits and lineage:** Maintainer confirms assisted grasping in collection and evaluation. July 2026 corrected base velocity frame; August corrected joint velocities. Evaluation docs require v 3.9.3 while baseline page still names v 3.9.2. Pin code and data independently. Deduplicate 2025/2026, raw/LeRobot, rerenders and language annotations by original trajectory identifiers. Do not add 10k and 20k without overlap audit.
- **Next useful check:** Use bounded raw state inspection first. Start with rigid pick/carry/place and export legally usable geometry. Replan grasp under physical contacts; assisted source success is not validation.

Inspected revision: code `a8247a8cc1633fe1ca0cc66aa07243d46c64f155`; dataset `f9d2901112993d75684223820eec70722806c2c4`.

Primary sources: [Dataset release](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/docs/challenge/dataset.md), [HDF5 writer](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/OmniGibson/omnigibson/envs/hdf5_data_wrapper.py), [Assisted grasp maintainer confirmation](https://github.com/StanfordVL/BEHAVIOR-1K/issues/2245#issuecomment-4597125008), [Corrections](https://behavior.stanford.edu/challenge/updates.html), [Evaluation version](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/docs/challenge/evaluation.md), [Asset licensing and format](https://behavior.stanford.edu/getting_started/important_concepts.html), [Raw release](https://huggingface.co/datasets/behavior-1k/2026-challenge-rawdata), [LeRobot metadata](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos/blob/main/meta/info.json).

#### Legacy BEHAVIOR-100 VR demonstrations

**Resource:** human teleoperation in simulation. **Priority:** B3. **Embodiment:** Legacy iGibson VR human/avatar embodiment.

- **Coverage:** 500 demonstrations covering 100 activities, five per activity.
- **Base and hands:** VR body/head/hand trajectories; mobile robot base is not directly measured. Tracked human hands/controllers; map to Reachy arm/gripper intent.
- **Objects and assets:** Processed simulator records and raw VR logs differ; inspect actual HDF5 schema before assuming object trajectories. Legacy iGibson scenes/object assets, not current OmniGibson serialization.
- **Access and license:** Official docs list processed ~250 GB, raw ~1.7 GB; raw example ~10 MB and processed example ~1 GB. Dataset and inherited iGibson asset terms require separate verification.
- **Limits and lineage:** Legacy embodiment and simulation dependency; VR trajectories may rely on grasp abstractions. Source states are references, not target-robot dynamics. Separate from BEHAVIOR-1K robot challenge; raw/processed/video views of each recording count once.
- **Next useful check:** Inspect a bounded source episode and its schema before adding an adapter.

Primary sources: [Official legacy dataset documentation](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/docs/behavior_100/dataset.md).

#### MoMaRT / Mobile Manipulation RoboTurk

**Resource:** simulation demonstrations. **Priority:** B2. **Embodiment:** Mobile robot in iGibson/PyBullet kitchen.

- **Coverage:** Five long-horizon table setup/cleanup and dishwasher tasks; expert, suboptimal, generalization, sample splits.
- **Base and hands:** Wrapper exposes base x/y and sin/cos yaw in observations, base linear/angular velocity; native WorldSaver captures world state. Arm/gripper proprioception and actions, task object observations.
- **Objects and assets:** WorldSaver.serialize(), with excluded body IDs; task observation object-state is source-task-specific. It is not robosuite flattened MuJoCo state. Requires historical iGibson momart branch and exact kitchen/object assets.
- **Access and license:** Registry estimates sample files 0.6–1.1 GB; full task/split files 14–36 GB. Direct official HDF5 links. No payload fetched. Data grant not verified separately from robomimic code and iGibson asset terms.
- **Limits and lineage:** Current robomimic constructor explicitly raises no-longer-supported; use historical compatible implementation in isolation. Body IDs and serialized state must match source model. No independence claim; deduplicate by original episode identifiers.
- **Next useful check:** Inspect table_setup_from_dishwasher_sample.hdf 5 schema and recover body mapping from pinned iGibson before writing any decoder.

Inspected revision: code `d309eaecc18acf4152a830a895a6984b8ac71b05`.

Primary sources: [Official dataset guide](https://robomimic.github.io/docs/datasets/momart.html), [Native state/observation wrapper](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/envs/env_ig_momart.py), [Download registry](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/__init__.py), [Paper](https://proceedings.mlr.press/v164/wong22a/wong22a.pdf).

#### M3Bench

**Resource:** generated whole-body trajectories + scene assets. **Priority:** B2. **Embodiment:** Mecanum/Kinova mobile manipulator.

- **Coverage:** 30k pick/place trajectories across 119 scenes; planner-generated, not 30k human teleoperations.
- **Base and hands:** Base-arm joint trajectories and initial robot configuration; global transforms can be derived from supplied URDF. Joint trajectory, language and target configuration; verify gripper joint/action timing in archive.
- **Objects and assets:** config.json contains target link and initial object pose. Published schema does not promise a measured per-frame object trajectory. Robot URDF/USD and complete scene URDF packages; nonuniform mesh scale handling is documented.
- **Access and license:** Public HF listing: 20, 115, 025, 023 bytes; robot URDF ZIP 56.7 MB, pick ZIP 2.81 GB, place ZIP 1.98 GB, scene ZIP 15.24 GB. No single-episode listing. HF card Apache-2.0; inspect asset-specific notices before redistribution.
- **Limits and lineage:** PyBullet evaluation checks collisions/limits/smoothness; Isaac Sim separately checks task completion. A kinematically feasible trajectory is not automatically a friction-valid grasp. No independence claim; deduplicate by original episode identifiers.
- **Next useful check:** Request/select one scene and trajectory from archive ranges if supported; decode base/arm and object initial pose, then run an independent Reachy physics attempt.

Inspected revision: code `97cec07e7c37cc36aec35ff79ee5d6b5db123a45`; dataset `5b543a8c93c82b16442b2c1c4a139b96bb313b59`.

Primary sources: [Official dataset structure](https://github.com/TooSchoolForCool/M3Bench), [Publisher dataset](https://huggingface.co/datasets/M3Bench/M3Bench), [Project and paper](https://zeyuzhang.com/papers/m3bench/).

#### MoMaGen

**Resource:** released simulation data-generation code; downloadable demo corpus unverified. **Priority:** B1-generator. **Embodiment:** Bimanual mobile manipulator.

- **Coverage:** Four tasks: pick cup, tidy table, clean pan, put dishes away.
- **Base and hands:** Planner jointly chooses base/torso/camera poses under reachability/visibility constraints. Object-centric end-effector subtask trajectories transformed from seed demos.
- **Objects and assets:** Generation uses object-centric frames and simulator state. A public recording package with a complete per-frame object-state schema was not verified. OmniGibson/BEHAVIOR dependencies, task configurations and scene-instance files are in the released generator. Exact assets and source demonstrations remain separate dependencies.
- **Access and license:** Public ChengshuLi/MoMaGen source and installation documentation are available. Repository revision 5da621667669b15b97d7a220d086283b3c8ad0e5 was checked. No downloadable source-demonstration package was verified. Website CC-BY-SA is not a dataset license. Code, source demonstration and inherited BEHAVIOR asset grants need separate inspection.
- **Limits and lineage:** The older momagen-rss.github.io page still says code coming soon; current momagen.github.io links the public repository. Do not infer source-data availability or physical-grasp fidelity from code availability. Synthetic augmentation derives from seed demonstrations; descendants are not independent human demos.
- **Next useful check:** Inspect the generator recorder and grasp settings, then plan one source task with explicit base reachability and visibility constraints. Verify seed data access and asset license before installation.

Inspected revision: code `5da621667669b15b97d7a220d086283b3c8ad0e5`.

Primary sources: [Current official project](https://momagen.github.io/), [Released code at inspected revision](https://github.com/ChengshuLi/MoMaGen/tree/5da621667669b15b97d7a220d086283b3c8ad0e5), [Installation documentation](https://chengshuli.github.io/MoMaGen/installation/), [Paper v4](https://arxiv.org/abs/2510.18316v4), [Stale earlier landing page](https://momagen-rss.github.io/).

#### ManiSkill mobile tasks and scene-manipulation environments

**Resource:** simulation task suite + selected demonstrations. **Priority:** B2. **Embodiment:** Fetch and other configurable mobile robots.

- **Coverage:** Mobile OpenCabinetDrawer plus exploratory scene environments. Documentation says large scene environments do not all have rewards/success definitions.
- **Base and hands:** Source simulator can export base articulation/root pose; requested trajectory modality must include state. Robot/controller-specific joint, end-effector and gripper actions.
- **Objects and assets:** Native env_states may contain actor/articulation pose and velocity; verify HDF5 modality and asset IDs for each download. SAPIEN/PhysX assets; PartNet-Mobility cabinets and RoboCasa/ReplicaCAD/AI2THOR scenes.
- **Access and license:** Explicit asset/demonstration downloader; per-environment availability differs. A scene package is not a demonstration collection. Code and each source asset dataset have separate terms; PartNet/ReplicaCAD/AI2THOR terms persist.
- **Limits and lineage:** MuJoCo conversion requires geometry, joint-axis, friction, mass and timestep reconciliation; reused RoboCasa scenes are not new independent scenes. No independence claim; deduplicate by original episode identifiers.
- **Next useful check:** Prefer one native cabinet trajectory with env_states over a random scene rollout; verify target handle and articulation before Reachy conversion.

Primary sources: [Mobile task specifications](https://maniskill.readthedocs.io/en/v3.0.0b20/tasks/mobile_manipulation/), [Scene dataset status](https://maniskill.readthedocs.io/en/v3.0.0b21/user_guide/datasets/scenes.html), [Official repository](https://github.com/haosulab/ManiSkill).

#### HumanoidBench manipulation subset

**Resource:** MuJoCo benchmark / rollout generator. **Priority:** B2. **Embodiment:** H1/G1 humanoids, dexterous or simplified hands.

- **Coverage:** Whole-body push, cabinet, door, package, bookshelf, kitchen and other manipulation. Exclude walk/stand/reach-only tasks.
- **Base and hands:** Free-root locomotion state is in model qpos/qvel; Reachy would need a wheeled-base adaptation. Joint/torque controls; privileged robot+environment observations and optional tactile/visual sensors.
- **Objects and assets:** MuJoCo task object state can be logged directly during generated rollouts. No released large demonstration corpus verified. MJCF assets and source task code available; robot models inherit upstream sources.
- **Access and license:** Public code, low-level policy weights and learning curves. Weights/curves are not demonstrations. Repository license plus upstream model/texture licenses; audit exact assets selected.
- **Limits and lineage:** Some examples strengthen humanoid hands for specific tasks. Keep the selected task physics explicit and do not treat source policies as Reachy success. No independence claim; deduplicate by original episode identifiers.
- **Next useful check:** Useful MuJoCo task generator: adapt a rigid push/package task with source mesh unchanged and save all outcomes.

Primary sources: [Official benchmark](https://github.com/carlosferrazza/humanoid-bench), [Release history](https://github.com/carlosferrazza/humanoid-bench/releases).

### Habitat tasks, traces and scene assets

#### HomeRobot / OVMM

**Resource:** task episodes + scene/object assets + baselines. **Priority:** B3. **Embodiment:** Hello Robot Stretch in Habitat and real-world systems.

- **Coverage:** Explore, locate an open-vocabulary object, find destination receptacle, pick and place.
- **Base and hands:** Simulation exposes root pose; episode start pose and goals are available. Benchmark package is not dense base-motion demonstrations. Arm/gripper control comes from simulator and baseline skills.
- **Objects and assets:** Episode object placements/targets and runtime object transforms; no universal released per-frame demo tracks. Pinned HSSD ovmm branch, OVMM object collection, robot model and episode files.
- **Access and license:** Official download script pins HSSD 4e0292b..., objects e9b714c..., episodes 9ad25fb.... Current object listing is ~1.34 GB; scenes separate. HomeRobot code MIT; HSSD CC-BY-NC-4.0; object collection inherits asset-specific licenses.
- **Limits and lineage:** Habitat magic/suction grasp actions use holding abstractions. Need physical gripper replacement before Reachy validation; preserve split/category boundaries. HSSD, OVMM objects and OVMM episodes are complementary components, not three demonstration datasets.
- **Next useful check:** Select a mini episode and explicit assets; generate and log base/object state, then replace assisted hold with Reachy contacts.

Inspected revision: code `ede6a67a2d0c0c8e12ad3b9726f330cea6cf90f5`; dataset `e69d754e7256f6f906a5d553f79ad38edc37e462`.

Primary sources: [Official repository](https://github.com/facebookresearch/home-robot), [Exact asset pins](https://github.com/facebookresearch/home-robot/blob/main/download_data.sh), [OVMM paper](https://ovmm.github.io/OVMM_ArXiv.pdf), [Grasp/action semantics](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/config/CONFIG_KEYS.md).

#### Habitat 2/3 rearrangement

**Resource:** task episodes + simulator. **Priority:** B3. **Embodiment:** Fetch/Stretch/Spot and human avatars, depending on task.

- **Coverage:** 2022 train: 50k problem definitions over 63 scenes; minival 20; val 1k over 21 scenes. Navigation and pick/place/open/close compose tasks.
- **Base and hands:** Start configuration and runtime mobile base pose; standard base action is forward velocity plus yaw velocity. Joint/EE arm action with magic or suction gripper controller.
- **Objects and assets:** Object initial/goal transforms, articulated furniture state, runtime transforms; task episodes are not dense demonstrations. ReplicaCAD/YCB for 2022 rearrangement; Habitat 3/PARTNR add other scenes and agents.
- **Access and license:** Public training/validation episode and asset downloads; held-out challenge test episodes are not public. Habitat code MIT, ReplicaCAD CC-BY-4.0; task and other asset terms separate.
- **Limits and lineage:** MagicGraspAction and SuctionGraspAction are not Reachy finger friction. Current repository announces no official Meta maintenance beyond v 0.3.4. CortexBench MobilePick and Galactic/Habitat evaluations reuse this ecosystem; do not count them as independent demonstrations.
- **Next useful check:** Use to sample scene/goal distributions; independently generate dense state trajectories and validate contact.

Primary sources: [Challenge episode definitions](https://aihabitat.org/challenge/rearrange_2022/), [Action and sensor schema](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/config/CONFIG_KEYS.md), [Maintenance status](https://github.com/facebookresearch/habitat-lab).

#### PARTNR episodes and HitL traces

**Resource:** task generator + problem instances + human-in-loop traces. **Priority:** B3. **Embodiment:** Human/robot collaborative agents in Habitat 3.

- **Coverage:** 131, 991 unverified generated problems; 111, 652 verified training subset; 2k/403 subsets; val 1k; CI 2. Released HitL traces are separate.
- **Base and hands:** Simulator/HitL trace can describe agents; audit recorded trace fields and actions rather than assuming robotic joint demonstrations. Motor skills pick/place/nav/open/close; humanoid interaction abstraction differs from Reachy gripper.
- **Objects and assets:** Episode world graph/placements and HitL traces; dense state/velocity coverage must be checked in trace reader. HSSD scenes and object collections; models/policies and concept graphs are additional artifacts.
- **Access and license:** HF listing 1, 045, 010, 217 bytes. CI problem file 2, 185 bytes; two largest HitL archives ~266/284 MB. Data CC-BY-NC-4.0; scene/asset source terms also apply.
- **Limits and lineage:** Verified problem feasibility does not mean 111k dense demonstrations. Same Habitat grasp limitations; distinguish user avatar motion from robot motor state. train_mini/train_2k are subsets of train, itself subset of unverified generation. Count each episode once.
- **Next useful check:** Inspect a bounded source episode and its schema before adding an adapter.

Inspected revision: code `ddfff19f4b6c098a31edea4d19e7b75db72433c2`; dataset `c6cffbdd82691c2f4240b75aa5289cd9efd0a32b`.

Primary sources: [Publisher dataset card](https://huggingface.co/datasets/ai-habitat/partnr_episodes), [Generator and baselines](https://github.com/facebookresearch/partnr-planner), [Trace analysis](https://github.com/facebookresearch/partnr-planner/tree/main/scripts/hitl_analysis).

#### Habitat-Web Pick-and-Place-HD

**Resource:** human demonstrations in simulator. **Priority:** B3. **Embodiment:** Browser-controlled Habitat agent.

- **Coverage:** 12k Pick-and-Place human demonstrations; separate 70k ObjectNav set is navigation-only and excluded.
- **Base and hands:** Navigation trajectory/action history from human demonstrations; embodiment is not necessarily an articulated mobile manipulator. Pick/place task abstraction; inspect action schema before mapping to a gripper.
- **Objects and assets:** Scene/object assets and replayable demonstration actions; dense physical object-state schema not verified. Matterport 3D scenes plus released pick/place object assets.
- **Access and license:** Official repository links axel81/habitat-web and object assets. Total/sample bytes not established here. Demonstrations CC-BY-NC-4.0; MP3D requires its own access/terms.
- **Limits and lineage:** Best for navigation-to-interaction sequencing. Discrete object interaction does not establish antipodal grasp or physical forces. No independence claim; deduplicate by original episode identifiers.
- **Next useful check:** Audit one Pick-and-Place trace, excluding ObjectNav-only demonstrations.

Primary sources: [Official download instructions](https://github.com/Ram81/habitat-imitation-baselines), [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ramrakhya_Habitat-Web_Learning_Embodied_Object-Search_Strategies_From_Human_Demonstrations_at_Scale_CVPR_2022_paper.pdf).

#### Galactic

**Resource:** kinematic simulator + tasks + assets. **Priority:** C3. **Embodiment:** Fetch-style mobile robot; extendable URDF articulation.

- **Coverage:** Mobile pick and tidy-house rearrangement at high simulation throughput.
- **Base and hands:** Robot articulation and start poses plus runtime kinematic state. Abstract grasping and dropping behavior.
- **Objects and assets:** Episode placements and runtime kinematic movable-object poses. Runtime datasets bundled under data: collection JSON, URDF, collision grids, mesh/texture packs and episode JSON.
- **Access and license:** Public repository includes runtime data; archived August2024. Source asset reconstruction is separate. Majority CC-BY-NC; Habitat submodules/bps 3D MIT; source asset terms retained.
- **Limits and lineage:** Explicitly kinematic collision/sliding/grasping approximations. Unsuitable for proving slip-free Reachy grasp dynamics. Reuses ReplicaCAD/YCB and Habitat components; count tasks/assets separately from recordings.
- **Next useful check:** Use scene placements for planning stress tests only; regenerate physical contacts independently.

Primary sources: [Simulation and license statement](https://github.com/facebookresearch/galactic), [Data format](https://github.com/facebookresearch/galactic/blob/main/DATA.md).

#### ReplicaCAD Interactive

**Resource:** scene/object asset dataset. **Priority:** A3-assets. **Embodiment:** Robot-independent; Fetch-oriented navigation meshes.

- **Coverage:** Apartment rearrangements; interactive package has 84 of 105 published variations, 21 held out.
- **Base and hands:** No measured base trajectory. Scene transforms and navmeshes only. No gripper demonstrations.
- **Objects and assets:** Static/initial object transforms and articulated asset definitions; no time-series interactions. 90+ object assets with convex collision meshes, mass/friction/restitution, six+ articulated URDFs, receptacle metadata.
- **Access and license:** Public, ungated HF listing: 157,450,746 bytes across 602 files at revision 3e8c7fe5759f64bfcbc3882f9cdf6de97f82a06d. The card describes a 132 MB original interactive package; the 525 MB baked-lighting package is a different rendering variant. CC-BY-4.0.
- **Limits and lineage:** Strong compact asset source, but scenes are related apartment variants, not independent household demonstrations. Baked-lighting version makes most furniture static. Interactive/baked versions and microvariations share asset and scene lineage.
- **Next useful check:** Import one object or articulated cabinet with source physical properties; create a new Reachy task and label it generated, not retargeted.

Inspected revision: dataset `3e8c7fe5759f64bfcbc3882f9cdf6de97f82a06d`.

Primary sources: [Publisher interactive asset card](https://huggingface.co/datasets/ai-habitat/ReplicaCAD_dataset), [Project](https://aihabitat.org/datasets/replica_cad/).

#### HSSD / HSSD-Hab / HSSD-mini

**Resource:** scene/object asset dataset. **Priority:** B3-assets. **Embodiment:** Robot-independent.

- **Coverage:** Home navigation and rearrangement contexts; wide visual/object diversity.
- **Base and hands:** No base trajectories. No gripper demonstrations.
- **Objects and assets:** Object instances and scene placements, not temporal manipulation tracks. 211 authored scenes, 18, 656 object models; Habitat/OVMM adaptations provide metadata needed for interaction.
- **Access and license:** Official HF hssd/hssd-hab; HomeRobot pins ovmm branch. Mini subset useful for bounded setup. Incorrect ai-habitat/HSSD path is not the official release. CC-BY-NC-4.0; asset/source notices remain relevant.
- **Limits and lineage:** Navigation-ready geometry need not have verified mass/inertia/contact meshes. Do not equate individual object models with demonstrations. HSSD-mini, OVMM and PARTNR reuse scenes/assets; they are not separate home scans.
- **Next useful check:** Use mini scene plus a selected dynamic object only after collision/inertia audit.

Primary sources: [Official project](https://3dlg-hcvc.github.io/hssd/), [Habitat package](https://huggingface.co/datasets/hssd/hssd-hab), [OVMM branch pin](https://github.com/facebookresearch/home-robot/blob/main/download_data.sh).

#### HM3D / HM3DSem

**Resource:** real-world scene scans / navigation assets. **Priority:** C3-assets. **Embodiment:** Robot-independent.

- **Coverage:** Residential/commercial/civic navigation contexts.
- **Base and hands:** No robot base trajectories in scene release. No manipulation demonstrations.
- **Objects and assets:** Semantic instance geometry in scanned scenes; no freely moving object tracks or calibrated object dynamics. 1, 000 building-scale scans; downloadable training/validation versus held-out test.
- **Access and license:** Access via Matterport research agreement; no acceptance performed. Byte totals depend on release/resolution. Academic noncommercial research terms, separate from Habitat MIT code.
- **Limits and lineage:** A scanned mug merged into static scene geometry cannot become a physical free object without segmentation, completion and assumed dynamics. HM3DSem adds annotations to HM3D; navigation derivatives are not new manipulation demonstrations.
- **Next useful check:** Use as background/navigation context, not priority object-contact ground truth.

Primary sources: [Official HM3D source](https://aihabitat.org/datasets/hm3d/).

### Real mobile-robot collections

#### Mobile ALOHA

**Resource:** real-robot demonstrations / collection code. **Priority:** C1-motion. **Embodiment:** Dual ViperX arms on differential mobile base.

- **Coverage:** Cooking, storage, cleaning and multi-step bimanual mobile tasks.
- **Base and hands:** Recorder saves measured linear/angular base velocity in /base_action[2]; optional T265 field is commented out. No measured global SE(2) field verified. 14-D arm/gripper qpos/qvel/effort and action, multiview video.
- **Objects and assets:** No object 6D trajectory or matched scene/object mesh in inspected writer. Robot hardware models available separately; real object geometry/dynamics require reconstruction.
- **Access and license:** Official project/code and linked data resources; individual HDF5 episodes. No complete official payload byte inventory established. Code MIT; verify exact data-host license before mirroring.
- **Limits and lineage:** Velocity integration gives derived drifting odometry, not measured world pose. Co-training stationary ALOHA data does not make it mobile. ACT/ALOHA stationary demos and converted mirrors overlap. Keep source episode IDs and distinguish mobile subset.
- **Next useful check:** Inspect a bounded source episode and its schema before adding an adapter.

Inspected revision: code `0e403249c76054a68e757e590d4da4dba401c9e3`.

Primary sources: [Official project](https://mobile-aloha.github.io/), [Recorder schema](https://github.com/MarkFzp/mobile-aloha/blob/main/aloha_scripts/record_episodes.py), [Paper](https://mobile-aloha.github.io/resources/mobile-aloha.pdf).

#### AIRoA MoMa preliminary release

**Resource:** real-robot demonstrations. **Priority:** C1-motion. **Embodiment:** Toyota HSR.

- **Coverage:** 23, 762 filtered PA episodes, 87 h, seven household task groups. Earlier paper/preliminary counts differ.
- **Base and hands:** Public card describes robot state; base-state names and frame not verified behind gate. Arm/gripper/head state and actions; RGB cameras and calibration.
- **Objects and assets:** No per-frame object 6D or matching meshes documented. One household laboratory; physical HSR model does not define the manipulated objects.
- **Access and license:** HF auto-gated; anonymous card/API visible, payload requires accepting contact-sharing terms. Listing89, 898, 503, 338 bytes. CC-BY-NC-SA-4.0.
- **Limits and lineage:** v 1.1 filters duration, state jumps and synchronization delay. Primitive episodes grouped by UUID may belong to one longer execution. MoMa 5k expands preliminary collection; audit UUID overlap before summing.
- **Next useful check:** Inspect a bounded source episode and its schema before adding an adapter.

Inspected revision: dataset `586f25e39d7b8d4d13b2b94dd7703a47a02cdd0b`.

Primary sources: [Publisher card and version notes](https://huggingface.co/datasets/airoa-org/airoa-moma), [Paper](https://arxiv.org/abs/2509.25032).

#### AIRoA MoMa 5k

**Resource:** real-robot demonstrations with wrench telemetry. **Priority:** C1-motion. **Embodiment:** Toyota HSR across 44 robots/five sites.

- **Coverage:** 1, 184, 259 successful primitive episodes, 425, 518 grouped short-horizon executions, 93 templates, 5, 025 h at 10 Hz.
- **Base and hands:** action.base[3] is x/y/theta delta. This is an action, not measured global base pose; 11-D servo fields include wheels. 8-D arm/wrist/gripper/head state; wrist wrench history600 values, freshness flags, 6-D EE poses.
- **Objects and assets:** No all-object 6D or mesh correspondence in documented feature schema. Real-world scenes; no matched simulator object assets verified.
- **Access and license:** Aug2026 release; gated v3 packed repository~4 TiB, v2.1 bucket has same records in per-episode form. Custom airoa-public-dataset-terms; own R&D use and restrictions on redistribution of data/derivatives/models. Do not substitute preliminary CC license.
- **Limits and lineage:** One site has 6-value wrench samples padded to 600 in v 3; freshness/validity matters. Success-only train split; PA count is not independent full-task count. Absolute EE frame needs schema clarification. v 2.1 and v 3 payloads represent the same episodes. Group by UUID; deduplicate against preliminary release.
- **Next useful check:** After authorized access, inspect one complete grouped navigation/pick/carry/place sequence and wrench validity; object reconstruction remains separate.

Inspected revision: dataset `2681f223af815be0465189103f198ae97cd694c3`.

Primary sources: [Publisher detailed schema and terms](https://huggingface.co/datasets/airoa-org/airoa-moma-5k), [Official release announcement](https://airoa.org/updates/20260803/487/), [License](https://huggingface.co/datasets/airoa-org/airoa-moma-5k/blob/main/LICENSE.ja.md).

#### Galaxea Open-World Dataset

**Resource:** real-robot demonstrations. **Priority:** C1-motion. **Embodiment:** Galaxea R1-Lite mobile dual-arm robot.

- **Coverage:** 500+ hours; 227 task-level archives across homes, kitchens, retail and offices; bed making/table bussing/microwave.
- **Base and hands:** The published schema records observation.state.chassis[3] positions, chassis.velocities[3], and chassis.imu[10]. The reference frame, pose convention and odometry calibration are not established by the card. action.chassis.velocities[6] is a commanded twist. Left/right arm position and velocity vectors have six values each; torso vectors have four. Each gripper observation ranges from 0 closed to 100 open. Each end-effector pose has position plus quaternion (seven values).
- **Objects and assets:** No per-frame all-object 6D or dynamics assets in published schema. 11 real collection sites; real objects only in visual observations/annotations.
- **Access and license:** HF auto-gated contact-sharing; listing2, 867, 672, 629, 620 bytes. Smallest task archive501, 166, 913 bytes. CC-BY-NC-SA-4.0 for this repository.
- **Limits and lineage:** Observed chassis position is stronger evidence than command-only mobility. Global reference frame and quaternion ordering still need calibration/schema verification. Object poses and physical assets are absent from the inspected schema. G0/G05 model training and RoboCOIN R1-Lite subsets are not automatically independent recordings; compare episode source IDs.
- **Next useful check:** Inspect a bounded source episode and its schema before adding an adapter.

Inspected revision: dataset `df670d8dc3f2b369f55d28a3428b8639f0dab37d`.

Primary sources: [Publisher schema and archives](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset), [Official project](https://opengalaxea.github.io/GalaxeaVLA/), [Paper](https://arxiv.org/abs/2509.00576).

#### RoboMIND 2.0 mobile subset and digital twins

**Resource:** real/sim dataset releases with availability caveat. **Priority:** C2-pending. **Embodiment:** Six embodiments including mobile bimanual platforms.

- **Coverage:** 2.0 reports 310k+ trajectories 739 tasks, including mobile and tactile subsets.1.0 primarily stationary multi-embodiment manipulation.
- **Base and hands:** Paper claims20k mobile trajectories; exact measured base-state fields not independently verified in released2.0 payload. Robot proprioception, multiview observations; 12k tactile-enhanced episodes reported.
- **Objects and assets:** 20k simulation trajectories/digital twins reported; per-frame object-state schema and matched assets not verified. Real scenes and Isaac Sim digital twins; actual per-task asset distribution needs inspection.
- **Access and license:** Official site points to ModelScope collection. HF2.0 indexed tree has only README/.gitattributes; API401 here. HF1.0 is gated12.28 TB, not evidence2.0 mobile data downloaded. 1.0 HF card Apache-2.0; 2.0 exact ModelScope data terms not verified.
- **Limits and lineage:** Do not equate paper scale with accessible payload. Need compare 2.0 with 1.0 and identify overlapping demos; mobile robot name alone does not prove base motion. No independence claim; deduplicate by original episode identifiers.
- **Next useful check:** Inspect a bounded source episode and its schema before adding an adapter.

Inspected revision: dataset `e7ffe31d1fe983a42c3d7b79d192f554fc05b86e`.

Primary sources: [Official site release link](https://x-humanoid-robomind.github.io/), [2.0 paper](https://arxiv.org/abs/2512.24653), [2.0 HF listing](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND2.0/tree/main), [Official ModelScope collection](https://modelscope.cn/collections/X-Humanoid/RoboMIND20), [1.0 publisher card](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND).

#### AgiBot World Alpha/Beta

**Resource:** real-robot demonstrations. **Priority:** C1-motion. **Embodiment:** Mobile dual-arm AgiBot robots.

- **Coverage:** Broad manipulation collection; select verified navigation-plus-manipulation episodes.
- **Base and hands:** Robot state/action telemetry; base coordinate fields must be checked by embodiment. Joint/end-effector/gripper or dexterous-hand channels depending on robot.
- **Objects and assets:** Visual/object annotations are not verified world-object 6D tracks. Real scenes; robot metadata does not provide calibrated object meshes.
- **Access and license:** Beta public card/listing but auto-gated; listing48, 050, 136, 152, 505 bytes. Task-state and observation archives separated. Custom publisher terms/card did not expose a standard license via API; verify before transfer.
- **Limits and lineage:** Large scale alone does not solve missing object state. Avoid downloading multi-tens-of-GB video archives for schema inspection. Publisher says Alpha is a subset of Beta; never add their episode counts.
- **Next useful check:** Inspect a bounded source episode and its schema before adding an adapter.

Inspected revision: dataset `2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278`.

Primary sources: [Official publisher organization and lineage](https://huggingface.co/agibot-world), [Beta release](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta), [Primary paper](https://agibot-world.com/blog/agibot_go1.pdf).

#### AgiBot World2026

**Resource:** real demonstrations + separately identified simulation/RL sections. **Priority:** C1-motion. **Embodiment:** AgiBotG2 mobile bimanual platform.

- **Coverage:** Long-horizon restocking, atomic-skill annotations and retained error/recovery trajectories.
- **Base and hands:** Per-task meta/info.json field_descriptions defines indices; documented action/robot/velocity is not a universal SE(2) pose. Joint position/velocity telemetry and end-effector actions; exact dimensions vary by subrelease.
- **Objects and assets:** Real collection includes2D boxes/object attributes, not guaranteed6D poses. Simulation section must be audited independently. Real service/commercial scenes; simulation assets via GenieSim separate.
- **Access and license:** Ungated listing13, 564, 582, 299, 458 bytes; current directories include simulation and ReinforcementLearning as well as imitation data. CC-BY-NC-SA-4.0 in current publisher card.
- **Limits and lineage:** Old card sample path task 3777/380098_380609.tar.gz may no longer match current tree; use recorded listing instead of blindly following sample text. Separate real, synthetic and policy-rollout provenance. Overlap with 2025 collection not resolved.
- **Next useful check:** Inspect a bounded source episode and its schema before adding an adapter.

Inspected revision: dataset `cf9841dffaa4bb6a7b226a30b145e4fa5127a064`.

Primary sources: [Publisher current card](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026), [Official announcement](https://www.agibot.com/article/231/detail/54.html).

#### AgiBot Digital World / GenieSim

**Resource:** synthetic data + scene/task generator. **Priority:** B3. **Embodiment:** AgiBotG1/G2; task config can use fixed or mobile base.

- **Coverage:** Current project advertises 10k+ synthetic hours and 200+ loco-manipulation tasks; precise public subset requires listing audit.
- **Base and hands:** Task template defines robot initial SE(3); runtime mobile state availability depends on recorder/selected task. Whole-body/gripper task controls and recordings; no uniform action width assumed.
- **Objects and assets:** Engine exposes object world/local poses and filters over motion; whether public LeRobot archives retain these is unverified. USD scenes/assets, JSON task recipes, object IDs, physical simulation; 3D Gaussian reconstruction is not itself collision geometry.
- **Access and license:** Public code/assets/data links; 2026 HF listing includes simulation metadata bundles as small as1.18 MB. Full stack heavy. Code/data/asset releases can differ; 2026 bundled data CC-BY-NC-SA; verify selected generator and asset license.
- **Limits and lineage:** Some templates explicitly attach objects. Do not import attachment-based behavior as physical grasp evidence. Source success filters need independent contact audit. Digital World, GenieSim versions and 2026 simulation packages can overlap; track generation seeds and source recipes.
- **Next useful check:** Read a metadata-only task bundle and recorder schema before a simulator install; prioritize free rigid objects without attachment constraints.

Primary sources: [Official platform](https://github.com/AgibotTech/genie_sim), [Task configuration](https://github.com/AgibotTech/genie_sim/blob/main/source/data_collection/TASK_CONFIG_GUIDE.md), [Object-state filters](https://github.com/AgibotTech/genie_sim/blob/main/source/data_collection/common/data_filter/README.md).

#### BRMData

**Resource:** real-robot bimanual/mobile demonstrations. **Priority:** C1-motion. **Embodiment:** Bimanual mobile household robot.

- **Coverage:** Ten original tasks incl bottle handoff, cup/plate placement, fruit, cleaning, human interaction; five added tasks announcedAug 2024.
- **Base and hands:** Mobile capability is described; exact base pose/action schema remains unverified. Arm/gripper demonstrations and multiview/depth sensing; inspect HDF5 fields.
- **Objects and assets:** No verified per-frame object 6D or matched scene dynamics in public overview. Real rigid/flexible objects; CAD correspondence not established.
- **Access and license:** Official JDBox dataset link uses password given in paper; no gate acceptance or payload download performed. Dataset license link on project did not resolve in this check; grant unverified.
- **Limits and lineage:** Mobile/tabletop and human-interaction subsets must be distinguished. Do not extrapolate ten-task paper to every later announced task. No independence claim; deduplicate by original episode identifiers.
- **Next useful check:** Inspect a bounded source episode and its schema before adding an adapter.

Primary sources: [Official project and download](https://embodiedrobot.github.io/BRMData.html), [Official code](https://github.com/Louis-ZhangLe/BRMData), [Paper](https://arxiv.org/abs/2405.18860).

#### RoboCOIN mobile-capable subsets

**Resource:** real-robot multi-embodiment data collection. **Priority:** C1-motion. **Embodiment:** 15 platforms including R1-Lite and other mobile-capable bimanual robots.

- **Coverage:** Paper 180k+ demonstrations 421 tasks 16 scenarios; choose actual mobile tasks, not all bimanual data.
- **Base and hands:** Per-task base channels not guaranteed by common schema; verify actual chassis motion. Standard arm/hand radians, normalized gripper, EE fields; actions may be derived from next observation if original actions absent.
- **Objects and assets:** eef_sim_pose_state is robot forward-kinematic EE pose, not simulated object ground truth. Unified robot-coordinate models; real scene/object geometry not generally supplied.
- **Access and license:** Task-level HF/ModelScope download manager; access application required. Example two-task selection10.8 GB. Toolkit Apache-2.0; each data repository terms require verification.
- **Limits and lineage:** Unified x-forward/y-left/z-up frame is at robot base/feet center. Derived FK and derived actions must be labelled; neither is measured contact. Check manufacturer collections for reused episodes and mirrors; embodiment count is not independent dataset count.
- **Next useful check:** Inspect a bounded source episode and its schema before adding an adapter.

Primary sources: [Official schema](https://github.com/FlagOpen/RoboCOIN), [Access manager](https://github.com/FlagOpen/RoboCOIN-DataManager), [Paper](https://arxiv.org/abs/2511.17441).

#### RT-1 / Fractal / OXE mobile-capable records

**Resource:** real-robot demonstrations / standardized mixture component. **Priority:** C1-motion. **Embodiment:** Everyday Robots mobile manipulator.

- **Coverage:** TFDS v 0.1.0 lists 87, 212 train episodes and 111.38 GiB; catalog description is tabletop manipulation with 17 objects. Must measure whether selected episodes actually move base.
- **Base and hands:** Schema contains commanded base displacement[2] and vertical rotation[1]; base_pose_tool_reached means EE pose relative to base, not robot global pose. Continuous gripper state/command, base-relative EE displacement/RPY action, RGB/language.
- **Objects and assets:** No object 6D tracks or matched meshes in documented TFDS schema. Real environments; visual object context only.
- **Access and license:** Public TFDS/OXE dataset distribution; shard-based partial inspection. Per-source OXE/RT-1 terms, not a blanket mixture license; inspect linked source release.
- **Limits and lineage:** Do not turn command deltas into measured global trajectory. Many conversions select 7-D arm action and discard base channels. Fractal/RT-1 in OXE and transformed mirrors are the same source recordings, not additional datasets.
- **Next useful check:** Inspect a bounded source episode and its schema before adding an adapter.

Primary sources: [Official TFDS schema](https://www.tensorflow.org/datasets/catalog/fractal20220817_data), [RT-1 project](https://robotics-transformer1.github.io/), [OXE collection](https://robotics-transformer-x.github.io/).

### Human interfaces and mobile manipulation systems

#### Universal Manipulation Interface

**Resource:** handheld human demonstration data. **Priority:** C1-motion. **Embodiment:** Human-held parallel gripper; no robot during recording.

- **Coverage:** Portable diverse object manipulation and bimanual collection.
- **Base and hands:** SLAM end-effector/world camera motion; no independently measured robot base. 6-DoF gripper trajectory, width and wrist video; relative/absolute frame conventions follow preprocessing.
- **Objects and assets:** No general all-object 6D/mass/friction/CAD ground truth. Real-world scenes and tools; reconstruction optional/derived.
- **Access and license:** Official community lists task-specific MP4/Zarr files and licenses. Bounded task downloads available. Original code MIT; community datasets have per-row terms.
- **Limits and lineage:** SLAM drift/failure and calibration determine pose quality; hand trajectory is insufficient to infer feasible mobile base or physical Reachy grasp. Raw MP4, processed Zarr and robot retargets share demonstration lineage.
- **Next useful check:** Inspect a bounded source episode and its schema before adding an adapter.

Primary sources: [Official implementation](https://github.com/real-stanford/universal_manipulation_interface), [Dataset registry](https://umi-data.github.io/), [Paper](https://umi-gripper.github.io/umi.pdf).

#### UMI on Legs

**Resource:** handheld demonstrations + whole-body controller framework. **Priority:** C1-motion. **Embodiment:** Quadruped with arm, trained from human-held UMI.

- **Coverage:** Prehensile, nonprehensile and dynamic tasks. The UMI registry includes 14 kettlebell-push and 500 basket-toss demonstrations.
- **Base and hands:** The deployed controller uses robot state and odometry to track a task-frame end-effector path; handheld training recordings do not directly measure the downstream robot base. Human-held parallel-gripper width and 6-DoF end-effector paths; a separate controller transfers the motion to the legged embodiment.
- **Objects and assets:** No universal moving-object pose and dynamics sequence was verified in the released UMI task records. The Isaac Gym controller environment and terrain are separate from the real demonstration object scenes.
- **Access and license:** Official code/data/checkpoint links and task-level UMI registry; no robot deployment invoked. Code MIT; registry marks these listed demos MIT.
- **Limits and lineage:** Successful robot deployment does not add object ground truth to handheld inputs. Controller-training rollouts and human task demos are different data types. UMI collection and robot controller rollouts have different provenance. Replaying one recording on another robot does not create an independent human demonstration.
- **Next useful check:** Inspect a bounded source episode and its schema before adding an adapter.

Inspected revision: code `d75c9c182d8044dadf53043612da2ffbf1936a97`.

Primary sources: [Official code](https://github.com/real-stanford/umi-on-legs), [Primary paper](https://arxiv.org/abs/2407.10353), [Task download registry](https://umi-data.github.io/).

#### Mobile UMI (2026)

**Resource:** research method / handheld data framework. **Priority:** C4. **Embodiment:** Human chest+wrist cameras; downstream mobile manipulator.

- **Coverage:** Four long-horizon tasks in the paper; asynchronous execution accounts for latency.
- **Base and hands:** A ChArUco anchor aligns visual-inertial frames. The method derives base SE(2) and chest-relative hand SE(3) from human motion; these are derived labels rather than measured robot-base trajectories. Portable wrist manipulation interface; downstream robot action must be reconstructed.
- **Objects and assets:** No released object-pose and matched-mesh corpus was verified. Real household environments; simulator assets and calibrated object dynamics were not verified.
- **Access and license:** The May 2026 paper is accessible. This audit did not establish an authoritative downloadable dataset or schema. Dataset license unknown; paper access does not establish data use rights.
- **Limits and lineage:** Do not infer a released corpus or measured robot odometry from the method description. No independence claim; deduplicate by original episode identifiers.
- **Next useful check:** Track source release; useful frame-factorization idea for mobility retargeting.

Primary sources: [Primary paper and submission](https://arxiv.org/abs/2605.20894).

#### Dobb-E / Homes of New York

**Resource:** handheld human demonstration dataset. **Priority:** C1-motion. **Embodiment:** The Stick/iPhone human collection; downstream Stretch deployment.

- **Coverage:** Homes of New York contains approximately 13 hours and 1.5 million RGB-D frames of household manipulation.
- **Base and hands:** Camera and handheld motion are recorded, rather than a measured mobile robot-base trajectory. Human demonstration motion with RGB-D; the downstream robot actions are learned mappings.
- **Objects and assets:** No measured all-object poses or matched dynamics meshes were verified. Real observations from 22 homes and 216 environments; calibrated physical scene assets are not established.
- **Access and license:** Official code and dataset documentation are public. This audit made no bulk transfer. Code MIT; dataset-specific terms and a complete payload byte inventory need confirmation.
- **Limits and lineage:** Deployment on a mobile Stretch does not establish simultaneous base-navigation labels in human training data. Reconstruction and contact labels would be derived. Raw and processed HoNY versions, and task fine-tuning subsets, can overlap. Retain original recording identifiers.
- **Next useful check:** Inspect a bounded source episode and its schema before adding an adapter.

Inspected revision: code `cf06a27e180625754c3508411117c157c615a2a7`.

Primary sources: [Official repository](https://github.com/notmahi/dobb-e), [Official documentation](https://docs.dobb-e.com/), [Paper](https://arxiv.org/abs/2311.16098).

#### TidyBot (2023)

**Resource:** preference benchmark + mobile manipulation system. **Priority:** C3-system. **Embodiment:** Mobile arm robot.

- **Coverage:** Personalized put-away choices and real object cleanup.
- **Base and hands:** The system navigates, but the released preference benchmark is not a base and arm trajectory dataset. Manipulation primitives and rule-based execution.
- **Objects and assets:** Objects, receptacles and language preferences; a dense contact-trajectory schema was not verified. Real experimental setups; no matched, calibrated MuJoCo episode package was verified.
- **Access and license:** Public code includes a benchmark directory. No dense motion corpus was verified. Inspect the repository license and exact benchmark files separately before reuse.
- **Limits and lineage:** Preference accuracy and source-robot success do not establish Reachy contact or manipulation success. No independence claim; deduplicate by original episode identifiers.
- **Next useful check:** Use the task semantics and preference evaluation; collect separate motion and object-state evidence for retargeting.

Primary sources: [Official repository](https://github.com/jimmyyhwu/tidybot), [Official project](https://tidybot.cs.princeton.edu/).

#### TidyBot++ / tidybot2

**Resource:** mobile robot demonstrations + MuJoCo collection environment. **Priority:** B3. **Embodiment:** Holonomic mobile base, single arm and parallel gripper.

- **Coverage:** Ground-level cube pickup sample demonstrations, with separate real household examples.
- **Base and hands:** base_pose[3] is episode-relative SE(2); the recorder checks that the initial base pose is approximately zero. Observations include arm_pos, arm_quat and gripper_pos. The simulator changes MuJoCo wxyz quaternions to xyzw; the dataset converter encodes action orientation as axis-angle.
- **Objects and assets:** The inspected recorder and HDF5 converter retain observations and actions, but no general MuJoCo full-state or object-qpos time series. The MuJoCo model and a cube pickup environment are provided.
- **Access and license:** The official README links sim-v1.tar.gz on Dropbox and converted HDF5 data. Payload size was not verified. Code MIT, verified from the repository license; the sample dataset grant was not separately established.
- **Limits and lineage:** The upstream example randomizes the cube at reset, so open-loop action playback is not expected to succeed every time. Its recorder keeps successful recordings; this repository must also preserve failed attempts. The raw sim-v 1 collection and its robomimic HDF5 conversion are the same recordings.
- **Next useful check:** Use the offline MuJoCo environment as a generator and add complete initial state, object trajectories and asset provenance. Do not label missing original cube poses as measured.

Inspected revision: code `c8dd23819f9cced476124a4b88538e6863ef53f7`.

Primary sources: [Official code and samplelinks](https://github.com/jimmyyhwu/tidybot2), [Writer](https://github.com/jimmyyhwu/tidybot2/blob/main/episode_storage.py), [Converter](https://github.com/jimmyyhwu/tidybot2/blob/main/convert_to_robomimic_hdf5.py), [Simulation](https://github.com/jimmyyhwu/tidybot2/blob/main/mujoco_env.py), [Inspected code license](https://raw.githubusercontent.com/jimmyyhwu/tidybot2/main/LICENSE).

#### OK-Robot

**Resource:** open-vocabulary mobile system / evaluation. **Priority:** C3-system. **Embodiment:** Stretch.

- **Coverage:** Zero-shot pick and drop in unfamiliar homes.
- **Base and hands:** The system uses iPhone scene scans and robot localization for navigation. No dense released base-action demonstration corpus was verified. Open-vocabulary grasp and drop modules.
- **Objects and assets:** Scene scans, semantic memory and localization estimates; no ground-truth object-pose time series was verified. Real home scans; no calibrated per-object mass, friction and matched simulation package was verified.
- **Access and license:** Public modular code, paper and videos. A system release is not itself a trajectory dataset. Code MIT, verified from the repository license. Dataset and scene-scan terms require separate verification.
- **Limits and lineage:** AnyGrasp and projected scene estimates are derived predictions. Videos and aggregate source success are not dense object-contact ground truth. No independence claim; deduplicate by original episode identifiers.
- **Next useful check:** Use the perception and navigation design as a reference; create separate state and contact validation data.

Primary sources: [Official code](https://github.com/ok-robot/ok-robot), [Primary paper](https://ok-robot.github.io/mfiles/paper/ok_robot.pdf), [Inspected code license](https://raw.githubusercontent.com/ok-robot/ok-robot/main/LICENSE).

#### DynaMem

**Resource:** dynamic scene-memory system / evaluation. **Priority:** C3-system. **Embodiment:** Stretch SE3.

- **Coverage:** Objects move, appear or disappear; the robot searches, navigates and manipulates under layout changes.
- **Base and hands:** The system maintains a 3D memory from robot observations. A released global-pose/action demonstration collection was not verified. Robot search, pick and drop pipeline.
- **Objects and assets:** Dynamic point-cloud memory and estimated object localization; no motion-capture-style object-pose ground truth was verified. The evaluation uses three real and nine offline scenes; scene memory is not a calibrated physical object library.
- **Access and license:** Official project, linked code and documentation are public. No complete all-trial motion/state corpus was verified. Inspect linked code and any scene-data terms; the publication website is not a dataset grant.
- **Limits and lineage:** Estimated object tracks and reported source-robot pick/drop results do not establish Reachy physical success. No independence claim; deduplicate by original episode identifiers.
- **Next useful check:** Use dynamic scene changes as a planning test; create explicit ground-truth free objects for contact validation.

Primary sources: [Official project](https://dynamem.github.io/), [Primary paper](https://arxiv.org/abs/2411.04999), [Linked StretchAI stack](https://github.com/hello-robot/stretch_ai).

#### StretchAI / official Stretch MuJoCo

**Resource:** offline simulation/collection tools. **Priority:** B2-generator. **Embodiment:** Stretch 3/4 mobile arm platforms.

- **Coverage:** Mobile teleoperation, grasping, placement and kitchen experimentation.
- **Base and hands:** MuJoCo root and joint states are available to an offline recorder. Base controls and wheel constraints need comparison with the Reachy model. Joint-position and base-velocity interfaces, with modeled grippers.
- **Objects and assets:** Full MuJoCo object states can be logged for newly created episodes. No existing demonstration corpus is assumed. Official Stretch MJCF and RoboCasa kitchen integration.
- **Access and license:** Public simulation repositories. Only offline simulation code is relevant; no hardware SDK or robot server was invoked. stretch_mujoco uses the Clear BSD license. Upstream MuJoCo Menagerie and RoboCasa asset notices still apply.
- **Limits and lineage:** Useful as an offline task generator and model-conversion reference. Real robot drivers and deployment are outside this project scope. Reusing a RoboCasa kitchen does not establish new asset or demonstration provenance.
- **Next useful check:** Compare mobile model construction and generate new state/contact logs under the Reachy model.

Inspected revision: code `d107e094cc295d92f7461bd233daef96e9e22fab`.

Primary sources: [Official MuJoCo stack](https://github.com/hello-robot/stretch_mujoco), [Official collection/planning stack](https://github.com/hello-robot/stretch_ai), [Current robot organization](https://github.com/hello-robot), [Inspected code license](https://raw.githubusercontent.com/hello-robot/stretch_mujoco/main/LICENSE).

## Relationship to the current repository and overlapping releases

The catalog snapshot inspected for this report had 333 entries and SHA-256 `ee6d4055a281ce66c7a73a39acc9dac276f38454e36ffec60fb0c722f42c4526`. Family or related-entry matches were found for RoboCasa, ManiSkill, ALOHA, OXE/RT-1, UMI, Dobb-E and TidyBot. Those matches are catalog references, not evidence that this repository already implements each mobile schema. The evidence JSON records the matched IDs. Broad token matches, especially ALOHA and OXE, should not be interpreted as exact mobile coverage.

No matching non-exclusion entry was found by those checks for many other profiled mobile resources, including BiGym, BEHAVIOR, MoMaRT, M3Bench, MoMaGen, AIRoA, Galaxea and the Habitat asset/task ecosystem. This is a comparison against the recorded snapshot, not a claim that every resource is an independent new recording collection. Catalog updates made elsewhere during validation may change the current count.

Keep these lineage relationships explicit:

- BEHAVIOR raw, processed LeRobot, rendered images, annotations and challenge versions can share recordings. Match original episode identifiers before adding counts.
- RoboCasa versions and MimicGen expansions can share source demonstrations. Synthetic descendants add generated trajectories, not independent human collections.
- AIRoA primitive episodes are parts of grouped executions. MoMa 5k v2.1 and v3 are alternate packaging, and preliminary-release overlap needs UUID comparison.
- AgiBot Alpha is a subset of Beta. Later real, synthetic and reinforcement-learning streams require separate provenance and overlap checks.
- PARTNR mini/2k/train sets are nested subsets of generated problems. HSSD/OVMM/PARTNR and Habitat/ReplicaCAD/Galactic share scene and object components.
- RT-1/Fractal in OXE, its TFDS source and converted mirrors are the same source recordings. Base channels can be lost in a conversion.
- UMI raw video, processed motion, robot retargets and controller rollout results represent distinct processing stages, not automatically distinct human demonstrations.
- TidyBot++ raw sim data and its robomimic conversion are the same recordings. Interactive and baked ReplicaCAD scenes share layout/asset lineage.

## How to turn a candidate into a defensible Reachy experiment

1. **Select and preserve one complete interaction.** Choose a rigid object with a meaningful navigate/approach/grasp/carry/place sequence, or an articulated handle task. Save the source URL, code and data revision, exact filename, checksum, original episode ID, simulator settings and all inherited asset terms. Bound acquisition and check free space before every new transfer.
2. **Decode source state with its own schema.** Recover source base, torso, arm, gripper and object frames through the saved model or documented serialized-state loader. Keep observed state separate from commands and inferred labels. Use an independent named observation or simulator forward-kinematics check where available. Record quaternion convention and velocity frame explicitly.
3. **Retain object identity and physical geometry.** Import the source collision mesh, articulation and available physical properties. Record missing mass, inertia and friction as assumptions. Keep the original object; choose a feasible Reachy grasp region and approach orientation rather than replacing it with a more convenient shape.
4. **Plan base and grasp together.** Select base poses that make the approach, closure and placement reachable without self/scene collision. Respect the target wheel/base model, torso limits, gripper aperture and collision geometry. Include carry acceleration and angular motion in the grasp evaluation because a grasp stable at rest may slip during navigation.
5. **Run dynamics without object-state intervention.** Initialize the scene once, then drive robot actuators. Never weld, teleport or overwrite the moving object's state during the physical attempt. Source object tracks remain reference targets. Source assisted grasp, a kinematic object follower or an imposed base trajectory must be disclosed and cannot establish full physical task success.
6. **Score the actual failure modes.** Log bilateral gripper contact, contact normals/forces, penetration, relative slip, lift clearance, object velocity, base/arm tracking, placement error and stability after release. Distinguish unreachable geometry, unsafe approach, missed grasp, collision, insufficient retention, unstable carry and failed placement. Save failed attempts and original parameter settings as well as successes.
7. **Broaden only after the adapter is trustworthy.** Vary rigid shapes, dimensions, graspable regions, mass, surface contact, receptacle geometry and mobile transitions while retaining source objects and recording any physics assumptions. Articulated, deformable, liquid and particle tasks require separate capability claims. Object counts, source families and scenario families should be reported separately from episode counts.

This workflow yields distinct records for accessible, fetched, parsed, retargeted, physically attempted and physics-validated data. The survey's rankings and source observations do not substitute for those run records.

## Evidence limitations

Public cards describe intended schemas; a gated archive may differ from its card. A missing license field in a host API means that the field was not populated, not that no license exists. A 401/404 on an incorrect or obsolete repository path does not establish that the official dataset is gated or unavailable. The failed `ai-habitat/HSSD` lookup is retained in the evidence for traceability; the official package is `hssd/hssd-hab`.

Repository HEAD revisions identify what was inspected, not guaranteed compatibility across packages. Where an exact payload revision, license, sample size or per-frame state schema could not be verified, the profile says so. No claim is made that every evaluated real-robot system releases its trials, that benchmark tasks are demonstrations, or that an upstream success rate transfers to Reachy.
