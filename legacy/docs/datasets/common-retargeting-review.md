# Common retargeting logic and dataset integration review

Historical review, written before the common-solver implementation. See the
[2026 method update](retargeting-2026-update.md) for the subsequent design and
implementation changes; findings below describe the earlier code snapshot.

Reviewed on 2026-10-06 against the then-current working tree and the linked primary
sources. This is an architecture and integration review, not a new implementation
or physics experiment. The existing survey evidence ledgers retain inspected
upstream revisions and hashes; pin the selected release again before acquisition.

**The proposed datasets can inform a shared retargeting system, but none is a
drop-in addition to the current physical pipeline.** RoboCasa365 and BiGym remain
the best first integration candidates. Their native MuJoCo models reduce scene
translation work; they do not remove the need for new input adapters, articulated
scene support, contact planning or Reachy validation.

“Same logic” should mean the same state interpretation, candidate generation,
optimization and controller for all supported tasks. Goals and physical inputs
can differ. Grasp pose, force, base motion and timing should be outputs of that
algorithm, rather than manually selected recipes indexed by a task name.

## Findings in the current implementation

| Priority | Finding and evidence | Consequence and required change |
| --- | --- | --- |
| High | [Scene extraction](../../reachy_retarget/object_scene.py), `_source_scene`, removes unselected top-level bodies containing joints. Selected objects must have a free root joint. `_resolved_source_values` rejects all remaining non-free joints. | Doors and drawers can be omitted or rejected. Preserve fixture articulation, passive joint state, limits, damping and contact geometry. Identify the source robot explicitly instead of treating other dynamic bodies as disposable. Reject incomplete scene coverage before claiming success. |
| High | [Planning](../../reachy_retarget/dynamics.py), `TASK_OBJECTS`, `source_grasp` and `plan`, assumes one active object, the right hand, a source lift, and the first close/open pair. | Pushing, pulling, repeated grasps, handovers and bimanual tasks cannot be represented generally. Introduce explicit interaction phases and stable body/hand references. The same phase interpreter must process all tasks. |
| High | [Validation](../../reachy_retarget/dynamics.py), `rollout`, always requires a bilateral lifted grasp and 95% bilateral carry contact. Final contact requirements also branch on task names. | A physically correct push or drawer opening would fail by definition. Apply grasp/carry gates only to the declared contact modes, while retaining shared collision, actuator, finite-state and completion checks. Preserve the existing grasp protocol and thresholds as a separately versioned baseline. |
| High | [Planning](../../reachy_retarget/dynamics.py), `plan`, branches on Can, Square and Threading names to select grasp orientation. [Frozen settings](../../configs/dynamics-validation-v2.json) select force, offsets, feedback options, release behavior and source windows per task. | This is a shared implementation with task-specific recipes, not the requested general method. Infer grasp geometry, support transitions and control parameters from canonical inputs. Moving the branches into JSON alone would not solve this. |
| Medium | [Search](../../reachy_retarget/dynamics.py), `objective` and `search`, ranks a bounded coordinate search. Object-reference SE(3) error is logged but is not in the search objective. | Failure gives limited guidance for insertion or long tasks. Add continuous object/relative-goal and articulation residuals, then refine actuator commands through a common predictive rollout optimizer. A lower objective must never override a failed physical gate. |
| Medium | [LeRobot normalization](../../reachy_retarget/normalize.py), `lerobot`, currently emits an empty object map and then requires object tracks. | RoboCasa's native LeRobot packaging does not make it supported by this loader. Its extra model/state files need a dedicated decoder. Missing local support is not evidence that the upstream dataset is robot-only. |

The existing provenance, source-preserving rigid-object scene conversion,
actuator-only rollout, saved failures and independent replay audit are useful
building blocks. The [executed baseline](../dynamics-validation.md) is still
6/20 passes with per-task settings: 4/6 development episodes and 2/14 additional
episodes. Lift/Stack use declared stationary source windows. These results do
not establish a common solver or unseen-object generalization.

## Proposed shared contract

The common flow is:

```text
Explicit acquisition -> source adapter -> canonical episode + scene + goal
    -> interaction phase/contact inference -> geometry-based candidate search
    -> shared base/arm/gripper optimization -> actuator-only MuJoCo validation
```

Source adapters may interpret dataset-specific files, units, clocks, frames,
action semantics and publisher success conditions. They must preserve the
meaning and provenance of observations. After conversion, the motion solver
should not receive dataset or task names as decision inputs.

The canonical input needs four components:

1. **Scene and robot:** rigid bodies, complete articulated trees, meshes and
   textures, collision geometry, initial poses and velocities, inertial/contact
   properties, Reachy limits and base model. Record missing or assumed physical
   values instead of silently supplying ground truth.
2. **Reference motion:** time-aligned hand/object/base trajectories and object
   joint states, validity masks, contacts and forces when available. Distinguish
   measured state, source-simulator reconstruction and derived estimates. Human
   pelvis motion and base commands are not observed Reachy base poses.
3. **Task specification:** goals such as object pose, relative placement,
   insertion alignment or drawer joint displacement; tolerances; required
   support, hold or release; allowed contact relations and phase dependencies.
   Describe the desired outcome without encoding a special trajectory recipe.
4. **Interaction evidence:** hand/object proximity and relative motion,
   gripper state and available contact signals, with uncertainty. Infer grasp,
   carry, push, pull, supported release and repeated interactions through shared
   rules. Missing grip labels in human data cannot be treated as an open hand.

The same candidate generator should use surface geometry, normals, available
gripper aperture, source contact evidence and reachability to propose grasps.
Object symmetry or handle direction should come from geometry or documented
object metadata. Reachy should select a stable region on the unchanged object.

Use a common objective combining object/relative-goal tracking, contact-mode
consistency, slip, collision, effort, smoothness and execution duration. Normalize
units and scales with one declared rule. Let force and timing adapt within fixed
robot limits using mass/friction evidence and observed slip; retain uncertainty
when those properties are unknown. Keep known object parameters fixed during
optimization. There is no basis to guarantee success on objects outside Reachy's
grasp, reach or payload capability.

A shared short-horizon predictive optimizer is a reasonable next method, with
geometry-based initialization and a longer-horizon base/phase plan. MuJoCo MPC
provides predictive sampling and other shooting planners as an implementation
reference, not an already integrated solution. Benchmark compute cost and
execution time before making speed or real-time claims.
[Official MuJoCo MPC repository](https://github.com/google-deepmind/mujoco_mpc)

Mobile support also requires base footprint/environment constraints. The current
ideal planar actuator model does not validate wheel traction. Do not copy
floating-base height/roll/pitch into Reachy or equate human root travel with
feasible navigation. Apply frame transformations consistently to the entire
scene and reference motion.

## Integration decisions for the named resources

The decisions below are engineering judgments based on the cited schemas and
release evidence. No new payload or Reachy physical pass is established here.

| Resource | What can be used | Required work and recommendation |
| --- | --- | --- |
| **RoboCasa365** | Native extras include episode MJCF, raw simulator states and scene metadata alongside LeRobot observations/actions. | **First integration.** Decode `model.xml.gz`, `states.npz` and `ep_meta.json`; resolve pinned assets and reconstruct object/base states using model names. Preserve kitchen articulation. Begin with one complete rigid-object transfer task and verify actual base travel before calling it mobile. [Native layout](https://robocasa.ai/docs/build/html/datasets/using_datasets.html) |
| **BiGym** | MuJoCo household scenes and bimanual demonstrations; lightweight saved demos retain actions and reset metadata. | **Second integration.** Replay the exact native version/seed first and export resulting object/joint states as reconstructed references. Upstream warns of unsuccessful replays. Adapt floating-base motion to Reachy's permitted base state. Intercept `DemoStore` auto-download through explicit fetch; exclude reach-only tasks. [Repository](https://github.com/NeuracoreAI/bigym), [saved demo schema](https://github.com/NeuracoreAI/bigym/blob/master/demonstrations/demo.py), [loader](https://github.com/NeuracoreAI/bigym/blob/master/demonstrations/demo_store.py) |
| **BEHAVIOR** | Raw replay data supports rich household task/state reconstruction. | **Later, rigid-object subset.** Translate OmniGibson assets and mechanics to MuJoCo and audit fidelity. Collection and evaluation use assisted grasping, so source success cannot establish Reachy friction-contact success. Particle, deformable and material-state tasks require additional models. [Dataset](https://behavior.stanford.edu/challenge/dataset.html), [maintainer confirmation](https://github.com/StanfordVL/BEHAVIOR-1K/issues/2245#issuecomment-4597125008) |
| **MoMaGen** | Released mobile manipulation generator and object-relative planning ideas. | **Method reference now.** Public code does not establish an acquired source-demo corpus. Verify seed availability, simulator dependencies and asset terms separately before treating generated results as dataset coverage. [Released code](https://github.com/ChengshuLi/MoMaGen), [installation](https://chengshuli.github.io/MoMaGen/installation/) |
| **M3Bench** | Scene/robot URDFs, initial object configuration and planned robot joint trajectories. | **Base/arm planning evidence.** The documented schema does not establish continuous moving-object state or physical grasp success. Inspect a sample before a dynamics adapter; never invent object tracks by assuming a rigid attachment. [Publisher schema](https://huggingface.co/datasets/M3Bench/M3Bench) |
| **MeLLO** | Locomotion with large-object interaction, motion capture and measured interaction signals. | **Human/mobile priority after reconstruction.** Fit object poses from markers, align sensor frames/clocks and obtain matching collision geometry and dynamics. Check grasp and payload feasibility without shrinking objects. [Author paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11939698/), [data/processing repository](https://github.com/nluttmer1/MeLLO-Data-Library) |
| **FORCE** | Human/object motion and object meshes with resistance variation. | **Conditional adapter.** Resolve dataset terms, mesh origins, units, wrist orientation and total mass. Added resistance is not total mass; force encodings must remain labeled as derived. [Release](https://github.com/xz6014/FORCE_dataset), [paper](https://arxiv.org/html/2403.11237v2) |
| **HIMO** | Multi-object human motion with object tracks and meshes. | **After access and multi-object support.** Follow the application/terms process, validate pose/mesh frames and clocks, then infer multiple contact phases. Multi-object activity does not itself establish locomotion. [Dataset repository](https://github.com/LvXinTao/HIMO_dataset) |
| **Object-aware KIT subsets** | Selected human motions, associated objects and models. | **Selective adapter.** Check each motion's object tracks and access conditions. The public FAQ says the API is disabled; individual web downloads remain available. Preserve recording lineage across subsets and exclude body-only imports. [Official FAQ](https://motion-database.humanoids.kit.edu/faq/) |

Detailed asset/license, missing-field and release notes remain in the
[robot survey](mobile-manipulation-survey.md),
[human survey](mobile-human-object-survey.md), and their
[robot](mobile-manipulation-survey-evidence.json) and
[human/method](mobile-methods-evidence.json) evidence ledgers. Research profile
counts include generators, assets and overlapping releases; they are not counts
of ready-to-retarget independent datasets.

## Implementation sequence and acceptance criteria

1. **Separate contracts from motion decisions.** Preserve the existing baseline
   and extract source decoders and goal evaluators. Introduce a versioned scene,
   goal and contact specification. Replace the six current task recipes with
   the same candidate generator and optimization policy before adding many
   dataset names. Retain earlier failures and report any regression.
2. **Add articulated scenes and two native pilots.** Acquire bounded native
   RoboCasa and BiGym samples explicitly, keeping at least 50 decimal GB free
   after extraction and planned outputs. Verify source replay first; compare
   body transforms, articulation, mass/inertia, collision/contact properties and
   assets after conversion. Preserve relevant passive mechanical constraints
   without transferring assisted hand-object attachments. Use a complete
   transfer and an articulated interaction to expose schema limitations early.
3. **Extend common contact and mobile optimization.** Exercise grasp/carry/place,
   push/pull, articulated interaction and repeated contacts with the same phase
   interpreter. Add bimanual coordination and obstacle-aware base planning as
   explicit capability increments. Then introduce selected human data and
   cross-simulator sources after their state/asset contracts pass.

The resulting implementation should pass these checks:

- Changing task/dataset display names, or consistently renaming body IDs, does
  not change controls given identical canonical inputs and random seed. Changing
  a physical property or goal changes the result through the shared model.
- Hold out object families and scenarios, not just episodes used for tuning.
  Freeze optimizer settings, budget, initialization rules and gate definitions
  globally. Per-episode optimization is allowed under that same procedure;
  manually chosen per-task force or yaw presets are not evidence of this goal.
- Report source availability, adapter support, source replay, retargeting and
  physical validation separately, including failures and unsupported inputs.
  Report planning wall time, execution duration and retiming alongside success.
- Use contact-mode-appropriate gates declared before evaluation. Retain existing
  grasp strictness and report new modes/protocols separately. Track object and
  goal error continuously, with task-relevant success tolerances.
- Advance all moving objects and passive articulation through dynamics. During
  a physical attempt, use robot actuators only; never weld, teleport or overwrite
  moving-object state. Keep independent control replay and asset/source hashes.
  Source-state reconstruction is a separate preparation operation, not validation.
- Preserve object shape and physical properties, Reachy visuals and the local
  visual tripod correction. Do not modify sibling repositories or import robot
  SDKs. Assumed human-object dynamics and the ideal planar base must remain
  visible limitations of any reported pass.
- Apply one documented initial-state validity policy. Preserve complete source
  recordings and report cropped windows separately; never count them as extra
  demonstrations or silently remove difficult motion to obtain a pass.

The recommended next deliverable is a common solver for the existing rigid-object
cases plus a faithfully reconstructed native mobile pilot. A broad dataset catalog
alone would not satisfy the requested generalization.
