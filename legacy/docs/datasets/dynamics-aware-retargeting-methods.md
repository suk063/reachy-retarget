# Dynamics-aware retargeting for the existing Reachy gripper

Checked: **2026-10-06**. This is a primary-source research report and engineering
proposal. Paper results belong to their authors; none establishes a successful
Reachy rollout in this repository. The accompanying
[evidence ledger](mobile-methods-evidence.json) records inspected repository
revisions and documentation hashes. No training stack, robot SDK, or dataset
payload was installed or executed for this survey.

The practical next step is **object-relative grasp selection followed by
actuator-control optimization in the same MuJoCo model used for validation**.
Retargeted wrist poses provide an initial guess. Success must depend on the
freely simulated object's motion. Keep the Reachy fingers and each object's
geometry, mass, inertia, and friction fixed throughout a comparison.

## What the literature actually supplies

| Method | Verified mechanism and release | Useful part for this project | Important boundary |
|---|---|---|---|
| MimicGen | Object-relative subtask trajectory transfer, then controller execution in the environment | Segment a demonstration by interacting object and contact phase | Transforming waypoints alone does not establish success |
| DexMimicGen | Dexterous and coordinated bimanual data generation; official simulation environments and datasets released | Preserve simultaneous hand phases and object identity | A dexterous-hand solution is not automatically feasible for Reachy |
| MoMaGen | Mobile-base placement and navigation combined with object-relative manipulation | Jointly select base pose, arm reach, and approach | Official stack uses OmniGibson/BEHAVIOR, not this MuJoCo scene |
| PhysicsGen | Kinematic seed followed by sampled forward-dynamics control refinement | Short action splines and object-dominant trajectory cost | Published implementation uses Drake; code link says forthcoming |
| MuJoCo MPC | Native CPU optimization, including derivative-free predictive sampling | Most direct local implementation route | Research framework; no established real-time budget for our model |
| SPIDER | IK followed by physics sampling and a virtual-contact guidance curriculum | Contact-aware initialization and refinement | GPU backends; assisted optimization must be followed by unassisted validation |
| OmniRetarget | Constrained interaction-mesh deformation | Preserve object/hand/environment relationships in the seed | Kinematic retargeting and learned dynamic tracking are separate stages |
| InterMimic | RL teachers repair/improve HOI, then policy distillation | Object/contact rewards and failure taxonomy | GPU learning route, not a small CPU-only patch |
| WristMimic | Body/wrist guidance with learned finger control | Do not force incompatible human finger configurations | Requires learned dynamics controller |
| Weave | Contact-aware retargeting, approach completion, whole-body policy learning | Include the approach before the grasp | Newly published learned humanoid system; derived rollouts are not new human captures |
| DexForge | Contact-aware kinematics and force-aware differentiable dynamics | Separate feasible contact construction from control refinement | Uses reconstructed Gaussian collision geometry; not a drop-in fixed-mesh MuJoCo optimizer |

### Object-centric transfer: MimicGen and DexMimicGen

MimicGen's documented interface records end-effector poses, object poses,
subtask boundaries, target poses, and gripper commands. Generation transforms
subtasks relative to an object and executes the resulting target actions through
the environment. Its implementation therefore distinguishes requested motion
from the state actually reached. The documentation also exposes per-attempt
success information. [MimicGen generation API](https://mimicgen.github.io/docs/modules/datagen.html)

DexMimicGen extends this research direction to dexterous and coordinated bimanual
tasks. Its project includes mobile bimanual BiGym examples as well as other
simulation tasks. The inspected official repository advertises simulation
environments, demonstration datasets, and policy-training configurations; this
does not establish that every paper generation component is released. Code uses
the repository's NVIDIA license and datasets are identified as CC BY 4.0.
[Project](https://dexmimicgen.github.io/),
[official repository](https://github.com/NVlabs/dexmimicgen)

**Proposed use:** preserve a phase-to-object association instead of treating all
hand trajectories as global wrist targets. For a source object `O` and gripper
reference `G`, form `T_OG = inverse(T_WO) @ T_WG`. Find a Reachy-compatible contact
transform near that reference, then use the target object's world pose to build
the approach seed. For two-handed carry or handover, synchronize the two contact
phases; do not retime each arm independently.

### Mobile manipulation: MoMaGen

MoMaGen separates object-relative manipulation from transitions between work
areas. The project describes hard reachability constraints and softer visibility
criteria when choosing base poses, with base/torso planning between subtasks.
The official installation requires OmniGibson and associated BEHAVIOR resources.
This is evidence for the planning decomposition, not a reason to import that
stack into this offline MuJoCo project. [Project](https://momagen.github.io/),
[installation](https://chengshuli.github.io/MoMaGen/installation/),
[repository](https://github.com/ChengshuLi/MoMaGen)

Release check: the live project page links public code and documentation. The
inspected source revision is `5da621667669b15b97d7a220d086283b3c8ad0e5`; its tree
contains the `momagen` package, setup file, documentation, and simulator/training
submodule references. Earlier forthcoming notices on another landing page do
not describe this inspected tree. A downloadable demonstration package and a
working local installation were not verified.

**Proposed use:** add a planar base variable only when the simulation contains a
supported base dynamics/controller model. Sample base placements that keep the
unchanged object reachable along the complete grasp/carry/place path. Check the
combined robot-plus-carried-object swept volume. A fixed pedestal cannot claim
mobile manipulation because a human root path was projected into the scene.
Likewise, prescribed base teleportation supports a kinematic preview, not a
physical mobile-manipulation result.

### Forward-dynamics refinement: PhysicsGen

PhysicsGen refines kinematically retargeted demonstrations using sampled control
trajectories and forward dynamics. Its appendix gives a useful compact reference:
six action knots with linear interpolation, 1.25 or 2.0 second horizons, and CEM
with 50 candidates and five elites. Reported trajectory weights favor object
tracking over robot tracking, with an increased terminal weight. The simulation
is Drake at 200 Hz. Those settings are experimental choices for its robots,
not validated defaults for MuJoCo or Reachy. The official project still labels
code as forthcoming. [Paper and appendix](https://arxiv.org/html/2502.20382v1#A1.SS1),
[project](https://lujieyang.github.io/physicsgen/)

**Proposed use:** keep the low-dimensional control parameterization, prioritize
object outcome, and measure the local search cost. Do not copy the paper's
physical domain randomization into a benchmark that requires unchanged objects.
Optimize grasp pose, approach, closure schedule, arm controls, and bounded timing;
object material parameters are inputs, not optimization variables.

### CPU implementation: MuJoCo MPC and native rollout

The official MuJoCo MPC repository supports macOS and Ubuntu and implements
predictive sampling alongside gradient-based planners. It includes a small
Python predictive-sampling example. The full framework has additional C++ task
integration and build requirements, so a local prototype can borrow its search
structure without introducing the complete application.
[MuJoCo MPC](https://github.com/google-deepmind/mujoco_mpc),
[pinned Python sampler](https://github.com/google-deepmind/mujoco_mpc/blob/ff572a21e7c2bf9fda62e1862a758da7e9a8719b/python/mujoco_mpc/demos/predictive_sampling/predictive_sampling.py)

MuJoCo's Python `rollout` runs a native simulation loop. Its default control input
is actuator controls. The documented thread-pool interface uses one `MjData`
per worker and supports batches of initial states and controls. Initial states
use the `mjSTATE_FULLPHYSICS` layout. Returned trajectories must be checked for
divergence; the persistent pool is not safe for concurrent use by independent
callers. This removes Python step-loop overhead without requiring a training GPU.
[Official Python rollout documentation](https://mujoco.readthedocs.io/en/stable/python.html#rollout)

**Proposed use:** benchmark one physics step and a batch of short candidate
rollouts first. Keep an explicit serial implementation for correctness checks.
Do not assert real-time control based on a paper's hardware or a different model.
Record planning wall time separately from trajectory duration and playback speed.

### Physics-based human transfer: SPIDER and learned alternatives

SPIDER combines a kinematic stage with physics sampling and gradually reduced
virtual contact guidance. The release documents MuJoCo-Warp and other GPU
backends. Its September 2026 trajectory release explicitly describes validation
by saved-state replay rather than equivalence under control replay. Its ARCTIC
processing also extracts short windows and simplifies the articulated object to
its rigid bottom part. These are material provenance and validation distinctions.
[SPIDER paper](https://arxiv.org/abs/2511.09484),
[official repository](https://github.com/facebookresearch/spider)

**Proposed use:** contact guidance may help explain an initialization strategy,
but final trials here must have zero artificial object guidance, no weld, and
no overwritten object state. Do not import a successful state sequence and call
it a successful actuator rollout. Do not silently convert articulated tasks
into rigid-object tasks.

OmniRetarget preserves interaction relationships through constrained deformation
of an interaction mesh. Its project separately discusses retargeted motion and
learned tracking policies. It is useful for a contact-preserving kinematic seed;
that seed is not itself a physical demonstration.
[OmniRetarget project](https://omniretarget.github.io/),
[official implementation](https://github.com/amazon-far/holosoma)

InterMimic uses reinforcement-learning teachers and distillation to produce
human-object control. Its official implementation uses Isaac simulation stacks.
WristMimic constrains body/wrist motion while learning finger behavior from
interaction objectives, illustrating why copying human finger angles can be
unnecessary. Both are relevant research directions, but neither is a verified
CPU Reachy controller. [InterMimic](https://github.com/Sirui-Xu/InterMimic),
[WristMimic](https://wongyun-yu.github.io/wristmimic/)

Two recent preprints merit tracking. Weave, submitted September 15, 2026,
combines contact-aware retargeting and approach completion with a whole-body
dexterous policy; its simulated rollout release is derived data. DexForge,
submitted October 5, 2026, constructs robot-adapted contacts and refines controls
in a differentiable simulator using reconstructed Gaussian object models, then
reports transfer to MuJoCo. Its collision representation differs from the fixed
original mesh required here. Neither paper's reported success rate measures this
repository. [Weave](https://arxiv.org/abs/2609.16683),
[DexForge](https://arxiv.org/abs/2610.06331)

## Proposed implementation with fixed objects and fixed fingers

Everything in this section is an engineering recommendation, not an implemented
or experimentally confirmed result.

### 1. Establish a model and data contract

Give every manipulated and background object a stable ID. Record its mesh and
collision hashes, scale, root transform, inertial properties, friction, support
surface, joint definitions, and source of each field. A missing measured mass
must remain a documented simulation assumption. Once the baseline model is
chosen, freeze these values for all optimizer candidates and comparisons.

Visual texture fidelity and contact geometry fidelity are separate checks. Keep
the existing textured Reachy model, but verify that the actual pad collision
surfaces align with the visible fingers. A mesh-origin or finger-joint error
cannot be compensated by a favorable grasp-search score. Record any required
model correction before comparing runs; never change collision geometry in the
middle of an optimization to make a trial succeed.

Preserve native source time and validity intervals. A trajectory needs a shared
metric world, hand/body poses, object poses, and object geometry. Do not silently
fill an unobserved object track with a constant pose. Retain articulated joint
state when the demonstrated action depends on articulation.

### 2. Search contact configurations before searching long motions

Generate candidate contact regions on the unchanged object. Prefer pairs whose
separation fits Reachy's actual opening and whose local normals allow opposing
pad forces. Include object center of mass and expected gravity/inertial torque in
the ranking. Test the real finger forward kinematics and pad surfaces, not an
ideal parallel-jaw width alone. Reject table strikes, finger intrusion, palm
collisions, arm-limit violations, and approach paths through the object.

Antipodal contacts and friction-cone reasoning are useful geometric filters, but
two point contacts do not automatically supply every 3D wrench. Finite pad
contact and torsional resistance still matter. This distinction follows standard
grasp mechanics; it is especially relevant for a can that rotates or slides
despite touching both fingers.
[MIT manipulation notes on grasping](https://manipulation.csail.mit.edu/clutter.html)

Use a short dynamic close/lift/hold experiment to eliminate candidates that look
good geometrically but slide. Only the best survivors enter a full transport and
placement search. A feasible grasp can differ from the human's wrist pose while
preserving which object is moved and where it goes.

### 3. Optimize actuator trajectories in short windows

Parameterize a small set of bounded action knots, plus grasp offset, closure
onset, lift onset, and phase durations. Position-control targets are acceptable
only through configured actuators with declared force limits. Evaluate each
candidate by forward integration. Keep the incumbent in every search generation
so noise cannot discard the best known trajectory.

An initial CPU prototype can use 24-64 samples, 4-8 elites, and a roughly
0.5-2 second horizon. These ranges are **proposed search budgets** to benchmark,
not measured performance. Begin with fewer active parameters around one grasp;
increase budget only when diagnosed failures justify it. Search short contact
transitions before long carry phases.

A conceptual objective is:

```text
hard invalid: divergence, model-contract violation, forbidden collision,
              object-state intervention, actuator-limit violation

soft cost = object position/orientation tracking
          + terminal task error
          + contact slip and unwanted separation
          + excessive penetration or force
          + non-target object disturbance
          + action variation and joint-limit proximity
          + modest robot-reference error
```

Normalize each residual by a declared physical tolerance before weighting it.
Orientations require rotation-space distance; direct subtraction of quaternion
components is sign-sensitive. Use task-specific symmetries only when stated in
the evaluation protocol. A cylinder may have an unimportant yaw for placing,
while its tilt remains important for a stable carry.

Receding-horizon refinement starts from the current integrated state. Copying that
state into independent planning rollouts is valid simulation branching; replacing
the live object's trajectory with its reference is not. Apply only the selected
control prefix, integrate, observe, and replan. Final open-loop replay is a useful
additional test and must be reported separately from feedback execution.

```text
freeze model and source metadata
construct feasible Reachy grasp seeds
for each seed:
    create independent initial-state rollout
    optimize close -> lift -> hold through actuator controls
    retain failure evidence and stable candidates
for each surviving candidate:
    optimize transport -> support contact -> release
    replay from initialization using the declared controller
    compute physical and task metrics for every object
```

### 4. Measure contact quality rather than contact count

For every contact, retain geom IDs, object ID, pad identity, signed distance, and
force. MuJoCo's `mj_contactForce` returns force/torque in the contact frame; its
first axis is the normal. Convert frames explicitly. The contact distance and
friction dimensionality have specific simulator meanings and must not be
interpreted from array position guesses.
[API reference](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce),
[contact computation](https://mujoco.readthedocs.io/en/stable/computation/index.html#contact)

For a grasp trial, distinguish opposed pad contact from a finger and palm touching
the same side. Compute relative contact-point velocity, including angular terms
`v + omega x r`, before taking its tangent-plane component. Count slip only during
the intended grasp/hold phase: sliding is often desired during a push or insertion.
Report force magnitude and friction utilization as diagnostics, with the actual
contact-pair coefficient and friction model. One threshold must not cover all
contact dimensions and task phases.

Require stable contact and low relative motion before lifting; require supported
placement and settling before release. These event checks can prevent a fixed
clock from lifting before closure completes. Keep timing changes bounded and
record their ratio to source duration. A tenfold slowdown must not be hidden in
a playback setting or described as equivalent task performance.

### 5. Extend to multiple objects and mobile carry honestly

Instantiate every relevant object and support in the physical scene. Objects
that the robot can contact should respond according to their declared dynamics;
the optimization should not receive free support from an omitted or frozen
moving body. Track the intended manipulated object separately from distractors,
containers, articulated links, and temporary supports.

Start with independent grasp-and-place tasks across different shapes. Then add
two-object transfer, placement near clutter, support exchange, bimanual carry,
and finally mobile transport. A task with two objects present is weaker evidence
than a task requiring an interaction between them. Record that distinction.
For mobility, use collision-aware base paths and base dynamics; otherwise report
an arm-only manipulation with a fixed base.

Do not turn a large human-carried object into a smaller or lighter one to create
Reachy success. Mark width, payload, workspace, missing support-person forces,
or unsupported articulation as an infeasibility reason. Valid rejection is a
useful result across heterogeneous datasets.

## Evidence required before claiming dynamics-aware success

| Evidence | Minimum artifact | Failure example it catches |
|---|---|---|
| Data provenance | Source/revision/checksum, parent capture ID, segment interval, units and frame transforms | Counting crops or retargeted hands as new demonstrations |
| Fixed physical model | Mesh/inertia/friction/actuator hashes and assumptions | Improving success by changing the can or increasing friction |
| Independent execution | Initial state, control stream/controller, seeds, simulator version, all trial outcomes | Replaying saved object states instead of producing motion |
| No hidden assistance | Zero artificial object force, no weld/mocap drive/state overwrite after initialization | Object following the hand without a physical grasp |
| Grasp | Correct-pad contact, slip, force, object-relative drift, lifted height and hold duration | Touching the object while it remains table-supported |
| Transport | Tracking error and peak error, collision/penetration, dropped-object detection | Good final pose after a mid-trajectory failure |
| Placement | Support, release, settling, final pose and velocities | Declaring success while still holding the object |
| Multiple objects | Per-object trajectory and disturbance metrics, task identity | Grasping the wrong object or benefiting from unmodeled support |
| Mobility | Dynamic base/controller, path length, swept-volume checks | Animating base poses through obstacles |
| Throughput | Simulated duration, source ratio, wall time, candidate count | Confusing fast playback with fast optimization |

Use held-out captures and object/scenario groups in addition to optimized
training examples. Save all failed attempts and summarize the denominator before
presenting the best video. Keep availability, acquisition, normalized data,
kinematic retargeting, and physical validation as separate statuses.

## Integration boundary with this repository

At inspection, explicit normalization dispatch in
[`normalize.py`](../../reachy_retarget/normalize.py) included robomimic, LeRobot,
HUMOTO, and ParaHome paths; these do not establish support for every survey
dataset. The existing [`physics.py`](../../reachy_retarget/physics.py),
[`contact.py`](../../reachy_retarget/contact.py), and
[`retarget.py`](../../reachy_retarget/retarget.py) are natural integration points.
This report does not modify them or certify concurrent implementation work.
Actual pass/fail status must come from saved local evaluation artifacts.

The [mobile human-object survey](mobile-human-object-survey.md) identifies data
that can test each proposed extension and data that cannot provide the required
dynamic object state.
