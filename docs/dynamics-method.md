# Implemented dynamics-aware retargeting

Checked against the implementation on **2026-10-06**. This document describes
the method and acceptance rules, without asserting a number of successful
episodes. Per-attempt artifacts and the final benchmark report are the evidence
for individual outcomes.

The implemented method builds an object-relative Reachy motion, executes it in
MuJoCo, and uses the measured physical outcome to select bounded controller and
grasp parameters. The freely simulated object determines success. This is a
task-specific, simulation-guided coordinate search; it does not implement MPC,
CEM, reinforcement learning, or a general contact planner. A feasible motion is
currently sought within a small family of right-gripper motions. Unsupported
tasks are rejected, and supported tasks can still fail every attempted candidate.

Implementation: [planner, controller, search and gates](../reachy_retarget/dynamics.py),
[source scene construction](../reachy_retarget/object_scene.py),
[independent actuator replay](../reachy_retarget/dynamics_audit.py), and
[ManiSkill task adapter](../reachy_retarget/dynamics_maniskill.py).

## Source contracts and physical scenes

The robosuite path accepts the inspected Panda OSC_POSE exports at 20 Hz:
PickPlaceCan, Lift and NutAssemblySquare with source version 1.5.1, and
MimicGen Stack_D0 and Threading_D0 with source version 1.4.1. It checks the
controller schema, exact object set, free-joint mapping, world-coordinate poses,
unit quaternions, valid samples, action shape and timing. Unverified versions and
task geometry configurations do not inherit success thresholds from a task name.

The separate ManiSkill path accepts the pinned Panda `pd_joint_pos` PickCube-v1
export. It preserves the source's T actions and T+1 measured states. Its derived
gripper command reverses the source sign and holds the last command at the
terminal state; it does not append an invented source action. See the
[adapter contract](maniskill-adapter.md).

For recorded robosuite scenes, the source collision geometry, object masses,
inertias, friction and contact parameters are retained and checked after merging
with Reachy. The complete environment and all retained objects receive one rigid,
gravity-preserving placement transform. Scene heading changes therefore move the
table, fixtures, objects and targets consistently. They never resize an object
or move it independently of its source environment to make a grasp work.

ManiSkill has no recorded MJCF. Its pinned source task, material and geometry
evidence is used for an explicitly labeled PhysX-to-MuJoCo reconstruction.
Equal parameter values do not imply equal contact dynamics between those engines.
The source table geometry is preserved through convex decomposition rather than
replaced with a collision plane. The detailed evidence, conversion assumptions,
and source-fidelity checks are in [object scenes](object-scenes.md).

An object's recorded pose at the selected interval start is assigned once in the
reset scene. After
reset, there are no object pose assignments, object actuators, virtual object
forces, welds, mocap attachments or gravity compensation on objects. Every
candidate starts a new physical episode. The Reachy gripper geometry and object
physical parameters remain fixed across candidates.

### Optional stationary pregrasp interval

`source_start_s` defaults to zero, retaining the complete source interval.
For robosuite recordings, `source_window` can select the first recorded frame at
or after a requested later start. It rejects a negative/out-of-range start or a
prefix containing any closed-gripper command, including the selected first
frame. Every object's two adjacent motion intervals around that frame must have
translation speed at most 0.01 m/s and rotation speed at most 0.2 rad/s. This is
a local stationarity check around the new start, not a claim that the entire
omitted prefix was stationary. ManiSkill T+1 recordings cannot be cropped by
this helper.

The selected poses supply only the new episode's reset state. Subsequent object
motion remains unconstrained dynamics. The helper creates a derived array view
and rebased clock; the original source arrays and HDF5 file are unchanged. The
report records the original start frame/time and both original and window grasp
anchor indices. A cropped interval remains the **same source episode**, never
an additional independent demonstration or a complete-interval success.

The current Lift and Stack configurations use a 0.25 s start after a proposed
0.2 s start failed the stationarity check. Their complete-interval failures remain
in [`frozen_matrix_v1`](../runs/object-matrix/runs/dynamics/frozen_matrix_v1),
alongside separate later attempts. Interval selection and its outcome must be
reported together.

## Motion construction and execution

1. **Locate a demonstrated grasp.** The source must show a closed-gripper lift.
   The first qualifying lifted state supplies an object-relative wrist reference.
   The selected Reachy grasp point is either this reference or the center of the
   source collision region nearest it; PickCube uses the original box center.
   This is a grasp-region heuristic, without an antipodal or force-closure proof.
2. **Construct Reachy targets.** Candidate depth, height, tilt, yaw and optional
   object-local x/y grasp offsets place the existing gripper around that region.
   Object-axis alignment is task dependent.
   The default trajectory seed follows the source object's SE(3) trajectory with
   a constant object-to-hand transform. The pregrasp and post-release withdrawal
   are reconstructed for Reachy. The alternative source-end-effector attachment
   mode remains available. Neither target sequence is assigned to the object.
3. **Solve inverse kinematics.** The right arm is active; the left arm stays at its
   rest reference. An optional planar base adds x, y and yaw to the IK variables.
   The solver uses joint margins and optional maximum base-x and minimum base-y
   bounds. Several deterministic initial arm seeds reduce dependence on one IK
   branch.
4. **Retime the command.** A source time multiplier and a closure pause are
   followed by velocity-dependent dilation. Planning caps are 0.4 m/s for each
   planar base component, 1.2 rad/s yaw and 0.65 rad/s arm joints. Plans requiring
   a dilation above eight are rejected. The final gripper state is held for two
   additional simulated seconds. The saved `source_time_scale`,
   `velocity_dilation` and duration explain each attempt's speed; visualization
   playback speed is a separate setting.
5. **Execute actuator commands.** MuJoCo advances at 0.002 s per step, with five
   steps per 100 Hz controller interval. Robot position actuators track the joint
   reference. A bounded integral correction compensates persistent tracking
   error by changing robot actuator references only. The original reference,
   applied reference, final actuator controls and resulting states are recorded.

The robot model includes ideal robot-body gravity compensation and bounded
position servos. These are simulation assumptions, not measured hardware
actuator identification. The object remains subject to gravity and contact.

### Optional contact-force feedback

`force_feedback` defaults to false. When enabled, measured contact normal force
changes the gripper position reference toward the candidate's requested force;
reference changes and the closure range are bounded. Force is measured from
MuJoCo contact constraints, not from a hardware sensor. The implementation uses
each distal finger's maximum normal force over the preceding five substeps.
By default the larger finger value is used; `balanced_force` selects the smaller
of those two interval maxima. The original feedback change is
`clip(0.01 * (measured_force - requested_force), -0.015, 0.015)` per controller
interval. With both `force_feedback` and `soft_force_feedback` enabled, free-space
closure advances by 0.015 per interval. Once either distal finger has a positive
normal-force contact, the change is
`clip(0.004 * (measured_force - requested_force), -0.005, 0.005)` instead. These
are changes to the gripper position command, not object forces. This is a simple
feedback rule, not force-closure optimization, and its success still depends on
all contact and slip gates.

### Optional controlled placement

`controlled_place` defaults to false and requires an object-relative grasp seed
with a demonstrated release. Instead of opening at the original release time,
it retains closure, moves above the final source object pose, lowers to that
pose, holds for 0.5 s, and then opens. The default `place_clearance` is 0.04 m;
the approach also respects a higher original release height. The added schedule
reaches the overhead waypoint at +1 s, the final placement at +2.5 s and opens at
+3 s relative to the original release waypoint. Subsequent velocity dilation
can lengthen these intervals, including the nominal 0.5 s hold.

This is a derived robot placement policy. It does not reproduce a source release
that relies on a ballistic drop. Only desired hand/object references and robot
commands change; neither the live object state nor source data is overwritten.
The final source pose is a placement reference, while the independent task
contract still determines success. Original source-release failures remain
saved as separate attempts, including the complete-interval frozen matrix.

### Optional measured-attachment calibration

`calibrate_attachment` also defaults to false. Once the object has risen more
than 15 mm and both distal fingers have positive normal-force contact at every
substep of an interval, the controller stores the measured transform
`T_hand_object = inverse(T_world_hand) @ T_world_object`.

While the gripper is commanded closed, the optional correction requests
`T_world_hand = T_world_object_reference @ inverse(T_hand_object)`. It blends from
the original hand target over one simulated second, reruns IK, and rate-limits
the applied robot reference with velocity and acceleration bounds. This corrects
the difference between an intended grasp attachment and the attachment that
actually formed. It does not reset the measured attachment to hide later slip.

This option reads exact simulated hand and object state: it is a **simulation
state oracle**. It is not an implemented perception system or a validated
hardware feedback loop. The free object still moves only through dynamics, and
the actuator replay must reproduce its trajectory without running calibration.

## Acceptance requires every gate

The task predicate is derived from the validated source scene and saved as
`plan.task_contract`. Source task success is adapted explicitly where the source
robot or final contact state differs:

| Task | Required physical outcome |
|---|---|
| Lift | Object center exceeds source tabletop + 0.04 m while held. |
| PickPlaceCan | Released Can center is strictly inside its positive-x, positive-y bin compartment. For the inspected default source, bounds are (0.1, 0.28, 0.8) to (0.295, 0.525, 0.9) m. Other compartments do not count. |
| NutAssemblySquare | Released nut is within 0.03 m of peg1 along each horizontal axis, above the source tabletop and below tabletop + 0.05 m. The inspected tabletop is 0.82 m. The lower bound rejects a nut lying on the floor beneath the peg. |
| Stack_D0 | Released cubeA has actual contact with cubeB, exceeds source tabletop + 0.04 m, and its center is more than 0.025 m above cubeB's center. |
| Threading_D0 | Needle collision-geometry center is within the source ring radius of the mean ring center while held. The inspected 20-box ring yields 0.012 m. This reproduces the publisher's geometric predicate rather than proving arbitrary insertion topology. |
| PickCube-v1 | Cube is within 0.025 m of the recorded goal and all Reachy arm joints are at most 0.2 rad/s; additional base stationarity rules apply. Release is not required. The recorded goal is distinct from the final demonstration pose. |

The source predicate references are saved with the task contract. Generic
physical acceptance additionally requires:

| Gate | Requirement |
|---|---|
| Final stability | Task satisfied continuously for at least the final 1 s; object speed below 0.02 m/s and angular speed below 0.2 rad/s. Held tasks additionally require bilateral contact at every physics substep of each final interval. |
| Established lifted grasp | A measured bilateral grasp anchor after more than 15 mm lift. |
| Contact during carry | At least 95% bilateral-contact occupancy after the anchor while closure is commanded. Occupancy counts all five substeps rather than treating any brief touch as a full interval. |
| Relative grasp slip | Maximum hand-relative object drift at most 3 mm and 3 degrees from the fixed measured anchor while closure is commanded. |
| Hand/object penetration | At most 1 mm throughout the rollout. |
| Other penetration | At most 2 mm for robot/environment, active-object/environment and robot self-contact. |
| Object/body contact | No active-object contact with robot parts outside the hands. |
| Actual speed | Arm joints at most approximately 1 rad/s; each base velocity component at most approximately 0.61 m/s and yaw at most 114 degrees/s. These are measured simulation velocities. |
| Joint and self clearance | At least 0.025 rad arm-joint-limit margin at every physics substep, with an additional robot geometry check; at least 0.009 m self-clearance in the separate robot geometry model. |
| Completion | The full planned rollout must finish without an exception. |

Penetration, bilateral contact, arm-joint-limit margins and actual velocity are
checked at physics substeps; object slip, task predicates and kinematic
self-clearance are evaluated at controller interval boundaries. Numerical
penetration tolerances are explicit acceptance thresholds,
not a claim that rigid contact is perfectly nonpenetrating. Severe robot
collisions terminate an attempt early and remain failures.

## Bounded search and replay evidence

The coordinate search starts from an explicit `Candidate`, evaluates it in
MuJoCo, and then changes one parameter of the best candidate at a time. The
proposal list contains grasp tilt, height, depth, requested contact force, yaw,
scene lateral placement and scene heading. Other parameters, including timing,
base bounds/use, local grasp offsets, force feedback, calibration, controlled
placement and source start, are chosen in the seed/configuration.
A force proposal changes behavior only when the force-feedback option is active.
The default proposal budget is 32, covering the heading proposals if the search
has not already stopped at a complete pass. Smaller budgets can stop before every
parameter is visited; repeated identical candidate configurations are skipped.

The ranking penalizes task failure, missing bilateral contact, penetration,
slip and failed gates. Ranking cannot override a failed acceptance rule. Search
stops at the first complete physical pass or returns its best failed attempt.
Each unique attempted candidate has its own directory; `search.json` identifies
the selection and failures. Trials used to select a candidate are development
data. A generalization claim requires separately identified, unmodified source
episodes evaluated with frozen candidate settings. Retries, crops and overlapping
release versions must not become additional independent demonstrations.

Each completed attempt records the scene XML and hash, hashes of external scene
assets, source HDF5 hash, loaded implementation snapshots, simulator versions,
controller identity, scene assumptions, candidate, task contract and gates.
`replay.h5` contains the initial state, interval-end states, object trajectories,
actuator controls, references and diagnostic metrics. The object-reference
residual measures the actual freely simulated object against the constructed
object target: normally the retimed source reference, or the modified placement
reference when `controlled_place` is enabled. It is diagnostic and currently has
no acceptance threshold. It must not be labeled an unmodified-source trajectory
error in the latter case. Physical task success does not establish exact
reproduction of the demonstration.

`target/right_hand_pose_world` is the planned hand seed, including any controlled
placement. `target/applied_right_hand_pose_world` additionally records the requested
TCP target after optional attachment calibration; it is a request before IK and
joint-reference rate limiting, not a measured hand pose. The applied joint
reference and final actuator control remain the execution evidence.
References apply during the recorded interval; measured states are its endpoint,
so plots must preserve that timing convention.

The independent replay verifier checks scene and asset hashes, complete recording
shapes and timeline, finite state/control data, free object roots, and the absence
of object assistance. It rejects object or descendant actuators, gravity
compensation, mocap, tendons and constraints; only whitelisted robot-joint
actuators are allowed. It resets once and applies the recorded robot actuator
controls without IK, force feedback, calibration or object state updates. Saved
joint and object trajectories must agree with replay below 1e-7 numerical error.
Replay equivalence is separate from task success; both must be reported.

## Current boundaries and next work

Multi-object scenes preserve the additional objects and their physical contacts.
The implemented Stack and Threading adapters nevertheless use one active right
gripper. They are not coordinated bimanual transport or general sequential
multi-object task planners. Scene compilation for ToolHang or TwoArmTransport
does not establish a complete dynamics adapter. A dataset with human/object
poses but no verified physical scene also cannot inherit these claims.

The mobile base is an ideal planar x/y/yaw mechanism with direct position
actuators. Wheel traction, nonholonomic motion, localization, obstacle-map
uncertainty and physical navigation have not been validated. Whole-scene heading
search and x/y bounds can improve reachability, but there is no general
table-footprint navigation planner. Contact failures can therefore reflect base
reachability even when a gripper attachment is stable.

Other limitations include hand-tuned material/controller assumptions, a narrow
grasp family, a single grasp/release phase, no learned recovery, no independent
target-object orientation-error gate, and exact-state dependence of optional
calibration. Source scene fidelity, source task support, a successful dynamic
attempt and hardware transfer are separate claims.

The [primary-source methods survey](datasets/dynamics-aware-retargeting-methods.md)
and [evidence ledger](datasets/mobile-methods-evidence.json) describe the relevant
MimicGen/DexMimicGen, mobile manipulation, contact optimization and MuJoCo MPC
work. Short-horizon actuator optimization with predictive sampling or CEM,
explicit base-footprint constraints, contact-phase segmentation and perturbation
tests are future engineering directions. None is an implemented MPC result here.
