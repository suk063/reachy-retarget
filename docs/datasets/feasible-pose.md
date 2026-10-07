# Geometric tool mapping and bounded mobile placement

`reachy_retarget.feasible_pose.retarget` produces derived Reachy joint/base
targets from observed native TCP, base, clock and task-object poses. It preserves
the original contact-reference positions and timestamps. It may adjust only the
robot's planar base placement/path and arm configuration. It does not establish
collision clearance, actuator tracking, physical contact or task success.

The previous diagnostic solver attached each source gripper rotation to Reachy's
home hand rotation. That constant heuristic can reverse the physical approach
axis and impose unnecessarily difficult orientations. The new mapping uses the
source gripper model and the Reachy CAD axes instead:

| Native reference | Approach axis | Unsigned jaw separation | Constant source-to-Reachy TCP rotation |
| --- | --- | --- | --- |
| BiGym Robotiq pinch site | +Z | Y | `diag(1,-1,-1)` |
| RoboCasa Panda grip site | +Z | X | `[[0,-1,0],[-1,0,0],[0,0,-1]]` |
| Reachy arm_tip | -Z | Y | Identity in its own frame |

The source revision and initial measured finger-body separation are checked at
runtime. BiGym evidence is the pinned
[Robotiq XML](https://github.com/NeuracoreAI/bigym/blob/52070fa73e8dd88a3d76eed4be575a6d01497931/bigym/envs/xmls/robotiq_2f85/2f85.xml).
RoboCasa uses its saved source model XML (SHA-256
`2a9652be9d833bf5bedba20195b73bee70277c8095e5e186ffa6e7ea9620439d`):
the grip site is on eef +Z and each finger's Y slide is rotated 90 degrees.
The jaw-separation sign is an explicit constant 180-degree roll hypothesis;
changing it does not permit arbitrary time-varying hand rotations.

The Reachy URDF SHA-256 is
`63a1ecab3312a72c35f19a4bb1142005d74e4d7101da87822d02fea96696d448`.
Its arm_tip fixed transform from the palm has translation `[0,0,0.10]` and
rotation `Rx(pi)`. Both hands' actual compiled collision meshes are inspected.
At the declared nominal gripper angle 0.63 rad, opposing distal face centroids
have midpoint `[-0.010196, 0, -0.046570151786]` m in TCP coordinates and
25.8077 mm separation. For target rotation `R` and source pinch position `p`,
the derived Reachy TCP position is `p - R @ midpoint`. Consequently the mapped
CAD pad midpoint coincides with the unchanged source reference point.

This is geometric calibration of model frames, not a measurement of deformable
rubber contact or proof that the source pinch site touched the object. Sampling
closed (-0.06 rad), nominal (0.63 rad), and open (2 rad) configurations found
6.573 mm maximum midpoint variation relative to nominal. That sampled variation
is recorded; it is not a bound over every possible aperture. Mesh checksums,
face centers, face areas, axis evidence and exact transforms are saved with
every attempt.

An explicitly evidenced `pad_calibration_gap_m` can replace the nominal aperture
for individual hands. The gap is measured from unchanged source object geometry;
the CAD inversion selects the monotone opening branch above minimum separation,
because the commanded negative closure can geometrically cross the face centers.
`pad_calibration_gripper_rad` is an alternative explicit angle input; a hand
cannot receive both definitions. Overrides require `pad_calibration_evidence`.
No gripper command changes: the selected aperture only calibrates the constant
TCP-to-pad attachment. The first BiGym plate follow-up uses a 7.90126 mm observed
width, giving 0.222199 rad and a 3.31087 mm TCP calibration correction; see
[the physical experiments](bigym-physics.md). This remains a geometric proxy
that needs complete contact validation.

The solver uses bounded SciPy trust-region least squares, warm starts, and a
tiny continuity penalty. Deterministic bounded alternative arm seeds are tried
only after a frame exceeds 20 mm translation or 0.15 rad rotation error. Default
robot corrections are ±0.5 m per horizontal axis and ±1.2 rad yaw relative to
the source base. After the initial derived placement, each horizontal axis is
limited to 0.6 m/s and yaw to 1.2 rad/s using the original irregular clock. These
are axis limits; diagonal planar speed can exceed 0.6 m/s. Empty intersections
between this speed limit and the source-relative placement window are retained
as explicit failures. Arm velocity is reported but not constrained. Every
documented hand remains constrained; the solver does not discard the other arm
to obtain a passing score. Inactive home-pose placeholders are declared.

Before the full-trajectory run, four previously failing BiGym frames (0, 272,
1063, 2252) were independently solved with this geometric mapping and bounded
base placement. Maximum positional errors were respectively 0.000037, 0.374,
0.059 and 0.000003 mm; maximum orientation error was 0.00137 rad. Required base
corrections stayed within 0.214 m per axis and 0.171 rad yaw. These independent
frame diagnostics do not establish a continuous feasible trajectory. Comparable
RoboCasa checks retained substantial errors, including roughly 97 mm at the
first frame; high, orientation-constrained source references remain difficult.
No source object or height was changed to remove those failures.

The callable accepts the same named input channels as `pose_retarget.retarget`,
plus optional constant `jaw_sign` and bounded solver settings. Its raw IK,
per-start configurations, calibration and common numeric HDF5 are immutable
attempt artifacts. Original task-object arrays are retained under `source/*`;
`reference/*` uses one shared planar world gauge without scale or relative
trajectory editing. New base/joint/gripper values are under `derived/*`, never
measured observations or falsely inferred applied commands. Source replay
failure remains in metadata independently of Reachy IK outcomes.

An optional `grasp_relocation` describes an explicit new planar-rim grasp.
`gravity_aligned_rim` uses the selected observed object's COM and verified plane
normal, and projects gravity into that plane at a declared source anchor. One
constant object-local rotation moves the original grasp radius above COM and
rotates the selected hand's full approach, carry and retreat around the unchanged
object. It changes both hand position and orientation. Other hands, original
source hand arrays, source clock and object poses remain untouched. The original
world-gauged hand reference remains `reference/source_contact_pose`; the new
contact reference is `derived/relocated_contact_pose`.

For development BiGym c795, original source geometry gives a 0.4 kg plate and
a side-rim grasp with 0.2704 Nm gravity moment about the closing axis. The first
candidate moves that grasp by 102.871 mm, rotating 88.13 degrees around its
geometry-derived plane normal. The model's predicted gravity moment becomes
0.001089 Nm. The new actual collider intersection is 7.98773 mm wide and is used
for pad calibration. This is a new grasp strategy, not minor calibration or
proof of physical success. Rotating the entire source-relative approach/retreat
changes some distant hand references by up to 433.5 mm; full-trajectory IK,
environment admission and all physical gates remain mandatory.

Future full solves first retain all-frame optimistic tip/wrist reach bounds in
`kinematic-preflight.json` and the unchanged references in
`reference-preflight.npz`. A proven necessary-bound failure returns
`kinematic_preflight_rejected`; a passing bound does not establish IK or scene
admission. During solving, `solver-trials.jsonl` preserves each bounded start,
`solver-progress.json` updates approximately every five seconds between starts,
and immutable `checkpoints/` files retain completed prefixes every 100 frames
or 15 seconds. These partial files never claim a complete or physical pass.

The optional `grasp_relocation.free_phase_blend` declares the original clock and
verified binary gripper-command channel, plus source pinch-to-COM clearance
distances. A quintic source-time weight preserves the original distant open
approach, reaches the complete rim rotation before closure, and remains exactly
one for the whole contiguous closed interval. It returns to the source reference
only after an observed open retreat crosses both declared distances and remains
clear. This distance rule is not a mesh collision admission. Object-local SE(3)
rotation around COM is scaled geometrically; no Euler-angle subtraction or
object edit is used. Weights and both original/derived contacts are retained.

The frozen c795 blend proposal uses 0.20 m for open-phase clearance and 0.30 m
for complete return. The observed approach crosses the 0.20 m threshold before
closure; the hand never withdraws beyond it after release. The candidate thus
retains the full relocated contact strategy through the available tail instead
of inventing a source retreat. The complete constant-rotation candidate remains
an independent retained attempt. Both require full IK, positive native-scene
fixture clearance and unchanged physical gates before any success claim.

The first full constant-rotation c795 attempt completed with zero valid frames
out of 2,256 (maximum 186.7 mm position and 0.516 rad orientation error), preserving
all 18,048 bounded starts. A separate native MuJoCo 3.1.5 reconstruction of all
2,256 retained integration states found left-arm/plate contacts in 1,407 frames
and no right-arm contacts with any task object; the right gripper never closed.
This contact-role audit does not infer forces or prove a Reachy parked-arm path.
Independent right-parking checks still failed several upper-rim poses.

The explicit `orientation_policy="preserve_source"` instead relocates only the
pinch position and preserves the original source hand orientation. It requires
separate `orientation_evidence`; the default remains `rotate_with_position`.
For the c795 anchor, the actual retained closing axis is 0.277 rad from the plate
normal, so its newly measured convex-collider intersection is 8.89560 mm, not the
7.98773 mm used by the rotated-hand candidate. Its modeled closing-axis gravity
moment is 0.01786 Nm. Four isolated previously failing frames then solved with
both hands constrained, sub-micrometer position residuals and 97–132 mm measured
robot self-clearance. These are independent-frame diagnostics, not a continuous
or physically validated trajectory. Full fixture clearance remains required.

The first complete side-approach run admitted 2,231 of 2,256 frames. The remaining
25 frames (429–453, all in approach) exceeded the unchanged 20 mm position limit;
maximum error was 22.419 mm. The base reached its planning speed limits on the
original source clock. Repeated failed-frame restarts also produced large joint
branch changes, so this retained trajectory was not sent to physics. A separate
516-frame prefix using explicitly declared planning bounds of 1.2 m/s per base
axis and 2.4 rad/s yaw passed all 516 frames without restarts, at 3.586 mm maximum
position error and 0.054 rad maximum adjacent joint change. This prefix is not a
full-trajectory result. Any subsequent physical execution still requires local
retiming to the existing 0.6 m/s base-axis, 1.2 rad/s base-yaw and 0.8 rad/s arm
command limits, plus the independent measured-state gates. Planning rates on an
unchanged source index clock are not measured physical execution rates.
