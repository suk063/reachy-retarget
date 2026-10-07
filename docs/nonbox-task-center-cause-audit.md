# Nonbox task-center failure audit

The 216 task-center candidates represent three original demonstrations: two
SquareNut sequences and one Threading sequence. Threading has two candidate
grasp axes. Each source/axis has 27 stationary bases and two arm seeds; these
variants are not additional demonstrations. All 216 completed their geometric
checks and were rejected before physics. Every candidate has an IK failure
during closed manipulation, not merely in the free initial pose.

The numeric source targets, complete clocks, joint solutions, phase labels,
collision reports and input hashes were audited from the retained immutable
archives. [The complete audit](nonbox-task-center-216-audit-v1.json) retains
every candidate rather than selecting only its best frame.

| Source/axis | Best stationary-bank maximum position error | Failed closed frames | Principal fixture overlap |
| --- | ---: | ---: | ---: |
| SquareNut 2aef / 0 | 2.581 mm | 6 / 320 | torso bar/table, 139.933 mm |
| SquareNut 31bb / 0 | 89.197 mm | 172 / 344 | torso bar/table, 101.355 mm |
| Threading / 0 | 212.175 mm | 995 / 995 | torso bar/table, 122.003 mm |
| Threading / 2 | 82.657 mm | 869 / 995 | torso bar/table, 48.897 mm |

The best environment-safe stationary candidates still have maximum TCP errors
of 196.583, 199.556 and 217.526 mm for the first three cases. None of the 54
Threading axis-2 candidates passes the environment check. Repeating selected
failed poses with 1,000 IK iterations does not materially improve the fixed-base
results. Shoulder and wrist joints reach their original limits. The two seed
variants frequently converge to the same constrained solution.

## Mobile pose tests distinguish reach from fixture interference

The source table occupies world X=[0.2, 1.0] m and Y=[-0.5, 0.3] m. Unconstrained
mobile IK restores the sampled TCP targets but generally moves the torso into
this table. For example, the environment-safe SquareNut candidates require
0.432 and 0.534 m base changes at their worst sampled poses and then overlap
the table by 34.698 and 54.144 mm. These observations are
[pose diagnostics](nonbox-mobile-pose-probe-v1.json), not path admissions.

A second diagnostic starts at the four geometric table sides and constrains
all six rigid torso bars to remain at least 3 mm from the unchanged table.
It preserves each exact TCP target and the original joint bounds. It changes
only robot pose during static inspection and never steps physics or writes an
object state. The [pose results](nonbox-table-constrained-pose-probe-v1.json)
show two different causes:

* SquareNut 2aef, closed frame 305 at 3.05 s: distinct torso-safe robot poses
  reach the TCP to below 0.2 micrometres, but the palm intersects the table by
  11.403 mm. This interference is implied by the desired hand pose itself.
* SquareNut 31bb, final closed frame 630 at 6.30 s: the exact hand pose
  intersects the square assembly peg by 20.828 mm at the distal finger. The
  later open-return target also hits the peg. Thus a free-return correction
  alone cannot fix this source. The additional
  [closed-frame check](nonbox-table-constrained-closed-pose-probe-v1.json)
  retains this distinction.
* Threading axis 0 has exact, collision-free representative solutions on the
  far X side of the table. Axis 2 has them on its negative-Y side. These poses
  are outside the previous stationary bank and justify a geometry-constrained
  path check. They do not establish whole-path feasibility.

## SquareNut pad-line rotation limit

The existing constant pad-line rotation preserves both calibrated contact-point
trajectories. An isolated-hand geometry screen applies that exact rigid
transform to the original compiled collision geoms, retaining fixed fixtures
and object reset. It checks every closed reference row at five declared
apertures. The transform queries do not alter qpos or moving-object state.
Finite contact area and object-following are not inferred from preserved
calibration points.

The [full-circle screen](nonbox-gripper-roll-fullcircle-v1.json) checks all 73
angles from -180 through +180 degrees at 5-degree spacing. Every angle fails.
The best worst overlap is 12.198 mm at +75 degrees for 2aef, and 16.531 mm at
+165 degrees for 31bb; both remaining obstructions involve the assembly peg.
The original zero-roll paths have 15.489 and 22.000 mm worst overlap.

This is not evidence that every possible SquareNut grasp is impossible. It
rejects this specific axis-0 contact line under a constant rotation. The
conservative bounding radius of every compiled hand geom about that line is
0.188913 m. Between neighboring angle samples, any hand point can move by at
most 8.242 mm. Even that bound cannot remove the sampled worst intersections
between samples. The [radius calculation](nonbox-gripper-roll-radius-bound-v1.json)
records this bound separately. A different contact line or grasped component
face would require fresh calibration, complete geometry admission, and actual
physics; expanding the base bank alone would retain the same obstruction.

## Complete Threading side-base checks

One stationary pose per Threading axis was selected from the exact table-side
pose solutions by minimum XY displacement from the original initial base.
Both checks retain all 1,257 original reference rows and five-point aperture
envelopes. [Both were rejected](nonbox-threading-side-path-v1.json):

* Axis 0 at base [1.2593, 0.1360, -0.6311 rad] has 218 failed closed IK rows,
  all 262 open-prefix rows fail IK, and the hand also hits the table.
* Axis 2 at base [0.2470, -0.5856, 1.0507 rad] has no robot/environment or
  robot-self overlap on the entire computed path and retains its joint/self
  margins. It still fails IK on all 262 open rows and 151 of 995 closed rows.
  The remaining 844 closed rows pass IK. Its complete geometric result is
  therefore rejected; no physical success is claimed.

An anchor-seeded mobile IK diagnostic subsequently inspected all 1,257 axis-2
rows, with the same table-bar constraints and original TCP targets. It remained
rejected: 97 closed and 173 open rows fail one or more checks. The complete
[mobile-path record](nonbox-threading-mobile-path-v1.json) preserves the joint
and collision failures, including a 42.924 mm environment overlap. There is a
target-geometry problem during acquisition as well: the exact axis-2 hand
target puts a proximal finger 35.193 mm into the table at 3.03 s. The axis-0
target has 14.967 mm palm/table overlap at the same source time. Those
[isolated-hand measurements](nonbox-threading-hand-zero-roll-v1.json) prevent
misdiagnosing the problem as base reachability alone.

The horizontal axis-0 contact line does admit a useful minimal rotation. Its
[full-circle hand screen](nonbox-threading-pad-line-roll-v1.json) finds that
+10 degrees is the smallest sampled roll clearing every closed row at all five
apertures: minimum fixture clearance is 3.158 mm. At +15 and +20 degrees the
minimum gaps are 4.377 and 4.812 mm. The two calibrated contact-point paths stay
unchanged to below 2.5e-16 m. The native 40 mm handle remains unchanged. These
are static hand/fixture candidates only; finite contact support, complete arm
and mobile-base admission, acquisition transitions, timing and actuator-driven
physics remain required. The old axis-2 mobile path cannot admit an axis-0
rolled target.

All these diagnostics use the pinned v23 code and original controller/assets.
They write only isolated worker-local scratch and local evidence files, not
the source datasets or original experiment records. Robot motion proposals
require complete new admission before any actuator-driven rollout.
The [artifact index](nonbox-cause-audit-v1-artifacts.json) binds every retained
script, numeric result and source input in the local diagnostic backup.
