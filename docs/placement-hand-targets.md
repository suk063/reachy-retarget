# Compiled hand geometry before placement IK

`placement_hand_targets.propose(prepared, robot, candidates, output)` screens
existing orientation proposals at their original placement XY, the logical task
cell center, and a declared bounded 5-by-5 interior XY grid. It introduces no new
orientation. Proposals rank by the magnitude of the declared rotation, then XY
displacement. The default interior margin is 40 mm and is a proposal parameter,
not a relaxed collision threshold.

The checker uses all compiled right-hand collision geoms and their actual mesh,
geom and finger transforms. It queries every physical hand/fixture pair, including
walls opposite the nearest wall. The manipulated object is excluded from these
fixture queries. Its intended grasp still requires the separate contact checks.
The source fixture reset is an explicit planning assumption.

Every query owns a separate MuJoCo `MjData`. Robot FK establishes the geometry at
each calibrated aperture; only derived hand geom transforms are then expressed
at the hypothetical TCP pose. It never calls `mj_step`, writes an object qpos,
changes a model field, or accesses physical rollout data. Tests compare these
distances with actual MuJoCo FK using a rotated mesh, an articulated finger and
a TCP offset. The object state and clock are checked for accidental changes.

A proposal passes only a **sampled rigid-hand screen**. It is not an admitted
robot path. Full mobile IK, all interpolated aperture and fixture checks, source
coverage, reference speed checks, actual physical gates and independent actuator
replay remain required. The report records this distinction explicitly.

The immutable output contains original source arrays, original candidates,
module bytes and hashes, a compiled collision-geometry signature, every rejected
candidate, and selected candidate IDs. The screening waypoints are tested for
exact equality with `supported_placement`'s Cartesian proposal.

In the a60 development case, three nearest-wall-facing targets put the final
palm 6.76–11.96 mm into the opposite bin wall. The logical cell center made that
collision worse. A 29.82 mm XY correction within the same verified task region
produced a sampled hand-clear target. This explains the original IK rejection;
it does not establish full robot or physical success.

## Interpolation repair remains a separate gate

`supported_placement.prepare(..., adaptive_midpoints=3)` optionally repairs
invalid interpolation chords after ordinary mobile IK has admitted every coarse
knot. `placement_midpoints` tests the chord's Cartesian midpoint and inserts a
new constrained IK solution only if the unchanged 2 mm / 0.02 rad TCP limits or
3 mm fixture clearance fail. Every original knot and both endpoints stay exact.
At most three rounds run; all failed midpoint states and original/refined knot
arrays are saved. Complete final control-row checks still decide admission.

The a60 first complete placement solved all 9,099 rows with zero fixture
penetration, but its separate positive-clearance gate found a 2.7436 mm gap
between the forearm and the opposite bin wall. That is an interpolation failure,
not permission to lower the 3 mm planning threshold. The adaptive repair uses
the same Cartesian target and scene to address that specific failure.

An independent `minimum_motion=True` optimizer option requires hard pose
constraints and normalizes motion by declared base/arm speed scales. It is
disabled by default. Its first a60 counterfactual selected a different redundant
branch and failed a lowering knot; it is retained as a failure, not used as the
default repair.

Local midpoint repair can instead use `midpoint_minimum_motion=True` with
`interval_knot_timing=True`. It starts each new solve from the interpolation
midpoint and assigns time to each unchanged geometric segment independently.
This avoids stretching every segment because one short interval needs more time.
The original coarse knots and complete source clock correspondence remain saved.

Explicit `midpoint_position_reserve_m` and `midpoint_fixture_reserve_m` are
optional conservative planning margins, each bounded to 0–1 mm and defaulting to
zero. They affect only newly solved midpoint knots. For example, a 100 µm
position reserve solves those knots inside 1.9 mm, while a 100 µm fixture reserve
solves at least 3.1 mm from fixtures. Final complete admission still uses the
unchanged 2 mm TCP and 3 mm fixture criteria. The fixture reserve uses an isolated
query with a wider distance cap, so a pair just outside the ordinary query cap
can enter the optimizer's active set. Per-call records bind both internal limits;
subsequent ordinary solves retain their original limits.

For a60, local midpoint minimum-motion and interval timing reduced the inserted
placement from 83.43 to 25.23 seconds and passed complete geometry. Its first
original-model FF1/FF0 physical pair passed 15 of 16 gates, failing only measured
base speed in the original source prefix/suffix. A fresh source-path planar norm
cap of 0.45 m/s then exposed a 55.72 µm inserted interpolation overshoot. Adding
the explicit 100 µm midpoint position reserve admitted all 3,323 control rows
and independent 3 mm fixture checks in 37.17 seconds of computation. This is
still geometric evidence; a new complete physical rollout and actuator replay
are required before calling the corrected trajectory physically successful.

`lower_acceleration_limits` optionally supplies 17 positive acceleration limits
in reference order: base X/Y (m/s²), base yaw (rad/s²), then both arms (rad/s²).
The default `None` retains the old timing exactly. This option changes only the
lowering phase and its reversed retreat. Each existing geometric interval uses
quintic progress with zero velocity and acceleration at the interval endpoints.
Analytic derivative bounds (15/8 for velocity, 10/√3 for acceleration) determine
the minimum duration. All joint and Cartesian knots remain unchanged, and the
original source and traverse timing remain exact. An explicit accelerated retreat
is rejected when this option is active, to avoid contradicting the declared caps.

These are bounds on the planned nominal reference, not measured robot
acceleration. The measured support guard can halt descent early. Complete
geometric admission, original physical gates and actuator replay still apply.
For the 6a80 diagnostic, the original lower path has only 15.1 mrad maximum
adjacent joint changes but a 39.2 rad/s² velocity corner. Caps of 1 m/s² base XY,
2 rad/s² yaw and 3 rad/s² arms predict 7.45 seconds instead of 3.36 seconds for
that same lower path, while leaving the rest of the source trajectory alone.
