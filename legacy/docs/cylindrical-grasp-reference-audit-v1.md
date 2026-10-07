# Original-mesh cylindrical contact proposal

The new opt-in `cylindrical_grasp.apply(prepared, robot, output)` changes only one
constant robot TCP attachment. It preserves every original source row, clock,
object reference, gripper-intent value, and physical scene property. It saves
both target trajectories and the exact compiled object vertices and finite CAD
pad triangles. It invalidates all prior IK, approach, placement, and fixture
admissions. Retimed or inserted-phase inputs are rejected because their saved
anchor index may be stale.

Matched physical comparisons subsequently improved two sources: both tested
closures on 47cc92 and 6e13ca pass every original physical gate and independent
actuator audit with the finite-pad proposal. The corresponding original-pad
plans fail the rotation-drift gate. Passing regression source 792e9d also passes
with both finite-pad closures. fe117e and 8ee64e remain failures in this
comparison. The full 20-attempt denominator, frozen release, archive hashes,
measured states, controls, and audit outcomes are preserved in
[`can-finite-pad-matched-v2-evidence.json`](can-finite-pad-matched-v2-evidence.json).
Static patch qualification remains distinct from physical validation; the
proposal's own report intentionally never marks itself physically validated.

## Geometry and minimum correction

The source Can's original collision mesh has a straight outer wall from
−38.090 to +37.129 mm along its native axis, with approximately 25 mm radius.
The +31.427 mm vertex ring is interior geometry. It does not terminate the outer
wall. Independent inspection confirms that both the original outer triangles
and the compiled convex hull span the longer interval.

The implementation derives the axis from repeated parallel long hull edges,
checks the surrounding side facets, and verifies their common axial interval
and circular boundary. The circle is an eligibility description only. Aperture
calibration uses the exact transverse support width of the unchanged mesh.

The Reachy inward CAD faces are approximately 28 × 32 mm. Forcing the longer
dimension parallel to the Can axis would add an unnecessary 88–90 degree hand
rotation on these sources. The implemented shortest jaw-leveling rotation
retains the source radial azimuth and tangential pad orientation. Both finite
faces must fit axially within the actual wall, with a declared 1 mm end guard.
The exact polygon's opposed support segments must lie inside each finite
projected CAD face, with a nonempty axial contact corridor. These are geometric
checks; projected support does not assert measured contact area or force.

Read-only probes on five immutable baseline plans produced:

| Source prefix | Rotation change | Midpoint before → after | New finite pad axial extent |
| --- | ---: | ---: | ---: |
| fe117e | 2.646° | 12.736 → 12.736 mm | −1.983 to 26.430 mm |
| 47cc92 | 5.648° | 23.277 → 21.788 mm | 7.700 to 36.129 mm |
| 6e13ca | 2.569° | 25.048 → 21.883 mm | 7.699 to 36.129 mm |
| 8ee64e | 0.869° | 16.128 → 16.128 mm | 1.882 to 30.313 mm |
| 792e9d | 3.639° | 13.058 → 13.058 mm | −1.280 to 27.149 mm |

The corrected anchor centroid lines have zero axial split to numerical
precision. Every projected contact-line edge margin exceeds 11 mm. The original
source files and baseline scene assets were checksum-verified before these
probes. Results and bindings are in
[`cylindrical-grasp-reference-audit-v1.json`](cylindrical-grasp-reference-audit-v1.json).
That file records the exact probe implementation hash; the subsequently frozen
module adds stale-clock rejection and explicit anchor-only scope without
changing the geometric calculation.

## The measured 22 mm split is not a 22 mm jaw-center tilt

At the end of source carry, before inserted traversal, 47cc92 has a 21.643 mm
height difference between measured force-weighted contacts, while the actual
CAD pad centroids differ by only 3.155 mm. For 6e13ca the values are 20.833 mm and
1.088 mm. One contact has moved to the +37.129 mm wall/rim transition. The large
split therefore reflects contact redistribution over finite pad faces and the
rim, not the orientation of the pad-centroid closing line alone.

The earlier contact-force replay matched saved qpos and qvel exactly. This
additional comparison reads saved state and recomputes CAD transforms; it does
not run a modified physical model or overwrite a moving object during rollout.
During inserted placement the archived object reference is deliberately paused;
comparisons against that paused reference are not physical object-tracking
measurements. The source-phase comparisons above precede that pause.

## Full source-relative audit

The original relative hand/object references are not perfectly rigid. Relative
TCP translation and rotation changes, measured from the original grasp anchor
over all closed-intent rows, are:

| Source prefix | Translation change | Rotation change | Max original jaw axial split |
| --- | ---: | ---: | ---: |
| fe117e | 16.822 mm | 1.870° | 3.141 mm |
| 47cc92 | 10.692 mm | 1.761° | 5.266 mm |
| 6e13ca | 11.754 mm | 1.774° | 2.334 mm |
| 8ee64e | 15.895 mm | 11.765° | 1.131 mm |
| 792e9d | 2.730 mm | 2.477° | 3.183 mm |

After the constant correction, maximum jaw axial splits over the same interval
are 0.849, 0.367, 0.105, 0.386, and 1.536 mm respectively. The large relative
rotation on 8ee64e mostly changes the other directions; it does not justify a
large time-varying jaw-leveling correction by itself.

For each source, the audit also intersects exact linear bounds on one constant
translation along the Can axis at the anchor. This gives the nearest additional
height change that keeps every finite pad vertex inside the same axial wall
guard over the specified reference interval:

| Source prefix | All closed-intent rows | Anchor through end of closed intent |
| --- | ---: | ---: |
| fe117e | −6.684 mm | 0 mm |
| 47cc92 | −11.429 mm | −0.126 mm |
| 6e13ca | −9.276 mm | −0.089 mm |
| 8ee64e | −6.971 mm | −6.971 mm |
| 792e9d | 0 mm | 0 mm |

Closed intent includes acquisition before the source grasp anchor. In fe117e,
all 13 axial violations occur before that anchor; treating them as an already
held-object constraint would unnecessarily lower the grasp. Most of the larger
all-intent correction for 47cc92 and 6e13ca has the same origin. Conditioning on
the source anchor is still a reference assumption, not a measured acquisition
event.

The strongest source-path height followup is therefore 8ee64e: 113 of its 140
post-anchor closed rows exceed the guard, and the maximum pad height is
42.992 mm. A further constant 6.971 mm downward robot attachment would satisfy
the axial interval on those references. Radial finite-patch support, full IK,
fixture clearance, supported placement, and physical validation remain required
before such a candidate can be accepted. The default anchor-only proposal is
unchanged.

An opt-in `height_policy="post_anchor_closed_corridor"` now computes this
intersection for the contiguous closed-intent block containing the source
anchor, starting at that anchor and ending at its first release. Each frame's
height response uses the exact object/hand rotations; the displacement is
constant in TCP coordinates, not a global vertical offset. It saves every
checked frame index, response, original/corrected axial bound, and per-frame
translation interval. The final constant attachment must also pass the original
finite-pad anchor checks. Default corrected targets were verified bit-identical
on all five actual baseline plans. Opt-in geometry results are preserved in
[`cylindrical-grasp-closed-corridor-v1.json`](cylindrical-grasp-closed-corridor-v1.json).
These corridor proposals do not yet establish physical success.

## Verification

The focused suite passed 31 tests covering cylindrical wall recovery under
rotation and translation, real MuJoCo compiled mesh/geom/rigid-child frame
transforms, unchanged object state, preserved native azimuth and relative hand
motion, finite face support and rejection cases, source-array preservation,
stale-anchor rejection, invalidation of earlier admission, nonunity axial
translation response, zero-response/incompatible interval rejection, and
termination at the first release. Independent
review reconstructed a separately rotated and translated compiled mesh to
within 5.4 nanometres of its analytic vertices, with unchanged qpos.
