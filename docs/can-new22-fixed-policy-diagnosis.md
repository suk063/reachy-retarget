# Fixed Can policy transfer: 22 independent episodes

The complete `additional-can-policy-v1` manifest contains 22 new independent
source episodes and four closure candidates per episode. All 88 attempts were
rejected before physical execution: 17 sources at robot initialization, five
at supported placement. This is **0/22 successful sources and 0/88 physical
rollouts**, not a measured 88-rollout grasp failure rate. All attempts completed
their relevant IK and static checks; no infrastructure failures were reported.

Every rejection is a robot/environment collision. All 22 checked paths pass
IK error, joint margin, robot self-collision and self-clearance gates. Changing
gripper force or closure cannot resolve this primary geometry bottleneck.

| Failure stage | Sources | Main collision | Maximum depth |
|---|---:|---|---:|
| Original full-path base/arm admission | 14 | Forearm / bin wall | 2.69–53.45 mm |
| Original full-path base/arm admission | 2 | Palm / bin1 wall | 42.50–46.86 mm |
| Original full-path base/arm admission | 1 | Right torso bar / bin1 | 30.44 mm |
| Inserted placement admission | 5 | Palm or forearm / bin2 wall | 3.24–25.16 mm |

Six sources have zero environment penetration throughout the original open and
closed portions; their forearm collision occurs only after original release:
`fe117e32c6f13e7ae68c4be4`, `c841bebf56398150e75ab844`,
`379ba29238cc2486ec03cebc`, `8ee64ec58eb602a29bbc0827`,
`792e9d23dcbe6e986bc8ac50`, and `2447e639d8694ab40028362b`.
Their maximum penetration ranges from 2.69 to 27.50 mm. The current pipeline
preserves the original post-release arm suffix, so that suffix can prevent
admission even when grasp and carry are geometrically clear.

All five placement failures occur at deepest lowering and the repeated hold,
opening and retreat endpoints. Their traverse and return phases are clear.
Two have palm collision as the largest contact; other cases can include both
palm and forearm contacts. A base or elbow branch change cannot generally
remove a palm contact at a fixed TCP pose, because the palm is rigidly attached
to that target. These require a different robot grasp or placement posture,
subject to the same source geometry and physical gates.

A general next intervention is a predeclared per-episode geometry search over
robot base offset, heading and arm seed, ranked solely by geometric admission
and robot correction magnitude. Placement posture and a safe post-release
withdrawal need separate geometry checks. The original fixed-policy outcome
must remain 0/22; later optimized attempts are a separate algorithm, with the
same 22-source denominator and all rejected candidates retained. No physical
outcomes should be used to hand-select episode-specific geometry.

The [evidence JSON](can-new22-fixed-policy-diagnosis.json) records source IDs,
artifact paths, phase maxima and contacts recomputed at each maximum. Object
geometry, mass, friction, source arrays and validation thresholds are unchanged.
Static contact inspection initializes only robot joints, preserving the source
object reset. Placement peak contacts were re-inspected at full opening to
identify rigid palm/forearm links; the original admission artifact retains its
five-aperture envelope used for the actual decision.

## Predeclared per-episode geometry search

`feasible_geometry_search.run(root, job)` now implements the next algorithm.
The 66-job manifest is `configs/additional-can-geometry-search-v1.json`: three
disjoint chunks for each of the same 22 sources. Each chunk tries 18 of 54
base/seed candidates. Offsets in the frozen world frame are X={0, 0.1, 0.2} m,
Y={-0.15, 0, 0.15} m and heading={-15, 0, 15} degrees, with either the recorded
initial right arm or the fixed earlier seed9. Every candidate retains the
original full base trajectory plus that declared offset.

Sparse screening includes uniform samples, closure/release boundaries and
hand/base position extrema. It is approximate: a sparse IK branch may reject a
candidate that a different solver path could realize. Screen failure therefore
means the bounded search rejected that candidate, not that the demonstration
is physically impossible. A surviving candidate must pass full original-source
admission, the same vertical approach and the same local speed retiming.

Placement candidates combine five local target-quadrant XY points (center and
the four +/-25 mm diagonals) with world pitch={30,40,50} and yaw={-20,0,20}
degrees. All points must lie strictly inside the original verified target
region. The existing 150 mm temporary base shift, partial opening, 2 mm descent
allowance and support controller remain fixed. Nine sampled poses screen each
placement candidate before its complete path and aperture-envelope checks.

Selection uses a declared lexicographic rank: base translation/heading
correction with a 0.3 m yaw radius, seed order, stable base ID, placement
correction, and stable placement ID. It never reads physical results. An error
in an earlier candidate prevents a claim of minimum-rank selection; the chunk
is `search_incomplete`, with successful later artifacts still retained.

The selected record contains both an admitted original-clock base plan and a
complete contact002 plan. Four later closure trials should use the base plan,
then reapply the declared approach, retiming and selected placement so each
closure receives its own complete admission. The complete plan already has the
inserted phase and must not receive a second insertion. Original source hand
and object references remain in `source-reference.npz`; no physics is executed
by this search. Its source SHA256 is
`61a527a29bb1ae94e17eb219fb5bf81bf3da3ce72f439fb8b28d73a757e5317f`.
The initial publication is runtime
`24ba99342c89362cda5d5f29a2108f09389be16d0625bb3e5f8346c15ad0ddf5`.
