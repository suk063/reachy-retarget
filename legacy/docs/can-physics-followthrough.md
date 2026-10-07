# Can physical follow-through

Both development Can sources now pass the unchanged physical model through the
same approach, posture-admission and support-gated placement pipeline. A fixed
bank of calibrated closure offsets supplies contact candidates: `contact_010`
and `contact_020` pass `1d0214a54bfcfdfd95faa196`, while `contact_005` passes
`6ad5cc23ec6d1c207da3e4b2`. Both use the same 40-degree placement policy with
2 mm maximum extra descent. Every original physical gate and the independent
actuator replay pass. These are two original development sources, not one
demonstration per experiment. A single numerical closure target has not been
shown to transfer across both sources.

No object geometry, mass, friction, simulator settings or validation threshold
changed for these primary attempts. Earlier `contact_002` passes `6ad5c...`
with 0.774 mm / 2.017 degree grasp drift, 0.487 mm object/environment penetration,
0.194 mm hand/object penetration, and 5.06 seconds of final task stability.
`1d0214...` needed a stronger closure plus the small support-height correction;
its successful contact010/020 rotation errors are 2.235/1.518 degrees. See the
[initial physical evidence](feasible-placement-v1-evidence.json) and
[closure/support follow-up](feasible-contact-followup-v1-evidence.json).

## Earlier approach and drop diagnosis

The training episode `6ad5cc23ec6d1c207da3e4b2`, under the `force_2n` controller,
passes every physical gate except `object_environment_penetration_2mm`.
The object lifts 130.3 mm and the acquired grasp drifts 0.638 mm and about 1.41
degrees. Independent actuator-only replay reproduces all recorded qpos and
qvel samples exactly. This remains a **physical failure**.

The attempt is stored at:

`/mnt/reachy-retarget/workspaces/feasible-geometry-v1-6ad5cc23ec6d1c207da3e4b2-can-force_2n/runs/dynamics/feasible-geometry-v1-6ad5cc23ec6d1c207da3e4b2-can-force_2n/6ad5cc23ec6d1c207da3e4b2`

| Phase | Maximum Can/floor penetration | Time | Contact |
|---|---:|---:|---|
| Open-hand approach | 3.217 mm | 2.680 s | Can / bin1 floor |
| Closed grasp and carry | 1.032 mm | 3.072 s | Can / bin1 floor |
| Release | 4.066 mm | 6.220 s | Can / bin2 floor |

Closure is requested at 2.85 s and release at 6.01 s. The approach violation
occurs from 1.966 to 2.722 s, before closure. Fixing the final drop alone cannot
make this trajectory pass.

## Open-hand approach

The saved robot references themselves sweep an open finger through the original
stationary Can. Static checks use the unchanged source reset pose, never a
repositioned object:

| Training episode | First planned overlap | Maximum planned overlap |
|---|---:|---:|
| `1d0214a54bfcfdfd95faa196` | 2.16 s | 6.263 mm |
| `6ad5cc23ec6d1c207da3e4b2` | 1.64 s | 9.920 mm |

Both original closure poses are collision-free with the gripper open. In the
physical executions, the early finger contact pushes or rolls the Can before
the commanded grasp. For example, episode `1d0214...` has no contact on one
distal finger at all under `contact005_velocity`; the other finger reaches
48.5 N and the hand misses its closure target by 52.6 mm.

A candidate vertical approach from 120 mm above each original closure pose to
that same pose was checked at 61 samples. Both training episodes have zero
hand/object overlap and maximum right-hand IK errors below 0.47 mm and 0.0049
rad. The base was held at its closure pose for this feasibility probe. It does
not establish feasibility of the complete pregrasp prefix or physical success.

`grasp_approach.prepare` now implements that complete-prefix policy. It retains
the exact initial TCP target, closure target, every post-closure joint target,
and the entire original clock. Backwards IK retains the admitted elbow branch.
Both development episodes pass all-frame IK, robot/environment and self-contact
admission, and have zero pregrasp robot/object overlap. The original moving
object reset remains unchanged. Five focused offline tests cover preservation,
collision rejection and interrupted-solver artifacts.

The subsequent actuator-only `feasible-acquisition-v1` runs using `contact_002`
pass the grasp, carry, task stability, speed and robot collision gates in both
episodes. Grasp drift is 0.621 mm / 2.04 degrees for `1d0214...` and 0.583 mm /
1.03 degrees for `6ad5c...`. Both remain physical failures because the unchanged
object/environment penetration gate fails. These are two development source
episodes, not additional demonstrations. See [batch evidence](feasible-acquisition-v1-evidence.json).

## Release

The near-passing run releases the Can at approximately z=0.980 m. Its final
resting body origin is approximately z=0.845 m on the bin2 floor. The resulting
drop causes the second penetration violation. The original hand reference
remains near z=1.01 m after opening, so delaying opening alone will not lower
the object onto support.

`supported_placement.prepare` inserts a bounded robot placement phase at the
original release after local retiming. The phase translates over the final
recorded object position, lowers the hand, waits for support, opens, lifts and
returns to the original release TCP before the unchanged source suffix. It
preserves every original row and saves a paused source-clock map plus original
arrays. All references receive IK and collision admission before use.

The runtime controller freezes downward motion when upward object support is
measured. Opening requires at least 120 ms of support at or above half the
object weight; otherwise the attempt freezes with its grasp closed and records
a failure. Retreat reverses only the admitted descent prefix that actually ran.
The module writes no simulator state. Both complete development paths now pass geometric admission using the same
explicit settings: a 150 mm temporary base-X offset, a held rotation of world
XYZ `[0, 40, 20]` degrees, and placement XY `[1.0275, -0.3225]` m. This XY is
25 mm in X and -25 mm in Y from the verified target quadrant center. The object
shape, physical properties and target region are unchanged. A 1.4 rad partial
opening releases the object inside the bin; full 2.0 rad opening occurs after
lifting above the walls. The original grasp patch remains unchanged.

The full paths have less than 1 mm / 0.01 rad IK error and zero environment or
self penetration. Their minimum self clearance is 124.6 mm and joint margin is
31 mrad. Simulated support stops at 10%, 50% and 90% of descent keep references
within the inherited 0.8 rad/s full-path speed bound. The inserted phase adds
16.79 and 12.05 seconds respectively in this conservative first policy; the
original recording itself is not slowed further or cropped. Nine placement
unit tests cover source preservation, interrupted IK, support failure,
transient support, partial/full opening, restored arm/base branches and an
intermediate-aperture collision that free endpoint apertures would miss. Each
inserted reference is checked at five aperture values between the closed and
declared open targets. The current source SHA256 is
`9ceacdc4ac3e1f8ddb2d39f60555e657e85a8f1b317217b1c974456e939f59b5`.

The preceding pure-translation attempt was rejected for palm/bin collisions
up to 61 mm. The second candidate's extra 8 mm descent was rejected on one
source for 6.75 mm forearm/bin contact. Both rejected attempts are retained.
Using measured TCP-to-Can attachments from the preceding successful-grasp
rollouts and the original compiled Can collision vertices, the admitted 40-degree
policy predicts the bottom surface crossing support by 0.37 mm and 4.91 mm.
These predictions suggest that measured support can stop descent before the
nominal endpoint. They are not measurements from the new placement rollout.

Physical execution confirms the pass above for `6ad5c...`. For `1d0214...`,
rotation first crosses three degrees at 7.21 seconds during free traverse,
with zero support force; support starts at 13.29 seconds. Its peak rotation
occurs at 10.63 seconds during unsupported lowering. Changing support debounce
or hold time cannot repair this failure. The 50-degree backup worsens drift
to 3.692 degrees and is not the selected policy.

The supported-control follow-up shows that tighter apertures can pass the
first episode's grasp gates, but its contact010/020 object bottom surfaces stop
only 0.029/0.569 mm above support. The 2 mm correction has now passed full geometric admission and physical
execution with the successful closures above. The controller stops on measured
support rather than pressing through the entire nominal descent.

The adjacent [evidence file](can-physics-followthrough.json) contains exact
replay bindings, contact events and feasibility measurements. Full records are
also under `/mnt/reachy-retarget/diagnostics/can-nearpass-physics-v1/` and
`/mnt/reachy-retarget/diagnostics/can-open-path-v1/`. Only the two training
episodes informed these proposals; no heldout policy tuning was performed.

## Optional post-release speedup

`retreat_speed_factor` defaults to one, preserving the executed baseline.
Factors two and three resample only the retreat after the hand has opened.
Every grasp, carry, support and opening row remains byte-equal to the baseline;
all original suffix rows and the source-clock mapping remain present. The new
references receive full IK/contact admission, including five aperture samples.
At factor three the inserted duration falls to 13.65 / 10.04 seconds, saving
3.24 / 2.11 seconds. Synthetic measured-support stops at 10%, 50%, 90% and 100%
of descent keep retreat references below 0.589 / 0.521 rad/s. All four subsequent
original-model rollouts pass every physics gate and the exact actuator replay
audit. Total duration changes from 25.49 to 23.06 / 22.25 seconds for the first
episode, and from 20.97 to 19.39 / 18.86 seconds for the second. Maximum measured
arm speed stays below 0.800 rad/s; object/environment penetration stays below
0.814 mm. These are repeated policies on two development episodes.

The same-input planning check preserves the prefix exactly. Cross-worker
regeneration is bit-identical for two physical comparisons; the other two
regenerated prefixes differ by at most 2.50e-15 in joint reference, which contact
dynamics amplifies into measurable state differences. Each run's own actuator
replay reproduces its qpos, qvel and object poses exactly. Thus the speedup has
four passing executions, not a claim of cross-worker bit-identical physics.
The optional implementation SHA256 is
`3a20dd5bdafce54e9004a29135b739a45fc8e934929d8e4e5909d72a9fdae29d`;
this is distinct from the code snapshot for the successful default-speed runs.
