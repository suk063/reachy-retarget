# Active physical-success improvement goal

The active goal is to maximize actual MuJoCo success on feasible object-interaction
trajectories. Work is on `main`; original sources and all failed attempts are
retained. The goal remains active. Grasp success, final task success, complete
physics validation and source-model fidelity are separate outcomes.

The unconditional starting denominator is 20 full-interval physical candidates,
not the number of controller trials. Five PickCube candidates originally put the
robot into the table. Nine Lift/Stack/Square candidates exceed the 2 mm object
contact-depth limit during a passive initial drop before any robot contact.
These inputs remain visible in [the source admission audit](passive-source-admission.json).
An initial-source failure is not removed by cropping the clock or overwriting
the object's pose. Source-model and alternative-model experiments are reported
separately.

## Current measured findings

* The same Can base correction, elbow seed and vertical open-gripper approach
  admit both development trajectories. With `contact_002`, both acquire and
  retain bilateral grasps, finish the placement task and pass every existing
  gate except object/environment penetration during the final drop. Drift is
  0.621 mm / 2.04 degrees and 0.583 mm / 1.03 degrees. Straight lowering fails hand/bin clearance. A shared temporary base shift,
  held-object rotation and partial opening now pass full-path IK, collision and
  intermediate-aperture admission on both development trajectories. The second development episode now passes all 16 gates and independent
  actuator replay under the original contact model: drift 0.774 mm / 2.017°,
  object/environment depth 0.487 mm and final stable time 5.06 s. The first
  episode also now passes: the shared 2 mm extra descent obtains measured
  support under `contact_010` / `contact_020`, with rotation drift 2.235° /
  1.518°. The second passes that same placement geometry with `contact_005`
  (2.371°). These are two unique source-model successes selected using the
  same declared closure bank, not one universally passing motor target.
  [Support follow-up evidence](feasible-contact-followup-v1-evidence.json).
  [Placement evidence](feasible-placement-v1-evidence.json) and
  [control comparisons](feasible-supported-control-v1-evidence.json) retain
  every outcome. Passing controller trials of an episode are counted once.
  A subsequent speed check passes both episodes with twofold and threefold
  post-opening retreat speed factors, independently replayed. The threefold
  factor reduces total execution from 25.49 to 22.25 seconds and from 20.97
  to 18.86 seconds. Grasp acquisition and loaded placement are unchanged by
  this timing policy. These are still the same two development sources.
  [Speed and pad evidence](feasible-speed-and-pad-v1-evidence.json).
* PickCube's corrected base and constant grasp attachment admit its complete
  path. A replay audit found that the source object translates about 8.6 mm
  before target acquisition, while the physical target object stays at reset.
  A constant robot-path translation toward the verified reset center passed
  fresh full-path IK but worsened physical slip in all 54 trials (roughly
  34–45 degrees). It remains an explicitly rejected experiment, not a default
  correction. See [the retained comparison](feasible-reset-v1-evidence.json).
* Five PickCube controller variants pass every physical gate and independent
  actuator replay under the explicitly alternative `impratio100` friction
  regularization profile. They are five trials of one demonstration, not five
  independent successes. Object geometry, mass, inertia, friction coefficients,
  contact compliance and timestep are unchanged. These PickCube batches have zero original-model passes; the separate Can
  supported-placement result above is the first original-model success. See [the complete comparison](feasible-acquisition-v2-evidence.json).
* Two of four distinct BiGym recordings reproduce source task success in the
  pinned native runtime. One failed source trial and one historical action-limit
  incompatibility remain recorded. The recording/source-code version gap and
  reconstructed clock remain explicit. One successful source now has a complete admitted Reachy IK path. Its
  4.510-second source interval maps to a 20.466-second robot execution using
  local speed limits; 813 of 2,255 original intervals remain at original speed.
  Full target physics is now complete and replay reproduces all 10,233
  actual actuator steps. The native task predicate becomes true, but common
  validation fails final stability, 34.7 mm / 26.2° grasp drift and 4.14 mm
  initial rack penetration. The native source itself has 4.74 mm rack depth
  before acquisition, so this source-contact failure is retained explicitly.
  This is not a physics-validated target demonstration. The second
  successful independent source is reserved for validation with frozen settings.

* With one explicit alternative global 4 ms contact profile, local 0.7 rad/s
  reference cap and bounded velocity/integral tracking, both development Lift
  episodes and the first Stack episode pass all physical gates and actuator
  replay using the same `contact005_velocity` settings. The other Stack fails.
  Original passive-contact failures remain in the denominator. Half-timestep checks retain success for the second Lift and first Stack;
  the first Lift exceeds the drift limit. Previously held-out Lift and Stack
  fail acquisition with those frozen settings, and held-out Can fails placement
  admission. These failures remain visible and are not used to tune the held-out
  settings. [Transfer evidence](feasible-transfer-v1-evidence.json). [Box contact evidence](feasible-box-contact-v1-evidence.json).

## Throughput and storage

All 50 persistent workers are Ready on 50 distinct hosts and use the existing
shared PVC at `/mnt/reachy-retarget`. Container service updates preserved pod UIDs
and storage. Terminal-job caching removes repeated CephFS scans, and each claim
selects an immutable source release. The acquisition-v2 batch used all 50 workers
with 49 overlapping claims; placement-v1 also used all 50 workers with 46
overlapping claims. This measures worker participation, not constant 100% CPU
utilization. One failed worker was automatically replaced; 50 current workers
remain Ready. A stale terminating pod is not counted as an active worker.

The state-observation bottleneck has been measured and fixed: on the actual
BiGym scene, 100 captures decreased from 6.363 s to 0.507 s (12.56 times faster).
All 63 arrays were bit-identical on six recorded states, and metadata matched.
No field, pose, control interval or image policy was changed. See
[the full evidence](state-observer-performance.json). Source publication now
verifies and reuses the unchanged large robot-asset reference instead of
transmitting it again on every iteration.

Every rollout keeps full robot state, actual issued controls, controller memory,
task-object state and terminal state in the common state-only format. RGB is
not stored. An IK/admission rejection retains its proposal and diagnostic but
is never mislabeled as a physical attempt. The acquisition-v1 Cube cache-loader
errors are preserved; acquisition-v2 retries those unchanged experiments after
correcting the admission guard.

## Reproducible evidence

* [Contact controls](feasible-contact-v1-evidence.json),
  [tracking compensation](feasible-tracking-v1-evidence.json),
  [local timing](feasible-timing-v1-evidence.json).
* [Base and grasp geometry](feasible-geometry-v1-evidence.json),
  [approach and acquisition](feasible-acquisition-v1-evidence.json),
  [closure and solver comparison](feasible-acquisition-v2-evidence.json).
* [PickCube acquisition cause](pickcube-grasp-acquisition-audit.json),
  [Can follow-through](can-physics-followthrough.md),
  [worker refresh](worker-service-final-refresh.json).

Experiment manifests in `configs/feasible-*.json` declare inputs, development
splits, source checksums and immutable runtime hashes. Alternative solver/contact
assumptions are never silently inherited as original-model successes. Held-out
episodes must not select controller or grasp settings.

## Additional independent episode batch

The completed frozen baseline evaluates 50 previously unexecuted source episodes:
22 Can, 21 Lift and seven Stack, selected by source sequence before outcomes.
The three already acquired raw files are copied with verified source hashes;
normalization runs in a new cluster import and each episode receives an isolated
workspace. Earlier source groups are explicitly excluded from this new count.
The original 20-episode denominator remains a separate cohort. Neither source
normalization nor planned batch size is counted as a physical success. All 50
were normalized, retargeted and physically executed; the baseline passed zero.
Its worker claim-to-completion interval was 100.75 seconds, excluding source
staging and normalization. It used 44 workers with 42 overlapping claims.

The frozen box policy passes eight of 21 new Lift and one of seven new Stack
sources (9/28), using the explicitly alternative global 4 ms contact model.
Every one of the 28 complete state archives and independent actuator replays
was checked. A subsequent, declared two-controller recovery bank on the 19
failures adds three distinct Lift successes, producing an adaptive union of
12/28. This adaptive result is separate from the frozen-policy 9/28 result.
The recovery bank does not recover any of the 11 acquisition failures.
A subsequent center/face calibration and short approach proposal recovers two
more Lift sources, raising the adaptive alternative-model union to 14/28.
Two final-stability-only Lift failures then pass with an explicit 0.2-second
terminal robot hold, giving an adaptive union of 16/28. All 28 existing numeric
plan/replay fields remain exactly identical over the original execution prefix;
only 20 terminal control intervals were appended. Final stable durations are
1.15 and 1.11 seconds, with the original one-second threshold retained.
[Terminal hold audit](additional-box-terminal-hold-v1-audit.json).
This remains separate from frozen-policy 9/28 and original-model passes.
[Centered grasp comparison](additional-box-centered-v1-evidence.json).
Geometry and approach diagnosis takes priority over additional force sweeps.
See [the fixed policy](additional-box-policy-v1.md) and
[its archives](additional-box-policy-v1-archive-audit.json).

All four predeclared closure variants fail geometric admission on each of the
22 new Can sources, giving 0/22 full passes and 88 retained plan rejections,
not 88 failed physical rollouts. The original fixed base/arm placement does
not generalize to these scenes. Seventeen sources fail the initial base-path
admission and five fail inserted placement admission; environment collisions
dominate, while IK, joint margin and self-clearance pass. A bounded base/arm
and verified-bin placement search is being prepared as 66 worker chunks over
the same 22 sources. Selection uses declared geometric correction costs,
never physical success from the evaluation trajectory. The completed search
admits eight sources. Four subsequently pass all original-model physical gates
and exact actuator replay under the same `contact_002` control, yielding 4/22
over the original Can cohort and 4/8 over admitted paths. Seven passing trials
are four unique sources. Fourteen geometric failures and four physical failures
remain in the denominator. Two physical failures only exceed rotational drift;
two lose carry contact and subsequently fail placement. No acceptance threshold
was changed. [Selected physical results](additional-can-selected-physics-v1-evidence.json).
The completed six-roll bank preserves both calibrated pad-point trajectories
and checks positive fixture clearance on those 14 still-unadmitted sources.
Its 252 chunks completed without infrastructure/search errors in 283.21 seconds
from first claim to last completion, using 49 workers with 45 overlapping
claims. Two additional source paths pass full admission. One passes all four
closure variants under the original contact model, raising the adaptive Can
union to 5/22 unique sources (5/10 of geometrically admitted paths). The other
retains small rotational drift or a support-acquisition failure, depending on
closure. This is one additional success, not four independent demonstrations.
[Roll search](additional-can-roll-search-v1-evidence.json),
[roll physics](additional-can-roll-physics-v1-evidence.json).

All four previously passing new Can sources retain physical success with a
threefold unloaded retreat speed factor. Their total times decrease from
36.96/25.21/26.93/26.05 seconds to 33.85/22.42/23.28/22.34 seconds.
[Speed comparison](additional-can-retreat-speed-v1-evidence.json).
See [the fixed-policy diagnosis](can-new22-fixed-policy-diagnosis.json) and
[the complete first search](additional-can-geometry-search-v1-evidence.json).

BiGym pad-gap recalibration passes all-frame IK, but its two physical trials
still fail common grasp drift and final stability, despite satisfying the
native task predicate. Neither trial is a physics-validated demonstration.
The diagnostic now identifies a large gravity moment about the narrow grasp
axis. A robot-only upper-rim grasp proposal is under investigation; original
object geometry, mass, friction and recorded references remain unchanged.
[Pad physics evidence](feasible-bigym-pad-physics-v1-evidence.json).

The ten rejected generic box plans were also checked with explicit intended
gripper aperture envelopes: initial open reset, sampled closing/opening sweep
and calibrated closed carry. They still fail admission, so the geometry changes
are not adopted as defaults. Can's existing admission behavior is retained.
A partial-opening force-feedback bug is fixed: a supported-placement opening
command now bypasses grip-force regulation. Earlier four force-control attempts
are retained as failures and are not policy evidence for that corrected behavior.

## Shared-storage interruption

The 2026-10-07 execution continuation encountered shared-storage NUL-record
corruption and EACCES/ENOSPC failures. Readiness or retained pods do not imply
continued useful computation. See [the incident and recovery evidence](storage-incident-20261007.md).
Physical failures and infrastructure retries remain separate. The goal remains
active. The five completed node-local batches are now durably exported to the
PVC with cross-pod verification; ongoing new attempts remain tracked separately.

## Node-local continuation during the storage interruption

The first node-local recovery batch completed 92 experiments on 48 persistent
pods in 143.33 seconds from first execution start to last execution completion.
All 92 complete artifacts (2.75 GB compressed) were backed up and every regular
file hash verified. One interrupted Kubernetes tar stream was resumed against
the original finished spool without repeating simulation. The outcomes are 60
plan rejections, 23 physical failures and nine physical passing trials; all 32
physical attempts reproduce exactly in independent actuator replay and pass the
opening-authorization audit. No shared-storage writes were attempted by the
guarded experiment processes. Subsequent durable PVC publication is independently
verified in [the publication audit](node-local-publication-audit-v1.json).
[Full recovery evidence](local-feasible-recovery-v1-evidence.json).

The only newly successful source is Can `aec964373b1b75854938d701`: its declared
contact005 grasp passes with 4 or 6 mm extra supported descent versus the retained
2 mm failure. The 4 mm trial drifts 1.097 mm / 2.106 degrees and remains stable for
13.34 seconds under the original object/contact model. This raises the new Can
adaptive union to **6/22 sources, or 6/10 geometrically admitted sources**. Together
with the separate two development Can successes, eight unique original-model
Can sources have passed. The remaining 12 new Can sources are unadmitted by the
current search, not proven globally impossible.

The loaded-rotation slowdown bank does not recover either rotational-slip source.
Prefix velocity compensation prevents source-bin contact for the dropped Can
sources but may introduce or worsen slip on previously passing sources. It is
therefore retained as an explicit comparison, not promoted globally. A two-trial
stronger closure comparison follows exact replay evidence of unsaturated grip
torque and near-friction-limit free carry.

Three Stack prefix/approach proposals also completed with a separately hashed,
explicitly unpublished pod-local runtime. All three are geometric rejections.
The completed 254-job batch combines 216 full-task-centered non-box base
candidates, 36 conditional Stack face/roll admission candidates and two Can
closure physical comparisons. A complete source interval, original scene and
unchanged physical thresholds remain mandatory for every trial.
All 216 non-box candidates fail full-path IK; the fixed-base search does not
establish that the original tasks are globally infeasible. Both stronger Can
closures fail grasp drift. Thirty Stack face candidates reject geometry and six
hit a floating-point quintic endpoint error; all six fresh corrected retries
then also reject geometry. The 254-job batch used 47 pods over 222.63 seconds of
execution. [Expansion evidence](local-feasible-expansion-v1-evidence.json).

The next Can comparison derives a minimum jaw-leveling rotation and supported
height from the unchanged collision mesh and finite Reachy CAD pad faces.
Rotations at the original grasp anchor range from 0.87 to 5.65 degrees across
five development cases; required height changes are 0 to -3.17 mm. A 90-degree
long-pad-axis alignment is unnecessary and is not applied. These static checks
do not prove dynamic contact, and a constant TCP correction does not make the
source's hand/object relative motion rigid. Every modified reference requires
fresh complete IK, fixture-clearance and physical validation.

## Matched finite-pad comparison

The matched comparison adds two unique original-model Can successes: `47cc92`
and `6e13ca`. The new Can adaptive union is now **8/22 sources, or 8/10 sources
with a fully admitted path**. Including the separate two development sources,
ten unique Can demonstrations have passed original-model physics. The remaining
two admitted sources (`fe117`, `8ee`) and twelve unadmitted sources remain visible.
This is an adaptive union, not a frozen policy success rate on unseen data.

At `contact_002`, the original versus finite-pad rotation drifts are 5.391 versus
0.886 degrees (`47cc92`) and 3.976 versus 0.655 degrees (`6e13ca`); corrected
translation drift is 0.918 and 0.769 mm. Both also pass `contact_005`. The passing
`792e9d` regression remains successful. All seven passing trials across three
unique sources pass independent replay with exactly zero state error. All twenty
paired physical attempts retain complete state and issued controls. No object
geometry, mass, friction, physical thresholds or source interval changed.
[Matched comparison evidence](can-finite-pad-matched-v2-evidence.json).

The first comparison is retained as a separate experiment: it accidentally used
the zero-yaw base for two cases whose previous selected base included +15 degrees,
and introduced a 3 mm positive-clearance planning heuristic absent from the
original five selected plans. The matched follow-up corrects those offsets and
uses the original complete geometric and physical criteria in both comparison
arms. It does not loosen any physical gate. All eight initial plan rejections and
two physical failures remain retained. Thirty-six separate geometry chunks over
the twelve previously unadmitted sources find no complete candidate under the
same bounded bank; this does not prove global infeasibility.
[Initial comparison evidence](can-finite-pad-comparison-v1-evidence.json).

The measured roughly 22 mm difference in contact heights is not a corresponding
tilt of the finger centers: actual centerline height differences are only about
1–3 mm. Contact shifts to pad edges and the Can rim explain the distinction.
The constant anchor proposal is therefore tested physically, rather than assumed
to establish a rigid source hand/object attachment. The remaining `8ee` reference
also moves the finite pad beyond the side wall after acquisition; a separately
declared full-carry height correction is under investigation. A previously held-out
Can is evaluated using the frozen geometry policy without physical-outcome tuning.
That held-out source (`e876dc`) now passes all original-model physical gates and
independent actuator replay: 0.994 mm / 1.610 degrees drift and 6.10 seconds final
stability, with exactly zero replay state error. The three complete geometry
chunks selected the minimum declared correction before this sole physical trial.
This is one held-out success, not an estimate of broad generalization. It is
separate from the adaptive 8/22 cohort; eleven unique original-model Can sources
have now passed across the cohorts.
[Held-out evidence](can-finite-pad-heldout-v1-evidence.json).

## Centered box recovery and matched source-model checks

The exact CAD face-alignment proposal preserves the three earlier passing Can
sources tested as regressions and recovers two new Can sources. The separate
held-out Can succeeds with the frozen finite-pad policy; its outcome was not
used to tune that policy. The remaining admitted Can failures still require
translation-slip and late-carry rotation diagnosis. Eight explicit source-prefix
tracking comparisons all fail the complete physical criteria. Two post-anchor
height-corridor proposals fail placement admission and are not adopted.

For boxes, retaining the source grasp's vertical offset caused palm/table
interference in three Lift sources. Aligning the pad closing axis to the original
box face and using the actual box center recovers all three under the explicit
alternative global 4 ms contact model. A bounded backward base offset recovers a
fourth source. The additional-cohort adaptive alternative-model union is now
**20/28 unique sources (19/21 Lift and 1/7 Stack)**, separate from the frozen
policy's 9/28 and from original-model validation. Four passing variants of three
sources in the first comparison and two variants of one source in the second
comparison are counted as four newly successful demonstrations.

The three matched original-contact-model checks all fail the original 2 mm
object/environment penetration gate; two also exceed the 3-degree grasp drift
gate. None is counted as an original-model pass. All ten newly executed physical
attempts in these two box batches reproduce exactly under independent actuator
replay, with complete state and actual-control recording. The 46-job bank ran on
46 persistent pods and completed its start-to-backup interval in 42.31 seconds.
Forty-one candidates were rejected geometrically, three failed original-model
physics, and two passed alternative-model physics. No moving object was adjusted
after reset. See [centered grasp results](additional-lift-center-clearance-v1-evidence.json)
and [bounded base results](additional-box-center-base-bank-v1-evidence.json).

## Common-format publication repair

An export audit found that node-local full archives retained the episode JSON
and dataset metadata, but the initial canonical index exposed only HDF5 files.
The publisher now verifies and indexes all three hash-bound contract files,
checks their consistency, and reads them back from a different pod. Old
publication receipts and all simulation bytes remain unchanged; additive
`common-files-publication.json` and `common-files-readback.json` record repair.
A batch is complete only when its exact expected job IDs have all been published.
The repair does not rerun physics or turn a failed attempt into a success.

The common-format repair is independently verified for 508 published attempts,
including 87 state-only physics archives. All 87 pass the real canonical
`inspect_archive` and measured-state/actual-control contract checks. This includes
failed physics and does not imply 87 successful demonstrations.
[Canonical verification](canonical-common-state-inspection-v2.json) and
[cross-pod durable evidence](feasible-goal-continuation-v2-publication.json).

The next matched Can diagnostics retain all failures. Dominant CAD plane
alignment preserves six passing regression trials but recovers neither remaining
admitted source. Optional source-prefix base damping compensation improves the
measured fe117 TCP tracking error but does not recover slip and breaks one
previously passing regression; it remains disabled by default. The comparisons
cross frozen versions with maximum reference roundoff below 1.6e-13; exact numeric
arrays are not claimed bit-identical. A new geometry-selected placement admits
the 8ee finite-height-corridor correction, but all four matched closure/height
physical trials still fail the original drift thresholds. No threshold changed.
[Base tracking audit](can-base-feedforward-tracking-audit-v1.json),
[base outcomes](can-prefix-base-tracking-v1-evidence.json),
[height outcomes](can-corridor-placement-physics-v1-evidence.json).

Three matched original-model Lift checks confirm the earlier passive-admission
finding: the unchanged objects fall approximately 10 mm from their source reset
and exceed 3.236 mm table penetration at 0.064 s, before Reachy hand contact.
The original native source samples themselves exceed 2 mm. Smaller timesteps or
tighter solver tolerances do not eliminate the failure. These remain original-
model failures, rather than being attributed to grasp control or hidden through
reset-state changes. [Bound passive-source audit](additional-lift-source-model-passive-audit-v1.json).

## Original-model friction diagnosis and fixture forecast correction

Seven explicitly alternative solver trials preserve the unchanged object model
and all physical thresholds. Both `impratio100` and `noslip3` recover the remaining
8ee rotation failure, while fe117 translation slip remains. Three earlier passing
Can regressions remain successful with `impratio100`. The original-model Can
counts remain unchanged: 8/22 additional sources, 8/10 fully admitted sources,
and eleven unique original-model successes across development/additional/held-out
cohorts. Solver alternatives are never added to that denominator or numerator.
The profiles also specify Newton/tolerance settings; this is not reported as a
single-parameter friction experiment.

A planning audit identifies a concrete false rejection in three Stack paths:
CubeB was held at its initial floating height throughout static collision checks.
Using its unchanged recorded source poses on a separate kinematic query removes
those fixture collisions. The new optional forecast binds the complete source
clock, source/scene hashes and exact frame-zero correspondence. It is used only
by admission queries; physical rollout resets and integrates both free cubes
without any forecast state assignments. Matched static/forecast and original/4ms
model trials are pending. Earlier 3 mm positive-clearance rejections remain
retained; the matched comparison uses identical existing full-physics criteria.

All 530 node-local attempts through the friction diagnosis are durably published.
All 109 corresponding canonical physics archives pass the full state/actual-control
contract, sidecar checksum agreement and no-RGB inspection. They include failures
and alternative-model attempts, not 109 successful demonstrations.
[Canonical audit](canonical-common-state-inspection-v4.json).

A further bounded geometry proposal rotates the gripper around the unchanged
cylinder axis to the nearest original opposed polygon facets. Actual retained CAD
calibration requires only 0.864 degrees for fe117 and retains an 11.01 mm projected
finite-pad edge margin. This is optional, not a replacement object or a demonstrated
contact-area claim; complete recalibration, IK and physical checks are pending.
[Actual CAD preflight](can-nearest-facet-cad-preflight-v1.json).

## Matched fixture/facet results and new held-out cohort

The 46-job comparison completed on 46 persistent pods in 118.12 seconds of
execution. Can 8ee now passes the original model with the single predeclared
midpoint closure offset of 0.035 rad on the previous jaw-level geometry. Drift
is 2.294 mm / 2.737 degrees. All three earlier passing regressions also pass this
midpoint. The nearest-facet/dominant-plane proposal with 0.05 rad closure also
passes 8ee, but breaks geometric admission for 6e13, so it is not adopted
universally. The original-model additional Can adaptive union is now 9/22, or
9/10 fully admitted sources. Twelve unique Can sources have passed across all
cohorts: nine additional, two earlier development, and one held-out. fe117
remains a translation-slip failure.

Stack 6 (b1d985) now passes the explicit global 4 ms alternative model after
correcting the planning fixture forecast. Its source-model matched trial passes
every gate except object/environment penetration. Both static-fixture arms reject
the same robot policy, independently demonstrating the planning error. The box
alternative-model union becomes 21/28 (19/21 Lift, 2/7 Stack); original-model box
passes remain zero. Stack 4 still fails physical grasp/stability. Stack 9 reveals
a second planning issue: approach admission forces the right gripper fully open
during closed carry frames. The declared phase envelope removes that false
collision on the exact saved path; a fresh full-pipeline check is required.

One result download ended before its compression footer. Its partial bytes and
error remain retained; the same completed remote result was transferred again
without rerunning simulation. All 576 attempts and 134 canonical physics archives
through this batch are now durably published and independently verified for
complete state/actual-control recording, checksums and no RGB.
[Batch evidence](fixture-facet-recovery-v1-evidence.json),
[canonical audit](canonical-common-state-inspection-v5.json).

A new 24-source Can evaluation cohort is selected from untouched source demos
25–48 before its physical outcomes. The frozen finite-pad/source-azimuth geometry
bank and 0.035 rad closure are declared in the policy file, with primary prefix
arm feedforward 1 and a separately reported feedforward 0 ablation. Policies are
scored separately; neither episode crops nor successful variants enlarge the
denominators. No new downloads are needed.
[Predeclared policy](../configs/can-heldout-cohort-v2-policy.json).

## Stack release clearance and portable new-source execution

A 20 mm robot-only post-release vertical return recovered Stack source `9295ddc7f5668cb10cae9f98` in the explicitly alternative global contact4ms model. It passed all physical gates, exact actuator replay, and the complete common state/control contract, with 2.47 s final stability, 0.320 mm drift and 1.841 degree rotation. The adaptive alternative box tally is now 22/28 unique sources (19/21 Lift, 3/7 Stack); original-model box passes remain zero. A 10 mm original-model comparison still failed, and 5/10 mm alternative comparisons were retained. Two earlier four-job infrastructure/planning retry batches remain separate from physical evaluations.

All 24 new Can baseline sources were retargeted and replayed completely with no successful baseline physics. Their frozen correction evaluation is pending. A portable-parent materializer preserves original archive/source/plan bytes, explicitly relocates bound mesh paths, and checks exact compiled models and 500 recorded physical steps before cross-pod followups. This removes dependencies on vanished pod-local assets without changing physical assumptions.

## New 24-source frozen policy result

The 72 geometry chunks used 46 persistent workers and found complete candidates for 2/24 previously unused Can sources. Both predeclared physical policies each passed 1/24 overall, or 1/2 among the geometry-admitted sources. Their passing sources differed: primary arm feedforward passed `6de99732f28d23128d8ea274` (0.911 mm / 2.015 degree drift), while the predeclared no-feedforward ablation passed `d93ada736f2b8ff8518a82b0` (0.804 mm / 2.113 degree drift). All four physical attempts passed exact actuator replay and complete measured state/control checks. The two-policy outcome union is not a frozen policy success rate.

The dominant limitation is geometry admission: 1,244 candidate base screens rejected trajectories, largely robot/bin collision even when IK passed. These bounded failures are not proofs of global infeasibility. Followup causal diagnosis and any modified geometry pipeline are development work, separate from the frozen held-out result. Across all cohorts, 14 unique Can sources now have at least one original-model physical success, with adaptive and frozen policy scores kept separate.

## Original-model acquisition and carry recovery

The remaining admitted additional Can source `fe117e32c6f13e7ae68c4be4` now passes all original-model physical gates. Its bounded fixed-patch rendezvous holds the robot command clock for measured alignment and bilateral contact while physics keeps integrating. Every original plan index remains present. A calibrated 0.20 rad closure offset reduces free-carry slip to 1.659 mm / 0.896 degree, with maximum hand/object penetration 0.804 mm, below the unchanged 1 mm gate. A separately qualified nearest-facet/plane version also passes (1.912 mm / 0.905 degree; 0.802 mm penetration). Both have exact actuator replay, independently recomputed measured acquisition guards, complete source coverage, verified support-gated release and full state/control records.

The additional development Can tally is 10/22 overall and 10/10 among completely geometry-admitted sources. This is an adaptive union of per-source policies, not a frozen generalization estimate. Fifteen unique Can sources across all cohorts now have original-model success. The new 24-source fixed-policy scores remain 1/24 for each of the two predeclared policies. Three previous successful source policies rerun with acquisition disabled all still pass; two show tiny newly planned reference/aperture roundoff differences and are not bit-identical across runs, while each exact actuator replay passes.

## Full-body contact diagnosis and approach recovery

The new24 source cohort retains the original frozen1/24 score for each policy. A separate read-only24worker diagnosis binds original scenes and saved screen configurations, classifying actual robot/fixture bodies at the retained aperture endpoints. Most worst closed collisions involve the forearm; palm collisions also occur and cannot be assumed solvable by a base-only correction. The f418 eleven-azimuth/static-base development bank produced no admitted candidate; all33 chunk failures are durably retained.

The f76b open-prefix followup tests18 approach parameters. Six have complete approach/positive-fixture checks, but their standalone retimed cache was correctly rejected because subsequent complete path admission had not occurred. These are retained launcher failures, not physical attempts or completed admissions. The chosen minimum declared approach lift0.01m and pivot0.65 now proceeds through all45 supported-placement candidates before physics. The initial TCP height dominates the tested above-waypoint values; the causal change is later traverse/descent timing, not a claim that the hand now moves only10mm upward.

Runtime v36 preserves raw measured states, issued substep controls, terminal state and actuator replay before metadata finalization. A provenance conflict is rejected before physical integration. The full731test suite passes.

## New original-model Can success after approach and base correction

Source f76b0aa318b7a80670eebb8d now passes both declared arm feedforward comparisons using its original object and contact model. The robot approach pivot changes from0.5to0.65, and the inserted placement base offset changes from0.15mto0.12m in X. One of24 bounded placement-base corrections passed complete IK and the3mmfixture envelope before physical outcomes. FF1 drift is0.809129mm/1.468372degrees, maximum hand penetration0.220379mm, and final stability5.82s. All2,654 simulated rows exactly replay from issued actuator controls, and state/control archives are complete.

This adds one development source, making16 unique original-model Can successes across the explicitly separate cohorts. It does not alter either frozen new-cohort policy score of1/24. Runtime v37 adds opt-in full-source base velocity compensation with strict servo-unit validation and an opt-in mobile/right-arm IK dispatcher. Defaults retain prior behavior; the full733test suite passes. Forty-five geometric candidates for15 other forearm-obstruction sources are running before any placement or physical success claim.

## Mobile source and placement recovery (2026-10-07, continuation)

The 45 original-azimuth mobile IK attempts admitted all source rows for six of fifteen trajectories. Three seeds per admitted source passed. Wider base bounds and yaw seeds on the nine remaining sources admitted none of 27 attempts; this is a bounded solver result, not a proof of infeasibility.

The admitted-parent cache initially lacked an explicit extraction step from its durable archive. All 18 launcher failures were preserved. The materializer now verifies the archive and three exact member checksums, preserves their bytes, flushes them, and independently checks the compiled assets, reset and full admission on a second node. The corrected 18 placement chunks completed the original 45-candidate bank for six sources without admitting a full placement.

Five declared axial grasp alternatives per remaining source, with a correspondingly rotated robot-base seed, admitted eight variants from seven of nine sources. The original Can geometry, mass, friction and all source object/clock arrays remain unchanged. Together with the six original-azimuth paths and independent a60 pilot, fourteen source paths have recovered geometry. None of these geometric outcomes is counted as a new physical success.

A coupled-DOF regression demonstrated a solver issue: the scalar weighted position/orientation objective can prefer a point outside the position tolerance even when both original tolerances are simultaneously feasible. Optional explicit pose constraints now enforce the existing tolerances; numerical guards are slightly tighter, never relaxed. Placement can additionally derive its palm orientation from the unchanged target-bin wall normals and use constrained mobile IK. Full interpolated-path admission and original physical/replay gates remain mandatory. Runtime v40 passed 759 tests. The 87-job comparison batch uses all 46 root worker slots while the four reserved diagnostic workers continue Threading, BiGym and independent placement work.

The durable publication audit now covers 979 attempts and the same 181 complete state/control physics archives. Geometry caches and failed staging attempts do not increase the physics count. Original Can success remains sixteen unique sources across explicitly separated development and frozen cohorts.

## Hand geometry, continuous base yaw and a seventeenth Can recovery

Source `7f6fd175b0b85f22d11fba0d` now passes both original-model arm feedforward comparisons after the declared axial grasp and complete bounded mobile source-path admission. The selected original placement bank candidate is 35. Each full 2,222-row actuator replay is exact. Primary drift is 1.330 mm / 1.870 degrees, maximum hand penetration is 0.221 mm, and final stability is 4.84 s. Seventeen unique Can sources now have original-model physical success across the separate cohorts. Neither frozen 1/24 policy score changes.

Source `3641974ddfbb197d490d6785` passes all contact, drift, penetration and robot gates but fails support-gated release and final stability. Retrospective compiled geometry finds the actual Can bottom 2.632 mm above the original bin floor top at 0.820 m. An additional 4 mm of robot descent is inside the existing proposal cap; its full geometric retry passes every check except the unchanged 3 mm interpolated fixture envelope, so no physical retry is counted yet.

The mobile placement solver now keeps yaw continuous across the +/- pi representation boundary. A real solver regression exercises the previous discontinuity without modifying any object state or source clock. A separate hand-target screen evaluates the actual compiled hand meshes against every task fixture, preventing a nearest-wall correction from hitting the opposite wall. Source-path caches are immutable and independently verified. Six of the remaining twelve additional development sources now pass complete source-path geometry; placement and physics remain separate.

The next hand-screened bank covers seventeen admitted source paths in fifty-one chunks. A worker's node-local disk fell below the 50 decimal GB reserve before any experiment started. Its retained launcher failure is followed by an explicit retry in an exclusive shared-spool subtree. The guard continues to reject writes through source-asset aliases or into any other shared path. No disk threshold was lowered and no old attempt was deleted.

The canonical publication audit now includes 1,021 attempts and 185 full measured state/control physics archives, including failures. All 185 pass checksum, state, issued-control and no-RGB checks. Later partial geometry batches are not included in that complete-batch count. [Physical success evidence](can-axial-7f6-physics-v1-evidence.json), [support failure evidence](can-mobile-wall-364-physics-v1-evidence.json), [canonical audit](canonical-common-state-inspection-v13.json).

## Support timing and local source speed recover more original-model trajectories

The a3e3 development source passes after extending only the pre-opening hold from 0.35 to 0.55 s. Measured floor support first appears 30 ms after the old opening deadline. On the same pod, all 1,943 preceding timestamps, nominal/applied references, issued actuator controls, qpos and qvel are bit-identical between the failed baseline and passing trial. The 0.12 s support debounce, object location and intact-grasp conditions remain unchanged. No new descent or object-state command was introduced.

Local midpoint minimum-motion and interval timing recover source 83eb under arm feedforward 0. Source 50bd passes under feedforward 1 after a locally applied 0.45 m/s planar reference speed norm and 7 mm extra robot descent. Its independent single-factor trials retain the speed-only support failure and descent-only speed failure. Source 1895 passes both feedforward comparisons at 0.4 and 0.3 m/s source-path norm caps; its original placement already succeeded but its original source prefix exceeded the physical base-speed gate.

There are now 21 unique original-model Can successes: 12 additional development, two earlier development, one earlier held-out, two new-cohort successes under different frozen policies, and four post-frozen development recoveries. These adaptive and distinct-policy outcomes are not one frozen success rate. The additional cohort has 12/22 adaptive successes, 13 full-placement admitted sources, and nine still lacking full-placement admission. Three of those nine have newly recovered complete source-path geometry, so bounded rejection is not interpreted as impossibility.

The standalone a60 path now places and releases stably for 9.87 s under both original-model comparisons, with exact replay and complete state/control archives. It still fails source base speed and is not counted as physical success. BiGym's native plate-facet correction now lifts the original object by 257.7 mm above reset, but free-carry slip and final task failure remain. Threading's extra tracking compensation and rigid attachment modifications worsened drift; both are retained failures and the defaults stay unchanged.

The canonical audit covers 1,269 individually published attempts and 205 complete state/control physics archives, including partial-batch receipts, with explicit batch completeness. All 205 pass checksum, field semantics and no-RGB checks. Separate manually published a60 archives are linked in the current status rather than silently folded into this index. Runtime v46 passes all 781 tests. [Current status](current-retarget-status.json), [a3 support comparison](can-a3-support-wait-counterfactual-v1.json), [83eb evidence](can-mobile-midpoint-physics-v2-evidence.json), [50bd evidence](can-50bd-speed-support-v1-evidence.json), [canonical audit](canonical-common-state-inspection-v14.json).

## Native-facet contact recovery: 23 original-model Can source successes

Two carry paths that placed successfully but exceeded the 3 degree grasp-rotation gate now pass every physical gate after small robot grasp azimuth corrections to existing opposed native cylinder facets. D091 changes 2.714 degrees and 379 changes 3.355 degrees; measured rotational drift falls to 0.881 and 0.930 degrees. Objects, closure, source interval, selected placement, 0.30 m/s reference speed cap, actuator policy and final gates stay unchanged. Fresh whole-path geometry, full MuJoCo rollout, exact actuator replay and complete common state/control archives were checked. See [native-facet comparison](can-native-facet-carry-recovery-v1-evidence.json).

The adaptive original-model union is 23 unique sources: 13 additional, two earlier development, three held-out sources under their separately reported policies, and five post-frozen development recoveries. The additional 22-source cohort has 13 fully placement-admitted sources, all 13 with a physical pass. This adaptive union is not a frozen-policy rate; the new 24-source frozen primary and ablation remain 1/24 each. Local source-speed comparisons retain all 12 trials, including eight failures; four successes represent only one newly successful 1895 source ([source-speed evidence](can-mobile-source-speed-v1-evidence.json)).

The next 44 jobs evaluate three new source paths across placement banks, five known fixture-chord failures, and a bounded 364 touchdown correction. Newly solved midpoint knots may use 100 micrometre tighter position and fixture constraints to create interior reserve. Original knots and final 2 mm position / 3 mm fixture gates remain unchanged. All 784 tests pass before publishing runtime v47.

The independent canonical audit now covers 1,349 published attempts and 219 common state/control physics archives. All 219 pass checksum, field-completeness and no-RGB checks. Failed and alternative-model attempts remain included as labelled archives, not original-model successes. Separate manual a60 archives remain explicitly outside this root batch index.

## Two more original-model Can successes and a shared interpolation failure

A60 now passes both original-model comparisons after capping only fast source planar reference segments at 0.40 m/s. The prior 0.45 m/s trials remain failures. All four source-speed attempts have exact actuator replay and independently verified common state/control exports; see [a60 evidence](can-a60-source-speed-recovery-v1.json). Source 364 passes with feedforward 0 after the already declared additional 4 mm robot descent is admitted using local midpoint interior reserves. Feedforward 1 still fails measured support and is retained ([364 evidence](can-364-interior-touchdown-v1-evidence.json)). The adaptive original-model union is now 25 unique Can sources; frozen policy scores remain unchanged.

Full-array diagnosis distinguishes failures that placement changes cannot fix. Four sources have valid original IK knots but lose a fraction of a millimetre of fixture clearance on newly retimed joint-interpolation samples. Thirteen of 17 complete placement failures occur only on these source prefix/suffix samples; two have both source and insertion deficits, and two a2e2 candidates fail only during inserted lowering. The next repair will constrain only the invalid interpolated robot samples, preserving original source arrays and all final gates.

MoMaGen pick-cup is not intrinsically unreachable merely because its inactive right-hand reference lies below Reachy's workspace. The publisher's task spec declares only the left hand as active; five of six explicit mobile-base seeds follow every one of its 909 original left-hand targets within 2 mm / 20 mrad while the idle right arm stays neutral. Original targets and clock remain intact. All six probes, including the failed seed, are durably published. This is task-aware kinematics, with source geometry, pad calibration, fixture clearance and physical validation still missing. See [task-hand probe evidence](datasets/momagen-task-hand-ik-probe-v1.json).

Runtime v51 passes805 unit tests. Four source trajectories (2447,e92,fd3,ba76) now pass complete placement admission after bounded correction of newly retimed robot samples; their original coarse knots, hand/object targets and source clock are unchanged. Eight physical rollouts (four sources, two declared feedforward settings) are running. Additional source710/c59 reuse the checksum-bound admitted mobile reference during the independent approach check. Optional quarter probes address narrow inserted-path defects missed by midpoint-only checks. These are development corrections, not frozen-policy success claims.

The v16 common-archive audit verifies1,447 individual published attempts and221 physical archives with complete measured state/control and noRGB. All checks pass. Manual a60 archives and the new MoMaGen kinematic archive are separately labelled outside this root physics index. Native source-identity evidence confirms25 distinct successful Can demonstrations without double-counting versions or crops.

The adaptive original-model Can union is now30 unique native demonstrations. New recoveries are ba76, e92, fd3,2447 and6a; source identity evidence v2 verifies their native source_id/source_sequence groups, with manual a60/6a archives explicitly included once. The additional22cohort reaches15/22; the frozen24policy results remain1/24 per policy, with ten separate post-frozen development recoveries. FD3passes both source-speed policies;2447passesFF1andretainsFF0failure.6apasses both after geometry-derived15mm placementX correction and1mm additional descent.

Source-only timing did not fix7029because the remaining actual-speed peak is in inserted traverse. A separate optional mobile base axis cap now time-dilates only inserted intervals while preserving original source arrays and geometric IK knots; all physical speed gates remain unchanged. The v17 archive audit independently verifies235state/control archives from1,502publishedattempts.

F15FF0adds the31st original-model Can source success; its FF1actual-speed failure is retained. Source identity v3 verifies31distinct native demonstrations. The v18 common audit covers1,508published attempts and241complete state/control physics archives, with noRGB. Manual a60/6a and other dataset pilots remain explicitly separate from this root archive index.

Runtime v53 passes819tests. It includes the bounded native-wall axial-grasp API, source-initial-target preservation, insertion-only mobile base axis timing cap, and direct shared-PVC spool publication. Shared publication requires the exact batch/job artifact path and all file hashes; it avoids an unnecessary operator upload and preserves cross-pod readback. The261fresh API targets match the admitted831-row path exactly; cross-pod cache loading reproduces the original compiled assets/reset. Placement and physics remain separately required.


### Continued recovery: 33 native Can sources

Original-model successes now cover 33 distinct native Can demonstrations, verified by `can-original-success-source-identity-v4.json`. The additional cohort remains 15/22 physical passes; 710 newly has full geometric admission but fails dynamics. The newer 24-source cohort has 13 post-frozen development recoveries in addition to its two distinct frozen-policy successes. Each frozen policy remains 1/24; adaptive unions are not frozen success rates.

7029 passed all 16 original physical gates and independent command replay after changing only the inserted mobile-placement base-axis speed cap from 0.30 to 0.25 m/s. Its source-only speed reduction had already fixed the prefix, but the actual remaining peak was in inserted traverse. FF1 now has 1.334 mm / 1.243 degree grasp drift and 11.35 seconds final stability. FF0 still fails rotation and measured-support release and is retained. The direct shared-PVC publication path was used and independently verified.

A2 passed all original gates with FF0 after a native-partition and measured-attachment diagnosis produced a 7 mm placement-X correction; drift is 1.276 mm / 1.686 degrees. The FF1 failure and all unchanged source-prefix evidence remain in its cross-pod-verified archive. 710 FF0 has clean source/traverse but hits the same native partition during lower; its correction is being derived independently from its own actual attachment.

The 261 axial cache had a missing aperture-variant label. Nine launcher failures were retained. A separately derived v2 cache restores the canonical variant metadata while proving every numeric Candidate and source/reference array unchanged, and passes full placement-parent validation on another pod before retry. A malformed derivative binding during root materialization was also retained and corrected before any placement execution.

The root common-state audit now covers 1,524 independently published attempts and 245/245 complete measured-state/actual-control archives without RGB. Separate manual A60, 6a and A2 exports remain explicitly outside that root index. Four previously static-screen-only source failures and three older admitted-source placement failures are now assigned useful updated-logic jobs rather than being treated as globally infeasible.

## Resumed execution: 36 native Can sources and shared-storage archives

The cluster had been idle after the last batches finished. Three retained physical failures were resumed with single-cause corrections. The source, object, contact model, clock and every physical gate stayed unchanged, and every attempt is listed in [the recovery evidence](can-resume-v1-evidence.json).

* 3873 had failed only `actual_base_speed_limits` (peak base vx 0.658 m/s against the 0.611 m/s limit). Lowering the source planar cap and the inserted base axis cap produced 16 passing variants of this one source. One combination, a 0.25 m/s source cap, was rejected by the unchanged retimed-source repair and is retained.
* d755 kept its grasp through carry, but the Can met a fixture about 20 mm above the bin floor during lowering. This was measured as a jump in the hand-frame translation at t ≈ 15 s. A bounded release-XY grid recovered it: seven variants at +12 or +16 mm X pass with feedforward 1.0 or 0.5. Of the four +12/+16 mm X combinations with Y offset 0 or +6 mm, all four pass at feedforward 1.0 and three pass at 0.5. Smaller shifts still drift, and −6 mm Y shifts fail placement IK.
* 710 passed with the attachment-derived +12 mm X shift together with a −10° placement palm yaw. All 12 variants with −10° yaw passed, and every 0° or +10° variant failed IK. This matches the earlier diagnosis that the wrist was near its range limit.

All 35 passing attempts replay exactly from issued actuator controls, use no object welds, and assign the object state only at the initial reset. The adaptive original-model union is now 36 unique native Can demonstrations ([identity v5](can-original-success-source-identity-v5.json)). These are 16/22 additional sources, two earlier development sources, three held-out sources under their separate policies, and 15 post-frozen development recoveries. Both frozen 24-source policies remain at 1/24.

For the ten remaining sources with an admitted source path, a [lower-knot census](can-placement-lower-knot-census-v1.json) of 285 retained placement failures found the following. 263 failures occur during `lower` at smooth weights 0.35–0.97, with IK errors at the unchanged 2 mm / 20 mrad tolerances. They are reach/orientation limits partway down, not failures at the release pose. The derived `can-placement-grid-v2` retries the two deepest-progress postures per source with ±8 mm release XY and ±12/±24° palm yaw, using the current runtime pin. `can-placement-grid-v1` reused stale per-parent runtime pins, so 54 launches failed before planning. Those failures are retained, and v1 was superseded without launching its remaining jobs. 106b's 88 correctly pinned v1 grid points produced no admission.

Throughput and storage changes:

* Operator archives are now written by the executing pod directly to `/mnt/reachy-retarget/operator-backups/local-spool-pool` and verified against the artifact manifest. They are then re-read on a second pod. Only receipts and logs stay on the operator disk.
* The export step publishes from that PVC archive (`pvc_backup` mode) without re-uploading.
* Per-pod operator leases let concurrent batches share all 50 pods. A busy pod defers a job instead of failing it.
* An interrupted orchestrator can resume a PVC archive only after re-verifying every file hash.
* The remote verifier source is captured when the module is imported. Previously, editing the file under a running batch sent a mis-sliced verifier and caused 50 retained orchestration errors; their simulations were then backed up on resume.
* `cluster/offload_local_archives.py` can remove the 19.6 GB of legacy local archives after a byte-identical cross-pod check. It has not been run.
