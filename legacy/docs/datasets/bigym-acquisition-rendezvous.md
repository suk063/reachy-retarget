# Measured acquisition rendezvous

This development experiment uses the existing c795 MovePlate source group. It
is not a new demonstration. The held-out 103d source is not used. Original native
samples, binary gripper intent, task-object trajectories, and the original full
Reachy reference remain archived. The native 4.0/4.1 replay-version uncertainty
and unavailable initial pre-action sample remain unchanged.

The previous failure began before lift: source closure row515 places the
requested relocated pad center 38.293 mm from the selected object-local patch.
Waiting blindly at the later row764 calibration anchor is unsuitable because
the reference plate has already risen about49 mm from its settled rack pose.
The finite mesh audit instead selected row550. At the actual settled plate pose,
the selected patch projects inside both finite pad faces by6.39/6.58 mm. Its
closing line intersects one actual plate collider over[-6.637,1.918] mm, giving
a2.359 mm robot-only centering correction. This is a geometric rendezvous
proposal, not a claim of successful grasping.

`acquisition_rendezvous.Rendezvous` owns only detector history and phase
transitions. A dataset geometry adapter supplies actual alignment and bilateral
contact guards. It never accesses physics state. BiGym's adapter binds compiled
scene geometry, the original source row, and its calibration evidence.

The declared experiment suppresses closure until source row550. It pauses source
command progression while physical time continues. A separate kinematic data
object is initialized from the measured integrator state; all subsequent
hypothetical assignments affect only robot joints in that clone. The actual
integrator receives actuator controls only. The measured plate remains free.
The nearest intersected convex plate collider defines a closing-axis-only
translation bounded by10 mm. The selected anchor must remain at least3 mm
inside both finite pad faces. Wrist orientation, base and inactive right-hand
reference are preserved.

Fresh IK and scene admission cover all2256 source-frame references and six
candidate-specific closed-to-open aperture samples. The three exact wheel/floor
pairs are exempted. Active-plate contact is checked during physical playback;
other inactive-hand fixture contacts are included in static admission. Limits
are3 mm fixture separation,9 mm sphere self separation,2 mm mesh self penetration,
2 mm two-hand position residual, and.02 rad orientation residual. These sampled
checks do not certify swept volume or servo-state clearance; unchanged actual
common gates remain required. The reference correction decays over.2 seconds of
the already declared baseline physical clock after resume. All original source
interval step counts remain unchanged; added time is recorded as explicit pauses.

Each wait is bounded by2 seconds and requires.1 seconds of contiguous guard
satisfaction. Open-hand settling requires measured TCP error<=2 mm/.02 rad,
TCP speed<=.02 m/s, the nominal closed-pad proxy's local chord-centering error
<=3 mm, and finite face margin>=3 mm. Closing additionally requires actual solved
normal force>=1 N on each distal pad and actual finger angle no more than.03 rad
above nominal contact. The closed-pad proxy is a CAD prediction transformed by
measured TCP, not an observed closed-pad position. Actual finger forces are
sampled immediately after the physics step and before cache synchronization.

The motor target uses the fixed CAD nominal angle minus.05 rad. Arm feedforward
is disabled. Original-contact and global4ms-contact variants use the same
controller, their existing2 ms/1 ms integrator steps, and a shared2 ms source
interval timing grid. Global4ms remains an explicitly altered contact model.
No object pose, mass, shape, material friction or task predicate changes.

Timeouts and admission rejections retain the observed prefix, terminal state,
issued controls and failed guards. A timeout cannot satisfy full-reference
coverage. Successful rendezvous would resume every remaining original row,
including the original release intent. No extra terminal hold is added. The
archive records original and actual clocks, phase codes, original source row,
per-physics commands and states, reference/correction arrays, exact independent
actuator replay, and common gates. No physical success is claimed merely from
passing the rendezvous or the native task predicate.

The first original-contact execution rejected the measured geometry at row550.
It saved4212 actual steps and their exact actuator replay (zero qpos/qvel/clock
error) to `datasets/bigym/episodes/bigym-acquisition550-margin-050-source.hdf5`.
The selected patch was3.3–3.5 mm outside both finite faces; the guard prevented
closure. Full-source coverage is false. This is a retained partial failure.

An independent prefix replay located first palm/plate force at source row465
(7.650 s), before the original row515 closure intent. The palm pushed the plate
about6 mm; delaying closure allowed it to rebound by row550. The new optional
`finite_patch_translation` proposal therefore solves a minimum-norm robot
translation with both finite-face halfspaces and actual collider centering.
The same object-local patch is fixed, total displacement remains<=10 mm, and
wrist orientation remains unchanged. Its desired geometric margin is4 mm,
leaving the actual measured guard at3 mm. This option requires fresh full
admission; it does not turn the rejected closing-axis-only attempt into a pass.

Memory use is tracked separately from physics. The recording contains21,904
observed numeric bytes per integration step. Once all rows are losslessly
stacked, the producer releases the duplicate row buffers and command lists
before NPZ export and independent replay. Every recorded value is retained;
no temporal decimation, sensor omission, or precision change is used.

The7.533 mm finite-patch proposal passed all2256 IK/fixture samples but timed out
in its2 s settle phase. It retained5212 steps; the1000 added2 ms steps have a
constant source time and increasing actual time. No closing or carry phase was
entered. Exact independent replay again had zero qpos/qvel/clock error. The
canonical archive is `bigym-acquisition550-patch-margin-050-source.hdf5`; both
partial failures have full checksum-verified exports and independent cross-pod
readback. Evidence JSON files beside this document bind the hashes and results.

The second failure isolated mechanical obstruction: the actual terminal palm
exerted12.0 N while both distal-pad forces were zero. The TCP stopped24.69 mm
short of its target. On a separate kinematic clone holding that measured plate
fixed, the commanded palm overlaps the plate by13.55 mm at both open and closed
apertures. A fixture-only check excluding the active object cannot catch this.
No alternate-contact rollout was used to bypass the obstruction. Subsequent
proposals require separate whole-hand/active-object geometry qualification,
with only verified inward distal pad faces eligible for intended contact.

The adapter now checks the measured rendezvous target against every non-pad
robot collision geom at all six command-aperture samples. A palm or linkage
within3 mm of the plate rejects the target before settling. Only the two named
distal bodies are omitted from this particular check; that omission does not
qualify their entire surfaces for contact. Finite inward-pad faces and actual
forces/penetration still require separate checks. The report is saved as
`acquisition/rendezvous-nonpad-object-admission.json`. This target check does
not certify the complete moving-object path or swept geometry.

A geometry-derived contact-frame proposal retains the exact original patch
`[-0.0003847501720931967, 0.07395797913321084, 0.009218442105546483]` meters in
the plate frame. Its local calibration changes the TCP by23.871 mm and the
wrist by19.23 degrees, and changes the nominal aperture from0.24569 to0.35 rad.
These are explicit robot-reference changes, not a minor timing adjustment.
Following the original continuous plate poses with that fixed tool transform
failed a first full-path IK/non-pad/self check when the prior base path was
held fixed. All2256 attempted frames and failures are retained in
`workspaces/bigym-fixed-patch-source-attached-preflight-20261007-01`.
That workspace contains zero physics steps and no physical success claim.

Allowing a bounded additional robot-base correction resolves the complete path:
all 2,256 both-hand targets pass within 0.200 mm / 1.608 mrad. The maximum added
base change is 64.92 mm and 5.33 degrees. Eighteen sampled aperture values across
the full path satisfy 3 mm non-pad/fixture clearance, with no open-pad overlap;
the nominal closest contacts lie on the inward finite pad faces. The exact
selected patch stays at least 11.59 mm inside their projected boundaries.
These are kinematic checks, and do not certify continuous swept geometry.
The declared interval retiming produces 17.010 seconds before any bounded
acquisition pauses. Source timestamps and task-object references are unchanged.
The retained rejected fixed-base attempt, interrupted optimizer checkpoint,
successful geometric proposal and independent cross-pod readback are bound in
[the contact-frame evidence](bigym-fixed-patch-contact-frame-v1-evidence.json).

The original-contact execution passed the measured rendezvous after 0.102 s
of settling and 0.436 s of closing. Both intended inward pad faces then carried
about 9 N, but contact weakened before lift and the complete task failed.
The retained full execution has 8,774 steps and exact actuator replay. The
global-4ms comparison stopped at row 550: its freely settled plate pose left
only 2.009 mm of palm clearance after the bounded translation correction.
Its 7,352-step prefix is also retained; the 3 mm gate was not relaxed.

An optional `post_acquisition_arm_feedforward_policy` now isolates velocity
compensation from acquisition. It requires `activation=after_measured_acquisition`
and reuses the bounded arm `kv/kp` helper on the remaining corrected reference.
Entry, settling and closing receive zero compensation. The first resumed
command starts at zero offset, followed by the declared smooth boundary taper.
Base, neck, gripper commands, model gains and object state remain unchanged.
The archive separates measured centering offsets from actual feedforward
offsets and records the enable flag, actual elapsed time since resumption,
activation source index and controller bounds. Physical gates and exact replay
remain required; improving target tracking alone does not establish a grasp.

The matched original-model comparison completed all 8,774 steps at both arm
compensation scales 0.5 and 1.0. The 3,945 entry/settle/close rows are bitwise
identical to the unmodified baseline, and each complete actuator replay is
exact. At source row 600, nominal TCP position error decreases from 10.721 mm
to 6.969 mm and 3.343 mm, respectively. Neither run lifts and carries the
plate successfully. The first scale-1 run on a different node had a numerical
prefix difference and is retained separately; the same-node repeat provides
the controlled comparison. All three complete failures have canonical common
archives and independent cross-node checksum verification.

The contact audit distinguishes the selected patch from actual load locations.
The selected patch remains about 10–11 mm inside the projected pad boundary,
but dominant measured load points lie only about 0–6 micrometers inside the
external boundary of the whole inward CAD face. This boundary is the convex
hull of all projected face vertices, not the edges of individual triangles.
The two measured inward pad normals have dot product -0.99958, corresponding
to a 1.665-degree departure from perfect opposition. Local plate surfaces
form a different wedge, so a geometric proposal must consider opposed finite
contact points rather than assume both surfaces can lie flush. Increasing
clamping strength has not been used to bypass this failure. The measured
comparison and artifact hashes are in
[the feedforward evidence](bigym-post-acquisition-feedforward-v1-evidence.json).

A subsequent measured facet-alignment candidate retains the same plate patch,
adds 10.18 degrees of wrist rotation and at most 2.244 mm of TCP translation
to the already-blended parent reference. Applying only this relative transform
avoids repeating the source-to-object approach; an earlier double-blend proposal
is retained as rejected because of joint discontinuities and excessive retiming.
All 2,256 both-hand frames and 18 aperture samples pass the discrete geometric
checks. This does not certify continuous swept geometry or grasp stability.

The original-contact rollout completes all 7,421 steps and lifts the plate
257.7 mm above the initial source reset pose. Measured bilateral acquisition
passes, but the overall result remains a physical failure. At 8.984 s, source
row 848, relative slip first exceeds 3 mm with zero fixture support force.
By source row 900 it reaches 14.54 mm and 8.02 degrees, still unsupported.
The first sampled target-rack contact occurs later, at 11.634 s. This is free
carry loss, rather than ambiguity in the placement phase. Every issued control
and state is retained, and independent actuator replay reproduces qpos, qvel
and time exactly. All 57 exported files and the common archive were verified
from another pod. The [complete evidence](bigym-opposed-contact-frame-v2-evidence.json)
keeps acquisition, carry, task outcome and source-version fidelity separate.

The solved contact audit shows `condim=3` at the pad/plate contacts. Sliding
friction is active, but the configured spin coefficient does not enable a
direct contact moment at this contact dimension. Actual spin moments are zero.
The 0.4 kg plate has local COM approximately `[0, 0, 0.01058005]` m. Its gravity
moment about the measured contact centroid grows from 0.0856 Nm at acquisition
to 0.1535 Nm at source row 900, with no fixture support. Sampled contacts already
reach the sliding friction cone boundary and remain near finite-face edges.
This supports evaluating grasp location and the moment carried by the contact
points; the presence of several newtons of normal force does not establish
stable transport. All exact one-step replays and the explicit correction of
an initially misleading inactive-spin capacity calculation are retained.

A robot-only alternative moves the patch 15.462 mm along the existing rim,
rotates its contact frame by 12 degrees, and adds a measured 0.5-degree normal
correction. The original object, source poses and timestamps, gains, materials
and 0.05-radian closure margin remain unchanged. Two complete rejected
geometric attempts are retained: one fails approach IK and another misses
the finite-face directional threshold. The accepted reference passes all
2,256 both-hand targets and 18 sampled aperture checks; its contact-frame
delta is blended over source rows 300–500 before the guarded rendezvous.

The original-model execution completes 8,037 steps and the final native task
predicate passes. Bilateral carry rises to 97.19%, and maximum lift is 281.6 mm.
Common physical validation still fails: relative slip reaches 9.151 mm and
6.935 degrees, original-model plate/rack overlap reaches 4.452 mm, and the
final stable segment is 0.466 s rather than the required 1 s. No undeclared
final hold was added. The first 3 mm slip occurs at source row 1074 without
fixture support; target-rack contact begins much later at row 1968. Exact
actuator replay and independent cross-pod archive verification both pass.
[The bound result](bigym-gravity-rim12-v3-evidence.json) distinguishes native
task completion from common physical success and retains the failed gates.

The native H1/Robotiq contact model also uses `condim=3`; its configured spin
coefficient is inactive. At four sampled carry poses the original native
geometry produces 5–7 pad/plate contact points with sliding coefficient
0.95–1.0. This differs from Reachy's sparse edge contacts. Source contact-model
inspection uses the compatible MuJoCo 3.1.5 runtime and archived states; it
does not recover the unavailable original 4.0 observations.

A bounded robot-only contact search checks finite pad support, original rack
clearance, opposed contact normals and the gravity moment about the pinch
line. Of 56 rim-neighborhood candidates, 37 pass static geometry and a
two-point balanced internal-force condition. This simplified force-cone
condition is not a full wrench or stability proof. Larger wrist rotations
must also pass the real Reachy joint limits and complete path checks before
any new physical attempt. The [bound search evidence](bigym-contact-force-cone-evidence.json)
retains failed searches and distinguishes static screening from whole-path
retargeting and physical validation.
