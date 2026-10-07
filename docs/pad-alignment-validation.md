# Translation-only gripper-pad alignment

Implemented and evaluated on `main`, October 6, 2026. This implements the
[narrow contact-alignment experiment](contact-alignment-reset.md). It does not
enable the earlier whole-trajectory optimizer or overwrite existing retargets.

**The geometric correction is implemented; broad physical success is not.**
Across six selected source episodes, the uncorrected condition acquired no
bilateral lifted grasps. Translation correction acquired grasps in Can and Lift.
No attempt passed every physical gate. Five corrected trajectories were simulated;
Square's correction was retained but rejected before simulation because its
fixed-base IK residual exceeded 2 mm. These are inspected development examples,
not an unseen-object evaluation.

## What changes

`pad_alignment.py` reads each existing `motion.h5` and its provenance. The
baseline uses that export's arm/base references and actual FK hand trajectory.
The source EEF origin in the object frame locates the demonstrated interaction
region. A line along Reachy's closing axis intersects the unchanged object's
convex collision regions. Connected overlapping pieces are merged so that a
thin convex decomposition piece cannot masquerade as a thin graspable object.

Opposing distal collision-face centroids define a pad proxy. The original visual
mesh also has a dark insert (`pincette_distal.dae`, `ID12-mesh`) at approximately
the same inward surface. This is a geometric proxy checked against visual
geometry, not a manufacturer annotation of rubber material or compliance.
No mesh, inertia, friction, contact parameter or object state is changed.

The gripper angle whose measured pad gap matches the local object section is
used to calculate the pad-to-TCP transform. This inferred angle does **not**
replace the gripper commands. One translation in the object frame then aligns
the pad midpoint with the source interaction region. The same rule applies to
every task; no task name selects an offset, grasp yaw or force.

The corrections measured here are approximately 49.6–54.5 mm because the original
export places the Reachy TCP at the source EEF position. This is different from
the approximately 18 mm endpoint-to-pad discrepancy in the earlier diagnostic:
the full TCP-to-pad displacement must be accounted for when no position
calibration was present at all. Neither value is installed as a universal offset.

Only the right-arm positional target changes. Target orientation remains exact.
Right-arm IK uses the frozen base trajectory; it never adds base motion to make
a correction reachable. A quintic translation blend finishes before the open
distal-finger envelope can reach the object and fades after release. The initial
state and unaffected free-motion intervals remain identical.

## Frozen comparison conditions

- Same complete source interval, rigid scene placement, scene bytes, object
  references, initial positions/velocities and controller gains.
- Same saved timing, with no new slowdown or inserted closure pause. A common
  two-second final-state observation hold follows each export.
- Same commanded base path, inactive arm and gripper commands. Their **actual
  actuator commands** are checked for exact equality on the common recorded
  prefix. Actual physical positions can differ in response to contact.
- Both conditions use the same joint-reference actuator controller. Integral
  compensation, attachment feedback, force feedback and support-triggered release
  are disabled in this experiment so they cannot alter the frozen channels.
  This is not a replay of the historical task-tuned or whole-body controllers.
- Some legacy exports lack a gripper channel. For those, the verified source
  closure is mapped identically in both conditions and explicitly marked derived.
- The existing strict MuJoCo validator and independent actuator replay audit
  remain unchanged. No object is welded, driven or overwritten during validation.

All five executed pairs pass the saved `frozen_factor_checks`. The sixth baseline
is still simulated and reported; its rejected corrected IK is not called a
physics-tested failure or an extra demonstration.

## Measured outcomes

| Task, first source episode | Before: bilateral lifted grasp | After: bilateral lifted grasp | Corrected outcome |
| --- | --- | --- | --- |
| PickPlaceCan | No | Yes | Lifts 31.5 mm, then slips; 19.1 mm / 12.0° drift, 1.35 mm hand/object penetration, collision/speed failures |
| Lift | No | Yes | 1.70 s final task hold and 100% bilateral carry; 1.22 mm / 5.18° drift, 3.61 mm hand/object penetration, 1.031 rad/s peak arm speed |
| NutAssemblySquare | No | Not simulated | 3.007 mm IK position residual exceeds the fixed 2 mm correction-execution limit |
| PickCube-v1 | No | No | Both stop on the same pre-existing robot/table collision, before correction can help |
| Stack_D0 | No | No | Acquisition, contact and joint-margin failures |
| Threading_D0 | No | No | Acquisition, environment collision and actual speed failures |

Lift also retains its passive initial object/table impact failure. The full
source interval was not cropped to remove it. Missing-grasp drift is marked
unavailable in the consolidated report, rather than interpreting zero as no slip.

Eleven physical recordings, including stopped prefixes, reproduce exactly from
actuator commands in the same pinned simulator. This establishes reproducibility,
not success. Original source data, failed trials and rejected corrected targets
are preserved. The initial late-blend experiment (`translation_v1`) and the
intermediate runs remain separate; they do not add independent source episodes.

The first experiment showed that blending only just before closure was too late:
open fingers already touched the Can during approach. The revised timing is
derived from the unchanged distal collision envelope and object section, not a
Can-specific timestamp. This fixes a calibration-transition issue without changing
the source clock, gripper events, orientation or approach-path planner.

## Evidence and reproduction

- [Consolidated result and invariant checks](pad-alignment-results.json).
- `runs/object-matrix/runs/pad-alignment/translation_final/summary.json` and each
  episode's `alignment.json`, `paired-targets.npz`, or `alignment-rejected.json`.
- `runs/object-matrix/runs/dynamics/pad_alignment/translation_final/{before,after}/ID/`
  contains exact scenes, implementation snapshots, controls, physical states,
  validator results and independent replay audits.
- **288 offline tests pass.** Tests cover frame-correct translation without input
  mutation, precontact blending, tapered object sections, merged convex pieces
  and opposing pad geometry. All 54 existing raw files / 673,639,339 bytes still
  match their recorded hashes and sizes.

```bash
export REACHY_RETARGET_CONTROL="$(cd ../reachy-control && pwd)"
export REACHY_RETARGET_BACKEND=reference
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m reachy_retarget.pad_alignment --root runs/object-matrix --label new_comparison --episode 6ad5cc23ec6d1c207da3e4b2 --episode 58e7f4f3c4178bc2c5f3f2c7
```

Use a fresh label. The command performs no downloads or robot access. Correction
is opt-in; it does not replace the default export with an unvalidated result.

Lift previews: [before](http://127.0.0.1:8094) and
[after](http://127.0.0.1:8095). Both retain Reachy source visual meshes/materials,
the visual tripod repair and available scene textures. Red markers show the TCP,
green markers the endpoints, orange markers the opposing surface centers and
their midpoint. Viewer FK mismatch is below 7e-13 m and object-position mismatch
is zero. Playback is stored-state visualization, separately labeled physical
failure; its adjustable 2× display speed does not alter the simulation clock.

To open another result, pass its `replay.h5` to `reachy_retarget.viewer` with
`--pad-markers --port PORT`. The overlays are read-only.

## Interpretation

Contact alignment can recover grasp acquisition without reconstructing the
whole task, as Can and Lift demonstrate. It is not sufficient for strict physical
success in this cohort. The experiment exposes remaining force/penetration,
tracking and pre-existing environment-collision failures. Those require separate
controlled changes; none is hidden by relaxing gates or silently replanning the
base. The results do not justify automatically applying this correction to every
dataset or restarting broad trajectory optimization.
