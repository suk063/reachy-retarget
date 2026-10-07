# Executed dynamics validation

Recorded after the frozen run completed: 2026-10-07T05:16:55.397358+00:00.

**Six of twenty evaluated source episodes pass every physical gate and an independent actuator-only replay.** Four passes are on the six development episodes used to select configurations; two are on fourteen additional episodes with those settings frozen. This is not broad object generalization. Successful shape families remain cans and cubes. No hardware was accessed.

Lift and Stack use an explicitly recorded source window starting at **0.25 s**, after passive initial settling and before any demonstrated closure. The original arrays are preserved. These windows count as the same source episodes, never as additional demonstrations. Other tasks use their complete source interval.

## Frozen results

| Source task | Complete physical passes | Development passes | Additional fixed-setting passes | Source interval |
| --- | ---: | ---: | ---: | --- |
| PickPlaceCan | 1/3 | 1/1 | 0/2 | Complete |
| Lift | 3/3 | 1/1 | 2/2 | From 0.25 s; same episode |
| Stack_D0 | 1/3 | 1/1 | 0/2 | From 0.25 s; same episode |
| PickCube-v1 | 1/5 | 1/1 | 0/4 | Complete |
| NutAssemblySquare | 0/3 | 0/1 | 0/2 | Complete |
| Threading_D0 | 0/3 | 0/1 | 0/2 | Complete |

All twenty attempts completed simulation and reproduced through actuator-only replay. Nine satisfy the sustained task endpoint predicate; only six also satisfy all grasp, contact, penetration, actual-speed, joint-margin and clearance gates. Endpoint success alone is not the reported physical pass. All replayed joint-position, joint-velocity and object-pose arrays matched the saved results exactly in this pinned runtime (maximum recorded numerical error **0.0**). This is an independent control replay in the same MuJoCo engine, not validation against a second engine or hardware.

Parameters are frozen per task, not shared universally across tasks. The additional episodes were excluded from this parameter search, but some had historical baseline evaluations. Both the first frozen run and this second run remain saved. These are small, deliberately selected samples, not unbiased dataset-wide rates or unseen-object tests.

Nine remaining local episodes are explicitly unsupported: three ToolHang, three TwoArmTransport, and three HUMOTO. ToolHang/Transport scenes compile but lack a complete sequential/bimanual task planner here. HUMOTO lacks a verified source physics scene. They are not counted as physical passes or failures.

## Measured passing attempts

| Episode | Slip translation (mm) | Slip rotation (deg) | Hand/object penetration (mm) | Bilateral carry | Final stable hold (s) | Simulated duration (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PickPlaceCan / demo_0 | 1.019 | 0.432 | 0.311 | 0.9873 | 3.84 | 24.06 |
| Lift / demo_0 | 0.568 | 2.216 | 0.620 | 1.0000 | 1.83 | 11.46 |
| Lift / demo_1 | 0.708 | 2.850 | 0.623 | 1.0000 | 1.81 | 11.31 |
| Lift / demo_2 | 0.582 | 1.273 | 0.638 | 1.0000 | 1.83 | 11.16 |
| PickCube-v1 / traj_0 | 2.836 | 0.866 | 0.935 | 1.0000 | 2.68 | 20.26 |
| Stack_D0 / demo_0 | 0.590 | 2.857 | 0.457 | 1.0000 | 3.18 | 19.66 |

The acceptance limits include slip ≤3 mm/3°, hand/object penetration ≤1 mm, other specified penetration ≤2 mm, at least 95% bilateral carry contact, at least the final one second of task stability, actual arm/base speed limits, 25 mrad joint margin, and 9 mm robot self-clearance. Held tasks need bilateral contact at every substep during the final hold. All gates and exact predicates are described in [the method](dynamics-method.md).

## What changed and what failed

- Retargeting now selects a source-object grasp region and plans Reachy-specific approach, closure, carry and retreat, with full source object SE(3) used as a reference. Simulation-in-the-loop trials select robot commands and timings while preserving object geometry, mass, inertia and source friction.
- Measured finger contact force regulates closure. PickCube uses slower corrections after contact, a deeper gripper grasp, and measured-attachment calibration through robot actuator commands. This relies on exact simulated object state and is not a perception implementation.
- Controlled placement moves to the demonstrated final support pose before opening. The original Can and Stack release clocks produced excessive impact penetration; the failed versions remain saved.
- The full-interval Lift source begins with a passive cube fall that exceeds the 2 mm environment-contact limit. The 0.2 s crop was rejected because the source was still settling. The explicit 0.25 s stationary source window passes, with unchanged thresholds and source files. The six Lift/Stack windows are conditional results and cannot be represented as full-interval passes.
- Can demo_1 still slips and loses contact; demo_2 loses the grasp. Stack demo_1 fails acquisition, and demo_2 exceeds the 3° drift limit. PickCube additional episodes fail through slip, missed grasp, joint margin or slight penetration.
- Square can form a stable grasp in one development setting but does not complete placement. Its approach/base path and thin handle remain problematic. Threading grasps and lifts the needle, but misses the ring and/or slips; contact can move the freely simulated tripod. A bounded base workspace improves tracking but is not a general collision-free navigation planner.

The first frozen full-interval run, `frozen_matrix_v1`, passed 1/20 under these gates. The final `frozen_matrix_v2` result combines controller improvements with the explicitly changed Lift/Stack input windows; **1/20 → 6/20 is therefore not a like-for-like full-trajectory performance comparison**. Development attempts, their failed states and the original frozen results are retained.

## Evidence and reproduction

- [Frozen final configuration](../configs/dynamics-validation-v2.json) and [first frozen configuration](../configs/dynamics-validation-v1.json). Each task records the selected development episode, parameters and original selection-result hash.
- `runs/object-matrix/runs/dynamics/frozen_matrix_v2/benchmark.json` contains all 20 results, nine unsupported episodes and split labels. Each episode directory contains `result.json`, `scene.xml`, `replay.h5`, exact loaded source snapshots and `actuator-replay-audit.json`.
- `runs/object-matrix/runs/dynamics/tuning-inventory-v1.json` preserves the earlier development inventory; `all-attempts-final.json` inventories every retained attempt without counting retries as new demonstrations.
- [Locked acquisition inputs](../configs/object-matrix-lock.jsonl.gz): 54 acquired raw files / 673,639,339 bytes across four demonstration sources plus assets/evidence. All raw hashes and sizes match. Six failed discovery probes remain excluded from the portable download lock, with failure evidence retained.
- [Scene construction and provenance](object-scenes.md): 26 scenes compile and preserve verified source geometry/inertial/contact fields. This is a prerequisite, not a manipulation-success claim.

```bash
export REACHY_RETARGET_CONTROL="$(cd ../reachy-control && pwd)"
export REACHY_RETARGET_BACKEND=reference
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m reachy_retarget.dynamics_benchmark --root runs/object-matrix --config configs/dynamics-validation-v2.json --label reproduction_v2
.venv/bin/python -m reachy_retarget.dynamics_audit runs/object-matrix/runs/dynamics/reproduction_v2/6ad5cc23ec6d1c207da3e4b2
```

Use a new label; attempts are never overwritten. For a fresh root, first run the explicit lock/discover/fetch/normalize commands in the [README](../README.md) and [scene reproduction guide](object-scenes.md). No dataset download occurs on import, during simulation, or in unit tests.

Validation used Python 3.12, MuJoCo 3.15.0, NumPy 2.5.3 and Pinocchio 4.1.0 with the read-only Reachy reference controller. **262 offline tests pass**, including source/task contracts, falsified-replay rejection, exact asset hashes, source-window safety and scene fidelity. `git diff --check` passes.

## Visualization and remaining scope

The generic mjviser replay preserves all recorded object states and the source scene, plus Reachy visual meshes/materials and the existing visual-only tripod repair. Can and PickCube viewer checks have zero object-position mismatch and hand-FK mismatch below 7e-13 m. Browser playback is stored-state visualization; it does not supply forces to the validation run. All 14 missing robosuite source textures were explicitly acquired. Primitive texture projection limitations and ManiSkill floor appearance are reported separately.

Local viewers: [Can](http://127.0.0.1:8082) and [PickCube](http://127.0.0.1:8083). The display defaults to 2× playback and is adjustable; physics was integrated at its recorded timestep. The older viewer on port 8081 was preserved.

The base is an ideal actuated planar x/y/yaw model. Wheel traction, locomotion, localization and long-horizon obstacle navigation have not been validated. The implemented search is bounded grasp/controller tuning, not MPC/CEM. Generalization needs a richer contact/grasp search and receding-horizon actuator optimization with base/environment constraints; source object tracking and robust phase transitions must also be addressed. The [methods study](datasets/dynamics-aware-retargeting-methods.md) separates these future methods from implemented results.

The [expanded mobile dataset survey](datasets/README.md) contains 37 mobile robot resource profiles, 12 human/mobile profiles and 11 methods, with access, schemas, object/base states, physical assumptions, licensing, source revisions and overlaps. Those profile counts are not independent dataset counts or downloaded demonstrations.
