# MoMaGen: fixed-source-base reachability diagnosis

All six v3 episodes executed all-frame Reachy IK and failed the kinematic gate. This audit distinguishes geometric infeasibility from numerical solver behavior; it does not claim physical validation.

The pinned Reachy model has shoulders at `[-0.010, ±0.200, 1.166]` m. Summing the exact shoulder-to-tip chain lengths gives the rigorous orientation-independent upper bound `0.28 + 0.28 + 0.10 = 0.660` m. Therefore the fixed-torso tip cannot reach below world height `0.506` m even with unrestricted planar base motion. The source initial hand targets require `0.822–0.836` m from the corresponding Reachy shoulders: every episode starts with at least `0.162–0.176` m unavoidable position error under the frozen source base path.

| Scenario | Frames provably outside the 2 cm bound, either hand | Minimum target world Z (m) |
|---|---:|---:|
| bringing_water | 3324/3324 | 0.434 |
| clean_pan | 1586/1751 | 0.392 |
| dishes_away | 1972/2788 | 0.356 |
| pick_cup | 909/909 | 0.415 |
| picking_up_trash | 2483/2483 | 0.195 |
| tidy_table | 1273/1273 | 0.386 |

The geometric count uses the actual saved target and planar base configuration at every original frame, not a sampled workspace or an optimizer result. Within this envelope is necessary, not sufficient: joint limits, orientation and collision constraints can further reduce reachability. Source clocks, object trajectories, geometry, mass and scale were untouched.

Mapping checks passed:

- MoMaGen's pinned bimanual interface concatenates achieved left then right EEF transforms in world coordinates.
- R1 inherits `get_position_orientation()` from HolonomicBaseRobot, which delegates to the moving `base_footprint_link`; R1 names that link `base_link`. The data do not describe a static articulation root.
- Reachy targets are `l_arm_tip` and `r_arm_tip`. Arm q/v indices are looked up by joint name, correctly skipping the three neck joints between the arms.
- Numerical finite differences of all 14 active arm Jacobian columns agree with the IK convention to `3.1e-8` maximum absolute error.
- Source base tilt is small (largest vertical-axis component deviation 0.0175); projecting it to the planar target base does not explain the decimeter errors.

Warm starting adds a secondary numerical failure. At clean_pan frame 1732, the saved 1.460 m position error stays unchanged after 300 more iterations from the same configuration. Restarting from Reachy's neutral seed gives 0.378 m, still infeasible. Similar seed sensitivity exists in other tasks; more iterations alone do not fix the morphology mismatch.

The current initial-frame tool rotation is an explicitly derived alignment, not a verified source-EEF-to-Reachy-pad calibration. No heuristic base offset was added after this diagnosis. A future experiment must preserve the source trajectories separately and distinguish task-active hand constraints, verified pad-frame mapping and derived robot base placement. Floor-reaching references remain blocked with the current fixed torso.

Evidence, exact runtime identity, per-hand bounds, joint indices, seed probes, and reproducible read-only scripts are in [momagen-reachability.json](momagen-reachability.json). The seeded random workspace sample is supplementary only; the stated impossibility bound comes from exact link lengths.

Primary source frame definitions: [MoMaGen bimanual interface](https://github.com/ChengshuLi/MoMaGen/blob/5da621667669b15b97d7a220d086283b3c8ad0e5/momagen/env_interfaces/omnigibson.py), [HolonomicBaseRobot](https://github.com/StanfordVL/BEHAVIOR-1K/blob/2ca5503895b2c81e02226dbe11c3fee1a68b5d6c/OmniGibson/omnigibson/robots/holonomic_base_robot.py), [R1 link names](https://github.com/StanfordVL/BEHAVIOR-1K/blob/2ca5503895b2c81e02226dbe11c3fee1a68b5d6c/OmniGibson/omnigibson/robots/r1.py).

## Task-aware mobile follow-up (2026-10-07)

The fixed-source-base audit above remains correct for its declared constraints.
It is not a proof that every task is infeasible for Reachy. In particular, the
pinned `r1_pick_cup` task spec declares only the left hand as manipulating
`coffee_cup_7`; the right hand has neither an object reference nor an attached
object. Its low idle source target should not constrain the manipulation hand.
The left tip's minimum target height is 0.678894 m, above the 0.506 m absolute
height bound, while the idle right tip violates that bound in 833 of 909 rows.

An explicit opt-in policy now solves the declared task-active arms and planar
base, keeping inactive arms at Reachy neutral. It binds the exact source task
configuration text, checksum, URL and revision. Both original hand target arrays,
all task-object references and the original clock remain archived. The default
fixed-base bimanual comparison is unchanged. The policy uses the stricter 2 mm /
20 mrad IK gate and records its missing pad calibration, fixture and physical
validation rather than publishing robot observations or issued controls.

Five of six declared base-yaw seeds pass all 909 rows. All six numeric attempts,
including the failed seed, were published and read back on another worker. The
zero-yaw seed was then run through the versioned native pipeline and its explicit
agent review, producing a common-format derived archive at
`/mnt/reachy-retarget/datasets/momagen/episodes/momagen-r1-pick-cup-task-hands-v4-ik-attempt.hdf5`.
Its maximum IK errors are 0.991 mm and 9.809 mrad. Source geometry availability,
cross-simulator asset terms and real grasp/fixture dynamics remain unresolved;
this result is not physics-validated or policy-ready.

[Probe evidence](momagen-task-hand-ik-probe-v1.json),
[pipeline and common-archive audit](momagen-task-hand-pipeline-v4.json).
