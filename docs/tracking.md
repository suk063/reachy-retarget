# Pure tracking retargeting (`reachy_retarget.tracking`)

## Purpose and separation

The manipulation retargeting (`reachy_retarget.retarget`, docs/design.md) is object-centric:
- it re-selects grasps;
- it relaxes orientation and idle-hand position away from objects;
- it rewrites `reference.tcp` from a first IK pass.

Tracking is the opposite. Reachy follows a **reference as given**:
- the world poses of both TCPs (`{l,r}_arm_tip`);
- a head rotation, the same rotation as reachy-control's `head_tip`;
- gripper openings;
- a nominal base path. The base itself is free; only the TCPs and the head are judged.

Frames Reachy cannot follow are tier-K failures. They are never reshaped into something reachable.

The package is kept apart from the manipulation pipeline:
- It lives in its own package, `reachy_retarget/tracking/`, with its own config (`TrackingConfig`) and CLI.
- Episodes carry `family = "tracking"` and their own datasets: `tracking/synthetic-v1` and `tracking/source/<source dataset>`.
- Output goes to its own run directories (`runs/tracking/...`).
- It reuses only embodiment-level pieces:
  - the robot model and self-collision spheres;
  - `retarget.wbik.FrameSolver` and `refine`;
  - `retarget.timing` (source paths only);
  - the schema;
  - `validate.kinematic.check`.
- It never loads the grasp, placement or gaze modules. `reachy_retarget/retarget/__init__.py` imports the manipulation pipeline lazily, and a test checks this.
- There is no physics tier: tracking episodes have only robot kinematics, so `tier.P` is `null`.
- reachy-control is a source of scenario ideas only. Its controller is not used, and none of its code is imported.

## Robot model and visualization (same as reachy-control)

The robot assets are byte-identical to reachy-control's:
- `robot/assets/reachy.urdf` (SHA-256 `63a1ecab…d448`);
- `collision_spheres.json` (`8d581c19…`);
- the meshes in `packages/`.

Cross-arm sphere pairs are enabled in both repos. The speed limits are the same:
- base 0.61 m/s per body axis and 114°/s;
- arms 1 rad/s;
- neck 30°/s;
- fingers 3 rad/s.

Hands are `l_arm_tip` / `r_arm_tip`.

The head target is a rotation in the base frame, as reachy-control computes it: `neck_R = base_from_world.R @ R_head`.
- The rotation of `head_tip` equals that of `head`; `head_tip` is `head` translated by (0.0564, 0, 0.062) m.
- Look-at targets aim the +X axis of `head_tip` from its origin (reachy-control `util/geometry.look_at`).

Self-clearance gate: 10 mm, reachy-control's CBF `MARGIN`. The manipulation retargeting uses 9 mm.

`python -m reachy_retarget.tracking.viewer --run DIR` ports reachy-control's Viser viewer (`experiment/viewer.py`, `util/visualization.py`):
- the same URDF meshes, floor grid, theme and colours;
- teal/blue target frames and paths for the left/right hand;
- orange/red actual hand frames and paths;
- purple base path, with the nominal base as a thin grey path and a target frame;
- a magenta gaze ray (`head_tip` +X): solid is actual, light is target;
- a collision-mesh toggle;
- play, time slider, speed, fit camera, previous and next.

Filters: cell, regime, neck mode, tier-K result, and final or all attempts.

`--collision` inspects the collision spheres with joint sliders. Spheres turn yellow below the 10 mm gate.

Link poses come from this repo's URDF kinematics. On 484 frames of a pilot episode they match pinocchio (reachy-control's library) to 1.2e-7 m. Every loaded episode is asserted to replay its stored TCPs (1e-5 m). Install with `uv pip install -e '.[viz]'`.

## Pipeline (`tracking.pipeline.track`)

1. **First frame.** `q_start` when given (synthetic references start at the robot's state), else a multi-start cold solve.
2. **Whole-body IK on every frame**, both TCPs, at full position and orientation weight:
   - No relaxation and no grasp offsets.
   - Base free, pulled to the nominal path (`w_base`) and boxed within 0.5 m / 0.8 rad of it.
   - References that are not retimed are also **boxed by the speed limits** around the previous frame (0.95 × limit; base per body axis at the current heading). A reference faster than Reachy becomes a residual, never a joint jump.
   - Arm redundancy (7 joints for a 6-D pose) is resolved by the IK alone: continuity with the
     previous frame and a weak pull toward the start posture. Nothing the generator knows about
     joints (keyframe postures, witness paths) reaches the IK.
3. **Neck** (`tracking.neck`): batched, bounded Gauss-Newton on the three neck joints for the head rotation in the base frame (from the solved base yaw), within limits − 0.03 rad. Rate-limited to 0.95 × 30°/s. Without a head reference the neck holds its start angles.
4. **Fingers**: the reference opening as finger angle, rate-limited to 0.95 × 3 rad/s.
5. **Timing.**
   - Synthetic references are generated within the speed limits on the 50 Hz output clock and are **not retimed**.
   - Source paths are slowed down where needed (`retarget.timing.clock`). They are re-timed with a smaller speed fraction until body-axis base speeds hold, then resampled to 50 Hz.
   - Residuals are refined on the output clock (`wbik.refine`). Refined rows that would break a speed limit are put back. Shared `refine` keeps a base axis whose neighbours are already apart, but still moves the other axis, which can exceed the body-axis base limit.
6. **Episode.** `reference.tcp` / `reference.head` / `reference.base` hold the reference unmodified. The head reference's position is the achieved head position, since only the rotation is tracked.

`objects = {}`. `validation` holds:
- `grasp_object = -1`;
- the tier-K frames;
- `head_rot_residual`;
- `witness_q` for witness cells.

`extra["tracking"]` holds:
- the scenario (cell, neck mode, length, speed, start, regions, grippers, base start) or the source description;
- the parameters (speeds, primitives, base segments, retry attempt);
- diagnostics;
- the `head_tip` offset.

**Tier K** (`tracking.validate.check`) is `validate.kinematic.check` with these tolerances:
- TCP 5 mm / 0.05 rad on every frame and both sides;
- joint limits (URDF);
- speed limits, fingers and neck included;
- self-clearance ≥ 10 mm;
- navigation destination within 3 cm.

On top of that, a head rotation residual ≤ 0.05 rad is checked. Base deviations from the nominal path are reported (`max_base_deviation_m`, `max_base_yaw_deviation_rad`), not gated.

## Synthetic scenarios (`tracking.synthetic`)

Scenario `<namespace>/<split>/<index>` is one independent lineage seed (SHA-256 of the id, as in `rl_tracking/data.py`). The cell is stratified: every block of 15 consecutive indices holds each cell once. All other axes are sampled independently.

| axis | values (weights) |
| --- | --- |
| cell, stationary base | `hold`, `left`, `right`, `bimanual_independent`, `bimanual_symmetric` (mirror), `bimanual_rigid` (virtual box), `handover`, `neck_only`, `witness_arms` |
| cell, moving base | `navigation`, `mobile_carry`, `mobile_reach` (drive, stop, manipulate, repeat), `mobile_world_hold` (hand fixed in the world while the base repositions), `mobile_free` (manipulation relative to a driving base), `witness_whole_body` |
| neck | `off` .25 (no head reference), `base_hold` .1, `world_hold` .1, `look_at_point` .1 (1–3 fixed points), `look_at_hand` .15 (alternating for two hands), `scan` .1, `look_ahead` .1 (travel direction; mobile only), `random` .1 |
| length | short 2–6 s .4, medium 6–20 s .4, long 20–60 s .2 (primitive chains) |
| speed | slow / normal / fast: hand 0.06–0.12 / 0.12–0.25 / 0.25–0.4 m/s and 0.25–0.5 / 0.5–0.9 / 0.9–1.3 rad/s; base cruise 0.2–0.3 / 0.3–0.42 / 0.42–0.55 m/s (at or above the ~0.2 m/s start threshold measured on the real base); yaw 0.3–1.4 rad/s |
| start posture | reachy-control experiment start .3, `ready` .2, `rest` .15, random clear posture .35 |
| region per hand | any, low, mid, high, front, side, cross-midline |
| gripper per hand | open, closed, toggle at keyframe arrivals (grasp/release-like), cycle, partial levels. Overrides: closed for carries, close-then-open for handover. Ramped at 0.9 × finger speed |
| base start | world x, y ∈ [−2, 2] m, yaw uniform |

**Hand motions.**
- **Keyframes** come from a joint-space random walk. Each keyframe is reachable and self-clear (≥ 25 mm) on its own, within limits − 0.12 rad.
- **Primitives** run between keyframes: line, circle, raster, press, hinge (door/lid arc), twist (exactly a wrist-yaw turn, kept within its range), pour/tilt, oscillation, smooth random wander. All are min-jerk, at the sampled speeds.
- **Base segments:** drive, strafe, arc, turn in place, splines with tangent or fixed heading (turning to the spline's initial tangent first). Each segment is stretched until body-frame speeds stay below 0.85 × the limits.
- **Witness cells** follow a PCHIP joint path through random-walk knots (no overshoot) with these bounds:
  - limits − 0.12 rad;
  - 0.85 × speed limits;
  - self-clearance ≥ 15 mm every 0.1 s (knots redrawn otherwise).
  Their FK is the reference; the joint path is stored (`witness_q`) for audit only.
- **Head references** are the rotation of a neck path within limits − 0.07 rad and 0.8 × neck speed, relative to the nominal base. They are reachable whenever the base follows its path.

**Retries** (`tracking.build`):
1. A reference that fails the cheap pre-screen is regenerated with motions × 0.75, at most 4 times, without IK. The pre-screen checks:
   - grasp-center height inside 0.46–1.87 m;
   - TCP within 0.665 m of the shoulder ball.
2. A tier-K failure is regenerated at most twice:
   - **slower** (durations × 1.5), when the IK was at its speed box where tracking was first lost and has not been slowed yet;
   - **smaller** (× 0.75) otherwise.

Every IK attempt is written. Earlier attempts get `episode_id` suffix `-try<k>` and `variant_of` = the last attempt's uid, and share its lineage seed.

## Source paths (`tracking.source`)

For every `SourceEpisode` a family adapter yields:
- **Targets.** TCP target = source grasp-center pose × the fixed Reachy grasp-center offset.
- **Sides.** By `side_hint`, or by lateral position for bimanual sources. A single-arm source is tracked with **both arms**: the better-scoring arm first, the other as `variant_of`.
- **Base.**
  - Fixed-base sources get one base pose: a 16-heading × 4-distance grid around the target centroid plus the source base hint, scored by a reachability proxy, the best 4 by keyframe IK.
  - Mobile sources follow the source base path composed with a constant body-frame offset chosen the same way.
- **Idle hand.** Holds the carry pose relative to the base.
- **Neck.** Looks at the active hand(s), or along the travel direction without one.
- **Gripper.** The source opening.
- **Timing.** Retimed.
- **Ignored.** Objects, grasp labels and scene geometry.
- **Lineage.** `lineage.seed` = the source's seed (plus `tracking_of` = source uid). These episodes are not independent of the manipulation retargets of the same demonstrations.

## Building

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m reachy_retarget.tracking.build synthetic \
    --namespace tracking-v1 --split train --count 20000 --jobs 24 --out runs/tracking/v1   # [--shard k/n]
.venv/bin/python -m reachy_retarget.tracking.build source --family robomimic \
    --path data/raw/hf__robomimic__robomimic_datasets/v1.5/lift/ph/low_dim_v15.hdf5 --demos 0-199 \
    --out runs/tracking/source-robomimic
.venv/bin/python -m reachy_retarget.tracking.build index runs/tracking/v1   # -> episodes/index.parquet
.venv/bin/python -m reachy_retarget.tracking.report runs/tracking/v1 --md summary.md
.venv/bin/python -m reachy_retarget.tracking.viewer --run runs/tracking/v1
```

Layout:
- `DIR/episodes/<dataset>/<episode_id>.h5`, for every attempt, K failures included;
- `DIR/records/tracking/*.jsonl`, one record per scenario or source episode, with the scenario axes, attempts, K reasons and metrics, `index_rows`, seconds and memory;
- `DIR/episodes/index.parquet`, built from the records.

`reachy_retarget.report` reads the same records.

## Pilot results (2026-10-08)

**Synthetic, namespace `dev`, split `pilot`, indices 0–299 (20 per cell).**
- 297 / 300 scenarios pass tier K (99.0 %).
  - 271 scenarios needed one IK attempt, 15 two, 14 three.
- Every cell passes ≥ 95 %. All 15 cells are at 100 % except `bimanual_independent`, `bimanual_rigid` and `bimanual_symmetric` (19/20 each).
- Volume: 1.43 h of motion, 343 files, 1.69 MB per file, 0.41 CPU-h (about 2 min on 24 workers).
- An earlier variant regularized the arm redundancy toward the generator's joint postures. It passed 296 / 300, but its joint paths differed by more than 0.2 rad in 142 of 296 episodes. It was removed: redundancy is the IK's own.
- Remaining failures: TCP rotation or position residuals (2 + 2), e.g. independent bimanual chains whose hands meet.

Passing episodes, base-relative hand path per arm:

| quantity | median | p90 | max |
| --- | --- | --- | --- |
| path per arm (m) | 0.5 | 2.0 | 9.2 |
| hand extent (m) | 0.19 | 0.5 | 1.0 |
| base travel (m) | 0.15 | 2.6 | 12.4 |
| base turn (rad) | — | 2.1 | 7.2 |
| neck range (rad) | 0.51 | — | 1.96 |
| gripper open/close changes | 1 | 11 | 49 |
| duration (s) | 13 | 35 | 65 |

**Source paths.** 33 / 46 pass:

| source | passing |
| --- | --- |
| robomimic Lift demos 0–9, both arms | 20 / 20 |
| robomimic Square demos 0–4 | 8 / 10 |
| robomimic Transport demos 0–3 | 0 / 4 |
| MolmoBot RB-Y1 door (mobile) and Franka pick fixtures | 3 / 3 |
| LIBERO fixture | 0 / 2 |
| DexMimicGen fixture | 0 / 1 |
| MobileManiBench G1 fixtures | 0 / 4 |

The failures are source poses Reachy cannot reach with strict tracking:
- Panda/G1 wrist orientations beyond Reachy's ±30° wrist;
- G1 start poses;
- Transport's two-robot handover.

The extracted RoboCasa episodes are not on this machine, so RoboCasa was not run.

## Known limits

- **Independent bimanual chains** plan the second arm's keyframes against the first arm's start posture, so the hands can meet (repulsion, then residuals).
- **The neck** is solved after the arms. Head–arm contacts are caught by tier K, not avoided.
- **The base box** (0.5 m / 0.8 rad around the nominal path) lets the base help an arm. A large deviation shifts the head reference, which assumes the nominal base yaw (0.04 rad of neck margin covers small deviations).
- **Source paths** ignore the source scene: no footprint obstacles, no objects.
