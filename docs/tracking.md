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
- Episodes carry `family = "tracking"` and their own datasets: `tracking/synthetic-v2` and `tracking/source/<source dataset>`.
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

The tripod bars are plain URDF boxes (there are no tripod meshes). The two front bars end 17 mm below
the bottom of `torso_visual.dae` and the back bar barely reaches it, so the torso looked detached.
For display only, the viewer extends each bar along its own axis to 1 cm above the torso bottom
(`visual.tripod_extensions`). The URDF, the kinematics and the collision model are unchanged.

Link poses come from this repo's URDF kinematics. On 484 frames of a pilot episode they match pinocchio (reachy-control's library) to 1.2e-7 m. Every loaded episode is asserted to replay its stored TCPs (1e-5 m). Install with `uv pip install -e '.[viz]'`.

`python -m reachy_retarget.tracking.mjviewer --run DIR` replays tracking records and build records (`records/<family>/*.jsonl`) on the MuJoCo model with mjviser (the viser-based MuJoCo viewer of reachy-control's mjlab environment):
- the robot is the tier-P model (`robot.mjcf.reachy_mjcf`) plus display-only visual geoms: the URDF visual meshes, one mesh per material colour, in group 2 without contact, and the tripod extensions;
- episodes with a `/scene` (manipulation builds) add their scene components from the asset library (`scene_assets.scene_mjcf`) as mocap bodies, Reachy links excluded since the robot is drawn from q; "Poses" switches between `/scene/poses` with `q` (kinematic) and `/scene/physics_poses` with the robot from `/physics/qpos` (tier-P rollout); the model is rebuilt only when the scene components change; textured primitive visual geoms become meshes with MuJoCo-style texture coordinates (cube textures as a six-face atlas), since mjviser textures only meshes with texture coordinates;
- the collision geoms (group 3) and frame sites (group 4) start hidden; mjviser's Groups tab shows them, its Visualization tab adds frames and contacts;
- the world frame is fixed and the camera frames the hands, the head and the moving objects of each episode; "Follow base" turns on mjviser's camera tracking, which shifts the whole world (scene, floor grid, overlays) by minus the base position every frame, so the grid then slides as the base drives;
- each frame writes q into qpos (mimic joints included) and runs `mj_kinematics`, nothing is simulated; every kinematic replay is asserted to reproduce the stored TCPs on the MuJoCo model (1e-5 m);
- filters: task or cell, tier-K result, final or all attempts.

Both viewers accept a dataset folder copied from the PVC as `--run`: records written on a pod keep the pod's `/tmp/.../episodes/...` paths, which resolve below `DIR/episodes/`.

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
- `witness_q` for `joint_move` scenarios (the executed joint path).

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

**One scenario is one stage.** Every body part does at most one action, going once from its start to
one goal: no return, no repetition, no cycle, no sequence of steps. Several parts may act at the same
time within the stage, e.g. the base drives while an arm reaches and the head turns. Parts that act
together start within the first 30 % of the stage. The stage has a 0.2–0.5 s still start and a
0.5–1 s still end. A test checks the rule on the references: for each arm (base-relative), the
base, the neck and each gripper, at most one movement.

The generator is version 2 (generator `tracking.synthetic/v2`, dataset `tracking/synthetic-v2`).
Version 1, a different cell set, is retired: reachy-policy refuses `tracking/synthetic-v1` episodes
in its splits and caches, and the code that made them is in the git history.

Scenario `<namespace>/<split>/<index>` is one independent lineage seed (SHA-256 of the id, as in
`rl_tracking/data.py`), or, with `--group-size k`, a member of a sibling group (below). The cell is
stratified: every block of 10 consecutive indices (groups) fills the 10 slots of `CELLS_V2` once.
All other axes are sampled independently.

| cell | slots of 10 | the action | everything else |
|---|---|---|---|
| `reach_target` | 2 | each active hand (one, or both with p = 1/3) reaches a TCP position drawn uniformly in x 0.15–0.60, outward y 0.00–0.55, z 0.55–1.30 m (base frame), through a self-clear posture within 4 cm of the draw; when 16 draws find none (about 1.5 % of hands), a joint-space step goal and `drawn_base` null | base on its path for the tracker, neck off, grippers kept |
| `gaze_target` | 2 | one head turn to a gaze direction (head +X yaw, pitch in base_link) drawn uniformly in ±0.9 × −0.4–0.35 rad, solved to 0.02 rad; the stage lasts until the turn ends | arms still, base on its path, grippers kept |
| `joint_move` | 1 | one or two named arm joints (or one neck joint, p = 1/4) move by 0.1–0.6 rad; the episode executes the joint path itself (`extra["exact_witness"]`: no IK), so the named joints are exact and the others hold | base on its path, grippers kept |
| `drive_target` | 1 | the base goes straight (holonomic) to a body-frame goal dx −0.5–2.5 m, dy ±1 m, dyaw ±1.6 rad | arms hold base-relative, neck off, grippers kept |
| `left`, `right` | 1 each | one hand motion (below) | base still |
| `bimanual_independent` | 1 | one hand motion per hand, staggered start | base still |
| `navigation` | 1 | one base motion (below) | arms hold base-relative, grippers kept |

The other axes; the target cells override the neck and grippers as listed above:

| axis | values (weights) |
| --- | --- |
| hand motion | `reach` .3 (to a reachable goal posture), `curved_reach` .15 (to such a goal along a bowed path), `line` .15 (push/pull/lift/lower/slide, orientation kept), `approach` .1 (along the tool axis, forward or back), `hinge` .1 (door/lid arc about an external axis), `twist` .1 (about the tool axis: exactly a wrist-yaw turn, within its range), `tilt` .1 (about a horizontal axis through the grasp center) |
| base motion | `drive` .3 (holonomic straight, optional heading change), `strafe` .15, `turn` .15 (in place), `arc` .2, `curve` .2 (one cubic Bezier leaving along the current heading, heading on the tangent; a straight drive below 0.3 m) |
| neck | `off` .25 (no head reference), `hold` .1, `world_hold` .1, `look_at_point` .15 (one gaze shift to a fixed point), `look_at_hand` .15 (follows the acting hand), `look_ahead` .1 (travel direction; mobile only), `shift` .15 (one turn to a new orientation). A gaze-following path that would stop and start again becomes one turn toward where the gaze ends |
| extent | small .35 / medium .4 / large .25: goal step 0.45 / 0.75 / 1.0 × the joint step; translation 4–9 / 9–17 / 17–28 cm; rotation 0.15–0.4 / 0.4–0.75 / 0.75–1.1 rad; base travel 0.2–0.6 / 0.6–1.5 / 1.5–3 m; base turn 0.3–0.8 / 0.8–1.6 / 1.6–3.1 rad |
| speed | slow / normal / fast: hand 0.06–0.12 / 0.12–0.25 / 0.25–0.4 m/s and 0.25–0.5 / 0.5–0.9 / 0.9–1.3 rad/s; base cruise 0.2–0.3 / 0.3–0.42 / 0.42–0.55 m/s (at or above the ~0.2 m/s start threshold measured on the real base); yaw 0.3–1.4 rad/s; `joint_move` joint speed 0.25–0.8 × the limit |
| start posture | reachy-control experiment start .3, `ready` .2, `rest` .15, random clear posture .35 |
| region per hand | goal region: any, low, mid, high, front, side, cross-midline |
| gripper per hand | `keep_open` .25, `keep_closed` .15, `close` .25, `open` .2, `partial` .15: at most one transition, when the hand's action ends; cells that keep their grippers hold the drawn state open or closed |
| base start | world x, y ∈ [−2, 2] m, yaw uniform |

With `--group-size k` (4 for the policy data), indices `k g … k g + k − 1` are **sibling group** `g`:
the members share the group's axes, start posture, speeds, base pose, still start and gripper
levels (the group's seed), then each draws its own goals (its own seed). The same start state thus
leads to different motions, which the policy can only tell apart from the instruction. Members
share `lineage.seed` (the group id; `lineage.group`, `lineage.member`) and are not independent
demonstrations. Parameters record the drawn and reached targets (`hand.<side>.target_base`,
`hand.<side>.drawn_base`, `gaze_target.drawn_rad`, `joint_move.joints/goal_rad/delta_rad`,
`base.goal_body`). `tests/test_tracking.py` regenerates built files (scenario, references, params,
lineage) to check that later edits keep the generator's output.

**Feasibility.**
- **Reach goals** of the hand motions are postures one joint-space step from the start: each is
  reachable and self-clear (≥ 25 mm) on its own, within limits − 0.12 rad.
- **Base motions** are stretched until body-frame speeds stay below 0.85 × the limits.
- **Joint moves** (`joint_move`) are one min-jerk move of the named joints with self-clearance ≥ 15 mm
  at 21 points along the path (deltas redrawn otherwise, shrinking every 10 draws, up to 40 draws;
  else the clearest). The episode is this joint path; it is stored as `witness_q`.
- **Head references** are the rotation of a neck path within limits − 0.07 rad and 0.8 × neck speed,
  relative to the nominal base. They are reachable whenever the base follows its path.

**Retries** (`tracking.build`):
1. A reference that fails the cheap pre-screen is regenerated with motions × 0.75, at most 4 times,
   without IK. The pre-screen checks:
   - grasp-center height inside 0.46–1.87 m;
   - TCP within 0.665 m of the shoulder ball.
2. A tier-K failure is regenerated at most twice:
   - **slower** (durations × 1.5), when the IK was at its speed box where tracking was first lost and
     has not been slowed yet;
   - **smaller** (× 0.75) otherwise.

Every IK attempt is written. Earlier attempts get `episode_id` suffix `-try<k>` and `variant_of` =
the last attempt's uid, and share its lineage seed.

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
- **Stages** (`tracking.stages`). A demonstration chains several actions (approach, close, carry,
  open, retreat), so it is cut into stages under the same rule as the synthetic scenarios: in each
  stage every hand (relative to the base), the base and each gripper make at most one movement
  toward one goal, and the head makes one movement.
  - A part moves while its smoothed speed is above a threshold: hand 2 cm/s or 0.15 rad/s, base 2 cm/s
    or 0.05 rad/s, gripper 0.2 opening/s.
  - Pauses under 0.3 s do not split a movement. Movements under 0.15 s and 1 cm, and gripper changes
    under 10 % of the stroke, are noise.
  - A movement that slows below 30 % of its peak and turns by more than 90° (a reach and return) is two
    movements.
  - Scanning movements by start time, a new stage begins when a part starts its second movement in the
    current stage. The cut lies within 0.5 s before that movement, at the latest instant where the
    other parts move least.
  - A movement still running at a cut continues in the new stage and counts there, unless it ends
    within 0.3 s. Demonstrations overlap their actions (the hand still settles as the gripper starts to
    close), and such a settling tail is not an action.
  - Each stage is one episode, `<episode>[-<arm>]-stageNN`. `extra["stage"]` holds the index, count,
    source rows, source times and the movements found. All stages share the source's lineage seed.
  - The head follows the active hand(s) within the stage when that is one movement, else it makes one
    turn toward where the gaze ends.
- **Timing.** Retimed.
- **Ignored.** Objects, grasp labels and scene geometry.
- **Lineage.** `lineage.seed` = the source's seed (plus `tracking_of` = source uid). These episodes are not independent of the manipulation retargets of the same demonstrations.

## Building

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m reachy_retarget.tracking.build synthetic --group-size 4 \
    --namespace tracking-v2 --split train --count 72000 --jobs 16 --out runs/tracking/v2   # [--shard k/n]
.venv/bin/python -m reachy_retarget.tracking.build source --family robomimic \
    --path data/raw/robomimic/v1.5/lift/ph/low_dim_v15.hdf5 --demos 0-199 \
    --out runs/tracking/source-robomimic
.venv/bin/python -m reachy_retarget.tracking.build index runs/tracking/v2   # -> episodes/index.parquet
.venv/bin/python -m reachy_retarget.tracking.report runs/tracking/v2 --md summary.md
.venv/bin/python -m reachy_retarget.tracking.viewer --run runs/tracking/v2
```

Layout:
- `DIR/episodes/<dataset>/<episode_id>.h5`, for every attempt, K failures included;
- `DIR/records/tracking/*.jsonl`, one record per scenario or source episode, with the scenario axes, attempts, K reasons and metrics, `index_rows`, seconds and memory;
- `DIR/episodes/index.parquet`, built from the records.

`reachy_retarget.report` reads the same records.

## Source-path pilot (2026-10-08)

Source paths cut into stages: 98 / 165 stage episodes pass tier K.

| source | stages per demo | stage episodes passing |
| --- | --- | --- |
| robomimic Lift demos 0–9, both arms | 2.3 (approach and open; close and lift) | 46 / 46 |
| robomimic Square demos 0–4, both arms | 4 | 37 / 40 |
| robomimic Transport demos 0–3 (bimanual) | 13 | 1 / 53 |
| MolmoBot RB-Y1 door (mobile) and Franka pick fixtures | 2 | 4 / 6 |
| MobileManiBench G1 fixtures | 2–3 | 4 / 10 |
| LIBERO fixture | 1 | 0 / 2 |
| DexMimicGen fixture | 2 | 0 / 2 |

On the source clock, in each of the 86 robomimic Lift and Square stage references, every hand and
gripper moves at most once. Checking the 50 Hz retimed episodes instead finds a few extra movements:
slow, near-threshold motion resampled from 10–20 Hz splits into pieces there.

The failures are source poses Reachy cannot reach with strict tracking:
- Panda/G1 wrist orientations beyond Reachy's ±30° wrist;
- G1 start poses;
- Transport's two-robot handover.

The extracted RoboCasa episodes are not on this machine, so RoboCasa was not run.

## Known limits

- **Independent bimanual actions** plan the second hand's goal against the first arm's start posture, so the hands can meet (repulsion, then residuals).
- **The neck** is solved after the arms. Head–arm contacts are caught by tier K, not avoided.
- **The free base** (boxed 0.5 m / 0.8 rad around the nominal path) can help an arm, which adds base motion the stage does not plan. A large deviation shifts the head reference, which assumes the nominal base yaw (0.04 rad of neck margin covers small deviations). `base_free=False` removes both effects.
- **Source stages** start wherever the demonstration's next action starts. A stage may begin with a part still moving (its movement from the previous stage continues), and stages can be short (0.1–0.5 s, e.g. a gripper closing).
- **Source paths** ignore the source scene: no footprint obstacles, no objects.
