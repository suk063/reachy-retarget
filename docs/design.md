# Reachy retargeting v2: design

## Goal

Acquire as many and as diverse object-interaction demonstrations as possible and turn
them into Reachy 2 state-only training data. Diversity axes are the body parts used
(one arm, both arms, head, mobile base), task type (pick-place, articulated objects,
insertion, tool use, bimanual coordination) and regime (tabletop, mobile manipulation,
navigation). The policy state/action format is not fixed yet, so every episode stores
canonical robot and object state plus precomputed views for all control modes. No
images are stored.

## Pipeline

```
source files ──adapter──▶ SourceEpisode ──retarget──▶ Reachy trajectory ──validate K/P──▶ ReachyEpisode (HDF5)
```

* `reachy_retarget.acquire`: pinned catalog and one explicit `fetch` for every acquisition kind
  (whole files, image-stripped HDF5, streamed tars, byte-range members and packages) with a
  50 GB reserve, digest verification, no image bytes and a per-file ledger; catalog
  generators in `acquire/generators/` (see docs/sources.md, Acquisition).
* `reachy_retarget.sources`: one adapter per source family. Adapters only read source
  files and produce `SourceEpisode` objects in the source's own world frame and clock.
* `reachy_retarget.retarget`: embodiment-independent mapping onto Reachy.
* `reachy_retarget.validate`: tier K (kinematic) for every episode, tier P (MuJoCo,
  free objects) when the source ships a MuJoCo scene.
* `reachy_retarget.schema`: data model, HDF5 I/O, control-mode registry.
* `cluster/`: job pool over the persistent worker pods (two jobs per pod); `cluster/manifests.py`
  writes size-batched fetch manifests.

Build jobs (`reachy_retarget.build`, one source file shard per job) must stay well below half
of a 4 GiB pod. Rules that keep them there: the shard is passed to the adapter as `select`
(adapters registered with `select=True` never read, compile or replay other shards'
episodes; the robosuite family does); a source reader keeps at most one compiled source
model and drops it before yielding unless the next episode records the same MJCF; the source
kinematic model is compiled without file textures (visual only); each episode is released and
the allocator trimmed before the next is read; MuJoCo's compiler asset cache is off
(`--mujoco-cache-mb`); per-frame checks over a whole episode are chunked (self-collision pair
distances take ~0.8 MB per frame); readers whose files are listed inline load the catalog
without its gzip TSV member tables (`load_catalog(tables=False)`, ~0.6 GB otherwise); the
BEHAVIOR reader, whose files are table rows, loads only `catalog/behavior.yaml` with its table
(`load_family_catalog("behavior")`, ~60 MB: a `--shard 0/20` build of one local task peaks at
444 MB RSS instead of 709 MB, episode content identical). Every
build record carries `rss_mb` and `max_rss_mb`.

## Reachy model

Assets are copied from `reachy-agent@d6d5e9fd` (`robot/assets`, URDF SHA-256
`63a1ecab…d448`). The tripod stays at 0 (its calibrated extension is in the URDF), and
antennas are not controlled.

Canonical joint vector `q` (22 values, this order everywhere):

| index | names |
| --- | --- |
| 0–2 | `base_x`, `base_y` (m, world), `base_yaw` (rad, world) |
| 3–9 | `l_shoulder_pitch`, `l_shoulder_roll`, `l_elbow_yaw`, `l_elbow_pitch`, `l_wrist_roll`, `l_wrist_pitch`, `l_wrist_yaw` |
| 10–16 | the same for `r_` |
| 17–19 | `neck_roll`, `neck_pitch`, `neck_yaw` |
| 20–21 | `l_hand_finger`, `r_hand_finger` (rad) |

Limits: URDF joint limits (wrist roll/pitch ±30°, neck roll/pitch ±30°, yaw ±60°), arm
joint speed 1 rad/s, neck 30°/s, base 0.61 m/s per axis and 114°/s (whole-body
controller limits). The more conservative task-space executor limits of reachy-agent
(0.22 m/s, 0.6 rad/s base) are reported as a flag, not enforced.

Frames: `world` (source world), `base_link` (Reachy base on the floor, +x forward, +y
left, +z up), TCP = `{l,r}_arm_tip`, head = `head`. Poses are stored as
`xyz + quaternion wxyz` unless a view says otherwise.

## SourceEpisode

Defined in `reachy_retarget/schema/source.py`. Everything is in the source world frame
(metres, z up) and the source clock.

* World frame: z up with the floor the robot stands on at z = 0 (adapters translate if needed).
* `effectors`: source end effectors, keyed by a source-specific id. Each has a world
  pose of the **grasp center** (the point between the pads, not the flange), with
  approach axis = +z and closing axis = +y (adapters convert their gripper convention to
  this), an opening in [0, 1] (1 = fully open), optional width in metres, an optional
  `side_hint` (`left`/`right`) and an optional recorded gripper `command` in [0, 1]
  (1 = commanded closed).
* `base`: source mobile base SE(2) path or `None` for fixed-base sources.
* `objects`: pose tracks with validity masks, a role (`manipulated`, `support`,
  `receptacle`, `fixture`) and simple geometry.
* `articulations`: articulated scene joints (drawers, doors) with joint names.
* `scene`: optional MuJoCo scene for tier P, with the bodies the source parks out of use
  (`inactive_bodies`) and `reference` values measured on the source's own states.
* provenance, license, lineage (`seed`, `variant_of`), task, instruction, source success.

## Retargeting method

1. **Tool alignment.** Adapters already express effector poses at the grasp center with
   a shared axis convention, so the Reachy TCP target is the source pose times a fixed
   Reachy grasp-center offset from `{l,r}_arm_tip`. Hand–object relative poses are
   preserved by construction.
   * *Grasp labels* (`targets.source_closed`): a source hand holds an object while it is
     closed and within 4 cm of the object's box or cylinder. With a recorded gripper command
     (robomimic, MimicGen, LIBERO) a hand is closed from the frame its fingers stop closing
     after a close command until the open command (the command leads the stall by 6–8 frames
     while the hand still descends; the release label ends 3–6 frames earlier than inferred
     and no longer splits on finger jitter). Otherwise "closed" is inferred from the opening: below
     0.5, or stalled (|rate| ≤ 0.25/s) more than 0.1 below the episode's open level, and not
     opening. A gripper that closes on an object stalls at the object's width (the robomimic
     Panda holds the Lift cube at opening 0.52, the Can at 0.62), so a fixed threshold missed
     every Lift and Can grasp. Three refinements (`targets.grasp_labels`, from the ManiSkill
     loop): a closed hand whose pad separation is below half the object's extent along the
     closing axis holds nothing (fingers closed on nothing push with their outsides or tips:
     ManiSkill PullCube, StackPyramid and TwoRobotPickCube push 40 mm cubes at width 0–15 mm and
     were labelled as grasps; an `aabb` is an envelope, e.g. the Square nut ring, so only its
     thinnest extent is used, and a union of `boxes` uses its thinnest part); gaps ≤ 0.25 s between two runs on the same object are bridged
     while the hand stays within contact distance (RL grippers jitter around the stall: the
     LiftPegUpright peg is lifted while the label flickered off for 0.5 s); runs shorter than
     0.2 s are dropped (a closed hand brushing an object). Source widths are pad-face gaps
     (robosuite: travel plus the gap at the closed limit; the Rethink gripper of MimicGen
     PickPlace read 14 mm narrow, so its cereal box and bread were "empty" hands), and robosuite
     objects with several collision geoms are unions of per-geom `boxes` (the mug is held by its
     9 mm handle, ThreePieceAssembly pieces are 34–40 mm voxel arms of 102–160 mm envelopes):
     before, 38 of the 105 closed-gripper runs of the MimicGen dev demos had no grasp label (no
     squeeze, no grasp-phase checks); now all 105 have one.
   * *Closed-hand pushes* (`targets.push_labels`, derived label recorded in
     `extra["retarget"]["push"]`, never a grasp label): frames where a closed but empty hand is
     within contact distance of an object that moves faster than 2 cm/s together with the hand
     (|v_obj − v_hand| ≤ 0.5 |v_obj|). They get the object-centric carrying rule below, so the
     hand keeps the contact geometry of the push onset instead of sliding off the object.
   * *Object-centric grasp re-selection* (`targets`, chosen by placement scoring, recorded
     in `extra["grasp_offsets"]`). The task is the object path, not the Panda hand pose, so
     each arm uses the source grasp frame composed with a constant offset
     `Rz(pi·flip + theta) · Ry(tilt)` about the grasp center: `flip` (half turn, always a
     parallel-jaw symmetry); `tilt` ∈ {0, ±15, ±30, ±45}° about the closing axis (pad planes
     unchanged, the pads press the same faces); `theta` = the rotation that lays the closing
     axis onto the nearest face normal of the grasped object's box (≤ 20°; the Panda's narrow
     pads hold the Lift cube 9–19° off its faces, Reachy's 30 × 39 mm flat pads then touch two
     edges and the cube turns between them), plus quarter turns when the object is symmetric
     under them (derived from the box/aabb geometry: approach within 15° of a box axis and
     equal cross-section half extents ±10 %; recorded with the reason). A cylinder grasped
     along its axis (the robomimic Can from the top) has no faces to align and is symmetric
     under any turn: `theta` ∈ {0, 45, 90, 135}° (flips add the rest). Offsets whose fingers
     would sink into scene boxes along the path more than 3 mm beyond the best offset are
     discarded before any IK (quarter turns are mostly rejected here: the source approach is
     off-centre along the other axis and a finger would land on the cube). Objects a closed hand
     touches on purpose (within contact distance, `targets.touch_labels`) are excluded from this
     screen like held ones: the StackPyramid fingers push cubeA, and counting that contact as
     depth rejected every tilt, leaving the cubeC carry outside Reachy's wrist range.
     *Per-segment offsets* (`placement.segment_offsets`, `targets.offset_track`, recorded in
     `extra["grasp_offset_segments"]`): a hand with several grasp segments (robomimic Transport:
     one arm lifts the lid, then takes the payload) re-selects the offset of each segment among
     that segment's own candidates, scored on the keyframes of its window at the chosen base
     placement; the episode-wide offset is kept unless a candidate scores lower. Each offset holds
     over its segment window and turns geodesically to the next one in the free time between
     windows.
   * *Object-centric carrying* (`targets.object_centric`): when the source object moves in its
     gripper by more than half the tier-K grasp tolerance during a grasp (robomimic Square: the
     nut slides 5–44 mm while pushed onto the peg), the hand follows the object pose times the
     grasp at the segment's third frame, and the correction to the source hand path decays to
     zero over the approach/retreat windows. Grasps the source holds rigidly keep its hand path.
   * *Free orientation away from grasps* (`targets.orientation_weight`): the TCP orientation is
     strict inside grasp windows (segment − 0.5 s … + 0.3 s) and its weight decays to 0.05 over
     1 s outside them. Outside grasp segments the weight is further capped by a proximity ramp
     (`contact_strict_distance` 5 cm → `contact_free_distance` 12 cm, grasp center to the nearest
     manipulated object): the orientation only matters where the fingers can touch something.
     Hands that never grasp (pushing, poking with the fingers) were strict on every frame and
     failed tier K at the first frames, where an RL Panda starts above the table with a wrist
     pose Reachy's ±30° wrist cannot copy; an approach that starts far away now starts free. A first IK pass with these weights gives the orientation Reachy prefers
     in free space; the reference rotation is that one blended into the source (offset)
     rotation by the weight, and a second pass tracks this reference strictly, so tier K still
     checks every frame against a stored reference (`reference.tcp`).
2. **Arm assignment.** Single-arm sources are assigned to the arm with the better
   reachability score over the trajectory. Bimanual sources keep their side hints. A
   mirrored or alternate-arm variant may be written but carries `variant_of`.
3. **Base placement.** Fixed-base sources: search a single base SE(2) pose (grid, then
   local refinement) that makes every TCP target reachable with joint-limit and
   manipulability margin and without base/scene collision. Mobile sources: map the
   source base path to Reachy's footprint, then refine with the same score. The base is
   allowed to move during whole-body IK only when the source base moved, or when a
   fixed placement cannot cover the whole trajectory.
   * *Body footprint by height* (`footprint`): Reachy is a stack of discs (base 0.25 m up to
     0.30 m, tripod column 0.14 m up to 0.95 m, torso/head 0.19 m above; from the MJCF
     collision geometry), and a scene box only blocks the discs at its height, so the base may
     stand under a table top whose legs are visual-only (robosuite). Box and body-frame aabb
     geometry (with centre offset) are both used; before, adapter aabbs were ignored and the
     base was placed inside the table. A union of `boxes` (ManiSkill PullCubeTool L tool,
     PlugCharger charger and receptacle, PegInsertionSide hole box) is seen by
     `box_geometry` as its enclosing box (an envelope, like an `aabb`: footprint, grasp
     symmetry and contact width), while hand–object distances and the finger screen use the
     exact parts (`footprint.box_parts`): the envelope of an L covers the empty corner beside
     the handle, where the fingers of a handle grasp are. Teaching `box_geometry` the kind
     keeps one geometry record per object (adapters need not duplicate it as an `aabb`).
   * robosuite world-child bodies without a joint of their own are fixtures through their static
     geoms, also when articulated parts hang below them (MimicGen Kitchen/HammerCleanup tables
     carry stove buttons and a drawer; the table was missing and the base was placed 6–8 cm into
     it: 6 of 10 dev episodes failed `robot_environment_penetration`, 1 after).
   * Placement candidates include the target centroid itself; the score adds the arm's
     self-clearance below the IK margin at the keyframes (the scoring IK has no repulsion
     term), with the inactive arm in its rest posture. Moving-object samples while a hand
     holds the object are not obstacles. Grasp offsets are searched per candidate (cold IK once
     per placement, offsets warm started, the first offset within 0.6 of the tolerance wins,
     else the two best of a 3-keyframe screen are scored on all keyframes).
   * *Base assistance* (`pipeline`): when the fixed-base IK leaves frames outside the tier-K
     tolerances, the IK is re-solved with the base free in a box (±0.6 m, ±1 rad) around the
     placement, a deviation penalty, and the footprint clearance ≥ 5 cm as a linearized
     constraint (re-solved with stiff rows when a step would cross it); the result is kept
     when it has fewer failing frames and a clear footprint, and `body_parts` gains `base`.
4. **Whole-body trajectory IK.** For each frame, warm-started damped least squares on
   `q[0:20]` with bounds: strong TCP position tracking, weighted orientation, posture
   regularization, base motion penalty, joint limits with a 0.045 rad margin (physics
   tracking dips up to 0.015 rad below the reference at the ±30° wrist limits; the tier-P gate
   needs 0.025) and a self-clearance penalty from the collision spheres. A smoothing pass
   follows. The bounded least squares is an exact active-set solver of the damped normal
   equations (3–4× faster than BVLS; IK, FK and Jacobian inner loops were profiled).
5. **Gaze.** Neck targets point the head at the currently manipulated object (or the
   active hand), within neck limits; the neck is scaled toward 0 where the turned head would
   come closer to an arm than the IK clearance margin (the arm IK assumes the neck at 0), with
   a backward pass so the rate limit can return it in time.
6. **Gripper.** Opening maps to `{l,r}_hand_finger` through the calibrated opening
   curve, continuous and binary. While grasping, the pad separation is at most the held
   object's extent along the (re-selected) closing axis, minus a 0.05 rad squeeze
   (0.4 Nm at kp 8, about 8 N per pad). A 0.15 rad squeeze made the cube creep 4–7° in the hand.
   *Release* (`targets.release_starts`, `release_ramp`, on the output clock): from the last
   post-grasp plateau row of the source-mapped angle (the source fingers start to leave the
   object; a Panda whose command ramps out of its squeeze stays at the object width for 0.2–0.3 s)
   the fingers open at their 3 rad/s speed limit (× `velocity_scale`) toward the source's next
   opening peak, never below the source-mapped angle (`extra["retarget"]["release_ramp"]`).
   *Placed drops* (`targets.place_labels`, derived label, `extra["retarget"]["placed_drops"]`):
   when the source object falls ≤ 4 cm away from the hand after the release and settles within
   0.6 s, the grasp label extends to the row where it has settled (≤ 3 mm and 0.05 rad from its
   rest pose); the hand carries it down object-centrically and opens there. Skipped when Reachy's
   fingers would sink into scene boxes on the way down or scene geometry stands within 1 cm
   beside the settled object (insertions: the coffee pod into its holder, objects into bins).
7. **Timing.** Phase-preserving time scaling so joint and base speed limits hold, then
   resampling at 50 Hz. The source↔target time map is stored. The source step into and out of
   each grasp lasts at least 0.3 s, so Reachy's fingers settle on the object before the arm
   lifts it (the source gripper stalls within one control step; without the dwell the Lift
   cube slid 2–3 cm down the pads).

## Validation tiers

* **K** (every episode): TCP residual (≤ 5 mm, ≤ 0.05 rad), joint limits, speed limits,
  self-clearance ≥ 9 mm, base footprint clear of static scene geometry, grasp-phase
  hand–object relative pose consistency. Failures are saved with reasons.
* **P** (sources with a MuJoCo scene): the source robot is removed from the source
  scene, Reachy is inserted, objects are free bodies, position servos track the
  retargeted joints, and only actuator commands drive the rollout. Gates follow the
  physical thresholds from the previous phase (`legacy/reachy_retarget/physical_gates.py`)
  plus a task predicate and actuator-only replay reproducibility.
  * Bodies the adapter declares inactive (`SceneRef.inactive_bodies`) are removed (robosuite
    PickPlaceCan parks Milk, Bread and Cereal overlapping at (10, 10, 10): the constant
    0.0399 m object–environment depth). Sources that declare none (`None`) fall back to
    removing free bodies that no episode object refers to and that start more than 2 m from
    the workspace.
  * *Source-relative object–environment gate.* MuJoCo contacts are soft, so source objects
    rest or land millimetres deep in their supports. The allowed object–environment depth is
    `max(2 mm, source reference + 1 mm)`, where the reference
    (`SceneRef.reference["object_environment_depth_m"]`, `sources/contact_reference.py`) is
    the deepest contact between a free object and any non-robot body (object–object included)
    over the source's recorded states (state set, `mj_forward`, never stepped). The
    reference, the applied threshold, and the absolute 2 mm verdict are all stored in the
    tier-P metrics, and the reason text names the threshold that applied. Sources without a
    reference (ManiSkill primitive scenes, RoboVerse) keep 2 mm. Robot gates (robot–environment,
    hand–object, self) stay absolute.
  * Servo commands lead the retargeted `q` by each servo's `kv/kp` (0.13 s for the arms):
    `kp (q(t + kv/kp) − q) − kv q̇ ≈ kp (q_ref − q) + kv (q̇_ref − q̇)`, a PD servo with
    velocity reference, still a deterministic function of the retargeted trajectory. Fingers
    take the more closed of the led and the current reference (lead while closing, never an
    early release). `tcp_tracking` compares against the reference, not the led command.
  * *Source end motion.* Sources may end while an object still moves (ManiSkill RollBall
    stops at the first success with the ball rolling at 0.6–1.0 m/s; a faithful replay can
    then neither end at the source's final pose after the 1 s hold nor at rest). Per tracked
    object, the source end speed is the pose change over the last 0.1 s of source time
    (`physics.source_end_motion`). At rest (< 0.02 m/s and < 0.2 rad/s, the existing rest
    thresholds) both gates apply as before (`rule = "after_hold"`). Otherwise
    `task_final_pose` compares the rollout pose at the instant the retargeted trajectory
    reaches the source's last frame (first row whose `source_time` reaches it) with that
    frame's pose (`rule = "source_end"`), and the object's final speed is reported, not
    gated, by `objects_at_rest`. Untracked free bodies and objects whose end speed cannot be
    measured keep the rest requirement. Rules, end speeds and both errors are stored per
    object (`metrics.task`, `metrics.objects_at_rest_rule`); robot gates and thresholds are
    unchanged.
  * Object–environment contacts during the settle phase (objects released from the
    source's initial state; robosuite cubes start ~1 cm above the table) are reported, not
    gated. Grasp gates also count a hand as closed while the episode's grasp label is set
    (large objects stall the fingers above the 0.5 opening point).

## Quality evaluation

`python -m reachy_retarget.evaluate --family robomimic --path <hdf5> --demos 0-19 --physics
--out runs/eval/<name>.jsonl [--write DIR] [--jobs N]` retargets each demo, runs tier K and
(with a source scene) tier P, prints per-episode reasons and an aggregate table, writes one
JSON line per episode, and with `--write` stores every episode (failures included) plus
`index.parquet`. `python -m reachy_retarget.report` buckets failure reasons by kind
(`report.reason_kind`: the tier-P gate name, or the K reason with sides, object names and numbers
stripped, e.g. "TCP rotation residual"), per tier, for the first and for any reason of each
episode. Results on robomimic v1.5 `ph` (dev = demos 0–19 used while developing,
held-out = demos 100–119, evaluated at the end and not used for tuning), episodes passing K / P:

| task | baseline dev | v5 dev | final dev | baseline held-out | v5 held-out | final held-out |
| --- | --- | --- | --- | --- | --- | --- |
| Lift | 20 / 0 | 20 / 20 | 20 / 20 | 19 / 0 | 19 / 19 | 19 / 19 (K & P 18) |
| Can | 0 / 0 | 17 / 0 | 15 / 5 (K & P 4) | 0 / 0 | 16 / 0 | 15 / 5 (K & P 4) |
| Square | 0 / 0 | 13 / 0 | 13 / 0 | 1 / 0 | 14 / 0 | 15 / 0 |
| Transport | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

"final" = source-relative object–environment gate, recorded gripper commands for grasp labels,
the Can as a cylinder and declared inactive bodies (v5 used the absolute 2 mm gate). No episode
passes P under the absolute 2 mm gate on Can, Square or Transport. The source references
(state-sampled, 20 Hz) are Lift 2.6–5.7 mm (cube resting in the table), Can 1.3–9.5 mm (median
3.2), Square 7.6–18.9 mm, Transport 11–27 mm. They agree with re-stepping the source at 2 ms for
Square (7.9–8.1 mm) and Transport (9–19 mm) but are lower for Can (5.4–7.7 mm): the Can's drop
into its bin peaks between recorded states. 8 dev and 4 held-out Can episodes still pass K and
every P gate except the object–environment one (the Reachy rollout drops the can 4–7 mm deep).
Square P is now bounded by `grasp_drift` (18 / 19 of 20), not by penetration. The Can K count
moves with the larger grasp-offset search (cylinder turns in 45° steps): dev loses demos 7
(0.0306 rad > 0.03), 9 and 12 and gains demo 19. Open K failure modes: IK
branch switches between consecutive source frames (0.44–0.92 rad in 5 of the 7 failing dev
Square episodes; the time scaling keeps the speed limits but the interpolated poses miss the
target by 2–5 cm) and, for Transport, the two-robot handover (position residual median
48 mm, down from 70 mm, but the payload orientations of both hands are not reachable with
one constant grasp offset per arm).

### ManiSkill loop (2026-10-07)

Per task one file (RL `pd_joint_delta_pos` where there are several; PickCube teleop, StackPyramid
and PullCubeTool motion planning, PegInsertionSide `rl/trajectory.h5`); dev = the first 10
`traj_*` keys, held-out = the last 10 (evaluated at the end; PickCube teleop has only 10
episodes, so its two samples coincide). Episodes passing K / P (runs in `runs/eval/ms/`):

| task | baseline dev | final dev | baseline held-out | final held-out |
| --- | --- | --- | --- | --- |
| PickCube (teleop) | 7 / 9 | 7 / 9 | (= dev) | (= dev) |
| PokeCube | 6 / 8 | 9 / 7 | 6 / 7 | 7 / 8 |
| PushCube | 8 / 8 | 9 / 7 | 10 / 6 | 10 / 5 |
| PullCube | 3 / 2 | 6 / 4 | 4 / 0 | 10 / 2 |
| StackPyramid | 7 / 2 | 10 / 6 | 7 / 1 | 9 / 5 |
| RollBall | 3 / 0 | 6 / 0 | 5 / 0 | 3 / 0 |
| LiftPegUpright | 1 / 0 | 0 / 0 | 2 / 0 | 0 / 0 |
| PegInsertionSide | 2 / 0 | 3 / 0 | 3 / 0 | 4 / 0 |
| PullCubeTool | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| TwoRobotPickCube | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 0 |
| **total** | 37 / 28 | 50 / 33 | 44 / 23 | 51 / 29 |

Robomimic demos 0–9 before → after: Lift 10/10 → 10/10, Can 7/3 → 7/2 (demo 3 misses the
source-relative object–environment threshold by 0.4 mm), Square 6/0 → 5/0, Transport 0/0 → 0/0
(97 s instead of 128 s per episode). The StackPyramid held-out drop to 1 K pass of an
intermediate version was diagnosed on held-out episodes (the touch exclusion of the finger
screen); the other changes were developed on dev only.

Root causes found, by gate: (1) *mislabelled pushes*: closed empty fingers next to a cube were
grasps (PullCube, StackPyramid cubeA, TwoRobotPickCube agent 0), so tier K demanded rigid
hand–cube poses during pushes; (2) *strict orientation far from objects*: hands without grasps
and the first frames of RL episodes failed the 0.05 rad gate at frame 0 (PullCube 6/10,
PokeCube 3/10 dev); (3) *jittering RL stall*: the LiftPegUpright label flickered off while the
peg was lifted (bridging it makes K stricter there: 1 → 0 dev, but the earlier passes skipped the
lift); (4) *finger screen* counting pushed objects as obstacles (StackPyramid). Rejected
experiments: tilts up to ±90° (+8 K, −10 P on dev: horizontal approaches put the forearm into the
objects, which the finger screen does not model) and ±60° (+1 K, −2 P); flips only for hands
without grasps (PullCube K 8 → 5, RollBall 6 → 1); open-hand pushes followed object-centrically
(PushCube P 8 → 5); sampling the offset pre-screen on strict-orientation keyframes only (Square
K 6 → 1).

Remaining failure modes: Reachy's ±30° wrist during carries that rotate the object
(PickCube teleop, PullCubeTool, LiftPegUpright, PegInsertionSide: TCP rotation residual 0.1–0.5
rad while holding); dynamic RL pushes and flicks (PullCube/PushCube RL hit the cube and let it
slide at 0.5–0.8 m/s; time scaling to Reachy's speed limits makes the hit 2–4× slower, so the
cube travels less: tier P `task_final_pose`); RollBall (see the follow-up below); the PullCubeTool L tool turns ~1 rad about the pad normal in Reachy's grasp
(grasped 12 cm from its centre of mass; the pads resist that torque only through contact
spread); two facing robots on one Reachy (TwoRobotPickCube, Transport handover: both hands near
one object from opposite sides, self-clearance and wrist limits).

Follow-up (same dev episodes, robomimic Lift demos 0–9), K / P before → after:

| change | task | before | after |
| --- | --- | --- | --- |
| source end motion | RollBall | 6 / 0 | 6 / 0 |
| | PushCube | 9 / 7 | 9 / 7 |
| | PokeCube | 9 / 7 | 9 / 7 |
| | Lift (robomimic) | 10 / 10 | 10 / 10 |
| `boxes` geometry | PullCubeTool | 0 / 0 | 0 / 0 |

RollBall no longer fails `objects_at_rest` (10 → 0) but still fails `task_final_pose` in all 10:
the gate is now attainable, the replay is not. Time scaling stretches the strike (traj_0: 2.15 s
source → 5.4 s), so Reachy's hand rolls the ball at ~0.1 m/s instead of 1 m/s and it stops
0.5–2.3 m short of the source's end position. Every PushCube, PokeCube and Lift object also ends
the source moving (PushCube 0.1–0.3 m/s, PokeCube 0.02–0.05 m/s, the Lift cube rises at
0.1–0.25 m/s at first success), so all use `source_end`; their errors change by ≤ 9 mm and no
verdict flips (PushCube traj_9, already failing, now also misses the 3 cm tolerance). The L
tool's geometry changes the grasp offset by 0.4° only: PullCubeTool fails on Reachy's ±30° wrist
(TCP rotation residual 0.18–0.52 rad, hand–tool relative drift) and in P on the tool turning
~1 rad in the grasp.

### Tier-P physics loop: in-hand slip (MimicGen sources, 2026-10-08)

Dev = MimicGen source demos 0–4 of the 12 local tasks plus robomimic `ph` Can and Lift demos 0–4;
held-out = demos 5–9, run once at the end. MimicGen `square` is the robomimic Square `ph` file
(identical states) and is counted once. Episodes passing K / P (K & P), runs in `runs/eval/pq/`:

| task | baseline dev | final dev | baseline held-out | final held-out |
| --- | --- | --- | --- | --- |
| stack | 5 / 0 (0) | 5 / 3 (3) | 1 / 0 (0) | 3 / 2 (2) |
| stack_three | 4 / 0 (0) | 4 / 3 (3) | 4 / 0 (0) | 5 / 0 (0) |
| threading | 2 / 5 (2) | 2 / 5 (2) | 2 / 3 (2) | 2 / 5 (2) |
| coffee | 2 / 1 (0) | 5 / 2 (2) | 4 / 0 (0) | 4 / 2 (2) |
| square | 2 / 0 (0) | 3 / 0 (0) | 3 / 0 (0) | 2 / 0 (0) |
| three_piece_assembly | 2 / 0 (0) | 2 / 0 (0) | 1 / 0 (0) | 0 / 0 (0) |
| mug_cleanup | 5 / 0 (0) | 4 / 0 (0) | 5 / 0 (0) | 4 / 0 (0) |
| coffee_preparation | 1 / 0 (0) | 1 / 0 (0) | 0 / 0 (0) | 0 / 0 (0) |
| kitchen | 1 / 0 (0) | 0 / 0 (0) | 1 / 0 (0) | 2 / 0 (0) |
| hammer_cleanup | 5 / 0 (0) | 5 / 0 (0) | 5 / 0 (0) | 5 / 0 (0) |
| nut_assembly | 0 / 0 (0) | 0 / 0 (0) | 0 / 0 (0) | 0 / 0 (0) |
| pick_place | 4 / 0 (0) | 0 / 0 (0) | 2 / 0 (0) | 0 / 0 (0) |
| robomimic Can | 5 / 1 (1) | 5 / 2 (2) | 2 / 1 (1) | 2 / 1 (1) |
| robomimic Lift | 5 / 5 (5) | 5 / 5 (5) | 5 / 5 (5) | 5 / 5 (5) |
| **total (70)** | 43 / 12 (8) | 41 / 20 (17) | 35 / 9 (8) | 34 / 15 (12) |

Dev P / K & P by step: baseline 12 / 8; release ramp 15 / 11; labels for multi-part objects and
pad-face widths 16 / 13; placed drops 20 / 17; fixtures with articulated children 20 / 17
(`robot_environment_penetration` first failures 6 → 1). K drops where grasps are now labelled
and checked strictly (PickPlace, Kitchen: Reachy's ±30° wrist while holding). ManiSkill dev check
(100 episodes, same commands): K & P 29 → 28, P 33 → 33 (StackPyramid P 6 → 7 but K 10 → 7: 3–5 mm
drops are placed and the strict grasp orientation now covers the landing; PushCube traj_5 loses
P after the release ramp on a brief RL grasp label).

Root causes, measured on every acquired grasp of the dev rollouts (`grasp_drift` anchor, contact
forces, finger actuator force, hand motion; 75 grasps, 53 over the 3 mm / 3° gate):
* *Release, 32 of 53*: the gate counts a hand as closed while its finger reference is below the
  0.5 opening (1.095 rad, a 46.8 mm pad gap), but the reference followed the source width, which
  stays at the object width for 0.2–0.3 s after the open command (the Panda's command ramping out
  of its squeeze), then the 0.3 s release dwell; a 40 mm cube needs 0.145 rad more, a 24 mm hammer
  handle 0.47 rad. 13 had no pad contact at the violation: the object falls in the "closed" hand.
  MimicGen operators release up to 3 cm above the stack (Stack: 8–21 mm in all 5 dev demos), longer
  drops go into bins, holders, drawers and onto pegs (Kitchen bread 9–12 cm, hammer 7–13 cm, Square
  nut 8–11 cm, PickPlace 3–8 cm).
* *Unlabelled grasps*: 38 of 105 closed-gripper runs had no grasp label (multi-part envelopes:
  mug, ThreePieceAssembly pieces; the Rethink gripper's widths 14 mm narrow), so no squeeze: 7
  grasps held with < 0.1 Nm; 8 PickPlace grasps saturated the 2 Nm finger actuator (commanded
  14 mm inside the object).
* *Carry, 19–23*: about half follow object–environment contact within 0.3 s (nut on the peg,
  pieces on the base, objects on bin walls and the coffee machine); the rest are steady soft-contact
  creep below the friction cone (friction use 0.2–0.6; the 7 g coffee pod lags the hand's turn
  about the approach axis by 25 %, the mug pivots about the pad normal at ~2°/s under 70–100 N).
  Solver sensitivity on coffee demo 4 (10.7°): noslip 3 iterations 0.9°, impratio 100 11.1°,
  impratio 1000 4.3°, 0.5 ms step 8.9°. Hand acceleration at the violation is low (median
  0.2 m/s², max 4.3), so slower carries were not indicated; lift-off accounts for 2.
* Contact geometry: both pads touch in every acquired grasp (median 5–6 contacts); normals are
  more than 25° off the closing axis only on mug handles and the Milk carton's gable top
  (8 of 79).

Hardware grounding: the finger speed limit is 3 rad/s (`robot.VELOCITY`). Reachy 2's grippers are
XM-series Dynamixels in current-based position control with a 0.4 A current limit
(reachy2_core `grippers.yaml`, `gripper_dynamixel_controller`), about 0.5–0.7 Nm at XM430 torque
constants, below the simulated ±2 Nm force range (reachy-agent `gripper_force`). The finger torque
at acquisition is 0.70 Nm median after the width correction (0.44 before; ThreadingNeedle 1.9 Nm,
6 of 172 dev and held-out grasps saturate), so squeeze was not raised; the actuator model was
left unchanged (with ±0.7 Nm, kv 0.8 would cap the finger speed at 0.9 rad/s).

Rejected: MuJoCo noslip (3 iterations) for all contacts (dev P 16 → 15, threading 5 → 3, more
`hand_object_penetration`; legacy treated it as a sensitivity only); placing 2 mm above the rest
pose (stacking P unchanged, 8 → 8); following the fall through the landing bounce (Stack demo 1:
the hand chases a 6 rad/s tumble, K fails; dev 19 / 16, kept: up to the settled pose, which gave
the same 20 / 17 as stopping at the first touch); a 5 mm minimum drop (Stack −1, ManiSkill +1);
placing insertions (coffee P 2 → 0: the pod sits on the holder rim).

Remaining failure modes (final, dev and held-out, 110 of 172 grasps over the gate): drops that are
not placed (bins, holders, drawers, pegs: 33 of 62 release violations have no pad contact;
Kitchen, HammerCleanup, PickPlace, NutAssembly, Square cannot pass `grasp_drift` while the closed
point is the absolute 0.5 opening); placed cubes rocking 3–7° about the pad normal as the pads
open (soft contact preloaded by 1–3 N); insertion contacts while carrying (26 of 48 carry
violations); creep of light objects (7–21 g); approach collisions of Reachy's wider fingers with
neighbouring objects (StackThree demo 0 knocks cubeC, ThreePieceAssembly fingers land on the
piece); drawer and lid manipulation that blocks the hand (MugCleanup `tcp_tracking` 3–8 cm).

## Episode storage (`reachy-retarget-episode-v2`)

One HDF5 file per episode plus a dataset `index.parquet`. See `docs/schema.md`.
