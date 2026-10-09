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
source files ──adapter──▶ SourceEpisode ──retarget──▶ Reachy trajectory ──validate K/P──▶ scene meshes ──▶ ReachyEpisode (HDF5) + assets/
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
* `reachy_retarget.meshes`: scene components with their visual/collision meshes, materials and
  textures, the content-addressed asset library and per-frame component poses (*Scene meshes*).
* `reachy_retarget.schema`: data model, HDF5 I/O, control-mode registry, scene reader
  (`schema.scene_assets`).
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
joint speed 1 rad/s, neck 30°/s, base 0.61 m/s per body axis and 114°/s (whole-body
controller limits; time scaling and tier K use base-frame translation, as tier P measures it).
The more conservative task-space executor limits of reachy-agent (0.22 m/s, 0.6 rad/s base) are
reported as a flag (`tier_k_metrics.base_exceeds_executor_limits`, `peak_base_speed_body`), not
enforced.

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
  (`inactive_bodies`) and `reference` values measured on the source's own states. It is also the
  source of the stored scene meshes; an episode without one is not written.
* `scene_qpos`: per-frame positions of every non-robot joint of `scene` (free objects including
  untracked ones such as distractors, articulated parts), columns named like the tier-P rollout
  (`<joint>`, `<joint>/x .. /qz`); robosuite family, RoboCasa and BiGym fill it from their recorded
  states. Used only to pose scene components.
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
     squeeze, no grasp-phase checks); now all 105 have one. An `aabb` closed run that carries the
     object (it travels > 3 cm and stays within 3 cm of its median pose in the grasp frame) is a grasp
     whatever the width: BiGym's Robotiq holds a mug by its wall at 15 mm, half of the 95 mm envelope.
   * *Closed-hand pushes* (`targets.push_labels`, derived label recorded in
     `extra["retarget"]["push"]`, never a grasp label): frames where a closed but empty hand is
     within contact distance of an object that moves faster than 2 cm/s together with the hand
     (|v_obj − v_hand| ≤ 0.5 |v_obj|). They get the object-centric carrying rule below, so the
     hand keeps the contact geometry of the push onset instead of sliding off the object.
   * *Object-centric grasp re-selection* (`targets`, chosen by placement scoring, recorded
     in `extra["grasp_offsets"]`). The task is the object path, not the Panda hand pose, so
     each arm uses the source grasp frame composed with a constant offset
     `Rz(pi·flip + theta) · Ry(tilt)` about the grasp center: `flip` (half turn, always a
     parallel-jaw symmetry); `tilt` ∈ {0, ±15, …, ±90}° about the closing axis (pad planes
     unchanged, the pads press the same faces; hands without grasp segments stop at ±45°, and the
     placement score penalizes arm links inside scene boxes, see the reachability loop); `theta` = the rotation that lays the closing
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
   * *Pad coverage and support lift* (loop 3, `targets.pad_coverage`, `targets.support_lift`): offsets
     whose pads hold less than 80 % of the closing extent the source frame's pads hold are dropped.
     When the source grasp frame's fingers sink more than 2 mm into support/fixture boxes during the
     pick-up (until the object has moved 1 cm), each candidate retracts the grasp center along the
     source approach axis by its own depth + 2 mm (≤ 15 mm, reduced while the coverage would fall below
     80 %): a fourth offset element `lift_m`, constant in the hand frame, so the object path is unchanged.
   * *Free orientation away from grasps* (`targets.orientation_weight`): the TCP orientation is
     strict inside grasp windows (segment − 0.5 s … + 0.3 s) and its weight decays to 0.05 over
     1 s outside them. Outside grasp segments the weight is further capped by a proximity ramp
     (`contact_strict_distance` 5 cm → `contact_free_distance` 12 cm, grasp center to the nearest
     manipulated object or, loop 3, articulated fixture with box geometry, at distance 0 while its
     joints move after the hand reached it within 2 cm): the orientation only matters where the fingers can touch something.
     Hands that never grasp (pushing, poking with the fingers) were strict on every frame and
     failed tier K at the first frames, where an RL Panda starts above the table with a wrist
     pose Reachy's ±30° wrist cannot copy; an approach that starts far away now starts free. A first IK pass with these weights gives the orientation Reachy prefers
     in free space; the reference rotation is that one blended into the source (offset)
     rotation by the weight, and a second pass tracks this reference strictly, so tier K still
     checks every frame against a stored reference (`reference.tcp`).
   * *Idle hands of mobile sources* (`targets.position_weight`, `idle_distance`): the position
     weight follows the same rule with a 0.15 m (strict) to 0.30 m (free, weight 0.05) ramp on the
     distance to the nearest task object (manipulated or receptacle surface; objects without box
     geometry: origin distance − 0.15 m; articulated fixtures without geometry and objects whose
     pose is unknown at that row count as 0). In the first pass an idle arm also gets an extra
     nominal-posture weight 0.3 × (1 − w); where w < 1 its position reference is the first-pass
     pose (`free_orientation.<side>.idle_shifted_frames`, `max_reference_shift_m`). BEHAVIOR's R1
     Pro carries its free arm at z 0.44 m, below Reachy's reach; tracking it twisted the arm and
     dilated time 4.6× (episode 00000010: K 7,996 failing frames → 0, 253 s → 45 s).
2. **Arm assignment.** Single-arm sources are assigned to the arm with the better
   reachability score over the trajectory. Bimanual sources keep their side hints. Navigation
   episodes (`regime = "navigation"`, RoboCasa NavigateKitchen holds the Panda hand fixed on the
   base) retarget no effector: the arms hold `stow` (`rest` with 16° shoulder roll toward the torso),
   only the base and the head move (`extra["notes"]`). A
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
   * *Projected candidates* (`placement._project`): a ring candidate whose footprint overlaps scene
     geometry is moved straight away from the target centroid (heading kept, 2 cm steps, ≤ 0.8 m)
     to the footprint margin, for episodes with grasps. *Arm screen* (`PlacementProblem.arm_contact`):
     the placement score adds the depth (per cm, mean + max over keyframes) of the upper arm,
     forearm and palm collision spheres in solid object boxes (a grasped object is exempt within its
     approach/retreat window).
   * *Scene geometry replaces envelopes* (`footprint.merge_scene`): with source-scene geometry, a
     static object envelope is dropped where a scene geom inside it reaches its top (± 2 cm), in
     placement, IK and tier K alike.
   * *Scene geometry* (`footprint.scene_obstacles`): when the source ships a MuJoCo scene, every
     colliding environment geom (walls, counters, cabinets, appliances, articulated parts at their
     initial state; not free objects) within 1.5 m of the source hands/base becomes a footprint
     polygon with its height interval, for placement, the free-base IK constraint and tier K (stored
     in `extra["scene_footprint"]`). The disc stack ends at 1.45 m (head top 1.41 m); a mobile
     episode with a resting arm adds the arm's band (`footprint.arm_band`: 0.44–1.21 m, 0.33 m for
     `rest`, 0.27 m for the navigation `stow` posture).
   * *Mobile sources* keep the free base (deviation penalty) and get the same linearized footprint
     constraint (static objects + scene geometry). Until 2026-10-08 the constraint rows were added on
     every iteration (an identity test on a fresh array view never held), which pinned a free base to
     the margin contour of the nearest obstacle: base assistance then turned the base by up to
     1.4 rad instead of stepping 0.3 m (RoboCasa OpenDrawer).
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
   active hand; without active hands, 1.5 m ahead along the direction of travel, toward the base
   position 1 m further along its path), within neck limits; the neck is scaled toward 0 where the turned head would
   come closer to an arm than the IK clearance margin (the arm IK assumes the neck at 0), with
   a backward pass so the rate limit can return it in time.
6. **Gripper.** Opening maps to `{l,r}_hand_finger` through the calibrated opening
   curve, continuous and binary. While grasping, the pad separation is at most the held
   object's extent along the (re-selected) closing axis, minus a 0.05 rad squeeze
   (0.4 Nm at kp 8, about 8 N per pad). A 0.15 rad squeeze made the cube creep 4–7° in the hand.
   *Held width* (`grasp_width_rise`, loop 3): while a grasp label is set the source width rises at
   most 2 mm above its running minimum (a bowl rim pried open as the operator presses it down is not
   a release).
   *Release* (`targets.release_starts`, `release_ramp`, on the output clock): from the last held
   row (the source's open command, `release_at_command`, loop 3; before, the last post-grasp plateau
   row of the source-mapped angle, which waited 0.2–0.4 s while a Panda ramped out of its squeeze
   and the hand already retreated) the fingers open at their 3 rad/s speed limit (× `velocity_scale`) toward the source's next
   opening peak, never below the source-mapped angle (`extra["retarget"]["release_ramp"]`).
   *Placed drops* (`targets.place_labels`, derived label, `extra["retarget"]["placed_drops"]`):
   when the source object falls ≤ 4 cm away from the hand after the release and settles within
   0.6 s, the grasp label extends to the row where it has settled (≤ 3 mm and 0.05 rad from its
   rest pose); the hand carries it down object-centrically, keeping the squeeze it had at the source
   release (the source fingers open during the fall; following them left the cube unsqueezed as it
   landed), and opens from the last held row at the finger speed. Skipped when Reachy's
   fingers would sink into scene boxes on the way down or scene geometry stands within 1 cm
   beside the settled object above half its height (insertions: the coffee pod into its holder,
   objects into bins; a plate rim 15 mm up a 64 mm bowl is not one, loop 3).
   *Approach opening* (`targets.approach_width`): within 12 cm of the next (or previous) grasped
   object the pads open only to its extent along the closing axis plus twice its centre offset
   plus 8 mm per side (Reachy's distal fingers are 27.5 mm thick, the Panda's about 10 mm).
   *Straight final approach* (`targets.straight_approach`): before a pick (the object rises ≥ 15 mm
   during the grasp), the path's offset across the grasp's approach axis is removed while the
   fingertips are within 5 mm of the object's far side, blending back over 3 cm; closed-hand pushes
   and the retreat are unchanged.
7. **Timing.** Phase-preserving time scaling so joint and base speed limits hold (base translation
   per body axis), then resampling at 50 Hz. Output-clock refinement skips rows between two source
   rows whose TCP position the source-clock IK already left more than 2 cm off (unreachable
   targets; smaller misses are often recovered on the output clock). The source↔target time map is stored. The source step into and out of
   each grasp lasts at least 0.3 s, so Reachy's fingers settle on the object before the arm
   lifts it (the source gripper stalls within one control step; without the dwell the Lift
   cube slid 2–3 cm down the pads). Articulated slide joints (drawers, typed `slide` in the source
   MJCF) move at most 0.04 m/s: drawer damping resists the position-servoed arm in proportion to
   the speed.

## Validation tiers

* **K** (every episode): TCP residual (≤ 5 mm, ≤ 0.05 rad; failing frames whose grasp-center target
  lies outside Reachy's reach band z 0.46–1.87 m are named as such, `<side>_unreachable_frames`),
  navigation episodes end within 3 cm of their base reference, joint limits, speed limits,
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
  * *Task predicate without a manipulated object* (missing measurements still fail otherwise): a
    `navigation` episode passes `task_final_pose` when the measured base ends within 3 cm of the
    last base reference; an articulated task (BiGym drawers, cupboards, dishwasher racks) when every
    scene joint the source moved by ≥ 1 cm / 0.05 rad ends within 3 cm / 0.05 rad of the source's
    final value after the hold (`metrics.task[...]`, rule `navigation_destination` |
    `articulation_final`). Before, both always failed.
  * *Floor bodies*: Reachy's wheels are excluded against every static body carrying the floor (a
    plane at z = 0, or a horizontal box wider than 1 m whose top is at z = 0: BiGym `floor`,
    RoboCasa `floor*_room_main`), as they already were against `world`. Wheel–floor friction dragged
    the planar base joints (stick-slip): BiGym DishwasherLoadCups measured base 0.86 m/s and 2.5 rad/s
    against a 0.44 m/s / 1.2 rad/s reference, then `arm_speed` 2.1 rad/s and `tcp_tracking` 7 cm;
    with the exclusion all three gates pass.
  * Object–environment contacts during the settle phase (objects released from the
    source's initial state; robosuite cubes start ~1 cm above the table) are reported, not
    gated. Grasp gates also count a hand as closed while the episode's grasp label is set
    (large objects stall the fingers above the 0.5 opening point), and only while its finger
    reference is at or below the pad-contact angle of the object (see the loop 2 section).

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

### Tier-P physics loop 2: release definition and approach geometry (2026-10-08)

Same dev / held-out split as loop 1 (MimicGen demos 0–4 / 5–9 of 12 tasks, robomimic Can and Lift);
ManiSkill spot check = first 10 episodes of PickCube (teleop), StackPyramid, PushCube. Runs and
audit inputs in `runs/eval/pq2/`.

*Decision: release at pad contact (measurement definition, thresholds unchanged).* In tier P a hand
is closed on object o for `grasp_drift` / `carry_contact` only while, in addition to the existing
condition (reference below the 0.5 opening angle or grasp label set), its finger reference is at
or below the angle at which its pads touch o on both sides: `gripper.width_to_angle` of o's extent
along the closing axis inside the pad prism (`targets.closing_extent`, the width logic of the
finger command, on the measured grasp-center and object poses; exact per part of a union of
boxes, `targets._clipped_span`; NaN, nothing between the pads, adds no condition). Commanded wider,
the object is released and judged by `task_final_pose`, `objects_at_rest` and
`object_environment_penetration`. Every carry records its release time, rule
(`pad_contact_angle` | `closed_point`), command, measured finger, contact angle and width in
`metrics.grasps[...].releases` (`PhysicsConfig.grasp_trace` stores the per-row traces). The first
attempt, the extent of the part nearest the grasp center, failed the audit: on the 32-box mug the
nearest part was 18 mm narrower than the pad gap, so the squeezed hand never counted as closed and
a 9.5° mug carry disappeared; reference poses instead of measured ones lost hands whose IK misses
the object (NutAssembly).

Reclassification audit (old and new rule on the same rollouts): dev `grasp_drift` failures 41 → 29
episodes, `carry_contact` 8 → 6, P 20 → 24, K & P 17 → 20; held-out `grasp_drift` 48 → 35, P 15 →
21, K & P 12 → 16. Every acquired carry keeps its acquisition (82 / 82 dev, 90 / 90 held-out). The
violations no longer counted (14 dev, 16 held-out) all start 0.00–0.06 s after the command passed
the contact angle with at most one pad touching (13 / 14 and 16 / 16: the object falls, 0.2 m/s,
or a placed cube rocks 3–5° as the pads open); the drift up to the release stays within the gate
(max 3.0 mm / 2.9°). Borderline: Square dev demo 2 (nut pushed onto the peg) reaches 2.97 mm while
commanded closed and 3.1 mm on the release row itself, with both pads still touching through the
0.02 rad finger servo lag. Releases caused by geometry (the object turned or slid in the hand so
its extent fell below the command, 5 held-out carries) all violated the gate before, under both
rules. No in-hand slip while the pads command contact is hidden.

Changes (dev P / K & P; ManiSkill 30: K / P / K & P):

| step | dev | ManiSkill |
| --- | --- | --- |
| baseline (loop 1, old rule) | 20 / 17 | 23 / 22 / 18 |
| release at pad contact | 24 / 20 | 23 / 22 / 18 |
| placed drops keep the squeeze, open from the last held row | 25 / 21 | 23 / 22 / 18 |
| narrowed approach opening | 26 / 21 | 23 / 23 / 18 |
| straight approach and retreat, every grasp (rejected) | 24 / 21 | not run |
| straight approach only, every grasp (rejected) | 31 / 24 | 21 / 18 / 11 |
| straight approach to picks only, pushes frozen | 31 / 24 | 22 / 25 / 18 |
| drawer slide speed 0.04 m/s (final) | 29 / 22 | 22 / 25 / 18 |

The last row's dev drop is Coffee demos 2 and 4 (pod `grasp_drift` 3.1° and 5.7°) with retargets
identical to the previous row within 5e-12 rad: the pod's in-hand rotation is chaotic at the
1e-12 level, so ±2 Coffee episodes are noise. Rejected: straightening the retreat (the Kitchen
pot and bread end 10 cm off), straightening approaches that end in a closed-hand push or an RL
grasp that never lifts the cube (StackPyramid K & P 6 → 0: the hand no longer pushed cubeA;
PushCube traj 2 missed its goal), selecting segments by the finger screen's neighbour depth
(`finger_penetration` did not predict the physics contacts: StackThree demo 0's depth rose
11 → 27 mm·rows with the fix that cleared cubeC; dev 26 / 21).

| task | baseline dev | release rule dev | final dev | baseline held-out | release rule held-out | final held-out |
| --- | --- | --- | --- | --- | --- | --- |
| stack | 5 / 3 (3) | 5 / 4 (4) | 5 / 5 (5) | 3 / 2 (2) | 3 / 3 (2) | 3 / 5 (3) |
| stack_three | 4 / 3 (3) | 4 / 3 (3) | 3 / 4 (3) | 5 / 0 (0) | 5 / 1 (1) | 4 / 3 (2) |
| threading | 2 / 5 (2) | 2 / 5 (2) | 3 / 4 (3) | 2 / 5 (2) | 2 / 5 (2) | 3 / 4 (3) |
| coffee | 5 / 2 (2) | 5 / 2 (2) | 5 / 3 (3) | 4 / 2 (2) | 4 / 2 (2) | 4 / 5 (4) |
| square | 3 / 0 (0) | 3 / 1 (1) | 4 / 0 (0) | 2 / 0 (0) | 2 / 0 (0) | 4 / 0 (0) |
| three_piece_assembly | 2 / 0 (0) | 2 / 0 (0) | 3 / 0 (0) | 0 / 0 (0) | 0 / 0 (0) | 0 / 1 (0) |
| mug_cleanup | 4 / 0 (0) | 4 / 0 (0) | 5 / 0 (0) | 4 / 0 (0) | 4 / 0 (0) | 5 / 0 (0) |
| coffee_preparation | 1 / 0 (0) | 1 / 0 (0) | 0 / 0 (0) | 0 / 0 (0) | 0 / 0 (0) | 0 / 0 (0) |
| kitchen | 0 / 0 (0) | 0 / 1 (0) | 1 / 4 (1) | 2 / 0 (0) | 2 / 3 (2) | 2 / 3 (2) |
| hammer_cleanup | 5 / 0 (0) | 5 / 1 (1) | 5 / 1 (1) | 5 / 0 (0) | 5 / 1 (1) | 5 / 2 (2) |
| nut_assembly | 0 / 0 (0) | 0 / 0 (0) | 0 / 0 (0) | 0 / 0 (0) | 0 / 0 (0) | 0 / 0 (0) |
| pick_place | 0 / 0 (0) | 0 / 0 (0) | 0 / 1 (0) | 0 / 0 (0) | 0 / 0 (0) | 0 / 0 (0) |
| robomimic Can | 5 / 2 (2) | 5 / 2 (2) | 4 / 2 (1) | 2 / 1 (1) | 2 / 1 (1) | 4 / 1 (1) |
| robomimic Lift | 5 / 5 (5) | 5 / 5 (5) | 5 / 5 (5) | 5 / 5 (5) | 5 / 5 (5) | 5 / 4 (4) |
| **total (70)** | 41 / 20 (17) | 41 / 24 (20) | 43 / 29 (22) | 34 / 15 (12) | 34 / 21 (16) | 39 / 28 (21) |

ManiSkill final per task (K / P / K & P, baseline → final): PickCube 7/9/6 → 6/10/6, PushCube
9/6/6 → 9/7/6, StackPyramid 7/7/6 → 7/8/6. Held-out Lift demo 8 fails `grasp_drift` (4.1° in a
carry that passed before); Can and StackThree lose K episodes to TCP rotation residuals on the
straightened approaches (0.05–0.1 rad near the ±30° wrist limits).

Measurements behind the remaining failure modes (final dev):
* *Placed cubes rocking*: with the release definition and the held squeeze, Stack passes 5 / 5 dev
  and held-out; the rocking itself (3–5° as the pads open) is now judged by the rest and final-pose
  gates, which it passes.
* *Neighbour contact on approach*: StackThree demo 0's mimic finger tipped cubeC while the hand
  descended 14 mm off cubeA's centre line with the source's 78 mm opening; fixed by the straight
  approach (narrowing alone does not clear it). Remaining: K residuals on straightened approaches.
* *Scene contact while carrying*: 35 carries violate `grasp_drift`; 23 touch scene geometry in the
  0.3 s before (pegs, bins, the coffee machine, assembly pieces). At the violation the reference
  tracks the source within 0.4 mm median (K residual; 3 of 35 have 0.15–0.23 rad orientation
  residual) and the base is still, so the carry paths do not deviate from the source near
  obstacles: these are the source's own insertions, which press the object against the scene and
  turn it in Reachy's pads. IK clearance to static geometry would not change them. The other 12 are
  pivots without scene contact (MugCleanup mug 3–24°, PickPlace Milk at 3.0–3.1°).
* *Drawers*: MugCleanup `tcp_tracking` (3–4 cm) is the hand stalled by the drawer's 100 N s/m
  damping at the source's 0.14 m/s; the slide speed limit leaves 1 of 5 dev episodes over 3 cm
  (3.2 cm). MugCleanup still fails `grasp_drift` in all 10 (the mug pivots 0.17–0.42 rad in the
  pads); CoffeePreparation fails on K, joint margins and the lid/pod sequence.
* NutAssembly (0 / 0) is bounded by K (TCP residuals 5–37 mm) and PickPlace by K (wrist rotation
  while holding); Square by `task_final_pose` after the nut is dropped onto the peg.

### Mobile / whole-body loop (2026-10-08)

Dev (52 episodes, fixed before any change): BEHAVIOR 10 (the first episode of each of the 8 local
tasks, the second of tasks 0 and 1), MobileManiBench 10 (2 per task type), MolmoBot 10 RB-Y1
(5 door, 2 open, 2 pick, 1 pick-and-place), RoboCasa 12 (episodes 0–3 of OpenDrawer,
PickPlaceCounterToCabinet, NavigateKitchen; P with the source scene), BiGym 10 (record 0 of every
4th task; P). Held-out (49, evaluated once at the end): the 6th/7th BEHAVIOR episodes, the next 2
MobileManiBench episodes per type, the other 10 RB-Y1 trajectories, RoboCasa episodes 5–7, record 0
of 10 other BiGym tasks. Runs: `runs/eval/mobile/{baseline,final}-{dev,ho}.jsonl` (baseline = HEAD
`54909d5`). Episodes passing K / P (K & P); P only where a scene exists:

| family | baseline dev | final dev | baseline held-out | final held-out | s/episode dev (base → final) |
| --- | --- | --- | --- | --- | --- |
| BEHAVIOR (R1 Pro) | 0 / – | 2 / – | 0 / – | 3 / – | 1001 → 257 |
| MobileManiBench (G1) | 4 / – | 4 / – | 3 / – | 5 / – | 30 → 15 |
| MolmoBot (RB-Y1) | 9 / – (1 error) | 10 / – | 7 / – (2 errors) | 8 / – | 9 → 8 |
| RoboCasa | 5 / 0 (0) | 4 / 0 (0) | 4 / 0 (0) | 4 / 0 (0) | 49 → 40 |
| BiGym (H1) | 2 / 0 (0) | 3 / 1 (1) | 2 / 0 (0) | 2 / 0 (0) | 106 → 101 |
| **total** | 20 / 0 | 23 / 1 | 16 / 0 | 22 / 0 | |

RoboCasa K includes the new navigation check (NavigateKitchen 4 / 4 → 0 / 4 dev, 3 / 3 → 1 / 3
held-out); without it the totals are 27 dev and 24 held-out. Per task (dev, K baseline → final):
OpenDrawer 1 → 4, radio 0 → 1 of 2, microwave popcorn 0 → 1, DishwasherLoadCups 0 → 1, MovePlate K & P
0 → 1, MolmoBot pick-and-place error → pass; unchanged: MobileManiBench close microwave and pick 4 / 4,
lever door, drawer (open table) and cart 0 / 6, MolmoBot door/open/pick 9 / 9. Held-out BEHAVIOR
timing 780 → 298 s, BiGym 255 → 122 s. Robomimic Lift demos 0–4: 5 / 5 K & P before and after;
ManiSkill PickCube teleop 10: 6 / 10 / 6 before and after (same episodes, same reasons).

Root causes (dev, measured):
* *Idle arms below reach*: BEHAVIOR failed K in 10 / 10 with 53–60 % of its frames on targets at
  z 0.43–0.44 m (R1 Pro arms parked at its side; Reachy's grasp center reaches 0.46–1.87 m, sampled
  over the arm workspace with self-clearance), far from every task object. Tracking them twisted the
  arm (self-clearance −103 mm) and dilated time up to 42× per interval (4.6× overall). Fix: idle
  position weights (above). Episode 00000010: K failing frames 7,996 → 0.
* *Targets Reachy cannot reach*: BEHAVIOR floor picks (shoes 0.05 m, trash 0.12–0.24 m, tripod
  0.07 m), MobileManiBench drawers at 0.30 m, BiGym crouched H1 (pelvis 0.40 m) reaching 0.42–0.56 m
  into dishwashers, and wall cabinets behind counters (TakeCups: shoulder distance 0.70–0.80 m with
  the base held off the counter by the footprint; Reachy's arm 0.62 m). These stay K failures and
  are now named (`TCP target outside Reachy's reach band`, 4 BEHAVIOR + 2 MMB + 1–3 BiGym episodes).
* *Footprint constraint pinned the base* (bug, all sources with base assistance or a free base):
  RoboCasa OpenDrawer 1 → 4 / 4 K.
* *No scene geometry in the footprint*: RoboCasa and BiGym bases drove into cabinet doors and
  dishwashers (P `robot_environment_penetration` 0.15 m on NavigateKitchen; BiGym K footprint overlap
  5 / 10). With `scene_obstacles` K now sees them; P robot–environment first failures: RoboCasa 4 → 3
  (PickPlaceCounterToCabinet stands where the counter blocks the base: K footprint reason), BiGym
  held-out 4 → 0.
* *World-axis base limit*: diagonal motion at 0.58 m/s per world axis is 0.82 m/s in the body; tier P
  measured 0.62 m/s (NavigateKitchen). Time scaling and K now use body-axis increments.
* *Wheel–floor friction in BiGym/RoboCasa scenes* (floor is its own body): `base_speed` 7 / 10 → 0,
  `arm_speed` 9 → 4–5 and `tcp_tracking` 9 → 6–7 BiGym P first/any reasons.
* *Envelope geometry hid BiGym grasps*: 15 mm Robotiq widths against 95 mm mug aabbs; the carried-run
  rule labels both DishwasherLoadCups mugs (left 174 / right 180 frames).
* *Task predicates that could not pass*: BiGym articulated tasks and navigation had no manipulated
  object; DrawerTopOpen now opens its drawer to 0.1 mm of the source in P (it fails on
  self-clearance: the two hands' collision spheres overlap by up to 10 mm at the handle).
* *Navigation*: the source Panda hand was tracked through the house. Now only base and head move;
  RoboCasa P gates all pass except the destination: Reachy with hanging arms cannot stand where the
  Omron base ended (source path clearance −0.10 to −0.24 m against Reachy's body discs + arm band;
  destination missed by 0.03–0.23 m after the `stow` posture, 0.10–3.78 m with `rest`). RoboCasa's own
  success region is 0.20 m around a target the recording does not store, so 3 cm of the source's final
  pose is the gate. Gaze looks along the travel direction.
* *Refinement of unreachable rows*: BiGym SaucepanToHob spent 42 of 68 s re-solving them.
* MolmoBot pick-and-place: receptacle ids with '/' broke episode assembly (adapter fix).

Base speed: sources drive 0.2–0.3 m/s (BEHAVIOR, MobileManiBench, MolmoBot) and up to 1.3 m/s and
2.3 rad/s (BiGym, RoboCasa navigation 0.95 m/s); time scaling keeps 0.61 m/s and 114°/s. 39 / 52 dev
and 33 / 49 held-out episodes exceed the reachy-agent executor flag (0.22 m/s, 0.6 rad/s).

Remaining failure modes: unreachable heights and distances above; MobileManiBench lever door
(handle at 0.62–0.65 m, rotation residual 0.5 rad at Reachy's lower reach edge) and cart (handle at
0.55–0.60 m: arm branch flips, self-clearance); BiGym bimanual handle grasps (hands within 1 cm of
each other), mug handle grasps slipping in Reachy's flat pads (P `grasp_drift`), object-environment
depth of the mug in the dishwasher rack; RoboCasa OpenDrawer P (wrist dips 0.02 rad past the margin
and the hand lags the drawer 3.6–7.7 cm: `joint_margin`, `tcp_tracking`); BEHAVIOR carries are not
labelled (asset geometry only, object origins 4–19 cm from the grasp center), so its K passes do not
check hand–object drift; BEHAVIOR, MobileManiBench and MolmoBot have no scene (no P, no footprint
geometry); articulated fixtures do not make the orientation strict (`hand_object_distance` uses
manipulated objects only).

### Reachability loop: LIBERO and bimanual sources (2026-10-08)

Dev (39): demos 0–1 of 11 LIBERO files (fetched with `--strip-images`: `libero_goal` bowl-on-plate,
cream-cheese-in-bowl, wine-bottle-on-cabinet, turn-on-stove; `libero_spatial` two black-bowl tasks;
`libero_object` salad dressing; `libero_10` STUDY_SCENE1 book, LIVING_ROOM_SCENE2 cheese+butter;
`libero_90` KITCHEN_SCENE3 moka pot, KITCHEN_SCENE5 drawer), DexMimicGen demos 0–3 of threading,
three-piece assembly and transport (state-only samples read by HTTP range requests,
`data/raw/dexmimicgen/range_samples/provenance.json`; generated variants), robomimic Transport `ph`
demos 0–4. Held-out (33, run once): LIBERO demos 10–11, DexMimicGen demos 500–501, Transport
100–104. Regression sets: the ManiSkill dev set of the ManiSkill loop (100), robomimic Can and
Square `ph` 0–4, Lift 0–4. Runs in `runs/eval/reach/` (`base-*` = HEAD `a771755`). K / P (K & P):

| set | baseline | final |
| --- | --- | --- |
| LIBERO dev (22) | 5 / 5 (4) | 14 / 5 (4) |
| DexMimicGen dev (12) | 1 / 1 (0) | 1 / 1 (0) |
| Transport dev (5) | 0 / 0 (0) | 0 / 0 (0) |
| LIBERO held-out (22) | 6 / 6 (5) | 16 / 5 (5) |
| DexMimicGen + Transport held-out (11) | 0 / 0 (0) | 0 / 0 (0) |
| ManiSkill dev (100) | 48 / 35 (28) | 68 / 38 (37) |
| robomimic Can + Square 0–4 (10) | 9 / 2 (2) | 8 / 1 (1) |
| robomimic Lift 0–4 | 5 / 5 (5) | 5 / 5 (5) |

LIBERO per task, K / P (K & P) of 2, dev baseline → final; held-out baseline → final:

| task | dev | held-out |
| --- | --- | --- |
| goal: put the bowl on the plate | 0/0 (0) → 2/0 (0) | 1/0 (0) → 2/0 (0) |
| goal: put the cream cheese in the bowl | 0/1 (0) → 2/0 (0) | 1/2 (1) → 2/0 (0) |
| goal: put the wine bottle on top of the cabinet | 0/0 (0) → 0/1 (0) | 0/0 (0) → 2/1 (1) |
| goal: turn on the stove | 2/2 (2) → 2/2 (2) | 2/2 (2) → 2/2 (2) |
| spatial: black bowl between plate and ramekin | 0/0 (0) → 2/0 (0) | 0/0 (0) → 2/0 (0) |
| spatial: black bowl on the cookie box | 1/0 (0) → 2/0 (0) | 0/0 (0) → 2/0 (0) |
| object: salad dressing in the basket (floor) | 0/0 (0) → 0/0 (0) | 0/0 (0) → 0/0 (0) |
| 10: STUDY_SCENE1 book into the caddy | 0/0 (0) → 2/0 (0) | 0/0 (0) → 2/0 (0) |
| 10: LIVING_ROOM_SCENE2 cheese and butter in the basket | 0/0 (0) → 0/0 (0) | 0/0 (0) → 0/0 (0) |
| 90: KITCHEN_SCENE3 moka pot on the stove | 0/0 (0) → 0/0 (0) | 0/0 (0) → 0/0 (0) |
| 90: KITCHEN_SCENE5 close the top drawer | 2/2 (2) → 2/2 (2) | 2/2 (2) → 2/2 (2) |

The floor (`libero_object`, grasp center z 0.12 m) and living-room (z 0.45 m) tasks stay below
Reachy's 0.46 m reach band. ManiSkill: PickCube teleop
6/10/6 → 10/10/10, StackPyramid 7/7/6 → 10/9/9, PegInsertionSide K 3 → 6, PullCubeTool K 0 → 4,
LiftPegUpright K 0 → 3, PullCube 6/4/3 unchanged. robomimic: Can demo 3 loses P (the can lands
7.1 mm deep in its bin, the known source-relative borderline) and Square demo 2 loses K (a 75° tilt
chosen on keyframes, 8 mm position residual between them). Placement takes 9.7 → 14.7 s per dev
episode (12.3 → 16.5 held-out, 5.9 → 7.1 ManiSkill); retargeting 28 → 32 s.

Root causes (dev, measured):
* *Table edge holds the base back.* LIBERO kitchen tables are a 0.85–0.90 m slab, inside the tripod
  column's band (0.30–0.95 m, radius 0.14 m): the base axis stays ≥ 0.19 m behind the edge, the
  shoulders (z 1.166 m) 0.2 m behind the Panda mount. 102 of 105 placement candidates overlapped
  the table, the 3 feasible ones were 0.2 m behind the mount, and the placement depended on
  Nelder-Mead finding the edge (bowl-on-plate: failing targets 0.61–0.65 m from the shoulder, moka pot
  0.64–0.65 m with the elbow straight; Reachy's arm reaches 0.62 m). With projected candidates the
  failing targets are ≤ 0.61 m away and sideways stances are chosen (yaw ±0.8–1.57 rad).
* *Envelope tables.* LIBERO's study table is the enclosing box of its collision geoms, a block from the
  floor to 0.88 m; its scene geometry is a 0.81–0.89 m slab: every candidate overlapped by 0.1–0.16 m
  (K `base footprint overlaps` at frame 0 in both book demos).
* *Wrist limits in holds.* With reach fixed, the remaining LIBERO failures are hold frames with the
  wrist roll and pitch both at the ±30° limit (minus the 0.045 rad margin): top-down grasps
  (approach z −0.95) carried at 0.95–1.3 m, near shoulder height. Of 2·10⁵ sampled right-arm
  postures (joint margin, self-clearance ≥ 9 mm, grasp center ≥ 0.15 m in front), a top-down grasp
  center (approach within 25° of vertical) lies at z 0.9–1.0 / 1.0–1.1 / 1.1–1.2 m in 353 / 85 / 4
  samples and never higher; approach 35–57° from vertical: 2233 / 1435 / 658 (179 at 1.2–1.3 m);
  horizontal (within 17°): 2259 / 3589 / 4770 (5199 at 1.2–1.3 m). Cold
  multi-start IK (60 seeds) at the failing frames finds the same residual: not an IK branch problem.
  Tilts to ±90° fix them where the arm stays clear of the scene: without the arm screen, dev ManiSkill
  went 68 / 31 (31), with it 71 / 34 (34) (the earlier rejection of ±90°, +8 K −10 P, was the
  forearm in the objects). LIBERO: a 75° tilt put the forearm into the wine bottle beside the
  cheese; with the screen it does not.
* *Pushes.* ManiSkill PullCube (no grasps): steep tilts took P 4 → 1, projected placements 4 → 0
  (the arm stalls against the pulled cube; elbow error grows to 0.7 rad); both are off for hands /
  episodes without grasp segments.
* *Bimanual.* DexMimicGen tables (top 0.75–0.80 m) hold the base at x −0.59 m: threading and
  assembly need both hands 0.63–0.69 m from the shoulders (two Pandas 0.5 m apart reach 0.85 m), and
  the remaining frames are wrist-limited holds; robomimic Transport needs 0.15–0.24 m more reach
  for the payload and base assistance is rejected (footprint −19 mm against the bins). Arm
  assignment by side hints is not the cause (the hands stay on their own sides).

Rejected: a 2 cm placement footprint margin (dev K 15 → 15, P 5 → 3); 16 keyframes (+1 K); 10 IK-scored
candidates (± 0); projecting only candidates within 60° of the source robot's side (ManiSkill
71 / 34 → 65 / 33); a finger screen against support boxes for tilts > 45° (meant for the 18 mm cream
cheese, whose 75–90° grasps pass K and fail P: dev + ManiSkill + robomimic K 91 → 86, K & P 42 → 40,
P equal; the support depth did not separate the cheese from PickCube grasps that pass).

Remaining failure modes: LIBERO black bowls pass K and fail P (`grasp_drift` 7–25 mm / 0.04–0.43
rad, `carry_contact` 0.65–0.92: the source Panda pushes the bowl 1.6 cm while closing on its rim and
Reachy's pads acquire it 0.4 s after the label starts); thin flat objects with steep tilts (cream
cheese: K passes, P loses 1 dev and 2 held-out episodes: no grasp or forearm contact); placements
above the shoulder (wine bottle onto the cabinet, 1.3 m); hold orientations beyond the wrist
(moka-pot handle reached sideways); targets below 0.46 m (LIBERO floor and living-room scenes);
bimanual reach (above).

### Tier-P physics loop 3: rims, flat objects, drawers (2026-10-08)

Dev (40, fixed before any change): LIBERO demos 0–1 of the 11 local files, RoboCasa OpenDrawer and
PickPlaceCounterToCabinet episodes 0–3, robomimic `ph` Can and Square demos 0–4. Held-out (38, run once at
the end): LIBERO demos 10–11, RoboCasa episodes 5–7, Can and Square demos 100–104. Regression: robomimic Lift
0–4, ManiSkill PickCube teleop 10, StackPyramid/PushCube/PokeCube/PullCube `traj_0`–`4`. Runs in
`runs/eval/pq3/` (`base-*` = HEAD `cecb3de`; `final2-dev`, `final-ho`, `final2-reg`). K / P (K & P):

| set | baseline dev | final dev | baseline held-out | final held-out |
| --- | --- | --- | --- | --- |
| LIBERO (22) | 14 / 5 (4) | 13 / 11 (9) | 16 / 5 (5) | 15 / 8 (8) |
| RoboCasa OpenDrawer (4 / 3) | 4 / 0 (0) | 3 / 1 (1) | 3 / 0 (0) | 3 / 1 (1) |
| RoboCasa PickPlaceCounterToCabinet (4 / 3) | 0 / 0 (0) | 0 / 0 (0) | 0 / 0 (0) | 0 / 0 (0) |
| robomimic Can (5) | 5 / 1 (1) | 5 / 2 (2) | 4 / 1 (1) | 4 / 0 (0) |
| robomimic Square (5) | 3 / 0 (0) | 3 / 0 (0) | 4 / 1 (1) | 4 / 1 (1) |
| **total** | 26 / 6 (5) | 24 / 14 (12) | 27 / 7 (7) | 26 / 10 (10) |

LIBERO dev per task (K & P, baseline → final): black bowl between plate and ramekin 0 → 2, black bowl on the
cookie box 0 → 1 (demo 0 passes P, fails K by 0.0625 rad on 15 frames of its now placed landing), cream cheese
in the bowl 0 → 2; bowl on the plate, book, wine bottle, moka pot, floor and living-room tasks unchanged.
Held-out gains: cream cheese 0 → 2, cookie-box bowl 0 → 1, OpenDrawer 0 → 1; losses: Can demo 103, bowl-on-plate
demo 10 K (0.41 rad rotation residual while held), Square demo 101 K. The held-out Can now misses
`task_final_pose` in all 5 (3.8–10.8 cm, before 1 of 5) and the bin depth in 1 (before 3): opening at the
command releases the thrown can earlier. Gating the earlier release on the held object being at rest (found
on held-out, so not an independent result) lost 2 dev episodes (Can demo 3, cream cheese demo 0) and was not
kept (`runs/eval/pq3/x-restgate-dev.jsonl`). Regression (35): 34 / 32 (31) → 34 / 33 (32): Lift 5 / 5 (5),
PickCube 10 / 10 (10) unchanged, PushCube traj 3 gains P. Seconds per episode (retarget + physics, one process): cream cheese 7.0 → 9.0,
cookie-box bowl 12.7 → 11.5, Can 6.8 → 8.0; the exact clipped span is vectorized over parts and plane
triples (a 40-part bowl's offset screen took 91 s per episode before).

LIBERO dev by step (22): baseline 14 / 5 (4); oriented boxes + release at the open command 14 / 7 (6); held
width 14 / 8 (7); plate rims are not insertions 13 / 9 (7); support lift 13 / 11 (9). Drawer orientation
(OpenDrawer dev 0 → 1 K & P, 8 episodes incl. 8–11: 0 → 1).

Root causes and changes (dev, measured):
* *Envelope contact width (tier-P measurement).* The LIBERO bowl walls are 1.4 mm box geoms turned in the
  bowl; their body-frame AABBs are 13–29 mm thick, so `closing_extent` gave a 37 mm contact width where the
  squeezed pads measured 5.2 mm. The release rule then never saw the release (finger reference 0.2–0.52 rad
  below the 0.89 rad "contact angle"), and the bowl tilting 18° on the plate as the hand opened was counted as
  carry drift (12 mm / 0.32 rad, `carry_contact` 0.65; the carry itself drifted 0.4 mm / 1.3°). robosuite
  `boxes` parts now also record their oriented box (`obb`: the geom's own box, exact for box geoms) and
  `closing_extent` / `contact_angles` use it (12 mm; thresholds and gates unchanged).
* *Release after the hand moved.* The ramp started at the end of the source-width plateau, 0.3–0.4 s after the
  open command, while the dwell-stretched retreat already lifted the hand: the rim was dragged 4 mm / 3°.
  The ramp now starts at the last held row (local maxima less than the squeeze + 0.01 rad above the start do
  not end it).
* *Pried fingers.* Operators press the bowl onto the plate after it lands; the rim pries the Panda's fingers
  from 4.2 to 11.8 mm under a closed command, and Reachy following that width let the landing bowl turn
  4–9° in its pads. Held width ≤ running minimum + 2 mm (0.5 mm: cookie-box bowl demo 1 2.2 → 3.2 mm, rejected).
* *Drops onto plates treated as insertions.* The source opens 7–9 mm above the plate; `_surrounded` saw the
  plate rim (15 mm above the bowl's bottom, bowl 64 mm) and kept a drop (8 mm / 8° while the pads opened).
* *Flat objects.* The Panda holds the 18 mm cream cheese 9 mm above the table; Reachy's pads reach 19–21 mm
  past the grasp center, so the source frame's fingers sink 6.3–7.0 mm into the table, and the 75° tilt the
  wrist preferred passed the finger screen (depth relative to the best candidate): Reachy's hand stalled on
  the table 17 mm above its reference and closed beside the cheese (no carry). The support lift (12 mm,
  tilt −30/−45°) picks it in all 4 dev and held-out episodes. Lifting only where the source frame itself sinks: lifted tilts beat untilted grasps in
  StackPyramid (P 4 → 0) and a 101° turn sank where the source did not (Lift demo 3, 3.3° drift); lifting
  for set-downs moved the book (caddy insertion, 23 mm) and lost its K.
* *Drawers.* The handle is no manipulated object and the drawer box (`stack_02…`, 0.38 N of damping at
  0.04 m/s) is a static fixture track, so the hand was free (orientation weight 0.06, 1.2 rad from the source)
  and the hand pressed into the drawer front with 10–62 N during the pull (160 N before it); the
  arm lagged 2–6 cm and slipped off the handle (drawer 13 cm short). Articulated fixtures with box geometry
  now count for the orientation ramp, at distance 0 while their joints move after the hand reached them.
  Dev `tcp_tracking` failures 4 / 4 → 1 / 4, `joint_margin` 3 / 4 → 2 / 4.

Rejected: narrowing the approach to the part between the pads at the grasp (the Panda slides over the rim
along the pad width and pushes the bowl 1.6 cm while closing; narrowed fingers landed on the rim: bowl-on-plate
demo 0 lost the bowl); re-straightening approaches along the tilted grasp frame's axis (LIBERO dev K & P −2,
Can −1).

Remaining failure modes: bowl-on-plate (dev demo 0: the source hand slides in over the rim along the pad
width; Reachy's open mimic finger, 27.5 mm thick and reaching 19–21 mm past the grasp center, strikes the upper
wall and tips the bowl 27° before the grasp; demo 1: the bowl turns 3.8° while pressed onto the
plate); drawer pulls: wrist roll/pitch at the IK margin (0.045 rad) dip 0.03–0.07 rad under the pull
(`joint_margin`, 2 of 4 dev, 2 of 3 held-out), and the hand slips off the handle in dev episode 3 (which now
fails K: 2.25 rad rotation residual); PickPlaceCounterToCabinet fails K (base held off the counter by its
footprint, 5–19 cm residuals); Can: the source throws the can
14–22 cm into the bin and the landing depth (3.3–5.5 mm) exceeds the 20 Hz-sampled source reference + 1 mm
(no change to the reference); Square: the nut pivots 4–5° about the handle grasp while carried and is pushed
onto the peg (`grasp_drift` 5–68 mm), and dropped nuts miss the peg (`task_final_pose` 11–36 cm); LIBERO
book, wine bottle and moka pot (`grasp_drift`, wrist limits).

## Scene meshes

Episodes must store the meshes of their scene: the policy builds a surface feature map per
component (reachy-agent `policy/build_map.py`: each body's textured visual parts in its local
frame, rendered with MuJoCo, DINO features sampled at surface points; the table keeps its upward
top faces) and assembles it at training time from per-frame component poses (`body_poses`,
xyz + wxyz, world) of every component including Reachy's links. Layout: docs/schema.md,
*Scene components and asset library*. Code: `reachy_retarget/meshes/`.

**Family policy** (`sources.registry.MESHES`, the single place): `available` for robomimic,
MimicGen, LIBERO, DexMimicGen (robosuite MJCF per demo + pinned asset archives), RoboCasa
(recorded kitchen MJCF + RoboCasa asset archives), BiGym (replay-record MJCF + exported assets)
and ManiSkill (primitive scenes); `pending` for RoboVerse and MobileManiBench; `excluded` for
BEHAVIOR (encrypted object assets) and MolmoBot (no per-frame object poses). `build` refuses
every family that is not `available` before reading anything. Within an available family an
episode whose scene cannot be resolved (adapter `state_route` kinematic-only, missing collision or
visual mesh or texture files) is not written: its build record says `status: "excluded"`,
`excluded: "no_meshes"` with the reason, and `report` counts it under `excluded`.

**Extraction from a MuJoCo `SceneRef`** (`meshes.mujoco_scene`): the scene is prepared as for tier P
(robot, mocap and declared inactive bodies removed, assets pruned and resolved), compiled without
textures, and read from the compiled model so that mesh `scale`, `refpos`/`refquat` and MuJoCo's
re-centring are applied exactly. Components are rigid groups (a body with a joint or a world child
plus its joint-less descendants; world-body geoms form `worldbody`). Visual parts are the visible
geoms that do not collide or lie in one of the scene's visual render groups (robosuite family and
RoboCasa: group 1, BiGym: group 2; the robosuite floor collides and renders); a component without
any falls back to its visible colliding geoms (BiGym floor, ManiSkill primitives). Collision parts
are the colliding geoms: primitives with their size, meshes stored like visual meshes (MuJoCo
collides with their hull). Meshes go to the library as `.msh` with compiled normals and texture
coordinates, visual primitives also get a generated surface mesh, textures keep their recorded
file bytes, builtin textures their parameters, materials their compiled values.

**Selection rule** (all families; it matters for RoboCasa kitchens): tracked objects and tracked
articulated parts always; every other rigid group (static scene parts, untracked free bodies,
untracked articulated parts) and every world-body geom when a geom's world AABB in the initial
state is within 2.5 m of the robot path (Reachy's base on the floor, its TCP and head positions and
the tracked object positions over the episode); omitted groups are listed with their distance in
`/scene` info. A robomimic table scene keeps everything but the far arena walls; a RoboCasa kitchen
keeps about 120 of its rigid groups.

**Poses**: tracked objects from the retargeted (resampled) source tracks; untracked moving groups by
forward kinematics of `SourceEpisode.scene_qpos` resampled onto the episode clock through the
episode's `source_time` (linear for hinge/slide, slerp for free joints; ManiSkill without
`scene_qpos`: the tracked free objects and articulations drive the kinematics); static groups
constant; Reachy links by FK of `q` over all URDF links with geometry (`reachy/<link>`). With
tier P, `physics_poses` gives the same components on the rollout clock from the rollout `qpos`.
The kinematics of the tracked objects is checked against their tracks
(`info.track_vs_scene_state`, ~1e-17 m on robomimic).

**Reachy links** (`meshes.reachy_links`, as reachy-agent `simulation/prepare.py`): the vendored
COLLADA visuals are parsed here (no trimesh/pycollada; `instance_node`/library nodes, node
transforms, units, per-corner normals), split per material into parts in the link frame with the
URDF visual origin and scale baked in; colours from the URDF material when named (antennas
`neckwhite`), else the COLLADA diffuse colour (raw effect values recorded); parts with fewer than 4
distinct vertices (single-triangle patches MuJoCo rejects) are skipped and listed. Collision parts
are the tier-P ones (URDF primitives, hulls of collider-mesh components). Reachy's 161 mesh files
(7.0 MB) enter each library once.

**Memory and size.** Assets are written once per output root and deduplicated by digest inside a
job (`meshes.library.open_library` keeps one writer per library directory); the extraction model is
compiled without textures and released after the episode. Scene extraction adds 0.05-2 s and about
0.3 GB transient memory (RoboCasa) per episode, below each family's existing retargeting / tier-P
peak. Component poses are stored as float32 (about 1 um at 10 m).

## Episode storage (`reachy-retarget-episode-v2`)

One HDF5 file per episode plus a dataset `index.parquet` and the dataset's asset library
`assets/` (scene meshes and textures, *Scene meshes*). See `docs/schema.md`.
