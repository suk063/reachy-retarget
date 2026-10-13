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
     every Lift and Can grasp. Three refinements (`targets.grasp_labels`): a closed hand whose pad separation is below half the object's extent along the
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
     9 mm handle, ThreePieceAssembly pieces are 34–40 mm voxel arms of 102–160 mm envelopes;
     with envelopes and travel-only widths, 38 of the 105 closed-gripper runs of the MimicGen dev
     demos had no grasp label, so no squeeze and no grasp-phase checks). An `aabb` closed run that carries the
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
     placement score penalizes arm links inside scene boxes, the *arm screen* of step 3); `theta` = the rotation that lays the closing
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
   * *Pad coverage and support lift* (`targets.pad_coverage`, `targets.support_lift`): offsets
     whose pads hold less than 80 % of the closing extent the source frame's pads hold are dropped.
     When the source grasp frame's fingers sink more than 2 mm into support/fixture boxes during the
     pick-up (until the object has moved 1 cm), each candidate retracts the grasp center along the
     source approach axis by its own depth + 2 mm (≤ 15 mm, reduced while the coverage would fall below
     80 %): a fourth offset element `lift_m`, constant in the hand frame, so the object path is unchanged.
   * *Free orientation away from grasps* (`targets.orientation_weight`): the TCP orientation is
     strict inside grasp windows (segment − 0.5 s … + 0.3 s) and its weight decays to 0.05 over
     1 s outside them. Outside grasp segments the weight is further capped by a proximity ramp
     (`contact_strict_distance` 5 cm → `contact_free_distance` 12 cm, grasp center to the nearest
     manipulated object or articulated fixture with box geometry, at distance 0 while its
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
     geometry (with centre offset) are both used. A union of `boxes` (ManiSkill PullCubeTool L tool,
     PlugCharger charger and receptacle, PegInsertionSide hole box) is seen by
     `box_geometry` as its enclosing box (an envelope, like an `aabb`: footprint, grasp
     symmetry and contact width), while hand–object distances and the finger screen use the
     exact parts (`footprint.box_parts`): the envelope of an L covers the empty corner beside
     the handle, where the fingers of a handle grasp are. Teaching `box_geometry` the kind
     keeps one geometry record per object (adapters need not duplicate it as an `aabb`).
   * robosuite world-child bodies without a joint of their own are fixtures through their static
     geoms, also when articulated parts hang below them (MimicGen Kitchen/HammerCleanup tables
     carry stove buttons and a drawer; without their static geoms the base stood 6–8 cm inside
     the table).
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
     constraint (static objects + scene geometry).
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
   (0.4 Nm at kp 8, about 8 N per pad). A 0.15 rad squeeze made the cube creep 4–7° in the hand,
   and Reachy 2's grippers are current-limited to about 0.5–0.7 Nm (reachy2_core `grippers.yaml`,
   0.4 A on XM-series Dynamixels), so the squeeze is not raised; the simulated finger actuator keeps
   reachy-agent's ±2 Nm range.
   *Held width* (`grasp_width_rise`): while a grasp label is set the source width rises at
   most 2 mm above its running minimum (a bowl rim pried open as the operator presses it down is not
   a release).
   *Release* (`targets.release_starts`, `release_ramp`, on the output clock): from the last held
   row (the source's open command, `release_at_command`; the source-mapped angle stays on its
   post-grasp plateau 0.2–0.4 s longer while a Panda ramps out of its squeeze and the hand already
   retreats) the fingers open at their 3 rad/s speed limit (× `velocity_scale`) toward the source's next
   opening peak, never below the source-mapped angle (`extra["retarget"]["release_ramp"]`).
   *Placed drops* (`targets.place_labels`, derived label, `extra["retarget"]["placed_drops"]`):
   when the source object falls ≤ 4 cm away from the hand after the release and settles within
   0.6 s, the grasp label extends to the row where it has settled (≤ 3 mm and 0.05 rad from its
   rest pose); the hand carries it down object-centrically, keeping the squeeze it had at the source
   release (the source fingers open during the fall; following them left the cube unsqueezed as it
   landed), and opens from the last held row at the finger speed. Skipped when Reachy's
   fingers would sink into scene boxes on the way down or scene geometry stands within 1 cm
   beside the settled object above half its height (insertions: the coffee pod into its holder,
   objects into bins; a plate rim 15 mm up a 64 mm bowl is not one).
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
  navigation episodes end within 3 cm of their base reference (RoboCasa's own success region lies
  around a target the recording does not store), joint limits, speed limits,
  self-clearance ≥ 9 mm, base footprint clear of static scene geometry, grasp-phase
  hand–object relative pose consistency. Failures are saved with reasons.
* **P** (sources with a MuJoCo scene): the source robot is removed from the source
  scene, Reachy is inserted, objects are free bodies, position servos track the
  retargeted joints, and only actuator commands drive the rollout. Gates follow the
  physical thresholds from the previous phase (`legacy/reachy_retarget/physical_gates.py` at git
  `fea7ea9`) plus a task predicate and actuator-only replay reproducibility.
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
    thresholds) both gates judge the pose after the hold (`rule = "after_hold"`). Otherwise
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
    `articulation_final`).
  * *Floor bodies*: Reachy's wheels are excluded against every static body carrying the floor (a
    plane at z = 0, or a horizontal box wider than 1 m whose top is at z = 0: BiGym `floor`,
    RoboCasa `floor*_room_main`), as they already were against `world`. Wheel–floor friction dragged
    the planar base joints (stick-slip): BiGym DishwasherLoadCups measured base 0.86 m/s and 2.5 rad/s
    against a 0.44 m/s / 1.2 rad/s reference, then `arm_speed` 2.1 rad/s and `tcp_tracking` 7 cm;
    with the exclusion all three gates pass.
  * Object–environment contacts during the settle phase (objects released from the
    source's initial state; robosuite cubes start ~1 cm above the table) are reported, not
    gated.
  * *Release at pad contact.* For `grasp_drift` and `carry_contact` a hand is closed on object o
    while its finger reference is below the 0.5 opening angle or the episode's grasp label is set
    (large objects stall the fingers above the 0.5 opening point), and only while that reference is
    at or below the angle at which its pads touch o on both sides: `gripper.width_to_angle` of o's
    extent along the closing axis inside the pad prism (`targets.closing_extent`, on the measured
    grasp-center and object poses; exact per part of a union of boxes, using each part's oriented
    box `obb`; NaN, nothing between the pads, adds no condition). Commanded wider, the object is
    released and judged by `task_final_pose`, `objects_at_rest` and
    `object_environment_penetration`; otherwise a placed cube rocking as the pads open, or an object
    falling from an opening hand, would count as in-hand drift. Every carry records its release time,
    rule (`pad_contact_angle` | `closed_point`), command, measured finger, contact angle and width in
    `metrics.grasps[...].releases` (`PhysicsConfig.grasp_trace` stores the per-row traces).

## Quality evaluation

`python -m reachy_retarget.evaluate --family robomimic --path <hdf5> --demos 0-19 --physics
--out runs/eval/<name>.jsonl [--write DIR] [--jobs N]` retargets each demo, runs tier K and
(with a source scene) tier P, prints per-episode reasons and an aggregate table, writes one
JSON line per episode, and with `--write` stores every episode (failures included) plus
`index.parquet`. `python -m reachy_retarget.report` buckets failure reasons by kind
(`report.reason_kind`: the tier-P gate name, or the K reason with sides, object names and numbers
stripped, e.g. "TCP rotation residual"), per tier, for the first and for any reason of each
episode.

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
ManiSkill (primitive scenes) and RoboVerse (CALVIN desk/block/floor scene from CALVIN's assets, RLBench
primitive scenes) and MobileManiBench (PartNet-Mobility object, support stage and ground; room omitted);
`excluded` for
BEHAVIOR (encrypted object assets) and MolmoBot (no per-step pose of free objects). `build` refuses
every family that is not `available` before reading anything. Within an available family an
episode whose scene cannot be resolved (adapter `state_route` kinematic-only, missing collision or
visual mesh or texture files) is not written: its build record says `status: "excluded"`,
`excluded: "no_meshes"` with the reason, and `report` counts it under `excluded`. Per-step object
state is required as well (`meshes.components.missing_object_poses`): an episode in which a
tracked object lacks a valid finite pose, an articulation finite joint positions, or a scene
component a pose at some step is not written (`excluded: "no_object_poses"`; checked on the source
before retargeting and on the assembled scene).

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
