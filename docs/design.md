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

* `reachy_retarget.acquire`: hash-pinned explicit downloads with a 50 GB reserve.
* `reachy_retarget.sources`: one adapter per source family. Adapters only read source
  files and produce `SourceEpisode` objects in the source's own world frame and clock.
* `reachy_retarget.retarget`: embodiment-independent mapping onto Reachy.
* `reachy_retarget.validate`: tier K (kinematic) for every episode, tier P (MuJoCo,
  free objects) when the source ships a MuJoCo scene.
* `reachy_retarget.schema`: data model, HDF5 I/O, control-mode registry.
* `cluster/`: job pool over the persistent worker pods (two jobs per pod).

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
  this), an opening in [0, 1] (1 = fully open), optional width in metres and an optional
  `side_hint` (`left`/`right`).
* `base`: source mobile base SE(2) path or `None` for fixed-base sources.
* `objects`: pose tracks with validity masks, a role (`manipulated`, `support`,
  `receptacle`, `fixture`) and simple geometry.
* `articulations`: articulated scene joints (drawers, doors) with joint names.
* `scene`: optional MuJoCo scene for tier P.
* provenance, license, lineage (`seed`, `variant_of`), task, instruction, source success.

## Retargeting method

1. **Tool alignment.** Adapters already express effector poses at the grasp center with
   a shared axis convention, so the Reachy TCP target is the source pose times a fixed
   Reachy grasp-center offset from `{l,r}_arm_tip`. Hand–object relative poses are
   preserved by construction.
   * *Grasp labels* (`targets.source_closed`): a source hand holds an object while it is
     closed and within 4 cm of the object's box. "Closed" is inferred from the opening: below
     0.5, or stalled (|rate| ≤ 0.25/s) more than 0.1 below the episode's open level, and not
     opening. A gripper that closes on an object stalls at the object's width (the robomimic
     Panda holds the Lift cube at opening 0.52, the Can at 0.62), so a fixed threshold missed
     every Lift and Can grasp.
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
     equal cross-section half extents ±10 %; recorded with the reason). Offsets whose fingers
     would sink into scene boxes along the path more than 3 mm beyond the best offset are
     discarded before any IK (quarter turns are mostly rejected here: the source approach is
     off-centre along the other axis and a finger would land on the cube).
   * *Object-centric carrying* (`targets.object_centric`): when the source object moves in its
     gripper by more than half the tier-K grasp tolerance during a grasp (robomimic Square: the
     nut slides 5–44 mm while pushed onto the peg), the hand follows the object pose times the
     grasp at the segment's third frame, and the correction to the source hand path decays to
     zero over the approach/retreat windows. Grasps the source holds rigidly keep its hand path.
   * *Free orientation away from grasps* (`targets.orientation_weight`): the TCP orientation is
     strict inside grasp windows (segment − 0.5 s … + 0.3 s) and its weight decays to 0.05 over
     1 s outside them. A first IK pass with these weights gives the orientation Reachy prefers
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
     base was placed inside the table.
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
  * Free bodies that no episode object refers to and that start more than 2 m from the
    workspace are removed as parked (robosuite PickPlaceCan parks Milk, Bread and Cereal
    overlapping at (10, 10, 10): the constant 0.0399 m object–environment depth).
  * Servo commands lead the retargeted `q` by each servo's `kv/kp` (0.13 s for the arms):
    `kp (q(t + kv/kp) − q) − kv q̇ ≈ kp (q_ref − q) + kv (q̇_ref − q̇)`, a PD servo with
    velocity reference, still a deterministic function of the retargeted trajectory. Fingers
    take the more closed of the led and the current reference (lead while closing, never an
    early release). `tcp_tracking` compares against the reference, not the led command.
  * Object–environment contacts during the settle phase (objects released from the
    source's initial state; robosuite cubes start ~1 cm above the table) are reported, not
    gated. Grasp gates also count a hand as closed while the episode's grasp label is set
    (large objects stall the fingers above the 0.5 opening point).

## Quality evaluation

`python -m reachy_retarget.evaluate --family robomimic --path <hdf5> --demos 0-19 --physics
--out runs/eval/<name>.jsonl [--write DIR] [--jobs N]` retargets each demo, runs tier K and
(with a source scene) tier P, prints per-episode reasons and an aggregate table, writes one
JSON line per episode, and with `--write` stores every episode (failures included) plus
`index.parquet`. Results on robomimic v1.5 `ph` (dev = demos 0–19 used while developing,
held-out = demos 100–119, evaluated at the end and not used for tuning), episodes passing K / P:

| task | baseline dev | final dev | baseline held-out | final held-out |
| --- | --- | --- | --- | --- |
| Lift | 20 / 0 | 20 / 20 | 19 / 0 | 19 / 19 (K & P 18) |
| Can | 0 / 0 | 17 / 0 | 0 / 0 | 16 / 0 |
| Square | 0 / 0 | 13 / 0 | 1 / 0 | 14 / 0 |
| Transport | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

Tier P on Can, Square and Transport is bounded by the source scenes themselves: re-stepping
the source MuJoCo model at 2 ms from its recorded states gives object–environment depths of
5.4–7.7 mm (Can dropped into its bin), 7.9–8.1 mm (Square nut on the peg / table) and
9–19 mm (Transport drops), above the 2 mm gate that is kept unchanged. 12 of 20 dev and 9 of
20 held-out Can episodes pass K and every P gate except that one. Open K failure modes: IK
branch switches between consecutive source frames (0.44–0.92 rad in 5 of the 7 failing dev
Square episodes; the time scaling keeps the speed limits but the interpolated poses miss the
target by 2–5 cm) and, for Transport, the two-robot handover (position residual median
48 mm, down from 70 mm, but the payload orientations of both hands are not reachable with
one constant grasp offset per arm).

## Episode storage (`reachy-retarget-episode-v2`)

One HDF5 file per episode plus a dataset `index.parquet`. See `docs/schema.md`.
