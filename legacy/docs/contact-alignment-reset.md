# Refocus on gripper contact alignment

The first controlled implementation and six-episode comparison are now recorded
in [translation-only pad alignment validation](pad-alignment-validation.md).

Decision on October 6, 2026, following user feedback: start from the original
trajectory-to-gripper-pad alignment problem. Pause broad trajectory optimization
as the preferred direction. Preserve all experimental implementations and runs;
do not infer that they improved the baseline or silently replace it.

## Findings from the current repository

- `retarget.aligned_goals` preserves the observed end-effector positions after
  a shared rigid scene placement and applies a constant tool orientation mapping.
- `physics.prepare_robot` creates synthetic fingertip endpoint sites at local
  `(0, 0, 0.0461)`. These sites are not annotations of the opposing pad surface.
- `contact.convert_grasp` uses the endpoint midpoint at a nominal gripper angle
  of 0.63 rad, then adds depth and height offsets. This was an approximate grasp
  calibration, not a measured pad/contact-frame conversion.
- The later `dynamics.plan` also replaced the hand path with an object-relative
  path, reconstructed the approach/retreat, inserted a closure hold, altered
  release/placement, and allowed base motion and timing dilation. The common
  optimizer then varied actuator trajectories. These simultaneous changes make
  the effect of pad alignment impossible to isolate.

An offline geometry inspection of the saved Can scene finds the midpoint of
inward-facing distal collision triangles approximately **17.892 mm shallower**
than the synthetic endpoint midpoint along TCP +Z at the same 0.63 rad opening.
The midpoint moves from TCP z=-64.463 mm to z=-46.570 mm. This is a geometric
surface proxy, not a verified semantic rubber-pad annotation or an optimal grasp.
It must not become another hard-coded offset. Pad shape, opening angle, source
contact region and object geometry all need to participate in calibration.

The inspected scene is
`runs/object-matrix/runs/dynamics/common_matrix_v2/6ad5cc23ec6d1c207da3e4b2/seed_00/6ad5cc23ec6d1c207da3e4b2/scene.xml`.
Inspection uses reset-state forward kinematics only, no physical stepping or
moving-object intervention. It changes no source assets or existing run files.

The earlier simple Can run
`runs/object-matrix/runs/contact/fixed_grasp_three_can/demo_0/result.json`
records a successful grasp/placement under its historical validator, with
0.467 mm / 1.809 degrees drift. It is a useful baseline to recover, not proof
under the newer validator. Likewise, historical 6/20 and common-policy 1/20 used
different source windows and configurations, so they do not isolate regression.

## Narrow next experiment

Freeze the same original retargeted trajectory, source interval, rigid scene
placement, base path, playback timing, orientation, gripper commands, controller
and physical acceptance gates for both conditions. Keep source objects unchanged.

1. Reproduce the uncorrected trajectory and measure where contact falls on the
   unchanged distal collision/visual geometry. Display the TCP, endpoint and pad
   surface candidates separately; quantify their alignment through closure.
2. Estimate one small translation per grasp interval from the source interaction
   region and Reachy pad geometry at the relevant opening. Express it in the
   object frame. Blend it near contact, retaining the original path elsewhere.
   Start with translation only; keep hand orientation fixed.
3. Compare uncorrected versus corrected trajectories using the same MuJoCo
   actuator-driven validator. Report pad placement/contact, penetration and slip
   separately from full-task completion. A geometric improvement alone is not
   a physical pass.

The same correction rule applies across tasks; task names must not choose hand
offsets. Parameters come from the demonstrated interaction and geometry. If a
correction cannot be reached with the frozen base and controller, report that
failure instead of silently replanning the task.

Only after isolating this effect should a demonstrated remaining slip or force
problem justify a separate, bounded dynamics correction. Whole-trajectory CEM,
new phase logic and new base planning are not prerequisites for diagnosing a
misaligned contact frame. The broad prototype and all failed samples remain
available as experimental evidence, not the default solution to this problem.
