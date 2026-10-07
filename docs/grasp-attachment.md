# Constant TCP attachment for flat-face grasping

`reachy_retarget.grasp_attachment.apply` changes the robot hand reference by one
constant SE(3) attachment. It uses the unchanged source box collision geometry
and the actual Reachy pad collision surfaces. It does not edit object geometry,
mass, friction, contact properties, initial state, reference poses, source clock,
or gripper intent. Physical success remains a separate measurement.

## Motivation from measured PickCube contacts

The preceding PickCube attempt lifted and held the object but exceeded the
grasp-drift gate: 8.549 mm and 21.817 degrees. An exact actuator replay reproduced
the recorded qpos and qvel without error. At the drift anchor, force-weighted
contact centers were about 9 mm off the cube center laterally and 12 mm above
it. By the end they had moved to approximately 16 mm above center, with some
contact points at the 20 mm top edge. Pad normal forces peaked near 22.5 N.

The original closing axis was 12.245 degrees from the closest box face normal.
Contact normals had already aligned substantially with the box faces before
the drift anchor, so rotation at acquisition alone does not explain later
drift. The off-center load and migration toward the top edge motivate a paired
physical trial of central, face-aligned contact. They do not prove its outcome.

Contact evidence: [pickcube-grasp-contact-audit.json](pickcube-grasp-contact-audit.json).

## Calibration and interface

```python
from reachy_retarget import feasible_maniskill, grasp_attachment

prepared = grasp_attachment.apply(
    prepared, robot, attachment_output,
    align_box_axis=True, center_box=True,
)
prepared = feasible_maniskill.prepare(
    prepared, robot, ik_output, base_xyyaw,
)
```

The input is the 12-element prepared tuple returned by `frozen_plan.load`.
`apply` returns a new tuple with copied details and corrected right-hand goals.
The old joint references are temporarily stale: `joint_reference_valid=False`
and `requires_full_ik_and_collision_admission=True` explicitly require the
subsequent IK stage. A fresh output directory preserves the module snapshot,
all original references, derived references, attachment, and checksums.

At the saved pad-alignment anchor, the method chooses the closest signed box
face normal and computes the minimal rotation of the closing axis onto it.
It aligns the opposing pad midpoint with the original box center. The actual
gripper angle is solved again from the opposing pad gap and unchanged box
width; downstream grip control must use the new
`details["pad_alignment"]["inferred_contact_angle_rad"]`.

For frozen hand references `E(t)`, corrected targets are `E(t) @ A`, with the
same attachment `A` at every time. Thus the world-relative rigid motion
`E(t2) @ inverse(E(t1))` is exactly preserved. The input here is the complete
frozen, previously pad-aligned hand reference, rather than raw source TCP poses.
No frame is cropped, and no object trajectory is followed by overwriting state.
Pad calibration only initializes an isolated robot model; object reset values
are checked for exact preservation.

`single_box_geometry(model, object_body=..., object_joint=...)` verifies one
physical box geom in a rigid subtree with the declared free root joint. It
rejects articulated descendants and multiple/non-box collision geoms. Explicit
contact pairs count as collisions even if a geom's masks are zero. Offset and
rotated fixed child bodies are included in the returned root-relative box frame.
The helper does not step, forward, reset or mutate a model/state. Eligibility is
based on geometry and articulation, with no dataset or task-name whitelist.
`align_box_axis=False` and `center_box=False` expose separate declared ablation
choices; they do not modify the source object.

## Actual admission result

The source-faithful PickCube reference `8a7847a26e8fbb00783e4114` was calibrated
and checked at stationary base `[0.3, -0.653, pi]` in the existing cluster:

| Quantity | Measured value |
| --- | ---: |
| Complete frozen reference | 726 frames |
| Attachment rotation | 12.245 degrees |
| Box width / desired pad midpoint | 40 mm / box center |
| New inferred contact angle | 0.938290484884 rad |
| Previous inferred contact angle | 0.958270586164 rad |
| Maximum IK position error | 0.994 mm |
| Maximum IK rotation error | 0.009934 rad |
| Maximum environment / self penetration | 0 / 0 mm |
| Minimum robot self clearance | 117.235 mm |
| Minimum joint margin | 0.031 rad |
| Peak arm reference speed at the original clock | 1.161416 rad/s |

All frames passed the implemented IK, environment, self-collision, clearance,
and joint-margin admission checks. These static checks leave the source object
at its unchanged reset and do not certify future hand/object contacts. The
reference speed exceeds the 1 rad/s physical gate, so the separately declared
timing policy must still be applied. Measured motion, contact depth, task
completion, drift, and replay gates require an actuator-only physical rollout.

Evidence: [pickcube-grasp-attachment-admission.json](pickcube-grasp-attachment-admission.json).
Its `attachment.joint_reference_valid=False` describes the calibration stage;
the separate `admission.admitted=True` records the subsequent successful IK
stage. Neither field claims physical validation.

Artifacts are preserved under
`/mnt/reachy-retarget/workspaces/pickcube-attachment-audit-v3/{attachment,ik}`.
Two setup-only failed attempts are retained separately. The report records the
original frozen-plan hashes, source release, controller source, and verified
Reachy-agent main revision. The original input files were not changed.

## Lift and Stack development calibration

The same implementation was applied to four existing development plans from
robomimic Lift and MimicGen Stack. Their actual compiled scenes each contain one
free rigid box. No task-specific offset or replacement geometry was introduced.

| Episode prefix | Task | Minimal face rotation | Actual face width | New pad angle |
| --- | --- | ---: | ---: | ---: |
| `0cacb4` | Lift | 12.92 degrees | 41.9342 mm | 0.979769 rad |
| `2dbc8c` | Lift | 11.86 degrees | 40.5666 mm | 0.950450 rad |
| `a3cc11` | Stack | 1.97 degrees | 40 mm | 0.938290 rad |
| `b6c2b8` | Stack | 0.94 degrees | 40 mm | 0.938290 rad |

All four calibrations retained clocks, gripper intent and object references
byte-for-byte. This calibration-only audit did not re-solve IK or execute a new
physical trial. Joint references are explicitly stale until complete shared IK
and collision admission. The previously measured passive support penetration
is unchanged by a robot TCP calibration.

[box-attachment-development-calibration.json](box-attachment-development-calibration.json)
binds each frozen plan, scene, module snapshot and derived target artifact.
Five attachment tests cover rigid-motion/input preservation, rotated fixed-child
geometry, and rejection of articulation, non-box shapes and explicit-pair extra
collisions.
