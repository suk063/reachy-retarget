# Box acquisition diagnosis and robot-only correction

The audited adaptive union is now **16/28** independent Lift/Stack sources under
the declared alternative global 4 ms contact model. The original fixed-policy
result remains 9/28. Source-fidelity passes remain zero. Earlier failures and
all adaptive attempts are retained; variants are not extra demonstrations.

The fixed contact policy passed 9 of 28 additional independent Lift/Stack
demonstrations under the separately declared global 4 ms contact model. A
predeclared, adaptive two-policy recovery bank recovered three more Lift
sources (12/28 cumulative). All 38 recovery archives passed independent actuator
replay, common-schema, state/control completeness and source-clock checks.
These are **alternative-model results**, not source-fidelity passes.

The eleven remaining acquisition failures are not explained by insufficient
gripper force alone. Their original fixed-policy recordings show 21–58 mm TCP
tracking errors near acquisition, despite sub-millimetre reference IK errors.
Five Lift inputs touch the object with an open finger before closure. Four of
those have no fixture contact at the sampled closure state; their object is
pushed before it reaches the calibrated pinch. The other Lift input is blocked
by forearm/table contact. Several Stack inputs are blocked by the table or the
other cube: at one observed state, cubeB/palm penetration is only 24–44 microns,
yet recomputed opposing shoulder constraint torques are 12–18 Nm. Reference
geometry at those same measured fixture poses overlaps by centimetres.

The recorded-state evidence is in
`additional-box-acquisition-geometry-audit.json` and
`additional-box-acquisition-obstruction-audit.json`. Forces in the second audit
are recomputed by isolated FK/forward dynamics from a saved state, explicitly
not exact solved-substep force observations. The original trajectories were
separately replayed exactly. Its nominal-reference comparisons initially use
the **measured fixture poses**, not a claim of original-reset clearance.

`grasp_roll.rotate_about_pad_line` applies one constant right attachment around
the calibrated opposing pad line, bounded to ±90 degrees. Both calibrated pad
contact-point trajectories are mathematically unchanged, at every source
timestamp. Thus a palm/forearm orientation can change without moving the
intended pinch, object reference or gripper intent. `grasp_roll.apply` preserves
the previous references and invalidates previous robot admissions. It requires
fresh complete IK, environment admission and actuator-only validation. The
calibrated line is defined at the contact aperture; all other apertures still
need geometric inspection.

`geometry_clearance.RobotFixtureClearance` checks signed robot/fixture distances
without advancing or forwarding simulator state. A positive gap supplements
the existing penetration thresholds: a servo can stall at nearly zero depth.
The checker respects physical collision masks, explicit pairs and body
exclusions. It excludes the selected active-object subtree and only explicitly
named wheel/floor support pairs. Nonactive cubes, distal fingers, palms, arms
and torso bars remain checked. Conservative compiled bounding spheres reject
far pairs before exact distance queries. A distance beyond the query bound is
labeled as a lower bound, not an exact global minimum.

`geometry_clearance.prepare` inspects every current reference frame and sampled
gripper aperture envelopes, preserving original reset fixtures. The inactive
left gripper remains open. The closed interval is tied to the actual numeric
command target and calibrated contact aperture; force-control policies require
an explicit static envelope. Its default 3 mm positive clearance is an
additional kinematic admission condition, **not a relaxed physical gate**.
Every rejection is saved. This finite static check does not establish
continuous swept-volume clearance, later moving-fixture clearance or physical
task success.

The first six full-path roll candidates were all rejected before physics:
three by environment overlap or sequential IK, one by approach IK, and two
by the new positive clearance inspection. Their sampled preflight was never
counted as success. The cumulative adaptive count after that stage was 12/28.
`additional-box-geometry-v1-evidence.json` retains each complete rejection.

The subsequent centered-box correction passed all physical gates for Lift
demo9 and demo22. Its three physical archives, complete state/control fields,
source maps and independent actuator replays were audited in
`additional-box-centered-v1-audit.json`. This increases the verified adaptive
union to 14/28, retaining the original fixed-policy result of 9/28.

A separate Lift demonstration misses only the existing one-second final
stability gate (0.95 seconds). Its last half-second joint reference is exactly
constant, task and bilateral contact remain true, and final object speed is
0.207 mm/s. The diagnostic is `additional-box-terminal-hold-diagnosis.json`.
`terminal_hold.apply` permits an explicit additional 0.2-second robot hold,
preserves all prior rows exactly, and holds the source-map endpoint. It does
not create native source observations or reduce the stability requirement.
Existing placement or adaptive phase controllers are rejected unless a
separate verified extension is implemented; original interaction phases and
the added robot-only phase are recorded separately. This hold still requires
an actual actuator rollout and independent replay verification. Both declared
hold attempts have now passed: final stability is 1.15 seconds for Lift demo3
and 1.11 seconds for Lift demo21. Every numeric field of both previous
`plan.h5` and `replay.h5` files is exactly equal over the entire old prefix,
including object poses, joint states and actual controls. Only twenty final
control intervals were appended. The canonical archives and independent
actuator audits passed; see `additional-box-terminal-hold-v1-audit.json`.

Remaining Stack diagnostics also exposed a planning-order issue: inspecting
the original colliding approach before generating its replacement can reject
a useful proposal prematurely. The target-only `grasp_approach.propose`
therefore precedes full-path IK; the existing approach validation must still
run afterward. `open_prefix_correction` optionally introduces a new attachment
and explicit base offset during open gripper intent, preserving the original
initial pose/base and the entire corrected closed suffix. It does not admit a
trajectory. Two full-path Stack trials removed the catastrophic initial
branch divergence but still failed IK by 23–27 mm; those failures are retained
and not promoted.
