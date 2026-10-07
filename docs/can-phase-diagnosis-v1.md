# Can failures separated by measured phase

The new Can cohort remains 22 independent source episodes. Geometric search
admits ten sources; five have passed original-model physics in at least one
declared closure candidate. Repeated policies and chunks are not additional
demonstrations. The fixed recipe remains a separate 0/22 result. This document
explains the next corrections; it does not promote pending experiments as passes.

The [evidence JSON](can-phase-diagnosis-v1.json) includes pinned source provenance,
complete failed-source phase summaries, recorded collision-surface gaps and
artifact paths. Every comparison retains object geometry, mass, friction, source
state, simulator settings and physical validation thresholds.

## Original carry and inserted rotation have different failures

| Source | Phase of first relevant failure | Evidence |
|---|---|---|
| `fe117e32c6f13e7ae68c4be4` | Original carry | Can/bin1 wall contact begins at 3.696 s, before bilateral grasp loss at 3.710 s. Peak wall normal force is 18.16 N. |
| `8ee64ec58eb602a29bbc0827` | Original carry | Wall contact begins at 4.976 s, before grasp loss at 4.990 s; peak normal force is 17.69 N. |
| `47cc92a1a140ab580b44c6c1` | Inserted traverse | Rotation drift first exceeds 3 degrees at 6.82 s, with both fingers contacting and zero measured environment support. Original carry stays below 1.38 degrees. |
| `6e13ca00e7023c648b2d22f8` | Inserted traverse | Rotation drift first exceeds 3 degrees at 6.73 s, also unsupported with bilateral contact. Original carry stays below 2.90 degrees. |

The first two cases have original-carry TCP errors of approximately 47 and
39 mm. Their robot links do not collide with the environment: the carried Can
hits the bin rim because actual motion lags the reference. Exact actuator
prefix replay reproduces recorded qpos and qvel with zero error. Contact
evidence is saved under `diagnostics/can-carry-contact-prefix-v1/` on the PVC.

`source_prefix_velocity_feedforward` applies bounded servo velocity compensation
only before inserted placement starts, with a 100 ms taper at both ends. It
does not change reference poses or source timing and is zero throughout the
feedback-controlled placement. In the first completed matched comparison,
`fe117e...` peak original-carry TCP error falls from 45.66 to 25.42 mm at gain
one. The Can reaches its destination and remains stable for 5.79 s, but the
attempt still fails: 19.66 mm translation drift and 3.566 degree rotation drift
exceed the unchanged grasp limits.

The remaining contact loss starts at 4.12 s during original carry. Exact replay
over 3.9–4.7 s contains only the two finger/Can contact pairs, with no Can/fixture
contact. The next matched stronger-closure comparison can therefore address
free-carry grip slip. Slower inserted rotation would not repair this source.
The regression source `3765ca...` passes all physical gates and independent
audit at prefix gains zero, 0.5 and one. At the first snapshot only five of 24
attempts had physical results; 19 interrupted attempts failed on existing output
directories and require separately identified infrastructure retries.

For the two rotation-only cases, `loaded_rotation_speed_rad_s` declares a
separate 0.8/0.6/0.4 rad/s experiment. It dilates only the loaded traverse.
Geometric IK knots, original prefix/suffix and unloaded return remain unchanged;
all current reference frames are admitted again. Longer contact may increase
creep, so a lower speed is not assumed to improve physical success. Reports
record the requested cap, actual angular peak and phase durations. The default
0.8 behavior remains the baseline.

## Support is measured near the intended destination

The earlier controller could mistake the dropped Can's reaction force on its
original bin floor for placement support. Final task gates already rejected
these failures. The new opening guard additionally requires measured object
position within 40 mm of the intended placement and an acquired bilateral grasp
within the previous 200 ms before first support. Once verified, support may
unload the fingers during the 120 ms debounce. Missing position or grasp history
fails closed. The independent audit verifies recorded previous-interval object
position, grasp history and opening authorization; force values themselves are
not independently recomputed by that additional check.

The gain-zero `fe117e...` replay now stays closed after the source-bin drop. Its
support and task gates correctly fail, while the independent control audit
passes. This guard corrects release authorization; it does not cure an earlier
grasp failure.

## Measured placement clearance

For `aec964373b1b75854938d701`, stronger closures maintain grasp but stop just
above the bin2 floor. The original compiled collision mesh and recorded qpos
give these end-of-hold bottom clearances:

| Closure | Bottom minus support surface | Maximum grasp rotation |
|---|---:|---:|
| `contact_002` | −0.209 mm | 3.260 degrees |
| `contact_005` | +0.899 mm | 2.140 degrees |
| `contact_010` | +2.623 mm | 2.015 degrees |
| `contact_020` | +3.686 mm | 1.711 degrees |

These are read-only geometry measurements from actual recorded simulation
states, not new physical executions. The existing maximum extra descent is
2 mm. A 4 mm candidate covers the smallest measured gap; 6 mm covers all three
nominal gaps with a small margin. Each candidate still needs complete IK and
collision admission and a robot-actuator rollout. Measured support stops the
descent, and no moving-object state is assigned. Descent and angular-speed
experiments remain separate so their effects can be identified.

## Target proposal before full IK admission

`grasp_approach.propose` saves a vertical open-hand target prefix before base/IK
admission. This prevents an already-repairable original prefix collision from
blocking the solver before the repair can be considered. The proposal certifies
no joint path, invalidates inherited admissions and preserves exact initial,
closing and closed-suffix targets plus all source clocks and object references.
Full base/IK admission, existing `grasp_approach.prepare` checks, retiming and
physical validation still follow. An offline regression confirms that an open
finger/object collision remains a rejection after this reordered proposal.
