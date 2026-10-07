# Can contact height and postrelease return diagnosis

The remaining 12 Can source demonstrations were checked against their unchanged
compiled collision mesh and actual opposing distal pad faces. Frozen plan
bindings and external scene assets were verified. Detailed measurements and
input hashes are in `can-height-and-return-diagnosis-v1.json`.

The convex collision surface has a straight side from object-local
−38.090 to +37.129 mm and approximately 25 mm radius. Existing calibrated grasp
centers are already 8.10–20.70 mm above the object origin. The source COM is
`[0.0589, −0.1337, 0.3584]` mm and mass is 0.0159962 kg. These values were read
from the original model, not replaced by a cylinder or another object.

The conservative geometric screen requires the complete opposing pad-face
axial projection to stay within the straight side, with a declared 1 mm end
guard. This is a proposal eligibility screen, not a changed physical gate.
At +20 mm, no tested roll/source retains full-face support. At +10 mm the numbers
of eligible sources for rolls −45°, −30°, −15°, 0°, +15°, +30°, +45° are respectively
4, 5, 8, 6, 0, 0, 0. A centroid-only section-width check misses this loss of finite
pad support near the rim.

Three paired zero-roll, fixed-base comparisons were run in node-local scratch:

| Source | Original maximum fixture penetration | With +10 mm grasp height | Scope |
| --- | ---: | ---: | --- |
|261e66 |46.544 mm |43.381 mm | Sparse full-episode screen |
|60f8d5 |11.673 mm |11.148 mm | Sparse full-episode screen |
|2447e6 |4.454 mm |4.921 mm | Complete 736-row path |

All comparisons retained source object references, clock, mesh, mass, friction
and reset state. The correction was one constant robot TCP translation derived
from the source anchor's object-local Z axis. It is not a prescribed object
motion. Raising an upright, centered grasp introduces no extra static gravity
moment about its contact line; under tilt the extra moment is bounded by
`mass × gravity × height`, approximately 1.57 mNm for 10 mm and 3.14 mNm for 20 mm.
This bound does not establish force closure, frictional stability or tolerance
to lateral acceleration. Raising also reduces available end clearance.

The fixed-base zero-roll phase audit found closed-intent fixture collisions in
11 of 12 candidates. This does not prove that every other base/roll is infeasible.
For 2447e6, the complete path instead has no prefix/carry penetration and only
eight bad open-return rows at 4.81–4.88 s. The maximum is an elbow/forearm overlap
with `source_scene_geom_9`. The current supported-placement helper inserts
phases and subsequently resumes the original suffix, so changing its admission
order alone cannot remove this collision.

`unloaded_return.propose` supplies an explicit robot-only world-Z excursion
after source release intent. It retains every source-clock/object/intent row,
all closed hand targets, the release boundary and hand orientations exactly;
it saves original robot references and invalidates prior admission. The source
evaluation hold remains stationary. A changed terminal height is optional and
explicit, never an implicit endpoint repair. Actual release remains a separate
support/contact-gated physical condition.

For 2447e6 at original base + [0, 0, 15°] and seed 9, both 30 mm and 60 mm return lifts
remove penetration across all 736 rows. The 30 mm proposal is rejected by the
unchanged additional 3 mm fixture-clearance requirement (minimum 2.370 mm).
The 60 mm proposal passes complete IK, joint margin, self-clearance, contact
depth and 3 mm fixture-clearance admission, with the original terminal target.
Retiming, inserted placement and an independently audited actuator rollout
remain required; this result is not physical success.

The pinned v25 downstream check passed the base, approach and retiming stages,
but none of the existing 45 placement candidates passed its sparse screen:
37 failed fixture clearance, seven failed IK, and one failed both. Four closest
fixture-only candidates were then examined at the identical failing TCP pose.
Increasing instantaneous base Y by 10 mm improved clearance by approximately
2.74–2.86 mm; reducing base yaw also improved it. The original bin wall has
world AABB `[.59,.61] × [−.41,.01] × [.8,.9]` m.

A bounded 12-candidate followup changed only the placement base offset: those
four targets each used +50 mm Y, +100 mm Y, or +50 mm Y with −5° yaw. This cleared
the first wall obstruction but exposed later target-bin collisions or IK
failures. No complete placement was admitted, and no physics success was added.
The full reports are `can-unloaded-return-complete-preflight-v1.json`,
`can-unloaded-return-base-gradient-v1.json` and
`can-unloaded-return-base-followup-v1.json`.

All diagnostics in this report read the PVC without writing it. New remote
outputs were kept under node-local `/tmp` on worker 4tn7w. Full evidence was also
saved locally. No dataset or source denominator changed.
All 204 files from the five generated diagnostic attempt directories, including
failed plans, are also retained in the verified local backup recorded by
`can-diagnostic-backup-v1.json`.
