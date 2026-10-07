# Cylindrical acquisition rendezvous

The failed `fe117e32c6f13e7ae68c4be4` rollout acquired the Can with the pad midpoint 5.56 mm laterally displaced from its calibrated patch. This displacement was already present at the first bilateral contact, before the later carry drift. At the original first close command, the pad midpoint was 15.19 mm above its calibrated height and the finite pad faces extended beyond the straight wall. Holding that first-close pose would preserve the wrong geometry.

The comparison `8ee64e` rollout acquired with about 1.45 mm lateral error. These measurements come from exact replay of each recorded actuator command; both replayed qpos and qvel matched the original recording. See `can-fe117-acquisition-patch-v1.json` and `can-8ee-acquisition-patch-v1.json`. The comparison recording uses its original `contact_005` variant (0.05 rad below calibrated contact), not the subsequently successful midpoint-closure trial.

## Admitted robot correction

`cylindrical_acquisition.prepare` selects the first source row near the fixed calibrated patch before the source lift and inserted placement. It reconstructs the calibrated object-relative TCP from saved geometric axes and midpoint, without indexing an anchor from an obsolete source clock. All original arrays are retained. The runtime command clock delays closure, enters the corrected pose, waits for measured alignment, closes only after alignment, and consumes every following original row through a 100 ms exit blend. Both real time and the original source-reference map are recorded.

For the actual fe plan, the original first close is reference row 272 and the qualified row is 300, at source time 2.682613117 s. The robot-only correction is 1.691 mm and 1.976 degrees. The exit consumes rows 301–310 exactly once and then resumes the unchanged suffix. All 107 static samples pass IK, fixture, self/joint, open-hand/object, non-distal/object and finite-pad checks. Maximum proposed arm reference speed is 0.77454 rad/s. The object reset remains unchanged. The complete report is `can-fe117-acquisition-qualified-v1.json`.

Runtime guards read the actual object pose and measured TCP. They check the calibrated closing patch using the saved contact-aperture CAD, both finite pad-face axial bounds, non-distal clearance, TCP error, and per-pad summed normal force. Near contact they additionally check the current-aperture pad midpoint. Open fingers have a different midpoint, so their current midpoint is reported without falsely treating it as the calibrated closing midpoint. No object state is assigned by the adapter or guards.

The adapter binds the final complete reference, source clock, gripper intent, hand/object targets, scene, numeric artifact and implementation checksums. Rejected qualification artifacts are retained. Geometry qualification is not physical success: the opt-in actuator-only rollout, original physical gates and independent replay audit are still required.

Frozen adapter revision for this qualification: SHA-256 `fce1a8af1b98dd766cb3c716fb2596f522f048d4e03f09d4160425c7c9b3b2fc`. Its geometry dependencies came from immutable release `7531371d1ef03d9405c81060aea5c236608c09fcfb48078402fe0babb8f5cf92` during the isolated preflight. Runtime publication and physical results are recorded separately.
