# Grasp component diagnosis

Three original development sources were independently replayed without changing
controls, source geometry, or object state after reset. Every recorded qpos and
qvel sample reproduced exactly. These diagnostics do not amend earlier results.

* SquareNut `2aef318f4c607ad8ed448123`: the proximal finger first hits ring
  component `SquareNut_g2`; one distal finger never develops force. The native
  grasp point is inside the separate handle component `SquareNut_g4`.
* SquareNut `31bb4a33156a48823d3f6daf`: distal contacts also first hit `g2`.
  Both finger forces occur, but no bilateral lifted grasp is acquired.
* Threading `c1f6fdd7b4fa919be783b8a6`: neither finger ever contacts the moving
  object. The earlier 81.25 mm calibration section crosses the needle and its
  handle. The original handle is a 40 mm cube. A line through that handle across
  the shaft is 40 mm wide; a line along the connected shaft is 160 mm wide.
  Thus selecting an explicit grasp component and closing axis matters; reducing
  the whole object's collision geometry would misrepresent the task.

The new opt-in `grasp_region` proposal selects the smallest unchanged collision
box containing the recorded native grasp point. Three declared box axes form a
finite bank. Section measurement still includes all original intersected pieces;
unsupported widths are rejected rather than clipped. A constant TCP attachment
changes robot targets, with a 150 mm translation bound. Every original hand and
object reference and timestamp is saved. Complete IK, environment and approach
admission precede any physical execution.

`configs/feasible-component-grasp-v1.json` declares 36 development trials over
these same three sources: three axes, two base offsets, and two IK seeds. It is
not an independent demonstration count. SquareNut's separate alternative global
4 ms contact model is explicit because its original passive drop already
violates the contact-depth criterion. Threading retains its original model.
This proposal has not yet produced a physical success.

All 36 first proposals were geometrically rejected. Besides genuine fixture
collisions, their static checks exposed a shared aperture-admission bug: a
scalar initializer closed both hands, including the unused left hand, and
tested the mechanical endpoint instead of the declared positive closed target.
The empty fingertip meshes then overlapped by 4.158 mm. The initializer now
supports separate gripper values, and planning selects the controller before
checking its aperture envelope. The inactive left hand stays open. Earlier
rejections are retained; every one also had an environment failure, so fixing
this bug alone is not evidence that any of these paths is feasible. Actual
actuator-driven rollouts and their original validation thresholds are unchanged.

Exact replay evidence is in `nonbox-acquisition-<source-id>.json` beside this file
and `/mnt/reachy-retarget/diagnostics/nonbox-acquisition-v1/` on shared storage.

## Derived base motion diagnosis

A 27-configuration sparse inactive-left-arm parking probe did not clear the
SquareNut task: the dominant obstacle is the torso left parallel bar against
the table, at 141.8 mm overlap, independent of the parked elbow. The original
source robot is stationary; the existing derived Reachy base moves up to
0.76 m laterally in this episode. A declared stationary-initial base bank
therefore checks 216 complete source paths across both SquareNut sources and
the Threading source, using calibrated original object components, 27 base
positions/yaws and two arm seeds. It changes only the derived robot base/IK,
keeps all hand/object source targets and times, and requires complete geometry
and positive fixture-clearance admission before physics. The parked-arm
proposal is not adopted. Evidence is retained at
`/mnt/reachy-retarget/diagnostics/nonbox-left-park-v1/result.json`.

## Full-task base bank after storage recovery bypass

All 216 initial-base-centered stationary candidates are now accounted for as
geometric rejections, including 60 immutable node-local retries after the storage
interruption. The 60 retries all fail full-path IK; none is a physical rollout.
The complete numeric targets reveal why that bank is too narrow: initial bases
have world Y about +0.39 m, while the full corrected TCP paths span Y from
−0.393 to −0.009 m for the second SquareNut, for example. Keeping candidate
bases within 0.10 m of that initial free-space base does not cover the task.

`configs/nonbox-task-center-v1.json` records each complete corrected TCP bounding
box, its immutable numeric-artifact hash, and 216 new stationary candidates for
the same three sources/four component axes. The bank places the trajectory-box
center at explicit robot-relative forward distances 0.4/0.5/0.6 m and lateral
offsets −0.1/−0.2/−0.3 m, with yaw −15/0/+15 degrees and two existing arm seeds.
Every candidate retains original targets, clocks and object geometry, and must
pass complete IK, self/fixture collision and positive-clearance admission before
any physical trial. These are geometric proposals, not new source demonstrations
or evidence that a source is globally feasible.
