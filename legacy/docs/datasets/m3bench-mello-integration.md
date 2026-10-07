# M3Bench and MeLLO: verified native adapters

Inspected 2026-10-06. The two adapters read real publisher formats and return the
same `{arrays, metadata, status, missing_fields}` contract. Both produce a
**partial source archive**, not a successful Reachy recording. Their retained
source, derived fields, missing channels, and simulation assumptions remain
separate. Primary-source URLs, revisions, native-member checksums, and measured
inspection results are in [the evidence record](m3bench-mello-integration-evidence.json).

Pose logging covers **task-related objects only**. M3Bench provides the target's
initial transform, but lacks continuous target motion and related support or
receptacle transforms. MeLLO retains task marker groups and a derived Box marker
frame whose collision-mesh calibration is unresolved. Both adapters explicitly
record `task_object_pose_coverage`; unrelated scene props are not a requirement.
The pipeline does not require image data.

## M3Bench

The publisher releases planned mobile-base and Kinova-arm paths with pick/place
instructions and scene assets. Code revision is
`97cec07e7c37cc36aec35ff79ee5d6b5db123a45`; dataset revision is
`5b543a8c93c82b16442b2c1c4a139b96bb313b59`. The publisher dataset card declares
Apache-2.0; preserve individual asset notices when collecting meshes.
[Publisher schema](https://huggingface.co/datasets/M3Bench/M3Bench),
[pinned code](https://github.com/TooSchoolForCool/M3Bench/tree/97cec07e7c37cc36aec35ff79ee5d6b5db123a45).

The verified pilot is
`pick_traj/physcene_4838/book_28_link/2024-10-03-02-53-19/23`.
Four native JSON files were inspected: config, planner request, named planner
return, and captioned trajectory. They occupy 65,171 decompressed bytes and only
15,382 compressed payload bytes. The adapter's `describe()["pilot"]` supplies
exact byte ranges, compression, CRC32, and SHA256. This avoids repeatedly reading
the pick archive's 133 MB central directory or downloading its 2.81 GB payload.
Checksums describe the selected members; the whole archive was not hashed.

The actual pilot contains 51 waypoints for ten named joints: two base translations,
base yaw, and seven arm angles. The named return and caption trajectory agree.
The object's **initial** 4x4 transform and attachment-planner metadata are present.
Physical timestamps, gripper aperture/closure commands, per-frame object motion,
and measured contact forces are absent. `source/waypoint_index` is retained;
`timestamp` is explicitly missing. Fifty-one waypoints must not silently become
51 frames at an invented sampling rate.

`describe()["fk_asset"]` provides another verified range: 3,702 compressed bytes
from `robot_urdf.zip` produce the 26,532-byte `Mec_kinova/main.urdf`. With this file
saved as `robot.urdf` beside the selected instance, normalization also derives
`source/ee_transform`. Its endpoint is **`robotiq_arg2f_base_link`**, the native
planner's gripper-base link, not a finger-pad contact frame. FK includes the
source base height and rotated arm-mount offset. A separate Pinocchio calculation
agreed across all 51 pilot frames to a maximum matrix element error of
`5.33e-15`. The source has one arm; assigning this transform to Reachy's right
hand must be recorded as an embodiment mapping, followed by pad calibration.

Two source assumptions require explicit handling:

- The inspected planner request sets the target's collision scale to
  `[0.001, 0.001, 0.001]`. Retain this as source metadata, but do not apply this
  collision reduction or its attachment constraint in physical validation.
- The official Isaac pick evaluator sets the object pose and places the robot
  at the **final** path configuration before closing and testing a lift. It
  does not execute the full approach path. Its setup requests Isaac Sim
  2023.1.1 and Python 3.10.13, and references a private `10.2.31.187` release
  server. This is not an unattended public replay recipe or evidence that our
  whole-trajectory dynamics checks passed.

These findings come from the native request and the pinned
[pick evaluator](https://github.com/TooSchoolForCool/M3Bench/blob/97cec07e7c37cc36aec35ff79ee5d6b5db123a45/evaluation/tv_evaluate/evaluate_pick.py#L195-L241)
and [setup script](https://github.com/TooSchoolForCool/M3Bench/blob/97cec07e7c37cc36aec35ff79ee5d6b5db123a45/tongverse/setup.sh).

The remaining physical pipeline is explicit: acquire the selected scene's
referenced meshes and collision assets; preserve native dimensions; review object
mass, inertia, and friction; select and record trajectory timing and gripper
closure; map the source gripper frame to Reachy's actual pad frame; and run
independent actuator-driven MuJoCo trials. The full scene ZIP is 15.24 GB and its
central directory is 61 MB; neither was downloaded locally for this inspection.
A robot URDF alone is sufficient for FK, but not for scene collision validation.

## MeLLO

The inspected source revision is
`ea82019e6093b85311b2c5e4865760a5d0945ce6`, with
[CC BY 4.0 attribution terms](https://github.com/nluttmer1/MeLLO-Data-Library/blob/ea82019e6093b85311b2c5e4865760a5d0945ce6/LICENSE.txt).
The library covers instrumented box, luggage, briefcase, walker, shopping cart,
wheelbarrow, and door interactions, including walking, carrying, pushing, and
pulling. The paper describes force/torque sensors and IMUs in addition to human
and object markers; those signals must be distinguished from derived motion.
Its instrumented box is 8 kg plus a 1.4 kg added load. Large-object handling may
be infeasible for a target embodiment; object size or mass must not be reduced
to make a retarget pass. [Author paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11939698/).

The directly accessible pilot is
[`Subject 1 - Box Walk 1.csv`](https://github.com/nluttmer1/MeLLO-Data-Library/blob/ea82019e6093b85311b2c5e4865760a5d0945ce6/Data%20Library/Subject%201/Box/Walk/Cut%20VICON%20Data/Subject%201%20-%20Box%20Walk%201.csv).
Its 506,651 bytes contain a Vicon `Trajectories` section, a recorded rate of
100 Hz, XYZ values in millimetres, and original frame indices 2198–2411.
The adapter preserves all 214 frames, 67 human and nine Box markers, and 1,617
missing marker observations. It converts units to metres without interpolation.
Relative time is derived from the recorded rate and original frame differences;
the file does not establish an absolute clock origin shared with other sensors.

The six named Box corner markers support a rigid correspondence fit. The adapter
derives a marker-centroid frame using the first fully observed reference frame,
retains fit RMS residuals, and masks frames with fewer than three noncollinear
correspondences or residual above 5 mm. There is no scale fitting or gap filling.
All 214 pilot frames meet that geometric threshold; maximum RMS residual is
1.361 mm and marker-frame translation is 2.166 m. These are reconstruction
measurements, **not grasp or physical success**. The origin is not the object's
mesh origin, centre of mass, handle, or calibrated contact frame.

The selected cut CSV has no load-cell stream. Its paired `exp1_022.mat` entry is
a Git LFS pointer to a 59,040,060-byte payload, not a 133-byte force recording.
The paper describes synchronized `Final Data`; the pinned public Git tree has
**no files in a `Final Data` directory**. Full-session Vicon trigger, load-cell,
and IMU streams therefore require explicit synchronization before they can be
associated with this cut. Existing transform code and three static CSVs are
present, but no OBJ/PLY/STL/GLB/USD object meshes were found in that tree.
Do not infer missing force values, hand orientations, articulated handle poses,
or collision geometry from a successful marker fit.

Next stages are calibrated marker-to-mesh/handle reconstruction, force-clock
alignment where needed, and target reach/grasp/payload feasibility review. The
first retarget attempt should preserve source motion and restrict modifications
to justified contact alignment. A complete physical Reachy episode requires
actual robot control and object dynamics. Task-object pose completeness requires
the relevant calibrated transforms; unrelated unmarked scene props do not block
an otherwise complete task recording.

## Persistent cluster and agent review contract

All acquisition and result paths use `/mnt/reachy-retarget` on the persistent
worker pool. Native downloads belong in an immutable source area; derived
episodes use the common dataset writer. Both adapters are offline:
`describe()` returns the acquisition/stage plan, `inspect(path)` reports verified
native fields, and `normalize(path)` returns shared arrays and provenance.
Network access is performed only by the explicit acquisition stage, subject to
the 50 decimal GB reserve.

| Stage | M3Bench output | MeLLO output | Required review before advancing |
|---|---|---|---|
| Inspect and normalize | Untimed source joint/base waypoints, initial object transform, optional source FK | Timed measured markers, validity masks, derived Box marker frame | Verify hashes, coordinate frames, original recording group, missing fields |
| Reconstruct | Collision scene, physical properties, declared timing and gripper closure | Calibrated object/handle geometry and relevant human contact targets | Keep source facts separate from authored assumptions |
| Retarget | Source wrist/base reference mapped to Reachy | Feasible human/object-relative contact mapped to Reachy | Review pad alignment and geometry/payload feasibility |
| Validate | Actuator-driven attempts and unchanged object dynamics | Same evidence requirements | Reject object state writes, welds, collision shrinking, and unsupported success claims |
| Export | Shared HDF5 source/derived archive; complete v5 only with required real channels | Same schema and completion gates | Preserve failures and identify incomplete training records |

The agent may propose different reconstruction logic for each dataset, but it
must leave review evidence and a new immutable attempt. An invalid or incomplete
episode stays archived with explicit blockers. Missing native timing is not a
reason to fabricate a 10 Hz policy-ready recording. Multiple crops, OpenSim
derivatives, and planner variants share their original capture/generation group.

Offline validation covers unit/clock conversion, occlusions, degenerate marker
sets, transform fitting, named-joint agreement, initial object/robot consistency,
untimed shared export, mask-preserving shared export, and source FK. The two
adapter test files and common exporter checks passed 37 tests at this inspection.
Both real native pilots were also written and inspected in the common archive
format. No new Reachy physical success is claimed by this adapter work.
