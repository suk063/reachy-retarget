# RoboCasa source-state reconstruction

The acquired `PickPlaceCounterToCabinet/20250819/episode_000000` now has
independently verified robot FK and all task fixture poses. The reconstruction
ran on the existing cluster access pod on 2026-10-07, reading
`/mnt/reachy-retarget/native/robocasa`. It used no downloads, new jobs, images,
hardware SDKs, physics steps, or changes to source files.

This is **source-state kinematics**, not asset-faithful source dynamics replay
or Reachy physical success. The result status is
`source_fk_verified_actions_preserved`.

## Corrected acquisition finding

The earlier statement in `native-mobile-archive.md` that source parquet actions
and observations were not acquired was incorrect. Both local and cluster
acquisition manifests include
`lerobot/data/chunk-000/episode_000000.parquet`, 39,262 bytes, SHA256
`5aa9350006bd028065cb4a1aee5e4d6e70bd8ae2c0fbd233cda776e5f6d4c992`.
It contains 232 rows of 16 source observations and 12 source actions, plus
timestamps, indices, rewards and done flags. These columns are now read and
retained without reading videos.

Source identity remains one original demonstration. Reconstructed derivatives
and subsequent attempts are not additional independent demonstrations.

## Frames and the EEF orientation trap

The source records RoboCasa **0.5.1**, robosuite **1.5.2**, MuJoCo **3.3.1**, and
the **PandaOmron** embodiment. The acquired source XML itself, rather than a
newly reset environment, defines the reconstruction.

The [official robosuite 1.5.2 mobile-robot observable](https://github.com/ARISE-Initiative/robosuite/blob/v1.5.2/robosuite/robots/mobile_robot.py)
uses the position of `gripper0_right_grip_site` and the rotation of
`robot0_right_hand` for the legacy relative EEF observation. These differ from
the grip site's rotation by 90 degrees in this episode. The
[pinned RoboCasa conversion](https://github.com/robocasa/robocasa/blob/456174f62b89b8fca99eaaf33949c29fec9cfc2a/robocasa/scripts/dataset_scripts/convert_hdf5_lerobot.py)
reorders observations/actions into the recorded modality layout.

The new adapter retains both conventions explicitly:

| Channel | Convention |
| --- | --- |
| `source/base_pose` | World pose of `mobilebase0_center`; mobile support center including torso height |
| `source/mobile_base_body_pose` | World pose of the mobile chassis body `mobilebase0_base` |
| `source/right_tcp_pose` | World position **and rotation** of the same grip site |
| `source/right_tcp_pose_base` | That true grip-site pose relative to the mobile support center |
| `source/right_eef_observable_pose_base` | Legacy source observation: grip-site position with hand-body rotation |
| `source/robot_body_poses` | The 20 named source robot bodies, with ordered names in metadata |

Derived poses use `xyz+wxyz`; original parquet quaternion columns remain
`xyzw` inside `source/observation_state`. A retargeter must not treat the legacy
observable quaternion as the grip site's quaternion. No left arm or neck is
invented for this single-arm source robot.

## Kinematic projection and verification

`compile_kinematic_model(xml)` keeps every explicit body, joint, site and camera
transform and verifies all compiled joint names and qpos/qvel addresses against
the expanded source traversal. It rejects unresolved defaults, includes,
frames, aligned-free coordinates and selected frames dependent on unrecorded
mocap state.

The source references 285 meshes and 76 textures. They are unnecessary for
body/joint/site FK and are omitted from the isolated projection, along with
geoms, actuators and contacts. Moving bodies lacking explicit inertials receive
declared compiler-only inertials so MuJoCo can compile the tree. Those values
are not object physics assumptions: the projected model is marked
`physical_model: false` and must never be used as a physical validation scene.
The original compressed model, full source states and their checksums remain
unchanged. MuJoCo's [kinematic pipeline](https://mujoco.readthedocs.io/en/stable/computation.html#consistency-in-mjdata)
updates poses from the restored coordinates without integrating motion.

All 232 reconstructed base/EEF observations were compared with the genuine
parquet rows in cluster MuJoCo **3.15.0**:

| Check | Maximum error |
| --- | ---: |
| Base position | 0 m |
| Base orientation | 3.684e-11 rad |
| Legacy EEF position relative to base | 7.056e-16 m |
| Legacy EEF orientation relative to base | 3.509e-7 rad |
| Gripper positions | 0 m |
| Cereal free-joint position | 0 m |
| Cereal free-joint orientation | 1.157e-15 rad |

The adapter fails if position errors exceed `1e-8 m` or rotation errors exceed
`2e-6 rad`. Source/runtime versions are retained rather than claiming source
dynamics equivalence across versions.

The 13 named robot qpos/qvel coordinates are preserved. Task pose channels cover
only the cereal `obj`, `counter_1_left_group_1`, and
`hingecabinet_3_left_group_1`, including its nine root/door/handle/shelf bodies.
Both cabinet hinge positions and velocities are retained. Distractors receive
no direct pose channels; their original coordinates remain only inside the
unaltered full source-state reference.

The simulator clock remains authoritative: **0.5 to 12.1 seconds**, with 230
50 ms intervals and one 100 ms interval. The relative common clock ends at
11.6 seconds. Parquet's nominal timestamp ends at approximately 11.55 seconds;
it is preserved as `source/parquet_timestamp` and never substituted for the
simulator clock. No action holding across the missing nominal sample is
assumed.

## Interface and remaining work

```python
from reachy_retarget.native_replay.robocasa import reconstruct
from reachy_retarget.agent_dataset import export_normalized

record = reconstruct("/mnt/reachy-retarget/native/robocasa")
# Same {arrays, metadata, status, missing_fields} interface as other sources.
# Publication uses a new immutable episode ID chosen by the cluster task.
archive = export_normalized(record, "/mnt/reachy-retarget", "robocasa", "new-id")
```

The actual reordered 12-component action vectors, their recorded modality
mapping and native controller configuration are preserved as source commands.
They are neither Reachy commands nor measured joint velocities. Controller
memory and every physics-substep control were not recorded in this source
selection and remain explicitly missing.

The reconstruction now supplies verified source TCP/base trajectories and task
poses to the retargeting stage. Asset-faithful source dynamics and an actual
Reachy rollout with contact/task validation remain separate work.

## Reproducibility evidence

| Artifact | SHA256 |
| --- | --- |
| `model.xml.gz` | `6018ec002d982808b6012de81fd340f931096f3dfab09a92fa9bcfb409ffd345` |
| Original expanded XML | `2a9652be9d833bf5bedba20195b73bee70277c8095e5e186ffa6e7ea9620439d` |
| FK projection XML | `e6c76dd37c876ada3b99cffbfae213c586a2dc312176646755869f210b4a0c98` |
| `states.npz` | `24981c73820bde85c14d209a85a80364c53bec14007700a195c51b860a63eace` |
| `modality.json` | `59589093ca5f946ed06328319cb43d0609492dbd6c583b1c2617803ae3e473fb` |
| Reconstruction module executed on cluster | `a3b18fc991ea35a46f6f7d8335417f4400f9d423abec9248d714801478ef7a71` |

The cluster had 1,098,135,896,064 free bytes; zero bytes were downloaded for this
reconstruction. Five new offline regression tests and the eight existing native
archive tests pass: **13 passed**. Tests verify primitive full-model FK
equivalence, source-observation agreement, the distinct EEF rotation convention,
task-only output scope, clock gaps, unchanged input states, common export, and
rejection of mismatched observations or unsupported compiler semantics.
