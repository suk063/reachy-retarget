# A common archive and the reachy-agent v5 bridge

Inspected on 2026-10-06, read-only, against sibling `reachy-agent` revision
`eb39c2e0f133e8955acfb949513512eaae193db1`. No robot SDK was imported and no
hardware connection was made. The common exporter lives in
[`agent_dataset.py`](../reachy_retarget/agent_dataset.py).
The publisher's remote `main` was also verified at this same commit;
[`reachy-agent-main-lock.json`](../configs/reachy-agent-main-lock.json) records
the remote verification time and the clean tracked working-tree identity.

All source datasets use the same **archive schema**,
`reachy-retarget-agent-archive-v1`. Real Reachy observations use the exact v5
channel names when they exist. Source robot/human observations, source actions,
and untimed planned waypoints remain explicitly identified as source data.
An incomplete archive does not become a policy episode by adding dummy images,
zero controller histories, inferred head targets, or a v5 schema attribute.

## Requested state-only recording profile

The current user requirement is **state observations without stored RGB**.
New canonical archives default to `storage_profile: state-only`,
`rgb_required:false`, and `rgb_stored:false`. Camera **poses** are retained.
RGB is not listed as a missing requirement of this profile. Missing native-v5
channels are listed separately under `missing_native_v5_fields`; the optional
visual-v5 bridge below is a separate compatibility path.

[`state_observations.py`](../reachy_retarget/state_observations.py) implements an
actual read-only `StateObserver` over MuJoCo `MjModel`/`MjData`. It reads current
state and cached transforms, not the reference trajectory. It never steps,
forwards, resets, renders, teleports objects or writes simulator/model buffers.
The application must synchronize MuJoCo position/velocity caches before capture;
the collector cannot correct stale caches by running physics itself.

| Channel | One captured row | Saved episode | Meaning |
| --- | --- | --- | --- |
| `timestamp` | scalar | T | Actual `mjData.time`, without retiming |
| `observation/joint_position` | J | T × J | All scalar robot and selected task-object joints in declared order |
| `observation/joint_velocity` | J | T × J | Measured `qvel` at the corresponding scalar DOFs |
| `observation/base_pose` | 7 | T × 7 | Actual world xyz + wxyz |
| `observation/base_twist` | 3 | T × 3 | vx, vy, wz at the base origin, expressed in base axes |
| `observation/base_twist_world`, `base_twist_local` | 6 | T × 6 | Angular-first complete base spatial velocity in world/base axes |
| `observation/left_tcp_pose`, `right_tcp_pose`, `head_pose` | 7 | T × 7 | Actual world poses; corresponding `*_pose_base` channels are relative to the current base |
| `observation/<left_tcp/right_tcp/head>_twist_world` | 6 | T × 6 | Actual angular-first spatial velocity at that part's origin in world axes |
| `observation/<part>_twist_base` | 6 | T × 6 | The same absolute velocity expressed in base axes |
| `observation/<part>_twist_relative_base` | 6 | T × 6 | Motion relative to the moving base, including subtraction of rotational transport |
| `observation/pickup_pose`, `receptacle_pose` | 7 | T × 7 | Only explicitly declared task roles; undefined roles remain absent |
| `observation/body_poses` | B × 7 | T × B × 7 | Robot bodies plus all parts of explicitly selected task objects |
| `observation/objects/<id>/pose` | 7 | T × 7 | Selected task object's unique root, where one exists |
| `observation/objects/<id>/parts/<body-id>/pose` | 7 | T × 7 | Every articulated/rigid part belonging to the task object's declared subtree |
| `observation/objects/<id>/qpos`, `qvel` | model-dependent | T × width | All free/ball/scalar states for that task object and its parts |
| `observation/cameras/<name>/pose` | 7 | T × 7 | Actual optical world pose; no RGB/rendering required |
| `observation/qpos`, `qvel`, `physics_state` | model-dependent | T × width | Full compiled replay state, including scene state needed to reproduce dynamics |
| `observation/actuator_control` | A | T × A | Actual `data.ctrl` snapshot, distinct from a desired joint or EE target |

The direct pose/body channels contain **task-relevant objects only**. A required
`task_objects` manifest maps semantic IDs to body roots; all their descendants,
including door/drawer parts and passive joints, are retained. An unrelated prop
or articulated scene object does not appear in direct object/body/joint streams.
The complete `qpos`/`qvel`/integration snapshot still preserves compiled replay
state; no source or simulator state is deleted to create the scoped observation.
Unknown task-object names, overlapping object scopes and roles outside the
manifest raise errors instead of being silently ignored.

Joint metadata gives ordered names, model/body IDs, scalar qpos/dof addresses,
joint types, ranges, units, and group **indices** for both arms, neck, grippers,
base, robot/task joints and direct-actuator membership. Full state vectors
preserve nonscalar free/ball joints. Missing standard Reachy arm/neck/gripper
joints are reported explicitly: recording every joint of a model with a frozen
neck does not establish complete Reachy control-state coverage. Actuator names
and transmission IDs identify raw controls without treating their numerical
values as universally position or velocity targets.

World/local velocity extraction uses MuJoCo's read-only
[`mj_objectVelocity`](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-objectvelocity)
with angular then linear components at the requested object's origin. Body,
base and head queries use **`mjOBJ_XBODY`**, whose origin/axes match `xpos/xmat`;
`mjOBJ_BODY` instead refers to the inertial/COM frame and would be wrong for
these stored poses when the inertial origin or orientation is offset. TCPs use
`mjOBJ_SITE`.
Tests compare the local base velocity against an analytical rotated-base case
and body/EE velocities against finite differences in a separate simulator state,
including translated and rotated base/head inertial frames.

```python
observer = StateObserver(
    model,
    robot_body_root="base_link", base_body="base_link", head_body="head",
    tcp_sites={"left": "l_arm_tip_tcp", "right": "r_arm_tip_tcp"},
    task_objects={"can": "can_body", "bin": "bin_body"},
    object_roles={"pickup": "can", "receptacle": "bin"},
)
row = observer.capture(data)       # each array has the per-row shape above
record = stack_observations(rows)  # introduces the leading T dimension
report = assess_state_observations(record)
```

`assess_state_observations` checks this state contract independently of the
strict native-v5 checker. It reports task-specific observation completeness,
table-field completeness, missing required joints, undefined task roles, and
control-recording completeness separately. It never requires RGB or declares a
physics success or native policy compatibility. A task without a receptacle may
have complete observations for its declared task while the receptacle table
field remains explicitly absent.

Requested commands remain separate from observations. Later absolute/delta EE,
joint-position and joint-velocity views must record their frame and timing.
Copying observed joint positions into `command/joint_position` does not create
an expert command. The state assessor checks separately for explicit command
semantics, actual controller snapshots and `applied_controls[T, substeps, A]`.
`issued_control_modes` or an explicit `command_profile.required_fields` declares
which command channels the controller actually issued. A complete position-servo
record does not require nonexistent velocity/Cartesian commands:
`actual_controls_complete` reports completeness of that declared profile,
while `all_control_modes_available` separately reports alternative channel
availability. Missing unissued command modes are never counted as lost data.
The collector does not invent any of those values; rollout code must append the
actual commands, controller state and all applied physics-substep controls.

```bash
python -m reachy_retarget.agent_dataset assess-state \
  --episode /mnt/reachy-retarget/datasets/DATASET/episodes/EPISODE.hdf5
```

## What native reachy-agent v5 saves

The writer is `simulation/writer.py`; action definitions are in
`robot/actions.py`; `simulation/validate.py` checks the recording against the
simulator; `policy/data.py` and `policy/inputs.py` consume it.

Its native episode schema is `reachy-mujoco-episode-v5`. The action contract is
`bimanual-head-observation-torso-camera-rotation6d-binary-grippers-reachy-control-v5`.

| Channel | Per-row shape | Meaning |
| --- | --- | --- |
| `timestamp`, `state_tick`, `image_tick` | scalar | Contiguous 10 Hz, synchronized pre-action state and image |
| `action` | 26 | Left TCP xyz and rotation6d; right TCP xyz and rotation6d; head rotation6d; left/right binary open commands |
| `stage_index` | scalar uint8 | Index into navigate, approach, grasp, lift, transfer, release, retreat, verify |
| `observation/joint_position`, `observation/joint_velocity` | J | Actual named joint observations, including recorded passive/mimic coordinates |
| `observation/base_pose` | 7 | World position and wxyz quaternion |
| `observation/base_twist` | 3 | Actual vx, vy, wz in the current base frame |
| `observation/left_tcp_pose`, `observation/right_tcp_pose` | 7 | Measured URDF arm-tip poses in world coordinates |
| `observation/pickup_pose`, `observation/receptacle_pose` | 7 | Actual object roles; the current generator uses can and bin |
| `observation/body_poses` | B × 7 | Named robot and scene component poses |
| `observation/cameras/<name>/pose` | 7 | Actual camera optical pose in world coordinates |
| `observation/cameras/<name>/rgb` | 256 × 256 × 3 uint8 | Calibration-aware RGB rendered from the same pre-action state |
| `physics_state` | model-dependent | Full `mjSTATE_INTEGRATION`, including the information omitted from joint observations |
| `controller_state_json` | UTF-8 JSON | Pre-action base, joints, targets, allocation/solver memory, servo targets, controller identity and format |
| `applied_controls` | substeps × actuators | Every physics substep's applied actuator controls |

Rotation6d is the first rotation column followed by the second, in order
`r00,r10,r20,r01,r11,r21`. Both hand goals and the head orientation are expressed
in the **torso camera frame observed at the current row**, not the world or a
fixed camera frame. Open is 1 and closed is 0. A source jaw angle or a continuous
source action is not automatically this binary label. Policy training later
re-expresses action chunks in the torso camera frame at the chunk start.

Dataset `metadata.json` includes the ordered action/joint/body/actuator names,
camera intrinsics/distortion/preprocessing, quaternion order, 10 Hz timing,
physics-state convention, prepared-scene manifest hash, source asset hashes,
the exact reachy-control revision/settings/runtime identity, and simulation
settings. Every native episode embeds the same metadata, layout JSON, seed,
episode ID/index, schema/contract and recorded success. The writer preserves
unfinished parts and attempt records and publishes successful episodes under
`episodes/000000.hdf5`, with a checksum sidecar.

### A real saved episode was inspected

The existing sibling fixture is
`runs/simulation/simple-scene/data/episodes/000000.hdf5`:

- SHA256: `29baba2418cec38acd773d55f9c21697de18ce01ef6e8a0202e51ca25c02f630`.
- 519 rows; `action` `(519,26)`; `applied_controls` `(519,50,22)`;
  `physics_state` `(519,800)`.
- 34 joint names and 34 body names. These counts describe this fixture, not a
  universal model-size requirement.
- `left_head` and `torso` RGB and poses; original controller-state format
  `reachy-control-state-v3`; physics timestep 0.002 s.
- The new offline structural checker found no missing fields or structural
  errors. It did **not** run this episode's physics/image/controller validator.
  The existing fixture is not counted as a newly collected or independent demo.

## The common archive layout

```text
/mnt/reachy-retarget/
  datasets/<dataset-id>/
    metadata.json
    episodes/<episode-id>.hdf5
    episodes/<episode-id>.json
    policy-v5/                 # created only by the evidence-gated native bridge
      metadata.json
      episodes/000000.hdf5
      episodes/000000.json
```

Common archives use one schema across BEHAVIOR, MoMaGen, M3Bench, MeLLO,
RoboCasa, BiGym and the existing sources. The `policy-v5` directory is a
separate native export profile; it contains original complete v5 files, not
renamed partial archives. Different prepared scenes/controllers must use
different native export roots because reachy-agent checks metadata equality.

`write_archive(root, dataset_id, episode_id, arrays, metadata)` writes an
immutable HDF5 file plus a hash-bound JSON sidecar. It refuses to overwrite
episodes. Root metadata and publication are protected by a filesystem lock.
Raw data, failures and prior attempts remain in their own acquisition/run
locations; an archive is an additional representation.

The common archive has these conventions:

- `timestamp[T]` contains the real episode time in seconds when known. Native
  source rates are retained during normalization; writing an archive does not
  silently resample to 10 Hz.
- If timing is unknown, omit `timestamp`, list it in `missing_fields`, and
  supply a strictly increasing `source/frame_index` or
  `source/waypoint_index`. The same schema records `timed:false`; a planned
  waypoint sequence is never assigned invented sample times or called an
  executed demonstration.
- `source/` preserves the original normalized arrays. For example, normalized
  `hand/right_pose` remains a source hand observation; it is not a Reachy TCP.
- Exact v5 names represent actual corresponding Reachy observations/commands
  when recorded. Additional objects use `observation/objects/<id>/pose` and
  articulated fixtures use `observation/fixtures/<id>/joint_position`.
- `export_normalized` makes identity aliases for source object/fixture
  channels in the observation hierarchy. Metadata explicitly labels these
  aliases as **source-world** observations. No Reachy-world transform is
  implied. Retargeted target-world data needs its actual transform provenance.
- Nonfinite canonical observations/targets need an explicit per-frame boolean
  `validity_masks` mapping. A mask may not mark a nonfinite sample as valid.
  Missing samples are retained, not interpolated across by the exporter.
  Declared source masks are retained for original channels and remapped to
  their observation aliases. Marker arrays may use a frame-by-marker mask;
  the mask shape must match the leading dimensions of the recorded quantity.
- Metadata records source URLs, revision or explicit null, source sequence,
  original source group, objects, missing fields, derived fields, simulation
  assumptions and the complete source metadata. Crops/versions retain their
  shared `source_group`; publishing another format does not add a demo.
- Every field has a dtype/shape/nonfinite-count inventory. The sidecar binds
  this inventory and metadata to the HDF5 SHA256.
- `policy_ready:false` and the archive schema remain explicit even when some
  channels happen to have v5 shapes. Export itself makes no new physics claim.

The dataset adapter interface is:

```python
record = {
    "arrays": {"time_s": times, "objects/object_id/pose": poses, ...},
    "metadata": {
        "source_id": "dataset-id", "episode_id": "episode-id",
        "source_sequence": "original-sequence", "source_group": "original-demo",
        "source_urls": ["https://publisher.example/data"], "source_revision": None,
        "objects": {"object_id": {"pose_frame": "world"}},
        "derived_fields": {}, "simulation_assumptions": [],
    },
    "status": "normalized",
    "missing_fields": [...],
}
path = export_normalized(record, root="/mnt/reachy-retarget")
```

`export_normalized` also accepts an existing normalized HDF5 path. It retains
the source HDF5 checksum and the source sidecar's provenance. Source URLs or
revisions that are not in the source metadata are explicitly listed as missing.

```bash
python -m reachy_retarget.agent_dataset export-normalized \
  --episode /path/to/normalized.h5 --root /mnt/reachy-retarget
python -m reachy_retarget.agent_dataset inspect \
  --episode /mnt/reachy-retarget/datasets/DATASET/episodes/EPISODE.hdf5
```

## Optional native visual-v5 bridge

This bridge preserves the existing sibling format. It is separate from the
requested state-only output and does not make RGB mandatory for canonical
state-only datasets.

1. Preserve source semantics and reconstruct the scene, object/fixture states,
   source hand paths and available base motion through the dataset adapter.
2. Retarget using the appropriate source-specific method. Begin with the
   minimal pad-position correction for the existing manipulation references;
   the archive format does not impose a new optimizer.
3. Execute the resulting camera-relative hand/head targets through the actual
   offline reachy-control controller in MuJoCo. Record actual pre-action images,
   body/camera/object poses and complete controller/physics snapshots with
   reachy-agent's `Environment`/`EpisodeWriter` or an equivalent audited writer.
   Never substitute robot joint PD logs for reachy-control history.
4. Validate task success, contacts, actuator-only dynamics, frame consistency,
   controller replay and images. Preserve failed trials as failures. A state
   restoration to initialize an independent replay is not permission to
   overwrite the moving object's state during its physical rollout.
5. Use `import_v5` to preserve complete recorded rows in the common archive.
   Native publication additionally requires an external validation report
   bound to the exact source episode SHA256. The bridge checks the recorded
   success, structure, report status and every supplied check, then copies the
   original file byte-for-byte and retains its original metadata/attributes.

The report is an evidence envelope produced by the caller after **actually**
running the validator; it is not a report the exporter generates or fabricates:

```json
{
  "validator": "simulation.validate",
  "episode_sha256": "EXACT_SHA256_OF_THE_VALIDATED_EPISODE",
  "status": "passed",
  "checks": [{"name": "ACTUAL_VALIDATOR_CHECK", "ok": true}]
}
```

```bash
python -m reachy_retarget.agent_dataset assess-v5 --episode /path/to/recorded.hdf5
python -m reachy_retarget.agent_dataset import-v5 \
  --episode /path/to/recorded.hdf5 --dataset DATASET --root /mnt/reachy-retarget \
  --validation-evidence /path/to/actual-validation.json
```

Omitting `--validation-evidence` creates only the common archive. The bridge
does not import or invoke reachy-agent itself and its structural assessment
continues to say `physics_validated:false`; any runtime pass is attributed
separately to the external evidence. The bridge rejects wrong episode hashes,
failed/empty checks, incompatible native-root metadata, and duplicate native
episode content. It does not count a format conversion as an independent demo.

### Current consumer limitations

Format compatibility and task support are distinct. The current sibling
`simulation.validate` assumes bodies named `can` and `bin`, checks their
pickup/receptacle correspondence and expects all eight task stages. A drawer,
door, push, long navigation episode or human-object scene needs a corresponding
scene/task validator; renaming those objects or fabricating a receptacle does
not establish compatibility. These limits were found in code, not inferred
from the README.

The existing policy loader also requires the exact prepared-scene/map manifest
hash, matching metadata across episodes, map components with recorded body
poses, a reachy-control controller, synchronized 10 Hz observations, binary
gripper labels and at least two successful episodes for a train/validation
split. Its current implementation is not a mixed-scene, mixed-manifest loader.
Source grouping must additionally be honored when constructing train/validation
sets because different crops of one demonstration are not independent episodes.

## Source evidence and verification

| Read-only sibling file | SHA256 |
| --- | --- |
| `robot/actions.py` | `3213b99e042ec30fc79b57114df74f6b3302c1c12e466b78bc7fc0bee80bf03a` |
| `simulation/writer.py` | `394b7c5cc3d1bee027a97f19e54f8279e8009f2251a43d73f68b92f9dfd850d4` |
| `simulation/validate.py` | `89e34b99c90d4ecd7a84c777de560e85fdca15da2610268b2afaaf81f33c57fa` |
| `policy/data.py` | `effacad51911546ef33965761f58c20f2d90659f8dccdb7def81faf08bfeaf9a` |
| `policy/inputs.py` | `fe69e7334241fe71a5c4905b8451c48cec0d43e7332cfe6b35498d5fe3c236eb` |
| `simulation/agent_loop.py` | `fd8e3563b08692a88f7f9476b3cb8b269795a2327e89f18d9cf3593db3573bd3` |

Offline tests in `tests/test_agent_dataset.py` check cross-source schema
identity, immutable publication, checksum tampering, original grouping,
nonfinite observation masks, unknown timing, derived-label preservation,
v5 synchronization, binary labels, controller-state presence, complete action
shapes, and evidence-gated byte-identical native export. Their synthetic format
fixtures are tests of serialization and rejection rules, not physics successes.
`tests/test_state_observations.py` additionally checks actual MuJoCo collection,
read-only model/integration/kinematic buffers, task-only body and joint scopes,
passive articulated states, camera optical poses, both spatial-velocity frame
conventions, missing-neck/role reporting, and state-only archives without RGB.
