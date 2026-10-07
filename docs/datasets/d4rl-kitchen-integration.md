# D4RL kitchen: acquired source and explicit FK reference

The historical complete, partial, and mixed kitchen files are acquired and
checksum verified on the existing cluster. One recorded segment from each file
has been reconstructed and exported to the common archive format. These are
noisy source configuration references, not exact simulator replay, Reachy
retargeting results, or physics successes. No RGB, robot SDK, rendering, or
hardware connection is required.

## Acquisition evidence

The acquisition root is `/mnt/reachy-retarget/native/d4rl_kitchen`.
`acquisition.json` records all source URLs, source Git blob checksums, SHA-256
checksums, sizes, and local paths. It contains 128 files totaling 55,823,156
bytes: three dataset files and 125 pinned source, asset, and license files.
Acquisition used the shared transfer lock and checked the 50 decimal GB disk
reserve before and during transfer. Existing originals and failed transfer
attempts are preserved; no worker or shared runtime was replaced.

The official source revision is
`89141a689b0353b0dac3da5cba60da4b1b16254d` in
[Farama-Foundation/D4RL](https://github.com/Farama-Foundation/D4RL/tree/89141a689b0353b0dac3da5cba60da4b1b16254d/d4rl/kitchen).
The historical dataset checksums match the repository's existing collection
lock. The data filenames below are relative to
`https://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/` and are saved under
the acquisition root's `raw/` directory.

| Variant | Filename | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| complete | `mini_kitchen_microwave_kettle_light_slider-v0.hdf5` | 556,544 | `cd797c38cd52dfbe3f960cef73935cc724847abd189e47a703a7a2315ed60a81` |
| partial | `kitchen_microwave_kettle_light_slider-v0.hdf5` | 19,558,459 | `57bfd9c5fe88a7cf702bbaf8b7cd800d30cd869e7f73416007238d26cd859800` |
| mixed | `kitchen_microwave_kettle_bottomburner_light-v0.hdf5` | 19,560,760 | `a30382a50278fdaea2e2d88e5f2836a95bcf9b581d252617872fd6537302721b` |

The adapter does not download anything on import, inspection, normalization,
or in tests. Only the explicit `fetch(cluster_root)` entry point transfers data.

## What the source actually records

All three files have exactly six fields: `observations` (N, 60), `actions`
(N, 9), and aligned `rewards`, `terminals`, `timeouts`, and `infos` vectors.
The pinned [environment observation implementation](https://github.com/Farama-Foundation/D4RL/blob/89141a689b0353b0dac3da5cba60da4b1b16254d/d4rl/kitchen/adept_envs/franka/kitchen_multitask_v0.py)
concatenates nine robot configuration values, 21 object configuration values,
and 30 task goal values. The last 30 columns are **goals, not velocities**.
The [robot observation implementation](https://github.com/Farama-Foundation/D4RL/blob/89141a689b0353b0dac3da5cba60da4b1b16254d/d4rl/kitchen/adept_envs/franka/robot/franka_robot.py)
adds noise to the observed configuration. Exact simulator qpos, qvel, recorded
timestamps, substep actuator controls, and controller memory are absent.

The original nine action values remain unchanged in `source/actions`. They
are normalized velocity requests that the source controller clips, scales,
limits, and converts to position targets using its cached robot observation.
They are not recorded applied actuator controls or Reachy commands.

| Variant | Rows | Segments delimited by recorded terminal/timeout flags | Distinct `infos` labels |
| --- | ---: | ---: | ---: |
| complete | 3,680 | 19 | 19 |
| partial | 136,950 | 613 | 40 |
| mixed | 136,950 | 613 | 40 |

`infos` labels are reused in the larger files. Segmentation uses only recorded
terminal/timeout flags, retaining any unclosed tail explicitly. It does not
guess boundaries from a fixed horizon or treat labels as unique capture IDs.
The releases overlap and may contain reordered or cropped source material.
They are assigned one shared corpus group; independent demonstration count
remains unknown. An exact configuration/action segment fingerprint supports
exact duplicate detection without claiming to detect every overlapping crop.

## Model reconstruction and task scope

The complete pinned model compiles on the cluster to 30 qpos, 29 qvel, 45
bodies, 241 geoms, and 43 meshes. Its legacy XML requires a recorded syntax
adaptation: expand verified includes in their original order, nest named
default classes below one unnamed main default, merge compiler attributes in
source order, and resolve original asset paths. Every dependency is checked
against the acquisition manifest. Original XML and assets remain unchanged.
The resulting model XML SHA-256 for the cluster acquisition path is
`28419d5741bc760bf05155b1b52b3c8d02b0b99fbc89179eab7d5c7be301a961`.
This checksum includes the resolved asset paths.

Reconstruction assigns the recorded noisy configuration to isolated MuJoCo
data and calls only `mj_kinematics`. It never steps physics or executes source
actions. The noisy kettle quaternion is normalized only in this FK working
copy. Original observations and original quaternion norms are retained. The
first segments have quaternion norms approximately 0.9901–1.0100; silently
treating these observations as exact simulator state would be incorrect.

The true `end_effector` site supplies the estimated source TCP position and
orientation. The fixed `panda0_link0` mounting body supplies `source/base_pose`.
In this model the mounting body's world z is 1.8 m; it is not a mobile base on
the floor. Any downstream workspace placement must be explicit and applied
consistently to source references and task objects.

Direct object pose channels are limited to the benchmark's declared task
elements. Each includes its articulation configuration, root pose, interaction
site pose, and all descendant part poses. The complete and partial variants
include microwave, kettle, light switch, and slide cabinet. The mixed variant
includes microwave, kettle, light switch, and bottom burner. Unrelated fixture
pose channels are omitted. The complete original model and source observations
remain available as provenance, not a claim that every fixture is task relevant.
The source task definitions come from the pinned
[kitchen environments](https://github.com/Farama-Foundation/D4RL/blob/89141a689b0353b0dac3da5cba60da4b1b16254d/d4rl/kitchen/kitchen_envs.py).

Distances to recorded task goals and the distance < 0.3 predicate are derived
observation labels. They do not establish contact, grasp stability, or physical
task success on Reachy.

## Timing and immutable common archives

By default, `normalize()` emits no `timestamp`, marks that field missing, and
retains monotonically increasing original `source/frame_index`. Its common
archive is explicitly untimed. A separate `source/nominal_time_s` is labeled
as derived reference information.

The optional `nominal_timing=True` explicitly selects the pinned environment's
40 physics substeps at the source XML's 0.002 s timestep, giving 0.08 s per
reference row. The timestep is declared in the pinned
[Franka asset XML](https://github.com/Farama-Foundation/D4RL/blob/89141a689b0353b0dac3da5cba60da4b1b16254d/d4rl/kitchen/third_party/franka/assets/assets.xml)
and the frame skip in `KitchenV0.__init__`. This is a reconstruction assumption,
not recovered measured timing. Missing/dropout information and the original
collection simulator revision remain unknown. Both representations preserve
native actions and observations.

The following real archives and their JSON sidecars passed common archive
inspection. Paths are relative to
`/mnt/reachy-retarget/datasets/d4rl_kitchen/episodes/`.

| Archive | Rows | Timing | SHA-256 |
| --- | ---: | --- | --- |
| `d4rl-complete-source-fk-v1.hdf5` | 184 | unknown | `9b8ec1f4417d6fc0aa034d1d323de049ec9cd7bba31e2869f075e1820a5d9bad` |
| `d4rl-partial-source-fk-v1.hdf5` | 255 | unknown | `49e878b7ceb52b8837bd0fa71c19234f9c6d02aec9b8013bb2ad03f941c3135b` |
| `d4rl-mixed-source-fk-v1.hdf5` | 228 | unknown | `e15f9998ad493b7b9b67b4ade871170e47feac24f3f98f769b7283272b9d5c2f` |
| `d4rl-complete-nominal-ik-v1-reference.hdf5` | 184 | nominal, 14.64 s span | `bc733afc6ac92f71663b3290ad58b588160452620152f74497c3c99995364cc3` |
| `d4rl-partial-nominal-ik-v1-reference.hdf5` | 255 | nominal, 20.32 s span | `9f7de7a94ead42b4f6984ee74aad9fcf0785f243b37c6675a5a1492132304eed` |
| `d4rl-mixed-nominal-ik-v1-reference.hdf5` | 228 | nominal, 18.16 s span | `479a0b057a9866966ca96dac13780cd9096441d1647296cf053dc7ca2b96f442` |

All six archives select the first recorded segment of their source file. The
nominal variants are derived views of the same segments, not additional
demonstrations. None has a Reachy physics pass.

## Adapter and Reachy IK handoff

```python
from reachy_retarget.adapters.d4rl_kitchen import normalize

record = normalize(
    "/mnt/reachy-retarget/native/d4rl_kitchen",
    variant="complete",  # or partial / mixed
    episode=0,
    nominal_timing=True,  # explicit reference timing assumption
)
# record = {"arrays": ..., "metadata": ..., "status": ..., "missing_fields": ...}
```

The shared pose retargeter should use `clock_key="timestamp"`,
`base_key="source/base_pose"`, and
`hand_keys={"right": "source/right_tcp_pose_estimate"}`. It must retain the
fixed source base interpretation and label any selected workspace or tool
rotation mapping. `ik_view()` additionally provides legacy aliases and the
sum of the two noisy source finger observations as an aperture reference.
It does not invent a Reachy gripper command or grasp contact label.

Representative direct worker payloads prepared for the shared pipeline are:

```json
[
  {"id": "d4rl-complete-pose-ik-v1", "operation": "source_pose_ik", "dataset": "d4rl_kitchen", "variant": "complete", "episode": 0, "nominal_timing": true},
  {"id": "d4rl-partial-pose-ik-v1", "operation": "source_pose_ik", "dataset": "d4rl_kitchen", "variant": "partial", "episode": 0, "nominal_timing": true},
  {"id": "d4rl-mixed-pose-ik-v1", "operation": "source_pose_ik", "dataset": "d4rl_kitchen", "variant": "mixed", "episode": 0, "nominal_timing": true}
]
```

These payloads require the shared worker route; this adapter does not enqueue
jobs or claim those jobs ran. Legacy input workspaces also exist under
`workspaces/d4rl-{variant}-nominal-ik-v1`, but were not enqueued. The direct
pose retargeting route is preferred. Exact task geometry placement, articulation
interaction controllers, gripper mapping, and physical task validation are
separate remaining work. Source FK restoration is never used as object state
assignment during physical validation.

## Validation

Six offline adapter tests cover recorded-marker segmentation, legacy XML
compilation without source edits, preservation of noisy observations and
quaternion norms, task-only pose scope, untimed common export, explicit nominal
timing, source checksum rejection, and IK reference aliases. Tests prohibit
physics stepping/forward dynamics during reconstruction and use local fixtures.
The adapter, RoboCasa replay, and native common archive suites passed together:
19 tests. Actual cluster reconstruction and common archive checks were run for
all three representative source segments listed above.
