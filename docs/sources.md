# Source families

Every file is pinned in `reachy_retarget/acquire/catalog/<family>.yaml` (URL at an immutable
revision, publisher SHA-256, size, license, kind). `fetch()` is the only code path that
uses the network; it writes `data/raw/<family>/<path>` and records each verified file as
its own JSON record `data/raw/ledger/<family>/<path>.json` (`read_ledger` aggregates
them). One record per file lets concurrent cluster jobs fetch into private roots that
are merged into the shared root by plain no-overwrite copies. A legacy single-file
`raw/ledger.json` is still read (per-file records win) and `migrate_ledger(root)` writes
per-file records for it. "Catalogued", "fetched", "adapted" (a `SourceEpisode` was
produced) and "physics-validated" are separate states; nothing here implies the next.

```python
from reachy_retarget.acquire import fetch, load_catalog
from reachy_retarget.sources import iter_episodes

fetch(["robomimic/v1.5/can/ph/low_dim_v15.hdf5",
       "robosuite/robosuite-1.5.1-py3-none-any.whl"], "data")
for ep in iter_episodes("robomimic", "data/raw/robomimic/v1.5/can/ph/low_dim_v15.hdf5"):
    ...
```

Fetch rules: resumable `*.part` files (HTTP Range; restarts if the server ignores the
range), size and SHA-256 checked before the file is renamed into place (a mismatch is
kept as `*.sha256-mismatch`), refusal before transfer if the file would leave less than
50 decimal GB free, abort (partial kept) if free space drops below that while
streaming, and refusal of `kind: images_embedded` files unless `strip_images=True`.

### Image stripping (`reachy_retarget/acquire/strip.py`)

`fetch(ids, root, strip_images=True)` is the supported way to acquire image-embedding
HDF5 files (MimicGen, LIBERO, DexMimicGen). After the publisher file is verified
(size + SHA-256), it is rewritten as `<stem>.state.hdf5` next to it: every group,
dataset, attribute and soft link is copied (same dtype, shape, chunks and filters)
except datasets whose full path matches the episode writer's image guard
(`schema.io.check_not_image`: image/img/rgb/depth/camera/pixel/video/segmentation) or
that are `uint8` with `ndim >= 3`. The copy is checked against the original (every kept
dataset and attribute equal, no image dataset left), the ledger records the original
URL, revision, SHA-256 and size plus `stripped.{path, sha256, bytes, dropped_names,
dropped_datasets, kept_datasets}`, and the original is deleted. The stripped file also
carries a root attribute `reachy_retarget_stripped` (JSON: original catalog id, URL,
revision, SHA-256, size, license, what was dropped), so `acquire.identify(path)` and the
adapters find the catalog entry of a stripped file without the ledger
(`provenance["sha256"]` = publisher digest, `provenance["file_sha256"]` = local file).
Re-fetching a stripped entry verifies the local copy against the ledger and transfers
nothing; an original already present locally is verified and stripped without transfer.
Only image observations are removed; states, `model_file`, actions, low-dimensional
observations and MimicGen `datagen_info` stay.

| file | original | stripped | ratio |
| --- | --- | --- | --- |
| MimicGen `source/*.hdf5` (12 files) | 471.4 MB | 56.8 MB | 12.1 % (9.1–21.0 %) |
| LIBERO `libero_90/KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5` | 377.0 MB | 7.1 MB | 1.9 % |

## robomimic (family `robomimic`)

* Repository: <https://huggingface.co/datasets/robomimic/robomimic_datasets>,
  revision `74fa018461f479cd9fd15b924a16103012096203`, license MIT (dataset card).
* Catalogued: `v1.5/{lift,can,square,transport}/{ph,mh}/low_dim_v15.hdf5` and
  `v1.5/tool_hang/ph/low_dim_v15.hdf5` (tool_hang has no MH). 9 files, 1.53 GB.
  `demo_v15.hdf5`, `paired/` and machine-generated `mg/` files are not catalogued.
* Recording: robosuite `env_version` 1.5.1, Panda + parallel gripper, OSC_POSE delta
  control, `control_freq` 20 Hz (from `data.attrs["env_args"]`), MuJoCo timestep
  0.002 s. Each demo has `states` = `[sim time, qpos, qvel]` rows and the complete MJCF
  as `model_file`. Episode time is the recorded sim time (0.05 s steps).
* PH = one proficient operator, MH = several operators of mixed skill. Different
  demonstrations, but `mask/*` splits (`train`, `valid`, `20_percent`, ...) overlap and
  are recorded per episode as `provenance["splits"]`, never as extra demos.
* No language instruction is recorded (`instruction=None`). `success` is derived:
  max sparse reward ≥ 1 (`provenance["success_source"]`); all PH demos are successful.

## MimicGen (family `mimicgen`)

* Repository: <https://huggingface.co/datasets/amandlek/mimicgen_datasets>,
  revision `33016f8a62c02334f929f2913af8fdd2a8a129e1`, license CC-BY-4.0.
* Catalogued: all five groups at that revision, 62 files, 148.97 GB:

  | group | files | GB | content | lineage |
  | --- | --- | --- | --- | --- |
  | `source/` | 12 | 0.47 | ~10 human demos per task (D0) | `generated: False` |
  | `core/` | 26 | 88.79 | ~1000 generated demos per task variant D0–D2 | generated from `source/<task>` |
  | `object/` | 2 | 6.65 | MugCleanup with other mugs (O1, O2) | generated from `source/mug_cleanup` |
  | `robot/` | 16 | 26.21 | Square, Threading on Panda / Sawyer / IIWA / UR5e | generated from `source/<task>` |
  | `large_interpolation/` | 6 | 26.85 | long interpolation segments | generated from `source/<task>` |

* Recording: robosuite 1.4.1 (`env_version`), 20 Hz, same `states`/`model_file` layout;
  v1.4 naming (`gripper0_grip_site`, `mount0_`). Source `nut_assembly` and `pick_place`
  (and the `robot/*_sawyer` files) use a Sawyer with the Rethink gripper; everything else
  in `source/` and `core/` is a Panda.
* **Every MimicGen file embeds 84×84 RGB observations** (`agentview_image`,
  `robot0_eye_in_hand_image`), so all are `kind: images_embedded` and must be fetched
  with `strip_images=True`. The 12 source files were fetched and stripped (56.8 MB kept).
* Lineage: every non-source demo is MimicGen-generated:
  `lineage = {"generated": True, "seed": "mimicgen/source/<task>", "seed_demo": None,
  "variant_group": <group>}` (task = file stem without `_d<k>`/`_o<k>` and robot suffix;
  every seed is a catalogued source file). The published files do not record the
  per-demo source index. Generated demos must not be counted as independent
  demonstrations, and `robot/` variants of one task are the same seeds on other arms.
* Object assets: the recorded MJCF of Coffee, CoffeePreparation and MugCleanup references
  `.../mimicgen_environments/mimicgen_envs/models/robosuite/assets/...`; they resolve into
  the pinned GitHub archive of NVlabs/mimicgen tag v0.1.1
  (`mimicgen/code/mimicgen-45db4b35a5a79e82ca8a70ce1321f855498ca82c.zip`, 16.1 MB,
  package `mimicgen_envs`, the version the datasets were recorded with; v1.0.0 renamed the
  package to `mimicgen`). License: NVIDIA Source Code License, non-commercial
  (research/evaluation) use only. Kitchen references one mesh of robosuite-task-zoo
  (`robosuite_task_zoo/models/kitchen/burnerplate.stl`), pinned as
  `robosuite/robosuite-task-zoo-74eab7f88214c21ca1ae8617c2b2f8d19718a9ed.zip` (1.7 MB,
  MIT). GitHub publishes no digests: the SHA-256 and size were computed from two
  identical downloads (2026-10-07). With these archives every source task gets a full
  scene. HammerCleanup and the robosuite-native tasks need only the robosuite wheel.

## robosuite assets (family `robosuite`)

* PyPI wheels `robosuite-1.5.1-py3-none-any.whl` (152 MB) and
  `robosuite-1.4.1-py3-none-any.whl` (194 MB), PyPI SHA-256 digests, license MIT, plus the
  robosuite-task-zoo archive above.
* Read as zip archives, never installed or imported. The recorded MJCF references meshes
  and textures by absolute paths of the recording machine (e.g.
  `/home/.../robosuite/models/assets/robots/panda/meshes/link0.stl`). Every `kind:
  assets` catalog entry declares an `asset_marker`; a reference containing the marker
  resolves to the archive member after it. The robosuite wheel is chosen by the recorded
  `env_version` (marker `robosuite/models/assets/`); other archives are used when their
  marker (or a family alias, e.g. LIBERO's `chiliocosm/assets/`) occurs in the MJCF.

## Adapter: `reachy_retarget/sources/robosuite.py`

Shared by the `robomimic`, `mimicgen`, `libero` and `dexmimicgen` families
(`read_robosuite_family` with a per-family `Profile`: task table, instruction, lineage,
success rule, asset aliases). Stripped `*.state.hdf5` files are read transparently. Per
demo: compile the recorded MJCF (cached by XML and archives), set `qpos`/`qvel` from each
state row, `mj_forward`, and read:

* **Effectors**: one per `gripper<N>[_right|_left]_grip_site` (key = that prefix, e.g.
  `gripper0_right`; Transport has `gripper0_right` and `gripper1_right`). The contract
  frame is measured from the model with the fingers open: +z = palm (finger parent body
  origin) → midpoint of the two finger pads, +y = finger-1 pad → finger-2 pad. For the
  Panda gripper this gives `R_contract = R_site · [[0,1,0],[-1,0,0],[0,0,1]]` (site +z is
  already the approach axis; the closing axis is site ±x) and a grasp center 3.6 mm
  behind the grip site along the approach axis (pad midpoint). Width = pad separation
  along +y minus the fully closed separation (equals the finger joint separation for the
  Panda, 0–0.08 m); opening = width / 0.08. `side_hint` comes from the robot base
  layout: when all arms face the same way (yaws within 30°), the base furthest along the
  shared left normal is `left` and the other `right` (`provenance["side_hint_source"]`);
  arms facing each other (robomimic Transport) or a single arm get `None`. The Rethink
  (Sawyer) gripper fits the same two-finger model; linkage grippers (Robotiq 85 on
  IIWA/UR5e) do not and raise.
* **Objects**: free-joint bodies named `<name>_main`/`<name>_root`. Per-task tables
  (`TASK_OBJECTS`) select task objects and roles; other free bodies (e.g. hidden Milk,
  Bread, Cereal in PickPlaceCan, RoundNut in NutAssemblySquare) are listed in
  `provenance["inactive_free_bodies"]`. Unknown envs keep every free body as
  `manipulated`. Static world-child bodies with collision geometry become constant
  tracks (`table*` → support, `*bin*` → receptacle, else fixture, e.g. square pegs).
  Geometry is the collision AABB in the body frame.
* **Articulations**: non-robot hinge/slide joints grouped by root body (none in the
  robomimic tasks; MimicGen drawers/lids will appear here).
* **base_hint**: `robot0_base` x, y, yaw; every arm's base is in
  `provenance["robot_bases_xy_yaw"]` (Transport: robot0 (0, −0.81, +90°),
  robot1 (0, 0.81, −90°)).
* **Asset compatibility**: MuJoCo's asset dictionary is keyed by base name, so identical
  files with the same base name share one key and different ones get a flattened unique
  name (`textures__wood.png`). MuJoCo 3 rejects some thin visual meshes of the Sawyer
  head ("mesh volume is too small"); those meshes get `inertia="shell"` as MuJoCo
  suggests and are listed in `provenance["mesh_inertia_shell"]` (robot links only; the
  robot is removed for tier P).
* **Scene**: `SceneRef` with the asset-resolved MJCF, the asset bytes, robot prefixes
  (`robot<N>_`, `gripper<N>_`, `mount<N>_`, `fixed_mount<N>_`) and initial `qpos` of
  every free and articulated scene joint.

Routes (`provenance["state_route"]`):

* `mjcf_states`: all assets resolved; exact kinematics and a physics scene.
* `mjcf_kinematic_only`: some asset was missing (`provenance["missing_assets"]`); mesh
  geoms become 1 mm spheres and file textures are dropped. Body, site and joint
  kinematics stay exact (geom shapes do not enter forward kinematics), but there is no
  `scene` and object geometry is empty.

A low-dimensional-observation route is not implemented: every catalogued file has
`states` and `model_file`. Files without them raise.

## LIBERO (family `libero`)

* Data: <https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets>, revision
  `f13aa24a3da8c43c7225569f28c562979fa0e35a`, the official mirror used by LIBERO's own
  `download_libero_datasets.py --use-huggingface` (the alternative UT Box zips have no
  per-file digests). License on the dataset card: Apache-2.0 (the LIBERO code is MIT).
  `openvla/modified_libero_rlds` (re-rendered, filtered RLDS) is not used.
* Catalogued (`catalog/libero.yaml`): 130 task files, 100.44 GB, 50 human teleoperated
  demos each: `libero_spatial` 10 (6.24 GB), `libero_object` 10 (7.44 GB), `libero_goal`
  10 (6.37 GB), `libero_10` 10 (13.73 GB), `libero_90` 90 (66.66 GB). All 130 digests are
  distinct. Every file embeds 128×128 `agentview_rgb`/`eye_in_hand_rgb`
  (`images_embedded`). Assets: GitHub archive of Lifelong-Robot-Learning/LIBERO at
  `8f1084e3132a39270c3a13ebe37270a43ece2a01` (master, 2025-03-15),
  `libero/code/LIBERO-8f1084e3….zip`, 243.6 MB, MIT, digest computed from two identical
  downloads; only `libero/libero/assets/*` and `libero/libero/bddl_files/*` are read.
* Recording: `env_args` has no `env_version` (LIBERO was built on a robosuite 1.4
  pre-release, the MJCF paths say `robosuite-master`); the robosuite 1.4.1 wheel serves the
  robot assets and `provenance["env_version_assumed"] = "1.4.1"` records the substitution.
  Scene assets were recorded under `chiliocosm/assets/` (LIBERO's pre-release name) and
  resolve into `libero/libero/assets/` (alias). Panda on a mount, 20 Hz, `states` =
  `[sim time, qpos, qvel]`.
* Adapter `reachy_retarget/sources/libero.py` (registered `libero`): `instruction` =
  `problem_info["language_instruction"]`; the BDDL file named by `bddl_file_name` is read
  from the archive and its language, objects of interest, fixtures, movable objects and
  goal go to `provenance["bddl"]`. Every free body is a movable BDDL object and stays
  `manipulated`; articulated fixtures (cabinet drawers, microwave doors, stove knobs) are
  articulations. `success`: max recorded reward ≥ 1 (LIBERO's 0/1 `uint8` completion
  reward; `reward_shaping` is set in `env_kwargs` but the recorded rewards are 0/1).
  `lineage = {"generated": False}`.
* LIBERO's recorded `obs/*` lag the states by one control step: `obs/ee_pos[t]` matches
  the replayed grip site of `states[t + 1]` to 0.43 mm. The adapter uses `states` only.

## DexMimicGen (family `dexmimicgen`)

* Repository: <https://huggingface.co/datasets/MimicGen/dexmimicgen_datasets>, revision
  `181967e10c6277a653e7c9761f3978312af3646a`, license CC-BY-NC-SA-4.0 (dataset card;
  non-commercial, share-alike). Only `generated/` exists (~1000 MimicGen-generated demos
  per task); the human source demos are not published.
* Catalogued (`catalog/dexmimicgen.yaml`): the three bimanual Panda tasks with
  parallel-jaw grippers (`env_configuration = "single-arm-parallel"`, robosuite 1.5.1),
  23.41 GB: `two_arm_threading` (3.91 GB, 1025 demos), `two_arm_three_piece_assembly`
  (4.32 GB, 1006), `two_arm_transport` (15.18 GB, 1029).
* Excluded (robot types read from `env_args` with HTTP range reads, 2026-10-07): Reachy
  has parallel grippers, so dexterous-hand tasks are not catalogued:
  `two_arm_box_cleanup`, `two_arm_drawer_cleanup`, `two_arm_lift_tray` (PandaDexRH +
  PandaDexLH hands; 5.17, 5.09, 9.76 GB) and `two_arm_can_sort_random`, `two_arm_coffee`,
  `two_arm_pouring` (GR-1 humanoid with dexterous hands; 4.43, 5.75, 6.32 GB). Note that
  DexMimicGen's coffee, can sort, box/drawer cleanup and tray lift are therefore *not*
  parallel-jaw tasks.
* Adapter `reachy_retarget/sources/dexmimicgen.py` (registered `dexmimicgen`): shared
  reader, two effectors `gripper0_right` (robot0 at y = −0.25 → `side_hint="right"`) and
  `gripper1_right` (robot1 at y = +0.25 → `left`), both arms facing +x. Other envs
  raise. `lineage = {"generated": True, "seed": None, ...}` (source demos unpublished).
  The files carry no `rewards`, so `success` is `None` (not derived).
* All robot and scene assets resolve from the robosuite 1.5.1 wheel; no DexMimicGen code
  archive is needed for these three tasks.

## Verification (2026-10-07)

| dataset | demos | T min–max | total steps | success | opening range | route |
| --- | --- | --- | --- | --- | --- | --- |
| robomimic/can/ph | 200 | 82–151 | 23207 | 200 | 0.378–1.000 | mjcf_states |
| robomimic/lift/ph | 200 | 36–64 | 9666 | 200 | 0.488–1.000 | mjcf_states |
| robomimic/square/ph | 200 | 107–236 | 30154 | 200 | 0.200–1.000 | mjcf_states |
| robomimic/transport/ph | 200 | 373–714 | 93752 | 200 | 0.012–1.000 | mjcf_states |
| mimicgen/source/coffee | 10 | 193–232 | 2082 | 10 | 0.521–0.999 | mjcf_states |
| mimicgen/source/coffee_preparation | 10 | 559–728 | 6600 | 10 | 0.122–1.000 | mjcf_states |
| mimicgen/source/hammer_cleanup | 10 | 240–309 | 2698 | 10 | 0.249–1.000 | mjcf_states |
| mimicgen/source/kitchen | 10 | 551–620 | 5823 | 10 | 0.188–1.000 | mjcf_states |
| mimicgen/source/mug_cleanup | 10 | 277–363 | 3183 | 10 | 0.105–1.000 | mjcf_states |
| mimicgen/source/nut_assembly (Sawyer) | 10 | 271–413 | 3330 | 10 | 0.204–1.000 | mjcf_states |
| mimicgen/source/pick_place (Sawyer) | 10 | 532–679 | 6288 | 10 | 0.000–1.000 | mjcf_states |
| mimicgen/source/square | 10 | 123–160 | 1427 | 10 | 0.290–1.000 | mjcf_states |
| mimicgen/source/stack | 10 | 86–117 | 1001 | 10 | 0.483–0.996 | mjcf_states |
| mimicgen/source/stack_three | 10 | 208–280 | 2333 | 10 | 0.483–0.998 | mjcf_states |
| mimicgen/source/threading | 10 | 182–249 | 2098 | 10 | 0.371–0.998 | mjcf_states |
| mimicgen/source/three_piece_assembly | 10 | 286–353 | 3114 | 10 | 0.409–0.999 | mjcf_states |
| libero/libero_90/KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet | 50 | 58–110 | 3762 | 50 | 0.850–0.999 | mjcf_states |

`mimicgen/source/stack` produces identical `SourceEpisode`s (all arrays bit-equal) from
the original and from the stripped file. The grip-site position reproduces the recorded
`obs/robot0_eef_pos` exactly for robomimic; the grasp center differs from it by 3.6 mm
along the approach axis, as designed. Width equals `robot0_gripper_qpos[0] −
robot0_gripper_qpos[1]`.

Spot checks without full downloads (demo_0 copied by HTTP range reads into scratch
files, whole-file digests therefore not verified; nothing stored under `data/`):

| dataset | T | route | notes |
| --- | --- | --- | --- |
| mimicgen/core/coffee_d0 | 227 | mjcf_states | MimicGen assets |
| mimicgen/core/kitchen_d1 | 585 | mjcf_states | task-zoo burner plate |
| mimicgen/object/mug_cleanup_o1 | 362 | mjcf_states | other mug resolves; rewards all 0 → `success=False` |
| mimicgen/large_interpolation/stack_d1 | 187 | mjcf_states | |
| mimicgen/robot/threading_d0_sawyer | 232 | mjcf_states | |
| mimicgen/robot/square_d0_iiwa, square_d0_ur5e | — | error | Robotiq 85 linkage gripper unsupported |
| dexmimicgen/two_arm_threading (2 demos) | 181–196 | mjcf_states | eef residual ≤ 0.57 mm, width exact |
| dexmimicgen/two_arm_three_piece_assembly | 229 | mjcf_states | sides right/left |
| dexmimicgen/two_arm_transport | 375 | mjcf_states | sides right/left |

For DexMimicGen the replayed grip sites match `obs/robot{0,1}_eef_pos` exactly at t = 0
and within 0.57 mm afterwards (zero-mean, ~0.14 mm std in the site frame: recording
jitter between the stored states and observations, not a frame offset); widths equal
the recorded finger joint separation to 1e-15.

## Known gaps

* MimicGen `robot/*_iiwa` and `robot/*_ur5e` (8 files, 12.6 GB) use the Robotiq 85
  linkage gripper, which the two-finger gripper model rejects; they are catalogued but
  not adaptable yet.
* MimicGen generated files' `rewards` are not a reliable success signal (MimicGen keeps
  only successful generations, yet `object/mug_cleanup_o1` demo_0 has all-zero rewards),
  so `success=False` there should not be trusted.
* Instructions are not available for robomimic, MimicGen or DexMimicGen.
* Only one LIBERO file (libero_90 KITCHEN_SCENE5 close drawer) and no full DexMimicGen
  file were fetched locally; bulk acquisition runs on the cluster with the same
  `fetch(..., strip_images=True)` call.

## ManiSkill3 (family `maniskill`)

* Repository: <https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations>,
  revision `d674485bbffdd533914e52d272fdda34c0515608`, license Apache-2.0 (dataset card).
  `*.h5` digests are the HF LFS sha256; `*.json` are plain git blobs, so their sha256 was
  computed from the bytes at the pinned revision (git blob sha1 checked first).
* Catalogued (`catalog/maniskill.yaml`): every `trajectory*.h5` + `.json` pair of the 13
  parallel-gripper tasks, 31 pairs, 0.777 GB (h5 0.771 GB, json 5.9 MB), plus the Panda
  URDFs `panda_v2.urdf` (robot uid `panda`) and `panda_v3.urdf` (`panda_wristcam`) from
  ManiSkill commit `baab60ed`. All 64 files were fetched and verified locally.
* Not catalogued: `PushT-v1` and `DrawTriangle-v1` (robot `panda_stick`, a stick end
  effector without gripper; 0.10 GB and 3.74 GB), `AnymalC-Reach-v1` (quadruped, no
  object), per-task `*.zip` (duplicate archives of each folder), `*.mp4` (rendered
  video) and `*.pt` (PPO checkpoints). `PlaceSphere-v1`, `OpenCabinetDrawer-v1` and other
  articulated-object tasks have no demonstrations at this revision, so no catalogued
  episode has a scene articulation.

| task | class | sources (files) | episodes | distinct trajectories | distinct initial states |
| --- | --- | --- | --- | --- | --- |
| PickCube-v1 | single arm | motionplanning, rl ×3, teleop | 4042 | 4042 | 2024 |
| PushCube-v1 | single arm | motionplanning, rl ×3 | 4062 | 4062 | 2024 |
| StackCube-v1 | single arm | motionplanning, rl ×3 | 3829 | 3829 | 2017 |
| PullCube-v1 | single arm | rl ×3 | 3072 | 3072 | 1024 |
| PokeCube-v1 (peg as tool) | single arm | rl ×3 | 2300 | 2300 | 1010 |
| RollBall-v1 | single arm | rl ×3 | 2373 | 2373 | 1014 |
| LiftPegUpright-v1 | single arm | rl ×2 | 2008 | 2008 | 1021 |
| PegInsertionSide-v1 | single arm | motionplanning, rl | 2000 | 1688 | 1688 |
| PlugCharger-v1 | single arm | motionplanning | 1000 | 1000 | 1000 |
| PullCubeTool-v1 (L tool) | single arm | motionplanning | 1000 | 987 | 937 |
| StackPyramid-v1 | single arm | motionplanning | 1000 | 1000 | 1000 |
| TwoRobotPickCube-v1 | two robots | rl | 983 | 983 | 983 |
| TwoRobotStackCube-v1 | two robots | rl | 1007 | 1007 | 1007 |
| **total** | | 31 | **28676** | **28351** | **16749** |

* Lineage: `motionplanning` (scripted planner, ManiSkill contributors), `rl` (PPO policy
  rollouts, one file per controller: `pd_ee_delta_pos`, `pd_ee_delta_pose`,
  `pd_joint_delta_pos`) and `teleoperation` (10 click-and-drag human demos). Only the
  teleop demos are human. `PegInsertionSide-v1/rl/trajectory.json` has no `source_type`;
  it is taken from the folder name.
* Overlap: the three RL files of a task reuse the same env seeds and start from
  bit-identical initial states (different policies, different trajectories); PickCube
  teleop seeds 0–9 equal motion-planning seeds 0–9. `lineage["initial_state"] =
  maniskill/<env>/env_seed/<seed>` groups them. Exact duplicates exist inside the
  motion-planning files: PegInsertionSide has 312 episodes that repeat an earlier
  episode byte for byte (same seed), PullCubeTool 13; another 50 PullCubeTool pairs share
  a seed but differ. `lineage["duplicate_of"]` names the earlier identical episode. Count
  distinct trajectories or distinct initial states, never the file totals.
* Success: the JSON `episodes[].success` (equals the h5 `success[-1]` everywhere). All
  published episodes succeed except PegInsertionSide RL (975/1000). RL episodes end at
  the first success or at `max_episode_steps`.
* Recording: `obs_mode=none` (no observations, no images, no TCP channel); the old
  `PegInsertionSide-v1/rl/trajectory.h5` carries a meaningless 1-D `obs` array, ignored.
  `sim_freq`/`control_freq` are not in the JSON; all tasks use the SimConfig defaults
  (100 Hz physics, 20 Hz control) at the recorded commits, so time = state index / 20 Hz.
  Each episode has T actions and T + 1 states (terminal state included); the adapter
  keeps all T + 1 states and does not export actions.

## Adapter: `reachy_retarget/sources/maniskill.py`

No SAPIEN/ManiSkill import. Per episode it reads `env_states` and the JSON:

* **State layout** (ManiSkill `Actor.get_state` / `Articulation.get_state` at
  `baab60ed`): actors = p, q (wxyz), v, ω (13); articulations = root p, q, v, ω, then
  qpos(9), qvel(9) in the order `panda_joint1..7, panda_finger_joint1, panda_finger_joint2`.
  The order is checked twice: every frame must lie inside the URDF limits (joint 4 and 6
  ranges are asymmetric, so a permutation fails), and for `pd_joint_pos` files the first
  action equals the first 7 qpos exactly (0.0 rad in all motion-planning and teleop files).
* **Effectors**: one per articulation (`panda`, `panda_wristcam`,
  `panda_wristcam-agent-0/1`). Grasp center by FK of the pinned URDF from recorded qpos
  and root pose (`reachy_retarget.robot.urdf`). The URDF's `mimic` tag on
  `panda_finger_joint2` is ignored, because SAPIEN simulates both finger joints
  independently (they differ under contact). Contract frame measured from the model with
  the fingers open: +z = `panda_hand` origin → midpoint of the two fingertip pad
  collision boxes, +y = `panda_leftfinger` pad → `panda_rightfinger` pad, i.e.
  `R_contract = R_hand · diag(−1, −1, 1)`; the grasp center is fixed in the hand frame at
  z = 0.10365 m, 0.25 mm beyond `panda_hand_tcp`. Width = pad separation along +y minus
  the closed separation (= `q_finger1 + q_finger2`, 0–0.08 m); opening = width / 0.08.
  `side_hint` is `None`; the two-robot tasks' arms face each other across the table
  (agent-0 at (0, −0.75) yaw +90°, agent-1 at (0, 0.75) yaw −90°), the env's
  `left_agent`/`right_agent` names are kept in `provenance["effector_env_roles"]`.
* **Objects**: per-task table `TASKS` (roles and geometry from the task source at the
  commit recorded in each JSON; constants checked identical across those commits):
  cubes 0.02 m half size, `LiftPegUpright`/`PokeCube` peg box (0.12, 0.025, 0.025), ball
  r = 0.035, PlugCharger charger and receptacle and the PullCubeTool L tool as box unions
  (`kind: boxes` + AABB). PegInsertionSide peg and hole sizes are random per episode and
  not stored; they are regenerated from the episode seed (`RandomState(seed)`: half
  length U(0.085, 0.125), radius U(0.015, 0.025), hole offset) and accepted only if the
  regenerated radius equals the recorded initial peg height (exact for all 2000
  episodes); otherwise the geometry is left empty. `provenance["geometry_status"]`
  records "task constant", "regenerated ..." or "unknown". Roles: StackCube `cubeB` and
  StackPyramid `cubeB` are `support`; PegInsertionSide `box_with_hole` and PlugCharger
  `receptacle` (kinematic) are `receptacle`; `table-workspace` is a `support` box
  (2.418 × 1.209 × 0.92 m; top at ManiSkill z = 0, i.e. z = 0.9196429 after translation). Goal markers (`goal_site`, `goal_region`,
  non-colliding visuals) are kept in `provenance["goal_markers"]`, not as objects.
  Unknown actors become `manipulated` with empty geometry and are listed.
* **World frame**: ManiSkill's origin is on the table top and its ground plane is at
  z = −0.9196429 (TableSceneBuilder table height). The `SourceEpisode` contract puts the
  floor at z = 0, so the adapter translates everything by +0.9196429 m in z
  (`WORLD_Z_OFFSET`): robot roots (hence effector poses), actor tracks, goal markers and the
  scene. Afterwards the floor is at z = 0 and the table top at z = 0.9196429.
  `provenance["world_z_offset_m"]` and `provenance["world_frame"]` record it. Raw-state
  checks (PegInsertionSide initial peg height = radius) use the untranslated values.
* **Table crop (recorded fixture adaptation)**: the 2.418 × 1.209 m table leaves the
  objects about 0.5 m from any edge Reachy's base can stand at, so per episode the table is
  cropped in its own x/y only (height and top surface unchanged) to the axis-aligned region,
  in the table frame, covering the footprint of every object (all frames, receptacles
  included; goal markers are not objects and are not covered) plus 0.10 m on each side,
  intersected with the original extent (`primitive_scene.crop_supports`,
  `TABLE_CROP_MARGIN`). The cropped box is the `table-workspace` track geometry (tier-K
  footprint) and the scene's table; `geometry["uncropped"]` and
  `provenance["scene_adaptations"]` hold the original and cropped extents and the rule.
  Fixture examples: PickCube 0.31 × 0.27 m, PegInsertionSide up to 0.90 × 0.46 m,
  TwoRobotPickCube 0.52 × 0.46 m (table-frame x × y).
* **base_hint**: root x, y, yaw of the first articulation; every robot base (and z, now
  0.9196 m above the floor) in provenance. Single-arm robots stand on the table top at
  (−0.615, 0) facing +x, except RollBall (−0.1, 1.0, yaw −90°).
* **Scene**: a `SceneRef` rebuilt from primitives by `sources/primitive_scene.py` when every
  actor's geometry is known (all 13 tasks; PegInsertionSide only when the seed regeneration
  matches, otherwise `scene=None` with the reason in `provenance["scene"]`). The MJCF is in
  the translated world frame: ground plane at z = 0, table top at z = 0.9196429, table and kinematic actors
  (`receptacle`, `box_with_hole`) as static bodies, dynamic actors as bodies named by the
  object id (`geometry["body"]`) with a free joint `<id>_freejoint`, posed at frame 0 (MJCF
  pose and `initial_qpos`). There is no source robot; `validate.scene.build_scene` refuses
  an empty prefix list, so `robot_prefixes = ["no_source_robot/"]`, which matches nothing.
  Parameters, their sources and the mapping assumptions are embedded in the MJCF
  (`<custom><text name="primitive_scene_provenance">`) and copied to
  `provenance["scene_physical"]`.

  | parameter | value | source (ManiSkill `baab60ed` unless noted) |
  | --- | --- | --- |
  | material of every shape (actors, table, ground) | static 0.3, dynamic 0.3, restitution 0 | `utils/structs/types.py` `DefaultMaterialsConfig` → `physx.set_default_material` (`envs/sapien_env.py`); no catalogued task passes a material |
  | density | 1000 kg/m³ | SAPIEN 3.0.0b1 `ActorBuilder.add_*_collision(density=1000)` (pinned `sapien==3.0.0.b1`) |
  | L tool density | handle 500, hook 1000 kg/m³ | `pull_cube_tool.py` `_build_l_shaped_tool` |
  | cube 0.04 m | 0.064 kg | density × volume |
  | LiftPegUpright / PokeCube peg 0.24 × 0.05 × 0.05 m | 0.6 kg | `build_twocolor_peg` (one box collision) |
  | RollBall ball r 0.035 m | 0.180 kg | `build_sphere` |
  | PullCubeTool L tool | 0.25 + 0.25 = 0.5 kg (the 0.05 × 0.025 × 0.05 m overlap of the two boxes is counted twice, as in PhysX) | |
  | PegInsertionSide peg | 0.15–0.63 kg (seed-dependent size) | `peg_insertion_side.py` |
  | PlugCharger charger | 1000 kg/m³ base + prongs | `plug_charger.py` (dynamic); receptacle `build_kinematic` → static |
  | table | kinematic box 2.418 × 1.209 × 0.9196429 m, top at z = 0 → static | `utils/scene_builder/table/scene_builder.py` |
  | ground | static plane at −table height (ManiSkill z −0.9196429, translated to 0) | `scene_builder.py`, `utils/building/ground.py` |
  | gravity, sim rate | −9.81 m/s², 100 Hz (MJCF option; tier P overrides with 2 ms) | `types.py` `SceneConfig`, `SimConfig` |
  | PhysX solver (not mapped) | contact_offset 0.02, rest_offset 0, 15/1 iterations, TGS, PCM | `types.py` `SceneConfig` |
  | actor damping (not mapped) | linear 0, angular 0.05 1/s | PhysX SDK defaults, not set by SAPIEN |

  `DefaultMaterialsConfig`, `SceneConfig` and the actor builders are identical at every
  commit recorded in the JSONs (`652ad935`, `e77e4ff3`, `9d5e0e01`, `95ea99d4`, `ee5f8826`,
  `baab60ed`); `ecc579b7` (PickCube teleop) predates the `mani_skill/` layout and was not
  compared. PhysX → MuJoCo mapping (assumptions, recorded per scene): MuJoCo sliding
  friction = PhysX dynamic friction (static = dynamic here, so nothing is lost); MuJoCo's
  `max` friction combination equals PhysX's `average` for equal materials; `condim` 3 (PhysX
  has no torsional or rolling friction); restitution 0 = critically damped `solref`; PhysX
  hard contacts (`rest_offset` 0) approximated by stiff `solref (0.004, 1)`, `solimp (0.95,
  0.99, 0.001)`, `margin` 0 (`contact_offset` is a detection distance, not a force margin);
  actor damping not modelled. Reachy geoms have `priority` 1, so Reachy's friction (1.0,
  condim 4) governs every hand–object contact, not ManiSkill's 0.3. Rest check (no robot,
  tier-P options, 2 s): every task's actors stay within 1 mm and 0.01 rad of their initial
  pose (measured: drift ≤ 0.009 mm, rotation ≤ 2e-6 rad, penetration ≤ 0.09 mm for a stacked
  cube; `tests/test_primitive_scene.py`).
  Tier-P smoke (2026-10-07, floor at z = 0 and cropped table; motion planning, default
  `RetargetConfig` and `PhysicsConfig`; tier P never welds or teleports the objects):

  | episode | K | P | notes |
  | --- | --- | --- | --- |
  | PickCube traj_0 | fail: TCP rotation residual 0.061 rad over 11 frames | **pass** | cube lifted 0.27 m, final error 2.2 mm, grasp drift 0.7 mm |
  | PickCube traj_1 | pass | **pass** | final error 2.7 mm |
  | PickCube traj_2 | pass | **pass** | final error 3.4 mm (lift 0.019 m) |
  | PickCube traj_3 | pass | **pass** | final error 3.2 mm |
  | PickCube traj_4 | fail: TCP rotation residual 0.066 rad over 11 frames | **pass** | final error 2.7 mm |
  | StackCube traj_0 | pass | fail: carry_contact 0.94 (< 0.95) | cubeA placed within 6.8 mm |
  | StackCube traj_1 | pass | fail: grasp_drift 5.4 mm / 0.023 rad, carry_contact 0.93 | cubeA placed within 4.7 mm |
  | PegInsertionSide traj_0 | fail: TCP 32 mm / 0.36 rad, hand–peg drift 22 mm / 0.49 rad | fail: hand–object 1.3 mm, l_wrist_pitch margin, TCP 0.084 m, grasp drift 38 mm / 0.40 rad, peg 0.11 m off | wrist pitch limit (±30°) |
  | PegInsertionSide traj_1 | fail: TCP rotation 0.56 rad, hand–peg drift 30 mm / 0.47 rad | fail: l_wrist_pitch margin, arm 1.21 rad/s, TCP 0.088 m, grasp drift 63 mm / 0.99 rad, peg 0.097 m off | wrist pitch limit |

* **FK cross-check**: no file records a TCP or link pose (`obs_mode=none`, env_states
  hold only root and joint states). `crosscheck=True` compares `panda_hand_tcp` and both
  finger links with MuJoCo's own URDF importer (independent parser and FK). Physical
  check: while a cube is held, its center lies at the grasp center.

## ManiSkill verification (2026-10-07)

All 28676 episodes of the 31 files were adapted (1,975,055 states, ~1 min total).
MuJoCo FK cross-check on the first 10 episodes of every file: max position error
8.4e-16 m, rotation error < 1e-7 rad (float rounding). Opening spans 0–1 in most files
(pushing tasks close the gripper fully).

| dataset | episodes checked | T (states) | grasp-center xyz range (m, ManiSkill frame before the +0.9196 m z translation) | opening | object motion (max ptp) |
| --- | --- | --- | --- | --- | --- |
| PickCube-v1/motionplanning | 1000 | 50–104 | [−0.02, −0.05, 0.02]–[0.05, 0.06, 0.29] (3 eps) | 0.457–1.0 | cube 0.27 m |
| PickCube-v1/teleop | 10 | 83–133 | [−0.02, −0.06, 0.01]–[0.06, 0.07, 0.30] (3 eps) | 0.458–1.0 | cube 0.27 m |
| StackCube-v1/motionplanning | 1000 | 81–413 | [−0.09, −0.10, 0.02]–[0.06, 0.11, 0.18] (3 eps) | 0.457–1.0 | cubeA 0.20 m, cubeB 0.003 m |
| PegInsertionSide-v1/motionplanning | 1000 | 103–424 | [−0.10, −0.31, 0.01]–[0.11, 0.21, 0.18] (3 eps) | 0.341–1.0 | peg 0.45 m, box 0 |
| TwoRobotPickCube-v1/rl | 983 | 83–101 | [−0.39, −0.27, 0.01]–[0.06, 0.18, 0.31] (3 eps) | 0.0–1.0 | cube 0.35 m |

Grasp-center check (first 50 episodes, frames with the object above 3 cm and opening
< 0.7): cube center in the grasp frame, mean (x, y, z) = (−2.7, 0.0, 0.0) mm, std
(0.4, 0.2, 0.9) mm for PickCube motion planning; (0.1, 0.0, −0.5) mm for StackCube
motion planning; RL and teleop grasps are less centred (median |offset| 10–14 mm) but
still symmetric along the closing axis (|y| ≈ 0.1 mm). The recorded width while
holding a 40 mm cube is 36.6 mm (joint sum; PhysX contact offset), so width is a joint
measurement, not a contact gap.

## MolmoBot-Data (family `molmobot`)

* Repository: <https://huggingface.co/datasets/allenai/molmobot-data>, revision
  `159a5aecaf2ba73855940ad325b477b27d85da73`, license ODC-BY 1.0 (dataset card); helper
  scripts Apache-2.0 (`CODE_LICENSE`). Objects come from Objaverse/THOR assets with
  per-asset licenses; `commercial_episodes.parquet` lists the scene packages (and the
  episode indices within them) whose objects all have non-NC licenses.
* Layout: nine generation configs, each with `train_pkgs`/`val_pkgs` parquet tables
  (`path, shard_id, offset, size, inflated_size, part`) and `{train,val}_shards/NNNNN.tar`.
  A table row is one scene package `<config>_house_<id>.tar.zst`, a zstd tar stored as a
  byte range inside a shard. Each package holds `house_<id>/trajectories_batch_<i>_of_<n>.h5`
  (state only) followed by one MP4 per episode and camera. The `shards/` folders of three
  configs and the top-level `arrow_table.json` index an older copy of the same data
  (MP4s first in each package); they are not catalogued.
* Totals (revision above): 324,497 packages, 8.73 TB compressed, almost all MP4.
  From the catalogued sample the `.h5` share is 6.5–23 % per config, i.e. about 1.2 TB of
  non-image data and roughly 0.9 M trajectories (3 packages per config; rough).

| config | robot | task | packages | compressed GB |
| --- | --- | --- | --- | --- |
| DoorOpeningDataGenConfig | rby1m | door_open | 17,102 | 858.5 |
| RBY1OpenDataGenConfig | rby1m | open (drawers, doors of furniture) | 10,544 | 399.6 |
| RBY1PickDataGenConfig | rby1m | pick | 30,801 | 440.1 |
| RBY1PickAndPlaceDataGenConfig | rby1m | pick_and_place | 9,932 | 148.8 |
| FrankaPickOmniCamConfig | franka_droid | pick | 73,485 | 1628.3 |
| FrankaPickAndPlaceOmniCamConfig | franka_droid | pick_and_place | 92,782 | 3472.1 |
| FrankaPickAndPlaceColorOmniCamConfig | franka_droid | pick_and_place_color | 5,456 | 200.9 |
| FrankaPickAndPlaceNextToOmniCamConfig | franka_droid | pick_and_place_next_to | 58,164 | 1280.1 |
| FrankaPickAndPlaceOmniCamConfig_ObjectBackfill | franka_droid | pick_and_place (extra object types, not used for the released models) | 26,231 | 301.1 |

### Trajectory file contents

Per `traj_<k>` (T steps at `policy_dt_ms` = 100 ms for RB-Y1, 66 ms for Franka; no
simulator clock): `obs/agent/qpos` and `qvel` (JSON per step, by move group: RB-Y1
`base` (world x, y, yaw), `torso` (6), `left_arm`/`right_arm` (7), `left_gripper`/
`right_gripper` (2 slide joints), `head` (2); Franka `arm` (7), `gripper` (2 Robotiq
driver joints)), `actions/{commanded_action, joint_pos, joint_pos_rel, ee_pose, ee_twist}`
(JSON), `env_states/articulations/panda` (robot joints only, padded to 31; despite the
name also for RB-Y1), `obs/extra/robot_base_pose` (world xyz + wxyz), `tcp_pose` (Franka
grasp site; RB-Y1 left `ee_site`) or `left_tcp_pose`/`right_tcp_pose` (door opening), all
in the robot base frame on the floor/pedestal, `obj_start`/`obj_end` (constant: initial
and goal pose of the task object), `grasp_pose` (planned grasp, Franka), `grasp_state_*`
(JSON `held`/`touching` per gripper), `task_info` (JSON: success terms, `position_error`,
receptacle displacement, `joint_position` for open/door tasks), `door_state` (JSON:
hinge position, handle position, joint angle, opening fraction), `policy_phase`,
camera intrinsics/extrinsics and projected object points, `rewards`, `success`, `fail`,
`terminated`, `truncated`, and `obs_scene` (JSON: task type, instruction, referral
expressions and `frozen_config`, a base64 pickle of the generation config: robot name and
MJCF file, `init_qpos`, Franka pedestal `base_size`, initial poses of every scene object,
task object names, start/goal poses, added objects, articulated joint name). Files of
some configs also have `valid_traj_mask` and per-trajectory `stats`.

**Not stored: per-step object state.** There is no object qpos or free-body pose after
`t = 0`, and no scene MJCF. Initial poses of all scene objects are in the config.

### Catalog (`catalog/molmobot.yaml`)

* `sources` (whole files, `acquire.fetch`): README, `commercial_episodes.parquet`,
  helper scripts, the 17 package tables (all LFS-hashed except the scripts/README, which
  were hashed at catalog time), and the MolmoSpaces robot shards from
  <https://huggingface.co/datasets/allenai/molmospaces> revision
  `5f802a46816ffc3520e59b5ef49fa8dafee8eeca`: `mujoco/robots/rby1m/20251224` (10.7 MB;
  `rby1_v1.2_site_control.xml`; no LICENSE file in the package) and
  `mujoco/robots/franka_droid/20260127` (42.9 MB; FR3 Apache-2.0 + Robotiq 2F-85
  BSD-3-Clause). These are the robots named in every trajectory's `frozen_config`.
  25 files, 70 MB.
* `molmobot_packages`: 27 packages (per config 2 train + 1 val; ObjectBackfill 3 train),
  all listed in `commercial_episodes.parquet`, size between the 15th and 45th percentile,
  ordered by sha1(part/path). Each records shard path and publisher shard SHA-256, byte
  offset/size, `range_sha256` of the compressed range and the SHA-256, size and
  trajectory count of every `.h5` member (both computed by one streaming pass of the
  generator). 65 trajectories, 11,602 steps, 31 `.h5` files, 46 MB on disk; fetching
  transfers 378 MB (the MP4s are streamed and discarded).
* `reachy_retarget/sources/molmobot_fetch.py`: `fetch_packages(ids, root)` streams each
  range (HTTP Range; refuses a 200 response), decompresses with `zstandard` or the `zstd`
  CLI, writes only the catalogued `.h5` members to
  `raw/molmobot/<config>/<split>/part<k>/<house>/` after checking member and range
  digests, applies the 50 GB reserve and records ledger entries.
  `unpack_robot_assets(root, entry)` extracts a fetched robot shard into
  `raw/molmobot/robots/<name>/<version>/` with a `MANIFEST.json` of file digests.

```python
from reachy_retarget.acquire import fetch, load_catalog
from reachy_retarget.sources import iter_episodes, molmobot  # import registers 'molmobot'
from reachy_retarget.sources.molmobot_fetch import fetch_packages, load_packages, unpack_robot_assets

cat = load_catalog()
ids = [k for k, e in cat.items() if e.family == "molmobot"]
fetch(ids, "data")
for k in ids:
    if cat[k].kind == "assets":
        unpack_robot_assets("data", cat[k])
fetch_packages(list(load_packages()), "data")
```

### Adapter: `reachy_retarget/sources/molmobot.py`

* **Robot models**: the pinned MolmoSpaces MJCF compiled with MuJoCo (includes inlined,
  sub-model and mesh paths resolved; missing meshes become 1 mm spheres, which keeps body,
  site and joint kinematics exact). Groups from `obs/agent/qpos` are written by joint
  name. Robotiq spring-link and follower joints are set equal to the driver; this
  satisfies the closed-chain `connect` constraints to < 1e-5 m (checked at load).
* **Frames**: RB-Y1 base joints are world x, y, yaw, so the model frame is the world
  (robot root 5 mm above the floor). Franka: world ← `fr3_link0` =
  `robot_base_pose` · translate(0, 0, `base_size[2]`) (pedestal height, 0.58 m in the
  sample).
* **Effectors**: RB-Y1 `left`/`right` (side hints), Franka `gripper`. Contract frame
  measured from collision geometry with the fingers open: contact-face centres = collision
  points within 1 mm of the innermost point of each finger, grasp center = their midpoint,
  +z = palm (finger parent body) → midpoint, +y = finger 1 → finger 2. RB-Y1:
  `R = R_ee_site · diag(−1, −1, 1)`, centre 16.5 mm behind `ee_site` along +z; Franka:
  `R = R_grasp_site`, centre 14.8 mm behind `grasp_site` (with the Robotiq pads open; the
  pads move ~3 cm forward while closing, the centre stays fixed in the site frame). Width
  = contact-face gap clipped at 0 (open: RB-Y1 100.2 mm, Robotiq 86.9 mm); opening =
  width / open gap, clipped to [0, 1]. RB-Y1 qpos slightly exceeds the finger limit
  (−0.056 vs −0.05) when open, so opening saturates at 1.
* **Base / torso**: RB-Y1 `base` = (x, y, unwrapped yaw) per step, `torso_height` =
  world z of `link_torso_5` (arm and neck mount; 1.31 m upright, down to 1.13 m when the
  torso bends in open tasks), regime `mobile_manipulation`. Franka: `base=None`,
  `base_hint` from `robot_base_pose`, regime `tabletop`.
* **Objects** (all `t = 0` only, `valid[1:] = False`, measured rows noted in geometry):
  pick object (`manipulated`, `pickup_obj_start_pose`), place receptacle
  (`receptacle`; `fixture` for `next_to`, where it is a reference object), articulated
  furniture root for `open` (`fixture`). Door opening: `<door>:handle` is valid at every
  step (position from `door_state`; orientation **derived** as yaw of hinge → handle).
* **Articulations**: door hinge angle (`door_state.joint_angle`, key = door body) and the
  opened joint of `open` tasks (`task_info.joint_position`, joint name from the config).
* **Instruction**: `task_description`; referral expressions are in provenance.
  **Success**: `success[-1]`. **Lineage**: `generated=False, synthetic=True` (planner
  rollouts, each its own scene/task sample; several trajectories share a house).
* **Scene**: `None`. Houses (`procthor-objaverse`, ~0.1 MB XML per house) and objects
  are published in `allenai/molmospaces`, but the scene assembly (house + objects +
  datagen-time added receptacles) is not reproduced; the initial poses of all objects are
  kept in `provenance["scene_initial_object_poses"]` for a later SceneRef.
* **Provenance**: file and package digests, config/split/part/house/batch, robot model
  ids and digests, FK check, grasp-state flags, final task info, goal pose,
  `commercial_valid_episodes` of the package.
* `rigid_grasp_track(ep, obj)` derives (never stores) a pick-object track: static until
  first contact, rigidly attached during the longest `held` run. RB-Y1 files never set
  `held` (all 10 RB-Y1 pick/place/open trajectories in the sample, including successes),
  so it only works for Franka.

### Verification (2026-10-07)

All 65 catalogued trajectories adapted (3.4 s). FK of the pinned models vs. recorded TCP
poses (pedestal/base frame), per effector-trajectory:

| robot | effector-trajectories | position median / p99 / max | rotation median / p99 / max | gate (p99 ≤ 2 mm, 2 mrad) |
| --- | --- | --- | --- | --- |
| rby1m (door: both arms; others: left) | 30 | ≤ 0.30 / 0.80 / 0.89 mm | ≤ 0.76 / 1.21 / 1.22 mrad | 30/30 |
| franka_droid | 45 | ≤ 0.19 / 0.63 / 0.80 mm | ≤ 0.44 / 2.25 / 5.1 mrad | 44/45 |

Isolated rotation outliers coincide with large joint steps (0.12 rad/step), i.e. a
sampling skew between `qpos` and the TCP observation, not a model mismatch.

| dataset | episodes | T | success | opening min | objects / articulations |
| --- | --- | --- | --- | --- | --- |
| DoorOpening train / val | 8 / 2 | 143–217 | 10 | 0.00 | handle valid all rows; hinge 0 → 1.05–1.08 rad |
| RBY1Open train / val | 2 / 1 | 106–201 | 3 | 0.00 | furniture t0; joint −1.21 … 0.19 |
| RBY1Pick train / val | 3 / 1 | 85–103 | 4 | 0.17 | pick object t0 |
| RBY1PickAndPlace train / val | 2 / 1 | 125–170 | 3 | 0.00 | pick object, receptacle t0 |
| FrankaPick train / val | 15 / 2 | 51–78 | 17 | 0.12 | pick object t0 |
| FrankaPickAndPlace train / val | 6 / 2 | 235–414 | 8 | 0.00 | pick object, receptacle t0 |
| FrankaPickAndPlaceColor train / val | 10 / 2 | 240–285 | 12 | 0.13 | pick object, receptacle t0 |
| FrankaPickAndPlaceNextTo train / val | 4 / 1 | 267–327 | 5 | 0.12 | pick object, reference t0 |
| FrankaPickAndPlace ObjectBackfill train | 3 | 246–272 | 3 | 0.01 | pick object, receptacle t0 |

All sampled trajectories are successes (the published data keeps valid trajectories;
2 carry `valid_traj_mask = False` and are still adapted, flagged in provenance).
Franka pick check of the derived track: |object–goal distance at the end of the held
run| vs. the recorded `task_info.position_error` differs by 3.3 mm median (max 23 mm,
17 episodes; 16 held to the end). Door handle yaw change equals the hinge angle to
< 2 mrad.

### Known gaps

* No per-step object state: grasp-phase hand–object checks and object-final-pose physics
  gates have nothing to compare against except the derived Franka track.
* RB-Y1 `grasp_state` flags are always false; `tcp_pose` is the left arm even when the
  right arm acts.
* No SceneRef (MolmoSpaces scene assembly not implemented).
* The rby1m package has no license file; its redistribution terms are unverified.
* Opening-joint semantics for `open`: `task_info.joint_position` is assumed to be the raw
  joint position (units of the joint); success uses a normalised threshold (0.67).

## BiGym (family `bigym`)

* Code: <https://github.com/chernyadev/bigym> (Apache-2.0), pinned at master head
  `d98124454dfa3bc697f75536daec5989a47bfa8d` (2026-05-06, version string 4.1.0). Catalog
  id `bigym/code/bigym-d98124…zip`: GitHub's archive of that commit, 59,310,942 bytes,
  SHA-256 `4a48cda1…a9a5` (computed on download). Scene assets under
  `bigym/envs/xmls` carry per-asset CC0 / CC BY / CC BY-NC licenses
  (`3D_MODELS_ATTRIBUTION.md`); a scene's asset set inherits them.
* Demonstrations: `bigym_data` release `v0.9.0` (the `DEMO_VERSION` of `bigym/const.py`
  at the pinned commit; published 2024-07-17),
  `https://github.com/chernyadev/bigym_data/releases/download/v0.9.0/demonstrations.zip`,
  123,362,114 bytes, SHA-256 `a7fd1e7f…86d2` (the release declares no digest; computed on
  download and pinned). Catalog id `bigym/v0.9.0/demonstrations.zip`, `kind: low_dim`.
* Content: **40 tasks, 1,933 recordings, 3,402 files** (per-task counts and derived
  instructions in `catalog/bigym.yaml`, `tasks:`). Each file is a *lightweight*
  safetensors recording: `info_demo_action` (T × 15 or 16) at 500 Hz, termination and
  truncation flags, and metadata (seed, env name, action mode, floating DOFs, recorded
  `bigym 4.0.0` / `mujoco 3.1.5`). No states, no images, no language, no success label.
  Termination flags mean success **or** failure at recording time and are reported
  separately, never as success.
* Overlap: 34 tasks store most recordings twice, as `…_absolute` and `…_delta` action-mode
  conversions of the same recording (same UUID). They are one demonstration; the
  absolute file is replayed, a delta file only when no absolute file of that UUID
  exists, and the other copy is listed in `lineage["other_versions"]`.
* Hardware: Unitree H1 upper body with two Robotiq 2F-85 grippers on a position-actuated
  "floating base" (pelvis slide x/y, hinge rz; 35 tasks also z). Actions: pelvis target
  increments, 10 absolute arm joint targets, two normalized gripper commands.

### Replay (`reachy_retarget/sources/bigym_replay.py`)

States exist only after replaying the actions. BiGym pins `mujoco==3.1.5`,
`numpy==1.26.*`, dm_control 1.0.19, a Gymnasium fork and Mojo, which conflict with this
project (numpy 2, MuJoCo ≥ 3.2), so replay runs in an **isolated interpreter**
(`data/envs/bigym-replay`, Python 3.11.14, 436 MB), created with uv:

```sh
uv venv --python 3.11 data/envs/bigym-replay
uv pip install --python data/envs/bigym-replay/bin/python numpy==1.26.4 mujoco==3.1.5 \
  dm_control==1.0.19 safetensors==0.6.2 pyquaternion mujoco_utils pyyaml imageio wget \
  "gymnasium @ git+https://github.com/stepjam/Gymnasium.git@2cbc4d34c3124ef4921977fe4ed1e4e532f33ed3" \
  "mojo @ git+https://github.com/stepjam/mojo.git@ccec1deaf9bde9fa7a2ac051c9376ad5a81a3aad"
uv pip install --python data/envs/bigym-replay/bin/python --no-deps \
  data/raw/bigym/code/bigym-d98124454dfa3bc697f75536daec5989a47bfa8d.zip
```

Resolved versions: bigym 4.1.0 (from the pinned archive, recorded via PEP 610
`direct_url.json`), mujoco 3.1.5, numpy 1.26.4, dm-control 1.0.19, gymnasium 0.29.2
(fork commit `2cbc4d3`), mojo 0.1.1 (`ccec1de`), safetensors 0.6.2, scipy 1.17.1,
lxml 6.1.3, labmaze 1.0.6. `dearpygui` and `pyopenxr` (VR/GUI only) are not installed.

```sh
data/envs/bigym-replay/bin/python reachy_retarget/sources/bigym_replay.py \
  --zip data/raw/bigym/v0.9.0/demonstrations.zip --out data/derived/bigym/replay-v1 --jobs 7
```

(or `reachy_retarget.sources.bigym.replay(...)`). The script never imports
`reachy_retarget` and uses no network. It follows `DemoPlayer.validate_in_env`: official
`Metadata`/`LightweightDemo` loading, `Metadata.get_env(500)`, `env.reset(seed=demo.seed)`,
`env.step(action, fast=True)` for every recorded action without decimation or clipping,
reading `env.success` after every step (without stopping at the first success). Per
recording it writes `<task>/<uuid>.npz`:

* `qpos`, `qvel`, `time` (`mjData.time`), `step`, `success` for the reset state and every
  10th post-action state plus the last one (50 Hz);
* `mjcf`: the scene **after the seeded reset**. BiGym's reset moves furniture and props
  by editing compiled `body_pos`/`body_quat`, and disables unused props through physics
  bindings (`Prop.disable()`: geom `contype`/`conaffinity` = 0, free-joint damping 1e7);
  none of this is in dm_control's `to_xml_string()`. The exporter writes body poses, geom
  collision flags and joint damping (a `<freejoint>` becomes `<joint type="free">`)
  back into the MJCF and recompiles. Every numeric model array is compared: remaining
  differences are only qpos0-derived fields, compiler bookkeeping (BVH, mesh graphs,
  per-body collision summaries), inertial frames of massless bodies, and texture pixels
  (one record); body poses at the reset state match to ≤ 1e-15 m (`meta.model_export`,
  `ok` for all 1,933 records);
* `meta`: zip member + SHA-256, zip SHA-256, seed, recorded and runtime versions, success
  summary (`success_any`, `success_final`, `first_success_step`), recorded termination
  flags, other stored versions, the exact failing action and traceback on error.

Meshes and textures are written once to `replay-v1/assets/` (dm_control content-hashed
names). Failed replays are saved with the states up to the failure.
`--clip-actions` is a labeled variant applying BiGym's own
`DemoConverter.clip_actions` (`meta.actions_clipped`, `lineage.replay_variant`); it is
never the default and its results are reported separately.

### Adapter (`reachy_retarget/sources/bigym.py`)

`iter_episodes("bigym", <record or folder>)` compiles the record's MJCF with the
project's MuJoCo (assets from the sibling `assets/` folder), sets `qpos`, runs
`mj_kinematics` and builds a `SourceEpisode`:

* **Effectors** `left`/`right` (`side_hint` = name). The contract frame is derived from
  geometry at the open reference configuration (qpos0): grasp center = midpoint of the two
  pad bodies' collision boxes, +z = gripper `base` (palm) origin → that midpoint, +y = left
  pad → right pad. It is a fixed transform of the palm: identity rotation, offset
  (0, 0, 0.13067) m (the BiGym `pinch` site is at 0.145 m). `opening` = 1 − mean
  normalized driver-joint angle (range 0–0.8 rad); `width` = gap between the inner pad
  faces along +y (0.0854 m open, ~0 when the pads touch).
* **Base**: `h1/pelvis` world pose (x, y, yaw, yaw unwrapped); pelvis height is
  `torso_height`. `regime = "mobile_manipulation"`, no `base_hint`.
* **Objects**: all free bodies outside the robot (`manipulated`, AABB geometry when assets
  are present; props BiGym disabled, i.e. without collision geometry, only in
  `provenance["inactive_free_bodies"]`) plus static furniture roots (`table`/`counter` →
  support, `drainer`/`rack` → receptacle, others fixture). The floor is skipped.
* **World frame**: the floor plane (body `floor`, `world.xml`) is at z = 0 in every
  checked record, so poses are used unchanged (`provenance["floor_z_in_source_m"] = 0`,
  `world_offset_m = [0, 0, 0]`). The adapter measures it per record and would translate
  all tracks (dropping the scene) if it ever differed.
* **Articulations**: hinge/slide joints outside the robot grouped by root body.
* **Success** = `success_any` of a complete replay; `None` for an incomplete replay
  (replay error), with the error in `provenance["replay_error"]`.
* **Instruction**: derived from the task class docstring (e.g. "Open top drawer of the
  cupboard"); `provenance["instruction_source"]` says so.
* **Scene** (`SceneRef`): the seeded MJCF, the asset bytes, `robot_prefixes = ["h1/"]`,
  initial free-joint and articulation qpos. Without assets (unit-test fixtures) mesh
  geoms become placeholders, `scene=None` and `state_route = bigym_replay_kinematic_only`.
  `with_scene=False` skips the assets (compiling is ~10x faster).

Simulation assumptions (in every episode's provenance): states are reconstructed by
action replay; the 4.0.0 recording code is not public, so 4.1.0 replays with the recorded
MuJoCo 3.1.5; the floating base is a position-actuated pelvis with animated legs (no legged
locomotion); kinematics are recomputed with the project's MuJoCo (body poses equal to the
3.1.5 ones to 4e-16 on checked records).

## BEHAVIOR-1K 2025 challenge (family `behavior`)

* Repositories (all MIT per dataset card):
  [`behavior-1k/2025-challenge-demos`](https://huggingface.co/datasets/behavior-1k/2025-challenge-demos)
  at `33639692425296185f245aaa648530178c7b3d2b`,
  [`behavior-1k/2025-challenge-rawdata`](https://huggingface.co/datasets/behavior-1k/2025-challenge-rawdata)
  at `b3c9a714b8730701c480e77173d8aa4a8f19e1c0`, and the R1 Pro URDF + import config from
  [`behavior-1k/omnigibson-robot-assets`](https://huggingface.co/datasets/behavior-1k/omnigibson-robot-assets)
  at `bc61e5e3c39c6fd347fd318cfe9d23d73e9d193b` (same git blob `ff1b63ce` as at the 2026
  head). The OmniGibson scenes and objects the episodes use are BEHAVIOR Data Bundle assets
  (non-commercial, encrypted, no redistribution); none are catalogued.
* Release: 10,000 human JoyLo teleoperation episodes, 50 tasks × 200, 119,094,660 frames at
  30 Hz (avg. 6.6 min). Robot Galaxea R1 Pro (holonomic base, 4-DoF torso, two 7-DoF arms,
  parallel grippers, `grasping_mode = assisted`), OmniGibson 3.7.0-alpha (git `875b5186`),
  og_dataset 3.7.0rc22. Recorded robot-assets hash `edec8837` is not public.
* Full-release sizes (HF tree API): `data/` parquet 164.5 GB, `meta/episodes/` JSON 19.0 GB
  (mostly `ins_id_mapping` and the full `scene_file`), `annotations/` 0.13 GB, rawdata HDF5
  792.6 GB, `videos/` (mp4 RGB/depth/seg, ~1.5 TB total dataset) not catalogued, nor
  `meta/episodes_stats.jsonl` (0.37 GB of feature statistics). The parquet files contain no
  images, so the low-dim release needs no stripping.
* Catalogued (`catalog/behavior.yaml`): the first 10 episodes by index of 8 tasks —
  turning_on_radio (0), picking_up_trash (1), cleaning_up_plates_and_food (3),
  set_up_a_coffee_station_in_your_kitchen (10), putting_shoes_on_rack (22),
  hanging_pictures (34), attach_a_camera_to_a_tripod (35), make_microwave_popcorn (40):
  80 episodes, 459,624 frames, parquet 0.655 GB + episode JSON 0.118 GB + annotations
  0.5 MB (= 0.774 GB non-image demos), plus `meta/{info.json,tasks.jsonl,episodes.jsonl}`
  and the raw HDF5 of the same 80 episodes (2.503 GB, optional). Fetched locally: all demo
  files (0.778 GB) and 40 raw files (0.704 GB: all of tasks 0, 34, 35, two each of 1, 3,
  10, 22, 40). Parquet/HDF5 digests are HF LFS sha256; JSON/URDF/YAML digests were computed
  after checking the git blob sha1 at the pinned revision.
* Overlap: the parquet and raw HDF5 of one episode index are the same demonstration (the
  parquet observations were produced by replaying the raw state); count episodes once.
  `behavior-1k/2026-challenge-{demos,rawdata}` reuse the same episode numbering (e.g.
  `task-0002/episode_00021890`); do not count them as independent before checking overlap.

### Layout (verified)

* `observation.state` (256 float64) = `PROPRIOCEPTION_INDICES["R1Pro"]` of
  `omnigibson/learning/utils/eval_utils.py` (BEHAVIOR-1K v3.7.2): joint qpos (28: 6 virtual
  base joints relative to the spawn frame, torso 4, arms interleaved L/R, grippers 2+2),
  sin/cos/vel/effort (efforts are invalid per the publisher), `robot_pos` and roll-pitch-yaw
  cos/sin of `base_link` in world, `robot_2d_ori`, per-arm qpos, `eef_{side}_pos/quat`
  (`{side}_eef_link` relative to `base_link`, quaternion xyzw), gripper qpos (2 prismatic
  fingers, 0-0.05 m each), trunk qpos, base qpos. The `grasp_{side}` entries listed in the
  config's `proprio_obs` are not in the 256 values.
* `observation.task_info` = the BehaviorTask low-dim obs in `task_obs_keys` order (episode
  JSON): per BDDL instance `_real`, `_pos` (world, object root link), `_ori_cos/_ori_sin`
  (roll-pitch-yaw), and for non-agent instances `_in_gripper_left/right`. Only
  task-relevant (BDDL-scoped) non-system objects; no articulation joint states.
* Raw HDF5 `data/demo_0`: `action` (T, 23), `reward`, `terminated`, `truncated` (T),
  `state` (T + 1, n) flat serialized sim state (object poses appear as pos + quat xyzw at
  scene-dependent offsets) and `state_size`; `data.attrs` `config`, `scene_file`.

### Adapter: `reachy_retarget/sources/behavior.py`

Input: one `data/task-XXXX/episode_N.parquet` (or a task folder); the episode JSON,
annotation and `meta/tasks.jsonl` are found from the layout, the raw HDF5 from the
sibling `2025-challenge-rawdata/` folder, the URDF from the catalog under the data root
(or `urdf=`).

* **Effectors** `left`/`right`: recorded `eef_{side}` pose composed with the recorded
  `base_link` world pose. OmniGibson defines `{side}_eef_link = {side}_gripper_link ·
  translate(0, 0, −0.06) · Ry(π)` (`r1pro_source_cfg.yaml`) with +z toward the fingertips,
  y the finger axis, and assisted-grasp points symmetric about its origin; the finger
  meshes put the distal pad region at gripper z −0.048 … −0.078 m, i.e. centred within
  ~3 mm of the eef origin. So the eef origin is used as the grasp center and
  `R_contract = R_eef · diag(−1, −1, 1)` (+y = finger 1 → finger 2). `width = q_f1 + q_f2`
  (the inner faces meet at q = 0), `opening = width / 0.10`.
* **World frame**: OmniGibson world translated by `world_z_offset_m` = −median(`base_link`
  z) so the floor the robot stands on is at z = 0 (`base_link` origin = bottom of the robot
  per `misc/metadata.json` `base_link_offset`/`bbox_size`). On the 8 checked episodes the
  offset is −0.005 m and `base_link` z varies by ≤ 1.2 mm within an episode; the BDDL
  `floor.n.01` object root is at z ≈ −0.15 m (object origin, not the surface).
  `house_double_floor_upper` scenes are loaded on their own and also stand at ~0.005 m.
* **Base** (x, y, `robot_2d_ori`) of `base_link`; `base_hint` = first frame;
  `torso_height` = world z of `torso_link4` (arm/head mount) by URDF FK (0.54–1.11 m in the
  subset). `regime = mobile_manipulation`, `scene = None` (OmniGibson/Isaac Sim house,
  `provenance["scene_model"]`).
* **Objects**: every non-agent BDDL instance of `task_info`, keyed by scene object name,
  pose = pos + quaternion from roll-pitch-yaw, `valid = _real`. Geometry is only
  `{kind: asset, category, model, scale, fixed_base, synset, bddl_inst}` (meshes are
  licensed assets). Roles: carried (derived, below) or annotated `manipulating_object_id` →
  manipulated; target of `place in`/`pour`/`insert` skills → receptacle; target of `place
  on`/`pick up from`/`hang`/… or `floor.n.01` → support; `fixed_base` → fixture; moved
  > 2 cm → manipulated; else fixture.
* **Derived carry label** (`provenance["carry_intervals"]`, rule in `carry_rule`): the
  released `_in_gripper` flags are −1 on every frame of all 80 episodes (the replay does
  not re-create the assisted-grasp constraint), so carrying is inferred: object speed
  > 3 cm/s, object still in the grasp frame (< 3 cm/s), within 0.25 m of the grasp center,
  gripper not fully open, ≥ 0.5 s, gaps < 0.5 s bridged.
* **Success**: raw `terminated[-1]` when the raw HDF5 is present, else `None`.
  `provenance["success"]` keeps the first success step and reward sum.
* **Instruction** = `tasks.jsonl` text; skills from the annotation in `provenance["skills"]`.
* **Articulations**: none. Cabinet/fridge/microwave joint positions are only in the raw
  serialized state, whose layout depends on the scene's object registry; decoding it needs
  OmniGibson's `Serializable` layout for that scene (object order and per-object state
  sizes from the scene registry), which is what the previous phase was blocked on
  (`legacy/reachy_retarget/adapters/behavior.py`). Not attempted here.
* Lineage: human teleoperation, not generated; `initial_state =
  behavior/<task>/instance/<activity_instance_id>`, `raw_source` names the raw file.

### BEHAVIOR verification (2026-10-07)

All 80 catalogued episodes (459,624 frames) adapted.

* FK: `r1pro.urdf` FK of the recorded torso/arm/finger qpos composed with the eef offset
  reproduces the recorded `eef_{side}` poses on every frame: max 3.9e-6 m, 2.1e-6 rad
  (float32 storage). `robot_2d_ori` equals the Euler yaw (≤ 7.4e-7 rad); the agent
  `task_info` position equals `robot_pos` exactly.
* Episode length equals `episodes.jsonl` `length` for all 80; the summed base path equals
  `distance_traveled` (ratio 0.999–1.000), confirming the base channel. The
  `{left,right}_eef_displacement` fields are 1.2–6.3× smaller than the world grasp-center
  path length (they are not world-frame path lengths).
* Raw HDF5 (40 episodes): step count equals the parquet frame count for all; 39 have
  `terminated[-1] = True` (first success 5–930 frames before the end), episode
  `task-0034/episode_00340020` (hanging_pictures) never terminates and is stored with
  `success = False`. The release therefore must not be assumed all-successful.
* Per task (10 episodes each): frames 1,544–17,411; objects 3–10 per episode; every
  episode has at least one derived carry interval; max opening 1.0 everywhere, minimum
  opening per task 0.00–0.09 except putting_shoes_on_rack (0.50) and
  attach_a_camera_to_a_tripod (0.30); no invalid object rows. Median object-origin distance from the grasp center while carried
  4–19 cm (object root frames are not grasp points, so this is not a centring test).
* The per-episode FK check can be disabled with `fk_check=False`.
* Tests: `tests/test_sources_behavior.py` (fixture: 51 rows of `episode_00000010`, ~98 KB).

## RoboCasa365 (family `robocasa`)

* Release: RoboCasa365 v1.0 datasets as listed by the official downloader
  (`robocasa/scripts/download_datasets.py`) at RoboCasa commit
  `456174f62b89b8fca99eaaf33949c29fec9cfc2a` (main, 2026-09-25; tag v1.0 = `8f3c96e`).
  One uncompressed `lerobot.tar` per task / split / source on Box
  (`robocasa/models/assets/box_links/box_links_ds.json`). License: datasets and assets
  CC BY 4.0, code MIT (RoboCasa README). No official HF copy with simulator extras exists:
  `lerobot/robocasa_target_human_unified` and the third-party per-task mirrors hold only
  parquet + videos (no MJCF/states) and are not catalogued; they overlap the target human
  data and must not be counted again.
* Catalog `catalog/robocasa.yaml`, generated by
  `sources.robocasa_fetch.generate_catalog(...)`:
  * `robocasa_tars`: 410 tars, 501.25 GB, 316 tasks. Pinned by Box file id, file version id
    and SHA-1 (Box publishes no SHA-256) plus size. Totals: pretrain atomic human 65 tars /
    6.1 GB, pretrain composite human 235 / 107.1 GB, target atomic human 18 / 9.2 GB, target
    composite human 32 / 52.4 GB, pretrain atomic MimicGen 60 / 326.5 GB. Each MimicGen tar has
    `seed` = the human dataset of the same task (registry `human_path`; for
    OpenToasterOvenDoor the MG folder predates it, see its `note`) and lineage
    `generated: true`. 35 registry paths (`mg_5x5`/`mg_5x1` variants and one human path) have
    no Box link and are listed under `registry_without_download`. Pretrain and target human
    sets are separate collections; MG demos are not independent demonstrations.
  * `sources` (whole files for `acquire.fetch`): HF `robocasa/robocasa-assets@1b92c3d`
    (`textures`, `generative_textures`, `fixtures`, `objaverse`, `aigen_objs` zips; LFS
    SHA-256), Box `fixtures_lightwheel.zip` / `objects_lightwheel.zip` (SHA-256 computed at
    catalog time from a stream whose size and Box SHA-1 matched), the GitHub source zip of the
    pinned commit (its `models/assets` tree: cabinets, handles, counters, arenas) and the
    robosuite 1.5.2 wheel (family `robosuite`; robot, base and gripper meshes). Asset total
    ≈ 11.4 GB. `asset_marker` is the recorded path fragment after which a reference equals the
    zip member name (`robocasa/models/assets/`, object zips `.../assets/objects/`).
* Tar layout (LeRobot v2.1, 20 Hz): `data/chunk-*/episode_k.parquet` (16-d state: base
  position + quaternion of site `mobilebase0_center`, grip-site position and hand quaternion
  relative to it, finger qpos; 12-d action; `next.reward`, `next.done`), `meta/`, `videos/`
  (3 × 256² MP4 per episode, ~80 % of the bytes) and `extras/` (`dataset_meta.json`; per
  episode `ep_meta.json`, `model.xml.gz`, `states.npz`). Human tars order members data,
  extras, meta, videos. **The MimicGen tar inspected (OpenDrawer
  `mg/demo/2025-08-20-21-55-00`, 9511 episodes) has no `extras/`**: only parquet
  observations/actions, no object or fixture state and no MJCF, so it cannot be adapted.
  Human tars of all four split/type groups were checked to carry extras.
* Acquisition: `sources.robocasa_fetch.fetch_tars(ids, root)` streams the whole tar once,
  verifies size and Box SHA-1, resumes dropped connections with HTTP Range, discards image
  members in memory and writes everything else to
  `raw/robocasa/<tar dir>/lerobot/...` with `tar_members.json` (offset, size, SHA-256 of
  each kept member, dropped counts) and a ledger record. Human tars keep ≈ 20 % of their
  bytes (extras ≈ 0.18 MB/episode). `fetch_asset_subset(rels, root)` copies single asset
  members (CRC-checked, read by Range from a zip's central directory) into
  `raw/robocasa/asset_subset/` when the full archives do not fit.

### Adapter: `reachy_retarget/sources/robocasa.py`

`read_robocasa(dir)` (registered as `robocasa`) takes an extracted dataset directory. Per
episode it decompresses the recorded MJCF, resolves assets, compiles it with the shared
robosuite replay (`_Model`), sets every `states.npz` row and runs `mj_forward`.

* **Effector** `gripper0_right` (PandaOmron has one arm): Panda grasp-center convention of
  the robosuite adapter (pad midpoint, +z approach, +y closing), width/opening from the
  pads; `side_hint=None`.
* **Base**: world pose of body `mobilebase0_base` (x, y, unwrapped yaw); the base joints act
  in the frame of `robot0_base`, which sits at (10, 10, 0) in every episode, so joint values
  are not world coordinates. `base` is set only when the base moves (> 1 cm or > 0.01 rad
  from frame 0); `base_hint` is always the initial pose. `torso_height` =
  `mobilebase0_joint_torso_height` (0–0.34 m).
* **Regime** (derived): `tabletop` when the base never moves, else `mobile_manipulation`
  when there are task objects or an operated referenced fixture, else `navigation`.
* **Objects** (derived roles, from `ep_meta`): `object_cfgs` not named `distr*` are task
  objects (`graspable: false` → receptacle, else manipulated); distractors are listed in
  `provenance["distractors"]` with their displacement. Fixtures named in `fixture_refs` get
  a static track of their root body with a role from their class (Counter/Stove… →
  support, cabinets/drawers/appliances/sinks → receptacle, else fixture). Objects placed
  inside an articulated fixture (e.g. `drawer_obj` in OpenDrawer) are task objects of the
  task definition but move with the drawer rather than being grasped.
* **Articulations**: every scene fixture joint (hinge/slide outside the robot) keyed by
  fixture name (`articulations="task"` keeps only `fixture_refs` fixtures);
  `provenance["task_articulations"]` and `["articulation_motion"]`.
* **Instruction** `ep_meta["lang"]`; **success** = max parquet `next.reward` ≥ 1.
* **World frame**: RoboCasa's room floor top is at z = 0 (checked per episode from the
  `floor*_room_main` boxes, refused if off by > 5 mm); no translation is applied.
* **Checks**: SHA-256 of every file read against `tar_members.json`; replayed base
  position, grip-site position relative to `mobilebase0_center` and finger qpos against the
  parquet `observation.state` (`provenance["observation_check"]`).
* **Assets / routes**: RoboCasa references resolve through all local catalogued archives
  plus `asset_subset/` (`provenance["asset_sources"]`, `["assets_from_subset"]`), robosuite
  references through the robosuite wheel of the recorded version. Asset names are the whole
  relative path flattened with `__`, because MuJoCo matches asset base names
  case-insensitively (`Prop.obj` / `prop.obj` of different objects). Missing assets give
  `mjcf_kinematic_only` (exact kinematics, no scene, empty AABBs).

### RoboCasa verification (2026-10-07)

Local samples (2.6 GB under `raw/robocasa`): the non-image members of three pretrain human
tars and of one MimicGen tar (OpenDrawer, parquet only, no extras), the source zip, the
robosuite 1.5.2 wheel, `textures`, `fixtures_lightwheel`, `objects_lightwheel` (whole), and
578 CRC-checked members of `objaverse`/`generative_textures` (252 MB, enough for the first
~10 episodes per task) in `asset_subset/`. Every episode of each dataset was adapted:

| dataset (pretrain) | eps | T min–max | frames | success | regime | base travel m (med / max) | task object disp. m | task fixture joint motion | full-asset scenes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OpenDrawer/20250819 | 102 | 135–250 | 20488 | 102 | tabletop 102 | 0.001 / 0.008 | 0.32–0.60 (object in drawer) | drawer 0.33–0.59 m | 15 |
| PickPlaceCounterToCabinet/20250819 | 108 | 161–309 | 24225 | 108 | tabletop 107, mobile 1 | 0.001 / 0.72 | 0.55–0.93 | door ≤ 0.07 rad | 12 |
| NavigateKitchen/20250821 | 503 | 66–340 | 79550 | 503 | navigation 456, mobile 47 | 3.14 / 9.33 (yaw ≤ 7.8 rad) | – | ≤ 0.32 (bumped doors) | 210 |

Replayed base position, relative grip-site position and finger qpos match the parquet
`observation.state` to ≤ 3.7e-15 on all 713 episodes. Torso lift is essentially unused
(≤ 1.1 cm). Scenes have 48–136 articulated fixture joints; a resolved scene carries
~90 MB of assets. Episodes without local objaverse/generative-texture members took the
`mjcf_kinematic_only` route. Time steps are 0.05 s with occasional 0.10 s gaps (recorded
simulator clock kept).

### Known gaps (RoboCasa)

* MimicGen tars (326.5 GB) lack simulator extras in the inspected sample; a parquet-only
  route (base + relative end-effector pose, no objects) is not implemented.
* `fetch_tars` streams whole tars (videos ≈ 80 % of the transfer) because MG tars put videos
  before data; human tars could stop early but the Box SHA-1 check needs the whole stream.
* Physics validation of RoboCasa scenes is untested here (scenes compile with full assets).

### BiGym verification (2026-10-07)

Full replay of the catalog (1,933 recordings, absolute version per UUID; 7 processes,
about 2 h on the laptop; 1.1 GB of records + 25 MB assets in `data/derived/bigym/replay-v1`):

| | recordings | env success | complete, no success | rejected (action out of bounds) |
| --- | ---: | ---: | ---: | ---: |
| all | 1,933 | **1,358 (70.3 %)** | 433 | 142 |
| articulated (12 tasks) | 674 | 479 | 73 | 122 |
| pick-place (22) | 970 | 640 | 311 | 19 |
| manipulation (3) | 149 | 99 | 49 | 1 |
| reach (3) | 140 | 140 | 0 | 0 |

Success at the last step: 1,251. Per task (success / recordings): CupboardsCloseAll 52/58,
CupboardsOpenAll 23/48, DishwasherClose 0/69 (69 rejected), DishwasherCloseTrays 42/60,
DishwasherLoadCups 58/60, DishwasherLoadCutlery 14/41, DishwasherLoadPlates 29/35,
DishwasherOpen 1/56 (53 rejected), DishwasherOpenTrays 56/60, DishwasherUnloadCups 0/57,
DishwasherUnloadCupsLong 5/51, DishwasherUnloadCutlery 36/47, DishwasherUnloadCutleryLong
29/34, DishwasherUnloadPlates 32/36, DishwasherUnloadPlatesLong 19/39, DrawerTopClose
51/51, DrawerTopOpen 41/47, DrawersAllClose 59/60, DrawersAllOpen 46/54, FlipCup 47/60
(1 rejected), FlipCutlery 46/50, FlipSandwich 43/62, GroceriesStoreLower 22/37,
GroceriesStoreUpper 27/36, MovePlate 54/60 (1 rejected), MoveTwoPlates 32/51 (3 rejected),
PickBox 35/37, PutCups 30/56 (15 rejected), ReachTarget 60/60, ReachTargetDual 50/50,
ReachTargetSingle 30/30, RemoveSandwich 32/39, SaucepanToHob 29/39, StackBlocks 6/39,
StoreBox 39/40, StoreKitchenware 24/36, TakeCups 29/38, ToastSandwich 22/39,
WallCupboardClose 60/60, WallCupboardOpen 48/51.

* The release is presumably curated successes, so the 575 non-successes are attributed to
  the 4.0.0 → 4.1.0 code gap (e.g. "adjusted stiffness of floating base positional
  actuators" in the 4.1.0 changelog), not to the operators; they are kept (with
  `success=False`, or `None` for a rejected replay) and never relabeled.
* Rejections: BiGym raises when an action is outside the 4.1.0 action space. DishwasherOpen
  /Close exceed `right_shoulder_yaw` by up to 0.037 rad; the others exceed a shoulder joint
  by ≤ 0.008 rad. The `--clip-actions` variant (`replay-v1-clipped`, labeled
  `lineage.replay_variant = "clipped_actions"`) succeeds for 102 of these 142
  (DishwasherClose 62/69, DishwasherOpen 27/53, PutCups 11/15, FlipCup 1/1, MovePlate 1/1,
  MoveTwoPlates 1/3). Clipped replays are a different action sequence and are reported
  separately; they are not counted in the 70.3 %.
* Recorded termination flags vs replay success (complete replays): 285 terminated and
  succeeded, 26 terminated but failed, 1,073 never terminated but succeeded, 407 neither.
  Termination is therefore not a usable success label.
* Prior work reproduced: the four MovePlate recordings from `legacy/docs/datasets/bigym-*`
  give the same outcomes (0943 fails, c795 and 103d succeed, d470 rejected at action 0).
* Determinism: replaying in another process gives bit-identical `qpos`/`qvel`.
* Kinematics across MuJoCo versions: body poses from the record's `qpos` in MuJoCo 3.1.5
  vs 3.15.0 agree to 4.4e-16 m (two records, all frames).

Adapter verification on three tasks (all records; scenes with assets for the first three
of each task, compiled again with MuJoCo 3.15.0 → 75 `h1/` robot bodies to remove):

| task | kind | records → episodes | success | base path (median) | check |
| --- | --- | --- | --- | --- | --- |
| DrawerTopOpen | articulated | 47 → 47 | 41 | 0.98 m, yaw range 0.17 rad, torso 0.51 m | successful: top drawer travels its full 0.38 m; failed: ≤ 0.33 m |
| MovePlate | pick-place | 60 → 59 (1 rejected at action 0, no states) | 54 | 0.70 m, yaw 0.32 rad | successful: plate moves ≥ 0.34 m; held plate ≈ 80 mm from the grasp center (rim grasp), 10.5 mm std while held |
| DishwasherLoadCups | base motion + pick-place | 60 → 60 | 58 | 0.95 m, yaw range 1.16 rad, torso 0.25 m | both mugs move ≥ 0.62 m; held mug 9.9 mm std relative to the grasp frame |

No adapter errors; every episode passes the `SourceEpisode` checks; gripper frames are
orthonormal; the grasp center equals the pad midpoint at reset (unit test).

### BiGym known gaps

* 30 % of the recordings do not reproduce their task in the public code (above); exact
  4.0.0 code is not available.
* States are sampled at 50 Hz (every 10th 500 Hz step); actuator commands (`ctrl`) are
  not stored in the records (they are recomputable from the recording's actions).
* Some scene assets are CC BY-NC (see `3D_MODELS_ATTRIBUTION.md`); scene redistribution
  must respect that.
