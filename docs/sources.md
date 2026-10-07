# Source families

Every file is pinned in `reachy_retarget/acquire/catalog.yaml` (URL at an immutable
revision, publisher SHA-256, size, license, kind). `fetch()` is the only code path that
uses the network; it writes `data/raw/<family>/<path>` and records each verified file in
`data/raw/ledger.json`. "Catalogued", "fetched", "adapted" (a `SourceEpisode` was
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
streaming, and refusal of `kind: images_embedded` files unless `allow_images=True`.

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
* Catalogued: `source/*.hdf5` (12 tasks, ~10 human demos each, 0.47 GB) and
  `core/*.hdf5` (26 generated task variants, ~1000 demos each, 88.8 GB). `object/`,
  `robot/` and `large_interpolation/` are not catalogued.
* Recording: robosuite 1.4.1 (`env_version`), Panda, 20 Hz, same `states`/`model_file`
  layout; v1.4 naming (`gripper0_grip_site`, `mount0_`).
* **Every MimicGen file embeds 84×84 RGB observations** (`agentview_image`,
  `robot0_eye_in_hand_image`), including the source files. They are therefore
  `kind: images_embedded` and `fetch` refuses them by default. The adapter never reads
  image datasets. Getting the core data without storing images needs a range-reading
  extractor (h5py over HTTP Range that copies only non-image datasets); not built yet.
* Lineage: `core/<task>_d<k>` demos are MimicGen-generated from `source/<task>`
  (`lineage = {"generated": True, "seed": "mimicgen/source/<task>", "seed_demo": None}`);
  the published files do not record the per-demo source index. They must not be counted
  as independent demonstrations.
* Tasks such as Coffee, Kitchen, MugCleanup, HammerCleanup use MimicGen's own object
  assets (`mimicgen/models/robosuite/assets`, <https://github.com/NVlabs/mimicgen>), which
  are not catalogued yet. Their files then fall back to the kinematic-only route below.
  Stack and the robosuite-native tasks need only the robosuite wheel.

## robosuite assets (family `robosuite`)

* PyPI wheels `robosuite-1.5.1-py3-none-any.whl` (152 MB) and
  `robosuite-1.4.1-py3-none-any.whl` (194 MB), PyPI SHA-256 digests, license MIT.
* Read as zip archives (`robosuite/models/assets/*`), never installed or imported. The
  recorded MJCF references meshes and textures by absolute paths of the recording
  machine (e.g. `/home/.../robosuite/models/assets/robots/panda/meshes/link0.stl`); the
  adapter maps everything after `robosuite/models/assets/` into the wheel of the
  recorded `env_version`.

## Adapter: `reachy_retarget/sources/robosuite.py`

Per demo: compile the recorded MJCF (cached by XML), set `qpos`/`qvel` from each state
row, `mj_forward`, and read:

* **Effectors**: one per `gripper<N>[_right|_left]_grip_site` (key = that prefix, e.g.
  `gripper0_right`; Transport has `gripper0_right` and `gripper1_right`). The contract
  frame is measured from the model with the fingers open: +z = palm (finger parent body
  origin) → midpoint of the two finger pads, +y = finger-1 pad → finger-2 pad. For the
  Panda gripper this gives `R_contract = R_site · [[0,1,0],[-1,0,0],[0,0,1]]` (site +z is
  already the approach axis; the closing axis is site ±x) and a grasp center 3.6 mm
  behind the grip site along the approach axis (pad midpoint). Width = pad separation
  along +y minus the fully closed separation (equals the finger joint separation for the
  Panda, 0–0.08 m); opening = width / 0.08. `side_hint` is `None` (Transport arms face
  each other, there is no left/right).
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

## Verification (2026-10-07)

| dataset | demos | T min–max | total steps | success | opening range | route |
| --- | --- | --- | --- | --- | --- | --- |
| robomimic/can/ph | 200 | 82–151 | 23207 | 200 | 0.378–1.000 | mjcf_states |
| robomimic/lift/ph | 200 | 36–64 | 9666 | 200 | 0.488–1.000 | mjcf_states |
| robomimic/square/ph | 200 | 107–236 | 30154 | 200 | 0.200–1.000 | mjcf_states |
| robomimic/transport/ph | 200 | 373–714 | 93752 | 200 | 0.012–1.000 | mjcf_states |
| mimicgen/source/stack | 10 | 86–117 | 1001 | 10 | 0.483–0.996 | mjcf_states |

The grip-site position reproduces the recorded `obs/robot0_eef_pos` exactly; the grasp
center differs from it by 3.6 mm along the approach axis, as designed. Width equals
`robot0_gripper_qpos[0] − robot0_gripper_qpos[1]`.

## Known gaps

* MimicGen data is image-embedded (see above); `mimicgen/source/stack.hdf5` (11.8 MB)
  was fetched with `allow_images=True` for adapter verification only.
* MimicGen custom assets are not catalogued.
* Instructions are not available for robomimic or MimicGen.
* LIBERO (robosuite 1.4 + BDDL, `problem_info` with language) and DexMimicGen
  (multi-finger hands) need task tables and, for DexMimicGen, a non-parallel gripper
  model; the state replay is shared.
