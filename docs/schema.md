# Episode schema `reachy-retarget-episode-v2`

One HDF5 file per retargeted episode, written by
`reachy_retarget.schema.io.write_episode` (atomic: temporary file in the same
directory, then rename) and read by `read_episode`. The in-memory model is
`reachy_retarget.schema.episode.ReachyEpisode`. Files hold robot and object **state
only**; the writer refuses any dataset whose path contains `image`, `img`, `rgb`,
`depth`, `camera`, `pixel`, `video` or `segmentation`, and any `uint8` array with
`ndim >= 3`.

## Conventions

* Clock: `time` is uniform at 50 Hz (`DT = 0.02 s`), `T >= 2` rows.
* Joint vector `q`, `qd`: 22 columns in the `JOINTS` order of `docs/design.md`
  (`base_x, base_y, base_yaw` in world, then left arm, right arm, neck, fingers). `qd`
  is the time derivative of `q`; base terms are world-frame rates.
* Frames: `world` = source world; `base_link` on the floor at `(base_x, base_y,
  base_yaw)`, so `X_world = planar(q[:, 0:3]) @ X_base`. TCP = `{l,r}_arm_tip`, head =
  `head` frame.
* Poses on disk: `(T, 7)` = `xyz + wxyz` quaternion with `w >= 0`. In memory robot
  poses are `(T, 4, 4)` matrices; object tracks stay `(T, 7)`.
* rotation6d = first two matrix columns `(r00, r10, r20, r01, r11, r21)` as in
  reachy-agent `rotation_values`; inverse by Gram–Schmidt.
* Gripper opening in `[0, 1]` (1 = open); width in metres.
* All gzip-compressed (with shuffle) float64 unless stated (`/scene` poses are float32).

## Layout

| path | shape | content |
| --- | --- | --- |
| attrs `schema` | str | `reachy-retarget-episode-v2` |
| attrs `joint_names` | JSON | `JOINTS` |
| attrs `metadata` | JSON | `uid, family, dataset, episode_id, task, instruction, regime, license, provenance, lineage, variant_of, body_parts, retarget_config, tier, extra` |
| `/time` | (T,) | seconds |
| `/source_time` | (T,) | source-clock time of each row (non-decreasing) |
| `/state/q`, `/state/qd` | (T, 22) | attr `columns` |
| `/state/base_pose_world` | (T, 3) | `q[:, 0:3]` (view) |
| `/state/tcp/<side>/base` | (T, 7) | canonical TCP pose in `base_link` |
| `/state/tcp/<side>/world` | (T, 7) | world view of the above |
| `/state/head/base`, `/state/head/world` | (T, 7) | head frame pose (base canonical, world view) |
| `/state/gripper/opening`, `/state/gripper/width` | (T, 2) | left, right |
| `/reference/tcp/<side>` | (T, 7) | pre-IK world TCP target (only sides that have one) |
| `/reference/head` | (T, 7) | pre-IK world head target (optional) |
| `/reference/base` | (T, 3) | pre-IK world base target x, y, yaw (optional) |
| `/objects/<id>/pose` | (T, 7) | world pose; rows with `valid = False` may be NaN |
| `/objects/<id>/valid` | (T,) bool | attrs `role`, `geometry` (JSON) |
| `/articulations/<id>/qpos` | (T, n) | attr `joint_names` (JSON) |
| `/validation/<name>` | (T, ...) | per-frame tier-K diagnostics, e.g. `tcp_pos_residual` (T, 2) |
| `/physics/{time,qpos,qvel,ctrl}` | (N, ·) | optional tier-P rollout on the simulator clock; attrs `columns` per array, group attr `info` (JSON: simulator, timestep, assumptions) |
| `/actions/<mode>/<array>` | (T-1 or T, k) | precomputed control views; attr `columns`, group attrs `frame`, `units`, `semantics` |
| `/scene` | group | scene components and their meshes (see *Scene components and asset library*); attrs `schema` = `reachy-retarget-scene-v1`, `library` |
| `/scene/description` | (n,) uint8 | gzip-compressed JSON `{"components", "materials", "textures", "info"}` |
| `/scene/poses` | (T, C, 7) float32 | world pose of every component at every episode row (NaN where not valid); attr `components` (names in column order) |
| `/scene/valid` | (T, C) bool | pose validity |
| `/scene/physics_poses` | (N, C, 7) float32 | optional: the same components on the tier-P clock (`/physics/time`) |

Metadata details: `uid` defaults to `<dataset>/<episode_id>`; `regime` is one of
`tabletop, mobile_manipulation, navigation`; `body_parts` is the subset of
`left_arm, right_arm, head, base` actually used; `lineage` holds at least `seed`;
`variant_of` is the uid of the episode a mirrored or alternate variant derives from;
`tier = {"K": {"passed", "reasons"}, "P": {"passed", "reasons"} | null}` (K and P are
never merged); `retarget_config` is the retarget configuration hash; `extra` is free JSON
(e.g. simulation assumptions).

Only canonical arrays are read back: world views and `/actions` are regenerated from
them. `recompute_modes(path)` rewrites every control view from the canonical state.

## Scene components and asset library

Every written episode stores the meshes of its scene (a policy builds a per-component surface
feature map from them, as reachy-agent `policy/build_map.py` does, and assembles it at training
time from per-frame component poses). Episodes whose meshes cannot be obtained are not written
(build record `status: "excluded"`, `excluded: "no_meshes"`), nor episodes in which a tracked
object, articulation or scene component lacks its pose at some step (`excluded: "no_object_poses"`);
see docs/design.md, *Scene meshes*.
Read with `reachy_retarget.schema.scene_assets`: `read_scene(path)`,
`load_component_meshes(path, kind="visual" | "collision", frame=None)` (trimesh-compatible arrays)
and `scene_mjcf(path, frame=0, physics=False)` (an MJCF string that MuJoCo compiles and renders; `frame=None` puts every component at the origin, e.g. to render one component in its own frame for a surface map).

**Asset library.** `<out>/assets/<sha256>.<ext>` next to `<out>/episodes/`, one per build output
root (= dataset release): every mesh and texture file once, named by the SHA-256 of its bytes,
written to a temporary name and hard-linked into place (never overwritten; identical content gives
the identical name, so libraries of several roots merge by copying missing names). There is no
index: every reference is `{"sha256", "format", "bytes"}` (meshes add `vertices`, `faces`,
`normals`, `uv`). `/scene` attr `library` is the library path relative to the episode's folder;
readers fall back to the first `assets/` folder above the episode. Formats:

* `msh` (all meshes): MuJoCo's binary mesh, int32 `nvertex, nnormal, ntexcoord, nface`, float32
  vertices (nvertex, 3), normals (nnormal, 3), texture coordinates (ntexcoord, 2, used as stored,
  no v flip) and int32 faces (nface, 3); `nnormal`, `ntexcoord` are 0 or `nvertex`.
* textures keep their recorded bytes and format (`png`, `jpg`, ...). Textures are asset files of
  the meshes, not observations; rendered images are never stored.

**Components** (`description.components`, in pose-column order): rigid bodies of the scene.

| key | content |
| --- | --- |
| `name` | unique: the object id for tracked objects (`/objects/<id>`), else the source body name; `worldbody` for geoms of the world body; `reachy/<link>` for Reachy links |
| `role` | `manipulated` (free object; `task` False for untracked free bodies such as distractors), `support`, `receptacle`, `fixture`, `articulated_part`, `robot_link` |
| `source_body`, `bodies` | the source body that defines the component frame (MuJoCo body / URDF link) and every source body merged into it |
| `kind` | `free`, `articulated`, `static`, `robot` |
| `task` | tracked object or tracked articulation |
| `object_id`, `articulation`, `joints` | the `/objects` id, the `/articulations` id and the source joints of the component |
| `pose_source` | `object_track` (`/objects/<id>/pose`), `scene_state` (forward kinematics of the source scene joints), `static`, `robot_fk` (Reachy FK of `q`) |
| `visual_from_collision` | the component has no visual-only geoms and its visible colliding geoms are its visual parts |
| `visual`, `collision` | parts (below) |

A **part** is one geom, in the component frame: `geom` (source name), `type` (`mesh`, `box`,
`sphere`, `capsule`, `cylinder`, `ellipsoid`, `plane`), `size` (MuJoCo geom size: box half
extents, radius / half length, ...), `pos`, `quat` (wxyz), `group`, `contype`, `conaffinity`,
`rgba` (geom rgba; MuJoCo uses it instead of the material colour when it differs from the default
0.5 0.5 0.5 1), `material` (name or null), and

* `mesh`: library reference of the compiled MuJoCo mesh (scale, refpos/refquat and MuJoCo's
  re-centring applied, vertices in the mesh frame; the part pose is the compiled geom pose) with
  `mesh_source` = `{name, file, sha256, scale, mesh_pos, mesh_quat}` (`stored vertex =
  R(mesh_quat)^T (scale * file vertex - mesh_pos)`); collision meshes add `collision_shape`
  (MuJoCo collides with the convex hull);
* `generated_mesh` (visual primitives only): a surface mesh of the primitive
  (`scene_assets.primitive_mesh`); collision primitives have type and size only.

`description.materials[name]`: `rgba, specular, shininess, reflectance, emission, metallic,
roughness, texrepeat, texuniform, texture` (+ `layers` `{role: texture}` for MuJoCo material
layers); Reachy materials keep their COLLADA/URDF origin under `source`.
`description.textures[name]`: `attributes` (every MJCF texture attribute except name and files:
`type`, `builtin`, `rgb1`, `rgb2`, `mark`, `width`, `height`, `gridsize`, `gridlayout`, ...),
`files` `{attribute: reference + source_file}` (`file`, `fileright`, ...), `file` (= `files.file`),
`builtin`. Builtin textures have no file.

`description.info`: selection rules and radius, `omitted` components (with distance), source
scene digest and removed robot elements, `track_vs_scene_state` (max deviation between tracked
object poses and the scene kinematics), Reachy URDF digest and skipped visual parts, `library`.

**Poses.** `poses[t, c]` is the world pose of component `c` at episode row `t`: tracked objects
from `/objects/<id>/pose` (the retargeted, resampled source track; NaN where invalid), untracked
moving parts by forward kinematics of the source scene joint positions resampled onto the episode
clock, static parts constant, Reachy links by FK of `q` (`base_link` = `planar(q[0:3])`).
`physics_poses[n, c]` holds the same components on the tier-P clock: forward kinematics of
`/physics/qpos` (objects as simulated) and Reachy FK of the rollout's joint positions. To assemble a
reachy-agent-style map observation: `body_names` = component names, `body_poses` = `poses[t]`.

**Exported source scenes.** Episodes keep only the digests of the scene their tier-P rollout used
(`/physics` attr `info` -> `scene`: `source_mjcf_sha256`, resolved `assets` with sha256,
`removed.parked_bodies`, `scene_sha256` of the compiled scene). `python -m
reachy_retarget.scene_export export` regenerates the `SceneRef` from the raw source file (on the
cluster, where the raw data is), rebuilds the tier-P scene and checks those digests (the compiled
`scene_sha256` only when the MuJoCo version equals the recorded `simulator`). A verified scene is
written by `SceneRef.save` as `scenes/<source_mjcf_sha256>/{scene.xml, scene_ref.json,
assets/<sha256>}` (format `reachy-retarget-scene-ref-v1`: the exact MJCF bytes, robot prefixes,
the exporting demo's initial qpos, inactive bodies, reference values and the asset name -> sha256
map; only assets left after removing the source robot are kept) and read back, hash-checked, by
`SceneRef.load`. `records/<job>.jsonl` holds one line per episode (status, checks). Rebuild an
episode's scene with `build_scene(SceneRef.load(dir), drop_bodies=info["scene"]["removed"]["parked_bodies"])`
and take its initial object state from `/physics/qpos`.

## Control modes

Registry: `reachy_retarget.schema.control_modes.MODES` (`ModeSpec` with `frame`,
`units`, `semantics`, `columns`, `compute(ep)`). **Action row `t` is issued at state
row `t` and targets state row `t+1`; there are `T-1` action rows.** Below, `P` is a TCP
pose in `base_link`, `H` the head rotation in `base_link`, `b = q[:, 0:3]`.

| mode | arrays (columns) | definition |
| --- | --- | --- |
| `joint_pos_abs` | `action` (22, `JOINTS`) | `q[t+1]` |
| `joint_pos_delta` | `action` (22) | `q[t+1] - q[t]`, `base_yaw` difference wrapped to `[-pi, pi)` |
| `joint_vel` | `action` (22) | `qd[t]` (m/s, rad/s) |
| `ee_abs_base` | `<side>_pos_rot6d` (9), `<side>_pos_quat` (7: x, y, z, qw, qx, qy, qz) | `P[t+1]` |
| `ee_abs_world` | same | world TCP pose at `t+1` |
| `ee_delta_base` | `<side>_tool`, `<side>_base` (6: dx, dy, dz, rx, ry, rz) | tool: `D = inv(P[t]) P[t+1]`, `(D.p, log D.R)`, so `P[t+1] = P[t] D`; base: `(p[t+1] - p[t], log(R[t+1] R[t]^T))` |
| `ee_twist_tool` | `<side>` (6: vx, vy, vz, wx, wy, wz) | `se3_log(inv(P[t]) P[t+1]) / DT`, body twist reaching `P[t+1]` in 20 ms |
| `head_rot_abs` | `rot6d` (6) | `rot6d(H[t+1])` |
| `head_rot_delta` | `rot6d` (6) | `rot6d(H[t]^T H[t+1])` |
| `base_twist_body` | `twist` (base_vx, base_vy, base_wz) | `se2_log(base_se2_delta, DT)`: constant body twist reaching `b[t+1]` |
| `base_se2_delta` | `delta` (base_dx, base_dy, base_dyaw) | body-frame displacement from `b[t]` to `b[t+1]` (20 ms), dyaw wrapped |
| `gripper_continuous` | `opening` (2) | `opening[t+1]` |
| `gripper_binary` | `open` (2) | `opening[t+1] >= 0.5` as 0/1 |
| `reachy_agent_v8` | `action` (29), `state` (30, **T rows**), float32 | see below |

Arm-only deltas (`ee_*` in `base_link`) exclude base motion; combine with a base mode
for whole-body control.

### `reachy_agent_v8`

Matches reachy-agent `d6d5e9fd` (`robot/actions.py`, `policy/inputs.py`), action
contract `current-base-tcp-head-rotation6d-local-base-delta-binary-grippers-v8`, state
contract `base-tcp-head-world-base-xy-sincos-opening-v8`.

* `action[t]` = `left_{x,y,z,r00,r10,r20,r01,r11,r21}` = `P_left[t+1]`, the same for
  `right_`, `head_{r00..r21}` = `rot6d(H[t+1])`, `base_dx, base_dy, base_dyaw` =
  `base_se2_delta[t]`, `left_gripper_open, right_gripper_open` = `gripper_binary[t]`.
* `state[t]` = the 24 pose columns at `t`, `base_world_x, base_world_y,
  base_world_sin_yaw, base_world_cos_yaw` from `b[t]`, then `left_gripper_opening,
  right_gripper_opening` = continuous `opening[t]`.

## Index

`write_index(dir, rows)` writes `dir/index.parquet` atomically from `index_row(ep,
file)` dicts: `uid, file, family, dataset, task, regime, body_parts (list),
tier_k_passed, tier_p_passed (null if no P), license, lineage_seed, variant_of, length,
duration`. Count independent demonstrations by `lineage_seed`, not by rows.

## Tracking episodes

`family = "tracking"` episodes (docs/tracking.md) use the same layout with these specifics.

They contain no `/objects` and no `/articulations`. `/validation` holds:
- `grasp_object` = −1;
- `head_rot_residual` (T,);
- `witness_q` (T, 22) for witness cells: the joint path whose FK produced the reference.

`/reference/tcp/*` and `/reference/head` hold the tracked reference unmodified. The head pose's
rotation is the target, and its position is the achieved head position. `/reference/base` is the
nominal base path; the base is free, so it is not a target.

`metadata.extra` holds:
- `tracking`: the scenario, its parameters, diagnostics and the `head_tip` offset;
- `tracking_config`.

`retarget_config` is the `TrackingConfig` digest, and `tier.P` is null.
