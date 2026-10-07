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
* All gzip-compressed (with shuffle) float64 unless stated.

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

Metadata details: `uid` defaults to `<dataset>/<episode_id>`; `regime` is one of
`tabletop, mobile_manipulation, navigation`; `body_parts` is the subset of
`left_arm, right_arm, head, base` actually used; `lineage` holds at least `seed`;
`variant_of` is the uid of the episode a mirrored or alternate variant derives from;
`tier = {"K": {"passed", "reasons"}, "P": {"passed", "reasons"} | null}` (K and P are
never merged); `retarget_config` is the retarget configuration hash; `extra` is free JSON
(e.g. simulation assumptions).

Only canonical arrays are read back: world views and `/actions` are regenerated from
them. `recompute_modes(path)` rewrites every control view from the canonical state.

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
