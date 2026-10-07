# Offline control views

`reachy_retarget.control_views.build_control_views(arrays, metadata)` derives
alternative representations from saved numeric state and explicitly described
commands. It never runs a robot, simulator, renderer, network transfer, or policy.
The result contains `arrays`, per-channel semantics in `metadata.views`, and
explicit `missing_fields`. These derived views are not new demonstrations or
evidence of successful physical execution. Images are unnecessary.

## Observed state and command targets remain separate

The default groups are `left_arm`, `right_arm`, `neck`, `base`, `left_gripper`,
and `right_gripper`. `joint_names` defines the saved scalar column order;
`joint_groups` explicitly selects indices or ordered names. No group is inferred
from a convenient slice of an undocumented state vector. Empty or absent groups
are reported as missing. Neck observations use `observation/head_pose`; this does
not establish the existence of a neck command.

| View | Meaning | Rows |
|---|---|---|
| `observed/joints/<group>/absolute` | Saved current joint position | T |
| `observed/joints/<group>/recorded_velocity` | Saved measured/simulated joint velocity | T |
| `observed/joints/<group>/observed_next_state` | Next observed position, **not** a controller target | T−1 |
| `observed/joints/<group>/raw_delta` | Direct coordinate difference | T−1 |
| `observed/joints/<group>/coordinate_delta` | Difference with only explicitly declared periodic wrapping | T−1 |
| `observed/joints/<group>/interval_velocity` | Derived coordinate delta divided by the actual time interval | T−1 |
| `recorded_commands/...` | Unmodified command channels with their recorded semantics | T or T−1 |
| `command_views/...` | Equivalent representations of an actual recorded command against its synchronous observation | T or T−1 |

`timestamp` or `time_s` must contain finite, strictly increasing times. Irregular
sampling is retained with `transition_start_s`, `transition_end_s`, and
`transition_dt_s`. No frame rate is guessed. Untimed data can still expose
absolute states and recorded velocities, but generates no transition rates.

Angular wrapping is opt-in through a mapping such as
`joint_periods={"base_yaw": 2*pi}`. An unlimited hinge is not automatically a
periodic coordinate: wheel winding may matter. A shortest periodic difference
cannot identify an inter-frame rotation larger than half a revolution; the raw
difference is always preserved. Joint units remain those in the observer's
ordered `joint_details` metadata.

## Cartesian pose and velocity frames

Left/right TCP, head, and base poses use XYZ followed by a unit **wxyz** quaternion.
The module validates quaternions instead of silently normalizing corrupt values.
Pose deltas use SE(3) multiplication, never Euler-angle subtraction:

```text
body delta       = inverse(T_current) @ T_next
world left delta = T_next @ inverse(T_current)
```

Both satisfy their appropriate reconstruction identity. A world-left delta's
translation is generally different from `p_next - p_current`; the latter is
stored separately. Finite interval SE(3) log rates are angular-first
`[rotation_vector, translation_log] / dt`. They are derived finite interval
quantities, distinct from a recorded instantaneous twist or a velocity command.
The principal rotational log has the usual half-turn ambiguity.

World views are stored under `observed/poses/<group>/world/`. If the saved base
pose is available, arm and head views also have a `base/` branch constructed from
`inverse(T_world_base[t]) @ T_world_body[t]`. Its left-delta and velocity labels
use `reference` rather than `world`. This is motion relative to the **moving**
base, not merely the world velocity rotated into base axes.

Saved observer twists remain separate:

- `*_twist_world`: absolute body-origin velocity in world axes.
- `*_twist_base`: absolute body-origin velocity expressed in base axes.
- `*_twist_relative_base`: velocity relative to the moving base, including the
  observer's base-motion subtraction.
- `observation/base_twist`: planar `[vx, vy, wz]` in base axes, retained as a
  three-component observation rather than silently becoming a six-component
  twist.

## Explicit command contracts

A source action or actuator value does not reveal the controller's intended
Cartesian target. Desired commands are recognized only through an explicit
`metadata.command_semantics` mapping. For example:

```python
metadata["command_semantics"] = {
    "command/right_target_pose": {
        "kind": "pose",
        "representation": "absolute",
        "timing": "pre_action",
        "group": "right_arm",
        "frame": "world",
        "quaternion_order": "wxyz",
    },
    "command/neck_target": {
        "kind": "joint_position",
        "representation": "absolute",
        "timing": "pre_action",
        "joint_names": ["neck_roll", "neck_pitch", "neck_yaw"],
    },
    "command/base_velocity": {
        "kind": "base_twist",
        "representation": "absolute",
        "timing": "pre_action",
        "group": "base",
        "frame": "base_axes",
        "component_order": "vx_vy_wz",
    },
}
```

Joint velocity commands require `kind="joint_velocity"` and ordered joint names.
Six-component twist commands require `kind="twist"`, a named frame, and an
explicit `angular_linear` or `linear_angular` order. Delta commands additionally
name their reference; `reference="observed_current"` permits conversion against
the corresponding synchronous pre-action state.

Actual absolute pose targets can be converted to body/right and reference/left
deltas. Actual body/reference delta targets can be composed into absolute poses.
Actual joint targets can be converted to raw differences from current joints.
These conversions use the **recorded target**, never the future observed state.
They do not invent a velocity target from a position target or a control period.
Conversions are skipped when timing or reference semantics do not establish
the required correspondence.

`observation/actuator_control` and `applied_controls` are retained with ordered
actuator names. They are evidence of actuator commands only; no EE or joint target
meaning is inferred from them. Command coverage is reported separately for each
group and representation. Missing left-arm, right-arm, neck, or base commands
remain missing even when the saved resulting motion is available.

The return value is an in-memory view bundle, not an in-place mutation of the
episode. Sample arrays have T rows and transition arrays have T−1 rows; callers
must preserve their separate timelines when storing or selecting training
targets. Selecting `observed_next_state` for imitation is a new, explicitly
declared supervision choice, not recovery of the original controller command.

Validation covers irregular clocks, periodic wrapping, explicit joint ordering,
quaternion sign equivalence, body/world SE(3) composition, SE(3) log reconstruction
near zero and pi, moving-base frames, planar base commands, actual-target
conversion, and absent command evidence. Twenty offline tests pass.
