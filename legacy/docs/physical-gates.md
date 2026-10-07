# Common measured manipulation validation

`reachy_retarget.physical_gates` evaluates an actual MuJoCo rollout independently
of whether IK passed or a native task predicate became true. It currently
supports one free moving task object and one contiguous grasp phase for each
explicitly selected hand. A release at the end is supported. Multiple grasp
phases are reported as missing validation, rather than silently reusing an
earlier grasp anchor.

The collector does not step, reset, forward, or otherwise write simulation
state. Every physics step must be observed. Model geometry, masses, friction,
contact parameters, the object trajectory and all acceptance thresholds remain
unchanged. Task identity, source provenance and any declared retiming remain
the producer's responsibility.

## Integration

Create `GateObserver(model, initial_data, active_object_body=..., grasp_hands=...,
initial_self_clearance_m=..., scene_sha256=...)` after the real initial reset and
cache synchronization. Both available Reachy arms are measured, even if only one
hand grasps. Pass the same positive robot clearance measurement used by the
existing robot geometry checker.

After each `mj_step`, call `sample_contacts(data)` immediately, before refreshing
the derived caches. This preserves the solved forces and contact manifold from
that step. Then synchronize observation caches and call:

```python
gates.update(
    data,
    contact_sample=solved_contacts,
    grasp_intent={"left": left_closed, "right": right_closed},
    task_satisfied=native_predicate["success"],
    self_clearance_m=measured_clearance,
)
```

Intent and native success must be explicit booleans. Contact force belongs to
the step's collision state; pose, velocity, margin and task measurements use
the synchronized interval-end state. All body velocities use `mjOBJ_XBODY`, so
the velocity origin and axes agree with the recorded body pose, including
models with offset or rotated inertia.

The active object's entire body subtree is included. For example, BiGym's
free root `plate/` and its collision child `plate/plate` are both recognized.
Bilateral grasp requires positive solved normal force on each of the mapped
distal pad bodies. It is not inferred from object motion or grasp commands.

## Fixed acceptance criteria

These values match the existing common manipulation validator. Acceleration is
reported diagnostically and has no acceptance threshold.

| Measurement | Requirement |
| --- | --- |
| Both arm joint speeds | At most 1.001 rad/s, including the existing numerical tolerance |
| Base body-local speed | Each planar component at most 0.611 m/s; yaw at most 114 degrees/s + 0.001 rad/s |
| Both arm joint limit margins | At least 0.025 rad |
| Positive robot self clearance | At least 0.009 m |
| Hand/object penetration | At most 0.001 m |
| Robot/environment, object/environment, robot/self penetration | At most 0.002 m each |
| Non-hand object/robot contacts | None |
| Acquisition | Explicit closed intent, both pads applying force, and lift above 0.015 m |
| Closed carry after acquisition | At least 95% bilateral contact; drift at most 0.003 m and 3 degrees |
| Final native success | Continuous final 1 s, object speed below 0.02 m/s and 0.2 rad/s |
| Coverage | Every expected step and the complete terminal boundary |
| Independent replay | Maximum full-state position/velocity, object-pose and clock error below 1e-7 |

The grasp anchor is the actual object pose relative to the actual TCP at first
acquisition. Carry includes every subsequently closed-intent step, including
placement while the command remains closed. Native source placement can exceed
this strict drift criterion; such a result is a failed gate, not grounds to
change the threshold or move the phase boundary retrospectively. An absent
acquisition produces `None` drift measurements and failed grasp gates.

For BiGym c795, recorded native binary commands select only the left hand and
one grasp phase. The source releases at the end, so
`require_terminal_contact=False` applies. Final stability must fit the complete
declared physical clock. No unrecorded terminal hold is added by the validator.

## Independent actuator replay

Before integration, use `dynamics_audit.bind_scene_assets` to record external
mesh, texture and height-field hashes alongside the scene hash. Preserve the
exact pre-action rows and a separate actual terminal boundary. Each row needs
the full `mjSTATE_INTEGRATION`, `observation/qpos`, `observation/qvel`, actual
actuator commands in `command/joint_position`, its `control_interval_s`, the
active object pose, and the timestamp. The command channel name reflects the
current position-servo producer; its width must equal the compiled model's
actuator count, not an assumed robot joint count.

`verify_actuator_replay(scene, arrays, expected_scene_sha256=...,
expected_assets=..., active_object_body=..., active_object_joint=...,
object_pose_key=..., initial_state=...)` loads a fresh model and data, verifies
the scene/assets, and rejects object actuators, assistance constraints, mocap
and gravity compensation. It restores recorded integration state exactly once,
checks a separate initial reset record when supplied, and subsequently writes
only the saved actuator controls. Full qpos, qvel, object pose and clock are
compared at every pre-action boundary and the terminal boundary. It never
rewrites an intermediate object state to obtain agreement. Input arrays are
read-only to this function.

Pass that audit into `finish(expected_steps=..., expected_final_time_s=...,
actuator_replay=audit)`. Missing clearance, task predicate, grasp intent, replay,
or matching scene identity blocks `validation_complete`. Failed measured gates
block `physics_validated`, even when validation itself was complete. A replay
success alone is never a task success. The native producer may retain a
stronger diagnostic-only restriction pending source/model review.

## Actual replay evidence

The independent audit in [bigym-common-actuator-replay.json](bigym-common-actuator-replay.json)
replayed all 2,252 steps of the original `bigym-physical-diagnostic-v1` attempt.
Maximum qpos, qvel and clock errors were exactly zero; maximum plate pose error
was `1.1102230246251565e-15`. The audit binds the archived input, reset, model,
assets and audit implementation by hash.

Those assets were hashed **after** the older recording. This is evidence of
reproducibility against the archived/current bound model, not retroactive proof
that asset identity was checked before that recording. New BiGym rollouts bind
assets before integration. Neither the older exact replay nor the existence of
this validator establishes physical task success. BiGym source-version and
MuJoCo-version differences and ideal base servos remain declared simulation
assumptions.

The subsequent timed c795 attempt had complete measured gates and exact actuator
replay, but failed drift, final stability and object/environment penetration.
An independent 10,233-step force audit found the first 3 mm drift violation at
10.964 s and first 3 degree violation at 11.018 s, with bilateral pad force and
no start-rack, target-rack or table force. Target-rack force first appeared at
18.604 s. Drift before that contact reached 31.127 mm and 23.14 degrees. This
rules out supported placement as the sole explanation for these failures; no
phase or threshold was changed. See
[bigym-timed-carry-phase-audit.json](bigym-timed-carry-phase-audit.json).
