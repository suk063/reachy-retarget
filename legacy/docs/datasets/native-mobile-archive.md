# RoboCasa and BiGym native pilot archives

`reachy_retarget.adapters.native_mobile.normalize(root, source)` reads already
acquired pilot members from `root/data/raw/{source}_native_pilot`. It verifies
lengths and SHA-256 against the acquisition manifest before reading values. It
performs no network access, simulator import, source modification, or RGB export.
The result uses the shared `{arrays, metadata, status, missing_fields}` interface
and can be passed directly to `agent_dataset.export_normalized`.

## Actual RoboCasa pilot

The acquired `PickPlaceCounterToCabinet` episode contains 232 MuJoCo state frames
of width 190, a compressed expanded model XML, native metadata, and the parquet
observations/actions. The initial normalizer did not decode the parquet members;
the separate [verified FK reconstruction](robocasa-replay.md) now reads them.

The adapter checks the expanded joint traversal and `time + nq + nv` width,
then retains these recorded values:

- Exact simulator time, 0.5 to 12.1 seconds. The relative archive clock preserves
  230 intervals of 0.05 seconds and one interval of 0.10 seconds; it does not
  manufacture a uniformly sampled 20 Hz sequence.
- All 13 named scalar robot joint positions and velocities from the source
  robot subtree. These remain source robot measurements, not Reachy state.
- The manipulated cereal object's world pose from its recorded free-joint qpos.
- Both named hinge positions of the task's target cabinet.

Only the manipulated object and explicitly referenced source counter/target
cabinet are task objects. Distractor objects receive no pose feature. Source
raw states and model metadata remain preserved as original reference data.
The initial normalizer's status `blocked_source_fk_assets_and_actions` describes
that earlier processing stage. The separate reconstruction verifies source TCP,
base and task-fixture FK and preserves all 12 source action channels. Its status
is `source_fk_verified_actions_preserved`. Asset-faithful source dynamics and
Reachy retargeting/physical validation remain unavailable; FK is not a rollout.

## Actual BiGym pilot

The acquired `MovePlate` lightweight safetensors member contains 2,253 native
source action vectors with 15 components, termination/truncation flags, an empty
reward tensor, reset configuration, random seed, and recorded package versions.
It does not contain measured joint/object/base trajectories. The plate remains
an explicitly declared, undecoded task object.

The source's `absolute` action-mode flag applies to its arm targets. The vector
also contains three pelvis target increments and two normalized gripper commands;
it must not be labeled as 15 absolute joint targets. Exact per-channel source
replay mapping is recorded separately by the native replay adapter.

The acquired header contains neither a timestamp tensor nor a control frequency.
The adapter therefore retains `source/frame_index` as an **untimed index** and
includes the literal `timestamp` missing-field marker. No timestamp, uniform
sample rate, object pose, measured joint state, or Reachy action is inferred.
The status is `blocked_source_replay_and_timestamp`. The original empty reward
array remains empty rather than becoming a fabricated zero-reward sequence.

## Common-format checks

Both actual local pilots were normalized, exported into temporary common-format
archives, and checked with `inspect_archive` on 2026-10-07:

| Pilot | Archive rows | Timed | RGB | Policy ready |
| --- | ---: | --- | --- | --- |
| RoboCasa | 232 | yes, recorded simulator clock | absent | false |
| BiGym | 2,253 | no, explicit source index | absent | false |

Eight offline tests cover recorded clock gaps, object selection, named measured
joint states, malformed quaternions/tensor extents, source checksum mismatches,
and common-writer exports of both a timed fixture and an untimed fixture with an
empty reward channel. These checks validate archive integrity and semantics;
they are not source replay or Reachy dynamics success checks.
