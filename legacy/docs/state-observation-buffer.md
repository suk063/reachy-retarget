# Complete observations with one episode metadata snapshot

`StateObservationBuffer` is an opt-in recording path for repeated captures from
one `StateObserver`. It retains every numeric channel at every requested state
and privately owns one detached metadata snapshot. It removes repeated copies
of identical joint/body/camera inventories; it does not omit observations or
change physics, controls, timestamps, validation or serialization precision.

```python
from reachy_retarget.state_observations import StateObservationBuffer

frames = StateObservationBuffer(observer)
# At each actual synchronized pre-action state:
frames.capture(data)
# After the complete run, including failed runs:
record = frames.finish()
```

`len(frames)` reports actual captured rows. The terminal boundary remains a
separate `observer.capture(data)`. The original `StateObserver.capture` API still
returns detached metadata with every frame. The new `capture_arrays` method
performs the same full numeric capture and checks without copying metadata.

The buffer never exposes its retained rows or metadata. Every `finish()` returns
fresh stacked arrays and detached metadata; changing that output cannot change
the buffer. Empty buffers and duplicate/nonincreasing clocks remain invalid.

The actual BiGym c795 benchmark sampled 500 existing recorded states throughout
the 10,233-step trajectory. Default and buffered capture returned all 63 fields
bit-for-bit identically, with equal metadata. Default/buffer calls alternated
order at each state. Numeric state restoration occurred outside capture timing.

| Measurement | Default frames | Buffer |
| --- | ---: | ---: |
| Capture wall time for 500 states | 0.49550 s | 0.36738 s |
| Retained Python/NumPy traced allocation | 36.25 MB | 17.40 MB |

This is a 1.35x capture speedup and 52.0% lower traced allocation for this workload,
not a claim of the same speedup for the whole pipeline. A separate allocation
pass includes observer construction. The buffered sequence also passed common
HDF5 state coverage assessment. The benchmark copy is explicitly profiling-only
and is not counted as another independent demonstration or physical validation.

Evidence, module/model/recording hashes and exact sampled indices are in
[state-observation-buffer-performance.json](state-observation-buffer-performance.json).
Real MuJoCo regression tests compare every channel and metadata, verify that
capture leaves state unchanged, exercise output/metadata alias isolation, and
check that the buffer still rejects duplicate timestamps.
