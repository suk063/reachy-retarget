# State collection performance without reducing the recording contract

`StateObserver.capture` now validates and converts all selected body, TCP,
head, base-relative, and optical-camera poses as one batch. It preserves the
same finite-value, orthogonality, determinant, quaternion-order, and frame
checks. `mjOBJ_XBODY` origin velocities, task-object selection, full replay
state, actuator snapshots, and every existing output channel are unchanged.
It still does not step, forward, reset, render, or mutate MuJoCo state.

Metadata is held as a private template detached once at construction. Each
capture and public `metadata` access receives an independent nested copy.
The optimized copier handles ordinary dict/list/tuple metadata directly and
uses normal deepcopy for unknown identity values. It preserves tuple values
and never shares a mutable metadata container between captured frames.

The original actual BiGym profile spent about 96% of its time in capture,
including 117 individual pose conversions per frame. An independent comparison
on the same recorded scene measured:

| Check | Result |
| --- | ---: |
| All-channel comparison | 63 arrays, every value exactly equal |
| Recorded states compared | Frames 0, 100, 500, 1000, 1500, 2251 |
| Maximum absolute difference | 0 |
| Metadata comparison | Equal at every checked state |
| 100 captures, previous implementation | 6.362950 s |
| 100 captures, batched implementation | 0.506743 s |
| Capture speedup | 12.5566 times |

This is a static-state capture benchmark, not an end-to-end worker speedup or
physics success claim. It does not change the clock, skip states, remove
channels, decimate observations, or change simulation parameters.

Evidence: [state-observer-performance.json](state-observer-performance.json).
The exact profile, old/new module snapshots, model, reset, and captured source
rows remain under
`/mnt/reachy-retarget/workspaces/bigym-physical-performance-20261007-01` and the
referenced immutable BiGym attempt. The new module SHA256 is
`70f3119f0df5b4c82e31dbe4ec49474b0af14d196630bb4975c0bddf8c56a7d4`.

The relevant local suite passes 69 tests across state observation, recording,
control-view derivation, and common archive validation. New regression checks
compare every output array against scalar pose conversion, reject corrupt
body/site/camera rotation caches, and verify nested metadata isolation. Existing
tests retain read-only model/data checks, body-origin velocity checks with
rotated inertial frames, and strict task-object scope.
