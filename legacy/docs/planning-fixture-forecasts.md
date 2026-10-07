# Recorded task-fixture forecasts for planning

`planning_fixtures.bind(prepared, source_path, output, fixture_ids=[...])` is an explicit planning assumption for selected **inactive rigid task objects**. For Stack, a source cubeB initially floats 10 mm above the table and then settles. Keeping its initial free-joint pose fixed during every collision query can report an obstruction that does not exist in the recorded scene at the corresponding time.

The binder reads the unchanged normalized HDF5 and its provenance sidecar. It records both hashes, the compiled XML hash, the complete native and initial control clocks, source object poses, original reset poses and passive scalar-fixture initialization. A single proper rigid source-to-scene transform must reproduce every manipulated-object reference. Every selected fixture's first transformed pose must match the original compiled reset. Missing poses, nonfinite values, unknown clocks, active-object selection, robot roots and articulated subtrees are rejected. A pose-only forecast cannot reconstruct articulation state.

The returned tuple preserves all original source arrays, object references, scene/model and robot arrays. Old initialization, approach and fixture-clearance admission claims are invalidated and retained in the binding report. The caller must run complete fresh IK and collision admission.

`KinematicFixtureQuery` owns a newly created `MjData`. It accepts no caller-owned simulation data and performs no integration. Only this isolated query data receives the selected fixture poses; the manipulated object and unrelated objects retain their original resets. A nonzero simulation clock or velocity is rejected. Base admission, grasp approach and positive fixture-clearance checks use the same optional forecast mechanism. Default behavior is unchanged when no forecast is bound.

After local retiming, the query requires a checksum-bound map with the exact original clock, exact current control clock, finite nondecreasing source correspondence and complete original endpoints. It also verifies that this correspondence reproduces the active-object reference. Evaluation and explicit terminal holds repeat the native final fixture pose; they do not create recorded source observations. Inserted supported-placement trajectories are currently rejected because their separate admission loop has not been verified with this forecast.

This mechanism is never called by physical integration. Physical validation still resets every free object once and integrates its original model normally; no forecast teleports, welds or assists either cube. Recorded future poses are an offline planning prior, not a promise of the Reachy rollout's future state. All actual contact, slip, motion, stability and replay gates remain authoritative. The independent 3 mm positive planning-clearance heuristic is distinct from the unchanged 2 mm penetration gates; passing one must not be inferred from the other.

## Evidence and matched evaluation

The three-source diagnostic in `stack-source-fixture-planning-diagnosis-v1.json` reproduced the old reset-static collision depths with zero error. Applying the original cubeB poses on the complete original clock reduced maximum robot/fixture depths as follows:

| Independent source | Reset-static maximum | Source-fixture maximum |
|---|---:|---:|
| Stack demo 4 | 3.3749 mm | 0.1015 mm |
| Stack demo 6 | 2.6033 mm | numerical zero |
| Stack demo 9 | 2.6705 mm | numerical zero |

The implementation verification in `planning-fixture-forecast-verification-v1.json` repeats all three checks through the new binder/query API, with exactly zero difference from that independent diagnosis, and verifies real retiming plus terminal-hold correspondence. It includes hashes and an operator-side backup of the underlying arrays. These are kinematic results, not physical successes.

`additional-stack-fixture-forecast-v1-pending.json` specifies twelve matched attempts: three existing independent sources × reset-static/source-forecast planning × original-source/explicit global-contact-4ms physical models. Robot attachment, base, approach and controller parameters match within each pair. Both planning arms use the same complete common admission gates; the earlier additional 3 mm heuristic is absent from both and remains uncertified. Within-profile planning comparisons establish the causal effect of the forecast; the alternative contact model is reported separately. No extra independent demonstrations are counted.
