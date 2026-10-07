# Native surface height correction for Can source 261e

The original `+120°` native-facet grasp fails at loaded source row 281. Both a warm restart and a minimum-motion constrained solve retain about 2.079 mm position error and 0.02036 rad rotation error, outside the unchanged 2 mm / 0.02 rad target tolerances. This is not evidence that the complete task or every grasp is impossible.

For the unchanged target, an exact native palm vertex provides a stronger local result. Under every rigid TCP pose within those tolerances, the vertex remains inside the original table's horizontal extent, cannot escape below it, and has at most 2.90309 mm clearance above it. The required fixture clearance is 3 mm. The target's tolerance envelope therefore contains no valid palm pose. Seven originally closed rows, 275–281, have the same obstruction. The object, table, robot geometry and thresholds were unchanged; these were isolated geometric queries without simulation steps.

One constant TCP translation along the Can's native axis resolves this obstruction. The minimum shift that clears the exact desired palm across all closed rows is 6.634336642 mm. The declared proposal adds a 0.1 mm planning reserve, giving **6.734336642069025 mm**. This is a robot grasp-site correction, not an object displacement. Original azimuth, orientation, object trajectory, gripper intent and complete clock remain unchanged. The previous first hand target is exact, and the existing full open approach connects it to the higher closing patch.

The unchanged finite CAD pads allow shifts from −15.353 to +9.386 mm across all 290 closed source rows. The selected anchor height changes from 12.841 to 19.575 mm. Complete pad axial extent changes from [−21.994, +26.744] mm to [−15.373, +33.478] mm; the native straight wall is [−38.090, +37.129] mm. The existing 1 mm end guard and finite anchor contact-corridor checks pass. These geometric statements do not imply a measured physical contact area.

Fresh mobile/right-arm IK admits **831/831 original rows**, followed by complete approach, open-hand/object, aperture-envelope and 3 mm fixture checks. Maximum target errors are 1.151 micrometres and 4.001 microradians; reported minimum fixture clearance is at least 3.001 mm. The inactive arm remains exact. Timing, inserted placement and physical rollout are separate subsequent stages.

The explicit replay parameters are:

```json
{
  "cylindrical_grasp": {
    "orientation_policy": "jaw_level",
    "azimuth_policy": "nearest_opposed_facets",
    "azimuth_offset_deg": 120,
    "end_margin_m": 0.001,
    "axial_offset_m": 0.006734336642069025,
    "preserve_pre_offset_initial_target": true
  },
  "approach_before_base_admission": true,
  "approach_reuse_admitted_reference": true,
  "grasp_approach": {"lift_m": 0.12, "traverse_fraction": 0.5}
}
```

Retain the original source-specific mobile anchor and base bounds from the evidence JSON; these parameters do not replace them. The optional axial API rejects unsupported offsets rather than clamping them, checks every closed source row, and requires explicit initial-target preservation. Its zero default follows the previous code path. Fresh reconstruction from the original parent produces bit-for-bit identical complete target arrays to the admitted pilot. Thirty-five focused cylindrical tests pass. API SHA `67993a23344e09a48ed01a421de95034f1d0ccf7f6cb247fc76be3670bff9a99` is published in v53.

The portable cache preserves a reproducible reset and complete planned references, with no measured state observations or physics-success label. Source clock, intent and joint references survive cache load exactly; pose serialization roundoff is at most 7.8e-16. `can-261e-native-height-v1-evidence.json` includes source URLs/revision/checksums, all proof and admission records, reconstruction settings and cache bindings. The 2,685,820-byte local artifact `runs/can-261e-native-height-v1/artifact.tar.gz` has SHA-256 `57fa7eeec988a0b9715e7f0525875f58587ad56fe9937c0ce9b97cdfbf231065` and retains the original failed attempts as well as the new geometric admission.

The first cache omitted the `experiment_variant` label, so placement dispatch correctly rejected it before simulation. The separate v2 cache derives `contact_0035` through the canonical `feasible_experiments.variant` function and asserts that every numeric Candidate field, source array and admitted aperture is unchanged. `validate_parent` passes before and after serialization; v1 bytes remain unchanged. This metadata correction is backed up as `runs/can-261e-native-height-v1/materialization-v2.tar.gz` (154,952 bytes, SHA-256 `e7ca1b86e3c1c72c2e5da90b9b19abf76389d4cbd6ddb70d745f82698fa55725`). The v2 cache, rather than v1, is the supported placement input.
