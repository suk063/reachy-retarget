# Local object-aware coverage audit

> Earlier kinematic and Can-only baseline. Current multi-object physical results
> are in [the executed dynamics validation](dynamics-validation.md); this saved
> audit is retained as historical evidence and is not the current physics matrix.

Generated: 2026-10-07T04:30:23.884005+00:00

Offline artifact inspection; no network, robot access, controller execution, or physics rerun.

Historical catalog exports are descriptions of a previous acquisition run. They do not establish that payloads exist on this machine. Public accessibility is not rechecked by this offline command.

| Root | Current downloaded source IDs | Normalized artifacts | Object-aware | Kinematic pass | Contact-validated episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| /Users/sunghwan/workspace/reachy-retarget | 0 | 0 | 0 | 0 | 0 |
| /Users/sunghwan/workspace/reachy-retarget/runs/local-preview | 2 | 1 | 1 | 1 | 0 |
| /Users/sunghwan/workspace/reachy-retarget/runs/local-preview-v2 | 2 | 1 | 1 | 1 | 1 |
| /Users/sunghwan/workspace/reachy-retarget/runs/object-matrix | 6 | 29 | 29 | 25 | 1 |

Counts of downloaded source IDs can include asset bundles; they are not counts of demonstration datasets.

Repeated roots and attempts collapse by source group or exact normalized fingerprint. Unknown re-timed/cropped/reformatted overlap remains unresolved; source groups are not claimed to be independent demonstrations.

## Episode evidence

| Source / sequence | Object IDs with usable poses | Scenario evidence | Kinematic | Contact |
| --- | --- | --- | --- | --- |
| hf__robomimic__robomimic_datasets / v1.5/can/ph/demo_0 | Can | PickPlaceCan | True | False |
| hf__robomimic__robomimic_datasets / v1.5/can/ph/demo_0 | Can | PickPlaceCan | True | True |
| hf__amandlek__mimicgen_datasets / source/threading/demo_1 | needle_obj, tripod_obj | Threading_D0 | True | False |
| hf__amandlek__mimicgen_datasets / source/stack/demo_2 | cubeA, cubeB | Stack_D0 | True | False |
| hf__amandlek__mimicgen_datasets / source/stack/demo_0 | cubeA, cubeB | Stack_D0 | True | False |
| hf__amandlek__mimicgen_datasets / source/threading/demo_2 | needle_obj, tripod_obj | Threading_D0 | True | False |
| hf__amandlek__mimicgen_datasets / source/threading/demo_0 | needle_obj, tripod_obj | Threading_D0 | True | False |
| hf__amandlek__mimicgen_datasets / source/stack/demo_1 | cubeA, cubeB | Stack_D0 | True | False |
| hf__haosulab__ManiSkill_Demonstrations / demos/PickCube-v1/teleop/traj_1 | cube | PickCube-v1 | True | False |
| hf__haosulab__ManiSkill_Demonstrations / demos/PickCube-v1/teleop/traj_2 | cube | PickCube-v1 | True | False |
| hf__haosulab__ManiSkill_Demonstrations / demos/PickCube-v1/teleop/traj_0 | cube | PickCube-v1 | True | False |
| hf__haosulab__ManiSkill_Demonstrations / demos/PickCube-v1/teleop/traj_4 | cube | PickCube-v1 | True | False |
| hf__haosulab__ManiSkill_Demonstrations / demos/PickCube-v1/teleop/traj_3 | cube | PickCube-v1 | True | False |
| hf__robomimic__robomimic_datasets / v1.5/tool_hang/ph/demo_2 | frame, stand, tool | ToolHang | True | False |
| hf__robomimic__robomimic_datasets / v1.5/lift/ph/demo_1 | cube | Lift | True | False |
| hf__robomimic__robomimic_datasets / v1.5/can/ph/demo_2 | Can | PickPlaceCan | True | False |
| hf__robomimic__robomimic_datasets / v1.5/square/ph/demo_2 | SquareNut | NutAssemblySquare | True | False |
| hf__robomimic__robomimic_datasets / v1.5/lift/ph/demo_2 | cube | Lift | True | False |
| hf__robomimic__robomimic_datasets / v1.5/square/ph/demo_1 | SquareNut | NutAssemblySquare | True | False |
| hf__robomimic__robomimic_datasets / v1.5/tool_hang/ph/demo_1 | frame, stand, tool | ToolHang | True | False |
| hf__robomimic__robomimic_datasets / v1.5/lift/ph/demo_0 | cube | Lift | True | False |
| hf__robomimic__robomimic_datasets / v1.5/can/ph/demo_0 | Can | PickPlaceCan | True | True |
| hf__robomimic__robomimic_datasets / v1.5/tool_hang/ph/demo_0 | frame, stand, tool | ToolHang | True | False |
| hf__robomimic__robomimic_datasets / v1.5/transport/ph/demo_0 | payload, transport_start_bin, transport_start_bin_lid, transport_target_bin, transport_trash_bin, trash | TwoArmTransport | False | False |
| hf__robomimic__robomimic_datasets / v1.5/square/ph/demo_0 | SquareNut | NutAssemblySquare | True | False |
| hf__robomimic__robomimic_datasets / v1.5/can/ph/demo_1 | Can | PickPlaceCan | True | False |
| hf__robomimic__robomimic_datasets / v1.5/transport/ph/demo_2 | payload, transport_start_bin, transport_start_bin_lid, transport_target_bin, transport_trash_bin, trash | TwoArmTransport | False | False |
| hf__robomimic__robomimic_datasets / v1.5/transport/ph/demo_1 | payload, transport_start_bin, transport_start_bin_lid, transport_target_bin, transport_trash_bin, trash | TwoArmTransport | False | False |
| humoto / carry_organizer_with_both_hands_at_chest_height-436 | draw_organizer_tray | unlabeled | True | False |
| humoto / drinking_from_mug_with_right_hand-815 | mug | unlabeled | True | False |
| humoto / baking_with_spatula_mixing_bowl_and_scooping_to_tray-244 | mixing_bowl, spatula, table, tray | unlabeled | False | False |

Unlabeled object types/scenarios remain unlabeled. A task name containing an object does not qualify an episode for this object-aware pipeline.

## Scenario coverage by run root

| Root | Dataset | Source scenario | Cases | Kinematic passes | Contact backend |
| --- | --- | --- | ---: | ---: | --- |
| local-preview | hf__robomimic__robomimic_datasets | PickPlaceCan | 1 | 1 | Can validator available |
| local-preview-v2 | hf__robomimic__robomimic_datasets | PickPlaceCan | 1 | 1 | Can validator available |
| object-matrix | hf__amandlek__mimicgen_datasets | Stack_D0 | 3 | 3 | unsupported scene |
| object-matrix | hf__amandlek__mimicgen_datasets | Threading_D0 | 3 | 3 | unsupported scene |
| object-matrix | hf__haosulab__ManiSkill_Demonstrations | PickCube-v1 | 5 | 5 | unsupported scene |
| object-matrix | hf__robomimic__robomimic_datasets | Lift | 3 | 3 | unsupported scene |
| object-matrix | hf__robomimic__robomimic_datasets | NutAssemblySquare | 3 | 3 | unsupported scene |
| object-matrix | hf__robomimic__robomimic_datasets | PickPlaceCan | 3 | 3 | Can validator available |
| object-matrix | hf__robomimic__robomimic_datasets | ToolHang | 3 | 3 | unsupported scene |
| object-matrix | hf__robomimic__robomimic_datasets | TwoArmTransport | 3 | 0 | unsupported scene |
| object-matrix | humoto | baking_with_spatula_mixing_bowl_and_scooping_to_tray-244 (source sequence label) | 1 | 0 | unsupported scene |
| object-matrix | humoto | carry_organizer_with_both_hands_at_chest_height-436 (source sequence label) | 1 | 1 | unsupported scene |
| object-matrix | humoto | drinking_from_mug_with_right_hand-815 (source sequence label) | 1 | 1 | unsupported scene |

Only the robomimic v1.5 proficient-human Can scene has a dynamic contact validator in the current implementation. All other object/scenario rows are unsupported for physical validation, including kinematically passing cube, nut, stacking, threading, tool-hanging and HUMOTO cases. They require scene/collision assets, contact and gripper adapters, and task-specific success criteria. No contact success is inferred from object-pose playback or source success labels.

## Kinematic failures

| Root / source sequence | Position error (mm) | Orientation error (degrees) | Minimum self-clearance (mm) | Time dilation | Neck pass |
| --- | ---: | ---: | ---: | ---: | --- |
| object-matrix / v1.5/transport/ph/demo_0 | 46.105 | 31.442 | -70.763 | 1.752 | True |
| object-matrix / v1.5/transport/ph/demo_2 | 84.119 | 33.283 | -65.556 | 3.309 | True |
| object-matrix / v1.5/transport/ph/demo_1 | 53.562 | 31.078 | -65.506 | 2.717 | True |
| object-matrix / baking_with_spatula_mixing_bowl_and_scooping_to_tray-244 | 24.550 | 6.682 | -10.754 | 11.810 | False |

Negative self-clearance indicates overlapping collision proxies. These failures remain failed despite retiming; the underlying files and measured errors are retained.

## Contact attempts and protocol limits

| Root / attempt | Source sequence | Saved result | Lift (mm) | Final stable time (s) | Maximum hand/object penetration (mm) | Failure reasons |
| --- | --- | --- | ---: | ---: | ---: | --- |
| local-preview-v2 / can_contact_v1/demo_0 | v1.5/can/ph/demo_0 | physical_fail | 0.296 | 0.000 | 0.000 | RuntimeError: allocation QP: maximum iterations reached; primal=0.000108, dual=0.0012, object_not_lifted_8cm, no_closed_gripper_object_contact, final_placement_not_stable_open_no_hand_contact_for_1s, actual_physics_velocity_limit_exceeded, self_clearance_below_9mm, joint_margin_below_25mrad, no_bilateral_finger_contact_during_lift |
| local-preview-v2 / can_contact_v2/demo_0 | v1.5/can/ph/demo_0 | physical_pass | 119.770 | 8.970 | 0.891 | none recorded |
| local-preview-v2 / can_contact_v3/demo_0 | v1.5/can/ph/demo_0 | physical_pass | 119.778 | 8.970 | 0.891 | none recorded |
| local-preview-v2 / speed4_baseline/demo_0 | v1.5/can/ph/demo_0 | physical_fail | 121.439 | 4.990 | 1.305 | hand_can_penetration_above_1mm, non_fingertip_can_penetration_above_1mm, no_bilateral_finger_contact_during_lift |
| local-preview-v2 / speed4_center15_fixed/demo_0 | v1.5/can/ph/demo_0 | physical_fail | 0.296 | 0.000 | 0.000 | RuntimeError: allocation QP: maximum iterations reached; primal=3.96e-05, dual=0.000606, object_not_lifted_8cm, no_closed_gripper_object_contact, final_placement_not_stable_open_no_hand_contact_for_1s, actual_physics_velocity_limit_exceeded, self_clearance_below_9mm, no_bilateral_finger_contact_during_lift |
| local-preview-v2 / speed4_center15_tilt20/demo_0 | v1.5/can/ph/demo_0 | physical_fail | 0.296 | 0.000 | 0.000 | RuntimeError: allocation QP: maximum iterations reached; primal=1.38e-06, dual=0.00118, object_not_lifted_8cm, no_closed_gripper_object_contact, final_placement_not_stable_open_no_hand_contact_for_1s, self_clearance_below_9mm, no_bilateral_finger_contact_during_lift |
| local-preview-v2 / speed4_center25/demo_0 | v1.5/can/ph/demo_0 | physical_fail | 0.296 | 0.000 | 0.000 | RuntimeError: allocation QP: maximum iterations reached; primal=5.29e-06, dual=0.00875, object_not_lifted_8cm, no_closed_gripper_object_contact, final_placement_not_stable_open_no_hand_contact_for_1s, actual_physics_velocity_limit_exceeded, self_clearance_below_9mm, joint_margin_below_25mrad, no_bilateral_finger_contact_during_lift |
| local-preview-v2 / speed4_center25_branch/demo_0 | v1.5/can/ph/demo_0 | physical_fail | 6.547 | 0.000 | 0.246 | RuntimeError: allocation QP: maximum iterations reached; primal=3.51e-05, dual=0.000213, object_not_lifted_8cm, final_placement_not_stable_open_no_hand_contact_for_1s, actual_physics_velocity_limit_exceeded, self_clearance_below_9mm, joint_margin_below_25mrad, no_bilateral_finger_contact_during_lift |
| local-preview-v2 / speed4_depth20/demo_0 | v1.5/can/ph/demo_0 | physical_fail | 0.296 | 0.000 | 0.000 | object_not_lifted_8cm, no_closed_gripper_object_contact, final_placement_not_stable_open_no_hand_contact_for_1s, actual_physics_velocity_limit_exceeded, self_clearance_below_9mm, no_bilateral_finger_contact_during_lift |
| local-preview-v2 / speed4_grasp10_tilt12/demo_0 | v1.5/can/ph/demo_0 | physical_fail | 108.856 | 0.000 | 1.073 | RuntimeError: allocation QP: maximum iterations reached; primal=2.56e-05, dual=0.000417, final_placement_not_stable_open_no_hand_contact_for_1s, hand_can_penetration_above_1mm, grasp_stability_criteria_not_met |
| local-preview-v2 / speed4_grasp15_motion/demo_0 | v1.5/can/ph/demo_0 | physical_fail | 144.597 | 5.060 | 1.059 | hand_can_penetration_above_1mm |
| local-preview-v2 / speed4_grasp15_soft/demo_0 | v1.5/can/ph/demo_0 | physical_pass | 144.911 | 5.150 | 0.858 | none recorded |
| local-preview-v2 / speed4_grasp15_tilt15/demo_0 | v1.5/can/ph/demo_0 | physical_fail | 144.589 | 0.000 | 0.966 | RuntimeError: allocation QP: maximum iterations reached; primal=6.81e-05, dual=0.000182, final_placement_not_stable_open_no_hand_contact_for_1s |
| local-preview-v2 / speed4_grasp15_tilt8/demo_0 | v1.5/can/ph/demo_0 | physical_fail | 138.691 | 5.100 | 1.797 | actual_physics_velocity_limit_exceeded, hand_can_penetration_above_1mm, grasp_stability_criteria_not_met |
| local-preview-v2 / speed5_grasp15_tilt15/demo_0 | v1.5/can/ph/demo_0 | physical_fail | 144.628 | 0.000 | 1.295 | RuntimeError: allocation QP: maximum iterations reached; primal=1.39e-06, dual=0.00147, final_placement_not_stable_open_no_hand_contact_for_1s, hand_can_penetration_above_1mm |
| object-matrix / fixed_grasp_three_can/demo_0 | v1.5/can/ph/demo_0 | physical_pass | 144.911 | 5.150 | 0.858 | none recorded |
| object-matrix / fixed_grasp_three_can/demo_1 | v1.5/can/ph/demo_1 | physical_fail | 134.345 | 5.410 | 0.966 | actual_physics_velocity_limit_exceeded, no_bilateral_finger_contact_during_lift, grasp_stability_criteria_not_met |
| object-matrix / fixed_grasp_three_can/demo_2 | v1.5/can/ph/demo_2 | physical_fail | 246.931 | 4.930 | 0.905 | actual_physics_velocity_limit_exceeded |

Each attempt retains its own criteria in the JSON. Some earlier Can development passes predate the newer grasp-drift gate and are not interchangeable with the fixed-grasp three-case matrix. Repeated Can tuning attempts and copies across preview roots are one source demonstration when their recorded source group matches. The current matrix is reported separately by root above.

## Aggregated evidence

```json
{
  "roots": 4,
  "catalogued_source_ids": 334,
  "locally_downloaded_source_ids": 6,
  "normalized_artifact_copies": 31,
  "distinct_recorded_source_groups": 29,
  "object_aware_source_groups": 29,
  "kinematic_pass_source_groups": 25,
  "contact_validated_source_groups": 1,
  "contact_attempts": 18,
  "contact_reported_passes": 4,
  "contact_artifact_verified_passes": 4,
  "object_aware_dataset_ids": [
    "hf__amandlek__mimicgen_datasets",
    "hf__haosulab__ManiSkill_Demonstrations",
    "hf__robomimic__robomimic_datasets",
    "humoto"
  ],
  "kinematic_pass_dataset_ids": [
    "hf__amandlek__mimicgen_datasets",
    "hf__haosulab__ManiSkill_Demonstrations",
    "hf__robomimic__robomimic_datasets",
    "humoto"
  ],
  "contact_validated_dataset_ids": [
    "hf__robomimic__robomimic_datasets"
  ],
  "source_scoped_object_ids": [
    "hf__amandlek__mimicgen_datasets/cubeA",
    "hf__amandlek__mimicgen_datasets/cubeB",
    "hf__amandlek__mimicgen_datasets/needle_obj",
    "hf__amandlek__mimicgen_datasets/tripod_obj",
    "hf__haosulab__ManiSkill_Demonstrations/cube",
    "hf__robomimic__robomimic_datasets/Can",
    "hf__robomimic__robomimic_datasets/SquareNut",
    "hf__robomimic__robomimic_datasets/cube",
    "hf__robomimic__robomimic_datasets/frame",
    "hf__robomimic__robomimic_datasets/payload",
    "hf__robomimic__robomimic_datasets/stand",
    "hf__robomimic__robomimic_datasets/tool",
    "hf__robomimic__robomimic_datasets/transport_start_bin",
    "hf__robomimic__robomimic_datasets/transport_start_bin_lid",
    "hf__robomimic__robomimic_datasets/transport_target_bin",
    "hf__robomimic__robomimic_datasets/transport_trash_bin",
    "hf__robomimic__robomimic_datasets/trash",
    "humoto/draw_organizer_tray",
    "humoto/mixing_bowl",
    "humoto/mug",
    "humoto/spatula",
    "humoto/table",
    "humoto/tray"
  ],
  "labeled_scenarios": [
    "Lift",
    "NutAssemblySquare",
    "PickCube-v1",
    "PickPlaceCan",
    "Stack_D0",
    "Threading_D0",
    "ToolHang",
    "TwoArmTransport"
  ]
}
```

Object-aware requires a numeric time-aligned xyz+wxyz object pose with at least one valid frame. Kinematic pass requires matching saved output and object channels. Contact pass requires the saved result, scene checksum, complete-state replay, linked object episode and declared free dynamic object policy; it is not an independent reproduction.

## Reproduce

```bash
.venv/bin/python -m reachy_retarget.assessment --root . --root runs/local-preview --root runs/local-preview-v2 --root runs/object-matrix --output runs/coverage/assessment.json --markdown docs/coverage-audit.md
```

The JSON retains per-source local file evidence, missing files, episode checksums, validity coverage, exact source groups, failed contact attempts and per-attempt criteria. No acquisition ledger or source payload is changed.
