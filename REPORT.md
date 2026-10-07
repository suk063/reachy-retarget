# Reachy retarget collection report

Snapshot: 2026-10-07T03:21:47.798237+00:00

**Acquisition is still incomplete:** 574,827 files remain in the durable queue. Provider cooldowns, live workers and individual failures are separate from completed conversions. See `runs/collector-status.json` and `runs/collector.log`.

This report separates acquired source states, canonical episodes, kinematic conversions, and actual contact simulation. Counts are measured from local payloads, not advertised repository totals.

| Measure | Observed result |
|---|---:|
| Releases with payloads (not independent corpora) | 159 |
| Corpus families with acquired payloads | 148 |
| Parsed sequences after known family overlap suppression | 257,780 |
| Known hours in those sequences | 1,285.85 |
| Unique acquired content bytes | 133.802 GB |
| Canonical representative episodes | 30 |
| Linked canonical object identities | 31 |
| Source-scoped objects with verified pose/mesh links | 214 |
| Kinematic passes / representative attempts | 14 / 25 |
| Existing tracking audit passes | 14 |
| Actual contact task successes | 4 / 10 |
| Contact successes passing every physical check | 2 / 10 |

**Counting limits:** Raw annotation archives not yet parsed are excluded from sequence/hour counts. Reformat variants and known subsets are not added as independent demonstrations. Canonical object IDs and verified raw pose/mesh links are separate counts; source-scoped IDs are not asserted to identify distinct physical objects across corpora. See `catalog/object_inventory.json`. The exact sequence inventory, representation choices and exclusions are in `runs/summary.json` and `catalog/sequence_inventory.jsonl`.

## Representative conversions

| Source | Canonical | Kinematic pass / attempted |
|---|---:|---:|
| hf__glannuzel__reachy2_pick_place | 5 | 5 / 5 |
| hf__pollen-robotics__pick_and_place_bottle | 5 | 3 / 5 |
| hf__robomimic__robomimic_datasets | 10 | 5 / 5 |
| humoto | 5 | 1 / 5 |
| parahome | 5 | 0 / 5 |

Rejected motion HDF5 files remain available with numerical reasons; only passing arm/base trajectories receive training NPZ files. Neck channels and their residuals are separate because the existing tracking environment locks the neck. Missing native gripper/object observations remain missing.

## Fixed Can contact benchmark

| Source sequence | Task | All checks | Lift (m) | Final stable open/no-contact (s) | Failure reasons |
|---|---|---|---:|---:|---|
| v1.5/can/ph/demo_0 | pass | pass | 0.1351 | 8.72 | none |
| v1.5/can/ph/demo_1 | fail | fail | 0.0022 | 0.00 | object_not_lifted_8cm, no_closed_gripper_object_contact, final_placement_not_stable_open_no_hand_contact_for_1s |
| v1.5/can/ph/demo_2 | pass | pass | 0.2367 | 8.96 | none |
| v1.5/can/ph/demo_3 | fail | fail | 0.0027 | 0.00 | object_not_lifted_8cm, final_placement_not_stable_open_no_hand_contact_for_1s |
| v1.5/can/ph/demo_4 | pass | fail | 0.1916 | 8.54 | actual_physics_velocity_limit_exceeded |
| v1.5/can/ph/demo_5 | fail | fail | 0.0010 | 0.00 | object_not_lifted_8cm, no_closed_gripper_object_contact, final_placement_not_stable_open_no_hand_contact_for_1s, actual_physics_velocity_limit_exceeded |
| v1.5/can/ph/demo_6 | fail | fail | 0.0056 | 0.00 | object_not_lifted_8cm, final_placement_not_stable_open_no_hand_contact_for_1s, actual_physics_velocity_limit_exceeded, self_clearance_below_9mm, joint_margin_below_25mrad |
| v1.5/can/ph/demo_7 | pass | fail | 0.2374 | 8.52 | actual_physics_velocity_limit_exceeded, self_clearance_below_9mm |
| v1.5/can/ph/demo_8 | fail | fail | 0.0003 | 0.00 | object_not_lifted_8cm, no_closed_gripper_object_contact, final_placement_not_stable_open_no_hand_contact_for_1s, actual_physics_velocity_limit_exceeded |
| v1.5/can/ph/demo_9 | fail | fail | 0.0039 | 0.00 | object_not_lifted_8cm, no_closed_gripper_object_contact, final_placement_not_stable_open_no_hand_contact_for_1s |

The Can stays a dynamic free body under gravity. Source mass, inertia and friction are retained. Replay uses 500 Hz physics and the pinned native controller at 100 Hz, transforming world targets into the measured base frame. A global time scale of 10 and a constant grasp attachment depth of 0.035 m are recorded morphology adjustments. No object weld, runtime pose assignment or target force is used. Videos and actual/command/goal state are in `runs/contact/can_fixed10_final/`.

**Simulation limits:** ideal planar base servos, robot-only gravity compensation, convex robot collision meshes and assumed robot/environment friction. Calibration used the same ten development demonstrations; these are reproducible feasibility tests, not a held-out generalization estimate. Four trials complete the contact task; stricter joint-speed/collision checks determine the final pass count.

## Initial candidate coverage

| Candidate | State | Evidence / scope |
|---|---|---|
| [omomo](https://github.com/lijiaman/omomo_release) | partial_acquired | State/geometry acquired from: intermimic, omomo. Per-release scope and pending files are explicit below. |
| [parahome](https://github.com/snuvclab/ParaHome) | partial_acquired | State/geometry acquired from: parahome. Per-release scope and pending files are explicit below. |
| [oakink](https://oakink.net/) | partial_acquired | State/geometry acquired from: hf__oakink__OakInk-v1. Per-release scope and pending files are explicit below. |
| [oakink2](https://github.com/oakink/OakInk2) | partial_acquired | State/geometry acquired from: hf__kelvin34501__OakInk-v2. Per-release scope and pending files are explicit below. |
| [taco](https://github.com/leolyliu/TACO-Instructions) | partial_acquired | State/geometry acquired from: taco. Per-release scope and pending files are explicit below. |
| [dexycb](https://dex-ycb.github.io/) | partial_acquired | State/geometry acquired from: dexycb, hf__QFun__MANUS-DexYCB. Per-release scope and pending files are explicit below. |
| [hot3d](https://facebookresearch.github.io/hot3d/) | partial_acquired | State/geometry acquired from: hf__MIT-Media-Lab__hot3d-annotations-v1, hf__bop-benchmark__hot3d. Per-release scope and pending files are explicit below. |
| [humoto](https://github.com/adobe-research/humoto) | partial_acquired | State/geometry acquired from: humoto. Per-release scope and pending files are explicit below. |
| [hoi4d](https://github.com/leolyliu/HOI4D-Instructions) | partial_acquired | State/geometry acquired from: hf__JingkunAn__lerobot_hoi4d_dataset. Per-release scope and pending files are explicit below. |
| [h2o](https://h2odataset.ethz.ch/) | partial_acquired | State/geometry acquired from: hf__MIT-Media-Lab__h2o-annotations-v1. Per-release scope and pending files are explicit below. |
| [ho3d](https://github.com/shreyashampali/ho3d) | partial_acquired | State/geometry acquired from: hf__Ronaldo-GOAT__ho3d_transfer. Per-release scope and pending files are explicit below. |
| [grab](https://grab.is.tue.mpg.de/) | access_required | No manual applications or new accounts; inspect official public release routes. |
| [behave](https://virtualhumans.mpi-inf.mpg.de/behave/) | partial_acquired | State/geometry acquired from: behave. Per-release scope and pending files are explicit below. |
| [intercap](https://intercap.is.tue.mpg.de/) | access_required | No manual applications or new accounts; inspect official public release routes. |
| [arctic](https://arctic.is.tue.mpg.de/) | access_required | No manual applications or new accounts; inspect official public release routes. |
| [hodome](https://github.com/Juzezhang/NeuralDome_Toolbox) | partial_acquired | State/geometry acquired from: hodome. Per-release scope and pending files are explicit below. |
| [chairs](https://jnnan.github.io/chairs/) | access_required | Official dataset page links an access request form. No public official state archive resolved; no form submitted. |
| [interact](https://github.com/wzyabcas/InterAct) | access_required | Official README Dataset Preparation step 1 requires non-commercial authorization form; no application submitted. Public original component corpora acquired separately. |
| [intermimic](https://github.com/Sirui-Xu/InterMimic) | partial_acquired | State/geometry acquired from: intermimic. Per-release scope and pending files are explicit below. |
| [hoi_retarget](https://github.com/leggedrobotics/hoi-retarget) | partial_acquired | State/geometry acquired from: hf__leggedrobotics__hoi-retarget. Per-release scope and pending files are explicit below. |
| [oxe](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | State/geometry acquired from: hf__lerobot__jaco_play, hf__lerobot__roboturk, hf__lerobot__taco_play, hf__lerobot__nyu_door_opening_surprising_effectiveness, hf__lerobot__toto, hf__lerobot__berkeley_autolab_ur5, hf__lerobot__columbia_cairlab_pusht_real, hf__lerobot__stanford_kuka_multimodal_dataset, hf__lerobot__stanford_hydra_dataset, hf__lerobot__nyu_rot_dataset, hf__lerobot__austin_buds_dataset, hf__lerobot__berkeley_cable_routing, hf__lerobot__nyu_franka_play_dataset, hf__lerobot__cmu_franka_exploration_dataset, hf__lerobot__ucsd_kitchen_dataset, hf__lerobot__ucsd_pick_and_place_dataset, hf__lerobot__austin_sailor_dataset, hf__lerobot__austin_sirius_dataset, hf__lerobot__usc_cloth_sim, hf__lerobot__utokyo_pr2_opening_fridge, hf__lerobot__utokyo_pr2_tabletop_manipulation, hf__lerobot__utokyo_saytap, hf__lerobot__utokyo_xarm_pick_and_place, hf__lerobot__utokyo_xarm_bimanual, hf__lerobot__berkeley_mvp, hf__lerobot__berkeley_rpt, hf__lerobot__tokyo_u_lsmo, hf__lerobot__dlr_sara_pour, hf__lerobot__kaist_nonprehensile, hf__lerobot__dlr_sara_grid_clamp, hf__lerobot__dlr_edan_shared_control, hf__lerobot__asu_table_top, hf__lerobot__stanford_robocook, hf__lerobot__iamlab_cmu_pickup_insert, hf__lerobot__utaustin_mutex, hf__lerobot__berkeley_fanuc_manipulation, hf__lerobot__imperialcollege_sawyer_wrist_cam, hf__lerobot__uiuc_d3field, hf__lerobot__cmu_stretch, hf__lerobot__cmu_play_fusion, hf__lerobot__berkeley_gnm_cory_hall, hf__lerobot__berkeley_gnm_sac_son, hf__lerobot__conq_hose_manipulation, hf__lerobot__fmb, hf__lerobot__berkeley_gnm_recon, hf__lerobot__aloha_sim_transfer_cube_human, hf__lerobot__aloha_sim_insertion_scripted, hf__lerobot__umi_cup_in_the_wild, hf__lerobot__aloha_sim_transfer_cube_scripted, hf__lerobot__libero, hf__lerobot__aloha_sim_insertion_human, hf__lerobot__aloha_static_candy, hf__lerobot__aloha_static_tape, hf__lerobot__aloha_static_screw_driver, hf__lerobot__aloha_mobile_wipe_wine, hf__lerobot__aloha_static_towel, hf__lerobot__aloha_static_coffee, hf__lerobot__aloha_static_battery, hf__lerobot__aloha_static_thread_velcro, hf__lerobot__aloha_static_vinh_cup_left, hf__lerobot__aloha_static_vinh_cup, hf__lerobot__aloha_static_coffee_new, hf__lerobot__aloha_static_ziploc_slide, hf__lerobot__aloha_static_pingpong_test, hf__lerobot__aloha_static_cups_open, hf__lerobot__aloha_static_fork_pick_up, hf__lerobot__aloha_static_pro_pencil, hf__lerobot__aloha_mobile_shrimp, hf__lerobot__aloha_mobile_wash_pan, hf__lerobot__aloha_mobile_cabinet, hf__lerobot__aloha_mobile_chair, hf__lerobot__aloha_mobile_elevator, hf__lerobot__droid_100, hf__lerobot__metaworld_mt50, hf__lerobot__droid_1.0.1, hf__lerobot__libero_10, hf__lerobot__libero_plus, hf__lerobot__robocasa_target_human_unified, oxe__aloha_mobile, oxe__bridge, oxe__cmu_playing_with_food, oxe__eth_agent_affordances, oxe__fractal20220817_data, oxe__io_ai_tech, oxe__maniskill_dataset_converted_externally_to_rlds, oxe__mimic_play, oxe__plex_robosuite, oxe__robo_net, oxe__robo_set, oxe__robot_vqa, oxe__spoc, oxe__stanford_mask_vit_converted_externally_to_rlds, oxe__tidybot, oxe__vima_converted_externally_to_rlds, oxe__kuka, oxe__viola, oxe__furniture_bench_dataset_converted_externally_to_rlds, oxe__bc_z, oxe__droid, oxe__dobbe, oxe__qut_dexterous_manipulation. Per-release scope and pending files are explicit below. |
| [droid](https://droid-dataset.github.io/droid/the-droid-dataset) | partial_acquired | State/geometry acquired from: hf__lerobot__droid_100, hf__lerobot__droid_1.0.1, oxe__droid. Per-release scope and pending files are explicit below. |
| [bridge](https://rail-berkeley.github.io/bridgedata/) | partial_acquired | State/geometry acquired from: oxe__bridge, hf__IPEC-COMMUNITY__bridge_orig_lerobot. Per-release scope and pending files are explicit below. |
| [aloha](https://github.com/tonyzhaozh/act) | partial_acquired | State/geometry acquired from: hf__lerobot__aloha_sim_transfer_cube_human, hf__lerobot__aloha_sim_insertion_scripted, hf__lerobot__aloha_sim_transfer_cube_scripted, hf__lerobot__aloha_sim_insertion_human, hf__lerobot__aloha_static_candy, hf__lerobot__aloha_static_tape, hf__lerobot__aloha_static_screw_driver, hf__lerobot__aloha_static_towel, hf__lerobot__aloha_static_coffee, hf__lerobot__aloha_static_battery, hf__lerobot__aloha_static_thread_velcro, hf__lerobot__aloha_static_vinh_cup_left, hf__lerobot__aloha_static_vinh_cup, hf__lerobot__aloha_static_coffee_new, hf__lerobot__aloha_static_ziploc_slide, hf__lerobot__aloha_static_pingpong_test, hf__lerobot__aloha_static_cups_open, hf__lerobot__aloha_static_fork_pick_up, hf__lerobot__aloha_static_pro_pencil. Per-release scope and pending files are explicit below. |
| [mobile_aloha](https://github.com/MarkFzp/mobile-aloha) | partial_acquired | State/geometry acquired from: hf__lerobot__aloha_mobile_wipe_wine, hf__lerobot__aloha_mobile_shrimp, hf__lerobot__aloha_mobile_wash_pan, hf__lerobot__aloha_mobile_cabinet, hf__lerobot__aloha_mobile_chair, hf__lerobot__aloha_mobile_elevator, oxe__aloha_mobile. Per-release scope and pending files are explicit below. |
| [umi](https://github.com/real-stanford/universal_manipulation_interface) | partial_acquired | State/geometry acquired from: hf__lerobot__umi_cup_in_the_wild. Per-release scope and pending files are explicit below. |
| [fastumi](https://github.com/YdingTeam/FastUMI) | partial_acquired | State/geometry acquired from: hf__IPEC-COMMUNITY__FastUMI_100k_lerobot. Per-release scope and pending files are explicit below. |
| [roboturk](https://roboturk.stanford.edu/dataset_real.html) | partial_acquired | State/geometry acquired from: hf__lerobot__roboturk. Per-release scope and pending files are explicit below. |
| [robomimic](https://github.com/ARISE-Initiative/robomimic) | partial_acquired | State/geometry acquired from: hf__robomimic__robomimic_datasets. Per-release scope and pending files are explicit below. |
| [mimicgen](https://github.com/NVlabs/mimicgen) | partial_acquired | State/geometry acquired from: hf__amandlek__mimicgen_datasets. Per-release scope and pending files are explicit below. |
| [robocasa](https://github.com/robocasa/robocasa) | partial_acquired | State/geometry acquired from: hf__lerobot__robocasa_target_human_unified. Per-release scope and pending files are explicit below. |
| [libero](https://github.com/Lifelong-Robot-Learning/LIBERO) | partial_acquired | State/geometry acquired from: hf__lerobot__libero, hf__lerobot__libero_10, hf__lerobot__libero_plus. Per-release scope and pending files are explicit below. |
| [maniskill](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html) | partial_acquired | State/geometry acquired from: hf__haosulab__ManiSkill_Demonstrations. Per-release scope and pending files are explicit below. |
| [calvin](https://github.com/mees/calvin) | partial_acquired | State/geometry acquired from: calvin. Per-release scope and pending files are explicit below. |
| [furniturebench](https://github.com/clvrai/furniture-bench) | partial_acquired | State/geometry acquired from: hf__IPEC-COMMUNITY__furniture_bench_dataset_lerobot, oxe__furniture_bench_dataset_converted_externally_to_rlds. Per-release scope and pending files are explicit below. |
| [d4rl_kitchen](https://github.com/Farama-Foundation/D4RL) | partial_acquired | State/geometry acquired from: d4rl_kitchen. Per-release scope and pending files are explicit below. |
| [cmu](https://mocap.cs.cmu.edu/) | partial_acquired | State/geometry acquired from: cmu. Per-release scope and pending files are explicit below. |
| [lafan1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) | partial_acquired | State/geometry acquired from: lafan1. Per-release scope and pending files are explicit below. |
| [amass](https://amass.is.tue.mpg.de/) | access_required | No manual applications or new accounts; inspect official public release routes. |

## Every discovered release

Full versions, licenses, feature names, file outcomes and reasons are in [datasets.json](catalog/datasets.json). The durable queue is `catalog/ledger.sqlite`.

| Release | State | Files | GB | Parsed sequences | License |
|---|---|---:|---:|---:|---|
| [hf__alekgomez__reachy_mini_info_benchmark](https://huggingface.co/datasets/alekgomez/reachy_mini_info_benchmark) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__binhpham__reachy-mini-massive-motion-library](https://huggingface.co/datasets/binhpham/reachy-mini-massive-motion-library) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__test_gripette_050526](https://huggingface.co/datasets/pollen-robotics/test_gripette_050526) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__binhpham__reachy-mini-motion-synth](https://huggingface.co/datasets/binhpham/reachy-mini-motion-synth) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__tfrere__reachy-mini-app-moderation](https://huggingface.co/datasets/tfrere/reachy-mini-app-moderation) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__tfrere__reachy-mini-onboarding-moves](https://huggingface.co/datasets/tfrere/reachy-mini-onboarding-moves) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__RemiFabre__reachy-mini-emotions-library-50hz-opus](https://huggingface.co/datasets/RemiFabre/reachy-mini-emotions-library-50hz-opus) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__tavis-benchmark__tavis-hands-reachy2](https://huggingface.co/datasets/tavis-benchmark/tavis-hands-reachy2) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__tfrere__reachy-personalities](https://huggingface.co/datasets/tfrere/reachy-personalities) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__BastienATOS__mini-reachy-animation](https://huggingface.co/datasets/BastienATOS/mini-reachy-animation) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__tfrere__reachy-mini-moves](https://huggingface.co/datasets/tfrere/reachy-mini-moves) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__grabette-community-open-a-door](https://huggingface.co/datasets/pollen-robotics/grabette-community-open-a-door) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__apirrone__reachy-mini-simon-scores](https://huggingface.co/datasets/apirrone/reachy-mini-simon-scores) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__fridaf__Reachy-state](https://huggingface.co/datasets/fridaf/Reachy-state) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__yourbench__reachy_mini_info_benchmark](https://huggingface.co/datasets/yourbench/reachy_mini_info_benchmark) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__pollen_robotic_fleet_usage](https://huggingface.co/datasets/pollen-robotics/pollen_robotic_fleet_usage) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__grabette-community-tasks](https://huggingface.co/datasets/pollen-robotics/grabette-community-tasks) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__tfrere__reachy-mini-app-categories](https://huggingface.co/datasets/tfrere/reachy-mini-app-categories) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__reachy-mini-emotions-library](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__tfrere__reachy-sticker-gallery](https://huggingface.co/datasets/tfrere/reachy-sticker-gallery) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__tavis-benchmark__tavis-head-reachy2](https://huggingface.co/datasets/tavis-benchmark/tavis-head-reachy2) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__tommulder__Reachy_RENS](https://huggingface.co/datasets/tommulder/Reachy_RENS) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__tfrere__reachy-mini-generated-moves](https://huggingface.co/datasets/tfrere/reachy-mini-generated-moves) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__reachy2_emotions_library](https://huggingface.co/datasets/pollen-robotics/reachy2_emotions_library) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__reachy_mini_store_data](https://huggingface.co/datasets/pollen-robotics/reachy_mini_store_data) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__tfrere__reachy-mini-trailer-video](https://huggingface.co/datasets/tfrere/reachy-mini-trailer-video) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__reachy-mini-wall-data](https://huggingface.co/datasets/pollen-robotics/reachy-mini-wall-data) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__tfrere__reachy-mini-move-previews](https://huggingface.co/datasets/tfrere/reachy-mini-move-previews) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__clem__opus4.7_reachy_mini_app_building](https://huggingface.co/datasets/clem/opus4.7_reachy_mini_app_building) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__FabienDanieau__reachy-mini-detection-dataset-v2](https://huggingface.co/datasets/FabienDanieau/reachy-mini-detection-dataset-v2) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__anyskin_slip_detection](https://huggingface.co/datasets/pollen-robotics/anyskin_slip_detection) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__FabienDanieau__reachy-mini-detection-dataset](https://huggingface.co/datasets/FabienDanieau/reachy-mini-detection-dataset) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__tavis-benchmark__tavis-head-sample-reachy2](https://huggingface.co/datasets/tavis-benchmark/tavis-head-sample-reachy2) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__haixuantao__reachy-mini-official-app-store](https://huggingface.co/datasets/haixuantao/reachy-mini-official-app-store) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__Anne-Charlotte__reachy-mini-arcade-leaderboard](https://huggingface.co/datasets/Anne-Charlotte/reachy-mini-arcade-leaderboard) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__squaredcuber__animacy-reachy-mini-lerobot](https://huggingface.co/datasets/squaredcuber/animacy-reachy-mini-lerobot) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__jyvet__reachy-mini-emotions-library-noaudio](https://huggingface.co/datasets/jyvet/reachy-mini-emotions-library-noaudio) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__soichgroups__reachytodo](https://huggingface.co/datasets/soichgroups/reachytodo) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__grabette-community-open-a-drawer](https://huggingface.co/datasets/pollen-robotics/grabette-community-open-a-drawer) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__dibsit__reachy-mini-emotions-library-silent](https://huggingface.co/datasets/dibsit/reachy-mini-emotions-library-silent) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__TwinPeaksTownie__Reachy_RENS](https://huggingface.co/datasets/TwinPeaksTownie/Reachy_RENS) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__microduck-emotions](https://huggingface.co/datasets/pollen-robotics/microduck-emotions) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__reachy-mini-official-app-store](https://huggingface.co/datasets/pollen-robotics/reachy-mini-official-app-store) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__Anne-Charlotte__reachy-songs](https://huggingface.co/datasets/Anne-Charlotte/reachy-songs) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__reachy-mini-dances-library](https://huggingface.co/datasets/pollen-robotics/reachy-mini-dances-library) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__cdeplanne__reachy-mini-arcade-leaderboard](https://huggingface.co/datasets/cdeplanne/reachy-mini-arcade-leaderboard) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__speech-commands-v0.02](https://huggingface.co/datasets/pollen-robotics/speech-commands-v0.02) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__grabette-community-grasp-a-spoon](https://huggingface.co/datasets/pollen-robotics/grabette-community-grasp-a-spoon) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__HTurlet15__reachy-alive](https://huggingface.co/datasets/HTurlet15/reachy-alive) | metadata_only | 3 | 0.001 | 0 | apache-2.0 |
| [hf__Voxel51__tavis-head-reachy2-800ep](https://huggingface.co/datasets/Voxel51/tavis-head-reachy2-800ep) | partial_acquired | 6 | 0.067 | 800 | cc-by-4.0 |
| [hf__THULab__reachy_wave](https://huggingface.co/datasets/THULab/reachy_wave) | metadata_only | 5 | 0.000 | 0 | apache-2.0 |
| [hf__aliberts__eval_reachy_test](https://huggingface.co/datasets/aliberts/eval_reachy_test) | partial_acquired | 6 | 0.000 | 1 | apache-2.0 |
| [hf__aliberts__reachy_wave](https://huggingface.co/datasets/aliberts/reachy_wave) | partial_acquired | 10 | 0.000 | 5 | apache-2.0 |
| [hf__aliberts__reachy_wave_2](https://huggingface.co/datasets/aliberts/reachy_wave_2) | partial_acquired | 55 | 0.003 | 50 | apache-2.0 |
| [hf__CompeteSAI__reachy2-kitchen-multimodal](https://huggingface.co/datasets/CompeteSAI/reachy2-kitchen-multimodal) | partial_acquired | 224 | 0.009 | 219 | apache-2.0 |
| [hf__cadene__reachy2_mobile_base](https://huggingface.co/datasets/cadene/reachy2_mobile_base) | partial_acquired | 5 | 0.003 | 22 | unspecified |
| [hf__cadene__reachy2_mobile_base_2](https://huggingface.co/datasets/cadene/reachy2_mobile_base_2) | partial_acquired | 5 | 0.006 | 19 | unspecified |
| [hf__cadene__reachy2_mobile_grasp_test](https://huggingface.co/datasets/cadene/reachy2_mobile_grasp_test) | partial_acquired | 5 | 0.000 | 3 | unspecified |
| [oakink](https://oakink.net/) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__cadene__reachy2_mobile_grasp_test3](https://huggingface.co/datasets/cadene/reachy2_mobile_grasp_test3) | partial_acquired | 5 | 0.000 | 1 | unspecified |
| [hf__cadene__reachy2_mobile_grasp_test2](https://huggingface.co/datasets/cadene/reachy2_mobile_grasp_test2) | partial_acquired | 5 | 0.000 | 1 | unspecified |
| [oakink2](https://github.com/oakink/OakInk2) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hot3d](https://facebookresearch.github.io/hot3d/) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__cdeplanne__reachy-arcade-scores](https://huggingface.co/datasets/cdeplanne/reachy-arcade-scores) | metadata_only | 3 | 0.000 | 0 | mit |
| [hf__erl-hub__eval_reachy-pick-and-place](https://huggingface.co/datasets/erl-hub/eval_reachy-pick-and-place) | partial_acquired | 8 | 0.000 | 2 | apache-2.0 |
| [hf__erl-hub__reachy-cleaning](https://huggingface.co/datasets/erl-hub/reachy-cleaning) | partial_acquired | 6 | 0.000 | 1 | apache-2.0 |
| [hoi4d](https://github.com/leolyliu/HOI4D-Instructions) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__erl-hub__reachy-cleaning-10Hz](https://huggingface.co/datasets/erl-hub/reachy-cleaning-10Hz) | partial_acquired | 32 | 0.002 | 14 | apache-2.0 |
| [ho3d](https://github.com/shreyashampali/ho3d) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__erl-hub__reachy-pick-and-place](https://huggingface.co/datasets/erl-hub/reachy-pick-and-place) | partial_acquired | 20 | 0.001 | 10 | apache-2.0 |
| [hf__erl-hub__reachy-pick-and-place-images](https://huggingface.co/datasets/erl-hub/reachy-pick-and-place-images) | partial_acquired | 54 | 0.003 | 25 | apache-2.0 |
| [hf__erl-hub__reachy-pick-and-place-large](https://huggingface.co/datasets/erl-hub/reachy-pick-and-place-large) | metadata_only | 1 | 0.000 | 0 | mit |
| [grab](https://grab.is.tue.mpg.de/) | access_required | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__erl-hub__reachy-pick-and-place-right-arm](https://huggingface.co/datasets/erl-hub/reachy-pick-and-place-right-arm) | partial_acquired | 51 | 0.002 | 23 | apache-2.0 |
| [hf__glannuzel__eval_grab_cube_reachy](https://huggingface.co/datasets/glannuzel/eval_grab_cube_reachy) | partial_acquired | 7 | 0.000 | 2 | apache-2.0 |
| [behave](https://virtualhumans.mpi-inf.mpg.de/behave/) | acquired_selected_scope | 2 | 0.355 | 0 | not verified; inspect upstream license before use |
| [intercap](https://intercap.is.tue.mpg.de/) | access_required | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [h2o](https://h2odataset.ethz.ch/) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__glannuzel__reachy2_pick_place](https://huggingface.co/datasets/glannuzel/reachy2_pick_place) | partial_acquired | 20 | 0.002 | 170 | apache-2.0 |
| [hf__haixuantao__reachy2_teleop_remi_eval](https://huggingface.co/datasets/haixuantao/reachy2_teleop_remi_eval) | unsuitable | 0 | 0.000 | 0 | unspecified |
| [droid](https://droid-dataset.github.io/droid/the-droid-dataset) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [arctic](https://arctic.is.tue.mpg.de/) | access_required | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__haixuantao__reachy2_teleop_remi_eval_2_raw](https://huggingface.co/datasets/haixuantao/reachy2_teleop_remi_eval_2_raw) | unsuitable | 0 | 0.000 | 0 | unspecified |
| [hf__haixuantao__reachy2_teleop_remi_eval_3_raw](https://huggingface.co/datasets/haixuantao/reachy2_teleop_remi_eval_3_raw) | unsuitable | 0 | 0.000 | 0 | unspecified |
| [hoi_retarget](https://github.com/leggedrobotics/hoi-retarget) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [bridge](https://rail-berkeley.github.io/bridgedata/) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__haixuantao__reachy2_teleop_remi_eval_4_raw](https://huggingface.co/datasets/haixuantao/reachy2_teleop_remi_eval_4_raw) | unsuitable | 0 | 0.000 | 0 | unspecified |
| [roboturk](https://roboturk.stanford.edu/dataset_real.html) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [aloha](https://github.com/tonyzhaozh/act) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [mobile_aloha](https://github.com/MarkFzp/mobile-aloha) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__apple_storage](https://huggingface.co/datasets/pollen-robotics/apple_storage) | partial_acquired | 55 | 0.003 | 50 | apache-2.0 |
| [umi](https://github.com/real-stanford/universal_manipulation_interface) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__apple_storage_2_modified](https://huggingface.co/datasets/pollen-robotics/apple_storage_2_modified) | partial_acquired | 57 | 0.003 | 52 | apache-2.0 |
| [maniskill](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [robomimic](https://github.com/ARISE-Initiative/robomimic) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__apple_storage_2](https://huggingface.co/datasets/pollen-robotics/apple_storage_2) | partial_acquired | 59 | 0.003 | 2 | apache-2.0 |
| [mimicgen](https://github.com/NVlabs/mimicgen) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [robocasa](https://github.com/robocasa/robocasa) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__eval_pick_and_place_bottle](https://huggingface.co/datasets/pollen-robotics/eval_pick_and_place_bottle) | partial_acquired | 20 | 0.000 | 15 | apache-2.0 |
| [libero](https://github.com/Lifelong-Robot-Learning/LIBERO) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__grasp_apple](https://huggingface.co/datasets/pollen-robotics/grasp_apple) | partial_acquired | 5 | 0.001 | 52 | unspecified |
| [calvin](https://github.com/mees/calvin) | acquired_selected_scope | 2 | 0.004 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__grasp_mug](https://huggingface.co/datasets/pollen-robotics/grasp_mug) | partial_acquired | 5 | 0.001 | 52 | unspecified |
| [d4rl_kitchen](https://github.com/Farama-Foundation/D4RL) | acquired_selected_scope | 3 | 0.040 | 1226 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__reachy-doing-things](https://huggingface.co/datasets/pollen-robotics/reachy-doing-things) | metadata_only | 1 | 0.000 | 0 | apache-2.0 |
| [furniturebench](https://github.com/clvrai/furniture-bench) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__pollen-robotics__pick_and_place_bottle](https://huggingface.co/datasets/pollen-robotics/pick_and_place_bottle) | partial_acquired | 91 | 0.002 | 86 | apache-2.0 |
| [hf__pollen-robotics__reachy2_static_cup](https://huggingface.co/datasets/pollen-robotics/reachy2_static_cup) | partial_acquired | 5 | 0.002 | 50 | unspecified |
| [hf__pollen-robotics__reachy2_mobile_household_apple](https://huggingface.co/datasets/pollen-robotics/reachy2_mobile_household_apple) | partial_acquired | 5 | 0.007 | 13 | unspecified |
| [hf__simheo__eval_act_reachy2_torso_cleaned](https://huggingface.co/datasets/simheo/eval_act_reachy2_torso_cleaned) | partial_acquired | 6 | 0.000 | 1 | apache-2.0 |
| [amass](https://amass.is.tue.mpg.de/) | access_required | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__simheo__reachy2_pick_place_larger](https://huggingface.co/datasets/simheo/reachy2_pick_place_larger) | partial_acquired | 26 | 0.004 | 88 | apache-2.0 |
| [hf__simheo__reachy2_pick_place_cleaned](https://huggingface.co/datasets/simheo/reachy2_pick_place_cleaned) | partial_acquired | 19 | 0.004 | 78 | unspecified |
| [hf__robomimic__robomimic_datasets](https://huggingface.co/datasets/robomimic/robomimic_datasets) | partial_acquired | 27 | 6.557 | 7650 | mit |
| [hf__kelvin34501__OakInk-v2](https://huggingface.co/datasets/kelvin34501/OakInk-v2) | partial_acquired | 632 | 36.992 | 0 | cc-by-sa-4.0 |
| [hf__leggedrobotics__hoi-retarget](https://huggingface.co/datasets/leggedrobotics/hoi-retarget) | partial_acquired | 62 | 0.235 | 0 | cc-by-nc-sa-4.0 |
| [cmu](https://mocap.cs.cmu.edu/) | acquired_selected_scope | 1 | 1.084 | 0 | not verified; inspect upstream license before use |
| [humoto](https://github.com/adobe-research/humoto) | partial_acquired | 144 | 0.442 | 70 | not verified; inspect upstream license before use |
| [parahome](https://github.com/snuvclab/ParaHome) | acquired_selected_scope | 4 | 2.369 | 207 | CC-BY-NC-SA-4.0 |
| [oxe](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__columbia_cairlab_pusht_real](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__stanford_kuka_multimodal_dataset_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__nyu_rot_dataset_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__cmu_franka_exploration_dataset_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__ucsd_pick_and_place_dataset_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__usc_cloth_sim_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__utokyo_pr2_opening_fridge_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__utokyo_saytap_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__utokyo_xarm_pick_and_place_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__utokyo_xarm_bimanual_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__kaist_nonprehensile_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__tokyo_u_lsmo_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__dlr_sara_pour_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__dlr_sara_grid_clamp_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__asu_table_top_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__stanford_robocook_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__imperialcollege_sawyer_wrist_cam](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__uiuc_d3field](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__berkeley_gnm_recon](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__berkeley_gnm_cory_hall](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__berkeley_gnm_sac_son](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__conq_hose_manipulation](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__fmb](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__lerobot__jaco_play](https://huggingface.co/datasets/lerobot/jaco_play) | partial_acquired | 6 | 0.004 | 1085 | cc-by-4.0 |
| [hf__lerobot__roboturk](https://huggingface.co/datasets/lerobot/roboturk) | partial_acquired | 6 | 0.009 | 1995 | apache-2.0 |
| [hf__lerobot__taco_play](https://huggingface.co/datasets/lerobot/taco_play) | partial_acquired | 6 | 0.017 | 3603 | cc-by-4.0 |
| [hf__lerobot__nyu_door_opening_surprising_effectiveness](https://huggingface.co/datasets/lerobot/nyu_door_opening_surprising_effectiveness) | partial_acquired | 6 | 0.001 | 484 | mit |
| [hf__lerobot__toto](https://huggingface.co/datasets/lerobot/toto) | partial_acquired | 6 | 0.022 | 1003 | mit |
| [hf__lerobot__berkeley_autolab_ur5](https://huggingface.co/datasets/lerobot/berkeley_autolab_ur5) | partial_acquired | 6 | 0.006 | 1000 | cc-by-4.0 |
| [hf__lerobot__columbia_cairlab_pusht_real](https://huggingface.co/datasets/lerobot/columbia_cairlab_pusht_real) | partial_acquired | 6 | 0.001 | 136 | mit |
| [hf__lerobot__stanford_kuka_multimodal_dataset](https://huggingface.co/datasets/lerobot/stanford_kuka_multimodal_dataset) | partial_acquired | 6 | 0.010 | 3000 | mit |
| [hf__lerobot__stanford_hydra_dataset](https://huggingface.co/datasets/lerobot/stanford_hydra_dataset) | partial_acquired | 6 | 0.023 | 570 | mit |
| [hf__lerobot__nyu_rot_dataset](https://huggingface.co/datasets/lerobot/nyu_rot_dataset) | partial_acquired | 6 | 0.000 | 14 | mit |
| [hf__lerobot__austin_buds_dataset](https://huggingface.co/datasets/lerobot/austin_buds_dataset) | partial_acquired | 6 | 0.004 | 50 | mit |
| [hf__lerobot__berkeley_cable_routing](https://huggingface.co/datasets/lerobot/berkeley_cable_routing) | partial_acquired | 1006 | 0.012 | 1647 | apache-2.0 |
| [hf__lerobot__nyu_franka_play_dataset](https://huggingface.co/datasets/lerobot/nyu_franka_play_dataset) | partial_acquired | 6 | 0.007 | 456 | mit |
| [hf__lerobot__cmu_franka_exploration_dataset](https://huggingface.co/datasets/lerobot/cmu_franka_exploration_dataset) | partial_acquired | 6 | 0.000 | 199 | mit |
| [hf__lerobot__ucsd_kitchen_dataset](https://huggingface.co/datasets/lerobot/ucsd_kitchen_dataset) | partial_acquired | 6 | 0.001 | 149 | mit |
| [hf__lerobot__ucsd_pick_and_place_dataset](https://huggingface.co/datasets/lerobot/ucsd_pick_and_place_dataset) | partial_acquired | 6 | 0.004 | 1355 | mit |
| [hf__lerobot__austin_sailor_dataset](https://huggingface.co/datasets/lerobot/austin_sailor_dataset) | partial_acquired | 6 | 0.017 | 240 | mit |
| [hf__lerobot__austin_sirius_dataset](https://huggingface.co/datasets/lerobot/austin_sirius_dataset) | partial_acquired | 6 | 0.016 | 559 | mit |
| [hf__lerobot__usc_cloth_sim](https://huggingface.co/datasets/lerobot/usc_cloth_sim) | partial_acquired | 6 | 0.001 | 1 | apache-2.0 |
| [hf__lerobot__utokyo_pr2_opening_fridge](https://huggingface.co/datasets/lerobot/utokyo_pr2_opening_fridge) | partial_acquired | 6 | 0.001 | 80 | mit |
| [hf__lerobot__utokyo_pr2_tabletop_manipulation](https://huggingface.co/datasets/lerobot/utokyo_pr2_tabletop_manipulation) | partial_acquired | 6 | 0.002 | 240 | mit |
| [hf__lerobot__utokyo_saytap](https://huggingface.co/datasets/lerobot/utokyo_saytap) | partial_acquired | 6 | 0.005 | 20 | mit |
| [hf__lerobot__utokyo_xarm_pick_and_place](https://huggingface.co/datasets/lerobot/utokyo_xarm_pick_and_place) | partial_acquired | 6 | 0.001 | 102 | apache-2.0 |
| [hf__lerobot__utokyo_xarm_bimanual](https://huggingface.co/datasets/lerobot/utokyo_xarm_bimanual) | partial_acquired | 6 | 0.000 | 70 | cc-by-4.0 |
| [hf__lerobot__berkeley_mvp](https://huggingface.co/datasets/lerobot/berkeley_mvp) | partial_acquired | 6 | 0.004 | 480 | mit |
| [hf__lerobot__berkeley_rpt](https://huggingface.co/datasets/lerobot/berkeley_rpt) | partial_acquired | 6 | 0.026 | 908 | mit |
| [hf__lerobot__tokyo_u_lsmo](https://huggingface.co/datasets/lerobot/tokyo_u_lsmo) | partial_acquired | 6 | 0.002 | 50 | mit |
| [hf__lerobot__dlr_sara_pour](https://huggingface.co/datasets/lerobot/dlr_sara_pour) | partial_acquired | 6 | 0.001 | 100 | mit |
| [hf__lerobot__kaist_nonprehensile](https://huggingface.co/datasets/lerobot/kaist_nonprehensile) | partial_acquired | 6 | 0.003 | 201 | cc-by-4.0 |
| [hf__lerobot__dlr_sara_grid_clamp](https://huggingface.co/datasets/lerobot/dlr_sara_grid_clamp) | partial_acquired | 6 | 0.001 | 107 | mit |
| [hf__lerobot__dlr_edan_shared_control](https://huggingface.co/datasets/lerobot/dlr_edan_shared_control) | partial_acquired | 6 | 0.001 | 104 | mit |
| [hf__lerobot__asu_table_top](https://huggingface.co/datasets/lerobot/asu_table_top) | partial_acquired | 6 | 0.003 | 110 | mit |
| [hf__lerobot__stanford_robocook](https://huggingface.co/datasets/lerobot/stanford_robocook) | partial_acquired | 6 | 0.010 | 2460 | mit |
| [hf__lerobot__iamlab_cmu_pickup_insert](https://huggingface.co/datasets/lerobot/iamlab_cmu_pickup_insert) | partial_acquired | 6 | 0.011 | 631 | apache-2.0 |
| [hf__lerobot__utaustin_mutex](https://huggingface.co/datasets/lerobot/utaustin_mutex) | partial_acquired | 6 | 0.018 | 1500 | mit |
| [hf__lerobot__berkeley_fanuc_manipulation](https://huggingface.co/datasets/lerobot/berkeley_fanuc_manipulation) | partial_acquired | 6 | 0.003 | 415 | mit |
| [hf__lerobot__imperialcollege_sawyer_wrist_cam](https://huggingface.co/datasets/lerobot/imperialcollege_sawyer_wrist_cam) | partial_acquired | 6 | 0.001 | 170 | apache-2.0 |
| [hf__lerobot__uiuc_d3field](https://huggingface.co/datasets/lerobot/uiuc_d3field) | partial_acquired | 6 | 0.001 | 183 | apache-2.0 |
| [hf__lerobot__cmu_stretch](https://huggingface.co/datasets/lerobot/cmu_stretch) | partial_acquired | 6 | 0.001 | 135 | mit |
| [hf__lerobot__cmu_play_fusion](https://huggingface.co/datasets/lerobot/cmu_play_fusion) | partial_acquired | 6 | 0.015 | 576 | mit |
| [hf__lerobot__berkeley_gnm_cory_hall](https://huggingface.co/datasets/lerobot/berkeley_gnm_cory_hall) | partial_acquired | 6 | 0.014 | 7273 | apache-2.0 |
| [hf__lerobot__berkeley_gnm_sac_son](https://huggingface.co/datasets/lerobot/berkeley_gnm_sac_son) | partial_acquired | 6 | 0.012 | 2955 | apache-2.0 |
| [hf__lerobot__conq_hose_manipulation](https://huggingface.co/datasets/lerobot/conq_hose_manipulation) | partial_acquired | 6 | 0.004 | 139 | apache-2.0 |
| [hf__lerobot__fmb](https://huggingface.co/datasets/lerobot/fmb) | partial_acquired | 6 | 0.020 | 1804 | apache-2.0 |
| [hf__lerobot__berkeley_gnm_recon](https://huggingface.co/datasets/lerobot/berkeley_gnm_recon) | partial_acquired | 1006 | 0.042 | 11834 | apache-2.0 |
| [hf__lerobot__aloha_sim_transfer_cube_human](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human) | partial_acquired | 8 | 0.002 | 50 | mit |
| [hf__lerobot__aloha_sim_insertion_scripted](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_scripted) | partial_acquired | 8 | 0.004 | 50 | mit |
| [hf__lerobot__umi_cup_in_the_wild](https://huggingface.co/datasets/lerobot/umi_cup_in_the_wild) | partial_acquired | 6 | 0.030 | 1447 | apache-2.0 |
| [hf__lerobot__aloha_sim_transfer_cube_scripted](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_scripted) | partial_acquired | 6 | 0.003 | 50 | apache-2.0 |
| [hf__lerobot__libero](https://huggingface.co/datasets/lerobot/libero) | partial_acquired | 382 | 0.020 | 1693 | apache-2.0 |
| [hf__lerobot__aloha_sim_insertion_human](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human) | partial_acquired | 9 | 0.003 | 50 | mit |
| [hf__lerobot__aloha_static_candy](https://huggingface.co/datasets/lerobot/aloha_static_candy) | partial_acquired | 6 | 0.002 | 50 | mit |
| [hf__lerobot__aloha_static_tape](https://huggingface.co/datasets/lerobot/aloha_static_tape) | partial_acquired | 6 | 0.002 | 50 | apache-2.0 |
| [hf__lerobot__aloha_static_screw_driver](https://huggingface.co/datasets/lerobot/aloha_static_screw_driver) | partial_acquired | 6 | 0.001 | 50 | mit |
| [hf__lerobot__aloha_mobile_wipe_wine](https://huggingface.co/datasets/lerobot/aloha_mobile_wipe_wine) | partial_acquired | 6 | 0.004 | 50 | mit |
| [hf__lerobot__aloha_static_towel](https://huggingface.co/datasets/lerobot/aloha_static_towel) | partial_acquired | 8 | 0.002 | 50 | apache-2.0 |
| [hf__lerobot__aloha_static_coffee](https://huggingface.co/datasets/lerobot/aloha_static_coffee) | partial_acquired | 6 | 0.003 | 50 | mit |
| [hf__lerobot__aloha_static_battery](https://huggingface.co/datasets/lerobot/aloha_static_battery) | partial_acquired | 6 | 0.001 | 49 | apache-2.0 |
| [hf__lerobot__aloha_static_thread_velcro](https://huggingface.co/datasets/lerobot/aloha_static_thread_velcro) | partial_acquired | 6 | 0.002 | 0 | apache-2.0 |
| [hf__lerobot__aloha_static_vinh_cup_left](https://huggingface.co/datasets/lerobot/aloha_static_vinh_cup_left) | partial_acquired | 6 | 0.003 | 100 | mit |
| [hf__lerobot__aloha_static_vinh_cup](https://huggingface.co/datasets/lerobot/aloha_static_vinh_cup) | partial_acquired | 6 | 0.003 | 101 | mit |
| [hf__lerobot__aloha_static_coffee_new](https://huggingface.co/datasets/lerobot/aloha_static_coffee_new) | partial_acquired | 6 | 0.005 | 50 | mit |
| [hf__lerobot__aloha_static_ziploc_slide](https://huggingface.co/datasets/lerobot/aloha_static_ziploc_slide) | partial_acquired | 6 | 0.001 | 56 | mit |
| [hf__lerobot__aloha_static_pingpong_test](https://huggingface.co/datasets/lerobot/aloha_static_pingpong_test) | partial_acquired | 6 | 0.001 | 10 | mit |
| [hf__lerobot__aloha_static_cups_open](https://huggingface.co/datasets/lerobot/aloha_static_cups_open) | partial_acquired | 6 | 0.001 | 50 | mit |
| [hf__lerobot__aloha_static_fork_pick_up](https://huggingface.co/datasets/lerobot/aloha_static_fork_pick_up) | partial_acquired | 6 | 0.004 | 100 | apache-2.0 |
| [hf__lerobot__aloha_static_pro_pencil](https://huggingface.co/datasets/lerobot/aloha_static_pro_pencil) | partial_acquired | 6 | 0.001 | 25 | mit |
| [hf__lerobot__aloha_mobile_shrimp](https://huggingface.co/datasets/lerobot/aloha_mobile_shrimp) | partial_acquired | 6 | 0.004 | 18 | apache-2.0 |
| [hf__lerobot__aloha_mobile_wash_pan](https://huggingface.co/datasets/lerobot/aloha_mobile_wash_pan) | partial_acquired | 6 | 0.003 | 50 | mit |
| [hf__lerobot__aloha_mobile_cabinet](https://huggingface.co/datasets/lerobot/aloha_mobile_cabinet) | partial_acquired | 6 | 0.009 | 85 | apache-2.0 |
| [hf__lerobot__aloha_mobile_chair](https://huggingface.co/datasets/lerobot/aloha_mobile_chair) | partial_acquired | 6 | 0.008 | 55 | mit |
| [hf__lerobot__aloha_sim_insertion_scripted_image](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_scripted_image) | not_acquired | 0 | 0.000 | 0 | mit |
| [hf__lerobot__aloha_mobile_elevator](https://huggingface.co/datasets/lerobot/aloha_mobile_elevator) | partial_acquired | 6 | 0.003 | 20 | mit |
| [hf__lerobot__aloha_sim_transfer_cube_human_image](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human_image) | not_acquired | 0 | 0.000 | 0 | mit |
| [hf__lerobot__aloha_sim_transfer_cube_scripted_image](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_scripted_image) | not_acquired | 0 | 0.000 | 0 | mit |
| [hf__lerobot__aloha_sim_insertion_human_image](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human_image) | not_acquired | 0 | 0.000 | 0 | mit |
| [hf__lerobot__droid_100](https://huggingface.co/datasets/lerobot/droid_100) | partial_acquired | 6 | 0.003 | 100 | mit |
| [hf__lerobot__metaworld_mt50_push_v2_image](https://huggingface.co/datasets/lerobot/metaworld_mt50_push_v2_image) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__lerobot__metaworld_mt50](https://huggingface.co/datasets/lerobot/metaworld_mt50) | partial_acquired | 287 | 0.016 | 1263 | apache-2.0 |
| [hf__lerobot__libero_spatial_image](https://huggingface.co/datasets/lerobot/libero_spatial_image) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__lerobot__libero_goal_image](https://huggingface.co/datasets/lerobot/libero_goal_image) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__lerobot__libero_10_image](https://huggingface.co/datasets/lerobot/libero_10_image) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__lerobot__aloha_mobile_shrimp_image](https://huggingface.co/datasets/lerobot/aloha_mobile_shrimp_image) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__lerobot__droid_1.0.1](https://huggingface.co/datasets/lerobot/droid_1.0.1) | partial_acquired | 167 | 12.645 | 95634 | apache-2.0 |
| [hf__lerobot__libero-assets](https://huggingface.co/datasets/lerobot/libero-assets) | unsuitable | 0 | 0.000 | 0 | unspecified |
| [hf__lerobot__libero_object_image](https://huggingface.co/datasets/lerobot/libero_object_image) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__lerobot__libero_10](https://huggingface.co/datasets/lerobot/libero_10) | partial_acquired | 152 | 0.008 | 0 | apache-2.0 |
| [hf__lerobot__libero_plus](https://huggingface.co/datasets/lerobot/libero_plus) | partial_acquired | 6 | 0.121 | 1438 | unspecified |
| [hf__lerobot__libero_10_image_subtask](https://huggingface.co/datasets/lerobot/libero_10_image_subtask) | not_acquired | 0 | 0.000 | 0 | unspecified |
| [hf__lerobot__robocasa_target_human_unified](https://huggingface.co/datasets/lerobot/robocasa_target_human_unified) | partial_acquired | 25 | 1.076 | 25307 | apache-2.0 |
| [hf__lerobot__libero_10_subtask](https://huggingface.co/datasets/lerobot/libero_10_subtask) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [robosuite_can_assets](https://github.com/ARISE-Initiative/robosuite/tree/v1.5.1) | metadata_only | 3 | 0.000 | 0 | MIT |
| [lafan1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) | acquired_selected_scope | 3 | 0.501 | 77 | CC-BY-NC-ND-4.0 |
| [hf__oakink__OakInk-v1](https://huggingface.co/datasets/oakink/OakInk-v1) | partial_acquired | 7 | 6.436 | 0 | cc-by-nc-sa-3.0 |
| [hf__JuzeZhang__HOI-M3](https://huggingface.co/datasets/JuzeZhang/HOI-M3) | unsuitable | 0 | 0.000 | 0 | unspecified |
| [hf__haosulab__ManiSkill_Demonstrations](https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations) | acquired_selected_scope | 73 | 4.667 | 32971 | apache-2.0 |
| [hf__amandlek__mimicgen_datasets](https://huggingface.co/datasets/amandlek/mimicgen_datasets) | partial_acquired | 19 | 1.266 | 6110 | cc-by-4.0 |
| [hf__MIT-Media-Lab__h2o-annotations-v1](https://huggingface.co/datasets/MIT-Media-Lab/h2o-annotations-v1) | acquired_selected_scope | 8 | 0.432 | 0 | cc-by-nc-4.0 |
| [hf__IPEC-COMMUNITY__FastUMI-Data](https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI-Data) | not_acquired | 0 | 0.000 | 0 | mit |
| [intermimic](https://github.com/Sirui-Xu/InterMimic) | acquired_selected_scope | 2 | 1.682 | 0 | not verified; inspect upstream license before use |
| [hf__MIT-Media-Lab__hot3d-annotations-v1](https://huggingface.co/datasets/MIT-Media-Lab/hot3d-annotations-v1) | partial_acquired | 7 | 1.629 | 0 | cc-by-nc-4.0 |
| [hf__JingkunAn__lerobot_hoi4d_dataset](https://huggingface.co/datasets/JingkunAn/lerobot_hoi4d_dataset) | partial_acquired | 298 | 1.024 | 294 | not verified; inspect upstream license before use |
| [hf__Ronaldo-GOAT__ho3d_transfer](https://huggingface.co/datasets/Ronaldo-GOAT/ho3d_transfer) | partial_acquired | 3 | 1.288 | 0 | other |
| [hf__IPEC-COMMUNITY__furniture_bench_dataset_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/furniture_bench_dataset_lerobot) | partial_acquired | 5104 | 0.336 | 5100 | apache-2.0 |
| [hf__bop-benchmark__hot3d](https://huggingface.co/datasets/bop-benchmark/hot3d) | acquired_selected_scope | 51 | 0.413 | 0 | unspecified |
| [hf__IPEC-COMMUNITY__FastUMI_100k_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI_100k_lerobot) | partial_acquired | 19709 | 1.362 | 19706 | not verified; inspect upstream license before use |
| [taco](https://github.com/leolyliu/TACO-Instructions) | acquired_selected_scope | 6 | 15.717 | 2317 | CC-BY-4.0 |
| [interact](https://github.com/wzyabcas/InterAct) | access_required | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [chairs](https://jnnan.github.io/chairs/) | access_required | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [fastumi](https://github.com/YdingTeam/FastUMI) | not_acquired | 0 | 0.000 | 0 | see each public release license |
| [hoim3](https://drive.google.com/drive/folders/1pXUHX0Y9Q6IVKNxjzz7iuRoFPCGociOF) | partial_acquired | 278 | 7.592 | 0 | Original dataset LICENSE.md pending inspection; code license not substituted for data license |
| [hodome](https://github.com/Juzezhang/NeuralDome_Toolbox) | partial_acquired | 187 | 1.583 | 0 | Custom non-commercial scientific research license; no redistribution; see raw/hodome/LICENSE.md |
| [hf__IPEC-COMMUNITY__fractal20220817_data_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [oxe__aloha_mobile](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.000 | 2 | not verified; inspect upstream license before use |
| [oxe__bridge](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.000 | 7 | not verified; inspect upstream license before use |
| [oxe__cmu_playing_with_food](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.000 | 4 | not verified; inspect upstream license before use |
| [oxe__eth_agent_affordances](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.001 | 2 | not verified; inspect upstream license before use |
| [oxe__fractal20220817_data](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.009 | 80 | not verified; inspect upstream license before use |
| [oxe__io_ai_tech](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.001 | 7 | not verified; inspect upstream license before use |
| [oxe__maniskill_dataset_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.004 | 24 | not verified; inspect upstream license before use |
| [oxe__mimic_play](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.003 | 9 | not verified; inspect upstream license before use |
| [oxe__plex_robosuite](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.003 | 25 | not verified; inspect upstream license before use |
| [oxe__robo_net](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.003 | 68 | not verified; inspect upstream license before use |
| [oxe__robo_set](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.001 | 14 | not verified; inspect upstream license before use |
| [oxe__robot_vqa](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.004 | 0 | not verified; inspect upstream license before use |
| [oxe__spoc](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.019 | 247 | not verified; inspect upstream license before use |
| [oxe__stanford_mask_vit_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.001 | 9 | not verified; inspect upstream license before use |
| [oxe__tidybot](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.001 | 12 | not verified; inspect upstream license before use |
| [oxe__vima_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.020 | 323 | not verified; inspect upstream license before use |
| [hf__IPEC-COMMUNITY__jaco_play_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/jaco_play_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__roboturk_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/roboturk_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__nyu_door_opening_surprising_effectiveness_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/nyu_door_opening_surprising_effectiveness_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__toto_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/toto_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__berkeley_autolab_ur5_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/berkeley_autolab_ur5_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__berkeley_cable_routing_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/berkeley_cable_routing_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__austin_buds_dataset_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/austin_buds_dataset_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__stanford_hydra_dataset_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/stanford_hydra_dataset_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__ucsd_kitchen_dataset_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/ucsd_kitchen_dataset_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__taco_play_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/taco_play_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__nyu_franka_play_dataset_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/nyu_franka_play_dataset_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__austin_sailor_dataset_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/austin_sailor_dataset_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__berkeley_mvp_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/berkeley_mvp_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__dlr_edan_shared_control_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/dlr_edan_shared_control_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__austin_sirius_dataset_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/austin_sirius_dataset_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__berkeley_rpt_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/berkeley_rpt_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__iamlab_cmu_pickup_insert_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/iamlab_cmu_pickup_insert_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__berkeley_fanuc_manipulation_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/berkeley_fanuc_manipulation_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__cmu_stretch_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/cmu_stretch_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__cmu_play_fusion_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/cmu_play_fusion_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__utaustin_mutex_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/utaustin_mutex_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__dobbe_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/dobbe_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__viola_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/viola_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__bc_z_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/bc_z_lerobot) | not_acquired | 0 | 0.000 | 0 | apache-2.0 |
| [hf__IPEC-COMMUNITY__kuka_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/kuka_lerobot) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__kuka](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.024 | 577 | not verified; inspect upstream license before use |
| [oxe__taco_play](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__jaco_play](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__berkeley_cable_routing](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__roboturk](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__nyu_door_opening_surprising_effectiveness](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__viola](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.000 | 2 | not verified; inspect upstream license before use |
| [oxe__berkeley_autolab_ur5](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__toto](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__IPEC-COMMUNITY__droid_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/droid_lerobot) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__stanford_hydra_dataset_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__austin_buds_dataset_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__nyu_franka_play_dataset_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__furniture_bench_dataset_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.001 | 3 | not verified; inspect upstream license before use |
| [oxe__ucsd_kitchen_dataset_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__austin_sailor_dataset_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__austin_sirius_dataset_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__bc_z](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.003 | 29 | not verified; inspect upstream license before use |
| [oxe__berkeley_mvp_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__berkeley_rpt_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__dlr_edan_shared_control_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__iamlab_cmu_pickup_insert_converted_externally_to_rlds](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__utaustin_mutex](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__berkeley_fanuc_manipulation](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__cmu_play_fusion](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__cmu_stretch](https://github.com/google-deepmind/open_x_embodiment) | not_acquired | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [oxe__droid](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.006 | 40 | not verified; inspect upstream license before use |
| [oxe__dobbe](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.003 | 25 | not verified; inspect upstream license before use |
| [oxe__qut_dexterous_manipulation](https://github.com/google-deepmind/open_x_embodiment) | partial_acquired | 3 | 0.000 | 1 | not verified; inspect upstream license before use |
| [oxe__language_table](https://github.com/google-deepmind/open_x_embodiment) | metadata_only | 2 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__IPEC-COMMUNITY__language_table_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/language_table_lerobot) | unsuitable | 0 | 0.000 | 0 | not verified; inspect upstream license before use |
| [hf__IPEC-COMMUNITY__bridge_orig_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) | partial_acquired | 2047 | 0.013 | 2047 | apache-2.0 |
| [dexycb](https://dex-ycb.github.io/) | partial_acquired | 3 | 2.706 | 0 | CC-BY-NC-4.0 |
| [hf__QFun__MANUS-DexYCB](https://huggingface.co/datasets/QFun/MANUS-DexYCB) | partial_acquired | 19 | 0.343 | 0 | CC-BY-NC-4.0 (source license ID preserved inside sample JSON) |
| [omomo](https://github.com/lijiaman/omomo_release) | acquired_selected_scope | 1 | 22.177 | 0 | not verified; inspect upstream license before use |

## Reproduce

See [README.md](README.md) for commands, environment pinning, schema, sampling/split policy, loader examples and acquisition restart. `validate` checks the full acquired ledger, representative schemas, checksums and existing tracking compatibility. Source-specific coordinate conventions are recorded only when verified; unverified raw schemas are not silently mapped into Reachy coordinates.
