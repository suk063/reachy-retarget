"""RLBench object table of the RoboVerse migration (generated, do not edit by hand).

Extracted from ``roboverse_pack/tasks/rlbench/*.py`` of github.com/RoboVerseOrg/RoboVerse at
commit ``5f3ec0185d2d3bcb59d53b0bd1f5b0f8a6f2ce14`` by executing each task module with the
MetaSim config classes stubbed out. Per task: object name -> (RoboVerse physics type, geometry).

* physics: ``RIGIDBODY`` (dynamic), ``GEOM`` (static collider), ``XFORM`` (visual only, no
  collision) or ``ARTICULATION`` (``ArticulationObjCfg``, fixed base, jointed).
* geometry: ``("box", half_extents)``, ``("sphere", radius)``, ``("cylinder", radius, height)``
  for MetaSim primitives (``PrimitiveCubeCfg.size`` is the full edge length, halved here), or
  ``("usd", path)`` for converted RLBench meshes (path relative to ``roboverse_data``).

Primitive objects have no mass or friction in their configs: MetaSim's ``_PrimitiveMixin``
default mass is 0.1 kg; friction is left to the simulator default.
"""
ROBOVERSE_COMMIT = "5f3ec0185d2d3bcb59d53b0bd1f5b0f8a6f2ce14"
PRIMITIVE_DEFAULT_MASS_KG = 0.1

RLBENCH_OBJECTS = {
    'basketball_in_hoop': {  # basketball_in_hoop.py
        'basket_ball_hoop_visual': ('XFORM', ('usd', 'assets/rlbench/basketball_in_hoop/basket_ball_hoop_visual/usd/basket_ball_hoop_visual_colored.usd')),
        'ball': ('RIGIDBODY', ('usd', 'assets/rlbench/basketball_in_hoop/ball/usd/ball_textured.usd')),
    },
    'beat_the_buzz': {  # beat_the_buzz.py
        'wand': ('RIGIDBODY', ('usd', 'assets/rlbench/beat_the_buzz/wand/usd/wand.usd')),
        'Cuboid': ('GEOM', ('usd', 'assets/rlbench/beat_the_buzz/Cuboid/usd/Cuboid.usd')),
    },
    'block_pyramid': {  # block_pyramid.py
        'block_pyramid_plane': ('GEOM', ('usd', 'assets/rlbench/block_pyramid/block_pyramid_plane/usd/block_pyramid_plane.usd')),
        'block_pyramid_block0': ('RIGIDBODY', ('box', [0.01875, 0.01875, 0.01875])),
        'block_pyramid_block1': ('RIGIDBODY', ('box', [0.01875, 0.01875, 0.01875])),
        'block_pyramid_block2': ('RIGIDBODY', ('box', [0.01875, 0.01875, 0.01875])),
        'block_pyramid_block3': ('RIGIDBODY', ('box', [0.01875, 0.01875, 0.01875])),
        'block_pyramid_block4': ('RIGIDBODY', ('box', [0.01875, 0.01875, 0.01875])),
        'block_pyramid_block5': ('RIGIDBODY', ('box', [0.01875, 0.01875, 0.01875])),
        'block_pyramid_distractor_block0': ('RIGIDBODY', ('box', [0.01875, 0.01875, 0.01875])),
        'block_pyramid_distractor_block1': ('RIGIDBODY', ('box', [0.01875, 0.01875, 0.01875])),
        'block_pyramid_distractor_block2': ('RIGIDBODY', ('box', [0.01875, 0.01875, 0.01875])),
        'block_pyramid_distractor_block3': ('RIGIDBODY', ('box', [0.01875, 0.01875, 0.01875])),
        'block_pyramid_distractor_block4': ('RIGIDBODY', ('box', [0.01875, 0.01875, 0.01875])),
        'block_pyramid_distractor_block5': ('RIGIDBODY', ('box', [0.01875, 0.01875, 0.01875])),
    },
    'change_channel': {  # change_channel.py
        'tv_remote': ('ARTICULATION', ('usd', 'assets/rlbench/change_channel/tv_remote/usd/tv_remote.usd')),
        'tv_frame': ('RIGIDBODY', ('usd', 'assets/rlbench/change_channel/tv_frame/usd/tv_frame.usd')),
    },
    'change_clock': {  # change_clock.py
        'clock': ('ARTICULATION', ('usd', 'assets/rlbench/change_clock/clock/usd/clock.usd')),
    },
    'close_box': {  # close_box.py
        'box_base': ('ARTICULATION', ('usd', 'assets/rlbench/close_box/box_base/usd/box_base.usd')),
    },
    'close_door': {  # close_door.py
        'door_frame': ('ARTICULATION', ('usd', 'assets/rlbench/close_door/door_frame/usd/door_frame.usd')),
    },
    'close_drawer': {  # close_drawer.py
        'drawer_frame': ('ARTICULATION', ('usd', 'assets/rlbench/close_drawer/drawer_frame/usd/drawer_frame.usd')),
    },
    'close_fridge': {  # close_fridge.py
        'fridge_base': ('ARTICULATION', ('usd', 'assets/rlbench/close_fridge/fridge_base/usd/fridge_base.usd')),
    },
    'close_grill': {  # close_grill.py
        'grill': ('ARTICULATION', ('usd', 'assets/rlbench/close_grill/grill/usd/grill.usd')),
    },
    'close_jar': {  # close_jar.py
        'jar0': ('RIGIDBODY', ('usd', 'assets/rlbench/close_jar/jar0/usd/jar0.usd')),
        'jar1': ('RIGIDBODY', ('usd', 'assets/rlbench/close_jar/jar1/usd/jar1.usd')),
        'jar_lid0': ('RIGIDBODY', ('usd', 'assets/rlbench/close_jar/jar_lid0/usd/jar_lid0.usd')),
    },
    'close_laptop_lid': {  # close_laptop_lid.py
        'base': ('ARTICULATION', ('usd', 'assets/rlbench/close_laptop_lid/base/usd/base.usd')),
        'laptop_holder': ('GEOM', ('usd', 'assets/rlbench/close_laptop_lid/laptop_holder/usd/laptop_holder.usd')),
    },
    'close_microwave': {  # close_microwave.py
        'microwave_frame_resp': ('ARTICULATION', ('usd', 'assets/rlbench/close_microwave/microwave_frame_resp/usd/microwave_frame_resp.usd')),
    },
    'empty_dishwasher': {  # empty_dishwasher.py
        'dishwasher': ('ARTICULATION', ('usd', 'assets/rlbench/empty_dishwasher/dishwasher/usd/dishwasher.usd')),
        'dishwasher_plate_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/empty_dishwasher/dishwasher_plate_visual/usd/dishwasher_plate_visual.usd')),
    },
    'get_ice_from_fridge': {  # get_ice_from_fridge.py
        'fridge_base': ('ARTICULATION', ('usd', 'assets/rlbench/get_ice_from_fridge/fridge_base/usd/fridge_base.usd')),
        'cup_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/get_ice_from_fridge/cup_visual/usd/cup_visual.usd')),
    },
    'hit_ball_with_queue': {  # hit_ball_with_queue.py
        'queue': ('RIGIDBODY', ('usd', 'assets/rlbench/hit_ball_with_queue/queue/usd/queue.usd')),
        'ball': ('RIGIDBODY', ('sphere', 0.01)),
        'hit_ball_with_queue_pocket': ('GEOM', ('box', [0.00375, 0.04, 0.0025])),
        'hit_ball_with_queue_pocket0': ('GEOM', ('box', [0.00375, 0.04, 0.0025])),
        'hit_ball_with_queue_pocket1': ('GEOM', ('box', [0.00375, 0.02, 0.0025])),
        'hit_ball_with_queue_stopper1': ('GEOM', ('box', [0.0002, 0.008, 0.0002])),
        'hit_ball_with_queue_stopper2': ('GEOM', ('box', [0.0002, 0.008, 0.0002])),
    },
    'hockey': {  # hockey.py
        'hockey_goal_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/hockey/hockey_goal_visual/usd/hockey_goal_visual.usd')),
        'hockey_stick': ('RIGIDBODY', ('usd', 'assets/rlbench/hockey/hockey_stick/usd/hockey_stick.usd')),
        'hockey_ball': ('RIGIDBODY', ('usd', 'assets/rlbench/hockey/hockey_ball/usd/hockey_ball.usd')),
    },
    'insert_onto_square_peg': {  # insert_onto_square_peg.py
        'square_ring': ('RIGIDBODY', ('usd', 'assets/rlbench/insert_onto_square_peg/square_ring/usd/square_ring.usd')),
        'pillar0': ('GEOM', ('box', [0.0125, 0.0125, 0.06])),
        'pillar1': ('GEOM', ('box', [0.0125, 0.0125, 0.06])),
        'pillar2': ('GEOM', ('box', [0.0125, 0.0125, 0.06])),
        'square_base': ('GEOM', ('box', [0.2, 0.05, 0.01])),
    },
    'lamp_off': {  # lamp_off.py
        'lamp_base': ('GEOM', ('usd', 'assets/rlbench/lamp_off/lamp_base/usd/lamp_base.usd')),
        'push_button_target': ('ARTICULATION', ('usd', 'assets/rlbench/lamp_off/push_button_target/usd/push_button_target.usd')),
    },
    'lamp_on': {  # lamp_off.py
        'lamp_base': ('GEOM', ('usd', 'assets/rlbench/lamp_off/lamp_base/usd/lamp_base.usd')),
        'push_button_target': ('ARTICULATION', ('usd', 'assets/rlbench/lamp_off/push_button_target/usd/push_button_target.usd')),
    },
    'lift_numbered_block': {  # lift_numbered_block.py
        'block1': ('RIGIDBODY', ('usd', 'assets/rlbench/lift_numbered_block/block1/usd/block1.usd')),
        'block2': ('RIGIDBODY', ('usd', 'assets/rlbench/lift_numbered_block/block2/usd/block2.usd')),
        'block3': ('RIGIDBODY', ('usd', 'assets/rlbench/lift_numbered_block/block3/usd/block3.usd')),
    },
    'light_bulb_in': {  # light_bulb_in.py
        'bulb0': ('RIGIDBODY', ('usd', 'assets/rlbench/light_bulb_in/bulb0/usd/bulb0.usd')),
        'bulb1': ('RIGIDBODY', ('usd', 'assets/rlbench/light_bulb_in/bulb1/usd/bulb1.usd')),
        'bulb_holder0': ('GEOM', ('usd', 'assets/rlbench/light_bulb_in/bulb_holder0/usd/bulb_holder0.usd')),
        'bulb_holder1': ('GEOM', ('usd', 'assets/rlbench/light_bulb_in/bulb_holder1/usd/bulb_holder1.usd')),
        'lamp_base': ('GEOM', ('usd', 'assets/rlbench/light_bulb_in/lamp_base/usd/lamp_base.usd')),
    },
    'light_bulb_out': {  # light_bulb_out.py
        'bulb': ('RIGIDBODY', ('usd', 'assets/rlbench/light_bulb_in/bulb0/usd/bulb0.usd')),
        'bulb_holder0': ('GEOM', ('usd', 'assets/rlbench/light_bulb_in/bulb_holder0/usd/bulb_holder0.usd')),
        'bulb_holder1': ('GEOM', ('usd', 'assets/rlbench/light_bulb_in/bulb_holder1/usd/bulb_holder1.usd')),
        'lamp_base': ('GEOM', ('usd', 'assets/rlbench/light_bulb_in/lamp_base/usd/lamp_base.usd')),
    },
    'meat_off_grill': {  # meat_off_grill.py
        'grill_visual': ('GEOM', ('usd', 'assets/rlbench/meat_off_grill/grill_visual/usd/grill_visual.usd')),
        'chicken_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/meat_off_grill/chicken_visual/usd/chicken_visual.usd')),
        'steak_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/meat_off_grill/steak_visual/usd/steak_visual.usd')),
    },
    'meat_on_grill': {  # meat_off_grill.py
        'grill_visual': ('GEOM', ('usd', 'assets/rlbench/meat_off_grill/grill_visual/usd/grill_visual.usd')),
        'chicken_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/meat_off_grill/chicken_visual/usd/chicken_visual.usd')),
        'steak_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/meat_off_grill/steak_visual/usd/steak_visual.usd')),
    },
    'open_box': {  # open_box.py
        'box_base': ('ARTICULATION', ('usd', 'assets/rlbench/close_box/box_base/usd/box_base.usd')),
    },
    'open_door': {  # close_door.py
        'door_frame': ('ARTICULATION', ('usd', 'assets/rlbench/close_door/door_frame/usd/door_frame.usd')),
    },
    'open_drawer': {  # close_drawer.py
        'drawer_frame': ('ARTICULATION', ('usd', 'assets/rlbench/close_drawer/drawer_frame/usd/drawer_frame.usd')),
    },
    'open_fridge': {  # close_fridge.py
        'fridge_base': ('ARTICULATION', ('usd', 'assets/rlbench/close_fridge/fridge_base/usd/fridge_base.usd')),
    },
    'open_grill': {  # open_grill.py
        'grill': ('ARTICULATION', ('usd', 'assets/rlbench/open_grill/grill/usd/grill.usd')),
    },
    'open_jar': {  # open_jar.py
        'jar0': ('RIGIDBODY', ('usd', 'assets/rlbench/open_jar/jar0/usd/jar0.usd')),
        'jar1': ('RIGIDBODY', ('usd', 'assets/rlbench/open_jar/jar1/usd/jar1.usd')),
        'jar_lid0': ('RIGIDBODY', ('usd', 'assets/rlbench/open_jar/jar_lid0/usd/jar_lid0.usd')),
        'jar_lid1': ('RIGIDBODY', ('usd', 'assets/rlbench/open_jar/jar_lid1/usd/jar_lid1.usd')),
        'target': ('GEOM', ('usd', 'assets/rlbench/open_jar/target/usd/target.usd')),
    },
    'open_microwave': {  # close_microwave.py
        'microwave_frame_resp': ('ARTICULATION', ('usd', 'assets/rlbench/close_microwave/microwave_frame_resp/usd/microwave_frame_resp.usd')),
    },
    'open_oven': {  # open_oven.py
        'oven_base': ('ARTICULATION', ('usd', 'assets/rlbench/open_oven/oven_base/usd/oven_base.usd')),
    },
    'open_washing_machine': {  # open_washing_machine.py
        'washer': ('ARTICULATION', ('usd', 'assets/rlbench/open_washing_machine/washer/usd/washer.usd')),
    },
    'open_window': {  # open_window.py
        'window_main': ('ARTICULATION', ('usd', 'assets/rlbench/open_window/window_main/usd/window_main.usd')),
    },
    'open_wine_bottle': {  # open_wine_bottle.py
        'bottle': ('ARTICULATION', ('usd', 'assets/rlbench/open_wine_bottle/bottle/usd/bottle.usd')),
    },
    'phone_on_base': {  # phone_on_base.py
        'phone_visual': ('XFORM', ('usd', 'assets/rlbench/phone_on_base/phone_visual/usd/phone_visual.usd')),
        'phone_case_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/phone_on_base/phone_case_visual/usd/phone_case_visual.usd')),
    },
    'pick_and_lift': {  # pick_and_lift.py
        'pick_and_lift_target': ('RIGIDBODY', ('box', [0.025, 0.025, 0.025])),
        'stack_blocks_distractor0': ('RIGIDBODY', ('box', [0.025, 0.025, 0.025])),
        'stack_blocks_distractor1': ('RIGIDBODY', ('box', [0.025, 0.025, 0.025])),
        'success_visual': ('XFORM', ('sphere', 0.04)),
    },
    'pick_and_lift_small': {  # pick_and_lift_small.py
        'triangular_prism': ('RIGIDBODY', ('usd', 'assets/rlbench/pick_and_lift_small/triangular_prism/usd/triangular_prism.usd')),
        'star_visual': ('XFORM', ('usd', 'assets/rlbench/pick_and_lift_small/star_visual/usd/star_visual.usd')),
        'moon_visual': ('XFORM', ('usd', 'assets/rlbench/pick_and_lift_small/moon_visual/usd/moon_visual.usd')),
        'cylinder': ('XFORM', ('usd', 'assets/rlbench/pick_and_lift_small/cylinder/usd/cylinder.usd')),
        'cube': ('RIGIDBODY', ('box', [0.010445, 0.010445, 0.010445])),
        'success_visual': ('XFORM', ('sphere', 0.04)),
    },
    'pick_up_cup': {  # pick_up_cup.py
        'cup1_visual': ('XFORM', ('usd', 'assets/rlbench/pick_up_cup/cup1_visual/usd/cup1_visual.usd')),
        'cup2_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/pick_up_cup/cup2_visual/usd/cup2_visual.usd')),
    },
    'place_cups': {  # place_cups.py
        'place_cups_holder_base': ('XFORM', ('usd', 'assets/rlbench/place_cups/place_cups_holder_base/usd/place_cups_holder_base.usd')),
        'mug_visual0': ('RIGIDBODY', ('usd', 'assets/rlbench/place_cups/mug_visual1/usd/mug_visual1.usd')),
        'mug_visual1': ('RIGIDBODY', ('usd', 'assets/rlbench/place_cups/mug_visual1/usd/mug_visual1.usd')),
        'mug_visual2': ('RIGIDBODY', ('usd', 'assets/rlbench/place_cups/mug_visual1/usd/mug_visual1.usd')),
        'mug_visual3': ('RIGIDBODY', ('usd', 'assets/rlbench/place_cups/mug_visual1/usd/mug_visual1.usd')),
    },
    'place_shape_in_shape_sorter': {  # place_shape_in_shape_sorter.py
        'shape_sorter': ('RIGIDBODY', ('usd', 'assets/rlbench/place_shape_in_shape_sorter/shape_sorter/usd/shape_sorter.usd')),
        'triangular_prism': ('RIGIDBODY', ('usd', 'assets/rlbench/pick_and_lift_small/triangular_prism/usd/triangular_prism.usd')),
        'star_visual': ('XFORM', ('usd', 'assets/rlbench/pick_and_lift_small/star_visual/usd/star_visual.usd')),
        'moon_visual': ('XFORM', ('usd', 'assets/rlbench/pick_and_lift_small/moon_visual/usd/moon_visual.usd')),
        'cylinder': ('XFORM', ('usd', 'assets/rlbench/pick_and_lift_small/cylinder/usd/cylinder.usd')),
        'cube': ('RIGIDBODY', ('box', [0.010445, 0.010445, 0.010445])),
    },
    'play_jenga': {  # play_jenga.py
        'Cuboid': ('RIGIDBODY', ('usd', 'assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd')),
        'Cuboid0': ('RIGIDBODY', ('usd', 'assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd')),
        'Cuboid1': ('RIGIDBODY', ('usd', 'assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd')),
        'Cuboid2': ('RIGIDBODY', ('usd', 'assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd')),
        'Cuboid3': ('RIGIDBODY', ('usd', 'assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd')),
        'Cuboid4': ('RIGIDBODY', ('usd', 'assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd')),
        'Cuboid5': ('RIGIDBODY', ('usd', 'assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd')),
        'Cuboid6': ('RIGIDBODY', ('usd', 'assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd')),
        'Cuboid7': ('RIGIDBODY', ('usd', 'assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd')),
        'Cuboid8': ('RIGIDBODY', ('usd', 'assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd')),
        'Cuboid9': ('RIGIDBODY', ('usd', 'assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd')),
        'Cuboid10': ('RIGIDBODY', ('usd', 'assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd')),
        'Cuboid11': ('RIGIDBODY', ('usd', 'assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd')),
        'Cuboid12': ('RIGIDBODY', ('usd', 'assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd')),
        'target_cuboid': ('RIGIDBODY', ('usd', 'assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd')),
    },
    'plug_charger_in_power_supply': {  # plug_charger_in_power_supply.py
        'charger': ('RIGIDBODY', ('usd', 'assets/rlbench/plug_charger_in_power_supply/charger/usd/charger.usd')),
        'task_wall': ('GEOM', ('usd', 'assets/rlbench/plug_charger_in_power_supply/task_wall/usd/task_wall.usd')),
        'plug': ('GEOM', ('usd', 'assets/rlbench/plug_charger_in_power_supply/plug/usd/plug.usd')),
    },
    'pour_from_cup_to_cup': {  # pour_from_cup_to_cup.py
        'cup_distractor_visual0': ('XFORM', ('usd', 'assets/rlbench/pour_from_cup_to_cup/cup_distractor_visual0/usd/cup_distractor_visual0.usd')),
        'cup_distractor_visual1': ('RIGIDBODY', ('usd', 'assets/rlbench/pour_from_cup_to_cup/cup_distractor_visual1/usd/cup_distractor_visual1.usd')),
        'cup_distractor_visual2': ('RIGIDBODY', ('usd', 'assets/rlbench/pour_from_cup_to_cup/cup_distractor_visual2/usd/cup_distractor_visual2.usd')),
        'cup_target_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/pour_from_cup_to_cup/cup_target_visual/usd/cup_target_visual.usd')),
        'cup_source_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/pour_from_cup_to_cup/cup_source_visual/usd/cup_source_visual.usd')),
    },
    'press_switch': {  # press_switch.py
        'switch_main': ('ARTICULATION', ('usd', 'assets/rlbench/press_switch/switch_main/usd/switch_main.usd')),
        'task_wall': ('GEOM', ('usd', 'assets/rlbench/press_switch/task_wall/usd/task_wall.usd')),
    },
    'push_button': {  # push_button.py
        'push_button_target': ('ARTICULATION', ('usd', 'assets/rlbench/push_button/push_button_target/usd/push_button_target.usd')),
    },
    'push_buttons': {  # push_buttons.py
        'push_buttons_target0': ('ARTICULATION', ('usd', 'assets/rlbench/push_buttons/push_buttons_target0/usd/push_buttons_target0.usd')),
        'push_buttons_target1': ('ARTICULATION', ('usd', 'assets/rlbench/push_buttons/push_buttons_target1/usd/push_buttons_target1.usd')),
        'push_buttons_target2': ('ARTICULATION', ('usd', 'assets/rlbench/push_buttons/push_buttons_target2/usd/push_buttons_target2.usd')),
    },
    'put_all_groceries_in_cupboard': {  # put_groceries_in_cupboard.py
        'cupboard': ('GEOM', ('usd', 'assets/rlbench/put_groceries_in_cupboard/cupboard/usd/cupboard.usd')),
        'chocolate_jello_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_groceries_in_cupboard/chocolate_jello_visual/usd/chocolate_jello_visual.usd')),
        'strawberry_jello_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_groceries_in_cupboard/strawberry_jello_visual/usd/strawberry_jello_visual.usd')),
        'spam_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_groceries_in_cupboard/spam_visual/usd/spam_visual.usd')),
        'sugar_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_groceries_in_cupboard/sugar_visual/usd/sugar_visual.usd')),
        'crackers_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_groceries_in_cupboard/crackers_visual/usd/crackers_visual.usd')),
        'mustard_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_groceries_in_cupboard/mustard_visual/usd/mustard_visual.usd')),
        'soup_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_groceries_in_cupboard/soup_visual/usd/soup_visual.usd')),
    },
    'put_books_on_bookshelf': {  # put_books_on_bookshelf.py
        'bookshelf_visual': ('GEOM', ('usd', 'assets/rlbench/put_books_on_bookshelf/bookshelf_visual/usd/bookshelf_visual.usd')),
        'book0_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_books_on_bookshelf/book0_visual/usd/book0_visual.usd')),
        'book1_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_books_on_bookshelf/book1_visual/usd/book1_visual.usd')),
        'book2_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_books_on_bookshelf/book2_visual/usd/book2_visual.usd')),
    },
    'put_bottle_in_fridge': {  # put_bottle_in_fridge.py
        'bottle_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_bottle_in_fridge/bottle_visual/usd/bottle_visual.usd')),
        'fridge_base': ('ARTICULATION', ('usd', 'assets/rlbench/put_bottle_in_fridge/fridge_base/usd/fridge_base.usd')),
    },
    'put_groceries_in_cupboard': {  # put_groceries_in_cupboard.py
        'cupboard': ('GEOM', ('usd', 'assets/rlbench/put_groceries_in_cupboard/cupboard/usd/cupboard.usd')),
        'chocolate_jello_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_groceries_in_cupboard/chocolate_jello_visual/usd/chocolate_jello_visual.usd')),
        'strawberry_jello_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_groceries_in_cupboard/strawberry_jello_visual/usd/strawberry_jello_visual.usd')),
        'spam_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_groceries_in_cupboard/spam_visual/usd/spam_visual.usd')),
        'sugar_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_groceries_in_cupboard/sugar_visual/usd/sugar_visual.usd')),
        'crackers_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_groceries_in_cupboard/crackers_visual/usd/crackers_visual.usd')),
        'mustard_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_groceries_in_cupboard/mustard_visual/usd/mustard_visual.usd')),
        'soup_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_groceries_in_cupboard/soup_visual/usd/soup_visual.usd')),
        'tuna_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_groceries_in_cupboard/tuna_visual/usd/tuna_visual.usd')),
        'coffee_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_groceries_in_cupboard/coffee_visual/usd/coffee_visual.usd')),
    },
    'put_item_in_drawer': {  # put_item_in_drawer.py
        'drawer_frame': ('ARTICULATION', ('usd', 'assets/rlbench/put_item_in_drawer/drawer_frame/usd/drawer_frame.usd')),
        'item': ('RIGIDBODY', ('box', [0.02, 0.02, 0.02])),
    },
    'put_knife_in_knife_block': {  # put_knife_in_knife_block.py
        'chopping_board_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_knife_in_knife_block/chopping_board_visual/usd/chopping_board_visual.usd')),
        'knife_block_visual': ('GEOM', ('usd', 'assets/rlbench/put_knife_in_knife_block/knife_block_visual/usd/knife_block_visual.usd')),
        'knife_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_knife_in_knife_block/knife_visual/usd/knife_visual.usd')),
    },
    'put_knife_on_chopping_board': {  # put_knife_in_knife_block.py
        'chopping_board_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_knife_in_knife_block/chopping_board_visual/usd/chopping_board_visual.usd')),
        'knife_block_visual': ('GEOM', ('usd', 'assets/rlbench/put_knife_in_knife_block/knife_block_visual/usd/knife_block_visual.usd')),
        'knife_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_knife_in_knife_block/knife_visual/usd/knife_visual.usd')),
    },
    'put_money_in_safe': {  # put_money_in_safe.py
        'dollar_stack': ('RIGIDBODY', ('usd', 'assets/rlbench/put_money_in_safe/dollar_stack/usd/dollar_stack.usd')),
        'safe_body': ('ARTICULATION', ('usd', 'assets/rlbench/put_money_in_safe/safe_body/usd/safe_body.usd')),
    },
    'put_plate_in_colored_dish_rack': {  # put_plate_in_colored_dish_rack.py
        'plate_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_plate_in_colored_dish_rack/plate_visual/usd/plate_visual.usd')),
        'dish_rack': ('GEOM', ('usd', 'assets/rlbench/put_plate_in_colored_dish_rack/dish_rack/usd/dish_rack.usd')),
        'plate_stand': ('GEOM', ('usd', 'assets/rlbench/put_plate_in_colored_dish_rack/plate_stand/usd/plate_stand.usd')),
    },
    'put_rubbish_in_bin': {  # put_rubbish_in_bin.py
        'bin_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_rubbish_in_bin/bin_visual/usd/bin_visual.usd')),
        'tomato1_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_rubbish_in_bin/tomato1_visual/usd/tomato1_visual.usd')),
        'tomato2_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_rubbish_in_bin/tomato2_visual/usd/tomato2_visual.usd')),
        'rubbish_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_rubbish_in_bin/rubbish_visual/usd/rubbish_visual.usd')),
    },
    'put_shoes_in_box': {  # put_shoes_in_box.py
        'box_base': ('ARTICULATION', ('usd', 'assets/rlbench/put_shoes_in_box/box_base/usd/box_base.usd')),
        'shoe1_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_shoes_in_box/shoe1_visual/usd/shoe1_visual.usd')),
        'shoe2_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_shoes_in_box/shoe2_visual/usd/shoe2_visual.usd')),
    },
    'put_toilet_roll_on_stand': {  # put_toilet_roll_on_stand.py
        'toilet_roll_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_toilet_roll_on_stand/toilet_roll_visual/usd/toilet_roll_visual.usd')),
        'stand_base': ('RIGIDBODY', ('usd', 'assets/rlbench/put_toilet_roll_on_stand/stand_base/usd/stand_base.usd')),
        'toilet_roll_box': ('RIGIDBODY', ('box', [0.05, 0.05, 0.05])),
    },
    'put_tray_in_oven': {  # open_oven.py
        'oven_base': ('ARTICULATION', ('usd', 'assets/rlbench/open_oven/oven_base/usd/oven_base.usd')),
        'tray_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_tray_in_oven/tray_visual/usd/tray_visual.usd')),
    },
    'put_umbrella_in_umbrella_stand': {  # put_umbrella_in_umbrella_stand.py
        'umbrella_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_umbrella_in_umbrella_stand/umbrella_visual/usd/umbrella_visual.usd')),
        'stand_visual': ('GEOM', ('usd', 'assets/rlbench/put_umbrella_in_umbrella_stand/stand_visual/usd/stand_visual.usd')),
    },
    'reach_and_drag': {  # reach_and_drag.py
        'cube': ('RIGIDBODY', ('box', [0.04, 0.04, 0.04])),
        'stick': ('RIGIDBODY', ('box', [0.0064, 0.0064, 0.18])),
        'target0': ('GEOM', ('usd', 'assets/rlbench/reach_and_drag/target0/usd/target0.usd')),
    },
    'reach_target': {  # reach_target.py
        'target': ('XFORM', ('sphere', 0.025)),
        'distractor0': ('XFORM', ('sphere', 0.025)),
        'distractor1': ('XFORM', ('sphere', 0.025)),
    },
    'remove_cups': {  # place_cups.py
        'place_cups_holder_base': ('XFORM', ('usd', 'assets/rlbench/place_cups/place_cups_holder_base/usd/place_cups_holder_base.usd')),
        'mug_visual0': ('RIGIDBODY', ('usd', 'assets/rlbench/place_cups/mug_visual1/usd/mug_visual1.usd')),
        'mug_visual1': ('RIGIDBODY', ('usd', 'assets/rlbench/place_cups/mug_visual1/usd/mug_visual1.usd')),
        'mug_visual2': ('RIGIDBODY', ('usd', 'assets/rlbench/place_cups/mug_visual1/usd/mug_visual1.usd')),
        'mug_visual3': ('RIGIDBODY', ('usd', 'assets/rlbench/place_cups/mug_visual1/usd/mug_visual1.usd')),
    },
    'scoop_with_spatula': {  # scoop_with_spatula.py
        'spatula_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/scoop_with_spatula/spatula_visual/usd/spatula_visual.usd')),
        'Cuboid': ('RIGIDBODY', ('box', [0.01, 0.01, 0.01])),
    },
    'screw_nail': {  # screw_nail.py
        'block': ('ARTICULATION', ('usd', 'assets/rlbench/screw_nail/block/usd/block.usd')),
        'screw_driver': ('RIGIDBODY', ('usd', 'assets/rlbench/screw_nail/screw_driver/usd/screw_driver.usd')),
    },
    'set_the_table': {  # set_the_table.py
        'holder': ('GEOM', ('usd', 'assets/rlbench/set_the_table/holder/usd/holder.usd')),
        'fork_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/set_the_table/fork_visual/usd/fork_visual.usd')),
        'knife_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/set_the_table/knife_visual/usd/knife_visual.usd')),
        'spoon_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/set_the_table/spoon_visual/usd/spoon_visual.usd')),
        'plate_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/set_the_table/plate_visual/usd/plate_visual.usd')),
        'glass_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/set_the_table/glass_visual/usd/glass_visual.usd')),
    },
    'setup_checkers': {  # setup_checkers.py
        'chess_board_base_visual': ('GEOM', ('usd', 'assets/rlbench/setup_chess/chess_board_base_visual/usd/chess_board_base_visual.usd')),
        'checker0': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker1': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker2': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker3': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker4': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker5': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker6': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker7': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker8': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker9': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker10': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker11': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker12': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker13': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker14': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker15': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker16': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker17': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker18': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker19': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker20': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker21': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker22': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
        'checker23': ('RIGIDBODY', ('cylinder', 0.017945, 0.00718)),
    },
    'setup_chess': {  # setup_chess.py
        'chess_board_base_visual': ('GEOM', ('usd', 'assets/rlbench/setup_chess/chess_board_base_visual/usd/chess_board_base_visual.usd')),
        'white_pawn_a': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/white_pawn_a/usd/white_pawn_a.usd')),
        'white_pawn_b': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/white_pawn_a/usd/white_pawn_a.usd')),
        'white_pawn_c': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/white_pawn_a/usd/white_pawn_a.usd')),
        'white_pawn_d': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/white_pawn_a/usd/white_pawn_a.usd')),
        'white_pawn_e': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/white_pawn_a/usd/white_pawn_a.usd')),
        'white_pawn_f': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/white_pawn_a/usd/white_pawn_a.usd')),
        'white_pawn_g': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/white_pawn_a/usd/white_pawn_a.usd')),
        'white_pawn_h': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/white_pawn_a/usd/white_pawn_a.usd')),
        'black_pawn_a': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/black_pawn_a/usd/black_pawn_a.usd')),
        'black_pawn_b': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/black_pawn_a/usd/black_pawn_a.usd')),
        'black_pawn_c': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/black_pawn_a/usd/black_pawn_a.usd')),
        'black_pawn_d': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/black_pawn_a/usd/black_pawn_a.usd')),
        'black_pawn_e': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/black_pawn_a/usd/black_pawn_a.usd')),
        'black_pawn_f': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/black_pawn_a/usd/black_pawn_a.usd')),
        'black_pawn_g': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/black_pawn_a/usd/black_pawn_a.usd')),
        'black_pawn_h': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/black_pawn_a/usd/black_pawn_a.usd')),
        'white_king': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/white_king/usd/white_king.usd')),
        'white_queen': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/white_queen/usd/white_queen.usd')),
        'black_king': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/black_king/usd/black_king.usd')),
        'black_queen': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/black_queen/usd/black_queen.usd')),
        'white_kingside_rook': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/white_kingside_rook/usd/white_kingside_rook.usd')),
        'white_queenside_rook': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/white_kingside_rook/usd/white_kingside_rook.usd')),
        'white_kingside_knight': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/white_kingside_knight/usd/white_kingside_knight.usd')),
        'white_queenside_knight': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/white_kingside_knight/usd/white_kingside_knight.usd')),
        'white_kingside_bishop': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/white_kingside_bishop/usd/white_kingside_bishop.usd')),
        'white_queenside_bishop': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/white_kingside_bishop/usd/white_kingside_bishop.usd')),
        'black_kingside_rook': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/black_kingside_rook/usd/black_kingside_rook.usd')),
        'black_queenside_rook': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/black_kingside_rook/usd/black_kingside_rook.usd')),
        'black_kingside_knight': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/black_kingside_knight/usd/black_kingside_knight.usd')),
        'black_queenside_knight': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/black_kingside_knight/usd/black_kingside_knight.usd')),
        'black_kingside_bishop': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/black_kingside_bishop/usd/black_kingside_bishop.usd')),
        'black_queenside_bishop': ('RIGIDBODY', ('usd', 'assets/rlbench/setup_chess/black_kingside_bishop/usd/black_kingside_bishop.usd')),
    },
    'slide_block_to_target': {  # slide_block_to_target.py
        'block': ('RIGIDBODY', ('box', [0.028125, 0.028125, 0.028125])),
        'target': ('GEOM', ('usd', 'assets/rlbench/slide_block_to_target/target/usd/target.usd')),
    },
    'slide_cabinet_open_and_place_cups': {  # slide_cabinet_open_and_place_cups.py
        'cabinet_base': ('ARTICULATION', ('usd', 'assets/rlbench/slide_cabinet_open_and_place_cups/cabinet_base/usd/cabinet_base.usd')),
        'cup_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/slide_cabinet_open_and_place_cups/cup_visual/usd/cup_visual.usd')),
    },
    'stack_blocks': {  # stack_blocks.py
        'stack_blocks_target_plane': ('GEOM', ('usd', 'assets/rlbench/stack_blocks/stack_blocks_target_plane/usd/stack_blocks_target_plane.usd')),
        'stack_blocks_target0': ('RIGIDBODY', ('box', [0.025, 0.025, 0.025])),
        'stack_blocks_target1': ('RIGIDBODY', ('box', [0.025, 0.025, 0.025])),
        'stack_blocks_target2': ('RIGIDBODY', ('box', [0.025, 0.025, 0.025])),
        'stack_blocks_target3': ('RIGIDBODY', ('box', [0.025, 0.025, 0.025])),
        'stack_blocks_distractor0': ('RIGIDBODY', ('box', [0.025, 0.025, 0.025])),
        'stack_blocks_distractor1': ('RIGIDBODY', ('box', [0.025, 0.025, 0.025])),
        'stack_blocks_distractor2': ('RIGIDBODY', ('box', [0.025, 0.025, 0.025])),
        'stack_blocks_distractor3': ('RIGIDBODY', ('box', [0.025, 0.025, 0.025])),
    },
    'stack_chairs': {  # stack_chairs.py
        'chair1': ('RIGIDBODY', ('usd', 'assets/rlbench/stack_chairs/chair1/usd/chair1.usd')),
        'chair2': ('RIGIDBODY', ('usd', 'assets/rlbench/stack_chairs/chair2/usd/chair2.usd')),
        'chair3': ('RIGIDBODY', ('usd', 'assets/rlbench/stack_chairs/chair3/usd/chair3.usd')),
    },
    'stack_cups': {  # stack_cups.py
        'cup1_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/stack_cups/cup1_visual/usd/cup1_visual.usd')),
        'cup2_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/stack_cups/cup2_visual/usd/cup2_visual.usd')),
        'cup3_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/stack_cups/cup3_visual/usd/cup3_visual.usd')),
    },
    'stack_wine': {  # stack_wine.py
        'wine_bottle_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/stack_wine/wine_bottle_visual/usd/wine_bottle_visual.usd')),
        'rack_bottom_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/stack_wine/rack_bottom_visual/usd/rack_bottom_visual.usd')),
        'rack_top_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/stack_wine/rack_top_visual/usd/rack_top_visual.usd')),
    },
    'sweep_to_dustpan': {  # sweep_to_dustpan.py
        'Dustpan_4': ('RIGIDBODY', ('usd', 'assets/rlbench/sweep_to_dustpan/Dustpan_4/usd/Dustpan_4.usd')),
        'sweep_to_dustpan_broom_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/sweep_to_dustpan/sweep_to_dustpan_broom_visual/usd/sweep_to_dustpan_broom_visual.usd')),
        'broom_holder': ('GEOM', ('box', [0.05, 0.05, 0.25])),
        'dirt0': ('RIGIDBODY', ('box', [0.005, 0.005, 0.005])),
        'dirt1': ('RIGIDBODY', ('box', [0.005, 0.005, 0.005])),
        'dirt2': ('RIGIDBODY', ('box', [0.005, 0.005, 0.005])),
        'dirt3': ('RIGIDBODY', ('box', [0.005, 0.005, 0.005])),
        'dirt4': ('RIGIDBODY', ('box', [0.005, 0.005, 0.005])),
    },
    'take_cup_out_from_cabinet': {  # slide_cabinet_open_and_place_cups.py
        'cabinet_base': ('ARTICULATION', ('usd', 'assets/rlbench/slide_cabinet_open_and_place_cups/cabinet_base/usd/cabinet_base.usd')),
        'cup_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/slide_cabinet_open_and_place_cups/cup_visual/usd/cup_visual.usd')),
    },
    'take_tray_out_of_oven': {  # open_oven.py
        'oven_base': ('ARTICULATION', ('usd', 'assets/rlbench/open_oven/oven_base/usd/oven_base.usd')),
        'tray_visual': ('RIGIDBODY', ('usd', 'assets/rlbench/put_tray_in_oven/tray_visual/usd/tray_visual.usd')),
    },
}
