"""Task predicates must reject plausible but unsuccessful physical outcomes.

Small MuJoCo scenes exercise compiled world transforms and actual contact
generation. They do not invoke the Reachy controller, acquire data, or pretend
that a geometric predicate alone establishes whole-episode success.
"""

from types import SimpleNamespace
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget import dynamics


def scene(bodies, placement=None):
    """Build independent free bodies at source-frame poses, then place the scene."""
    placement = np.eye(4) if placement is None else placement
    xml = ET.Element("mujoco")
    ET.SubElement(xml, "option", gravity="0 0 0")
    world = ET.SubElement(xml, "worldbody")
    for name, position, size in bodies:
        p = (placement @ np.r_[position, 1.])[:3]
        quat = Rotation.from_matrix(placement[:3, :3]).as_quat()[[3, 0, 1, 2]]
        body = ET.SubElement(world, "body", name=name,
                             pos=" ".join(map(str, p)), quat=" ".join(map(str, quat)))
        ET.SubElement(body, "freejoint", name=name + "_joint")
        ET.SubElement(body, "geom", name=name + "_collision", type="box",
                      size=" ".join(map(str, size)), mass=".1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, {name: model.body(name).id for name, _, _ in bodies}, placement


def satisfied(task, oid, state, *, closed=False, hand_contact=False):
    model, data, body_ids, placement = state
    return dynamics.task_state(task, oid, model, data, body_ids, placement,
                               closed, hand_contact, data.xpos[body_ids[oid]].copy())


@pytest.mark.parametrize("position", [
    (0., .15, .89),  # Lower-left compartment: inside bin, wrong Can target.
    (.2, .15, .89),  # Lower-right compartment.
    (0., .4, .89),  # Upper-left compartment.
    (.1, .4, .89),  # Divider/boundary is not inside the target compartment.
    (.2, .28, .89),
    (.31, .4, .89),  # Beyond the far wall.
    (.2, .4, .02),  # Dropped below the arena.
    (.2, .4, .92),  # Above publisher bin2_z + .1; e.g. supported on a divider.
])
def test_can_rejects_other_compartments_boundaries_and_floor(position):
    state = scene([("Can", position, (.02, .02, .04))])
    assert not satisfied("PickPlaceCan", "Can", state)


@pytest.mark.parametrize("closed,hand_contact,expected", [
    (False, False, True),
    (True, False, False),
    (False, True, False),
])
def test_can_requires_release_in_correct_compartment(closed, hand_contact, expected):
    placement = np.eye(4)
    placement[:3, :3] = Rotation.from_euler("z", 1.1).as_matrix()
    placement[:3, 3] = [.6, -.2, .1]
    state = scene([("Can", (.2, .4, .89), (.02, .02, .04))], placement)
    assert satisfied("PickPlaceCan", "Can", state,
                     closed=closed, hand_contact=hand_contact) is expected


@pytest.mark.parametrize("position", [
    (.1, .1, .02),  # Settled on the floor directly beneath the peg.
    (.1, .1, .80),  # Below the assembly band.
    (.1, .1, .89),  # Hovering above the peg, not assembled.
    (.14, .1, .83),  # Correct height but beside the peg.
])
def test_nut_rejects_floor_hover_and_wrong_peg_region(position):
    state = scene([("SquareNut", position, (.01, .01, .005)),
                   ("peg1", (.1, .1, .82), (.005, .005, .02))])
    assert not satisfied("NutAssemblySquare", "SquareNut", state)


def test_nut_region_alone_does_not_bypass_release():
    state = scene([("SquareNut", (.1, .1, .83), (.01, .01, .005)),
                   ("peg1", (.1, .1, .82), (.005, .005, .02))])
    assert not satisfied("NutAssemblySquare", "SquareNut", state, closed=True)
    assert not satisfied("NutAssemblySquare", "SquareNut", state, hand_contact=True)


def test_stack_requires_actual_contact_not_just_relative_height():
    touching = scene([("cubeA", (0., 0., .8695), (.02, .02, .02)),
                      ("cubeB", (0., 0., .825), (.025, .025, .025))])
    assert touching[1].ncon > 0
    assert satisfied("Stack_D0", "cubeA", touching)
    separated = scene([("cubeA", (0., 0., .89), (.02, .02, .02)),
                       ("cubeB", (0., 0., .825), (.025, .025, .025))])
    assert separated[1].ncon == 0
    assert not satisfied("Stack_D0", "cubeA", separated)


def test_stack_on_floor_does_not_count_as_tabletop_stacking():
    state = scene([("cubeA", (0., 0., .0695), (.02, .02, .02)),
                   ("cubeB", (0., 0., .025), (.025, .025, .025))])
    assert state[1].ncon > 0
    assert not satisfied("Stack_D0", "cubeA", state)


def test_contact_with_unrelated_object_is_not_stacking_target():
    state = scene([("cubeA", (0., 0., .8695), (.02, .02, .02)),
                   ("cubeB", (.3, 0., .825), (.025, .025, .025)),
                   ("distractor", (0., 0., .825), (.025, .025, .025))])
    assert state[1].ncon > 0
    assert not satisfied("Stack_D0", "cubeA", state)


@pytest.mark.parametrize("closed,hand_contact", [(True, False), (False, True)])
def test_supported_stack_still_requires_release(closed, hand_contact):
    state = scene([("cubeA", (0., 0., .8695), (.02, .02, .02)),
                   ("cubeB", (0., 0., .825), (.025, .025, .025))])
    assert not satisfied("Stack_D0", "cubeA", state,
                         closed=closed, hand_contact=hand_contact)


@pytest.mark.parametrize("height,closed,hand_contact,expected", [
    (.83, True, True, False),
    (.90, False, True, False),
    (.90, True, False, False),
    (.90, True, True, True),
])
def test_lift_requires_height_closed_gripper_and_contact(height, closed, hand_contact, expected):
    state = scene([("cube", (0., 0., height), (.02, .02, .02))])
    assert satisfied("Lift", "cube", state,
                     closed=closed, hand_contact=hand_contact) is expected


def threading_scene(offset):
    xml = ET.Element("mujoco")
    world = ET.SubElement(xml, "worldbody")
    needle = ET.SubElement(world, "body", name="needle_obj",
                           pos=" ".join(map(str, [offset, 0., 1.])))
    ET.SubElement(needle, "freejoint", name="needle_joint")
    ET.SubElement(needle, "geom", name="needle_obj_needle", type="box",
                  size=".03 .001 .001", mass=".02")
    tripod = ET.SubElement(world, "body", name="tripod_obj", pos="0 0 1")
    ET.SubElement(tripod, "freejoint", name="tripod_joint")
    index = 0
    for row in range(6):
        for column in range(6):
            if row not in (0, 5) and column not in (0, 5):
                continue
            ET.SubElement(tripod, "geom", name=f"tripod_obj_ring_{index}",
                          type="box", size=".005 .002 .002", mass=".001",
                          pos=f"0 {column * .004 - .01} {row * .004 - .01}")
            index += 1
    # A rendering-only duplicate must never shift the evaluated ring center.
    ET.SubElement(tripod, "geom", name="tripod_obj_ring_0_vis", type="box",
                  size=".005 .002 .002", pos=".5 0 0", contype="0", conaffinity="0", mass="0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, {name: model.body(name).id for name in ("needle_obj", "tripod_obj")}, np.eye(4)


@pytest.mark.parametrize("offset,expected", [(.009, True), (.011, True), (.013, False)])
def test_threading_uses_publisher_twelve_mm_radius_and_ignores_visuals(offset, expected):
    # RingTripodObject unit_size[1] * pattern.shape[2] = .002 * 6.
    # https://github.com/NVlabs/mimicgen/blob/main/mimicgen/models/robosuite/objects/composite/ring_tripod.py
    assert satisfied("Threading_D0", "needle_obj", threading_scene(offset),
                     closed=True, hand_contact=True) is expected


@pytest.mark.parametrize("closed,hand_contact", [(False, True), (True, False)])
def test_threading_proximity_alone_does_not_establish_held_insertion(closed, hand_contact):
    assert not satisfied("Threading_D0", "needle_obj", threading_scene(.005),
                         closed=closed, hand_contact=hand_contact)


def demonstration():
    poses = np.zeros((4, 7))
    poses[:, 2] = [.82, .82, .86, .90]
    poses[:, 3] = 1.
    hand = poses.copy()
    hand[:, 0] += .03
    actions = np.zeros((4, 7))
    actions[:, 6] = [-1., 1., 1., 1.]
    return {"hand/right_pose": hand, "objects/cube/pose": poses,
            "source/action": actions, "time_s": np.arange(4) * .05}


@pytest.mark.parametrize("value", [0., .4, np.nan, np.inf])
def test_source_gripper_contract_does_not_guess_other_action_encodings(value):
    arrays = demonstration()
    arrays["source/action"][1, 6] = value
    with pytest.raises(ValueError, match="gripper schema"):
        dynamics.source_grasp(arrays, {}, "cube", geometry_center=False)


def test_source_attachment_requires_demonstrated_closed_lift():
    arrays = demonstration()
    arrays["source/action"][:, 6] = -1.
    with pytest.raises(ValueError, match="closed gripper"):
        dynamics.source_grasp(arrays, {}, "cube", geometry_center=False)
    arrays = demonstration()
    arrays["objects/cube/pose"][:, 2] = .82
    with pytest.raises(ValueError, match="source lift"):
        dynamics.source_grasp(arrays, {}, "cube", geometry_center=False)


def test_source_grasp_does_not_mutate_reference_arrays():
    arrays = demonstration()
    originals = {key: value.copy() for key, value in arrays.items()}
    _, _, _, anchor, offset, _ = dynamics.source_grasp(arrays, {}, "cube", geometry_center=False)
    assert anchor == 2
    np.testing.assert_allclose(offset, [.03, 0., 0.])
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], originals[key])


@pytest.mark.parametrize("task", ["ToolHang", "HUMOTO", "UnknownTask"])
def test_plan_rejects_tasks_without_explicit_dynamics_adapter(tmp_path, monkeypatch, task):
    monkeypatch.setattr(dynamics, "read_episode",
                        lambda path: ({}, {"env_args": {"env_name": task}}))
    with pytest.raises(ValueError, match="No complete dynamics task adapter"):
        dynamics.plan(SimpleNamespace(root=tmp_path), {"path": "not-read.h5"},
                      dynamics.Candidate(), None)


def contract_scene(task="Lift", placement=None, table_top=None):
    """A small compiled source arena with the inspected release's metadata."""
    placement = np.eye(4) if placement is None else placement
    objects = {"Lift": ["cube"], "PickPlaceCan": ["Can"],
               "NutAssemblySquare": ["SquareNut"], "Stack_D0": ["cubeA", "cubeB"],
               "Threading_D0": ["needle_obj", "tripod_obj"]}[task]
    old_release = task in {"Stack_D0", "Threading_D0"}
    controller = {"type": "OSC_POSE"}
    if not old_release:
        controller = {"type": "BASIC", "body_parts": {"right": {
            "type": "OSC_POSE", "gripper": {"type": "GRIP"}}}}
    metadata = {
        "env_args": {"env_name": task, "env_version": "1.4.1" if old_release else "1.5.1",
                     "env_kwargs": {"robots": ["Panda"], "control_freq": 20,
                                    "controller_configs": controller}},
        "objects": {oid: {"body": oid, "joint": oid + "_joint", "pose_frame": "world"}
                    for oid in objects}}
    arrays = demonstration()
    poses = arrays.pop("objects/cube/pose")
    arrays.update({f"objects/{oid}/pose": poses.copy() for oid in objects})
    xml = ET.Element("mujoco")
    world = ET.SubElement(xml, "worldbody")

    def body(name, pos, free=False):
        quat = Rotation.from_matrix(placement[:3, :3]).as_quat()[[3, 0, 1, 2]]
        position = (placement @ np.r_[pos, 1.])[:3]
        element = ET.SubElement(world, "body", name=name,
                                pos=" ".join(map(str, position)), quat=" ".join(map(str, quat)))
        if free:
            ET.SubElement(element, "freejoint", name=name + "_joint")
        return element

    if task == "PickPlaceCan":
        ET.SubElement(body("bin2", [.1, .28, .8]), "geom", name="bin_floor",
                      type="box", size=".2 .25 .02")
    else:
        top = (.82 if task == "NutAssemblySquare" else .8) if table_top is None else table_top
        ET.SubElement(body("table", [0., 0., top - .025]), "geom", name="table_collision",
                      type="box", size=".4 .4 .025")
    for oid in objects:
        element = body(oid, [0., 0., .9], free=True)
        if oid == "tripod_obj":
            index = 0
            for row in range(6):
                for column in range(6):
                    if row not in (0, 5) and column not in (0, 5):
                        continue
                    ET.SubElement(element, "geom", name=f"tripod_obj_ring_{index}",
                                  type="box", size=".005 .002 .002", mass=".001",
                                  pos=f"0 {column * .004 - .01} {row * .004 - .01}")
                    index += 1
        else:
            name = "needle_obj_needle" if oid == "needle_obj" else oid + "_collision"
            ET.SubElement(element, "geom", name=name, type="box", size=".01 .01 .01", mass=".02")
    if task == "NutAssemblySquare":
        ET.SubElement(body("peg1", [.23, .1, .85]), "geom", type="box", size=".016 .016 .1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return metadata, arrays, model, data, placement


@pytest.mark.parametrize("task", ["Lift", "PickPlaceCan", "NutAssemblySquare", "Stack_D0", "Threading_D0"])
def test_known_source_contracts_are_explicit_json_records_without_array_mutation(task):
    args = contract_scene(task)
    originals = {key: value.copy() for key, value in args[1].items()}
    contract = dynamics.validate_task_contract(*args)
    assert contract["task"] == task
    assert contract["source_version"] == args[0]["env_args"]["env_version"]
    assert contract["object_ids"] == sorted(args[0]["objects"])
    assert json.loads(json.dumps(contract)) == contract
    for key, original in originals.items():
        np.testing.assert_array_equal(args[1][key], original)


def test_contract_derives_square_tabletop_and_preserves_source_frame_under_scene_placement():
    placement = np.eye(4)
    placement[:3, :3] = Rotation.from_euler("z", 1.1).as_matrix()
    placement[:3, 3] = [.6, -.2, .1]
    contract = dynamics.validate_task_contract(*contract_scene("NutAssemblySquare", placement))
    assert contract["table_top"] == pytest.approx(.82)
    assert contract["assembly_min_height"] == pytest.approx(.82)
    assert contract["assembly_max_height"] == pytest.approx(.87)


def test_contract_uses_can_logical_compartment_bounds_not_collision_floor_extents():
    contract = dynamics.validate_task_contract(*contract_scene("PickPlaceCan"))
    np.testing.assert_allclose(contract["bin_lower"], [.1, .28, .8])
    np.testing.assert_allclose(contract["bin_upper"], [.295, .525, .9])
    assert contract["bin_index"] == 3


def test_contract_uses_compiled_twelve_mm_threading_ring():
    contract = dynamics.validate_task_contract(*contract_scene("Threading_D0"))
    assert contract["threading_radius"] == pytest.approx(.012)
    assert len(contract["ring_geoms"]) == 20


@pytest.mark.parametrize("field,value", [("env_version", "1.4.1"), ("env_version", None),
                                         ("env_name", "UnverifiedLift")])
def test_contract_rejects_unknown_task_release_pairs(field, value):
    args = contract_scene()
    args[0]["env_args"][field] = value
    with pytest.raises(ValueError, match="task/version contract"):
        dynamics.validate_task_contract(*args)


@pytest.mark.parametrize("field,value,message", [
    ("robots", ["Sawyer"], "robot or control frequency"),
    ("control_freq", 10, "robot or control frequency"),
    ("table_offset", [0., 0., .8], "task-specific"),
    ("table_full_size", [.4, .4, .05], "table-size"),
    ("controller_configs", {"type": "JOINT_POSITION"}, "composite-controller"),
])
def test_contract_rejects_unverified_robot_controller_and_arena_configuration(field, value, message):
    args = contract_scene()
    args[0]["env_args"]["env_kwargs"][field] = value
    with pytest.raises(ValueError, match=message):
        dynamics.validate_task_contract(*args)


@pytest.mark.parametrize("task", ["Lift", "NutAssemblySquare"])
def test_contract_rejects_unverified_compiled_table_height_even_when_metadata_looks_supported(task):
    with pytest.raises(ValueError, match="table height"):
        dynamics.validate_task_contract(*contract_scene(task, table_top=.9))


def test_contract_rejects_bin_pose_that_disagrees_with_source_configuration():
    args = contract_scene("PickPlaceCan")
    args[0]["env_args"]["env_kwargs"]["bin2_pos"] = [.2, .28, .8]
    with pytest.raises(ValueError, match="bin pose"):
        dynamics.validate_task_contract(*args)


def test_contract_rejects_a_deformed_threading_ring():
    args = contract_scene("Threading_D0")
    model, data = args[2:4]
    model.geom_pos[model.geom("tripod_obj_ring_0").id, 1] += .003
    mujoco.mj_forward(model, data)
    with pytest.raises(ValueError, match="ring pattern"):
        dynamics.validate_task_contract(*args)


@pytest.mark.parametrize("problem", ["object_scope", "pose_frame", "invalid_mask", "wrong_time",
                                     "action_shape", "nonunit_quaternion"])
def test_contract_rejects_metadata_and_trajectory_inputs_that_cannot_support_its_semantics(problem):
    args = contract_scene()
    metadata, arrays = args[:2]
    if problem == "object_scope":
        metadata["objects"]["distractor"] = metadata["objects"]["cube"].copy()
    elif problem == "pose_frame":
        metadata["objects"]["cube"]["pose_frame"] = "camera"
    elif problem == "invalid_mask":
        arrays["objects/cube/valid"] = np.array([True, True, False, True])
    elif problem == "wrong_time":
        arrays["time_s"] *= 2
    elif problem == "action_shape":
        arrays["source/action"] = arrays["source/action"][:, :6]
    elif problem == "nonunit_quaternion":
        arrays["objects/cube/pose"][2, 3:] *= .5
    with pytest.raises(ValueError):
        dynamics.validate_task_contract(*args)


def test_task_predicate_uses_passed_contract_and_rejects_contract_for_another_task():
    state = scene([("cube", (0., 0., .86), (.02, .02, .02))])
    model, data, ids, placement = state
    args = ("Lift", "cube", model, data, ids, placement, True, True, data.xpos[ids["cube"]].copy())
    assert dynamics.task_state(*args, contract={"task": "Lift", "lift_height": .84})
    assert not dynamics.task_state(*args, contract={"task": "Lift", "lift_height": .88})
    with pytest.raises(ValueError, match="Task predicate"):
        dynamics.task_state(*args, contract={"task": "PickPlaceCan", "lift_height": .84})
