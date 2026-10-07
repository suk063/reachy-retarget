from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
import pytest

from reachy_retarget.native_replay.bigym import SOURCE_REVISION, TASK_OBJECTS
from reachy_retarget.native_replay import bigym_retarget as adapter


def record():
    count = 5
    pose = np.tile([0., 0., 1., 1., 0., 0., 0.], (count, 1))
    arrays = {"timestamp": np.arange(1, count + 1) * .002, "source/base_pose": pose.copy(),
              "source/action": np.zeros((count, 15))}
    for side, y in (("left", .2), ("right", -.2)):
        arrays[f"source/{side}_pinch_pose"] = pose.copy()
        arrays[f"source/{side}_pinch_pose"][:, :3] = [.5, y, 1.2]
    for name in TASK_OBJECTS:
        arrays[f"objects/{name}/pose"] = pose.copy()
    arrays["source/action"][:, 13] = [0, 0, 1, 1, 0]
    channels = [{"index": i, "kind": "base_position_target_increment" if i < 3 else "absolute_joint_position_target"} for i in range(13)]
    channels.extend({"index": 13 + i, "kind": "normalized_gripper_command", "name": side + "_gripper", "range": [0, 1],
        "actuators": [f"h1/robotiq_2f85_{side}/fingers_actuator"]} for i, side in enumerate(("left", "right")))
    return {"arrays": arrays, "metadata": {"source_revision": SOURCE_REVISION, "replay_bigym_version": "4.1.0",
        "source_replay_completed": True, "source_task_success": False, "action_channels": channels,
        "source_sequence": "MovePlate/record", "source_group": "bigym/MovePlate/record"},
        "status": "source_replayed", "missing_fields": ["exact recorded BiGym 4.0.0 source revision"]}


class FakeRobot:
    def __init__(self, root=None):
        self.q = np.zeros(1)
        transform = lambda p: NS(rotation=np.eye(3), translation=np.asarray(p))
        frames = [NS(parentJoint=3, placement=transform([.26, 0, 0]), name="l_arm_tip"),
                  NS(parentJoint=5, placement=transform([.26, 0, 0]), name="r_arm_tip")]
        placements = [transform([0, 0, 0]) for _ in range(6)]
        placements[3] = placements[5] = transform([.4, 0, 0])
        model = NS(njoints=6, getJointId=lambda name: {"root_joint": 1, "l_shoulder_pitch": 2, "r_shoulder_pitch": 4}[name],
            frames=frames, names=["universe", "root_joint", "l_shoulder_pitch", "l_wrist", "r_shoulder_pitch", "r_wrist"],
            parents=[0, 0, 1, 2, 1, 4], jointPlacements=placements)
        world = [transform([0, 0, 0]) for _ in range(6)]
        world[2], world[4] = transform([-.01, .2, 1.166]), transform([-.01, -.2, 1.166])
        self.r = NS(model=model, data=NS(oMi=world), hands=[0, 1])

    def fk(self, q):
        return np.tile(np.eye(4), (2, 1, 1))


def test_verified_polarity_maps_commands_without_inventing_contact():
    source = record()
    before = deepcopy(source)
    grippers, metadata = adapter.prepare(source)
    np.testing.assert_array_equal(grippers["left"], [2., 2., -.06, -.06, 2.])
    np.testing.assert_array_equal(grippers["right"], np.full(5, 2.))
    assert metadata["source_task_success"] is False
    assert not metadata["hands"]["left"]["physical_pad_calibration_verified"]
    assert metadata["head_neck_semantics"].endswith("no Reachy neck target inferred")
    for key, value in before["arrays"].items():
        np.testing.assert_array_equal(value, source["arrays"][key])


@pytest.mark.parametrize("change", ["order", "continuous", "revision", "partial", "base_absolute"])
def test_unverified_source_mapping_is_rejected(change):
    source = record()
    if change == "order":
        source["metadata"]["action_channels"][13]["name"] = "right_gripper"
    elif change == "continuous":
        source["arrays"]["source/action"][2, 13] = .5
    elif change == "revision":
        source["metadata"]["source_revision"] = "unknown"
    elif change == "partial":
        source["metadata"]["source_replay_completed"] = False
    else:
        source["metadata"]["action_channels"][0]["kind"] = "absolute_joint_position_target"
    with pytest.raises(ValueError):
        adapter.prepare(source)


def test_reach_bound_proves_failure_without_moving_source_or_optimizing_base():
    source = record()
    source["arrays"]["source/left_pinch_pose"][2, 0] = .8
    before = deepcopy(source["arrays"])
    raw, report = adapter.geometric_preflight(source, FakeRobot())
    left = report["hands"]["left"]
    assert left["maximum_geometric_reach_m"] == pytest.approx(.66)
    assert left["position_error_lower_bound_max_m"] > .15
    assert left["worst_frame_index"] == 2
    assert left["worst_frame_time_s"] == .006
    assert report["proven_infeasible_at_2cm"]
    assert not report["kinematic_passed"] and not report["physics_validated"]
    np.testing.assert_array_equal(raw["source_clock_s"], source["arrays"]["timestamp"])
    for key, value in before.items():
        np.testing.assert_array_equal(source["arrays"][key], value)


def test_wrapper_still_delegates_full_trajectory_after_preflight_failure(tmp_path, monkeypatch):
    import reachy_retarget.robot
    import reachy_retarget.pose_retarget
    monkeypatch.setattr(reachy_retarget.robot, "Robot", FakeRobot)
    source = record()
    source["arrays"]["source/left_pinch_pose"][2, 0] = .8
    calls = []

    def solve(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "kinematic_failed", "frames": len(args[0]["arrays"]["timestamp"]), "physics_validated": False}

    monkeypatch.setattr(reachy_retarget.pose_retarget, "retarget", solve)
    result = adapter.retarget(source, tmp_path, "bigym", "source-episode", tmp_path / "attempt")
    assert result["frames"] == 5
    assert result["geometric_preflight"]["proven_infeasible_at_2cm"]
    assert calls[0][0][0] is source
    assert calls[0][1]["hand_keys"] == adapter.HAND_KEYS
    assert calls[0][1]["clock_key"] == "timestamp"
    assert (tmp_path / "attempt/geometric-preflight.npz").is_file()
    assert (tmp_path / "attempt/geometric-preflight.json").is_file()
    with pytest.raises(FileExistsError):
        adapter.retarget(source, tmp_path, "bigym", "source-episode", tmp_path / "attempt")
