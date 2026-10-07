import json

import h5py
import numpy as np
import pytest

from reachy_retarget.adapters import behavior, momagen


def native_file(path, enriched=True, extra=False):
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["config"] = json.dumps({"env": {"action_frequency": 30},
                                         "robots": [{"name": "robot_r1", "grasping_mode": "assisted"}],
                                         "task": {"activity_name": "put_away"}})
        data.attrs["scene_file"] = json.dumps({"versions": {"omnigibson": {"git_hash": "recorded"}}})
        group = data.create_group("demo_0")
        group.create_dataset("state", data=np.zeros((4, 13)))
        group.create_dataset("state_size", data=np.full(4, 13, dtype=int))
        group.create_dataset("action", data=np.zeros((3, 21)))
        group.create_dataset("reward", data=np.zeros(3))
        group.create_dataset("terminated", data=np.zeros(3, dtype=bool))
        group.create_dataset("truncated", data=np.zeros(3, dtype=bool))
        if enriched:
            info = group.create_group("datagen_info")
            info.attrs["env_interface_type"] = "omnigibson_bimanual"
            left = np.tile(np.eye(4), (3, 1, 1)); left[:, 0, 3] = np.arange(3)
            right = left.copy(); right[:, 1, 3] = 2
            info.create_dataset("eef_pose", data=np.concatenate([left, right], axis=1))
            info.create_dataset("base_pose", data=left)
            info.create_dataset("gripper_action", data=np.tile([-1., 1.], (3, 1)))
            for name in ["cup", "robot_r1", "torso_link4"]:
                info.create_dataset("object_poses/" + name, data=right)
        if extra:
            raw = data.create_group("demo_1")
            for name in ["state", "state_size", "action", "reward", "terminated", "truncated"]:
                raw.create_dataset(name, data=group[name][()])
    return path


def test_behavior_preserves_t_plus_one_and_reports_blocker(tmp_path):
    path = native_file(tmp_path / "raw.hdf5", enriched=False)
    result = behavior.normalize(path)
    assert result["status"] == "blocked_native_state_decode"
    assert result["arrays"]["source/state"].shape == (4, 13)
    assert result["arrays"]["source/action"].shape == (3, 21)
    np.testing.assert_allclose(result["arrays"]["time_s"], np.arange(4) / 30)
    assert not any(k.startswith("objects/") for k in result["arrays"])
    assert result["metadata"]["recorded_versions"]["omnigibson"]["git_hash"] == "recorded"
    assert result["metadata"]["source_revision"] is None
    assert result["metadata"]["source_grasping_modes"] == ["assisted"]
    assert not result["metadata"]["physics_validated"]


def test_momagen_exact_world_poses_commands_and_robot_roles(tmp_path):
    path = native_file(tmp_path / "r1_test.hdf5", extra=True)
    before = path.read_bytes()
    report = momagen.inspect(path)
    assert report["enriched_episodes"] == ["demo_0"]
    assert report["unannotated_episodes"] == ["demo_1"]
    result = momagen.normalize(path)
    assert result["status"] == "normalized_source_reference"
    arrays = result["arrays"]
    np.testing.assert_array_equal(arrays["hand/left_pose"][:, 1], 0)
    np.testing.assert_array_equal(arrays["hand/right_pose"][:, 1], 2)
    np.testing.assert_array_equal(arrays["hand/right_pose"][:, 3:], np.tile([1, 0, 0, 0], (3, 1)))
    np.testing.assert_array_equal(arrays["source/gripper_action"], np.tile([-1, 1], (3, 1)))
    assert set(result["metadata"]["objects"]) == {"cup"}
    assert set(result["metadata"]["excluded_robot_link_roles"]) == {"robot_r1", "torso_link4"}
    assert arrays["source/state"].shape[0] == 4
    assert result["metadata"]["source_group"] == "momagen/r1_test/demo_0"
    assert path.read_bytes() == before


def test_momagen_raw_attempt_retained_without_fabricated_pose(tmp_path):
    path = native_file(tmp_path / "raw.hdf5", extra=True)
    result = momagen.normalize(path, "demo_1")
    assert result["status"] == "blocked_native_state_decode"
    assert result["metadata"]["source_group"] == "momagen/raw/demo_1"
    assert not any(k.startswith("hand/") for k in result["arrays"])


@pytest.mark.parametrize("corruption", ["nan", "scale", "reflection", "homogeneous", "interface"])
def test_invalid_recorded_transforms_fail_closed(tmp_path, corruption):
    path = native_file(tmp_path / "source.hdf5")
    with h5py.File(path, "r+") as f:
        group = f["data/demo_0/datagen_info"]
        if corruption == "interface":
            group.attrs["env_interface_type"] = "unknown"
        else:
            values = group["eef_pose"][()]
            if corruption == "nan": values[1, 0, 0] = np.nan
            if corruption == "scale": values[1, 0, 0] = 2
            if corruption == "reflection": values[1, 0, 0] = -1
            if corruption == "homogeneous": values[1, 3, 3] = 0
            group["eef_pose"][...] = values
    with pytest.raises(ValueError):
        momagen.normalize(path)


def test_no_default_frequency_or_guessed_serialized_offsets(tmp_path):
    path = native_file(tmp_path / "source.hdf5", enriched=False)
    with h5py.File(path, "r+") as f:
        f["data"].attrs["config"] = json.dumps({"env": {}, "robots": []})
    with pytest.raises(ValueError, match="action_frequency"):
        behavior.normalize(path)


def test_pinned_plans_have_no_duplicate_view_counts():
    plans = momagen.describe()
    assert len(plans["fetch"]) == 6
    assert sum(row["reference_frames"] for row in plans["fetch"]) == 12528
    assert all("processed_source_demos" in row["path"] for row in plans["fetch"])
    assert all(len(row["git_blob_sha1"]) == 40 for row in plans["fetch"])
    assert behavior.describe()["fetch"][0]["sha256"] == behavior.PILOT["sha256"]
    plans["fetch"].clear()
    assert len(momagen.describe()["fetch"]) == 6


def test_behavior_task_object_roles_preserved_without_poses(tmp_path):
    path = native_file(tmp_path / "source.hdf5", enriched=False)
    with h5py.File(path, "r+") as f:
        f["data"].attrs["scene_file"] = json.dumps({
            "objects_info": {"init_info": {
                "cup": {"class_name": "DatasetObject", "args": {"category": "cup", "scale": [1, 1, 1]}},
                "robot_r1": {"class_module": "omnigibson.robots.r1", "class_name": "R1"},
            }}, "metadata": {"task": {"inst_to_name": {"cup.n.01_1": "cup", "agent.n.01_1": "robot_r1"}}},
        })
    result = behavior.normalize(path)
    assert set(result["metadata"]["objects"]) == {"cup"}
    assert result["metadata"]["objects"]["cup"]["decoded_pose_available"] is False
    assert result["metadata"]["objects"]["cup"]["source_roles"] == ["cup.n.01_1"]
    assert not any(k.startswith("objects/") for k in result["arrays"])


def test_source_replay_callback_only_reads_and_converts_xyzw():
    from types import SimpleNamespace

    class Body:
        def get_position_orientation(self):
            return np.array([1, 2, 3.]), np.array([0, 0, 0, 1.])

        def get_joint_positions(self):
            return np.array([.2])

    body = Body()
    robot = Body()
    robot.name = "robot"
    robot.eef_links = {"left": Body(), "right": Body()}
    scene = SimpleNamespace(object_registry=lambda kind, name: body if kind == "name" and name == "door" else None)
    env = SimpleNamespace(robots=[robot], scene=scene)
    frame = behavior.capture_replay_frame(env, ["door"])
    np.testing.assert_array_equal(frame["objects/door/pose"], [1, 2, 3, 1, 0, 0, 0])
    np.testing.assert_array_equal(frame["objects/door/joint_position"], [.2])
    with pytest.raises(ValueError, match="Robot cannot"):
        behavior.capture_replay_frame(env, ["robot"])


def test_only_declared_task_object_poses_are_required():
    config = {"robots": [{"name": "robot_r1"}]}
    scene = {"objects_info": {"init_info": {"cup": {}, "cabinet": {}, "robot_r1": {}}},
             "metadata": {"task": {"inst_to_name": {"cup.n.01_1": "cup", "agent.n.01_1": "robot_r1"}}}}
    coverage = behavior.object_coverage(scene, config, ["cup"])
    assert coverage["scene_object_count"] == 2
    assert coverage["scene_pose_completeness_required"] is False
    assert coverage["missing_task_object_pose_names"] == []
    assert coverage["all_task_object_poses_decoded"] is True
    del scene["metadata"]
    no_roles = behavior.object_coverage(scene, config, ["cup", "cabinet"])
    assert no_roles["task_scope_available"] is False
    assert no_roles["all_task_object_poses_decoded"] is False
    assert behavior.object_coverage({}, config, ["cup"])["all_task_object_poses_decoded"] is False
    declared = behavior.object_coverage(scene, config, ["cup"], selected_names=["cup", "cabinet"])
    assert declared["missing_task_object_pose_names"] == ["cabinet"]
    assert declared["all_task_object_poses_decoded"] is False


def test_momagen_checks_task_refs_and_preserves_support_context():
    config = {"robots": [{"name": "robot_r1"}]}
    scene = {"objects_info": {"init_info": {"coffee_cup_7": {}, "breakfast_table_6": {}, "lamp": {}, "floor": {}}},
             "metadata": {"task": {"inst_to_name": {"cup.n.01_1": "coffee_cup_7", "floor.n.01_1": "floor"}}}}
    coverage = momagen._coverage(scene, config, ["coffee_cup_7", "breakfast_table_6"], "r1_pick_cup")
    assert coverage["task_object_names"] == ["breakfast_table_6", "coffee_cup_7"]
    assert coverage["all_task_object_poses_decoded"] is True
    assert coverage["excluded_fixed_ground_pose_names"] == ["floor"]
    assert "lamp" not in coverage["task_object_names"]
    missing = momagen._coverage(scene, config, ["breakfast_table_6"], "r1_pick_cup")
    assert missing["missing_manipulation_reference_pose_names"] == ["coffee_cup_7"]
    assert missing["all_task_object_poses_decoded"] is False
