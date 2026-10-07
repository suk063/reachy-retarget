"""Offline native-format and evidence-boundary checks for mobile adapters."""

import csv
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget.adapters import m3bench, mello


def write_vicon(path, *, unit="mm", object_prefix="Box", subframe=0):
    names = ["Subject:LWRA"] + [object_prefix + ":" + label for label in ("BFBR", "BFBL", "BFTL", "BFTR", "BBR", "BBL")]
    labels = ["", ""] + [v for name in names for v in (name, "", "")]
    points = [[1000, 2000, 3000], [0, 0, 1000], [100, 0, 1000], [0, 100, 1000],
              [100, 100, 1000], [0, 0, 1100], [100, 0, 1100]]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows([
            ["Trajectories"], [100], labels,
            ["Frame", "Sub Frame"] + [v for _ in names for v in ("X", "Y", "Z")],
            ["", ""] + [unit] * (3 * len(names)),
            [2198, subframe] + [v for point in points for v in point],
            [2200, subframe] + ["", "", ""] + [v for point in points[1:] for v in point],
        ])
    return path


def native_m3bench(tmp_path):
    q = np.zeros((3, 10))
    q[:, 0] = [1, 1.1, 1.2]
    q[:, 1] = 2
    config = {"env": {"scene": {"name": "scene"}, "agent": {"name": "Mec_kinova", "position": [1, 2, 0]},
                       "object": {"name": "book", "transformation_matrix": np.eye(4).tolist()}},
              "attachments": {"given_attach_trans": [0, 0, 0]}}
    trajectory = {"trajectory": q.tolist(), "action_type": "pick", "link_name": "book", "caption": "Pick book"}
    result = {name: q[:, i].tolist() for i, name in enumerate(m3bench.JOINT_NAMES)}
    result.update(joint_names=m3bench.JOINT_NAMES, n_step=3, converged=True)
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "pick_vkc_caption_trajectory.json").write_text(json.dumps(trajectory))
    (tmp_path / "pick_vkc_return.json").write_text(json.dumps(result))
    (tmp_path / "vkc_request.json").write_text(json.dumps({"vkc_env": {"collision_scale": {"book": [.001] * 3}}}))
    return tmp_path


def test_mello_clock_units_and_occlusion_are_preserved(tmp_path):
    path = write_vicon(tmp_path / "vicon.csv")
    result = mello.normalize(path)
    np.testing.assert_allclose(result["arrays"]["time_s"], [0, .02])
    np.testing.assert_array_equal(result["arrays"]["source/frame_index"], [2198, 2200])
    np.testing.assert_allclose(result["arrays"]["markers/position_m"][0, 0], [1, 2, 3])
    assert np.isnan(result["arrays"]["markers/position_m"][1, 0]).all()
    assert not result["arrays"]["markers/valid"][1, 0]
    assert not any(key.startswith("observation/") for key in result["arrays"])
    assert result["status"] == "normalized_partial"
    assert result["metadata"]["physics_validated"] is False
    assert not result["metadata"]["source_revision_verified_for_pilot"]
    assert result["metadata"]["task_object_pose_coverage"]["all_task_object_poses_saved"] is False
    assert result["metadata"]["task_object_pose_coverage"]["all_native_marker_observations_saved"] is True


@pytest.mark.parametrize("unit,subframe", [("cm", 0), ("mm", 1)])
def test_mello_rejects_unverified_units_or_subframe_clock(tmp_path, unit, subframe):
    with pytest.raises(ValueError):
        mello.normalize(write_vicon(tmp_path / "bad.csv", unit=unit, subframe=subframe))


def test_mello_rejects_human_only_and_lfs_pointer(tmp_path):
    with pytest.raises(ValueError, match="object-interaction"):
        mello.normalize(write_vicon(tmp_path / "gait.csv", object_prefix="Subject"))
    path = tmp_path / "pointer.csv"
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 200\n")
    with pytest.raises(ValueError, match="LFS pointer"):
        mello.normalize(path)


def test_marker_fit_recovers_motion_without_filling_bad_frames():
    reference = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]])
    rotation = Rotation.from_euler("z", .7).as_matrix()
    observed = reference @ rotation.T + [1, 2, 3]
    values = np.stack([reference, observed, observed.copy(), observed.copy()])
    valid = np.ones((4, 4), bool)
    values[1, 0] = np.nan
    valid[1, 0] = False
    values[2, 1] += [.1, -.2, .2]
    valid[3, :2] = False
    values[3, :2] = np.nan
    pose, residual, accepted, first = mello.fit_marker_frame(values, valid)
    np.testing.assert_array_equal(accepted, [True, True, False, False])
    np.testing.assert_allclose(pose[1, :3], reference.mean(axis=0) @ rotation.T + [1, 2, 3])
    np.testing.assert_allclose(Rotation.from_quat(pose[1, [4, 5, 6, 3]]).as_matrix(), rotation, atol=1e-12)
    assert first == 0 and residual[2] > .005
    assert np.isnan(pose[2:]).all()


def test_collinear_marker_constellation_never_becomes_object_pose():
    values = np.array([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]] * 2, float)
    pose, _, accepted, first = mello.fit_marker_frame(values, np.ones((2, 3), bool))
    assert first is None and not accepted.any() and np.isnan(pose).all()


def test_m3bench_preserves_planner_data_without_invented_time_or_objects(tmp_path):
    result = m3bench.normalize(native_m3bench(tmp_path))
    assert "time_s" not in result["arrays"]
    assert not any("object" in key and "trajectory" in key for key in result["arrays"])
    assert result["arrays"]["source/object_initial_transform"].shape == (4, 4)
    assert result["metadata"]["planner_converged"] is True
    assert result["metadata"]["physics_validated"] is False
    assert result["metadata"]["source_planner_collision_scale"] == {"book": [.001] * 3}
    assert result["metadata"]["simulation_assumptions"] == []
    assert "continuous_object_pose_and_contact_forces" in result["missing_fields"]
    assert result["metadata"]["task_object_pose_coverage"]["all_task_object_poses_saved"] is False
    assert result["metadata"]["task_object_pose_coverage"]["continuous_object_ids"] == []


def test_m3bench_crosschecks_named_joint_trajectory(tmp_path):
    root = native_m3bench(tmp_path)
    path = root / "pick_vkc_return.json"
    native = json.loads(path.read_text())
    native["joint_1"][1] = 2
    path.write_text(json.dumps(native))
    with pytest.raises(ValueError, match="differ from the named VKC"):
        m3bench.normalize(root)


@pytest.mark.parametrize("bad", ["scale", "wrong_object", "wrong_base"])
def test_m3bench_rejects_invalid_initial_contract(tmp_path, bad):
    root = native_m3bench(tmp_path)
    path = root / "config.json"
    config = json.loads(path.read_text())
    if bad == "scale":
        config["env"]["object"]["transformation_matrix"][0][0] = .1
    elif bad == "wrong_object":
        config["env"]["object"]["name"] = "other"
    else:
        config["env"]["agent"]["position"][0] = 50
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        m3bench.normalize(root)


def test_m3bench_requires_paired_object_configuration(tmp_path):
    (tmp_path / "pick_vkc_caption_trajectory.json").write_text('{"trajectory": [[0, 1]]}')
    with pytest.raises(ValueError, match="config.json"):
        m3bench.normalize(tmp_path)


def test_descriptions_are_json_and_ranges_are_bounded():
    for adapter in (m3bench, mello):
        json.dumps(adapter.describe())
        assert adapter.describe()["physics_validated"] is False
    pilot = m3bench.describe()["pilot"]
    assert sum(x["compressed_bytes"] for x in pilot["members"]) < 20000
    assert all(x["offset"] + x["compressed_bytes"] <= pilot["archive_bytes"] for x in pilot["members"])
    assert m3bench.describe()["source_replay"]["ready_for_unattended_install"] is False


def test_untimed_m3bench_uses_shared_archive_without_fabricated_clock(tmp_path):
    from reachy_retarget.agent_dataset import export_normalized, inspect_archive
    native = tmp_path / "native"
    native.mkdir()
    result = m3bench.normalize(native_m3bench(native))
    path = export_normalized(result, tmp_path / "out", "m3bench", "synthetic-native")
    report = inspect_archive(path)
    assert report["rows"] == 3 and report["timed"] is False
    assert report["policy_ready"] is False
    assert "timestamp" in report["missing_fields"] and "timestamp" not in report["fields"]
    assert "source/waypoint_index" in report["fields"]


def test_missing_mello_marker_fits_remain_masked_in_shared_archive(tmp_path):
    from reachy_retarget.agent_dataset import export_normalized, inspect_archive
    path = write_vicon(tmp_path / "vicon.csv")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    rows[6][5:17] = [""] * 12  # Only two Box markers observed in frame two.
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        csv.writer(stream).writerows(rows)
    result = mello.normalize(path)
    assert not result["arrays"]["objects/Box/marker_frame_valid"][1]
    assert np.isnan(result["arrays"]["objects/Box/marker_frame_pose_wxyz"][1]).all()
    exported = export_normalized(result, tmp_path / "out", "mello", "synthetic-occlusion")
    report = inspect_archive(exported)
    assert report["timed"] is True and report["policy_ready"] is False
    assert report["fields"]["observation/objects/Box/marker_frame_pose_wxyz"]["nonfinite_count"] == 7


def test_m3bench_optional_fk_is_derived_from_named_chain(tmp_path):
    root = native_m3bench(tmp_path)
    links = ["world"] + ["link_" + str(i) for i in range(10)] + ["robotiq_arg2f_base_link"]
    xml = ["<robot name='test'>"] + [f"<link name='{link}'/>" for link in links]
    for index, name in enumerate(m3bench.JOINT_NAMES):
        kind = "prismatic" if index < 2 else "revolute"
        axis = ("1 0 0", "0 1 0")[index] if index < 2 else "0 0 1"
        xml.append(f"<joint name='{name}' type='{kind}'><parent link='{links[index]}'/><child link='{links[index + 1]}'/><axis xyz='{axis}'/></joint>")
    xml.append("<joint name='tcp' type='fixed'><parent link='link_9'/><child link='robotiq_arg2f_base_link'/><origin xyz='.2 0 .5'/></joint></robot>")
    (root / "robot.urdf").write_text("".join(xml))
    result = m3bench.normalize(root)
    matrices = result["arrays"]["source/ee_transform"]
    np.testing.assert_allclose(matrices[:, :3, 3], [[1.2, 2, .5], [1.3, 2, .5], [1.4, 2, .5]])
    assert "source/ee_transform" in result["metadata"]["derived_fields"]
    assert "timestamp" in result["missing_fields"]
    assert not any(key.startswith("observation/") for key in result["arrays"])
