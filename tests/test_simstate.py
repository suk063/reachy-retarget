"""Offline checks for recorded free-body extraction and source sensor frames.

Synthetic tests run without datasets. The final parameterized test also audits
explicitly acquired local files when present; it never downloads missing data.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from reachy_retarget.simstate import free_object_poses


def _simple_xml(extra="", name="cube_main"):
    return (
        '<mujoco><worldbody><body name="' + name + '"><freejoint name="cube_free"/>'
        '</body>' + extra + '</worldbody></mujoco>'
    )


def _one_state():
    # robosuite 1.4.1 and 1.5.1 MjSimState.flatten: time, qpos, qvel.
    return np.array([[0.125, 0.2, 0.3, 0.4, 1., 0., 0., 0., 2., 3., 4., 5., 6., 7.]])


def test_joint_addresses_match_independent_mujoco_compiler():
    mujoco = pytest.importorskip("mujoco")
    xml = """<mujoco>
      <compiler angle="radian"/>
      <default><geom type="sphere" size="0.02" mass="1"/></default>
      <worldbody>
        <body name="unused_before"><freejoint/><geom/></body>
        <body name="arm">
          <body name="wrist"><joint name="ball" type="ball"/><geom/></body>
          <joint name="slide" type="slide" axis="1 0 0"/>
          <joint name="hinge" type="hinge" axis="0 0 1"/><geom/>
        </body>
        <body name="cube_main" pos="7 8 9">
          <freejoint name="cube_free"/><geom/>
        </body>
        <body name="unused_after"><freejoint/><geom/></body>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    recorded, body_poses = [], []
    joint = model.joint("cube_free").id
    address = int(model.jnt_qposadr[joint])
    body = model.body("cube_main").id
    for frame in range(3):
        data.time = frame * 0.05
        angle = frame * 0.3
        data.qpos[address:address + 7] = [
            0.1 + frame, 0.2, 0.3, np.cos(angle / 2), 0, 0, np.sin(angle / 2)
        ]
        data.qvel[:] = 100 + np.arange(model.nv)
        mujoco.mj_forward(model, data)
        recorded.append(np.r_[data.time, data.qpos.copy(), data.qvel.copy()])
        body_poses.append(np.r_[data.xpos[body], data.xquat[body]])
    recorded = np.array(recorded)
    before = recorded.copy()
    arrays, objects, evidence = free_object_poses(xml, recorded, "Lift")
    np.testing.assert_allclose(arrays["objects/cube/pose"], body_poses, atol=1e-14)
    assert objects["cube"]["qpos_address"] == address
    assert evidence["nq"] == model.nq
    assert evidence["nv"] == model.nv
    assert evidence["inactive_free_bodies_omitted"] == ["unused_after", "unused_before"]
    np.testing.assert_array_equal(recorded, before)
    assert not np.shares_memory(arrays["objects/cube/pose"], recorded)


@pytest.mark.parametrize("mutation", [
    lambda x: x[:, :-1],
    lambda x: np.c_[x, np.zeros(len(x))],
    lambda x: x[0],
    lambda x: np.full_like(x, np.nan),
])
def test_malformed_state_layout_is_rejected(mutation):
    with pytest.raises(ValueError, match="Recorded state width"):
        free_object_poses(_simple_xml(), mutation(_one_state()), "Lift")


def test_nonunit_quaternion_is_rejected_without_silent_normalization():
    state = _one_state()
    state[0, 4] = 0.9
    before = state.copy()
    with pytest.raises(ValueError, match="Non-unit"):
        free_object_poses(_simple_xml(), state, "Lift")
    np.testing.assert_array_equal(state, before)


@pytest.mark.parametrize("tag", ["include", "replicate", "attach", "composite", "flexcomp", "frame"])
def test_unexpanded_structure_is_rejected(tag):
    xml = _simple_xml().replace("<worldbody>", f"<{tag}/><worldbody>")
    with pytest.raises(ValueError, match="Unexpanded XML"):
        free_object_poses(xml, _one_state(), "Lift")


def test_default_joint_type_is_not_guessed():
    xml = _simple_xml().replace("<worldbody>", '<default><joint type="ball"/></default><worldbody>')
    with pytest.raises(ValueError, match="Joint-type defaults"):
        free_object_poses(xml, _one_state(), "Lift")


@pytest.mark.parametrize("change", [
    lambda xml: xml.replace("<worldbody>", '<compiler alignfree="true"/><worldbody>'),
    lambda xml: xml.replace('<freejoint name="cube_free"', '<freejoint align="true" name="cube_free"'),
])
def test_compiler_frame_alignment_is_not_misread_as_original_body_pose(change):
    with pytest.raises(ValueError, match="Aligned free-joint frames"):
        free_object_poses(change(_simple_xml()), _one_state(), "Lift")


def test_body_identity_is_not_selected_from_an_object_tail_guess():
    with pytest.raises(ValueError, match="Required object bodies missing"):
        free_object_poses(_simple_xml(name="unrelated"), _one_state(), "Lift")
    with pytest.raises(ValueError, match="Unverified object schema"):
        free_object_poses(_simple_xml(), _one_state(), "UnverifiedTask")


def test_duplicate_canonical_object_ids_are_rejected():
    xml = _simple_xml('<body name="cube_root"><freejoint/></body>')
    with pytest.raises(ValueError, match="ambiguous canonical"):
        free_object_poses(xml, _one_state(), "Lift")


def test_nested_free_body_is_rejected():
    xml = '<mujoco><worldbody><body name="parent"><body name="cube_main"><freejoint/></body></body></worldbody></mujoco>'
    with pytest.raises(ValueError, match="direct child"):
        free_object_poses(xml, _one_state(), "Lift")


def test_source_files_in_one_directory_keep_distinct_episode_identity(tmp_path):
    from reachy_retarget.episodes import read_episode
    from reachy_retarget.normalize import robomimic
    from reachy_retarget.store import Store

    source = "hf__amandlek__mimicgen_datasets"
    directory = tmp_path / "data/raw" / source / "source"
    directory.mkdir(parents=True)
    for filename, env_name, objects in [
        ("stack", "Stack_D0", ["cubeA", "cubeB"]),
        ("threading", "Threading_D0", ["needle_obj", "tripod_obj"]),
    ]:
        xml = '<mujoco><worldbody>' + ''.join(
            f'<body name="{name}"><freejoint/></body>' for name in objects
        ) + '</worldbody></mujoco>'
        states = np.zeros((2, 27))  # time + 2*(7 qpos) + 2*(6 qvel)
        states[:, [4, 11]] = 1
        states[1, 0] = 0.05
        with h5py.File(directory / f"{filename}.hdf5", "w") as handle:
            data = handle.create_group("data")
            data.attrs["env_args"] = json.dumps({
                "env_name": env_name, "env_version": "1.4.1",
                "env_kwargs": {"control_freq": 20},
            })
            demo = data.create_group("demo_0")
            demo.attrs["model_file"] = xml
            demo["states"] = states
            demo["actions"] = np.zeros((2, 7))
            demo.create_group("obs")
    paths = robomimic(Store(tmp_path), source, 1)
    assert len(paths) == len(set(paths)) == 2
    records = [read_episode(path)[1] for path in paths]
    assert {record["source_sequence"] for record in records} == {
        "source/stack/demo_0", "source/threading/demo_0"
    }
    assert len({record["source_group"] for record in records}) == 2


# These are independently documented *observation* slices, not qpos offsets.
# Source: robosuite/environments/manipulation/{lift,nut_assembly,tool_hang,
# two_arm_transport,pick_place}.py at v1.5.1; stack.py at v1.4.1;
# NVlabs/mimicgen/mimicgen/envs/robosuite/threading.py _setup_references and
# _create_obj_sensors. Pose sensors use xyz+xyzw; simulator qpos uses xyz+wxyz.
# ToolHang stand_base is a geom and frame_intersection_site is a site, so only
# its tool body is compared directly. Matrix-derived quaternions may flip sign.
_CASES = [
    ("hf__robomimic__robomimic_datasets/v1.5/can/ph/low_dim_v15.hdf5", "PickPlaceCan", "1.5.1", 14, {"Can": 7}),
    ("hf__robomimic__robomimic_datasets/v1.5/lift/ph/low_dim_v15.hdf5", "Lift", "1.5.1", 10, {"cube": 0}),
    ("hf__robomimic__robomimic_datasets/v1.5/square/ph/low_dim_v15.hdf5", "NutAssemblySquare", "1.5.1", 14, {"SquareNut": 7}),
    ("hf__robomimic__robomimic_datasets/v1.5/tool_hang/ph/low_dim_v15.hdf5", "ToolHang", "1.5.1", 44, {"tool": 35}),
    ("hf__robomimic__robomimic_datasets/v1.5/transport/ph/low_dim_v15.hdf5", "TwoArmTransport", "1.5.1", 41, {"payload": 0, "trash": 7}),
    ("hf__amandlek__mimicgen_datasets/source/stack.hdf5", "Stack_D0", "1.4.1", 23, {"cubeA": 0, "cubeB": 7}),
    ("hf__amandlek__mimicgen_datasets/source/threading.hdf5", "Threading_D0", "1.4.1", 28, {"needle_obj": 0, "tripod_obj": 14}),
]


@pytest.mark.parametrize("relative,env_name,version,observation_width,slices", _CASES,
                         ids=[case[1] for case in _CASES])
def test_acquired_object_poses_agree_with_source_observation_sensors(
    relative, env_name, version, observation_width, slices
):
    path = Path(__file__).resolve().parents[1] / "runs/object-matrix/data/raw" / relative
    if not path.is_file():
        pytest.skip("Optional, explicitly acquired dataset is absent; no download is attempted")
    with h5py.File(path, "r") as handle:
        env = json.loads(handle["data"].attrs["env_args"])
        assert env["env_name"] == env_name
        assert env["env_version"] == version
        for episode in ("demo_0", "demo_1", "demo_2"):
            demo = handle[f"data/{episode}"]
            arrays, _, _ = free_object_poses(demo.attrs["model_file"], demo["states"][()], env_name)
            observed = demo["obs/object"][()]
            assert observed.shape[1] == observation_width
            for object_name, start in slices.items():
                pose = arrays[f"objects/{object_name}/pose"]
                expected = observed[:, start:start + 7]
                np.testing.assert_allclose(pose[:, :3], expected[:, :3], atol=1e-7, rtol=0)
                quaternion = expected[:, [6, 3, 4, 5]]
                residual = np.minimum(
                    np.linalg.norm(pose[:, 3:] - quaternion, axis=1),
                    np.linalg.norm(pose[:, 3:] + quaternion, axis=1),
                )
                assert np.max(residual) < 1e-6, (episode, object_name, np.max(residual))
