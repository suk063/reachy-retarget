import json

import h5py
import mujoco
import numpy as np
import pytest

from reachy_retarget.dynamics_audit import verify
from reachy_retarget.store import json_write, sha256


def falling_object(tmp_path, extra=""):
    xml = ('<mujoco><option timestep=".002"/><worldbody>'
           '<geom type="plane" size="2 2 .1"/><body name="object" pos="0 0 1">'
           '<freejoint name="free"/><geom type="box" size=".02 .02 .02" mass=".1"/>'
           '</body></worldbody>'+extra+'</mujoco>')
    path=tmp_path/"scene.xml";path.write_text(xml)
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    initial={"qpos":data.qpos.copy(),"qvel":data.qvel.copy(),"ctrl":data.ctrl.copy()}
    records=[]
    for _ in range(80):
        for _ in range(5):mujoco.mj_step(model,data)
        mujoco.mj_forward(model,data)
        records.append((data.time,data.qpos.copy(),data.qvel.copy(),data.ctrl.copy(),
                        np.r_[data.xpos[1],data.xquat[1]]))
    with h5py.File(tmp_path/"replay.h5","w") as f:
        for name,value in initial.items():f["initial/"+name]=value
        for col,key in enumerate(("time_s","simulation/qpos","simulation/qvel",
                                  "simulation/actuator_control","objects/cube/pose")):
            f[key]=np.asarray([r[col] for r in records])
    json_write(tmp_path/"result.json",{"scene_sha256":sha256(path),
               "physics":{"objects":{"cube":{"body":"object","joint":"free"}}}})
    return tmp_path


def test_actuator_replay_matches_unconstrained_free_fall(tmp_path):
    result=verify(falling_object(tmp_path))
    assert result["actuator_replay_pass"]
    assert result["max_object_pose_error"] == 0
    with h5py.File(tmp_path/"replay.h5") as f:
        assert f["objects/cube/pose"][-1,2] < .1


def test_saved_pose_teleport_does_not_pass_control_replay(tmp_path):
    falling_object(tmp_path)
    with h5py.File(tmp_path/"replay.h5","r+") as f:
        f["objects/cube/pose"][20,0] = .1
    result=verify(tmp_path)
    assert not result["actuator_replay_pass"]
    assert result["max_object_pose_error"] >= .1


@pytest.mark.parametrize("extra,reason",[
    ('<equality><weld body1="object"/></equality>',"constraints"),
    ('<actuator><motor joint="free" gear="1 0 0 0 0 0"/></actuator>',"actuator"),
])
def test_object_assistance_is_rejected(tmp_path,extra,reason):
    falling_object(tmp_path,extra)
    with pytest.raises(ValueError,match=reason):verify(tmp_path)


def test_initial_object_cannot_be_moved_away_from_source_scene(tmp_path):
    falling_object(tmp_path)
    with h5py.File(tmp_path/"replay.h5","r+") as f:
        f["initial/qpos"][0] = .2
    with pytest.raises(ValueError,match="Initial object state"):
        verify(tmp_path)


@pytest.mark.parametrize("key",["initial/qpos","simulation/qvel","objects/cube/pose"])
def test_nan_records_cannot_silently_count_as_zero_error(tmp_path,key):
    falling_object(tmp_path)
    with h5py.File(tmp_path/"replay.h5","r+") as f:
        values=f[key][()];values.flat[0]=np.nan;f[key][...]=values
    with pytest.raises(ValueError,match="Nonfinite"):
        verify(tmp_path)


@pytest.mark.parametrize("key,bad_shape", [
    ("initial/qpos", (7, 1)), ("initial/qvel", (1, 6)), ("initial/ctrl", (0, 1)),
    ("simulation/qpos", (80, 8)), ("simulation/qvel", (79, 6)),
    ("simulation/actuator_control", (81, 0)), ("objects/cube/pose", (80, 8)),
    ("time_s", (80, 1)),
])
def test_recording_dimensions_must_match_model_and_timeline(tmp_path, key, bad_shape):
    falling_object(tmp_path)
    with h5py.File(tmp_path / "replay.h5", "r+") as f:
        del f[key]
        f[key] = np.zeros(bad_shape)
    with pytest.raises(ValueError, match="shape mismatch|one-dimensional"):
        verify(tmp_path)


@pytest.mark.parametrize("bad", [np.array([]), np.array([np.nan]), np.array([np.inf]),
                                    np.array([0.]), np.array([.02, .01])])
def test_empty_or_invalid_timeline_is_rejected(tmp_path, bad):
    falling_object(tmp_path)
    with h5py.File(tmp_path / "replay.h5", "r+") as f:
        del f["time_s"]
        f["time_s"] = bad
    with pytest.raises(ValueError, match="timeline|strictly increasing"):
        verify(tmp_path)


def _set_report(tmp_path, **fields):
    path = tmp_path / "result.json"
    report = json.loads(path.read_text())
    report.update(fields)
    json_write(path, report)
    return report


def _remove_frame(tmp_path, index):
    with h5py.File(tmp_path / "replay.h5", "r+") as f:
        for key in ["time_s", "simulation/qpos", "simulation/qvel",
                    "simulation/actuator_control", "objects/cube/pose"]:
            array = np.delete(f[key][()], index, axis=0)
            del f[key]
            f[key] = array


def test_missing_interior_frame_cannot_hide_behind_matching_array_lengths(tmp_path):
    falling_object(tmp_path)
    _remove_frame(tmp_path, 12)
    with pytest.raises(ValueError, match="incomplete control intervals"):
        verify(tmp_path)


def test_truncated_last_frame_is_bound_to_reported_simulation_duration(tmp_path):
    falling_object(tmp_path)
    _set_report(tmp_path, simulated_s=.8)
    _remove_frame(tmp_path, -1)
    with pytest.raises(ValueError, match="simulated_s"):
        verify(tmp_path)


def test_complete_recording_can_check_optional_reported_duration(tmp_path):
    falling_object(tmp_path)
    _set_report(tmp_path, simulated_s=.8, recorded_frames=80)
    result = verify(tmp_path)
    assert result["actuator_replay_pass"] and result["reported_duration_checked"]


@pytest.mark.parametrize("duration", [np.nan, np.inf, -.8, .81])
def test_invalid_reported_duration_is_rejected(tmp_path, duration):
    falling_object(tmp_path)
    _set_report(tmp_path, simulated_s=duration)
    with pytest.raises(ValueError, match="simulated_s"):
        verify(tmp_path)


def test_optional_channels_must_have_complete_frame_counts(tmp_path):
    falling_object(tmp_path)
    with h5py.File(tmp_path / "replay.h5", "r+") as f:
        f["metrics/task_satisfied"] = np.ones(79, dtype=bool)
    with pytest.raises(ValueError, match="frame count"):
        verify(tmp_path)


def test_pregrasp_nan_drift_is_allowed_but_not_nan_state(tmp_path):
    falling_object(tmp_path)
    with h5py.File(tmp_path / "replay.h5", "r+") as f:
        f["metrics/grasp_drift_m_rad"] = np.full((80, 2), np.nan)
    assert verify(tmp_path)["actuator_replay_pass"]


def _scene_with_descendant(tmp_path, *, tail="", child_attributes="", child_joint="child_hinge"):
    xml = f'''<mujoco><option timestep=".002"/><worldbody>
      <body name="anchor"><geom size=".01"/><site name="anchor_site"/></body>
      <body name="unrelated"><joint name="unrelated_joint"/><geom size=".01" mass=".1"/></body>
      <body name="object" pos="0 0 1"><freejoint name="free"/><geom size=".03" mass=".1"/>
        <body name="child" pos="0 0 .1" {child_attributes}>
          <joint name="{child_joint}"/><geom size=".01" mass=".01"/><site name="child_site"/>
        </body>
      </body>
      </worldbody>{tail}</mujoco>'''
    path = tmp_path / "scene.xml"
    path.write_text(xml)
    json_write(tmp_path / "result.json", {"scene_sha256": sha256(path),
               "physics": {"objects": {"cube": {"body": "object", "joint": "free"}}}})
    return xml


@pytest.mark.parametrize("tail,reason", [
    ('<actuator><motor joint="child_hinge"/></actuator>', "actuator"),
    ('<actuator><motor joint="unrelated_joint"/></actuator>', "actuator"),
    ('<actuator><general site="child_site" gear="1 0 0 0 0 0"/></actuator>', "actuator"),
    ('<actuator><adhesion body="child" gain="1" ctrlrange="0 1"/></actuator>', "actuator"),
    ('<tendon><fixed name="t"><joint joint="child_hinge" coef="1"/></fixed></tendon>'
     '<actuator><motor tendon="t"/></actuator>', "actuator"),
    ('<equality><weld body1="child" body2="anchor"/></equality>', "constraints"),
    ('<equality><joint joint1="child_hinge"/></equality>', "constraints"),
    ('<equality><connect site1="child_site" site2="anchor_site"/></equality>', "constraints"),
    ('<equality><weld site1="child_site" site2="anchor_site"/></equality>', "constraints"),
    ('<tendon><fixed name="t" stiffness="1"><joint joint="child_hinge" coef="1"/></fixed></tendon>', "tendon"),
    ('<tendon><spatial name="t" stiffness="1"><site site="anchor_site"/><site site="child_site"/></spatial></tendon>', "tendon"),
])
def test_descendant_and_indirect_object_assistance_is_rejected(tmp_path, tail, reason):
    _scene_with_descendant(tmp_path, tail=tail)
    with pytest.raises(ValueError, match=reason):
        verify(tmp_path)


def test_robot_joint_name_cannot_disguise_actuation_on_object_descendant(tmp_path):
    _scene_with_descendant(tmp_path, child_joint="r_hand_finger",
                          tail='<actuator><motor joint="r_hand_finger"/></actuator>')
    with pytest.raises(ValueError, match="descendant"):
        verify(tmp_path)


def test_descendant_gravity_compensation_is_rejected(tmp_path):
    _scene_with_descendant(tmp_path, child_attributes='gravcomp="1"')
    with pytest.raises(ValueError, match="gravity compensation"):
        verify(tmp_path)


def test_only_approved_robot_joint_transmissions_are_allowed(tmp_path):
    from reachy_retarget.dynamics_audit import _verify_no_object_assistance
    xml = _scene_with_descendant(tmp_path).replace('name="unrelated_joint"', 'name="base_x"')
    xml = xml.replace('</mujoco>', '<actuator><motor joint="base_x"/></actuator></mujoco>')
    model = mujoco.MjModel.from_xml_string(xml)
    assert _verify_no_object_assistance(model, {"cube": {"body": "object", "joint": "free"}})


def test_mocap_flag_on_descendant_is_rejected(tmp_path):
    from reachy_retarget.dynamics_audit import _verify_no_object_assistance
    model = mujoco.MjModel.from_xml_string(_scene_with_descendant(tmp_path))
    model.body_mocapid[model.body("child").id] = 0
    with pytest.raises(ValueError, match="mocap"):
        _verify_no_object_assistance(model, {"cube": {"body": "object", "joint": "free"}})


def test_exact_referenced_mesh_bytes_must_be_bound_not_just_source_files(tmp_path):
    from reachy_retarget.dynamics_audit import bind_scene_assets, _verify_assets
    (tmp_path / "assets").mkdir()
    mesh = tmp_path / "assets" / "converted.stl"
    mesh.write_bytes(b"original converted mesh bytes")
    texture = tmp_path / "assets" / "color.png"
    texture.write_bytes(b"original texture bytes")
    scene = tmp_path / "scene.xml"
    scene.write_text('<mujoco><compiler assetdir="assets"/><asset>'
                     '<mesh name="m" file="converted.stl"/><texture name="t" type="2d" file="color.png"/>'
                     '</asset></mujoco>')
    binding = bind_scene_assets(scene.read_text(), tmp_path)
    assert binding["meshes"] == {str(mesh): sha256(mesh)}
    assert binding["textures"] == {str(texture): sha256(texture)}
    with pytest.raises(ValueError, match="Unbound"):
        _verify_assets(scene, {"meshes": {"original.dae": "unrelated-source-hash"}})
    assert _verify_assets(scene, {"scene_asset_hashes": binding}) == binding
    mesh.write_bytes(b"changed converted mesh bytes")
    with pytest.raises(ValueError, match="checksum changed"):
        _verify_assets(scene, {"scene_asset_hashes": binding})


def test_changed_asset_set_and_unexpanded_includes_are_rejected(tmp_path):
    from reachy_retarget.dynamics_audit import bind_scene_assets, _verify_assets
    scene = tmp_path / "scene.xml"
    scene.write_text('<mujoco/>')
    with pytest.raises(ValueError, match="differ from bound manifest"):
        _verify_assets(scene, {"scene_asset_hashes": {"meshes": {"removed.stl": "x"}, "textures": {}, "hfields": {}}})
    with pytest.raises(ValueError, match="includes must be expanded"):
        bind_scene_assets('<mujoco><include file="external.xml"/></mujoco>', tmp_path)


def test_mesh_and_texture_directory_overrides_and_strippath(tmp_path):
    from reachy_retarget.dynamics_audit import bind_scene_assets
    meshes, textures = tmp_path / "mesh", tmp_path / "texture"
    meshes.mkdir(); textures.mkdir()
    mesh = meshes / "m.stl"; mesh.write_bytes(b"mesh")
    height = meshes / "terrain.bin"; height.write_bytes(b"height field")
    image = textures / "face.png"; image.write_bytes(b"texture")
    xml = ('<mujoco><compiler assetdir="wrong" meshdir="mesh" texturedir="texture" strippath="true"/>'
           '<asset><mesh name="m" file="discarded/path/m.stl"/>'
           '<hfield name="h" file="discarded/terrain.bin"/>'
           '<texture name="t" type="cube" fileleft="discarded/face.png"/>'
           '</asset></mujoco>')
    result = bind_scene_assets(xml, tmp_path)
    assert result == {"meshes": {str(mesh): sha256(mesh)},
                      "textures": {str(image): sha256(image)},
                      "hfields": {str(height): sha256(height)}}


def test_attached_model_cannot_escape_external_asset_binding(tmp_path):
    from reachy_retarget.dynamics_audit import bind_scene_assets
    with pytest.raises(ValueError, match="attached models"):
        bind_scene_assets('<mujoco><asset><model name="external" file="outside.xml"/></asset></mujoco>', tmp_path)
