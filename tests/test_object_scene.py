import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget.episodes import pose_to_matrices
from reachy_retarget.object_scene import MissingSceneAssets, build_scene
from reachy_retarget import object_scene


def robot_cache(root):
    folder = root / "data/assets/reachy_mujoco"
    folder.mkdir(parents=True)
    (folder / "robot.xml").write_text('''<mujoco><compiler angle="radian"/>
      <option timestep=".002"/><default><geom friction="9 .1 .01" solref=".004 1"/></default>
      <asset/><worldbody><geom name="floor" type="plane" size="10 10 .1"/>
      <body name="base_link"><joint name="base_x" type="slide" axis="1 0 0"/>
      <inertial mass="1" pos="0 0 0" diaginertia=".01 .01 .01"/>
      <geom name="robot_sphere" type="sphere" size=".05"/></body></worldbody>
      <actuator/><equality/><contact/></mujoco>''')
    (folder / "manifest.json").write_text(json.dumps({"mimics": {}}))


def source(*, nested=False, extras="", mesh=False, texture=False):
    asset = '<asset/>'
    material = ''
    geom = '<geom name="cube_collision" type="box" size=".02 .03 .04"/>'
    if mesh:
        asset = '<asset><mesh name="missing" file="/old/robosuite/models/assets/objects/meshes/missing.stl"/></asset>'
        geom = '<geom name="cube_collision" type="mesh" mesh="missing"/>'
    if texture:
        asset = '<asset><texture name="missing_texture" type="2d" file="/old/robosuite/models/assets/textures/missing.png"/><material name="wood" texture="missing_texture"/></asset>'
        material = ' material="wood"'
        geom = geom.replace('/>', material + '/>')
    if nested:
        geom = '<body name="cube_piece" pos=".1 .2 .3">' + geom + '</body>'
    xml = f'''<mujoco><compiler angle="radian"/><option density="1.2" viscosity=".00002"/>
      <default><geom friction=".42 .006 .0002" solref=".03 .7" density="350"/>
      <joint damping=".004"/><default class="payload"><geom density="2000"/></default></default>
      {asset}<worldbody><geom name="floor" type="plane" size="3 3 .1"/>
      <body name="table" pos="0 0 .4"><geom name="table_collision" type="box" size=".5 .4 .02"/></body>
      <body name="robot0_base"><geom type="sphere" size=".1"/></body>
      <body name="cube_main" childclass="payload"><joint type="free" name="cube_joint0"/>{geom}</body>
      </worldbody>{extras}</mujoco>'''
    metadata = {"model_xml": xml, "env_args": {"env_name": "Lift", "env_version": "1.5.1"},
                "objects": {"cube": {"body": "cube_main", "joint": "cube_joint0"}}}
    pose = np.array([[.2, -.1, .7, 1, 0, 0, 0], [.2, -.1, .8, 1, 0, 0, 0]])
    arrays = {"time_s": np.array([0., .05]), "objects/cube/pose": pose}
    return metadata, arrays


def test_preserve_source_defaults_mass_friction_and_world_placement(tmp_path):
    robot_cache(tmp_path)
    metadata, arrays = source()
    original_xml = metadata["model_xml"]
    initial_pose = arrays["objects/cube/pose"].copy()
    placement = np.eye(4)
    placement[:3, :3] = Rotation.from_euler("z", .7).as_matrix()
    placement[:3, 3] = [.4, .1, 0]
    xml, manifest = build_scene(tmp_path, metadata, arrays, placement)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    info = manifest["objects"]["cube"]
    assert info["mass_kg"] == pytest.approx(.04 * .06 * .08 * 2000)
    np.testing.assert_allclose(model.geom_friction[model.geom("cube_collision").id], [.42, .006, .0002])
    np.testing.assert_allclose(model.geom_solref[model.geom("cube_collision").id], [.03, .7])
    np.testing.assert_allclose(model.geom_friction[model.geom("robot_sphere").id], [9, .1, .01])
    np.testing.assert_allclose(data.xpos[info["body_id"]], (placement @ pose_to_matrices(initial_pose[0]))[:3, 3])
    np.testing.assert_allclose(model.dof_damping[info["qvel_address"]:info["qvel_address"] + 6], .004)
    assert model.jnt_type[info["joint_id"]] == mujoco.mjtJoint.mjJNT_FREE
    assert model.body_gravcomp[info["body_id"]] == 0 and model.nu == 0 and model.neq == 0
    assert max(manifest["validation"].values()) < 1e-7
    assert metadata["model_xml"] == original_xml
    np.testing.assert_array_equal(arrays["objects/cube/pose"], initial_pose)
    # Demonstrate actual unconstrained gravity, without resetting the object.
    before = float(data.qpos[info["qpos_address"] + 2])
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert data.qpos[info["qpos_address"] + 2] < before


def test_compound_massless_carrier_retains_child_inertia(tmp_path):
    robot_cache(tmp_path)
    metadata, arrays = source(nested=True)
    placement = np.eye(4); placement[0, 3] = .5
    _, manifest = build_scene(tmp_path, metadata, arrays, placement)
    info = manifest["objects"]["cube"]
    assert info["source_root_inertial"]["mass_kg"] == 0
    assert info["mass_kg"] == pytest.approx(.384)
    assert info["rigid_component_masses_kg"]["cube_piece"] == pytest.approx(.384)
    assert len(info["collision_geom_ids"]) == 1


def test_missing_collision_mesh_reports_pinned_explicit_fetch(tmp_path):
    robot_cache(tmp_path)
    metadata, arrays = source(mesh=True)
    with pytest.raises(MissingSceneAssets) as error:
        build_scene(tmp_path, metadata, arrays, np.eye(4))
    asset = error.value.assets[0]
    assert asset["url"] == "https://raw.githubusercontent.com/ARISE-Initiative/robosuite/v1.5.1/robosuite/models/assets/objects/meshes/missing.stl"
    assert not list((tmp_path / "data/raw").glob("**/*"))


def test_missing_texture_is_explicit_and_does_not_change_collision(tmp_path):
    robot_cache(tmp_path)
    metadata, arrays = source(texture=True)
    xml, manifest = build_scene(tmp_path, metadata, arrays, np.eye(4))
    assert len(manifest["missing_visual_assets"]) == 1
    assert ET.fromstring(xml).find('asset/material[@name="wood"]').get("texture") is None
    assert manifest["objects"]["cube"]["mass_kg"] == pytest.approx(.384)


@pytest.mark.parametrize("constraint", ['<equality><weld body1="cube_main"/></equality>',
                                         '<actuator><position joint="cube_joint0" kp="1"/></actuator>'])
def test_object_welds_and_servos_are_rejected(tmp_path, constraint):
    robot_cache(tmp_path)
    metadata, arrays = source(extras=constraint)
    with pytest.raises(ValueError, match="constraint/actuator"):
        build_scene(tmp_path, metadata, arrays, np.eye(4))


def articulated_source():
    metadata, arrays = source()
    fixture = '''<body name="cabinet" pos="1 0 .6"><body name="door">
      <joint name="door_hinge" type="hinge" axis="0 0 1" damping=".12"
       range="-1 1" limited="true" stiffness=".4"/>
      <geom name="door_shape" type="box" size=".2 .02 .3" pos=".2 0 0" mass=".5"/>
      </body></body>'''
    metadata["model_xml"] = metadata["model_xml"].replace("</worldbody>",fixture+"</worldbody>")
    return metadata, arrays


def test_unknown_articulated_fixture_is_never_silently_deleted(tmp_path):
    robot_cache(tmp_path)
    metadata, arrays = articulated_source()
    with pytest.raises(ValueError,match="refusing omission"):
        build_scene(tmp_path,metadata,arrays,np.eye(4))


def test_passive_door_preserves_state_limits_and_dynamics(tmp_path):
    robot_cache(tmp_path)
    metadata, arrays = articulated_source()
    metadata["fixtures"] = {"cabinet":{"body":"cabinet","joints":["door_hinge"]}}
    arrays["fixtures/cabinet/joint_position"] = np.array([[.3],[.4]])
    placement=np.eye(4);placement[:3,:3]=Rotation.from_euler("z",.5).as_matrix()
    xml, manifest=build_scene(tmp_path,metadata,arrays,placement)
    assert max(manifest["validation"].values())<1e-7
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    object_scene.initialize_fixtures(model,data,manifest)
    j=model.joint("door_hinge").id;address=model.jnt_qposadr[j]
    assert data.qpos[address]==.3
    np.testing.assert_allclose(model.jnt_range[j],[-1,1])
    assert model.dof_damping[model.jnt_dofadr[j]]==.12
    before=float(data.qpos[address])
    for _ in range(10):mujoco.mj_step(model,data)
    assert data.qpos[address] < before  # Passive spring, not prescribed playback.
    with pytest.raises(ValueError,match="reset-only"):
        object_scene.initialize_fixtures(model,data,manifest)


def extruded_l_profile():
    import trimesh
    yz = np.array([[-.4, 0], [.4, 0], [.4, .05], [-.3, .05], [-.3, .5], [-.4, .5]])
    vertices = np.vstack([np.c_[np.full(6, -.02), yz], np.c_[np.full(6, .02), yz]])
    triangles = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]]
    faces = [[a, c, b] for a, b, c in triangles] + [[a + 6, b + 6, c + 6] for a, b, c in triangles]
    for i in range(6):
        j = (i + 1) % 6
        faces += [[i, j, j + 6], [i, j + 6, i + 6]]
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(vertices), 2)),
                    material=trimesh.visual.material.PBRMaterial(baseColorFactor=[204, 204, 204, 255]))
    return mesh


def test_table_profile_decomposition_preserves_concavity():
    from scipy.spatial import ConvexHull
    mesh = extruded_l_profile()
    parts, check = object_scene._extrusion_parts(mesh)
    assert len(parts) == 4
    assert check["source_volume"] == pytest.approx(.04 * (.8 * .05 + .1 * .45))
    assert check["decomposed_volume"] == pytest.approx(check["source_volume"])
    # The whole-mesh hull would fill this open notch; none of our prisms do.
    notch = np.array([0, 0, .2])
    assert np.all(ConvexHull(mesh.vertices).equations[:, :3] @ notch + ConvexHull(mesh.vertices).equations[:, 3] < 0)
    for vertices in parts:
        equations = ConvexHull(vertices).equations
        assert np.any(equations[:, :3] @ notch + equations[:, 3] > 0)


def maniskill_fixture(tmp_path, monkeypatch):
    import trimesh
    folder = tmp_path / "evidence"
    path = folder / object_scene.MANISKILL_TABLE
    path.parent.mkdir(parents=True)
    trimesh.Scene(extruded_l_profile()).export(path)
    monkeypatch.setattr(object_scene, "_maniskill_evidence", lambda root: (folder, []))
    metadata = {"source_format": "ManiSkill-HDF5", "source_commit": object_scene.MANISKILL_REVISION,
                "env_args": {"env_name": "PickCube-v1", "env_kwargs": {"obs_mode": "none"}},
                "objects": {"cube": {"type": "cube"}}}
    arrays = {"time_s": np.array([0., .05]),
              "objects/cube/pose": np.array([[.2, 0, .2, 1, 0, 0, 0], [.2, 0, .2, 1, 0, 0, 0]]),
              "source/env_states/actors/table-workspace": np.tile([-.12, 0, -.875, 2**-.5, 0, 0, 2**-.5, 0, 0, 0, 0, 0, 0], (2, 1))}
    return metadata, arrays


def test_maniskill_parameters_and_unconstrained_freefall(tmp_path, monkeypatch):
    robot_cache(tmp_path)
    metadata, arrays = maniskill_fixture(tmp_path, monkeypatch)
    placement = np.eye(4); placement[:3, 3] = [.5, 0, .8]
    xml, manifest = build_scene(tmp_path, metadata, arrays, placement)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    info = manifest["objects"]["cube"]
    assert info["body"] == "cube_main" and info["joint"] == "cube_joint0"
    assert info["mass_kg"] == pytest.approx(.064)
    np.testing.assert_allclose(model.body_inertia[info["body_id"]], .064 * .04**2 / 6)
    geom = model.geom("cube_collision").id
    np.testing.assert_allclose(model.geom_size[geom], [.02] * 3)
    np.testing.assert_allclose(model.geom_friction[geom], [.3, 0, 0])
    assert model.geom_condim[geom] == 3
    assert manifest["source_physical_parameters"]["restitution"] == 0
    assert model.nu == 0 and model.neq == 0 and model.body_gravcomp[info["body_id"]] == 0
    assert manifest["conversion_assumptions"] and manifest["source_xml_sha256"] is None
    assert max(manifest["validation"].values()) < 1e-7
    address = info["qpos_address"]
    initial = data.qpos[address:address + 7].copy()
    for _ in range(10):
        mujoco.mj_step(model, data)
    expected_z = initial[2] - .5 * 9.81 * data.time**2
    assert data.qpos[address + 2] == pytest.approx(expected_z, abs=.00021)
    np.testing.assert_allclose(data.qpos[address:address + 2], initial[:2], atol=1e-12)


def test_maniskill_refuses_moving_table(tmp_path, monkeypatch):
    metadata, arrays = maniskill_fixture(tmp_path, monkeypatch)
    arrays["source/env_states/actors/table-workspace"][1, 0] += .01
    with pytest.raises(ValueError, match="Moving source table"):
        build_scene(tmp_path, metadata, arrays, np.eye(4))


def test_maniskill_requires_pinned_local_evidence(tmp_path):
    with pytest.raises(MissingSceneAssets) as error:
        object_scene._maniskill_evidence(tmp_path)
    assert any(record["url"].endswith("/3.0.0.dev2/python/py_package/wrapper/actor_builder.py") for record in error.value.assets)
    path = tmp_path / "data/raw/maniskill_scene_evidence/setup.py"
    path.parent.mkdir(parents=True)
    path.write_text("modified evidence")
    with pytest.raises(ValueError, match="checksum mismatch"):
        object_scene._maniskill_evidence(tmp_path)
