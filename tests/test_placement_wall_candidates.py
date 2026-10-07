from types import SimpleNamespace
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from reachy_retarget import placement_wall_candidates as wall


def scene():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="source_scene_geom_x" type="box" size=".01 .5 .1" pos="0 .5 .8"/>
      <geom name="source_scene_geom_y" type="box" size=".5 .01 .1" pos=".5 0 .8"/>
      <geom name="source_scene_geom_floor" type="box" size=".5 .5 .01" pos=".5 .5 .7"/>
      <geom name="source_scene_geom_visual" type="box" size=".001 .5 .1" pos=".15 .5 .8" contype="0" conaffinity="0"/>
      <body name="r_hand_palm_link" pos=".25 .4 .9"/>
      <body name="Can" pos=".2 .3 .8"><freejoint name="Can_joint"/><geom type="sphere" size=".03"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    return model, data


def test_compiled_collision_walls_and_rotated_palm_define_inward_yaw():
    model, data = scene()
    selected = wall.nearest_walls(model, data, [.2, .3])
    assert [w['geom'] for w in selected] == ['source_scene_geom_x', 'source_scene_geom_y']
    np.testing.assert_allclose([w['distance'] for w in selected], [.19, .29])
    offset = np.array([.05, .1, .1])
    for pitch in (30., 40., 50.):
        for direction in ([1., 0.], [0., 1.], np.ones(2)/np.sqrt(2)):
            yaw = wall.yaw_toward(offset, pitch, direction)
            actual = Rotation.from_euler('xyz', [0., pitch, yaw], degrees=True).apply(offset)[:2]
            np.testing.assert_allclose(actual/np.linalg.norm(actual), direction, atol=1e-12)
    with pytest.raises(ValueError, match='axis'):
        wall.nearest_walls(model, data, [3., 3.])


def test_nine_mobile_proposals_preserve_source_arrays_and_actual_object(monkeypatch):
    model, data = scene()
    refs = np.zeros((4, 17)); refs[:, 0] = [-.3, -.1, .1, .2]
    objects = np.tile(np.eye(4), (4, 1, 1)); objects[:, :3, 3] = [.2, .3, .8]
    source = [None]*12
    source[0] = model
    source[2] = dict(objects={'Can': dict(joint='Can_joint', body='Can')}, mimics={})
    source[5] = dict(task='PickPlaceCan', task_contract=dict(bin_lower=[.1, .1, .7], bin_upper=[.9, .9, .9]),
                     world_placement=np.eye(4), pad_alignment=dict(inferred_contact_angle_rad=1.))
    source[7:12] = [np.arange(4)*.01, refs, np.array([2., -.06, -.06, 2.]), objects.copy(), objects]
    before = [v.copy() for v in source[7:12]]
    def query(model, data, *args):
        original = data.qpos.copy(); mujoco.mj_forward(model, data)
        np.testing.assert_array_equal(data.qpos, original)
    monkeypatch.setattr(wall, 'initialize', query)
    robot = SimpleNamespace(pack=lambda arm, base: np.r_[base, arm])
    bank, metadata = wall.bank(source, robot)
    assert len(bank) == 9 and len({p['id'] for p in bank}) == 9
    assert [p['rank'] for p in bank] == sorted(p['rank'] for p in bank)
    assert metadata['release_frame'] == 3
    steep, steep_metadata = wall.bank(source, robot, pitch_angles_deg=(60., 70., 80.))
    assert len(steep) == 9
    assert {p['parameters']['rotation_xyz_deg'][1] for p in steep} == {60., 70., 80.}
    assert steep_metadata['pitch_angles_deg'] == [60., 70., 80.]
    for angles in [[], [50., 50.], [30., 40., 50., 60.], [91.], [float('nan')]]:
        with pytest.raises(ValueError, match='pitch angles'):
            wall.bank(source, robot, pitch_angles_deg=angles)
    for proposal in bank:
        p = proposal['parameters']; assert p['base_offset_xyyaw'] == [0., 0., 0.]
        assert p['mobile_ik']['orientation_transport'] and p['mobile_ik']['pose_tolerance_constraints']
        bounds = np.array(p['mobile_ik']['base_bounds_xyyaw'])
        assert np.all(refs[:, :3] >= bounds[:, 0]) and np.all(refs[:, :3] <= bounds[:, 1])
    for old, new in zip(before, source[7:12]):
        np.testing.assert_array_equal(old, new)
