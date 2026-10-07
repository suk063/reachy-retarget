import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget.grasp_region import component, attachment
from reachy_retarget.pad_alignment import collision_section


def needle_model(extra=''):
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <body name="needle"><freejoint name="object_free"/>
            <geom name="shaft" type="box" pos="0 -.02 0" size=".005 .06 .005"/>
            <geom name="handle" type="box" pos="0 .06 0" size=".02 .02 .02"/>
            <geom name="visual" type="box" pos="0 .06 0" size=".001 .001 .001"
                  contype="0" conaffinity="0"/>
        '''+extra+'''</body></worldbody></mujoco>''')


def test_native_grasp_selects_original_handle_without_mutating_scene(monkeypatch):
    model = needle_model(); before = model.qpos0.copy()
    def forbidden(*args, **kwargs):
        raise AssertionError('Shape eligibility must not reset or advance physics')
    for method in ('mj_forward', 'mj_step', 'mj_resetData'):
        monkeypatch.setattr(mujoco, method, forbidden)
    selected = component(model, 'needle', 'object_free', [0, .059, -.007])
    assert selected['component_geom'] == 'handle'
    np.testing.assert_array_equal(selected['half_size_m'], [.02, .02, .02])
    np.testing.assert_array_equal(model.qpos0, before)
    assert selected['other_collision_components_preserved']


def test_closing_width_includes_connected_neighbor_instead_of_changing_object():
    model = needle_model(); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    center = np.array([0., .06, 0.]); body = model.body('needle').id
    horizontal = collision_section(model, data, body, center, np.array([1., 0., 0.]))
    along_shaft = collision_section(model, data, body, center, np.array([0., 1., 0.]))
    assert horizontal['width_m'] == pytest.approx(.04)
    assert along_shaft['width_m'] == pytest.approx(.16)
    assert set(along_shaft['collision_regions']) == {'shaft', 'handle'}
    np.testing.assert_array_equal(model.geom_size[model.geom('shaft').id], [.005, .06, .005])


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_declared_face_alignment_preserves_rigid_motion_and_centers_pads(axis):
    hand = np.eye(4); obj = np.eye(4)
    hand[:3, :3] = Rotation.from_euler('xyz', [20, -15, 5], degrees=True).as_matrix()
    obj[:3, :3] = Rotation.from_euler('z', 25, degrees=True).as_matrix()
    obj[:3, 3] = [.3, .2, .8]
    center, midpoint = np.array([0., .06, 0.]), np.array([-.01, 0., -.045])
    transform, rotation = attachment(hand, obj, [0, 1, 0], midpoint, np.eye(3), center, axis)
    goal = hand@transform
    np.testing.assert_allclose(goal[:3, 3]+goal[:3, :3]@midpoint, obj[:3, 3]+obj[:3, :3]@center)
    closing = obj[:3, :3].T@goal[:3, :3]@np.array([0., 1., 0.])
    assert abs(closing[axis]) == pytest.approx(1.)
    assert 0 <= rotation <= np.pi/2
    second = hand.copy(); second[:3, 3] = [.1, .2, .3]
    np.testing.assert_allclose((second@transform)@np.linalg.inv(goal), second@np.linalg.inv(hand), atol=1e-14)


def test_invented_grasp_component_and_articulated_subtree_are_rejected():
    with pytest.raises(ValueError, match='outside'):
        component(needle_model(), 'needle', 'object_free', [2., 0., 0.])
    model = needle_model('<body><joint type="hinge"/><geom type="sphere" size=".01"/></body>')
    with pytest.raises(ValueError, match='nonarticulated'):
        component(model, 'needle', 'object_free', [0., .06, 0.])
