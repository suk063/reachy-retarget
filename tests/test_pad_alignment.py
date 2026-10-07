"""The correction changes contact translation without replanning other channels."""

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget.pad_alignment import (
    blend_weights, translate_only, convex_line_interval, collision_section, pad_surfaces,
)


def test_translation_preserves_rotation_and_free_motion_without_mutation():
    times = np.arange(0, 4.01, .01)
    closed = (times >= 1) & (times < 3)
    weights = blend_weights(times,closed)
    hands = np.tile(np.eye(4),(len(times),1,1))
    hands[:,0,3] = times
    objects = hands.copy()
    objects[:,:3,:3] = Rotation.from_euler('z',times[:,None]).as_matrix()
    original_hands, original_objects = hands.copy(),objects.copy()
    delta = np.array([.01,.02,.03])
    result = translate_only(hands,objects,delta,weights)
    np.testing.assert_array_equal(result[:,:3,:3],hands[:,:3,:3])
    np.testing.assert_array_equal(result[weights==0],hands[weights==0])
    held_delta = np.einsum('nji,nj->ni',objects[closed,:3,:3],result[closed,:3,3]-hands[closed,:3,3])
    np.testing.assert_allclose(held_delta,np.tile(delta,(closed.sum(),1)),atol=1e-15)
    np.testing.assert_array_equal(hands,original_hands)
    np.testing.assert_array_equal(objects,original_objects)
    assert weights[0] == weights[-1] == 0


@pytest.mark.parametrize('closed', [[1,1,0,0],[0,1,0,1],[0,0,0,0]])
def test_invalid_contact_intervals_rejected(closed):
    with pytest.raises(ValueError):blend_weights(np.arange(4),closed)


def test_alignment_completes_before_open_fingers_enter_contact_envelope():
    times=np.arange(0,4.01,.01)
    closed=(times>=2.6)&(times<3.5)
    weights=blend_weights(times,closed,ready_time=1.2)
    assert weights[0] == 0 and weights[np.searchsorted(times,1.2)] == 1
    assert np.all(weights[(times>=1.2)&(times<=3.5)] == 1)
    with pytest.raises(ValueError,match="free pregrasp"):
        blend_weights(times,closed,ready_time=0)


def test_cross_section_is_at_source_contact_height_not_object_bounding_box():
    # Tapered geometry is narrow near its top. An AABB would return width 2.
    vertices = np.array([[x*s,y*s,z] for z,s in [(0,1),(1,.1)] for x in (-1,1) for y in (-1,1)])
    interval = convex_line_interval(vertices,np.array([.05,0,.8]),np.array([1.,0,0]))
    np.testing.assert_allclose(interval,[-.33,.23],atol=1e-12)
    assert convex_line_interval(vertices,[0,0,2],[1,0,0]) is None


def test_overlapping_collision_pieces_do_not_propose_an_artificial_thin_grasp():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="object" pos="3 2 1">
      <geom name="a" type="box" size=".02 .02 .02" pos="-.01 0 0"/>
      <geom name="b" type="box" size=".02 .02 .02" pos=".01 0 0"/>
    </body></worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    result=collision_section(model,data,model.body('object').id,np.array([.004,0,0]),np.array([1.,0,0]))
    assert result['width_m'] == pytest.approx(.06)
    np.testing.assert_allclose(result['center_object'],[0,0,0],atol=1e-12)
    assert set(result['collision_regions']) == {'a','b'}


def test_opposing_pad_centroids_transform_with_tcp_and_ignore_outward_faces():
    # MuJoCo computes the closed box mesh hull and its consistent face winding.
    vertices=' '.join(str(v) for x in (-.01,.01) for y in (-.003,.003) for z in (-.02,.02) for v in (x,y,z))
    xml=f'''<mujoco><asset><mesh name="pad" vertex="{vertices}"/></asset><worldbody>
      <body pos="1 2 3" euler="20 30 40"><site name="r_arm_tip_tcp"/>
        <body name="r_hand_distal_link" pos="0 -.03 .06"><geom type="mesh" mesh="pad"/></body>
        <body name="r_hand_distal_mimic_link" pos="0 .03 .06"><geom type="mesh" mesh="pad"/></body>
      </body></worldbody></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    pads=pad_surfaces(model,data)
    np.testing.assert_allclose(pads['midpoint_tcp'],[0,0,.06],atol=1e-8)
    assert pads['gap_m'] == pytest.approx(.054,abs=1e-8)
