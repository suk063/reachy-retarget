import numpy as np
import mujoco
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget.grasp_attachment import derive_attachment, single_box_geometry


def test_constant_attachment_aligns_face_and_centers_pad_without_input_mutation():
    hand, obj = np.eye(4), np.eye(4)
    hand[:3, :3] = Rotation.from_euler("z", 12, degrees=True).as_matrix()
    hand[:3, 3] = [.2, .3, .8]
    obj[:3, 3] = [.2, .3, .84]
    jaw, midpoint = np.array([0., 2., 0.]), np.array([0., 0., .04])
    before = [a.copy() for a in (hand, obj, jaw, midpoint)]
    attachment, report = derive_attachment(hand, obj, jaw, midpoint, np.eye(3), np.zeros(3))
    goal = hand @ attachment
    np.testing.assert_allclose(goal[:3, :3] @ [0., 1., 0.], [0., 1., 0.], atol=1e-12)
    np.testing.assert_allclose(goal[:3, :3] @ midpoint + goal[:3, 3], obj[:3, 3], atol=1e-12)
    assert report["rotation_change_rad"] == pytest.approx(np.deg2rad(12))
    for actual, original in zip((hand, obj, jaw, midpoint), before):
        np.testing.assert_array_equal(actual, original)


def test_constant_tcp_attachment_preserves_every_relative_hand_motion():
    first = np.eye(4)
    first[:3, :3] = Rotation.from_euler("xyz", [5, 8, 12], degrees=True).as_matrix()
    attachment, _ = derive_attachment(first, np.eye(4), [0, 1, 0], [0, 0, -.05],
                                      np.eye(3), [.01, -.01, 0])
    second = np.eye(4)
    second[:3, :3] = Rotation.from_euler("xyz", [10, 30, 40], degrees=True).as_matrix()
    second[:3, 3] = [.3, -.1, .2]
    # A common right attachment preserves world-relative rigid motion between
    # timestamps exactly, unlike per-frame pose matching to an object.
    np.testing.assert_allclose((second @ attachment) @ np.linalg.inv(first @ attachment),
                               second @ np.linalg.inv(first), atol=1e-12)


def test_center_only_keeps_hand_rotation_and_rejects_bad_frame():
    first = np.eye(4)
    first[:3, :3] = Rotation.from_euler("z", 13, degrees=True).as_matrix()
    attachment, report = derive_attachment(first, np.eye(4), [0, 1, 0], [0, 0, .03],
                                           np.eye(3), [0, 0, 0], align_box_axis=False)
    np.testing.assert_allclose((first @ attachment)[:3, :3], first[:3, :3], atol=1e-12)
    assert report["rotation_change_rad"] == 0
    with pytest.raises(ValueError, match="orthonormal"):
        derive_attachment(first, np.eye(4), [0, 1, 0], [0, 0, .03], 2*np.eye(3), [0, 0, 0])


def box_model(child="", extra="", pair=""):
    return mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
        <geom name="floor" type="plane" size="1 1 .1"/>
        <body name="task_box" pos=".3 .4 .5" euler="10 20 30"><freejoint name="task_free"/>
          <geom name="visual" type="sphere" size=".01" mass=".001" contype="0" conaffinity="0"/>
          <body name="rigid_part" pos=".01 .02 .03" euler="0 0 90">{child}
            <geom name="collision" type="box" size=".02 .03 .04" pos=".1 0 0" euler="0 90 0"/>
          </body>{extra}
        </body></worldbody>{pair}</mujoco>''')


def test_generic_box_eligibility_preserves_rotated_offset_geometry_and_state(monkeypatch):
    model = box_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    before = data.qpos.copy()
    root, geom = model.body("task_box").id, model.geom("collision").id
    expected_axes = data.xmat[root].reshape(3, 3).T @ data.geom_xmat[geom].reshape(3, 3)
    expected_center = data.xmat[root].reshape(3, 3).T @ (data.geom_xpos[geom]-data.xpos[root])
    def forbidden(*args, **kwargs):
        raise AssertionError("Eligibility cannot advance, forward or reset simulation")
    for name in ("mj_step", "mj_forward", "mj_setState", "mj_resetData"):
        monkeypatch.setattr(mujoco, name, forbidden)
    report = single_box_geometry(model, object_body="task_box", object_joint="task_free")
    assert report["eligible"] and report["rigid_body_names"] == ["task_box", "rigid_part"]
    np.testing.assert_allclose(report["box_axes_object"], expected_axes, atol=1e-14)
    np.testing.assert_allclose(report["box_center_object_m"], expected_center, atol=1e-14)
    np.testing.assert_array_equal(data.qpos, before)


def test_box_eligibility_rejects_articulation_and_explicit_pair_extra_collision():
    articulated = box_model(child='<joint name="hinge" axis="0 0 1"/>')
    with pytest.raises(ValueError, match="articulated"):
        single_box_geometry(articulated, object_body="task_box", object_joint="task_free")
    paired = box_model(pair='<contact><pair geom1="visual" geom2="floor"/></contact>')
    with pytest.raises(ValueError, match="one verified source box"):
        single_box_geometry(paired, object_body="task_box", object_joint="task_free")
    sphere = box_model()
    sphere.geom_type[sphere.geom("collision").id] = mujoco.mjtGeom.mjGEOM_SPHERE
    with pytest.raises(ValueError, match="one verified source box"):
        single_box_geometry(sphere, object_body="task_box", object_joint="task_free")
