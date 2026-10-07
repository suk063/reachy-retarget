import mujoco
import numpy as np
import pytest

from reachy_retarget.geometry_clearance import RobotFixtureClearance, closed_aperture_interval


def scene(gap=.001, explicit=False):
    pair = '<contact><pair geom1="palm" geom2="fixture"/></contact>' if explicit else ''
    mask = 'contype="0" conaffinity="0"' if explicit else ''
    return mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <geom name="floor" type="plane" size="1 1 .1"/>
      <body name="base_link"><freejoint/>
        <geom name="wheel" type="sphere" size=".01" pos="0 0 .01"/>
        <body name="hand"><geom name="palm" type="sphere" size=".02" pos="0 0 1" {mask}/></body>
      </body><body name="task_box" pos=".03 0 1"><freejoint/><geom name="active" size=".02"/></body>
      <body name="receptacle" pos="{.04+gap} 0 1"><geom name="fixture" size=".02" {mask}/></body>
      <body name="far_fixture" pos="10 0 1"><geom name="far" size=".1"/></body>
      </worldbody>{pair}</mujoco>''')


def test_positive_clearance_rejects_nonpenetrating_obstruction_without_mutation(monkeypatch):
    model = scene(); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    observer = RobotFixtureClearance(model, active_object_bodies=('task_box',),
                                    support_pairs=(('wheel', 'floor'),), minimum_m=.003)
    before = [getattr(data, n).copy() for n in ('qpos', 'qvel', 'ctrl', 'geom_xpos')]
    def forbidden(*args, **kwargs):
        raise AssertionError('The clearance observer cannot advance or forward state')
    monkeypatch.setattr(mujoco, 'mj_step', forbidden)
    monkeypatch.setattr(mujoco, 'mj_forward', forbidden)
    report = observer.inspect(data)
    assert report['admitted'] is False
    assert report['minimum_signed_distance_or_lower_bound_m'] == pytest.approx(.001)
    assert report['closest_geom_pair'] == ['palm', 'fixture']
    assert report['certified_far_pairs'] > 0
    for name, saved in zip(('qpos', 'qvel', 'ctrl', 'geom_xpos'), before):
        np.testing.assert_array_equal(getattr(data, name), saved)


def test_no_implicit_floor_exemption_and_explicit_pairs_ignore_zero_masks():
    model = scene(explicit=True); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    no_exemption = RobotFixtureClearance(model, active_object_bodies=('task_box',))
    assert not no_exemption.inspect(data)['admitted']
    clear = RobotFixtureClearance(model, active_object_bodies=('task_box',), support_pairs=(('wheel', 'floor'),))
    assert clear.inspect(data)['closest_geom_pair'] == ['palm', 'fixture']
    with pytest.raises(ValueError, match='robot/fixture'):
        RobotFixtureClearance(model, active_object_bodies=('task_box',), support_pairs=(('palm', 'active'),))


def test_far_geometry_returns_honest_distance_bound():
    model = scene(gap=.01); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    checker = RobotFixtureClearance(model, active_object_bodies=('task_box',), support_pairs=(('wheel', 'floor'),))
    report = checker.inspect(data)
    assert report['admitted'] and report['distance_is_lower_bound']
    assert report['minimum_signed_distance_or_lower_bound_m'] == pytest.approx(.003001)
    assert report['closest_geom_pair'] is None


def test_numeric_actuator_envelope_and_force_control_are_explicit():
    details = dict(pad_alignment=dict(inferred_contact_angle_rad=.94),
                   effective_candidate=dict(gripper_closed_target=.89, force_feedback=False))
    np.testing.assert_allclose(closed_aperture_interval(details), [.89, .94])
    with pytest.raises(ValueError, match='actual numeric'):
        closed_aperture_interval(details, [.9, .94])
    details['effective_candidate']['force_feedback'] = True
    with pytest.raises(ValueError, match='explicit static'):
        closed_aperture_interval(details)
    np.testing.assert_allclose(closed_aperture_interval(details, [.8, .94]), [.8, .94])
