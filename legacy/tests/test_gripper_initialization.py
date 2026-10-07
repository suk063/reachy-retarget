"""Static robot aperture inspection must keep the inactive hand open."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from reachy_retarget.physics import ARMS, BASE, GRIPPERS, initialize


def scene():
    names = [*BASE, *ARMS, *GRIPPERS, 'left_mimic', 'right_mimic']
    bodies = ''.join(f'<body name="b{i}"><joint name="{name}"/><geom type="sphere" '
                     'size=".01" contype="0" conaffinity="0"/></body>'
                     for i, name in enumerate(names))
    actuators = ''.join(f'<position name="{name}" joint="{name}"/>' for name in [*BASE, *ARMS, *GRIPPERS])
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+bodies+
        '<body name="object" pos=".2 .3 .8"><freejoint name="object_free"/>'
        '<geom type="box" size=".02 .02 .02"/></body></worldbody><actuator>'+
        actuators+'</actuator></mujoco>')
    robot = SimpleNamespace(arm_ids=np.arange(len(ARMS)), base=lambda q: np.array([.1, .2, .3]),
                            r=SimpleNamespace(q_indices={}))
    mimics = {'left_mimic': ('l_hand_finger', -1., .1), 'right_mimic': ('r_hand_finger', -1., .1)}
    return model, robot, mimics


def test_asymmetric_apertures_reach_actuators_and_mimics_without_object_writes():
    model, robot, mimics = scene(); data = mujoco.MjData(model)
    address = int(model.joint('object_free').qposadr[0]); original = data.qpos[address:].copy()
    initialize(model, data, robot, np.zeros(len(ARMS)), mimics,
               {'l_hand_finger': 2., 'r_hand_finger': .7})
    for name, expected in [('l_hand_finger', 2.), ('r_hand_finger', .7),
                           ('left_mimic', -1.9), ('right_mimic', -.6)]:
        assert data.qpos[model.joint(name).qposadr[0]] == pytest.approx(expected)
    assert data.ctrl[model.actuator('l_hand_finger').id] == 2.
    assert data.ctrl[model.actuator('r_hand_finger').id] == .7
    np.testing.assert_array_equal(data.qpos[address:], original)
    assert data.time == 0.


def test_scalar_reset_is_backward_compatible_and_partial_mapping_is_rejected():
    model, robot, mimics = scene(); data = mujoco.MjData(model)
    initialize(model, data, robot, np.zeros(len(ARMS)), mimics, 2.)
    assert all(data.qpos[model.joint(g).qposadr[0]] == 2. for g in GRIPPERS)
    with pytest.raises(ValueError, match='both finite'):
        initialize(model, data, robot, np.zeros(len(ARMS)), mimics, {'r_hand_finger': .7})
