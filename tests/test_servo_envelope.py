"""Exercise command saturation and recovery through an actual MuJoCo servo."""
import mujoco
import numpy as np
from reachy_retarget import servo_envelope


def test_feedforward_and_integral_cannot_command_past_reserved_joint_margin():
    model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
    <option timestep=".002" gravity="0 0 0"/>
    <worldbody><body><joint name="j" type="slide" axis="1 0 0" range="-1 1"/>
    <geom size=".01" mass=".1"/></body></worldbody>
    <actuator><position joint="j" kp="180" kv="24"/></actuator></mujoco>''')
    bounds = servo_envelope.limits(model, [0], .03)
    data = mujoco.MjData(model)
    memory = np.array([.02])
    peak = 0.
    for index in range(400):
        reference = np.array([.96 if index < 200 else -.4])
        feedforward = np.array([.08 if index < 200 else 0.])
        data.ctrl[:] = servo_envelope.target(reference, memory, feedforward, bounds)
        for _ in range(5):
            mujoco.mj_step(model, data)
            peak = max(peak, data.qpos[0])
        memory = servo_envelope.integrate(memory, reference-data.qpos, reference, feedforward, bounds)
    assert peak <= .9701
    assert abs(data.qpos[0]+.4) < .005
    assert abs(memory[0]) < .005


def test_saturation_does_not_accumulate_error_in_the_blocked_direction():
    # A persistent position demand outside the reserved margin must not store
    # a correction that worsens future reversal/recovery.
    memory = np.zeros(2)
    reference = np.array([1.1, -1.1])
    bounds = np.array([-.97, -.97]), np.array([.97, .97])
    for _ in range(1000):
        memory = servo_envelope.integrate(memory, [.13, -.13], reference, np.zeros(2), bounds)
    np.testing.assert_array_equal(memory, 0.)
    recovered = servo_envelope.integrate(memory, [-.2, .2], reference, np.zeros(2), bounds)
    assert recovered[0] < 0 < recovered[1]
