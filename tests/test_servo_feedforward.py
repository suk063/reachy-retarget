"""Check feedforward against actual MuJoCo servo tracking, without robot I/O."""
import mujoco
import numpy as np
from reachy_retarget.servo_feedforward import velocity_offsets
from reachy_retarget.servo_feedforward import source_prefix_offsets
from reachy_retarget.servo_feedforward import source_prefix_base_offsets
from reachy_retarget.servo_feedforward import base_velocity_offsets
from reachy_retarget.servo_feedforward import PostAcquisitionBaseOffsets


def test_velocity_compensation_reduces_physical_tracking_lag_without_changing_reference():
    model=mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".002" gravity="0 0 0"/>
      <worldbody><body><joint name="j" type="slide" axis="1 0 0"/>
      <geom size=".01" mass=".1"/></body></worldbody>
      <actuator><position joint="j" kp="180" kv="24"/></actuator></mujoco>''')
    seconds=np.arange(0,2,.01)
    reference=np.zeros((len(seconds),4));reference[:,3]=.3*seconds
    saved=reference.copy()
    # The helper skips three base columns; map the remaining arm to the servo.
    feedforward=velocity_offsets(model,[0,0,0,0],reference,seconds,1.)
    errors=[]
    for gain in (0.,1.):
        data=mujoco.MjData(model)
        for i in range(len(seconds)):
            data.ctrl[0]=reference[i,3]+gain*feedforward[i,3]
            for _ in range(5):mujoco.mj_step(model,data)
        errors.append(abs(data.qpos[0]-.3*data.time))
    assert errors[1]<errors[0]*.1
    # The 100 Hz zero-order hold has a small within-interval velocity ripple.
    assert abs(data.qvel[0]-.3)<.01
    np.testing.assert_array_equal(reference,saved)
    np.testing.assert_array_equal(feedforward[:,:3],0.)


def test_source_prefix_compensation_is_bounded_and_zero_for_feedback_phases():
    from types import SimpleNamespace
    model=SimpleNamespace(actuator_gainprm=np.array([[180.]]),actuator_biasprm=np.array([[0.,0.,-24.]]))
    seconds=np.arange(200)*.01
    reference=np.zeros((200,4));reference[:,3]=.8*seconds
    result=source_prefix_offsets(model,[0,0,0,0],reference,seconds,1.,100)
    assert np.max(abs(result))<=.2
    np.testing.assert_allclose(result[20:80,3],24./180.*.8)
    np.testing.assert_array_equal(result[:,:3],0.)
    np.testing.assert_array_equal(result[99:],0.)
    assert result[0,3]==0 and 0<result[95,3]<result[80,3]
    np.testing.assert_array_equal(source_prefix_offsets(model,[0,0,0,0],reference,seconds,0.,100),0.)


def test_base_compensation_reduces_physical_damping_lag_and_preserves_other_commands():
    model=mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".001" gravity="0 0 0"/>
      <worldbody><body><joint name="x" type="slide" axis="1 0 0"/>
      <joint name="y" type="slide" axis="0 1 0"/><joint name="yaw" axis="0 0 1"/>
      <geom size=".1" mass="25"/></body></worldbody>
      <actuator><position joint="x" kp="3000" kv="600"/>
      <position joint="y" kp="3000" kv="600"/><position joint="yaw" kp="300" kv="60"/>
      </actuator></mujoco>''')
    seconds=np.arange(300)*.01
    reference=np.zeros((300,4));reference[:,0]=.1*seconds;reference[:,3]=.4
    saved=reference.copy()
    offsets=source_prefix_base_offsets(model,[0,1,2,0],reference,seconds,1.,250)
    errors=[]
    for gain in (0.,1.):
        data=mujoco.MjData(model)
        for i in range(200):
            data.ctrl[:]=reference[i,:3]+gain*offsets[i,:3]
            for _ in range(10):mujoco.mj_step(model,data)
        errors.append(abs(data.qpos[0]-.1*data.time))
    assert errors[1]<errors[0]*.1
    np.testing.assert_array_equal(offsets[:,3:],0.)
    np.testing.assert_array_equal(offsets[249:],0.)
    np.testing.assert_array_equal(reference,saved)


def test_base_compensation_caps_offsets_and_excludes_feedback_boundary_velocity():
    from types import SimpleNamespace
    model=SimpleNamespace(actuator_gainprm=np.full((3,1),10.),actuator_biasprm=np.tile([0.,0.,-100.],(3,1)))
    seconds=np.arange(200)*.01
    reference=np.zeros((200,5));reference[:,:3]=seconds[:,None]
    result=source_prefix_base_offsets(model,[0,1,2,0,1],reference,seconds,1.,100)
    np.testing.assert_allclose(result[20:80,:3],np.tile([.05,.05,.1],(60,1)))
    np.testing.assert_array_equal(result[99:],0.)
    reference[100:,:3]=100000.
    np.testing.assert_array_equal(result,source_prefix_base_offsets(model,[0,1,2,0,1],reference,seconds,1.,100))
    np.testing.assert_array_equal(source_prefix_base_offsets(model,[0,1,2,0,1],reference,seconds,0.,100),0.)
    import pytest
    with pytest.raises(ValueError,match='increasing clock'):
        source_prefix_base_offsets(model,[0,1,2,0,1],reference,np.zeros(200),1.,100)


def whole_base_model():
    return mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".001" gravity="0 0 0"/>
      <worldbody><body><joint name="x" type="slide" axis="1 0 0"/>
      <joint name="y" type="slide" axis="0 1 0"/><joint name="yaw" axis="0 0 1"/>
      <geom size=".1" mass="25"/></body></worldbody>
      <actuator><position joint="x" kp="3000" kv="600"/>
      <position joint="y" kp="3000" kv="600"/><position joint="yaw" kp="300" kv="60"/>
      </actuator></mujoco>''')


def test_whole_base_compensation_reduces_tracking_lag_and_preserves_model_clock():
    model = whole_base_model()
    seconds = np.arange(300)*.01
    reference = np.c_[.1*seconds, -.05*seconds, .2*seconds, np.ones(300)]
    before, clock = reference.copy(), seconds.copy()
    gains, bias = model.actuator_gainprm.copy(), model.actuator_biasprm.copy()
    offsets = base_velocity_offsets(model, [0, 1, 2, 0], reference, seconds, 1.)
    errors = []
    for scale in (0., 1.):
        data = mujoco.MjData(model)
        for i in range(200):
            data.ctrl[:] = reference[i, :3]+scale*offsets[i, :3]
            for _ in range(10):
                mujoco.mj_step(model, data)
        errors.append(np.abs(data.qpos-np.array([.1, -.05, .2])*data.time))
    assert np.all(errors[1] < errors[0]*.1)
    np.testing.assert_array_equal(offsets[:, 3:], 0.)
    np.testing.assert_array_equal(offsets[[0, -1]], 0.)
    np.testing.assert_array_equal(reference, before)
    np.testing.assert_array_equal(seconds, clock)
    np.testing.assert_array_equal(model.actuator_gainprm, gains)
    np.testing.assert_array_equal(model.actuator_biasprm, bias)
    np.testing.assert_array_equal(base_velocity_offsets(model, [0, 1, 2, 0], reference, seconds, 0.), 0.)


def test_whole_base_compensation_caps_offsets_and_checks_units_and_input_contract():
    import pytest
    model = whole_base_model()
    seconds = np.arange(100)*.01
    reference = np.tile(10.*seconds[:, None], (1, 3))
    result = base_velocity_offsets(model, [0, 1, 2], reference, seconds, .5)
    np.testing.assert_allclose(result[20:80], np.tile([.05, .05, .1], (60, 1)))
    for scale in (-.01, 1.01, np.nan):
        with pytest.raises(ValueError, match='scale'):
            base_velocity_offsets(model, [0, 1, 2], reference, seconds, scale)
    for clock in (seconds[:, None], np.zeros(100), seconds[:-1], np.r_[seconds[:-1], np.nan]):
        with pytest.raises(ValueError, match='clock'):
            base_velocity_offsets(model, [0, 1, 2], reference, clock, 1.)
    with pytest.raises(ValueError, match='units'):
        base_velocity_offsets(model, [2, 1, 0], reference, seconds, 1.)
    model.actuator_gear[0, 0] = 2.
    with pytest.raises(ValueError, match='units'):
        base_velocity_offsets(model, [0, 1, 2], reference, seconds, 1.)
    model.actuator_gear[0, 0] = 1.
    model.actuator_biasprm[0, 1] = -1.
    with pytest.raises(ValueError, match='position servos'):
        base_velocity_offsets(model, [0, 1, 2], reference, seconds, 1.)


def test_post_acquisition_compensation_uses_actual_resume_and_source_clock_with_exact_prefix():
    from reachy_retarget.acquisition_clock import AcquisitionClock
    model = whole_base_model()
    seconds = np.arange(220)*.01
    reference = np.c_[.1*seconds, -.05*seconds, .2*seconds]
    reference_before, clock_before = reference.copy(), seconds.copy()
    states, commands, errors = [], [], []
    for scale in (0., 1.):
        feedback = PostAcquisitionBaseOffsets(model, [0, 1, 2], reference, seconds, scale, 12)
        clock = AcquisitionClock(source_rows=len(seconds), first_close_index=5,
            acquisition_index=8, entry_rows=3, exit_source_indices=[9, 10, 11])
        data = mujoco.MjData(model)
        qpos, controls = [], []
        active_weights = []
        for frame in clock.frames():
            index = frame.reference_index
            offset = feedback.update(index, frame.phase, float(data.time))
            if index <= 12:
                np.testing.assert_array_equal(offset, 0.)
            else:
                active_weights.append(feedback.state['weight'])
            data.ctrl[:] = reference[index]+offset
            controls.append(data.ctrl.copy())
            for _ in range(10):
                mujoco.mj_step(model, data)
            qpos.append(data.qpos.copy())
            if frame.phase in ('settle', 'close'):
                clock.observe(alignment=True, bilateral_force=True)
            if index == 180:
                errors.append(np.abs(data.qpos-reference[index]))
        assert clock.complete
        assert active_weights[0] < active_weights[4] < active_weights[9]+1e-12
        assert active_weights[9] == 1.
        assert feedback.state['actual_resume_time_s'] > seconds[12]
        np.testing.assert_array_equal(offset, 0.)  # Original final boundary taper.
        states.append(np.array(qpos)); commands.append(np.array(controls))
    first_effect = np.flatnonzero(np.any(commands[0] != commands[1], axis=1))[0]
    np.testing.assert_array_equal(commands[0][:first_effect], commands[1][:first_effect])
    np.testing.assert_array_equal(states[0][:first_effect], states[1][:first_effect])
    assert np.all(errors[1] < errors[0]*.2)
    np.testing.assert_array_equal(reference, reference_before)
    np.testing.assert_array_equal(seconds, clock_before)


def test_post_acquisition_compensation_rejects_paused_suffix_and_clock_or_unit_mismatch():
    import pytest
    model = whole_base_model()
    seconds = np.arange(20)*.01
    reference = np.tile(seconds[:, None], (1, 3))
    for resume in (0, len(seconds), True):
        with pytest.raises(ValueError, match='source row'):
            PostAcquisitionBaseOffsets(model, [0, 1, 2], reference, seconds, 1., resume)
    with pytest.raises(ValueError, match='units'):
        PostAcquisitionBaseOffsets(model, [2, 1, 0], reference, seconds, 1., 5)
    with pytest.raises(ValueError, match='scale'):
        PostAcquisitionBaseOffsets(model, [0, 1, 2], reference, seconds, 1.01, 5)
    for bad in ('pause', 'clock', 'phase'):
        feedback = PostAcquisitionBaseOffsets(model, [0, 1, 2], reference, seconds, 1., 5)
        for i in range(6):
            feedback.update(i, 'source', float(seconds[i]))
        with pytest.raises(ValueError, match={'pause': 'paused', 'clock': 'resume clock', 'phase': 'source suffix'}[bad]):
            feedback.update(5 if bad == 'pause' else 6,
                'settle' if bad == 'phase' else 'source', .08 if bad == 'clock' else .06)
