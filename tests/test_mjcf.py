import mujoco
import numpy as np
import pytest

from reachy_retarget.robot import JOINTS, LOWER, UPPER, Reachy, gripper, mjcf


@pytest.fixture(scope="module")
def model():
    xml, assets = mjcf.reachy_mjcf("rb_")
    return mujoco.MjModel.from_xml_string(xml, assets)


def test_compiles_with_actuators_in_canonical_order(model):
    assert model.nu == len(JOINTS) == 22
    assert [model.actuator(i).name for i in range(model.nu)] == [f"rb_{n}" for n in JOINTS]
    for i, n in enumerate(JOINTS):  # every actuator drives the joint of the same name
        assert model.joint(int(model.actuator_trnid[i, 0])).name == f"rb_{n}"
    assert model.neq == 8  # 4 mimic joints per hand
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC


def test_fk_matches_kinematic_model(model):
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    lo = np.where(np.isfinite(LOWER), LOWER, -2.0)
    hi = np.where(np.isfinite(UPPER), UPPER, 2.0)
    reachy = Reachy.load()
    for _ in range(10):
        q = rng.uniform(lo, hi)
        mjcf.set_reachy_state(model, data, q, "rb_")
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(mjcf.reachy_q(model, data, "rb_"), q)
        for frame, T in reachy.fk(q).items():
            s = model.site(f"rb_{frame}").id
            np.testing.assert_allclose(data.site_xpos[s], T[:3, 3], atol=1e-6)
            np.testing.assert_allclose(data.site_xmat[s].reshape(3, 3), T[:3, :3], atol=1e-6)


def test_finger_mimic_coupling_and_pad_contact(model):
    data = mujoco.MjData(model)
    q = np.zeros(22)
    q[20:22] = 1.2
    mjcf.set_reachy_state(model, data, q, "rb_")
    q[20:22] = gripper.LOWER  # command a squeeze of the empty hands
    data.ctrl[:] = q
    for _ in range(500):
        mujoco.mj_step(model, data)
    f = data.qpos[model.joint("rb_r_hand_finger").qposadr[0]]
    assert f < 0.2
    for j, (src, mult, off) in mjcf.mimic_joints().items():
        value = data.qpos[model.joint(f"rb_{j}").qposadr[0]]
        source = data.qpos[model.joint(f"rb_{src}").qposadr[0]]
        assert abs(value - (mult * source + off)) < 2e-3, j
    pads = {model.body(f"rb_{b}").id for b in mjcf.FINGER_BODIES["right"]}
    touching = [c for c in data.contact[:data.ncon]
                if {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]} == pads]
    assert touching, "the pads of an empty closed hand must touch"


def test_attach_keeps_reachy_parameters_and_excludes_wheels():
    scene = """<mujoco><default><geom friction="0.2 0 0" solref="0.05 1"/><joint damping="7"/></default>
      <worldbody><geom name="floor" type="plane" size="3 3 .1"/>
      <body name="box" pos="1 0 .1"><freejoint/><geom type="box" size=".1 .1 .1"/></body></worldbody></mujoco>"""
    spec = mujoco.MjSpec.from_string(scene)
    mjcf.attach_reachy(spec, prefix="reachy/", pos=(0.5, 0.2, 0.0))
    m = spec.compile()
    assert [m.actuator(i).name for i in range(m.nu)] == [f"reachy/{n}" for n in JOINTS]
    pad = m.geom("reachy/r_hand_distal_link_c0_0")
    np.testing.assert_allclose(pad.friction, [1, 0.01, 0.001])
    np.testing.assert_allclose(pad.solref, [0.004, 1])
    assert m.joint("reachy/r_shoulder_pitch").damping[0] == pytest.approx(mjcf.JOINT_DAMPING)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    np.testing.assert_allclose(d.xpos[m.body("reachy/base_link").id], [0.5, 0.2, 0.0], atol=1e-12)
    robot = {b for b in range(m.nbody) if m.body(b).name.startswith("reachy/")}
    assert not [c for c in d.contact[:d.ncon] if {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} & robot]
