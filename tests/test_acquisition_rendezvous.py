import numpy as np
import pytest
from reachy_retarget.acquisition_rendezvous import Rendezvous
from reachy_retarget.native_replay.bigym_acquisition import line_intervals, choose_chord, face_margin


def test_waits_require_contiguous_actual_guards_and_have_bounded_timeout():
    r = Rendezvous(timestep=.01)
    r.begin()
    for _ in range(9):
        assert r.update(alignment=True, bilateral_force=True) == 'settle'
    r.update(alignment=False, bilateral_force=True)
    for _ in range(10):
        r.update(alignment=True, bilateral_force=False)
    assert r.phase == 'close'
    for _ in range(200):
        r.update(alignment=True, bilateral_force=False)
    assert r.phase == 'failed' and r.failure == 'close_timeout'
    assert r.hold_steps == 220
    assert r.report()['inserted_duration_s'] == 2.2


def test_bilateral_cannot_skip_alignment_or_settle():
    r = Rendezvous(timestep=.002)
    r.begin()
    for _ in range(100):
        r.update(alignment=False, bilateral_force=True)
    assert r.phase == 'settle'
    for _ in range(50):
        r.update(alignment=True, bilateral_force=False)
    assert r.phase == 'close'
    for _ in range(50):
        r.update(alignment=True, bilateral_force=True)
    assert r.phase == 'carry' and not r.failure
    with pytest.raises(ValueError):
        r.begin()
    with pytest.raises(ValueError):
        Rendezvous(timestep=.002, wait_limit_s=2.01)


def test_finite_geometry_does_not_inflate_disconnected_plate_parts():
    from scipy.spatial import ConvexHull
    vertices = np.array([[x, y, z] for x in (-.004, .004) for y in (-.015, .015) for z in (-.01, .01)])
    eq = ConvexHull(vertices).equations
    interval = choose_chord(line_intervals([('plate', eq)], np.array([.002, 0, 0]), np.array([1., 0, 0])))
    assert np.allclose(interval[1:], [-.006, .002])
    assert np.isclose((interval[1]+interval[2])/2, -.002)
    assert not line_intervals([('plate', eq)], np.array([0, .02, 0]), np.array([1., 0, 0]))
    face = np.array([[0, y, z] for y in (-.015, .015) for z in (-.01, .01)])
    assert np.isclose(face_margin(face, [0, 0, 0]), .01)
    assert face_margin(face, [0, .02, 0]) < 0


def test_minimum_patch_correction_has_tangent_only_when_finite_face_requires():
    from reachy_retarget.native_replay.bigym_acquisition import minimum_patch_translation
    faces = [np.array([[x, y, z] for y in (-.015, .015) for z in (-.01, .01)]) for x in [-.004, .004]]
    shift = minimum_patch_translation(faces, [.002, .018, 0], np.array([1., 0, 0]), .002)
    assert np.allclose(shift, [.002, .007, 0], atol=1e-9)
    assert np.linalg.norm(shift) < .01
    assert min(face_margin(v+shift, [.002, .018, 0]) for v in faces) >= .004-1e-9
    axis_only = minimum_patch_translation(faces, [0, 0, 0], np.array([1., 0, 0]), .002)
    assert np.allclose(axis_only, [.002, 0, 0])


def test_nonpad_guard_catches_palm_even_when_fingers_are_allowed():
    mujoco = pytest.importorskip('mujoco')
    from reachy_retarget.native_replay.bigym_acquisition import nonpad_object_clearance
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="robot"><geom name="palm" type="box" size=".02 .02 .02"/>
        <body name="pad" pos=".021 0 0"><geom type="box" size=".002 .01 .01"/></body>
      </body>
      <body name="object" pos=".025 0 0"><freejoint/>
        <geom name="plate" type="box" size=".01 .02 .02"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    before = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.time)
    kwargs = dict(robot_bodies={model.body('robot').id, model.body('pad').id},
                  object_bodies={model.body('object').id},
                  allowed_pad_bodies={model.body('pad').id})
    report = nonpad_object_clearance(model, data, **kwargs)
    assert not report['admitted_nonpad_only']
    assert report['violating_pairs'][0]['robot_geom'] == 'palm'
    assert report['minimum_signed_distance_or_lower_bound_m'] == pytest.approx(-.005)
    assert all(np.array_equal(a, b) for a, b in zip(before[:3], (data.qpos, data.qvel, data.ctrl)))
    assert data.time == before[3]
    # A distant free object passes the non-pad check without changing masks.
    data.qpos[0] = .1
    mujoco.mj_forward(model, data)
    assert nonpad_object_clearance(model, data, **kwargs)['admitted_nonpad_only']
    with pytest.raises(ValueError):
        nonpad_object_clearance(model, data, **{**kwargs, 'allowed_pad_bodies': {model.body('object').id}})
