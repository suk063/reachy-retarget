"""Pure tracking retargeting (reachy_retarget.tracking): generator, neck, pipeline, storage, separation."""
import json
import subprocess
import sys

import numpy as np
import pytest

from reachy_retarget.robot import LOWER, UPPER, Reachy
from reachy_retarget.robot.reachy import LEFT_ARM, NECK, RIGHT_ARM
from reachy_retarget.schema.episode import DT, SIDES
from reachy_retarget.schema.io import read_episode, write_episode
from reachy_retarget.tracking import neck as N
from reachy_retarget.tracking import primitives as pr
from reachy_retarget.tracking import synthetic as S
from reachy_retarget.tracking.build import run_scenario
from reachy_retarget.tracking.pipeline import track
from reachy_retarget.tracking.reference import TrackingReference

NS = "test"


def first(cell, extent=None, start=0, limit=400):
    """The first scenario of ``cell`` (and extent) in namespace ``test``."""
    for i in range(start, limit):
        sc = S.scenario(NS, "unit", i)
        if sc.cell == cell and (extent is None or sc.extent == extent):
            return sc
    raise LookupError(cell)


def test_scenarios_are_stratified_and_deterministic():
    n = len(S.CELLS)
    cells = [S.scenario(NS, "unit", i).cell for i in range(2 * n)]
    assert sorted(cells[:n]) == sorted(S.CELLS) and sorted(cells[n:]) == sorted(S.CELLS)
    a, b = S.scenario(NS, "unit", 7), S.scenario(NS, "unit", 7)
    assert a == b and a.seed == S.seed_of(a.source_id)
    ra, rb = S.generate(a), S.generate(b)
    for s in SIDES:
        assert np.array_equal(ra.tcp[s], rb.tcp[s])
    assert np.array_equal(ra.base, rb.base) and np.array_equal(ra.opening, rb.opening)


@pytest.mark.parametrize("cell", S.CELLS)
def test_every_cell_generates_a_valid_reference(cell):
    sc = first(cell)
    ref = S.generate(sc)
    assert isinstance(ref, TrackingReference) and not ref.retime
    assert np.allclose(np.diff(ref.time), DT)
    assert (ref.head is None) == (sc.neck == "off")
    assert ref.q_start is not None
    # the reference starts at the robot's start state
    fk = Reachy.load().fk(ref.q_start)
    for s in SIDES:
        assert np.allclose(fk[f"{s}_tcp"], ref.tcp[s][0], atol=1e-5)  # mirrored cells: URDF symmetric to ~2 um
    if ref.head is not None:
        assert N.rotation_error(fk["head"][:3, :3], ref.head[0]) < 1e-9
    assert ref.opening.min() >= 0 and ref.opening.max() <= 1
    moving_base = np.ptp(ref.base[:, :2], axis=0).max() > 1e-3 or np.ptp(ref.base[:, 2]) > 0.01
    assert moving_base == (cell in S.MOBILE)
    assert ref.regime == ("tabletop" if not moving_base else ref.regime)
    assert ref.lineage["seed"] == sc.source_id and ref.extra["scenario"]["cell"] == cell


def test_witness_reference_passes_tier_k():
    sc = first("witness_arms", "small")
    ref = S.generate(sc)
    assert ref.witness is not None
    ep = track(ref).episode
    assert ep.tier["K"]["passed"], ep.tier["K"]["reasons"]
    assert ep.family == "tracking" and ep.tier["P"] is None
    assert np.array_equal(ep.validation["witness_q"], ref.witness)
    for s in SIDES:  # the reference is stored unmodified
        assert np.array_equal(ep.reference.tcp[s], ref.tcp[s])


def _static_reference(z=None):
    """A 1 s reference holding the ready posture; ``z`` moves the left target to that height."""
    from reachy_retarget.tracking.pipeline import _nominal
    T = 51
    q0 = np.zeros(22)
    q0[LEFT_ARM] = np.radians([0, -10, 10, -90, 0, 0, 0])
    q0[RIGHT_ARM] = np.radians([0, 10, -10, -90, 0, 0, 0])
    fk = Reachy.load().fk(q0)
    tcp = {s: np.repeat(fk[f"{s}_tcp"][None], T, axis=0) for s in SIDES}
    if z is not None:
        tcp["left"][T // 2:, 2, 3] = z
    _ = _nominal
    return TrackingReference(dataset="tracking/unit", episode_id="static", task="hold", time=np.arange(T) * DT,
                             tcp=tcp, base=np.zeros((T, 3)), opening=np.ones((T, 2)),
                             head=np.repeat(fk["head"][None, :3, :3], T, axis=0), q_start=q0,
                             body_parts=("left_arm", "right_arm", "head"))


def test_unreachable_reference_is_a_named_k_failure_and_round_trips(tmp_path):
    ep = track(_static_reference(z=0.2)).episode
    assert not ep.tier["K"]["passed"]
    assert any("reach band" in r for r in ep.tier["K"]["reasons"])
    path = write_episode(tmp_path / "ep.h5", ep)
    back = read_episode(path)
    assert back.tier == ep.tier and back.family == "tracking"
    assert np.allclose(back.reference.tcp["left"], ep.reference.tcp["left"])
    assert np.allclose(back.reference.head, ep.reference.head)


def test_static_reference_passes():
    ep = track(_static_reference()).episode
    assert ep.tier["K"]["passed"], ep.tier["K"]["reasons"]
    assert ep.extra["tier_k_metrics"]["head_max_rot_residual"] < 1e-6


def test_neck_solver_round_trip_and_limits():
    rng = np.random.default_rng(0)
    lo, hi = N.limits(0.05)
    n = rng.uniform(lo, hi, (100, 3))
    R = N.head_base(n)[:, :3, :3]
    assert N.rotation_error(N.head_base(N.solve_base(R))[:, :3, :3], R).max() < 1e-6
    far = np.array([[0.0, 0.0, 1.6]])  # neck yaw beyond its 60 deg limit
    sol = N.solve_base(N.head_base(far)[:, :3, :3], margin=0.03)
    assert sol[0, 2] <= UPPER[NECK][2] - 0.03 + 1e-9
    limited = N.rate_limit(np.array([[0, 0, 0], [0, 0, 1.0], [0, 0, 1.0]]), np.zeros(3), np.full(2, 0.02))
    assert np.abs(np.diff(limited, axis=0)).max() <= np.radians(30) * 0.02 + 1e-12
    assert np.allclose(N.HEAD_TIP[:3, :3], np.eye(3))  # head_tip only translates the head frame


def test_mirror_matches_mirrored_joints():
    rng = np.random.default_rng(1)
    q = np.zeros(22)
    q[LEFT_ARM] = rng.uniform(LOWER[LEFT_ARM] + 0.2, UPPER[LEFT_ARM] - 0.2)
    q[RIGHT_ARM] = q[LEFT_ARM] * pr.MIRROR_SIGNS
    fk = Reachy.load().fk_base(q)
    assert np.allclose(pr.mirror(fk["left_tcp"]), fk["right_tcp"], atol=1e-5)


def test_track_continuity():
    P = np.eye(4)
    P[:3, 3] = [0.4, 0.2, 1.0]
    tr = pr.Track(P)
    speed = dict(lin=0.2, ang=0.6)
    Q = P.copy()
    Q[:3, 3] += [0.1, 0, 0]
    tr.add(*pr.reach(P, Q, speed), "reach")
    tr.add(*pr.line(tr.end, [0, 0, 1], 0.05, speed), "line")
    t = np.arange(0, tr.t + 0.5, DT)
    X = tr.sample(t)
    R = Q.copy()
    R[2, 3] += 0.05
    assert np.allclose(X[0], P) and np.allclose(X[-1], R)
    assert np.linalg.norm(np.diff(X[:, :3, 3], axis=0), axis=1).max() < 0.25 * DT * 1.01


def test_ik_does_not_use_the_witness():
    """The witness (the joint path that generated a reference) is stored for audit only."""
    sc = first("witness_arms", "small")
    ref = S.generate(sc)
    a = track(ref).episode
    ref.witness = None
    b = track(ref).episode
    assert np.array_equal(a.q, b.q)


def test_run_scenario_writes_episode_record_and_index(tmp_path):
    sc = first("gripper_only")
    rec = run_scenario(NS, "unit", sc.index, tmp_path)
    assert rec["K"]["passed"] and rec["P"] is None
    rows = rec["index_rows"]
    assert len(rows) == len(rec["attempts"]) and rows[-1]["family"] == "tracking"
    ep = read_episode(tmp_path / "episodes" / rows[-1]["file"])
    assert ep.uid == rec["uid"] and ep.lineage["seed"] == sc.source_id
    json.dumps(rec)


def test_mjviewer_replays_a_pod_record_on_the_mujoco_model(tmp_path):
    pytest.importorskip("trimesh")
    pytest.importorskip("mujoco")
    from reachy_retarget.tracking import mjviewer, viewer
    rec = run_scenario(NS, "unit", first("gripper_only").index, tmp_path)
    rel = rec["index_rows"][-1]["file"]
    path = viewer._resolve(tmp_path, f"/tmp/rr2/jobs/j/out/episodes/{rel}")  # as written on a pod
    assert path == tmp_path / "episodes" / rel
    model = mjviewer.display_model()
    visual = model.geom_group == mjviewer.VISUAL_GROUP
    assert visual.sum() > 50 and not model.geom_contype[visual].any() and not model.geom_conaffinity[visual].any()
    ep = read_episode(path)
    st = mjviewer.episode_states(model, ep)
    assert st["qpos"].shape == (ep.length, model.nq) and st["xpos"].shape == (ep.length, model.nbody, 3)
    assert model.nmocap == 0 and st["mocap_pos"].shape == (ep.length, 0, 3)  # tracking: no scene
    ep.q[:, LEFT_ARM.start] += 0.05
    with pytest.raises(ValueError, match="does not match"):
        mjviewer.episode_states(model, ep)


def test_mjviewer_textures_primitives_through_uv_meshes():
    mujoco = pytest.importorskip("mujoco")
    conversions = pytest.importorskip("mjviser.conversions")
    from reachy_retarget.tracking import mjviewer
    spec = mujoco.MjSpec.from_string("""<mujoco><asset>
      <texture name="cube" type="cube" builtin="checker" width="8" height="8" rgb1="1 0 0" rgb2="0 0 1"/>
      <texture name="flat" type="2d" builtin="checker" width="8" height="8"/>
      <material name="wood" texture="cube"/><material name="tile" texture="flat" texrepeat="2 2" texuniform="true"/>
      </asset><worldbody><body name="b" mocap="true">
      <geom name="box" type="box" size=".2 .1 .05" material="wood" contype="0" conaffinity="0"/>
      <geom name="floor" type="plane" size="1 1 .1" material="tile" contype="0" conaffinity="0"/>
      <geom name="solid" type="box" size=".1 .1 .1" material="wood"/>
      </body></worldbody></mujoco>""")
    assert mjviewer.uv_textured_primitives(spec) == 2
    m = spec.compile()
    uv = {}
    for name in ("box", "floor"):
        g = m.geom(name).id
        assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and conversions.get_geom_texture_id(m, g) >= 0
        mesh = m.geom_dataid[g]
        tc = m.mesh_texcoord[m.mesh_texcoordadr[mesh]:m.mesh_texcoordadr[mesh] + m.mesh_texcoordnum[mesh]]
        faces = m.mesh_facetexcoord[m.mesh_faceadr[mesh]:m.mesh_faceadr[mesh] + m.mesh_facenum[mesh]]
        uv[name] = tc[faces]  # (faces, 3 corners, 2)
    assert m.geom_type[m.geom("solid").id] == mujoco.mjtGeom.mjGEOM_BOX  # colliding geoms are left alone
    strip = np.floor(uv["box"][..., 1] * 6).astype(int)  # atlas face rows [k/6, (k+1)/6] of each corner
    centre = np.floor(uv["box"][..., 1].mean(1) * 6).astype(int)
    assert ((strip == centre[:, None]) | (strip == centre[:, None] + 1)).all()  # a triangle stays in one face
    assert np.bincount(centre, minlength=6).tolist() == [2] * 6  # two triangles per cube face
    np.testing.assert_allclose([uv["floor"].min(), uv["floor"].max()], [-2, 2])  # 2 repeats per metre


def test_tracking_does_not_load_the_manipulation_pipeline():
    code = ("import sys, reachy_retarget.tracking.build, reachy_retarget.tracking.synthetic, "
            "reachy_retarget.tracking.source, reachy_retarget.tracking.pipeline; "
            "print(sorted(m for m in sys.modules if m.startswith('reachy_retarget.retarget')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    loaded = set(eval(out))
    assert not loaded & {"reachy_retarget.retarget.pipeline", "reachy_retarget.retarget.targets",
                         "reachy_retarget.retarget.placement", "reachy_retarget.retarget.assign",
                         "reachy_retarget.retarget.gaze"}


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return 0 if not len(st) else 1 + int(np.sum((st[1:] - en[:-1]) * DT > 0.1))


@pytest.mark.parametrize("cell", S.CELLS)
def test_one_stage_each_part_acts_at_most_once(cell):
    """One scenario is one stage: every body part goes once from its start to one goal (or holds)."""
    from reachy_retarget.robot import planar
    from reachy_retarget.schema.rotations import so3_log
    for sc in [first(cell, start=k) for k in (0, 60)]:
        ref = S.generate(sc)
        W = np.linalg.inv(planar(ref.base[:, 0], ref.base[:, 1], ref.base[:, 2]))
        for s in SIDES:
            L = W @ ref.tcp[s]
            v = np.linalg.norm(np.diff(L[:, :3, 3], axis=0), axis=1) / DT
            w = np.linalg.norm(so3_log(np.swapaxes(L[:-1, :3, :3], 1, 2) @ L[1:, :3, :3]), axis=1) / DT
            assert _runs((v > 0.005) | (w > 0.02)) <= 1, (sc, s)
            assert _runs(np.abs(np.diff(ref.opening[:, SIDES.index(s)])) / DT > 0.01) <= 1, (sc, s)
        b = ref.base
        assert _runs((np.linalg.norm(np.diff(b[:, :2], axis=0), axis=1) / DT > 0.005)
                     | (np.abs(np.diff(b[:, 2])) / DT > 0.02)) <= 1, sc
        if ref.head is not None:
            Rb = np.swapaxes(planar(b[:, 0], b[:, 1], b[:, 2])[:, :3, :3], 1, 2) @ ref.head
            n = N.solve_base(Rb, None, 0.0, 30)
            assert _runs(np.abs(np.diff(n, axis=0)).max(axis=1) / DT > 0.02) <= 1, sc


def test_stage_cuts_start_one_movement_per_part_per_stage():
    from reachy_retarget.tracking import stages
    times = np.arange(100) * 0.1
    parts = {"hand": [(0, 20), (40, 60)], "grip": [(25, 30), (70, 75)]}
    bounds = stages.cuts(times, parts)
    # the hand's second movement forces a cut just before it; the gripper's second one fits that stage
    assert bounds == [0, 40, 99]
    for a, b in zip(bounds[:-1], bounds[1:]):
        for runs in parts.values():
            assert sum(a <= s < b for s, _ in runs) <= 1


def _pick_place_source():
    """A Panda-like pick and place: approach down, close, lift and carry, open, retreat."""
    from reachy_retarget.schema.source import Effector, SourceEpisode
    t = np.arange(0, 7.0, 0.05)
    p0 = np.array([0.45, -0.15, 0.95])
    down, carry, back = np.array([0, 0, -0.15]), np.array([0, 0.2, 0.15]), np.array([0, 0, 0.12])

    def ramp(a, b):
        return pr.min_jerk((t - a) / (b - a))[:, None]
    pos = p0 + ramp(0.3, 1.5) * down + ramp(2.4, 3.8) * carry + ramp(4.8, 6.0) * back
    pose = np.tile(np.eye(4), (len(t), 1, 1))
    pose[:, :3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])  # approach (+z) pointing down
    pose[:, :3, 3] = pos
    opening = 1.0 - 0.6 * ramp(1.7, 2.2)[:, 0] + 0.6 * ramp(4.0, 4.5)[:, 0]
    return SourceEpisode(family="synthetic", dataset="synthetic/pick", episode_id="pp0", task="pick",
                         time=t, effectors={"hand": Effector(pose=pose, opening=opening)},
                         lineage={"seed": "pp0"})


def test_source_demonstration_is_cut_into_single_stages():
    from reachy_retarget.robot import planar
    from reachy_retarget.tracking import source as TS
    from reachy_retarget.tracking import stages
    th = stages.Thresholds()
    refs = TS.references(_pick_place_source())
    primary = [r for r in refs if r.variant_of is None]
    assert len(primary) >= 3 and len(refs) == 2 * len(primary)  # single-arm source: both arms
    assert all(r.lineage["seed"] == "pp0" for r in refs)
    for ref in refs:
        t = ref.time - ref.time[0]
        W = np.linalg.inv(planar(ref.base[:, 0], ref.base[:, 1], ref.base[:, 2]))
        for s in SIDES:
            for runs in (stages.movements(t, pose=W @ ref.tcp[s]),
                         stages.movements(t, scalar=ref.opening[:, SIDES.index(s)])):
                assert sum(1 for a, b in runs if not (a == 0 and t[b - 1] <= th.tail)) <= 1, ref.episode_id
        Rb = np.swapaxes(planar(ref.base[:, 0], ref.base[:, 1], ref.base[:, 2])[:, :3, :3], 1, 2) @ ref.head
        assert N.movements(N.solve_base(Rb, None, 0.0, 30), t) <= 1
    rows = [r.extra["stage"]["source_rows"] for r in primary]
    assert rows[0][0] == 0 and all(a[1] == b[0] for a, b in zip(rows[:-1], rows[1:]))
