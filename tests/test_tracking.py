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


def first(cell, length=None, start=0, limit=400):
    """Index of the first scenario of ``cell`` (and length bucket) in namespace ``test``."""
    for i in range(start, limit):
        sc = S.scenario(NS, "unit", i)
        if sc.cell == cell and (length is None or sc.length == length):
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
    sc = first("witness_arms", "short")
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
    tr.add(*pr.circle(tr.end, 0.05, [0, 0, 1], 1, speed), "circle")
    t = np.arange(0, tr.t + 0.5, DT)
    X = tr.sample(t)
    assert np.allclose(X[0], P) and np.allclose(X[-1], Q)
    assert np.linalg.norm(np.diff(X[:, :3, 3], axis=0), axis=1).max() < 0.25 * DT * 1.01


def test_ik_does_not_use_the_witness():
    """The witness (the joint path that generated a reference) is stored for audit only."""
    sc = first("witness_arms", "short")
    ref = S.generate(sc)
    a = track(ref).episode
    ref.witness = None
    b = track(ref).episode
    assert np.array_equal(a.q, b.q)


def test_run_scenario_writes_episode_record_and_index(tmp_path):
    sc = first("hold", "short")
    rec = run_scenario(NS, "unit", sc.index, tmp_path)
    assert rec["K"]["passed"] and rec["P"] is None
    rows = rec["index_rows"]
    assert len(rows) == len(rec["attempts"]) and rows[-1]["family"] == "tracking"
    ep = read_episode(tmp_path / "episodes" / rows[-1]["file"])
    assert ep.uid == rec["uid"] and ep.lineage["seed"] == sc.source_id
    json.dumps(rec)


def test_tracking_does_not_load_the_manipulation_pipeline():
    code = ("import sys, reachy_retarget.tracking.build, reachy_retarget.tracking.synthetic, "
            "reachy_retarget.tracking.source, reachy_retarget.tracking.pipeline; "
            "print(sorted(m for m in sys.modules if m.startswith('reachy_retarget.retarget')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    loaded = set(eval(out))
    assert not loaded & {"reachy_retarget.retarget.pipeline", "reachy_retarget.retarget.targets",
                         "reachy_retarget.retarget.placement", "reachy_retarget.retarget.assign",
                         "reachy_retarget.retarget.gaze"}
