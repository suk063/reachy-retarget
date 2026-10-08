"""Tier-P physics validation on a synthetic scene: a table, a free box and a stand-in source robot."""
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget.robot import JOINTS, LOWER, UPPER, Reachy, gripper
from reachy_retarget.schema.episode import DT, PhysicsRollout, ReachyEpisode
from reachy_retarget.schema.source import ObjectTrack, SceneRef
from reachy_retarget.validate import physics
from reachy_retarget.validate.scene import MissingSceneAssets, build_scene, reset

TABLE_TOP = 0.70
BOX_HALF = (0.02, 0.02, 0.05)
BOX0 = np.array([0.45, -0.22, TABLE_TOP + BOX_HALF[2]])
SHIFT = np.array([0.0, -0.12, 0.0])  # put-down displacement
GRASP = BOX0 + [0.0, 0.0, 0.02]       # grasp center 2 cm above the box centre
# Horizontal grasp, forearm forward: approach +x world, closing axis -y world.
ROT = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], float)
OPEN, CLOSED = gripper.UPPER - 0.3, gripper.LOWER
T_CLOSE, T_RELEASE = 3.2, 8.6
BOX_OBJ = f"""v -1 -1 -1\nv 1 -1 -1\nv 1 1 -1\nv -1 1 -1\nv -1 -1 1\nv 1 -1 1\nv 1 1 1\nv -1 1 1
f 1 3 2\nf 1 4 3\nf 5 6 7\nf 5 7 8\nf 1 2 6\nf 1 6 5\nf 2 3 7\nf 2 7 6\nf 3 4 8\nf 3 8 7\nf 4 1 5\nf 4 5 8\n"""


def scene_xml(*, leg_mesh_file=None, texture_file=None):
    """Source-like MJCF: robosuite-style prefixed robot with actuator/sensor/weld, a mocap target."""
    leg = "" if leg_mesh_file is None else (
        '<geom name="table_leg" type="mesh" mesh="leg" pos="0 0.3 -0.2"/>'
        '<geom name="table_deco" type="mesh" mesh="deco" contype="0" conaffinity="0"/>')
    assets = "" if leg_mesh_file is None else (
        f'<mesh name="leg" file="{leg_mesh_file}" scale="0.02 0.02 0.02"/>'
        '<mesh name="deco" file="/nowhere/robosuite/models/assets/objects/meshes/deco.obj"/>')
    if texture_file:
        assets += (f'<texture name="wood" type="2d" file="{texture_file}"/>'
                   '<material name="wood_mat" texture="wood"/>')
    mat = ' material="wood_mat"' if texture_file else ""
    return f"""<mujoco model="synthetic"><compiler angle="radian" meshdir="/home/recorder/meshes"/>
<default><geom friction="1 .005 .0001"/></default>
<asset>{assets}<mesh name="robot0_link_mesh" file="/home/recorder/robosuite/models/assets/robots/panda/meshes/link0.stl"/></asset>
<worldbody><geom name="floor" type="plane" size="4 4 .1"/>
<body name="table" pos="0.62 -0.1 {TABLE_TOP / 2}">
  <geom name="table_top" type="box" size="0.2 0.35 {TABLE_TOP / 2}"{mat}/>{leg}</body>
<body name="box_main" pos="{BOX0[0]} {BOX0[1]} {BOX0[2] + 0.1}"><freejoint name="box_joint"/>
  <geom name="box_g" type="box" size="{BOX_HALF[0]} {BOX_HALF[1]} {BOX_HALF[2]}" mass="0.1"/></body>
<body name="robot0_base" pos="-0.5 0 0.9"><joint name="robot0_j" type="hinge"/>
  <geom type="mesh" mesh="robot0_link_mesh" mass="1"/>
  <body name="gripper0_pad"><joint name="gripper0_f" type="slide"/><geom size=".01" mass=".1"/></body></body>
<body name="left_eef_target" mocap="true" pos="0 0 2"><site name="left_eef_site"/></body>
</worldbody>
<actuator><position name="robot0_act" joint="robot0_j" kp="10"/><position name="gripper0_act" joint="gripper0_f" kp="1"/></actuator>
<equality><weld name="target_weld" body1="left_eef_target" body2="gripper0_pad"/></equality>
<sensor><jointpos name="robot0_js" joint="robot0_j"/><framepos name="box_pos" objtype="body" objname="box_main"/></sensor>
<contact><exclude body1="robot0_base" body2="table"/></contact>
<keyframe><key name="home" qpos="{' '.join(['0'] * 9)}"/></keyframe>
</mujoco>"""


def scene_ref(**kw):
    return SceneRef(mjcf=scene_xml(**kw), robot_prefixes=["robot0_", "gripper0_"],
                    initial_qpos={"box_joint": [*BOX0, 1, 0, 0, 0]})


def _ik(reachy, q, T_target, iters):
    """Damped least squares on the right arm with a null-space pull toward the start posture."""
    arm, rest = slice(10, 17), np.radians([0, 10, -10, -90, 0, 0, 0])
    for _ in range(iters):
        T, J = reachy.jacobian(q, "right_grasp")
        e = np.r_[T_target[:3, 3] - T[:3, 3], Rotation.from_matrix(T_target[:3, :3] @ T[:3, :3].T).as_rotvec()]
        Ja = J[:, arm]
        Jp = Ja.T @ np.linalg.inv(Ja @ Ja.T + 1e-4 * np.eye(6))
        q[arm] = np.clip(q[arm] + Jp @ e + (np.eye(7) - Jp @ Ja) @ (0.1 * (rest - q[arm])),
                         LOWER[arm] + 0.05, UPPER[arm] - 0.05)
    return q


@pytest.fixture(scope="module")
def trajectory():
    """Reach, close on the box, lift 8 cm, carry 12 cm sideways, put down, open, retreat (10 s)."""
    reachy = Reachy.load()
    g, s = GRASP, SHIFT
    wps = [(0.0, g + [-0.12, 0, 0], OPEN), (0.6, g + [-0.12, 0, 0], OPEN), (2.0, g, OPEN), (2.2, g, OPEN),
           (T_CLOSE, g, CLOSED), (3.4, g, CLOSED), (4.8, g + [0, 0, .08], CLOSED), (6.0, g + s + [0, 0, .08], CLOSED),
           (7.4, g + s + [0, 0, .002], CLOSED), (7.6, g + s + [0, 0, .002], CLOSED), (T_RELEASE, g + s + [0, 0, .002], OPEN),
           (8.8, g + s, OPEN), (10.0, g + s + [-0.06, 0, 0.06], OPEN)]
    wt = np.array([w[0] for w in wps])
    times = np.arange(0, wt[-1] + 1e-9, DT)
    q = np.zeros((len(times), 22))
    qc = np.zeros(22)
    qc[10:17] = np.radians([0, 10, -10, -90, 0, 0, 0])
    qc[3:10] = np.radians([0, -10, 10, -90, 0, 0, 0])
    qc[20] = OPEN
    T = np.eye(4)
    T[:3, :3] = ROT
    T[:3, 3] = wps[0][1]
    qc = _ik(reachy, qc, T, 300)
    for i, t in enumerate(times):
        k = min(np.searchsorted(wt, t, side="right") - 1, len(wt) - 2)
        a = (t - wt[k]) / (wt[k + 1] - wt[k])
        a = 3 * a * a - 2 * a ** 3
        T[:3, 3] = (1 - a) * wps[k][1] + a * wps[k + 1][1]
        qc = _ik(reachy, qc, T, 8)
        qc[21] = (1 - a) * wps[k][2] + a * wps[k + 1][2]
        q[i] = qc
    return times, q


def make_episode(times, q):
    reachy = Reachy.load()
    n = len(times)
    fkb = reachy.fk_base(q)
    # Source object track: the box follows the grasp center while held and is set down at the end.
    box = np.tile(np.r_[BOX0, 1, 0, 0, 0], (n, 1))
    gpos = reachy.fk(q)["right_grasp"][:, :3, 3]
    i_close, i_release = int(round(T_CLOSE / DT)), int(round(T_RELEASE / DT))
    if n > i_release:
        box[i_close:i_release + 1, :3] = gpos[i_close:i_release + 1] + (BOX0 - gpos[i_close])
        box[i_release + 1:, :3] = box[i_release, :3]
        box[i_release + 1:, 2] = BOX0[2]
    return ReachyEpisode(
        family="synthetic", dataset="synthetic/box", episode_id="0", task="pick and place the box",
        time=times, q=q, qd=np.gradient(q, DT, axis=0), tcp_base={"left": fkb["left_tcp"], "right": fkb["right_tcp"]},
        head_base=fkb["head"], gripper_opening=gripper.angle_to_opening(q[:, 20:22]),
        gripper_width=gripper.angle_to_width(q[:, 20:22]), source_time=times,
        tier={"K": {"passed": True, "reasons": []}, "P": None}, retarget_config="test",
        objects={"box": ObjectTrack(pose=box, valid=np.ones(n, bool), role="manipulated", geometry={"body": "box_main"}),
                 "table": ObjectTrack(pose=np.tile([0.62, -0.1, TABLE_TOP / 2, 1, 0, 0, 0], (n, 1)),
                                      valid=np.ones(n, bool), role="support")})


@pytest.fixture(scope="module")
def success(trajectory):
    ep = make_episode(*trajectory)
    return ep, *physics.simulate(ep, scene_ref())


def test_scene_replaces_source_robot():
    ref = scene_ref()
    scene = build_scene(ref)
    m = scene.model
    names = {m.body(b).name for b in range(m.nbody)}
    assert not {n for n in names if n.startswith(("robot0_", "gripper0_"))}
    assert "left_eef_target" not in names and m.nmocap == 0
    assert [m.actuator(i).name for i in range(m.nu)] == [f"reachy/{n}" for n in JOINTS]
    assert m.nsensor == 1 and m.sensor(0).name == "box_pos"  # object sensor kept, robot sensor removed
    assert m.nkey == 0 and m.neq == 8  # only Reachy's mimic couplings remain
    assert scene.free_bodies == {"box_joint": "box_main"}
    removed = scene.info["removed"]
    assert removed["mocap_bodies"] == ["left_eef_target"]
    assert set(removed["elements"]["actuator"]) == {"robot0_act", "gripper0_act"}
    assert "mesh:robot0_link_mesh" in scene.info["pruned_assets"]  # robot mesh file never needed
    d = reset(scene, np.zeros(22))
    adr = m.joint("box_joint").qposadr[0]
    np.testing.assert_allclose(d.qpos[adr:adr + 7], [*BOX0, 1, 0, 0, 0])  # initial state, not the MJCF pose
    assert ref.mjcf == scene_xml()  # the SceneRef is not modified


def test_scene_assets_from_dict_resolver_and_meshdir(tmp_path):
    rec = "/home/recorder/robosuite/models/assets/objects/meshes/leg.obj"
    ref = scene_ref(leg_mesh_file=rec, texture_file="/home/recorder/textures/wood.png")
    ref.assets = {"objects/meshes/leg.obj": BOX_OBJ.encode()}  # archive-relative key, absolute file attr
    scene = build_scene(ref)
    assert scene.model.geom("table_leg").type == mujoco.mjtGeom.mjGEOM_MESH
    missing = {(r["tag"], r["name"]) for r in scene.info["missing_visual_assets"]}
    assert missing == {("mesh", "deco"), ("texture", "wood")}  # visual-only: dropped and recorded
    assert scene.info["assets"][0]["found"] == "assets:objects/meshes/leg.obj"

    ref.assets = {}
    calls = []
    scene = build_scene(ref, asset_resolver=lambda f: calls.append(f) or (BOX_OBJ.encode() if f == rec else None))
    assert rec in calls and scene.info["assets"][0]["found"] == "resolver"

    (tmp_path / "objects" / "meshes").mkdir(parents=True)
    (tmp_path / "objects" / "meshes" / "leg.obj").write_text(BOX_OBJ)
    scene = build_scene(ref, meshdir=tmp_path)
    assert scene.info["assets"][0]["found"].startswith("meshdir:")

    with pytest.raises(MissingSceneAssets):
        build_scene(ref)
    tier, rollout = physics.simulate(make_episode(np.arange(3) * DT, np.zeros((3, 22))), ref)
    assert rollout is None and not tier["passed"] and tier["reasons"][0].startswith("scene: missing collision")


def test_pick_and_place_passes(success):
    ep, tier, rollout = success
    gates = tier["metrics"]["gates"]
    assert tier["passed"], tier["reasons"]
    assert all(gates.values()) and tier["reasons"] == []
    assert set(gates) >= {"rollout_complete", "robot_environment_penetration", "object_environment_penetration",
                          "robot_self_penetration", "self_clearance", "joint_margin", "arm_speed", "base_speed",
                          "tcp_tracking", "grasp_drift", "task_final_pose", "actuator_replay"}
    grasp = tier["metrics"]["grasps"]["right/box"]
    assert grasp["acquired"] and grasp["phases"] == 1 and grasp["carry_s"] > 3.0
    task = tier["metrics"]["task"]["box"]
    assert task["max_lift_m"] > 0.06 and task["position_error_m"] < 0.01
    # the source box rests at its end: final pose after the hold, rest required
    assert task["rule"] == "after_hold" and task["position_error_m"] == task["after_hold_position_error_m"]
    assert tier["metrics"]["objects_at_rest_rule"]["box_main"]["required"]
    # The rollout is on the simulator clock at 50 Hz: settle + episode + hold.
    assert isinstance(rollout, PhysicsRollout)
    info = rollout.info
    assert len(rollout.time) == round((info["settle_s"] + ep.duration + info["hold_s"]) / DT) + 1
    np.testing.assert_allclose(np.diff(rollout.time), DT, atol=1e-9)
    assert rollout.ctrl_names == [f"reachy/{n}" for n in JOINTS]
    k = round(info["settle_s"] / DT) + len(ep.time) // 2  # mid-episode row: reference q at that instant
    every = round(DT / info["timestep"])
    ref = physics.control_sequence(ep, info["timestep"], info["settle_s"], info["hold_s"])
    np.testing.assert_allclose(ref[k * every - 1], ep.q[len(ep.time) // 2], atol=1e-9)
    # ... and the servos receive it led by kv / kp (velocity reference), fingers only while closing
    lead = np.asarray(info["ctrl_lead_s"])
    np.testing.assert_allclose(lead[3:17], 24.0 / 180.0)
    led = physics.control_sequence(ep, info["timestep"], info["settle_s"], info["hold_s"], lead)
    np.testing.assert_allclose(rollout.ctrl[k], led[k * every - 1], atol=1e-9)
    # The box starts from the source initial state (not the MJCF body pose) and is never written.
    box = [rollout.qpos_names.index(f"box_joint/{c}") for c in ("x", "y", "z")]
    np.testing.assert_allclose(rollout.qpos[0, box], BOX0)


def test_gripper_never_closing_fails_the_task(trajectory):
    times, q = trajectory
    q = q.copy()
    q[:, 21] = OPEN
    tier, rollout = physics.simulate(make_episode(times, q), scene_ref(), physics.PhysicsConfig(replay=False))
    gates = tier["metrics"]["gates"]
    assert not tier["passed"] and not gates["task_final_pose"]
    assert any(r.startswith("task_final_pose") for r in tier["reasons"])
    assert tier["metrics"]["task"]["box"]["position_error_m"] > 0.1  # the box stayed where it was
    assert tier["metrics"]["grasps"] == {}
    assert not gates["actuator_replay"]  # not run = not passed
    assert rollout is not None  # failed rollouts are returned for saving


def test_source_end_motion_uses_the_source_clock():
    """End speed over the last source window; the target row is where the source clock reaches
    the last frame (the retargeted episode may hold it longer)."""
    n = 40
    times = np.arange(n) * DT
    ep = make_episode(times, np.zeros((n, 22)))
    ep.source_time = np.minimum(times, times[29])     # rows 29.. hold the last source frame
    box = ep.objects["box"]
    box.pose[:, 0] = BOX0[0] + 0.5 * ep.source_time   # 0.5 m/s along x until the source ends
    mo = physics.source_end_motion(ep, box, 0.1, 0.02, 0.2)
    assert mo["source_row"] == n - 1 and mo["target_row"] == 29
    assert mo["linear_m_s"] == pytest.approx(0.5) and mo["window_s"] == pytest.approx(0.1) and mo["at_rest"] is False
    box.pose[:, 0] = BOX0[0]
    assert physics.source_end_motion(ep, box, 0.1, 0.02, 0.2)["at_rest"] is True
    box.valid[:] = False
    assert physics.source_end_motion(ep, box, 0.1, 0.02, 0.2) is None
    box.valid[-1] = True  # one valid row: speed unknown
    assert physics.source_end_motion(ep, box, 0.1, 0.02, 0.2)["at_rest"] is None


def test_object_moving_at_source_end_is_compared_at_the_source_end(trajectory):
    """A source that ends while the box still moves (RollBall-like): task_final_pose compares
    the pose when the retargeted trajectory reaches the source's last frame, and rest is only
    reported for that object."""
    ep = make_episode(*trajectory)
    box = ep.objects["box"]
    box.pose[-6:-1, 0] -= 0.05  # the source's last 0.1 s: the box arrives at 0.5 m/s
    tier, rollout = physics.simulate(ep, scene_ref(), physics.PhysicsConfig(replay=False))
    m = tier["metrics"]
    task = m["task"]["box"]
    assert task["rule"] == "source_end" and task["source_end_speed"]["linear_m_s"] == pytest.approx(0.5)
    assert task["position_error_m"] == task["source_end_position_error_m"] < 0.01
    assert task["source_end_time_s"] == pytest.approx(rollout.info["settle_s"] + ep.duration)
    assert m["gates"]["task_final_pose"] and m["gates"]["objects_at_rest"]
    rule = m["objects_at_rest_rule"]["box_main"]
    assert rule["object"] == "box" and not rule["required"] and not m["objects_final_speed"]["box_main"]["required"]
    assert "source moving" in rule["reason"]


def test_actuator_only_replay_is_deterministic(success):
    ep, tier, rollout = success
    assert tier["metrics"]["actuator_replay_max_abs_diff"] <= physics.THRESHOLDS["replay_absolute_tolerance"]
    scene = build_scene(scene_ref())
    m = scene.model
    d = reset(scene, ep.q[0])
    state = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(m, d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    info = rollout.info
    ctrl = physics.control_sequence(ep, m.opt.timestep, info["settle_s"], info["hold_s"], info["ctrl_lead_s"])
    act = np.array([m.actuator(f"reachy/{n}").id for n in JOINTS])
    every = round(DT / m.opt.timestep)
    qpos, qvel = physics.replay(m, state, ctrl, act, every)
    np.testing.assert_allclose(qpos, rollout.qpos, rtol=0, atol=1e-7)
    np.testing.assert_allclose(qvel, rollout.qvel, rtol=0, atol=1e-7)
    np.testing.assert_allclose(ctrl[every - 1::every], rollout.ctrl[1:, act], rtol=0, atol=0)


def test_parked_free_bodies_are_removed():
    """Free bodies that no episode object refers to and that start far from the workspace are
    dropped (robosuite parks unused objects overlapping at (10, 10, 10))."""
    from reachy_retarget.validate.scene import parked_bodies
    xml = scene_xml().replace("</worldbody>", "".join(
        f'<body name="{n}_main" pos="10 10 10"><freejoint name="{n}_joint"/><geom size=".03" mass=".1"/></body>'
        for n in ("milk", "bread")) + "</worldbody>")
    ref = SceneRef(mjcf=xml, robot_prefixes=["robot0_", "gripper0_"],
                   initial_qpos={"box_joint": [*BOX0, 1, 0, 0, 0], "milk_joint": [10, 10, 10, 1, 0, 0, 0],
                                 "bread_joint": [10, 10, 10, 1, 0, 0, 0]})
    parked = parked_bodies(ref, {"box_main"}, np.array([[0.6, 0.0, 0.8]]), 2.0)
    assert set(parked) == {"milk_main", "bread_main"}
    assert parked_bodies(ref, {"box_main"}, np.array([[9.0, 9.0, 9.0]]), 2.0) == {}
    scene = build_scene(ref, drop_bodies=sorted(parked))
    assert set(scene.free_bodies.values()) == {"box_main"}
    assert scene.info["removed"]["parked_bodies"] == ["milk_main", "bread_main"]


def test_finger_commands_lead_only_while_closing():
    times = np.arange(11) * DT
    q = np.zeros((11, 22))
    q[:, 20] = np.r_[np.linspace(2.0, 1.0, 6), np.linspace(1.0, 2.0, 6)[1:]]  # close, then open
    ep = make_episode(times, q)
    lead = np.full(22, 0.04)
    led = physics.control_sequence(ep, 0.002, 0.0, 0.0, lead)
    now = physics.control_sequence(ep, 0.002, 0.0, 0.0)
    closing, opening = slice(0, 30), slice(60, 90)
    assert np.all(led[closing, 20] < now[closing, 20] - 1e-9)    # led while closing
    np.testing.assert_allclose(led[opening, 20], now[opening, 20])  # never opens early


def test_object_environment_gate_is_source_relative():
    """Soft source contacts let objects rest a few mm deep; the gate allows the source's own
    reference depth + 1 mm (never less than the absolute 2 mm) and records both verdicts."""
    xml = scene_xml().replace('mass="0.1"/></body>', 'mass="0.1" solref="0.3 1"/></body>', 1)
    ep = make_episode(np.arange(3) * DT, np.zeros((3, 22)))
    cfg = physics.PhysicsConfig(replay=False)
    base = dict(mjcf=xml, robot_prefixes=["robot0_", "gripper0_"], initial_qpos={"box_joint": [*BOX0, 1, 0, 0, 0]})
    tier, _ = physics.simulate(ep, SceneRef(**base), cfg)
    m = tier["metrics"]
    depth = m["max_depth_m"]["object_environment"]
    assert depth > 0.003  # the soft box rests deeper than the absolute gate
    assert not m["gates"]["object_environment_penetration"] and not m["object_environment_passed_absolute"]
    assert m["object_environment_reference_depth_m"] is None and m["object_environment_threshold_m"] == 0.002
    assert any(r.startswith("object_environment_penetration") and "absolute threshold" in r for r in tier["reasons"])

    tier, _ = physics.simulate(ep, SceneRef(**base, reference={"object_environment_depth_m": depth - 0.0005}), cfg)
    m = tier["metrics"]
    assert m["gates"]["object_environment_penetration"] and not m["object_environment_passed_absolute"]
    assert m["object_environment_threshold_m"] == pytest.approx(depth + 0.0005)
    assert "source-relative" in m["object_environment_threshold_rule"]

    tier, _ = physics.simulate(ep, SceneRef(**base, reference={"object_environment_depth_m": depth - 0.002}), cfg)
    assert not tier["metrics"]["gates"]["object_environment_penetration"]
    assert any("source reference" in r for r in tier["reasons"] if r.startswith("object_environment"))


def test_declared_inactive_bodies_replace_the_distance_rule():
    xml = scene_xml().replace("</worldbody>", '<body name="milk_main" pos="0.3 0 2"><freejoint name="milk_joint"/>'
                              '<geom size=".03" mass=".1"/></body></worldbody>')
    ep = make_episode(np.arange(3) * DT, np.zeros((3, 22)))
    ref = SceneRef(mjcf=xml, robot_prefixes=["robot0_", "gripper0_"], inactive_bodies=["milk_main"],
                   initial_qpos={"box_joint": [*BOX0, 1, 0, 0, 0], "milk_joint": [0.3, 0, 2, 1, 0, 0, 0]})
    tier, rollout = physics.simulate(ep, ref, physics.PhysicsConfig(replay=False))
    assert rollout.info["scene"]["removed"]["parked_bodies"] == ["milk_main"]  # near, but declared inactive
    assert tier["metrics"]["removed_inactive_bodies"]["rule"].startswith("declared")
