"""Tier P: physics validation of a retargeted episode in the source's MuJoCo scene.

`simulate(ep, scene_ref, cfg)` builds the scene (`validate.scene.build_scene`: source robot
removed, Reachy attached at the world origin; the bodies the adapter declares inactive in
`SceneRef.inactive_bodies` are removed; when an adapter declares none (`None`), free bodies that
no episode object track refers to and whose initial position lies more than
`cfg.park_distance_m` from every object position, TCP and base position of the episode are
removed as parked instead, see `scene.parked_bodies`), resets it once and then drives it with
actuator commands only:

1. **Reset.** MJCF defaults, the source's initial object qpos (`SceneRef.initial_qpos`) and
   Reachy at `ep.q[0]` (mimic finger joints coupled), `ctrl = ep.q[0]`, `mj_forward`. The
   integration state is saved for the replay check.
2. **Settle** for `cfg.settle_s` with `ctrl = ep.q[0]` (objects come to rest on their supports).
3. **Track** the episode: during physics step k the position servos receive the canonical
   `q` linearly interpolated at the end of the step (episode time `t0 + (k + 1) dt - settle_s`)
   plus the servo's lead `kv / kp` (`cfg.velocity_reference`, `servo_lead`): a position servo
   `kp (ctrl - q) - kv qd` commanded `q_ref(t + kv/kp) ~ q_ref + (kv/kp) qd_ref` is the PD servo
   with velocity reference `kp (q_ref - q) + kv (qd_ref - qd)`, so it tracks without the
   `kv/kp * speed` lag (0.13 s for the arms). The commands are a deterministic function of the
   retargeted `q` only. The finger actuators receive the commanded finger angles `q[:, 20:22]`
   (lead only while closing: their command is the more closed of the led and the current
   reference, so a hand stalled on an object never opens before its reference). Nothing else is written: objects are never welded, teleported or have
   their state overwritten.
4. **Hold** the final `q` for `cfg.hold_s` so the final object pose is a resting pose.

`qpos`, `qvel` and the applied `ctrl` are recorded every `1 / 50 s` on the simulator clock
(row 0 = reset state; episode time `t0` is at simulator time `settle_s`) into a
`PhysicsRollout`. Per-step `ctrl` is not stored: it is the deterministic interpolation
`control_sequence(ep, ...)` and is regenerated for the replay check.

Gates (thresholds: `THRESHOLDS`, copied from legacy `physical_gates.THRESHOLDS`, plus the tier-P
additions marked new):

| gate | criterion |
| --- | --- |
| `rollout_complete` | every planned step ran, the state stayed finite, no MuJoCo BADQACC/BADQPOS/BADQVEL/BADCTRL warning |
| `robot_environment_penetration` | Reachy vs non-object scene geometry depth <= 2 mm (every step) |
| `object_environment_penetration` | free object vs scene/other objects depth <= max(2 mm, source reference + 1 mm), source-relative (every step after the settle phase; while settling, objects released from the source's initial state may drop onto their supports, e.g. robosuite cubes start about 1 cm above the table, and the impact depth is reported as `settle_object_environment_depth_m`, not gated). The source reference `SceneRef.reference["object_environment_depth_m"]` is the deepest object-environment contact of the source's own recorded states (soft contacts: the robomimic Square nut rests 7.6 mm deep in the table); without it the absolute 2 mm applies. The reference, the applied threshold and the absolute verdict are recorded (`object_environment_reference_depth_m`, `object_environment_threshold_m`, `object_environment_passed_absolute`) |
| `hand_object_penetration` | Reachy hand links vs objects depth <= 1 mm |
| `no_nonhand_object_contact` | no object contact with a Reachy link outside the hands |
| `robot_self_penetration` | Reachy self contacts depth <= 2 mm (contacts between the finger links of one hand are reported, not gated: closing an empty hand presses its pads together) |
| `self_clearance` | sphere-model self clearance (robot.collision) of measured q >= 9 mm |
| `joint_margin` | measured arm and neck joints >= 0.025 rad inside the URDF limits |
| `arm_speed`, `neck_speed`, `base_speed` | measured arm <= 1.001 rad/s, neck <= 30 deg/s + 0.001 (new), base body-frame vx, vy <= 0.611 m/s, wz <= 114 deg/s + 0.001 |
| `tcp_tracking` | measured TCP (`{l,r}_arm_tip`) vs FK of the retargeted q at the same instant (the reference, not the lead-shifted ctrl): position <= 3 cm, rotation <= 0.2 rad (new) |
| `grasp_drift` | while a hand is commanded closed (finger reference below the 0.5 opening angle, or the episode's grasp label `validation["grasp_object"]` set: large objects stall the fingers above the 0.5 opening) and holds an object lifted by > 15 mm with both pads (positive normal force), the object pose relative to the grasp-center site drifts <= 3 mm / 3 deg from its pose at acquisition |
| `carry_contact` | both pads touch the carried object during >= 95 % of each carry |
| `task_final_pose` | every `manipulated` object with a scene body is within `cfg.object_position_tol_m` (default 3 cm, about a can radius; new) of its final pose in `ep.objects` (the source's final object pose): after the hold when the source object is at rest at its end, else at the instant the retargeted trajectory reaches the source's last frame (see *Source end motion*); orientation is reported only (symmetric objects) |
| `objects_at_rest` | at the end of the hold every free object moves < 0.02 m/s and < 0.2 rad/s; required for every free body except tracked objects that still move at the end of the source (reported only, see *Source end motion*) |
| `actuator_replay` | re-running from the saved reset state with only the recorded ctrl sequence reproduces every recorded qpos/qvel within 1e-7 |

*Source end motion.* Some sources end while an object still moves (ManiSkill RollBall stops at
the first success with the ball rolling at 0.6-1.0 m/s); a faithful replay then cannot leave it
at the source's final pose after the hold, nor at rest. Per tracked object
(`source_end_motion`): the source's end is its last valid row ``e`` and the retargeted
trajectory reaches it at the first row whose ``ep.source_time`` reaches ``source_time[e]``; the
source speed is the pose change over the last ``cfg.source_rest_window_s`` (0.1 s) of source time
before it. Below the rest thresholds (0.02 m/s, 0.2 rad/s) the object is *at rest* and both
gates apply as written (``rule = "after_hold"``, rest required). Otherwise ``task_final_pose``
compares the rollout pose at that instant (simulator time ``settle_s + time[row] - time[0]``)
with the source's final pose (``rule = "source_end"``), and its final speed is reported, not
gated. Free bodies without a track, or whose source speed cannot be measured, keep the rest
requirement. The rule, the source end speed and both errors are recorded per object
(``metrics["task"][id]``, ``metrics["objects_at_rest_rule"]``).

Missing measurements fail: e.g. no manipulated object with a scene body fails
`task_final_pose`. `simulate` returns `({"passed", "reasons", "metrics"}, rollout)`; store
`{"passed", "reasons"}` as `ep.tier["P"]`. When the scene cannot be built the rollout is `None`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..robot import JOINTS, Reachy
from ..robot import gripper as rgripper
from ..robot.collision import min_clearance
from ..robot.mjcf import FINGER_BODIES
from ..schema.episode import DT, PhysicsRollout, ReachyEpisode
from ..schema.source import SceneRef
from .scene import MissingSceneAssets, Scene, build_scene, parked_bodies, reset, subtree

THRESHOLDS = dict(  # legacy physical_gates.THRESHOLDS (+ new tier-P entries at the end)
    arm_speed_rad_s=1.001, base_vx_m_s=.611, base_vy_m_s=.611, base_wz_rad_s=float(np.deg2rad(114) + .001),
    joint_margin_rad=.025, self_clearance_m=.009, hand_object_depth_m=.001, robot_environment_depth_m=.002,
    object_environment_depth_m=.002, robot_self_depth_m=.002, acquisition_lift_m=.015,
    carry_bilateral_fraction=.95, grasp_translation_m=.003, grasp_rotation_rad=float(np.deg2rad(3)),
    final_stable_s=1., stable_linear_speed_m_s=.02, stable_angular_speed_rad_s=.2,
    positive_contact_force_n=1e-6, replay_absolute_tolerance=1e-7,
    # new in tier P
    neck_speed_rad_s=float(np.deg2rad(30) + .001), tcp_position_m=.03, tcp_rotation_rad=.2,
    object_position_m=.03,
    # object-environment depth allowed above the source's own reference depth (source-relative gate)
    object_environment_reference_margin_m=.001)
ARM_NECK = JOINTS[3:20]
NECK_JOINTS = JOINTS[17:20]
BAD_WARNINGS = ("mjWARN_BADQACC", "mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADCTRL")
# Finger commands below this angle count as "closed" (the 0.5 opening point).
CLOSED_ANGLE = float(rgripper.opening_to_angle(0.5))
_ENV, _ROBOT, _HAND, _OBJECT = 0, 1, 2, 3


@dataclass
class PhysicsConfig:
    settle_s: float = 0.5
    hold_s: float = 1.0
    prefix: str = "reachy/"
    floor_z: float = 0.0
    object_position_tol_m: float = THRESHOLDS["object_position_m"]
    object_bodies: dict[str, str] | None = None   # object id -> scene body (default: geometry["body"] or id)
    replay: bool = True
    velocity_reference: bool = True  # lead each servo command by its kv/kp (see servo_lead)
    park_distance_m: float = 2.0   # untracked free bodies farther than this from the workspace are removed
    source_rest_window_s: float = 0.1  # source time over which the end speed of an object is measured
    thresholds: dict = field(default_factory=lambda: dict(THRESHOLDS))


def control_sequence(ep: ReachyEpisode, timestep: float, settle_s: float, hold_s: float, lead=None) -> np.ndarray:
    """(n_steps, 22) servo commands: q interpolated at the end of every physics step, each joint
    ``lead`` (22,) seconds ahead (default 0; clamped to the episode end). The fingers (columns
    20, 21) take the more closed of their led and current reference: the lead acts while a hand
    closes and never opens a hand before its reference does."""
    n_settle, n_track, n_hold = (int(round(x / timestep)) for x in (settle_s, ep.duration, hold_s))
    k = np.arange(n_settle + n_track + n_hold)
    t0 = ep.time[0] + (k + 1) * timestep - n_settle * timestep

    def at(t):
        t = np.clip(t, ep.time[0], ep.time[-1])
        i = np.clip(np.searchsorted(ep.time, t, side="right") - 1, 0, len(ep.time) - 2)
        w = (t - ep.time[i]) / (ep.time[i + 1] - ep.time[i])
        cols = np.arange(22)[None]
        return (1 - w) * ep.q[i, cols] + w * ep.q[i + 1, cols]

    now = at(np.repeat(t0[:, None], 22, axis=1))
    if lead is None or not np.any(lead):
        return now
    ahead = at(t0[:, None] + np.asarray(lead, float)[None])
    ahead[:, 20:22] = np.minimum(ahead[:, 20:22], now[:, 20:22])
    return ahead


def servo_lead(model, act) -> np.ndarray:
    """Per-actuator kv / kp (s) of position servos: commanding q(t + kv/kp) equals a PD servo
    with position and velocity reference, kp (q_ref - q) + kv (qd_ref - qd), to first order."""
    kp, kv = model.actuator_gainprm[act, 0], -model.actuator_biasprm[act, 2]
    return np.where(kp > 0, kv / np.maximum(kp, 1e-12), 0.0)


def _names(model):
    qpos, qvel = [], []
    for j in range(model.njnt):
        n, t = model.joint(j).name, int(model.jnt_type[j])
        if t == mujoco.mjtJoint.mjJNT_FREE:
            qpos += [f"{n}/{c}" for c in ("x", "y", "z", "qw", "qx", "qy", "qz")]
            qvel += [f"{n}/{c}" for c in ("vx", "vy", "vz", "wx", "wy", "wz")]
        elif t == mujoco.mjtJoint.mjJNT_BALL:
            qpos += [f"{n}/{c}" for c in ("qw", "qx", "qy", "qz")]
            qvel += [f"{n}/{c}" for c in ("wx", "wy", "wz")]
        else:
            qpos.append(n)
            qvel.append(n)
    return qpos, qvel


def _rot_angle(R):
    return float(Rotation.from_matrix(R).magnitude())


class _Monitor:
    """Contact, speed and margin measurements; reads MuJoCo data only."""

    def __init__(self, scene: Scene, th: dict):
        m, p = scene.model, scene.prefix
        self.m, self.th = m, th
        self.robot = scene.reachy_bodies()
        self.free_roots = {m.body(b).id for b in scene.free_bodies.values()}
        cls = np.full(m.nbody, _ENV)
        self.root_of = np.full(m.nbody, -1)
        for r in self.free_roots:
            for b in subtree(m, r):
                cls[b], self.root_of[b] = _OBJECT, r
        for b in self.robot:
            cls[b] = _HAND if "hand_" in m.body(b).name else _ROBOT
        self.cls = cls
        self.finger_group = np.full(m.nbody, -1)
        self.pads = {}
        for k, (side, s) in enumerate((("left", "l"), ("right", "r"))):
            for link in ("proximal_link", "proximal_mimic_link", "distal_link", "distal_mimic_link"):
                self.finger_group[m.body(f"{p}{s}_hand_{link}").id] = k
            self.pads[side] = tuple(m.body(f"{p}{b}").id for b in FINGER_BODIES[side])
        self.depth = {k: 0.0 for k in ("robot_environment", "object_environment", "hand_object", "robot_self",
                                       "finger_finger")}
        self.depth_event = {k: None for k in self.depth}
        self.settling = False          # object-environment contacts while settling go to settle_depth
        self.settle_depth, self.settle_event = 0.0, None
        self.nonhand_object_contacts = 0
        self.nonhand_event = None
        jid = [m.joint(f"{p}{n}").id for n in ARM_NECK]
        self.qadr = m.jnt_qposadr[jid]
        self.vadr = m.jnt_dofadr[jid]
        self.range = m.jnt_range[jid]
        self.is_neck = np.array([n in NECK_JOINTS for n in ARM_NECK])
        self.base_q = [m.jnt_qposadr[m.joint(f"{p}base_{c}").id] for c in ("x", "y", "yaw")]
        self.base_v = [m.jnt_dofadr[m.joint(f"{p}base_{c}").id] for c in ("x", "y", "yaw")]
        self.peak_joint = np.zeros(len(ARM_NECK))
        self.peak_base = np.zeros(3)
        self.min_margin = np.inf
        self.margin_joint = None

    def step(self, d):
        n = d.ncon
        if n:
            g = d.contact.geom[:n]
            b = self.m.geom_bodyid[g]
            c = self.cls[b]
            depth = np.maximum(0.0, -d.contact.dist[:n])
            robot = (c == _ROBOT) | (c == _HAND)
            obj = c == _OBJECT
            r1, r2, o1, o2 = robot[:, 0], robot[:, 1], obj[:, 0], obj[:, 1]
            fg = self.finger_group[b]
            same_hand = (fg[:, 0] >= 0) & (fg[:, 0] == fg[:, 1])
            kinds = {
                "robot_environment": (r1 & (c[:, 1] == _ENV)) | (r2 & (c[:, 0] == _ENV)),
                "object_environment": (o1 & ~r2) | (o2 & ~r1),
                "hand_object": ((c[:, 0] == _HAND) & o2) | ((c[:, 1] == _HAND) & o1),
                "robot_self": r1 & r2 & ~same_hand,
                "finger_finger": same_hand,
            }
            for k, mask in kinds.items():
                if mask.any():
                    i = int(np.argmax(np.where(mask, depth, -1)))
                    if self.settling and k == "object_environment":
                        if depth[i] > self.settle_depth:
                            self.settle_depth = float(depth[i])
                            self.settle_event = {"depth_m": float(depth[i]), "time_s": float(d.time),
                                                 "bodies": [self.m.body(int(x)).name for x in b[i]]}
                        continue
                    if depth[i] > self.depth[k]:
                        self.depth[k] = float(depth[i])
                        self.depth_event[k] = {"depth_m": float(depth[i]), "time_s": float(d.time),
                                               "geoms": [self.m.geom(int(x)).name or f"geom{int(x)}" for x in g[i]],
                                               "bodies": [self.m.body(int(x)).name for x in b[i]]}
            bad = ((c[:, 0] == _ROBOT) & o2) | ((c[:, 1] == _ROBOT) & o1)
            if bad.any():
                self.nonhand_object_contacts += int(bad.sum())
                if self.nonhand_event is None:
                    i = int(np.argmax(bad))
                    self.nonhand_event = {"time_s": float(d.time),
                                          "bodies": [self.m.body(int(x)).name for x in b[i]]}
        v = np.abs(d.qvel[self.vadr])
        self.peak_joint = np.maximum(self.peak_joint, v)
        yaw = d.qpos[self.base_q[2]]
        vx, vy, wz = d.qvel[self.base_v]
        c, s = np.cos(yaw), np.sin(yaw)
        self.peak_base = np.maximum(self.peak_base, np.abs([c * vx + s * vy, -s * vx + c * vy, wz]))
        q = d.qpos[self.qadr]
        margin = np.minimum(q - self.range[:, 0], self.range[:, 1] - q)
        k = int(np.argmin(margin))
        if margin[k] < self.min_margin:
            self.min_margin, self.margin_joint = float(margin[k]), ARM_NECK[k]

    def pad_contacts(self, d) -> dict[str, set[int]]:
        """{side: object roots touched by BOTH pads of that hand with positive normal force}."""
        touched = {side: {pad: set() for pad in pads} for side, pads in self.pads.items()}
        n = d.ncon
        if not n:
            return {side: set() for side in self.pads}
        g = d.contact.geom[:n]
        b = self.m.geom_bodyid[g]
        force = np.zeros(6)
        for i in np.flatnonzero(((self.cls[b[:, 0]] == _HAND) & (self.cls[b[:, 1]] == _OBJECT))
                                | ((self.cls[b[:, 1]] == _HAND) & (self.cls[b[:, 0]] == _OBJECT))):
            mujoco.mj_contactForce(self.m, d, int(i), force)
            if force[0] <= self.th["positive_contact_force_n"]:
                continue
            hand, obj = (b[i, 0], b[i, 1]) if self.cls[b[i, 0]] == _HAND else (b[i, 1], b[i, 0])
            for side, pads in self.pads.items():
                if hand in pads:
                    touched[side][hand].add(int(self.root_of[obj]))
        return {side: set.intersection(*per_pad.values()) for side, per_pad in touched.items()}


def _site_pose(d, sid):
    T = np.eye(4)
    T[:3, :3] = d.site_xmat[sid].reshape(3, 3)
    T[:3, 3] = d.site_xpos[sid]
    return T


def _body_pose(d, bid):
    T = np.eye(4)
    T[:3, :3] = d.xmat[bid].reshape(3, 3)
    T[:3, 3] = d.xpos[bid]
    return T


def replay(model, initial_state: np.ndarray, ctrl_steps: np.ndarray, act: np.ndarray, every: int):
    """Actuator-only rerun: restore the reset state once, then write only ctrl.

    Returns (qpos rows, qvel rows) every `every` steps, row 0 = reset state.
    """
    d = mujoco.MjData(model)
    mujoco.mj_setState(model, d, initial_state, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, d)
    qpos, qvel = [d.qpos.copy()], [d.qvel.copy()]
    for k in range(len(ctrl_steps)):
        d.ctrl[act] = ctrl_steps[k]
        mujoco.mj_step(model, d)
        if (k + 1) % every == 0:
            qpos.append(d.qpos.copy())
            qvel.append(d.qvel.copy())
    return np.array(qpos), np.array(qvel)


def source_end_motion(ep: ReachyEpisode, track, window_s: float, linear_m_s: float, angular_rad_s: float):
    """End of one object's source track (see the module docstring, *Source end motion*):
    ``{"source_row", "target_row", "window_s", "linear_m_s", "angular_rad_s", "at_rest"}`` or
    None without valid rows. ``target_row`` is the first episode row whose source time reaches the
    source time of the last valid row; speeds are None (and ``at_rest`` None) when the window holds
    no earlier source time."""
    valid = np.flatnonzero(track.valid)
    if not len(valid):
        return None
    e = int(valid[-1])
    st = ep.source_time
    t_end = float(st[e])
    target = int(np.argmax(st >= t_end - 1e-9))
    out = {"source_row": e, "target_row": target, "window_s": None, "linear_m_s": None, "angular_rad_s": None,
           "at_rest": None}
    early = valid[(st[valid] >= t_end - window_s - 1e-9) & (st[valid] < t_end - 1e-9)]
    if not len(early):
        return out
    i0 = int(early[0])
    dt = t_end - float(st[i0])
    a, b = track.pose[i0], track.pose[e]
    rot = Rotation.from_quat(np.r_[b[4:], b[3]]) * Rotation.from_quat(np.r_[a[4:], a[3]]).inv()
    lin, ang = float(np.linalg.norm(b[:3] - a[:3]) / dt), float(rot.magnitude() / dt)
    out.update(window_s=dt, linear_m_s=lin, angular_rad_s=ang, at_rest=bool(lin < linear_m_s and ang < angular_rad_s))
    return out


def _track_body(oid, track, free, cfg):
    candidates = [(cfg.object_bodies or {}).get(oid), track.geometry.get("body"), oid, f"{oid}_main"]
    return next((c for c in candidates if c and c in free), None)


def _object_bodies(ep: ReachyEpisode, scene: Scene, cfg: PhysicsConfig):
    m = scene.model
    free = set(scene.free_bodies.values())
    out, missing = {}, []
    for oid, track in ep.objects.items():
        if track.role != "manipulated":
            continue
        body = _track_body(oid, track, free, cfg)
        if body is None:
            missing.append(oid)
        else:
            out[oid] = m.body(body).id
    return out, missing


def _parked(ep: ReachyEpisode, scene_ref: SceneRef, cfg: PhysicsConfig) -> dict[str, float | None]:
    """Bodies removed before the rollout: ``scene_ref.inactive_bodies`` when the adapter declares
    them (value None), else untracked free bodies parked far from the workspace
    (scene.parked_bodies, value = distance): the workspace is every valid object position of the
    episode, the robot's TCP path and its base path."""
    if scene_ref.inactive_bodies is not None:
        return {b: None for b in scene_ref.inactive_bodies}
    keep = set((cfg.object_bodies or {}).values())
    pts = [np.c_[ep.q[:, :2], np.zeros(len(ep.q))]]
    for oid, track in ep.objects.items():
        keep |= {oid, f"{oid}_main", track.geometry.get("body")}
        pts.append(track.pose[track.valid, :3])
    pts += [X[:, :3, 3] for X in ep.tcp_world.values()]
    return parked_bodies(scene_ref, keep, np.concatenate(pts), cfg.park_distance_m)


def simulate(ep: ReachyEpisode, scene_ref: SceneRef, cfg: PhysicsConfig | None = None, *,
             asset_resolver=None, meshdir=None):
    """Run tier P; returns ({"passed", "reasons", "metrics"}, PhysicsRollout | None)."""
    cfg = cfg or PhysicsConfig()
    th = cfg.thresholds
    parked = _parked(ep, scene_ref, cfg)
    try:
        scene = build_scene(scene_ref, prefix=cfg.prefix, floor_z=cfg.floor_z,
                            asset_resolver=asset_resolver, meshdir=meshdir, drop_bodies=sorted(parked))
    except (MissingSceneAssets, ValueError) as e:
        return {"passed": False, "reasons": [f"scene: {e}"], "metrics": {}}, None
    m, p = scene.model, cfg.prefix
    timestep = float(m.opt.timestep)
    every = int(round(DT / timestep))
    if not np.isclose(every * timestep, DT, rtol=0, atol=1e-12):
        raise ValueError(f"physics timestep {timestep} does not divide the {DT} s record period")
    settle_s, hold_s = round(cfg.settle_s / DT) * DT, round(cfg.hold_s / DT) * DT
    act = np.array([m.actuator(f"{p}{n}").id for n in JOINTS])
    lead = servo_lead(m, act) if cfg.velocity_reference else np.zeros(len(act))
    ref_steps = control_sequence(ep, timestep, settle_s, hold_s)
    ctrl_steps = control_sequence(ep, timestep, settle_s, hold_s, lead)
    n_steps = len(ctrl_steps)
    objects, unmapped = _object_bodies(ep, scene, cfg)

    d = reset(scene, ep.q[0])
    initial_state = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(m, d, initial_state, mujoco.mjtState.mjSTATE_INTEGRATION)
    mon = _Monitor(scene, th)
    n_settle = int(round(settle_s / timestep))
    mon.settling = n_settle > 0
    mon.step(d)
    sites = {s: m.site(f"{p}{s}_tcp").id for s in ("left", "right")}
    grasp_sites = {s: m.site(f"{p}{s}_grasp").id for s in ("left", "right")}
    z0 = {o: float(d.xpos[b][2]) for o, b in objects.items()}
    carry = {(s, o): dict(anchor=None, phases=0, acquired=False, carry_s=0.0, bilateral_s=0.0, translation=0.0,
                          rotation=0.0, phase_bilateral=[]) for s in ("left", "right") for o in objects}

    rows = {"time": [0.0], "qpos": [d.qpos.copy()], "qvel": [d.qvel.copy()], "ctrl": [d.ctrl.copy()],
            "ref": [ep.q[0].copy()],
            "tcp": [[_site_pose(d, sites[s]) for s in ("left", "right")]]}
    obj_rows = {o: [np.r_[d.xpos[b], d.xquat[b]]] for o, b in objects.items()}
    finite, steps_done, warnings = True, 0, {}

    labels = np.asarray(ep.validation.get("grasp_object", np.full((ep.length, 2), -1)))

    def held_at(k):
        """Retargeting grasp labels (left, right) at the end of physics step k."""
        t = ep.time[0] + (k + 1) * timestep - n_settle * timestep
        if t < ep.time[0] or t > ep.time[-1]:
            return (False, False)
        i = min(int(np.searchsorted(ep.time, t, side="right")) - 1, ep.length - 1)
        return tuple(bool(x) for x in labels[i] >= 0)

    def grasp_update(d, cmd, held):
        holds = mon.pad_contacts(d)
        for (side, o), st in carry.items():
            i = 0 if side == "left" else 1
            closed = cmd[20 + i] < CLOSED_ANGLE or held[i]
            b = objects[o]
            bilateral = b in holds[side]
            lift = float(d.xpos[b][2]) - z0[o]
            rel = np.linalg.inv(_site_pose(d, grasp_sites[side])) @ _body_pose(d, b)
            if not closed:
                st["anchor"] = None
                continue
            if st["anchor"] is None and bilateral and lift > th["acquisition_lift_m"]:
                st["anchor"], st["acquired"] = rel, True
                st["phases"] += 1
            if st["anchor"] is not None:
                delta = np.linalg.inv(st["anchor"]) @ rel
                st["translation"] = max(st["translation"], float(np.linalg.norm(delta[:3, 3])))
                st["rotation"] = max(st["rotation"], _rot_angle(delta[:3, :3]))
                st["carry_s"] += DT
                st["bilateral_s"] += DT * bilateral

    for k in range(n_steps):
        d.ctrl[act] = ctrl_steps[k]
        mujoco.mj_step(m, d)
        steps_done += 1
        mon.settling = k + 1 < n_settle
        mon.step(d)
        if (k + 1) % every == 0:
            if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                finite = False
                break
            rows["time"].append((k + 1) * timestep)
            rows["qpos"].append(d.qpos.copy())
            rows["qvel"].append(d.qvel.copy())
            rows["ctrl"].append(d.ctrl.copy())
            rows["ref"].append(ref_steps[k])
            rows["tcp"].append([_site_pose(d, sites[s]) for s in ("left", "right")])
            for o, b in objects.items():
                obj_rows[o].append(np.r_[d.xpos[b], d.xquat[b]])
            grasp_update(d, ref_steps[k], held_at(k))
    for w in BAD_WARNINGS:
        count = int(d.warning[int(getattr(mujoco.mjtWarning, w))].number)
        if count:
            warnings[w] = count

    qpos_names, qvel_names = _names(m)
    ctrl_names = [m.actuator(i).name for i in range(m.nu)]
    rollout = PhysicsRollout(
        time=np.array(rows["time"]), qpos=np.array(rows["qpos"]), qvel=np.array(rows["qvel"]),
        ctrl=np.array(rows["ctrl"]), qpos_names=qpos_names, qvel_names=qvel_names, ctrl_names=ctrl_names,
        info={"simulator": f"mujoco {mujoco.__version__}", "timestep": timestep, "record_hz": int(round(1 / DT)),
              "settle_s": settle_s, "hold_s": hold_s, "episode_time_offset_s": settle_s - float(ep.time[0]),
              "control": "position servos only; ctrl = canonical q linearly interpolated at the end of each "
                         "physics step plus the joint's lead ctrl_lead_s = kv/kp (a PD servo with velocity "
                         "reference; settle: q[0], hold: q[-1]); fingers follow q[:, 20:22]",
              "ctrl_lead_s": lead.tolist(),
              "reachy_prefix": p, "object_bodies": {o: m.body(b).name for o, b in objects.items()},
              "scene": scene.info,
              "assumptions": ["ideal planar base servos, wheels not simulated (wheel/floor contact excluded)",
                              "Reachy gravity-compensated (gravcomp=1); objects are not",
                              "Reachy collision = URDF primitives + convex hulls of collider-mesh components",
                              "global options overridden to elliptic cones, impratio 10, implicitfast, 2 ms",
                              "objects keep their source mass, friction and contact parameters"]})

    # ------------------------------------------------------------------ metrics
    n_rows = len(rows["time"])
    ref_q = np.array(rows["ref"])
    meas_q = np.array([q[[m.jnt_qposadr[m.joint(f'{p}{n}').id] for n in JOINTS]] for q in rows["qpos"]])
    fk = Reachy.load().fk(ref_q)
    tcp_actual = np.array(rows["tcp"])
    pos_err, rot_err = np.zeros((n_rows, 2)), np.zeros((n_rows, 2))
    for i, side in enumerate(("left", "right")):
        ref = fk[f"{side}_tcp"]
        pos_err[:, i] = np.linalg.norm(tcp_actual[:, i, :3, 3] - ref[:, :3, 3], axis=-1)
        rel = np.swapaxes(ref[:, :3, :3], 1, 2) @ tcp_actual[:, i, :3, :3]
        rot_err[:, i] = Rotation.from_matrix(rel).magnitude()
    clearance = min_clearance(meas_q)

    motion = {o: source_end_motion(ep, t, cfg.source_rest_window_s, th["stable_linear_speed_m_s"],
                                   th["stable_angular_speed_rad_s"]) for o, t in ep.objects.items()}
    obj_arr = {o: np.array(r) for o, r in obj_rows.items()}

    def pose_error(final, target):
        rq = Rotation.from_quat(np.r_[final[4:], final[3]]) * Rotation.from_quat(np.r_[target[4:], target[3]]).inv()
        return float(np.linalg.norm(final[:3] - target[:3])), float(rq.magnitude())

    task, task_ok = {}, bool(objects) and not unmapped
    for o, b in objects.items():
        track, mo = ep.objects[o], motion[o]
        if mo is None:
            task[o] = {"error": "no valid source pose"}
            task_ok = False
            continue
        target = track.pose[mo["source_row"]]
        rows_o = obj_arr[o]
        final = rows_o[-1]
        # rollout row at which the retargeted trajectory reaches the source's last frame of this object
        r_end = min(int(round((settle_s + ep.time[mo["target_row"]] - ep.time[0]) / DT)), len(rows_o) - 1)
        at_end = rows_o[r_end]
        err_hold, rot_hold = pose_error(final, target)
        err_end, rot_end = pose_error(at_end, target)
        rule = "source_end" if mo["at_rest"] is False else "after_hold"
        err, rot = (err_end, rot_end) if rule == "source_end" else (err_hold, rot_hold)
        task[o] = {"rule": rule, "final_pose": final.tolist(), "source_final_pose": target.tolist(),
                   "position_error_m": err, "rotation_error_rad": rot,
                   "after_hold_position_error_m": err_hold, "source_end_position_error_m": err_end,
                   "source_end_pose": at_end.tolist(), "source_end_time_s": float(rollout.time[r_end]),
                   "source_end_speed": {k: mo[k] for k in ("linear_m_s", "angular_rad_s", "window_s")},
                   "max_lift_m": float(np.max(rows_o[:, 2]) - z0[o])}
        task_ok &= err <= cfg.object_position_tol_m
    free = set(scene.free_bodies.values())
    body_track = {}
    for oid, track in ep.objects.items():
        body = _track_body(oid, track, free, cfg)
        if body is not None:
            body_track.setdefault(body, oid)
    grasp_ids = list(ep.extra.get("grasp_object_ids", []))
    held_at_end = {grasp_ids[i] for i in labels[-1] if 0 <= i < len(grasp_ids)}
    rest, rest_rule = {}, {}
    vel = np.zeros(6)
    for name in scene.free_bodies.values():
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, m.body(name).id, vel, 0)
        rest[name] = {"linear_m_s": float(np.linalg.norm(vel[3:])), "angular_rad_s": float(np.linalg.norm(vel[:3]))}
        oid = body_track.get(name)
        mo = motion.get(oid) if oid else None
        if oid is None:
            rest_rule[name] = {"required": True, "object": None, "reason": "no source track"}
        elif mo is None or mo["at_rest"] is None:
            rest_rule[name] = {"required": True, "object": oid, "reason": "source end speed unknown"}
        elif oid in held_at_end:  # still in a Reachy hand after the hold: it must be motionless
            rest_rule[name] = {"required": True, "object": oid, "reason": "held by a hand at the episode end"}
        else:
            rest_rule[name] = {"required": bool(mo["at_rest"]), "object": oid,
                               "reason": f"source {'at rest' if mo['at_rest'] else 'moving'} at its end: "
                                         f"{mo['linear_m_s']:.3f} m/s {mo['angular_rad_s']:.3f} rad/s"}
        rest[name]["required"] = rest_rule[name]["required"]

    grasps = {f"{s}/{o}": {k: v for k, v in st.items() if k not in ("anchor", "phase_bilateral")}
              for (s, o), st in carry.items() if st["acquired"] or st["carry_s"]}
    for g in grasps.values():
        g["bilateral_fraction"] = g["bilateral_s"] / g["carry_s"] if g["carry_s"] else None

    replay_diff = None
    if cfg.replay and finite and steps_done == n_steps:
        rq_, rv_ = replay(m, initial_state, ctrl_steps, act, every)
        if rq_.shape == rollout.qpos.shape:
            replay_diff = float(max(np.abs(rq_ - rollout.qpos).max(), np.abs(rv_ - rollout.qvel).max()))

    peak_arm = mon.peak_joint[~mon.is_neck]
    peak_neck = mon.peak_joint[mon.is_neck]
    # Object-environment gate, source-relative: max(absolute, source reference + margin).
    oe_depth, oe_abs = mon.depth["object_environment"], th["object_environment_depth_m"]
    oe_ref = (scene_ref.reference or {}).get("object_environment_depth_m")
    if oe_ref is None:
        oe_th, oe_rule = oe_abs, f"absolute threshold {oe_abs:.4f} m (no source reference)"
    else:
        margin = th["object_environment_reference_margin_m"]
        oe_th = max(oe_abs, float(oe_ref) + margin)
        oe_rule = (f"source-relative threshold {oe_th:.4f} m = max(absolute {oe_abs:.4f} m, source reference "
                   f"{oe_ref:.4f} m + {margin:.4f} m), {'source reference' if oe_th > oe_abs else 'absolute'} "
                   f"applies; absolute verdict {'pass' if oe_depth <= oe_abs else 'fail'}")
    gates = {
        "rollout_complete": bool(finite and steps_done == n_steps and not warnings),
        "robot_environment_penetration": mon.depth["robot_environment"] <= th["robot_environment_depth_m"],
        "object_environment_penetration": oe_depth <= oe_th,
        "hand_object_penetration": mon.depth["hand_object"] <= th["hand_object_depth_m"],
        "no_nonhand_object_contact": mon.nonhand_object_contacts == 0,
        "robot_self_penetration": mon.depth["robot_self"] <= th["robot_self_depth_m"],
        "self_clearance": bool(clearance.min() >= th["self_clearance_m"]),
        "joint_margin": mon.min_margin >= th["joint_margin_rad"],
        "arm_speed": bool(peak_arm.max() <= th["arm_speed_rad_s"]),
        "neck_speed": bool(peak_neck.max() <= th["neck_speed_rad_s"]),
        "base_speed": bool(mon.peak_base[0] <= th["base_vx_m_s"] and mon.peak_base[1] <= th["base_vy_m_s"]
                           and mon.peak_base[2] <= th["base_wz_rad_s"]),
        "tcp_tracking": bool(pos_err.max() <= th["tcp_position_m"] and rot_err.max() <= th["tcp_rotation_rad"]),
        "grasp_drift": all(g["translation"] <= th["grasp_translation_m"] and g["rotation"] <= th["grasp_rotation_rad"]
                           for g in grasps.values() if g["acquired"]),
        "carry_contact": all(g["bilateral_fraction"] is None or g["bilateral_fraction"] >= th["carry_bilateral_fraction"]
                             for g in grasps.values()),
        "task_final_pose": bool(task_ok),
        "objects_at_rest": all(r["linear_m_s"] < th["stable_linear_speed_m_s"]
                               and r["angular_rad_s"] < th["stable_angular_speed_rad_s"]
                               for r in rest.values() if r["required"]),
        "actuator_replay": replay_diff is not None and replay_diff <= th["replay_absolute_tolerance"],
    }
    details = {
        "rollout_complete": f"steps {steps_done}/{n_steps}, finite={finite}, warnings={warnings}",
        "robot_environment_penetration": f"{mon.depth['robot_environment']:.4f} m",
        "object_environment_penetration": f"{oe_depth:.4f} m > {oe_rule}",
        "hand_object_penetration": f"{mon.depth['hand_object']:.4f} m",
        "no_nonhand_object_contact": f"{mon.nonhand_object_contacts} contacts, first {mon.nonhand_event}",
        "robot_self_penetration": f"{mon.depth['robot_self']:.4f} m",
        "self_clearance": f"{clearance.min():.4f} m",
        "joint_margin": f"{mon.min_margin:.4f} rad at {mon.margin_joint}",
        "arm_speed": f"{peak_arm.max():.3f} rad/s",
        "neck_speed": f"{peak_neck.max():.3f} rad/s",
        "base_speed": f"body vx, vy, wz = {np.round(mon.peak_base, 3).tolist()}",
        "tcp_tracking": f"{pos_err.max():.4f} m, {rot_err.max():.4f} rad",
        "grasp_drift": "; ".join(f"{k}: {g['translation']:.4f} m, {g['rotation']:.4f} rad"
                                 for k, g in grasps.items() if g["acquired"]) or "no carry",
        "carry_contact": "; ".join(f"{k}: {g['bilateral_fraction']}" for k, g in grasps.items()) or "no carry",
        "task_final_pose": ("no manipulated object with a scene body" if not objects else
                            "; ".join(f"{o}: {t.get('position_error_m', float('nan')):.4f} m ({t.get('rule')})"
                                      for o, t in task.items()))
                           + (f"; unmapped objects {unmapped}" if unmapped else ""),
        "objects_at_rest": "; ".join(f"{k}: {v['linear_m_s']:.3f} m/s {v['angular_rad_s']:.3f} rad/s"
                                     + ("" if v["required"] else " (reported only)") for k, v in rest.items()),
        "actuator_replay": f"max |diff| {replay_diff}",
    }
    reasons = [f"{k}: {details[k]}" for k, ok in gates.items() if not ok]
    metrics = {
        "gates": gates, "thresholds": dict(th), "object_position_tol_m": cfg.object_position_tol_m,
        "max_depth_m": dict(mon.depth), "worst_contacts": dict(mon.depth_event),
        "settle_object_environment_depth_m": mon.settle_depth, "settle_object_environment_contact": mon.settle_event,
        "object_environment_reference_depth_m": None if oe_ref is None else float(oe_ref),
        "object_environment_threshold_m": oe_th, "object_environment_threshold_rule": oe_rule,
        "object_environment_passed_absolute": bool(oe_depth <= oe_abs),
        "removed_inactive_bodies": {"bodies": sorted(parked), "rule": "declared by the adapter (SceneRef.inactive_bodies)"
                                    if scene_ref.inactive_bodies is not None else
                                    f"untracked free bodies > {cfg.park_distance_m} m from the workspace"},
        "nonhand_object_contacts": mon.nonhand_object_contacts,
        "min_self_clearance_m": float(clearance.min()), "min_joint_margin_rad": mon.min_margin,
        "min_joint_margin_joint": mon.margin_joint,
        "peak_arm_speed_rad_s": float(peak_arm.max()), "peak_neck_speed_rad_s": float(peak_neck.max()),
        "peak_base_speed_body": mon.peak_base.tolist(),
        "tcp_position_error_max_m": pos_err.max(axis=0).tolist(), "tcp_rotation_error_max_rad": rot_err.max(axis=0).tolist(),
        "tcp_position_error_rms_m": np.sqrt((pos_err ** 2).mean(axis=0)).tolist(),
        "grasps": grasps, "task": task, "unmapped_objects": unmapped, "objects_final_speed": rest,
        "objects_at_rest_rule": rest_rule,
        "actuator_replay_max_abs_diff": replay_diff, "simulator_warnings": warnings,
        "steps": steps_done, "expected_steps": n_steps,
    }
    return {"passed": all(gates.values()), "reasons": reasons, "metrics": metrics}, rollout
