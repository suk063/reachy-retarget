"""Correct a colliding robot initialization without changing a PickCube scene.

The stationary base is an explicit experiment input. All right-hand targets,
object references, gripper intent and timestamps are retained. This is a robot
configuration admission check, never evidence of physical success.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import mujoco
import numpy as np

from .physics import initialize
from .store import json_write, sha256


def collision_depths(model, data, robot_bodies, active_object_body):
    """Use the same robot/environment classification as dynamics.rollout."""
    depths = {"environment": 0., "self": 0.}
    worst = {"environment": None, "self": None}
    for contact in data.contact:
        pair = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
        kind = ("self" if pair <= robot_bodies else
                "environment" if pair & robot_bodies and active_object_body not in pair else None)
        if kind and -float(contact.dist) > depths[kind]:
            depths[kind] = -float(contact.dist)
            worst[kind] = [model.geom(contact.geom1).name, model.geom(contact.geom2).name]
    return depths, worst


def prepare(prepared, robot, output, base_xyyaw):
    """Re-solve the unchanged right-hand trajectory at one stationary base.

    Save every computed configuration and rejection before returning or raising.
    Object poses are never assigned, including during this static inspection.
    The caller runs admitted plans through the existing actuator-only validator.
    """
    return _prepare(prepared, robot, output, base_xyyaw, expected_task="PickCube-v1")


def intent_parameters(details):
    """Numeric command/contact bounds shared by phase-aware static queries."""
    contact = float(details['pad_alignment']['inferred_contact_angle_rad'])
    effective = details.get('effective_candidate', {})
    target = float(effective.get('gripper_closed_target', -.06))
    if effective.get('force_feedback'):
        raise ValueError('Unbounded force-feedback aperture cannot receive static intent-envelope admission')
    if (not np.isfinite([contact, target]).all()
            or not -.06 <= contact <= 2. or not -.06 <= target <= 2.):
        raise ValueError('Intent envelope requires a verified finite contact aperture')
    return contact, target


def intent_apertures(commands, index, contact_angle, closed_target):
    """Sample declared closed bounds and the entire intent-transition sweep."""
    command = float(commands[index])
    closed = command < 0
    lower, upper = ((min(closed_target, contact_angle), max(closed_target, contact_angle))
                    if closed else (command, command))
    if index == 0 or not closed or closed != (commands[index-1] < 0):
        previous = (2. if index == 0 else
                    float(commands[index-1]) if commands[index-1] >= 0 else closed_target)
        lower, upper = min(lower, previous), max(upper, previous)
    return np.unique(np.linspace(lower, upper, 5))


def _prepare(prepared, robot, output, base_xyyaw, *, expected_task, right_arm_seed=None,
             gripper_admission="open_envelope"):
    """Shared right-arm solver; callers must declare the supported task."""
    if prepared[5].get("task") != expected_task:
        raise ValueError("Base correction is scoped to " + expected_task)
    if gripper_admission not in ("open_envelope", "intent_envelope"):
        raise ValueError("Unknown gripper admission policy")
    contact_angle = None
    closed_target = None
    if gripper_admission == "intent_envelope":
        contact_angle, closed_target = intent_parameters(prepared[5])
    base = np.asarray(base_xyyaw, dtype=float)
    times, previous, goals = prepared[7], prepared[8], prepared[10]
    if base.shape not in ((3,), (len(times), 3)) or not np.isfinite(base).all():
        raise ValueError("base_xyyaw must be a finite stationary pose or one pose per original timestamp")
    stationary = base.ndim == 1
    base_track = np.broadcast_to(base, (len(times), 3)) if stationary else base
    seed = None if right_arm_seed is None else np.asarray(right_arm_seed, dtype=float)
    if seed is not None and (seed.shape != (7,) or not np.isfinite(seed).all()):
        raise ValueError("right_arm_seed must contain seven finite joint angles")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / Path(__file__).name).write_text(Path(__file__).read_text())
    model, _, manifest = prepared[:3]
    reference = np.array(previous, copy=True)
    reference[:, :3] = base_track
    data = mujoco.MjData(model)
    robot_bodies = {model.body("base_link").id}
    for body in range(1, model.nbody):
        if int(model.body_parentid[body]) in robot_bodies:
            robot_bodies.add(body)
    active_body = model.body(manifest["objects"][prepared[5]["object_id"]]["body"]).id
    previous_active = robot.active.copy()
    robot.active = robot.arm_v[7:]
    q = robot.pack(previous[0, 3:], base_track[0])
    if seed is not None:
        q[robot.arm_ids[7:]] = seed
    solutions, errors, collisions, clearances, margins = [], [], [], [], []
    aperture_envelopes = []
    worst = {"environment": None, "self": None}
    worst_depth = {"environment": 0., "self": 0.}
    failure = None
    fixture_query = None
    try:
        from .planning_fixtures import make_query
        fixture_query = make_query(prepared)
        if fixture_query is not None:
            data = fixture_query.data
        for index in range(len(times)):
            q = robot.pack(q[robot.arm_ids], base_track[index])
            # Preserve the inactive arm's original joint reference exactly.
            q[robot.arm_ids[:7]] = previous[index, 3:10]
            target = robot.fk(q)
            target[1] = goals[index]
            q, pe, re = robot.ik(target, q, active_hands=(1,), iterations=150,
                                 joint_margin=.031)
            reference[index, 10:] = q[robot.arm_ids[7:]]
            solutions.append(q.copy()); errors.append([pe, re])
            apertures = np.array([2.])
            if gripper_admission == "intent_envelope":
                apertures = intent_apertures(prepared[9], index, contact_angle, closed_target)
            aperture_envelopes.append([float(apertures.min()), float(apertures.max())])
            depths = {"environment": 0., "self": 0.}
            for aperture in apertures:
                initialize(model, data, robot, q, manifest["mimics"],
                           {'l_hand_finger': 2., 'r_hand_finger': float(aperture)})
                if fixture_query is not None:
                    fixture_query.synchronize(index)
                current, pairs = collision_depths(model, data, robot_bodies, active_body)
                for kind in worst:
                    depths[kind] = max(depths[kind], current[kind])
                    if current[kind] > worst_depth[kind]:
                        worst[kind] = pairs[kind]
                        worst_depth[kind] = current[kind]
            collisions.append([depths["environment"], depths["self"]])
            gap, margin, _ = robot.r.geometry(q)
            clearances.append(gap); margins.append(margin)
    except Exception as exc:
        failure = type(exc).__name__ + ": " + str(exc)
    finally:
        robot.active = previous_active
    # Save partial results before any metric or export can fail.
    artifact = output / "robot-initialization.npz"
    np.savez_compressed(artifact, time_s=times, reference_before=previous,
                        reference_after=reference, q=np.asarray(solutions),
                        ik_errors=np.asarray(errors), collision_depths=np.asarray(collisions),
                        self_clearance_m=clearances, joint_margin_rad=margins,
                        gripper_aperture_envelopes_rad=np.asarray(aperture_envelopes),
                        original_hand_goals=goals, original_object_goals=prepared[11],
                        original_gripper_commands=prepared[9])
    if "time_s" in prepared[3]:
        np.save(output / "native-source-time-s.npy", prepared[3]["time_s"], allow_pickle=False)
    completed = len(solutions) == len(times) and failure is None
    errors = np.asarray(errors).reshape(-1, 2)
    collisions = np.asarray(collisions).reshape(-1, 2)
    report = dict(base_xyyaw=base.tolist() if stationary else None,
                  right_arm_seed=None if seed is None else seed.tolist(),
                  gripper_admission=gripper_admission,
                  gripper_closed_target_rad=closed_target,
                  inactive_left_gripper_rad=2.,
                  gripper_admission_scope="Static sampled robot aperture envelopes; actual gripper tracking and all contacts remain independently physics-gated",
                  initial_base_xyyaw=base_track[0].tolist(), final_base_xyyaw=base_track[-1].tolist(),
                  base_policy="stationary" if stationary else "explicit original-clock base trajectory",
                  completed_frames=len(solutions),
                  total_frames=len(times), exception=failure, physics_validated=False,
                  max_ik_position_error_m=float(errors[:, 0].max()) if len(errors) else None,
                  max_ik_rotation_error_rad=float(errors[:, 1].max()) if len(errors) else None,
                  max_environment_penetration_m=float(collisions[:, 0].max()) if len(collisions) else None,
                  max_self_penetration_m=float(collisions[:, 1].max()) if len(collisions) else None,
                  min_self_clearance_m=float(min(clearances)) if clearances else None,
                  min_joint_margin_rad=float(min(margins)) if margins else None,
                  worst_collision_pairs=worst,
                  artifact_sha256=sha256(artifact),
                  frozen_scene_sha256=hashlib.sha256(prepared[1].encode()).hexdigest(),
                  controller_identity=getattr(robot, "identity", None),
                  native_source_clock_sha256=(sha256(output / "native-source-time-s.npy")
                                              if "time_s" in prepared[3] else None),
                  scope="Robot base and right-arm joint configuration only; no source or scene changes",
                  object_collision_scope="Source reset object stays unchanged; static admission checks environment and robot self only")
    if fixture_query is not None:
        report['planning_fixture_forecast'] = fixture_query.save(output)
        report['object_collision_scope'] = 'Manipulated object stays at reset; selected inactive task fixtures use checksum-bound source poses in isolated planning data only'
    checks = dict(completed=completed,
                  right_hand_ik=bool(completed and np.all(errors <= [.002, .02])),
                  environment=bool(completed and np.max(collisions[:, 0]) <= .002),
                  self_collision=bool(completed and np.max(collisions[:, 1]) <= .002),
                  self_clearance=bool(completed and min(clearances) >= .009),
                  joint_margin=bool(completed and min(margins) >= .025))
    report.update(admission_checks=checks, admitted=all(checks.values()))
    if completed:
        report["peak_reference_base_speed_xyyaw"] = np.max(
            np.abs(np.diff(reference[:, :3], axis=0) / np.diff(times)[:, None]), axis=0).tolist()
        report["peak_reference_arm_speed_rad_s"] = np.max(
            np.abs(np.diff(reference[:, 3:], axis=0) / np.diff(times)[:, None]), axis=0).tolist()
    json_write(output / "result.json", report)
    if not report["admitted"]:
        raise ValueError("Robot initialization rejected; inspect " + str(output / "result.json"))
    result = list(prepared)
    details = deepcopy(prepared[5])
    details.update(robot_initialization=report,
                   trajectory_seed="Existing hand/object targets and source clock; right-arm IK at an explicit collision-checked robot base reference",
                   base_planning="stationary" if stationary else "explicit original-clock trajectory; saved in robot-initialization.npz")
    result[5], result[6], result[8] = details, solutions[0], reference
    return tuple(result)
