"""Geometric invariants and bounded continuation without a robot or network."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget import feasible_pose as fp
from reachy_retarget.episodes import matrices_to_pose


def test_axis_mapping_is_proper_and_approach_preserving_for_both_sources():
    for close in ([0, -1, 0], [-1, 0, 0]):
        for sign in (-1, 1):
            rotation = fp.tool_rotation(close, jaw_sign=sign)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3))
            assert np.linalg.det(rotation) == pytest.approx(1)
            np.testing.assert_allclose(rotation @ [0, 0, -1], [0, 0, 1])
            np.testing.assert_allclose(rotation @ [0, 1, 0], np.asarray(close)*sign)
    with pytest.raises(ValueError):
        fp.tool_rotation([0, 0, 1])


def test_declared_aperture_calibration_does_not_change_other_hand_or_accept_invalid_angles():
    source = {"left": .22}
    assert fp.calibration_apertures(source) == {"left": .22, "right": .63}
    assert source == {"left": .22}
    assert fp.calibration_apertures() == {"left": .63, "right": .63}
    for invalid in ({"left": np.nan}, {"right": 3.}, {"left": -.2}, {"unknown": .2}, [.2, .3]):
        with pytest.raises(ValueError):
            fp.calibration_apertures(invalid)


def test_geometry_gap_inversion_rejects_unreachable_and_nonmonotone_apertures():
    assert fp.angle_for_pad_gap(lambda angle: .004+.04*(angle+.06), .012) == pytest.approx(.14)
    with pytest.raises(ValueError, match="outside"):
        fp.angle_for_pad_gap(lambda angle: .004+.04*(angle+.06), .2)
    with pytest.raises(ValueError):
        fp.angle_for_pad_gap(lambda angle: .02+.01*np.sin(3*angle), .025)
    assert fp.angle_for_pad_gap(lambda angle: abs(.04*(angle-.03)), .008) == pytest.approx(.23)
    with pytest.raises(ValueError, match="positive"):
        fp.angle_for_pad_gap(lambda angle: .004+.04*(angle+.06), np.nan)


def test_rim_rule_changes_selected_robot_grasp_preserving_all_observed_arrays():
    obj = np.tile(np.eye(4), (2, 1, 1))
    obj[1, :3, :3] = Rotation.from_euler("z", .4).as_matrix()
    obj[1, :3, 3] = [2., 3., 1.]
    relative = np.eye(4); relative[:3, 3] = [.1, 0., .003]
    left = obj @ relative
    right = obj.copy(); right[:, 2, 3] += .4
    a = {"objects/plate/pose": matrices_to_pose(obj), "left": matrices_to_pose(left),
         "right": matrices_to_pose(right), "timestamp": np.array([.002, .004])}
    before = {key: value.copy() for key, value in a.items()}
    spec = dict(hand="left", object_key="objects/plate/pose", anchor_frame=0,
        center_of_mass_local_m=[0, 0, 0], plane_normal_local=[0, 0, 1],
        gravity_world_m_s2=[0, -9.81, 0], geometry_evidence={"method": "synthetic planar disk fixture"})
    adjusted, report = fp.gravity_aligned_rim(a, {"left": "left", "right": "right"}, spec)
    target = fp._pose(adjusted["left"])
    local = np.linalg.inv(obj) @ target
    np.testing.assert_allclose(local[:, :3, 3], [[0, .1, .003]]*2, atol=1e-12)
    assert report["rim_rotation_rad"] == pytest.approx(np.pi/2)
    assert report["anchor_grasp_relocation_m"] == pytest.approx(np.sqrt(.02))
    assert report["object_reference_changes"] is False
    for key, value in a.items():
        np.testing.assert_array_equal(value, before[key])
    np.testing.assert_array_equal(adjusted["right"], before["right"])
    np.testing.assert_array_equal(adjusted["objects/plate/pose"], before["objects/plate/pose"])
    retained, retained_report = fp.gravity_aligned_rim(a, {"left": "left", "right": "right"},
        dict(spec, orientation_policy="preserve_source", orientation_evidence={"geometry": "synthetic closing-line test"}))
    np.testing.assert_allclose(fp._pose(retained["left"])[:, :3, :3], left[:, :3, :3], atol=1e-12)
    np.testing.assert_allclose(fp._pose(retained["left"])[:, :3, 3], target[:, :3, 3], atol=1e-12)
    assert retained_report["orientation_policy"] == "preserve_source"
    with pytest.raises(ValueError, match="geometry evidence"):
        fp.gravity_aligned_rim(a, {"left": "left"}, dict(spec, orientation_policy="preserve_source"))
    with pytest.raises(ValueError, match="gravity projection"):
        fp.gravity_aligned_rim(a, {"left": "left"}, dict(spec, gravity_world_m_s2=[0, 0, -9.81]))


def test_cad_center_hits_source_reference_without_modifying_source_or_object():
    base = np.tile(np.eye(4), (3, 1, 1))
    base[:, :3, :3] = Rotation.from_euler("z", .7).as_matrix()
    base[:, :3, 3] = [[2., 3., .8], [2.1, 3., .8], [2.3, 3., .8]]
    hand = base.copy(); hand[:, :3, 3] += [.4, .2, .5]
    hand[:, :3, :3] = Rotation.from_euler("xyz", [.4, -.2, 1.1]).as_matrix()
    a = {"base": matrices_to_pose(base), "hand": matrices_to_pose(hand), "objects/plate/pose": matrices_to_pose(hand)}
    before = {k: v.copy() for k, v in a.items()}
    offset = np.array([-.010196, 0, -.04657015])
    b, target, contact, placement, rotations, active = fp.contact_references(
        a, np.tile(np.eye(4), (2, 1, 1)), "base", {"left": "hand"},
        {"left": fp.tool_rotation([0, -1, 0])}, {"left": offset})
    assert active == (0,)
    np.testing.assert_allclose(b[0, :2, 3], 0, atol=1e-14)
    center = target[:, 0, :3, 3] + np.einsum("nij,j->ni", target[:, 0, :3, :3], offset)
    np.testing.assert_allclose(center, (placement @ hand)[:, :3, 3], atol=1e-14)
    np.testing.assert_allclose(contact[:, 0], placement @ hand)
    for key in a:
        np.testing.assert_array_equal(a[key], before[key])
    bad = np.eye(3); bad[0, 0] = -1
    with pytest.raises(ValueError, match="proper"):
        fp.contact_references(a, np.tile(np.eye(4), (2, 1, 1)), "base", {"left": "hand"}, {"left": bad}, {"left": offset})


class LinearRobot:
    """Both arms independently translate/rotate; base supplies planar motion."""
    def __init__(self):
        self.q = np.zeros(18); self.q[2] = 1
        self.arm_ids = np.arange(4, 18)
        self.r = SimpleNamespace(model=SimpleNamespace(lowerPositionLimit=np.full(18, -3.), upperPositionLimit=np.full(18, 3.)))
        self.pin = SimpleNamespace(log3=lambda matrix: Rotation.from_matrix(matrix).as_rotvec())
    def fk(self, q):
        result = np.tile(np.eye(4), (2, 1, 1))
        for h in (0, 1):
            j = q[self.arm_ids[7*h:7*(h+1)]]
            result[h, :3, 3] = j[:3]+[q[0], q[1], 0]
            result[h, :3, :3] = Rotation.from_rotvec(j[3:6]).as_matrix()
        return result


def test_irregular_clock_bounded_base_and_both_hands_warmstart():
    robot = LinearRobot()
    times = np.array([0., .01, .05, .2])
    base = np.tile(np.eye(4), (len(times), 1, 1))
    base[:, 0, 3] = times*.2
    targets = np.tile(np.eye(4), (len(times), 2, 1, 1))
    targets[:, 0, :3, 3] = np.c_[.4+times, .2+times*.2, np.ones(4)*.1]
    targets[:, 1, :3, 3] = np.c_[.1+times, -.4+times*.1, np.ones(4)*.3]
    out = fp.solve_frames(robot, times, base, targets, (0, 1), max_starts=3, max_evaluations=60)
    assert out["errors"].max() < 1e-5
    assert len(out["audit"]) == len(times)  # No unnecessary restarts for passing rows.
    assert np.abs(out["base_interval_velocity_xyyaw"][:, :2]).max() <= .600001
    assert np.abs(out["base_interval_velocity_xyyaw"][:, 2]).max() <= 1.200001
    assert out["base_tube_failure_frames"] == []
    np.testing.assert_allclose(out["achieved"][:, :, :3, 3], targets[:, :, :3, 3], atol=1e-5)


def test_unsatisfiable_reference_saves_all_bounded_starts():
    robot = LinearRobot()
    times = np.array([0., .01])
    base = np.tile(np.eye(4), (2, 1, 1))
    targets = np.tile(np.eye(4), (2, 2, 1, 1))
    targets[:, 1, 2, 3] = 10.
    out = fp.solve_frames(robot, times, base, targets, (1,), max_starts=3, max_evaluations=5)
    assert len(out["audit"]) == 6
    assert (out["errors"][:, 0] > 6).all()
    assert all(len(row["qpos"]) == 18 for row in out["audit"])


def test_clock_repeated_or_nonfinite_is_rejected():
    for clock in ([0, 0], [0, np.nan]):
        with pytest.raises(ValueError, match="clock"):
            fp.solve_frames(LinearRobot(), clock, np.tile(np.eye(4), (2, 1, 1)), np.tile(np.eye(4), (2, 2, 1, 1)), (0,))


def test_open_far_blend_preserves_closed_phase_and_does_not_invent_withdrawal():
    n = 9
    objects = np.tile(np.eye(4), (n, 1, 1))
    hands = objects.copy()
    hands[:, 0, 3] = [.4, .3, .22, .12, .08, .08, .09, .1, .12]
    a = {'timestamp': np.array([0., .03, .08, .1, .15, .2, .3, .5, .8]),
         'action': np.zeros((n, 1))}
    a['action'][4:6] = 1
    spec = dict(clock_key='timestamp', gripper_command=dict(array='action', column=0, closed_value=1),
                clear_distance_m=.2, return_distance_m=.3)
    weights, report = fp.free_phase_weights(a, hands, objects, np.zeros(3), spec)
    np.testing.assert_array_equal(weights[:3], 0)
    np.testing.assert_array_equal(weights[3:], 1)
    assert report['no_observed_withdrawal'] is True
    hands[6:, 0, 3] = [.21, .25, .31]
    weights, report = fp.free_phase_weights(a, hands, objects, np.zeros(3), spec)
    np.testing.assert_array_equal(weights[4:7], 1)
    assert 0 < weights[7] < 1 and weights[8] == 0
    assert report['retreat_start_frame'] == 6
    a['action'][2] = 1
    with pytest.raises(ValueError, match='contiguous'):
        fp.free_phase_weights(a, hands, objects, np.zeros(3), spec)


def test_solver_progress_retains_each_trial_and_completed_prefix():
    events = []
    out = fp.solve_frames(LinearRobot(), np.array([0., .01]),
        np.tile(np.eye(4), (2, 1, 1)), np.tile(np.eye(4), (2, 2, 1, 1)), (0, 1),
        max_starts=2, max_evaluations=3,
        progress=lambda event, values: events.append((event, values.get('frame', values.get('completed_frames')))))
    assert events == [('trial', 0), ('frame', 1), ('trial', 1), ('frame', 2)]
    assert len(out['audit']) == 2


def test_optimistic_tip_wrist_preflight_can_only_prove_large_distance_failure():
    placement = lambda p: SimpleNamespace(rotation=np.eye(3), translation=np.asarray(p, dtype=float))
    model = SimpleNamespace(getJointId=lambda name: {'root_joint':1,'l_shoulder_pitch':2}[name],
        frames=[SimpleNamespace(parentJoint=3, placement=placement([.1,0,0]))],
        parents=[0,0,1,2], jointPlacements=[placement([0,0,0])]*3+[placement([.4,0,0])])
    data = SimpleNamespace(oMi=[placement([0,0,0]), placement([0,0,0]), placement([0,.1,1]), placement([.4,.1,1])])
    robot = SimpleNamespace(q=np.zeros(1), fk=lambda q: None, r=SimpleNamespace(model=model,data=data,hands=[0]))
    base = np.tile(np.eye(4),(2,1,1)); targets=np.tile(np.eye(4),(2,2,1,1))
    targets[:,0,:3,3] = [[.5,0,1],[3,0,1]]
    rows, report = fp.reach_preflight(robot,base,targets,(0,))
    assert report['proven_infeasible']
    assert report['hands']['left']['tip']['proven_reject_frames'] == [1]
    assert report['hands']['left']['wrist']['proven_reject_frames'] == [1]
    assert rows['left/tip_reach_excess_m'][0] < 0
