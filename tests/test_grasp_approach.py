import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget import grasp_approach as approach


class Robot:
    def __init__(self, fail_at=None):
        self.arm_ids = self.arm_v = np.arange(3, 17)
        self.active = np.arange(17)
        self.calls, self.fail_at = 0, fail_at
        self.r = SimpleNamespace(geometry=lambda q: (.02, .04, None))

    def pack(self, arms, base):
        return np.r_[base, arms]

    def fk(self, q):
        result = np.tile(np.eye(4), (2, 1, 1))
        result[1, :3, 3] = q[-6:-3]
        result[1, :3, :3] = Rotation.from_rotvec(q[-3:]).as_matrix()
        return result

    def ik(self, goals, q, **kwargs):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('solver stopped')
        q = q.copy()
        q[-6:-3] = goals[1, :3, 3]
        q[-3:] = Rotation.from_matrix(goals[1, :3, :3]).as_rotvec()
        return q, 0., 0.


def fixture(monkeypatch):
    xml = '<mujoco><worldbody><body name="base_link"><joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/><joint axis="0 0 1"/><geom type="box" size=".01 .01 .01" pos="0 0 -.5"/><body name="r_hand"><joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/><joint type="slide" axis="0 0 1"/><geom name="hand" type="sphere" size=".01"/></body></body><body name="object" pos=".2 0 .4"><freejoint/><geom name="object" type="box" size=".035 .035 .035"/></body></worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)

    def initialize(model, data, robot, q, mimics, grip):
        np.testing.assert_array_equal(data.qpos[6:], model.qpos0[6:])
        data.qpos[:3] = q[:3]
        data.qpos[3:6] = q[-6:-3]
        mujoco.mj_forward(model, data)

    monkeypatch.setattr(approach, 'initialize', initialize)
    times = np.arange(7) * .01
    hands = np.tile(np.eye(4), (7, 1, 1))
    hands[:, 0, 3] = [0, .1, .2, .3, .4, .45, .5]
    hands[:, 2, 3] = .4
    reference = np.zeros((7, 17))
    reference[:, -6:-3] = hands[:, :3, 3]
    objects = np.tile(np.eye(4), (7, 1, 1))
    objects[:, :3, 3] = [.2, 0, .4]
    details = dict(task='PickPlaceCan', object_id='object', effective_candidate={'controller': 'joint_reference'})
    return (model, xml, {'objects': {'object': {'body': 'object'}}, 'mimics': {}}, {}, {},
            details, reference[0].copy(), times, reference,
            np.array([2, 2, 2, 2, -.06, -.06, 2]), hands, objects)


def test_exact_endpoints_and_closed_suffix_in_continuous_full_prefix():
    times = np.arange(11) * .1
    hands = np.tile(np.eye(4), (11, 1, 1))
    hands[:, 0, 3] = times
    hands[6:, :3, :3] = Rotation.from_euler('z', .4).as_matrix()
    commands = np.r_[np.full(6, 2.), np.full(5, -.06)]
    result, report = approach.targets(times, hands, commands, lift_m=.12)
    np.testing.assert_array_equal(result[0], hands[0])
    np.testing.assert_array_equal(result[6:], hands[6:])
    np.testing.assert_allclose(result[3, :3, 3], [.6, 0, .12])
    np.testing.assert_allclose(result[3:7, :2, 3], np.tile([.6, 0], (4, 1)))
    assert report['close_frame'] == 6


def test_full_admission_preserves_objects_clock_gripper_base_left_arm_and_suffix(monkeypatch, tmp_path):
    original = fixture(monkeypatch)
    robot = Robot()
    active = robot.active.copy()
    result = approach.prepare(original, robot, tmp_path/'pass')
    for index in (0, 1, 2, 3, 4, 7, 9, 11):
        assert result[index] is original[index]
    np.testing.assert_array_equal(result[8][:, :10], original[8][:, :10])
    np.testing.assert_array_equal(result[8][4:], original[8][4:])
    np.testing.assert_array_equal(result[10][4:], original[10][4:])
    np.testing.assert_array_equal(robot.active, active)
    assert 'robot_grasp_approach' not in original[5]
    assert result[5]['robot_grasp_approach']['admitted']
    assert not result[5]['robot_grasp_approach']['physics_validated']


def test_low_approach_rejected_for_open_finger_object_overlap(monkeypatch, tmp_path):
    original = fixture(monkeypatch)
    with pytest.raises(ValueError, match='approach rejected'):
        approach.prepare(original, Robot(), tmp_path/'collision', lift_m=.001)
    report = json.loads((tmp_path/'collision/result.json').read_text())
    assert not report['admission_checks']['open_robot_object_no_overlap']
    assert report['max_collision_depths_m'][2] > .001
    assert (tmp_path/'collision/approach-admission.npz').exists()


def test_failed_backward_ik_saves_partial_computation_and_restores_active(monkeypatch, tmp_path):
    original = fixture(monkeypatch)
    robot = Robot(fail_at=2)
    active = robot.active.copy()
    with pytest.raises(ValueError, match='approach rejected'):
        approach.prepare(original, robot, tmp_path/'interrupted')
    np.testing.assert_array_equal(robot.active, active)
    report = json.loads((tmp_path/'interrupted/result.json').read_text())
    assert 'solver stopped' in report['exception']
    with np.load(tmp_path/'interrupted/approach-admission.npz') as f:
        assert np.isfinite(f['q'][3]).all()
        assert np.isnan(f['q'][2]).all()


def test_no_open_prefix_is_not_fabricated():
    with pytest.raises(ValueError, match='open pregrasp'):
        approach.targets(np.arange(3), np.tile(np.eye(4), (3, 1, 1)), np.full(3, -.06))


def test_real_control_clock_roundoff_cannot_push_slerp_outside_endpoints():
    times=np.arange(300)*.01
    hands=np.tile(np.eye(4),(len(times),1,1))
    hands[:,0,3]=times*.1
    hands[255:,:3,:3]=Rotation.from_euler('z',.3).as_matrix()
    commands=np.r_[np.full(255,2.),np.full(45,-.06)]
    changed,report=approach.targets(times,hands,commands,traverse_fraction=.8)
    assert report['close_frame']==255 and np.isfinite(changed).all()
    np.testing.assert_array_equal(changed[0],hands[0])
    np.testing.assert_array_equal(changed[255:],hands[255:])
    np.testing.assert_allclose(changed[204,:3,:3],hands[255,:3,:3],atol=1e-15)


def test_proposal_invalidates_old_admission_without_certifying_or_solving_joints(monkeypatch,tmp_path):
    original=list(fixture(monkeypatch))
    original[5]=dict(original[5],robot_initialization={'admitted':True},
        robot_grasp_approach={'admitted':True},robot_fixture_clearance={'admitted':True})
    result=approach.propose(tuple(original),tmp_path/'proposal')
    for index in (0,1,2,3,4,6,7,8,9,11):assert result[index] is original[index]
    np.testing.assert_array_equal(result[10][0],original[10][0])
    np.testing.assert_array_equal(result[10][4:],original[10][4:])
    assert not np.array_equal(result[10][1:4],original[10][1:4])
    assert original[5]['robot_initialization']['admitted']
    assert 'robot_initialization' not in result[5]
    assert 'robot_grasp_approach' not in result[5]
    assert 'robot_fixture_clearance' not in result[5]
    report=result[5]['robot_grasp_approach_proposal']
    assert not report['admitted'] and not report['physics_validated']
    assert report['completed_frames']==0 and report['requires_full_ik_and_collision_admission']
    assert not result[5]['joint_reference_valid']
    with np.load(report['artifact']) as saved:
        np.testing.assert_array_equal(saved['original_hand_targets'],original[10])
        np.testing.assert_array_equal(saved['proposed_hand_targets'],result[10])


def test_proposal_does_not_bypass_subsequent_open_object_collision_admission(monkeypatch,tmp_path):
    original=fixture(monkeypatch)
    proposed=approach.propose(original,tmp_path/'proposal',lift_m=.001)
    with pytest.raises(ValueError,match='approach rejected'):
        approach.prepare(proposed,Robot(),tmp_path/'checked',lift_m=.001)
    report=json.loads((tmp_path/'checked/result.json').read_text())
    assert not report['admission_checks']['open_robot_object_no_overlap']
    assert not report['admitted']


def aperture_fixture(monkeypatch):
    """A fixture intersects a fully open finger only on the closed suffix."""
    original = list(fixture(monkeypatch))
    original[1] = original[1].replace('<geom name="hand" type="sphere" size=".01"/>',
        '<geom name="hand" type="sphere" size=".01"/>'
        '<body name="finger"><joint name="finger_angle" axis="0 0 1"/>'
        '<geom name="finger" type="capsule" fromto="0 0 0 .1 0 0" size=".003"/></body>')
    original[1] = original[1].replace('</worldbody>',
        '<geom name="closed_suffix_fixture" type="sphere" size=".008" pos=".458385316 .090929743 .4"/></worldbody>')
    original[0] = mujoco.MjModel.from_xml_string(original[1])
    original[5].update(pad_alignment={'inferred_contact_angle_rad': .9},
                       effective_candidate={'gripper_closed_target': .85})
    original[9][-1] = -.06
    def initialize(model, data, robot, q, mimics, grip):
        np.testing.assert_array_equal(data.qpos[7:], model.qpos0[7:])
        data.qpos[:3] = q[:3]
        data.qpos[3:6] = q[-6:-3]
        data.qpos[6] = grip['r_hand_finger']
        assert grip['l_hand_finger'] == 2.
        mujoco.mj_forward(model, data)
    monkeypatch.setattr(approach, 'initialize', initialize)
    return tuple(original)


def test_phase_envelope_rejects_false_open_suffix_without_changing_trajectory(monkeypatch, tmp_path):
    original = aperture_fixture(monkeypatch)
    with pytest.raises(ValueError, match='approach rejected'):
        approach.prepare(original, Robot(), tmp_path/'forced-open')
    old = json.loads((tmp_path/'forced-open/result.json').read_text())
    assert not old['admission_checks']['environment']
    assert old['worst_collision_pairs']['environment']['frame'] == 6
    corrected = approach.prepare(original, Robot(), tmp_path/'phase-aware',
                                  gripper_admission='intent_envelope')
    assert corrected[5]['robot_grasp_approach']['admitted']
    for index in (0, 1, 2, 3, 4, 7, 9, 11):
        assert corrected[index] is original[index]
    np.testing.assert_array_equal(corrected[8][4:], original[8][4:])
    with np.load(tmp_path/'phase-aware/approach-admission.npz') as f:
        # First closure includes the full sweep; the already-closed suffix
        # checks only the declared target-to-contact interval.
        np.testing.assert_array_equal(f['gripper_aperture_envelopes_rad'][4], [.85, 2.])
        np.testing.assert_array_equal(f['gripper_aperture_envelopes_rad'][5:], [[.85, .9], [.85, .9]])
        assert np.max(f['collision_depths'][:, 0]) < 1e-10


def test_phase_envelope_keeps_true_opening_collision_rejected(monkeypatch, tmp_path):
    original = aperture_fixture(monkeypatch)
    original[9][-1] = 2.
    with pytest.raises(ValueError, match='approach rejected'):
        approach.prepare(original, Robot(), tmp_path/'opening', gripper_admission='intent_envelope')
    result = json.loads((tmp_path/'opening/result.json').read_text())
    assert not result['admission_checks']['environment']
    assert result['worst_collision_pairs']['environment']['frame'] == 6
    with np.load(tmp_path/'opening/approach-admission.npz') as f:
        np.testing.assert_array_equal(f['gripper_aperture_envelopes_rad'][-1], [.85, 2.])


def test_intent_proposal_is_target_only_and_requires_later_phase_admission(monkeypatch, tmp_path):
    original = aperture_fixture(monkeypatch)
    result = approach.propose(original, tmp_path/'proposal', gripper_admission='intent_envelope')
    assert result[8] is original[8]
    assert result[5]['robot_grasp_approach_proposal']['requested_followup_gripper_admission'] == 'intent_envelope'
    assert not result[5]['robot_grasp_approach_proposal']['admitted']
    original[5]['effective_candidate']['force_feedback'] = True
    with pytest.raises(ValueError, match='Unbounded'):
        approach.prepare(original, Robot(), tmp_path/'unbounded', gripper_admission='intent_envelope')


def bound_mobile_proposal(monkeypatch, tmp_path, *, lift_m=.12):
    import hashlib
    original = fixture(monkeypatch)
    proposed = approach.propose(original, tmp_path/'proposal', lift_m=lift_m)
    result = list(proposed)
    reference = proposed[8].copy()
    reference[:, -6:-3] = proposed[10][:, :3, 3]
    reference[:, -3:] = Rotation.from_matrix(proposed[10][:, :3, :3]).as_rotvec()
    result[8] = reference
    path = tmp_path/'mobile.npz'
    np.savez_compressed(path, original_time_s=proposed[7], reference=reference,
        original_hand_goals=proposed[10], original_object_goals=proposed[11],
        original_gripper_intent=proposed[9])
    result[5] = dict(proposed[5], robot_mobile_fixture_ik=dict(admitted=True,
        completed_frames=len(reference), total_frames=len(reference),
        admission_checks={'completed': True, 'ik': True, 'positive_fixture_clearance': True},
        artifact=str(path), artifact_sha256=approach.sha256(path),
        scene_sha256=hashlib.sha256(proposed[1].encode()).hexdigest()))
    return tuple(result)


def test_reuse_skips_second_ik_but_checks_every_unchanged_mobile_row(monkeypatch, tmp_path):
    prepared = bound_mobile_proposal(monkeypatch, tmp_path)
    robot = Robot(fail_at=1)
    result = approach.prepare(prepared, robot, tmp_path/'checked', preserve_admitted_reference=True)
    assert robot.calls == 0
    np.testing.assert_array_equal(result[8], prepared[8])
    for index in (7, 9, 10, 11):
        np.testing.assert_array_equal(result[index], prepared[index])
    report = result[5]['robot_grasp_approach']
    assert report['completed_frames'] == report['total_frames'] == len(prepared[7])
    assert all(report['admission_checks'].values())
    assert report['reused_admission_binding']['reference_exact']


def test_reuse_still_rejects_open_object_overlap_and_saves_evidence(monkeypatch, tmp_path):
    prepared = bound_mobile_proposal(monkeypatch, tmp_path, lift_m=.001)
    robot = Robot(fail_at=1)
    with pytest.raises(ValueError, match='approach rejected'):
        approach.prepare(prepared, robot, tmp_path/'checked', lift_m=.001,
                         preserve_admitted_reference=True)
    assert robot.calls == 0
    report = json.loads((tmp_path/'checked/result.json').read_text())
    assert report['completed_frames'] == len(prepared[7])
    assert not report['admission_checks']['open_robot_object_no_overlap']


@pytest.mark.parametrize('change', ['reference', 'proposal_hash', 'mobile_complete', 'scene', 'different_approach'])
def test_reuse_rejects_stale_or_incomplete_admission(monkeypatch, tmp_path, change):
    prepared = list(bound_mobile_proposal(monkeypatch, tmp_path))
    kwargs = {}
    if change == 'reference':
        prepared[8] = prepared[8].copy(); prepared[8][1, -1] += 1e-6
    elif change == 'proposal_hash':
        prepared[5]['robot_grasp_approach_proposal']['artifact_sha256'] = '0'*64
    elif change == 'mobile_complete':
        prepared[5]['robot_mobile_fixture_ik']['completed_frames'] -= 1
    elif change == 'scene':
        prepared[5]['robot_mobile_fixture_ik']['scene_sha256'] = '0'*64
    else:
        kwargs['lift_m'] = .13
    robot = Robot(fail_at=1)
    with pytest.raises(ValueError, match='approach rejected'):
        approach.prepare(tuple(prepared), robot, tmp_path/'checked',
                         preserve_admitted_reference=True, **kwargs)
    report = json.loads((tmp_path/'checked/result.json').read_text())
    assert report['completed_frames'] == 0 and report['exception']
    assert robot.calls == 0
