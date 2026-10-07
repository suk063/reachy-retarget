import copy

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget import cylindrical_grasp as grasp


def cylinder_vertices():
    angle = np.arange(24)*2*np.pi/24
    # Original polygonal outer wall with narrower cap rings; no fitted mesh.
    return np.array([[radius*np.cos(a), radius*np.sin(a), z]
                     for z, radius in [(-.04, .023), (-.038, .025),
                                       (.037, .025), (.04, .023)] for a in angle])


def pad_geometry():
    faces = []
    gap = .05
    for y in [-gap/2, gap/2]:
        vertices = np.array([[x, y, z] for x, z in [(-.014, -.016), (.014, -.016),
                                                   (.014, .016), (-.014, .016)]])
        faces.append(vertices[[[0, 1, 2], [0, 2, 3]]])
    return dict(points_tcp=np.array([[0., -gap/2, 0.], [0., gap/2, 0.]]),
                midpoint_tcp=np.zeros(3), gap_m=gap), faces


def test_mesh_axis_and_wall_ignore_internal_and_cap_vertices_under_rigid_rotation():
    vertices = np.vstack([cylinder_vertices(), [[.005, 0, .031427], [-.01, .005, .031427]]])
    rotation = Rotation.from_euler('xyz', [21, 31, 57], degrees=True).as_matrix()
    translation = np.array([.2, -.4, .7])
    wall = grasp.mesh_wall(vertices @ rotation.T + translation)
    axis = np.asarray(wall['axis_object'])
    expected = rotation[:, 2]
    sign = 1 if axis @ expected > 0 else -1
    np.testing.assert_allclose(axis, sign*expected, atol=1e-10)
    bounds = np.sort(np.array([-.038, .037])*sign+translation @ axis)
    np.testing.assert_allclose(wall['straight_wall_interval_m'], bounds, atol=1e-10)
    assert wall['radius_fit_m'] == pytest.approx(.025, abs=1e-10)
    assert wall['side_facet_count'] == 24


def test_alignment_removes_jaw_height_difference_and_clamps_only_to_supported_height():
    vertices = cylinder_vertices()
    wall = grasp.mesh_wall(vertices)
    pad, faces = pad_geometry()
    hand, obj = np.eye(4), np.eye(4)
    hand[:3, :3] = Rotation.from_euler('x', 30, degrees=True).as_matrix()
    hand[:3, 3] = [.1, .2, .5]
    obj[:3, 3] = [.1, .2, .5]
    originals = copy.deepcopy((hand, obj, pad, faces, vertices))
    attachment, report = grasp.derive_attachment(hand, obj, pad, faces, wall, vertices, [0, 0, .027])
    corrected = hand @ attachment
    np.testing.assert_allclose(corrected[:3, :3] @ [0, 1, 0], [0, 1, 0], atol=1e-10)
    np.testing.assert_allclose(corrected[:3, :3] @ [0, 0, 1], [0, 0, 1], atol=1e-10)
    assert report['selected_midpoint_height_m'] == pytest.approx(.020)
    assert report['rotation_change_rad'] == pytest.approx(np.pi/6)
    assert report['axial_contact_centroid_separation_m'] < 1e-12
    assert all(v[1] <= .036+1e-12 for v in report['pad_axial_bounds_object_m'])
    assert all(report['static_finite_patch_checks'].values())
    assert all(c['supported_axis_interval_m'][1] > c['supported_axis_interval_m'][0]
               for c in report['finite_contact_corridors'])
    for actual, previous in zip((hand, obj, vertices), (originals[0], originals[1], originals[4])):
        np.testing.assert_array_equal(actual, previous)


def test_native_azimuth_and_fully_supported_height_remain_exact():
    vertices, (pad, faces) = cylinder_vertices(), pad_geometry()
    wall = grasp.mesh_wall(vertices)
    hand, obj = np.eye(4), np.eye(4)
    hand[:3, :3] = Rotation.from_euler('zx', [37, -25], degrees=True).as_matrix()
    before = hand[:3, :3] @ [0, 1, 0]
    expected = before.copy()
    expected[2] = 0
    expected /= np.linalg.norm(expected)
    attachment, report = grasp.derive_attachment(hand, obj, pad, faces, wall, vertices, [0, 0, .005])
    np.testing.assert_allclose((hand @ attachment)[:3, :3] @ [0, 1, 0], expected, atol=1e-12)
    assert report['selected_midpoint_height_m'] == pytest.approx(.005)
    second = np.eye(4)
    second[:3, :3] = Rotation.from_euler('xyz', [11, 72, 5], degrees=True).as_matrix()
    second[:3, 3] = [.3, -.2, .6]
    np.testing.assert_allclose((second @ attachment) @ np.linalg.inv(hand @ attachment),
                              second @ np.linalg.inv(hand), atol=1e-12)


def test_supported_short_pad_dimension_avoids_unnecessary_ninety_degree_roll():
    vertices, (pad, faces) = cylinder_vertices(), pad_geometry()
    wall = grasp.mesh_wall(vertices)
    hand = np.eye(4)
    hand[:3, :3] = Rotation.from_euler('yx', [90, 25], degrees=True).as_matrix()
    attachment, report = grasp.derive_attachment(hand, np.eye(4), pad, faces, wall, vertices, [0, 0, .027])
    assert report['rotation_change_rad'] == pytest.approx(np.deg2rad(25))
    assert report['rotation_if_longest_pad_axis_fully_aligned_rad'] > np.deg2rad(80)
    assert report['selected_midpoint_height_m'] == pytest.approx(.022)
    assert not report['pad_long_axis_alignment_applied']
    np.testing.assert_allclose((hand @ attachment)[:3, :3] @ [0, 1, 0], [0, 1, 0], atol=1e-12)
    assert all(report['static_finite_patch_checks'].values())


def wedge_pad_faces():
    pad, faces = pad_geometry()
    faces[0] = faces[0][:, ::-1]
    for i, angle in enumerate([.02, -.02]):
        center = pad['points_tcp'][i]
        rotation = Rotation.from_rotvec([angle, 0, 0]).as_matrix()
        faces[i] = (faces[i]-center) @ rotation.T+center
    return pad, faces


def test_nearest_original_facet_minimizes_azimuth_and_uses_polygon_support_width():
    vertices, (pad, faces) = cylinder_vertices(), pad_geometry()
    original_vertices = vertices.copy()
    hand = np.eye(4)
    hand[:3, :3] = (Rotation.from_euler('z', 2., degrees=True).as_matrix()
                       @ Rotation.from_euler('x', 20., degrees=True).as_matrix())
    original_hand = hand.copy()
    wall = grasp.mesh_wall(vertices)
    attachment, report = grasp.derive_attachment(hand, np.eye(4), pad, faces, wall,
        vertices, [0, 0, 0], azimuth_policy='nearest_opposed_facets')
    assert abs(report['facet_azimuth']['signed_rotation_rad']) == pytest.approx(np.deg2rad(5.5))
    assert report['contact_width_m'] == pytest.approx(.05*np.cos(np.pi/24), abs=1e-12)
    normal = np.asarray(report['facet_azimuth']['selected_normal_object'])
    np.testing.assert_allclose((hand @ attachment)[:3, :3] @ [0, 1, 0], normal, atol=1e-12)
    np.testing.assert_allclose(report['facet_azimuth']['opposed_normal_object'], -normal, atol=1e-12)
    np.testing.assert_array_equal(vertices, original_vertices)
    np.testing.assert_array_equal(hand, original_hand)
    default, _ = grasp.derive_attachment(hand, np.eye(4), pad, faces, wall, vertices, [0, 0, 0])
    explicit, _ = grasp.derive_attachment(hand, np.eye(4), pad, faces, wall, vertices,
        [0, 0, 0], azimuth_policy='source')
    np.testing.assert_array_equal(default, explicit)


def test_facet_selection_is_equivariant_and_preserves_dominant_plane_tangency():
    vertices = cylinder_vertices()
    axis, jaw = np.array([0., 0., 1.]), np.array([.3, .95, 0.])
    rotation = Rotation.from_euler('xyz', [17, 31, 22], degrees=True).as_matrix()
    correction, report = grasp.nearest_opposed_facets(vertices, axis, jaw)
    transformed, changed = grasp.nearest_opposed_facets(vertices @ rotation.T+[.4, -.2, .7],
        rotation @ axis, rotation @ jaw)
    np.testing.assert_allclose(transformed, rotation @ correction @ rotation.T, atol=1e-12)
    assert changed['signed_rotation_rad'] == pytest.approx(report['signed_rotation_rad'], abs=1e-12)
    pad, faces = wedge_pad_faces()
    hand = np.eye(4)
    hand[:3, :3] = Rotation.from_euler('zy', [2., -88.], degrees=True).as_matrix()
    attachment, result = grasp.derive_attachment(hand, np.eye(4), pad, faces, grasp.mesh_wall(vertices),
        vertices, [0, 0, 0], orientation_policy='dominant_pad_planes', azimuth_policy='nearest_opposed_facets')
    normals = np.array([p['normal_tcp'] for p in result['dominant_pad_planes']['planes']])
    np.testing.assert_allclose(normals @ (hand @ attachment)[:3, :3].T @ axis, 0, atol=1e-12)


def test_odd_polygon_without_opposed_facets_and_unknown_policy_are_rejected():
    angles = np.arange(25)*2*np.pi/25
    vertices = np.array([[.025*np.cos(a), .025*np.sin(a), z] for z in [-.038, .037] for a in angles])
    with pytest.raises(ValueError, match='No opposed'):
        grasp.nearest_opposed_facets(vertices, [0, 0, 1], [0, 1, 0])
    pad, faces = pad_geometry()
    with pytest.raises(ValueError, match='azimuth policy'):
        grasp.derive_attachment(np.eye(4), np.eye(4), pad, faces, grasp.mesh_wall(vertices),
            vertices, [0, 0, 0], azimuth_policy='invalid')


def test_dominant_plane_policy_corrects_small_common_tangent_error_without_pca_roll():
    pad, faces = wedge_pad_faces()
    hand = np.eye(4)
    hand[:3, :3] = Rotation.from_euler('y', -88, degrees=True).as_matrix()
    vertices = cylinder_vertices()
    attachment, report = grasp.derive_attachment(hand, np.eye(4), pad, faces,
        grasp.mesh_wall(vertices), vertices, [0, 0, 0], orientation_policy='dominant_pad_planes')
    assert report['rotation_change_rad'] == pytest.approx(np.deg2rad(2))
    planes = report['dominant_pad_planes']
    assert all(p['area_fraction'] == pytest.approx(1) for p in planes['planes'])
    np.testing.assert_allclose(planes['normal_dot_cylinder_axis_after'], 0, atol=1e-12)
    common = np.asarray(planes['common_tangent_tcp'])
    np.testing.assert_allclose((hand @ attachment)[:3, :3] @ common, [0, 0, 1], atol=1e-12)
    assert report['static_finite_patch_checks']['common_pad_plane_tangent_aligned_to_cylinder_axis']
    assert 'jaw_perpendicular_to_cylinder_axis' not in report['static_finite_patch_checks']


def test_dominant_plane_axis_discloses_nonzero_jaw_axial_residual():
    pad, faces = wedge_pad_faces()
    skew = Rotation.from_euler('z', .1, degrees=True).as_matrix()
    faces = [face @ skew.T for face in faces]
    hand = np.eye(4)
    hand[:3, :3] = Rotation.from_euler('y', -88, degrees=True).as_matrix()
    vertices = cylinder_vertices()
    _, report = grasp.derive_attachment(hand, np.eye(4), pad, faces, grasp.mesh_wall(vertices),
        vertices, [0, 0, 0], orientation_policy='dominant_pad_planes')
    assert abs(report['dominant_pad_planes']['closing_axis_axial_residual']) == pytest.approx(np.sin(np.deg2rad(.1)))
    assert report['axial_contact_centroid_separation_m'] > .00008
    assert report['static_finite_patch_checks']['jaw_axial_residual_explicitly_reported']


def test_parallel_or_fragmented_pad_normals_do_not_identify_dominant_common_plane():
    _, faces = pad_geometry()
    with pytest.raises(ValueError, match='Nearly parallel'):
        grasp.dominant_pad_planes(faces)
    _, wedge = wedge_pad_faces()
    fragmented = [np.r_[face, face @ Rotation.from_euler('x', 20, degrees=True).as_matrix().T]
                  for face in wedge]
    with pytest.raises(ValueError, match='sufficiently dominant'):
        grasp.dominant_pad_planes(fragmented)


def test_box_or_unsupported_pad_cannot_be_labelled_cylindrical():
    box = np.array([[x, y, z] for x in [-.025, .025] for y in [-.025, .025] for z in [-.04, .04]])
    with pytest.raises(ValueError, match='side edges|common cylinder axis|side facets'):
        grasp.mesh_wall(box)
    vertices, (pad, faces) = cylinder_vertices(), pad_geometry()
    wall = grasp.mesh_wall(vertices)
    wall['straight_wall_interval_m'] = [-.01, .01]
    with pytest.raises(ValueError, match='do not fit'):
        grasp.derive_attachment(np.eye(4), np.eye(4), pad, faces, wall, vertices, [0, 0, 0])
    wall = grasp.mesh_wall(vertices)
    hand = np.eye(4)
    hand[:3, :3] = Rotation.from_euler('x', 90, degrees=True).as_matrix()
    with pytest.raises(ValueError, match='too axial'):
        grasp.derive_attachment(hand, np.eye(4), pad, faces, wall, vertices, [0, 0, 0])


def test_compiled_mesh_geom_and_rigid_body_frames_are_transformed_to_object_root():
    vertices = cylinder_vertices()
    text = ' '.join(map(str, vertices.ravel()))
    child_rotation = Rotation.from_euler('xyz', [0, 10, 0], degrees=True)
    geom_rotation = Rotation.from_euler('xyz', [23, 31, 12], degrees=True)
    child_quat = ' '.join(map(str, child_rotation.as_quat()[[3, 0, 1, 2]]))
    geom_quat = ' '.join(map(str, geom_rotation.as_quat()[[3, 0, 1, 2]]))
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><asset><mesh name="native" vertex="{text}"/></asset>
    <worldbody><body name="object" pos=".3 .2 .6" euler="15 25 35"><freejoint name="object_free"/>
      <body name="rigid_child" pos=".01 -.02 .03" quat="{child_quat}">
        <geom name="native_collision" type="mesh" mesh="native" pos=".04 -.01 .02" quat="{geom_quat}"/>
      </body></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    before = data.qpos.copy()
    extracted, report = grasp.collision_mesh(model, data, object_body='object', object_joint='object_free')
    child, geom = child_rotation.as_matrix(), geom_rotation.as_matrix()
    expected = (vertices @ geom.T+[.04, -.01, .02]) @ child.T+[.01, -.02, .03]
    # MuJoCo principal-axis mesh compilation changes vertex order and geom
    # quaternion. Compare the complete resulting shape rather than raw order.
    from scipy.spatial.distance import cdist
    assert np.max(np.min(cdist(expected, extracted), axis=1)) < 1e-8
    assert np.max(np.min(cdist(extracted, expected), axis=1)) < 1e-8
    assert abs(np.asarray(report['axis_object']) @ (child @ geom[:, 2])) == pytest.approx(1., abs=1e-8)
    np.testing.assert_array_equal(data.qpos, before)


@pytest.mark.parametrize('invalid', ['robot_control_retiming', 'robot_supported_placement'])
def test_retimed_or_inserted_parents_cannot_reuse_stale_anchor(tmp_path, invalid):
    prepared = (None, None, None, None, None, {invalid: {'admitted': True}},
                None, None, None, None, None, None)
    with pytest.raises(ValueError, match='original untimed, uninserted'):
        grasp.apply(prepared, None, tmp_path/invalid)


def test_post_anchor_corridor_uses_exact_rotated_response_and_stops_at_first_release():
    _, faces = pad_geometry()
    wall = grasp.mesh_wall(cylinder_vertices())
    hands, objects = np.tile(np.eye(4), (4, 1, 1)), np.tile(np.eye(4), (4, 1, 1))
    hands[1, :3, :3] = Rotation.from_euler('x', 30, degrees=True).as_matrix()
    hands[1, 2, 3] = .015
    hands[2:, 2, 3] = 99.  # Separate open and later-grasp rows are not this carry.
    commands = np.array([-1., -1., 2., -1.])
    originals = [v.copy() for v in (hands, objects, commands)]
    shift, report, arrays = grasp.post_anchor_corridor(hands, objects, commands, 0, faces, wall)
    peak = .015+.025*np.sin(np.pi/6)+.016*np.cos(np.pi/6)
    expected = (.036-peak)/np.cos(np.pi/6)
    assert report['additional_anchor_axis_translation_m'] == pytest.approx(expected)
    assert abs(expected-(.036-peak)) > .0005  # The global-axis shortcut is wrong.
    np.testing.assert_allclose(shift[:3, 3], [0, 0, expected], atol=1e-12)
    np.testing.assert_array_equal(arrays['corridor_frame_indices'], [0, 1])
    np.testing.assert_allclose(arrays['corridor_axial_response'], [1, np.cos(np.pi/6)], atol=1e-12)
    assert arrays['corridor_pad_axial_bounds_after_m'][:, 1].max() == pytest.approx(.036)
    assert report['completed_frames'] == report['total_frames'] == 2
    for actual, previous in zip((hands, objects, commands), originals):
        np.testing.assert_array_equal(actual, previous)


@pytest.mark.parametrize('degrees, reason', [(180, 'No common'), (90, 'zero-response')])
def test_post_anchor_corridor_rejects_inconsistent_or_zero_response_constraints(degrees, reason):
    _, faces = pad_geometry()
    hands, objects = np.tile(np.eye(4), (2, 1, 1)), np.tile(np.eye(4), (2, 1, 1))
    hands[1, :3, :3] = Rotation.from_euler('x', degrees, degrees=True).as_matrix()
    hands[1, 2, 3] = .1
    with pytest.raises(ValueError, match=reason):
        grasp.post_anchor_corridor(hands, objects, [-1, -1], 0, faces,
                                   grasp.mesh_wall(cylinder_vertices()))


@pytest.mark.parametrize('settings', [dict(height_policy='anchor'),
    dict(height_policy='post_anchor_closed_corridor'),
    dict(axial_offset_m=.004, preserve_pre_offset_initial_target=True)])
def test_apply_keeps_source_object_state_and_arrays_and_invalidates_all_admissions(tmp_path, monkeypatch, settings):
    vertices = cylinder_vertices()
    text = ' '.join(map(str, vertices.ravel()))
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><asset><mesh name="native" vertex="{text}"/></asset>
    <worldbody><body name="object"><freejoint name="object_free"/><geom type="mesh" mesh="native"/></body>
    </worldbody></mujoco>''')
    state = {'angle': 1.}
    def initialize(model, data, robot, q0, mimics, angles):
        assert angles['l_hand_finger'] == 2.
        state['angle'] = angles['r_hand_finger']
        mujoco.mj_forward(model, data)
    def pads(model, data):
        pad, _ = pad_geometry()
        pad['points_tcp'] *= state['angle']
        pad['gap_m'] *= state['angle']
        return pad
    def faces(model, data):
        _, result = pad_geometry()
        for face in result:
            face[:, :, 1] *= state['angle']
        return result
    monkeypatch.setattr(grasp, 'initialize', initialize)
    monkeypatch.setattr(grasp, 'pad_surfaces', pads)
    monkeypatch.setattr(grasp, 'finite_pad_faces', faces)
    hand = np.tile(np.eye(4), (3, 1, 1))
    hand[:, :3, :3] = Rotation.from_euler('x', 20, degrees=True).as_matrix()
    objects = np.tile(np.eye(4), (3, 1, 1))
    details = dict(object_id='target', pad_alignment=dict(retarget_anchor_frame=1,
        inferred_contact_angle_rad=1., section=dict(center_object=[0, 0, .005])),
        robot_initialization={'admitted': True}, robot_grasp_approach={'admitted': True},
        robot_fixture_clearance={'admitted': True})
    manifest = dict(objects={'target': dict(body='object', joint='object_free')}, mimics={})
    prepared = (model, 'original xml', manifest, {'original': np.arange(3)}, {}, details,
                np.zeros(21), np.array([0., .01, .02]), np.zeros((3, 17)),
                np.array([2., -.06, 2.]), hand, objects)
    original = copy.deepcopy(details)
    result = grasp.apply(prepared, None, tmp_path/'proposal', **settings)
    for index in [0, 1, 2, 3, 4, 6, 7, 8, 9, 11]:
        assert result[index] is prepared[index]
    assert prepared[5] == original
    for key in ['robot_initialization', 'robot_grasp_approach', 'robot_fixture_clearance']:
        assert key not in result[5]
        assert key in result[5]['cylindrical_grasp']['invalidated_prior_admissions']
    assert result[5]['requires_full_ik_and_collision_admission']
    assert not result[5]['joint_reference_valid']
    assert not result[5]['cylindrical_grasp']['physics_validated']
    with np.load(tmp_path/'proposal'/'attachment-targets.npz') as artifact:
        np.testing.assert_array_equal(artifact['original_hand_goals'], hand)
        np.testing.assert_array_equal(artifact['original_object_goals'], objects)
        np.testing.assert_array_equal(artifact['original_time_s'], prepared[7])
        if settings.get('axial_offset_m'):
            previous = artifact['pre_axial_offset_hand_goals']
            shift = artifact['explicit_axial_offset_attachment']
            np.testing.assert_array_equal(result[10][0], previous[0])
            np.testing.assert_array_equal(result[10][1:], (previous @ shift)[1:])
            assert result[5]['cylindrical_grasp']['selected_midpoint_height_m'] == pytest.approx(.009)
            assert result[5]['cylindrical_grasp']['explicit_axial_offset']['every_closed_source_row_axially_supported']


def test_explicit_axial_offset_checks_pre_anchor_closed_rows_and_exact_rotated_response():
    _, faces = pad_geometry()
    hands = np.tile(np.eye(4), (4, 1, 1))
    objects = hands.copy()
    hands[1, :3, :3] = Rotation.from_euler('x', 30, degrees=True).as_matrix()
    hands[1, 2, 3] = .005
    commands = np.array([2., -1., -1., 2.])
    before = hands.copy()
    shift, report, arrays = grasp.axial_offset(hands, objects, commands, 2, faces,
                                               grasp.mesh_wall(cylinder_vertices()), .003)
    np.testing.assert_allclose(shift[:3, 3], [0, 0, .003], atol=1e-12)
    np.testing.assert_array_equal(arrays['axial_offset_closed_source_rows'], [1, 2])
    np.testing.assert_allclose(arrays['axial_offset_response'], [np.cos(np.pi/6), 1.], atol=1e-12)
    np.testing.assert_allclose(arrays['axial_offset_bounds_after_m']-arrays['axial_offset_bounds_before_m'],
                               np.repeat((.003*np.array([np.cos(np.pi/6), 1.]))[:, None], 2, axis=1))
    assert report['completed_frames'] == report['total_frames'] == 2
    np.testing.assert_array_equal(hands, before)
    # The pre-anchor frame is the limiting one. An anchor-only check would
    # incorrectly admit this larger request; there is no silent wall clamping.
    with pytest.raises(ValueError, match='no clamping'):
        grasp.axial_offset(hands, objects, commands, 2, faces,
                           grasp.mesh_wall(cylinder_vertices()), .012)


@pytest.mark.parametrize('settings', [dict(axial_offset_m=.004),
    dict(preserve_pre_offset_initial_target=True), dict(axial_offset_m=float('nan')),
    dict(axial_offset_m=.021, preserve_pre_offset_initial_target=True),
    dict(axial_offset_m=.004, preserve_pre_offset_initial_target=True,
         height_policy='post_anchor_closed_corridor')])
def test_explicit_axial_offset_rejects_ambiguous_or_unbounded_api(tmp_path, settings):
    prepared = (None, None, None, None, None, {}, None, None, None, None, None, None)
    with pytest.raises(ValueError, match='explicit pre-offset'):
        grasp.apply(prepared, None, tmp_path/'invalid', **settings)


@pytest.mark.parametrize('offset', [-90., -30., 15., 90., 180.])
def test_explicit_grasp_azimuth_rotates_robot_about_unchanged_cylinder(offset):
    vertices, (pad, faces) = cylinder_vertices(), pad_geometry()
    wall=grasp.mesh_wall(vertices)
    hand,obj=np.eye(4),np.eye(4)
    hand[:3,:3]=Rotation.from_euler('x',20.,degrees=True).as_matrix()
    original=copy.deepcopy((vertices,hand,obj))
    zero,baseline=grasp.derive_attachment(hand,obj,pad,faces,wall,vertices,[0,0,.005])
    attachment,report=grasp.derive_attachment(hand,obj,pad,faces,wall,vertices,[0,0,.005],azimuth_offset_deg=offset)
    around=Rotation.from_rotvec(np.asarray(wall['axis_object'])*np.deg2rad(offset)).as_matrix()
    np.testing.assert_allclose((hand@attachment)[:3,:3],around@(hand@zero)[:3,:3],atol=1e-12)
    np.testing.assert_allclose(report['desired_pad_midpoint_object_m'],baseline['desired_pad_midpoint_object_m'],atol=1e-12)
    assert report['requested_azimuth_offset_deg']==offset
    assert all(report['static_finite_patch_checks'].values())
    for value,before in zip((vertices,hand,obj),original):np.testing.assert_array_equal(value,before)
    explicit,_=grasp.derive_attachment(hand,obj,pad,faces,wall,vertices,[0,0,.005],azimuth_offset_deg=0.)
    np.testing.assert_array_equal(explicit,zero)


@pytest.mark.parametrize('offset',[181.,-181.,float('nan'),float('inf')])
def test_invalid_grasp_azimuth_fails_closed(offset):
    vertices,(pad,faces)=cylinder_vertices(),pad_geometry()
    with pytest.raises(ValueError,match='azimuth'):
        grasp.derive_attachment(np.eye(4),np.eye(4),pad,faces,grasp.mesh_wall(vertices),vertices,[0,0,0],azimuth_offset_deg=offset)
