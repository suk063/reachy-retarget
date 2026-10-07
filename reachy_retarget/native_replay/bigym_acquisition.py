"""Measured plate rendezvous geometry, independent of the free-object integrator.

All hypothetical qpos assignments occur in a separately created kinematic data
object. Only robot actuator targets returned here are used by physical playback.
"""
import json
from pathlib import Path
import numpy as np
from ..acquisition_rendezvous import Rendezvous


def _save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def line_intervals(equations, midpoint, axis):
    result = []
    for name, eq in equations:
        n, den = eq[:, :3]@midpoint+eq[:, 3], eq[:, :3]@axis
        if ((abs(den) < 1e-12) & (n > 1e-10)).any():
            continue
        lo = np.max(-n[den < -1e-12]/den[den < -1e-12], initial=-np.inf)
        hi = np.min(-n[den > 1e-12]/den[den > 1e-12], initial=np.inf)
        if lo <= hi and np.isfinite([lo, hi]).all():
            result.append((name, float(lo), float(hi)))
    return result


def choose_chord(intervals):
    if not intervals:
        raise ValueError('Closing line does not intersect finite plate geometry')
    # A local nearest connected convex collider, not the hull of the whole plate.
    return min(intervals, key=lambda v: abs((v[1]+v[2])/2))


def _vertices(model, data, geom):
    mesh = int(model.geom_dataid[geom])
    if mesh < 0:
        raise ValueError('Verified plate and pad mesh geometry required')
    a, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    return model.mesh_vert[a:a+n]@data.geom_xmat[geom].reshape(3, 3).T+data.geom_xpos[geom]


def _face_vertices(model, data):
    ids = [model.body(n).id for n in ('l_hand_distal_link', 'l_hand_distal_mimic_link')]
    result = []
    for side, body in enumerate(ids):
        inward = data.xpos[ids[1-side]]-data.xpos[body]
        inward /= np.linalg.norm(inward)
        triangles = []
        for g in np.flatnonzero(model.geom_bodyid == body):
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            mesh = int(model.geom_dataid[g]); v = _vertices(model, data, g)
            tri = v[model.mesh_face[model.mesh_faceadr[mesh]:model.mesh_faceadr[mesh]+model.mesh_facenum[mesh]]]
            n = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
            triangles.extend(tri[n@inward > .95*np.linalg.norm(n, axis=1)])
        result.append(np.unique(np.asarray(triangles).reshape(-1, 3), axis=0))
    return result


def face_margin(vertices, point):
    from scipy.spatial import ConvexHull
    center = vertices.mean(0)
    _, _, basis = np.linalg.svd(vertices-center)
    eq = ConvexHull((vertices-center)@basis[:2].T).equations
    return float(np.min(-(eq[:, :2]@((point-center)@basis[:2].T)+eq[:, 2])))


def minimum_patch_translation(faces, anchor, axis, axis_distance, *, margin=.004):
    """Minimum robot translation placing one fixed patch within both pad faces.

    Face halfspaces and the closing-axis displacement are linear constraints.
    This does not move the patch, rotate the hand, or inflate finite pad area.
    """
    from scipy.spatial import ConvexHull
    from scipy.optimize import minimize
    A, b = [], []
    for vertices in faces:
        center = vertices.mean(0)
        _, _, basis = np.linalg.svd(vertices-center)
        eq = ConvexHull((vertices-center)@basis[:2].T).equations
        normals = eq[:, :2]@basis[:2]
        A.extend(normals)
        b.extend(margin+normals@(np.asarray(anchor)-center)+eq[:, 2])
    A, b = np.asarray(A), np.asarray(b)
    axis = np.asarray(axis, dtype=float)
    # Millimeter optimization coordinates avoid numerical tolerances at 1e-6.
    fit = minimize(lambda x: .5*x@x, axis*axis_distance*1000,
        jac=lambda x: x, method='SLSQP', constraints=[
            {'type': 'ineq', 'fun': lambda x: A@x-b*1000, 'jac': lambda x: A},
            {'type': 'eq', 'fun': lambda x: axis@x-axis_distance*1000, 'jac': lambda x: axis}],
        options={'ftol': 1e-11, 'maxiter': 100})
    correction = fit.x/1000
    if not fit.success or not np.isfinite(correction).all() or (A@correction < b-1e-9).any():
        raise ValueError('Finite-patch minimum translation is infeasible')
    return correction


def nonpad_object_clearance(model, data, *, robot_bodies, object_bodies,
                            allowed_pad_bodies, minimum_m=.003):
    """Read-only signed clearance including palms and finger linkages.

    Only explicitly identified pad bodies are omitted. Their finite inward
    faces, forces and actual penetration require separate qualification; this
    result is never an admission of all distal-body contact.
    """
    import mujoco
    robot_bodies, object_bodies, pads = map(set,
        (robot_bodies, object_bodies, allowed_pad_bodies))
    if (not robot_bodies or not object_bodies or not pads <= robot_bodies
            or robot_bodies & object_bodies or not np.isfinite(minimum_m) or minimum_m <= 0):
        raise ValueError('Disjoint robot/object bodies and explicit robot pads required')
    active = [g for g in range(model.ngeom)
              if model.geom_contype[g] or model.geom_conaffinity[g]]
    robot_geoms = [g for g in active if int(model.geom_bodyid[g]) in robot_bodies-pads]
    object_geoms = [g for g in active if int(model.geom_bodyid[g]) in object_bodies]
    if not robot_geoms or not object_geoms:
        raise ValueError('Both robot non-pad and object collision geometry are required')
    bound, near = float(minimum_m+1e-6), []
    for g in robot_geoms:
        for h in object_geoms:
            # Bounding spheres give a conservative rejection of distant pairs.
            lower = (np.linalg.norm(data.geom_xpos[g]-data.geom_xpos[h])
                     -model.geom_rbound[g]-model.geom_rbound[h])
            if lower > minimum_m+1e-6:
                continue
            line = np.empty(6)
            distance = float(mujoco.mj_geomDistance(model, data, g, h, minimum_m+1e-6, line))
            bound = min(bound, distance)
            if distance < minimum_m:
                near.append({'robot_geom': model.geom(g).name,
                             'object_geom': model.geom(h).name, 'signed_distance_m': distance})
    return {'admitted_nonpad_only': not near,
            'minimum_signed_distance_or_lower_bound_m': bound,
            'minimum_required_m': float(minimum_m), 'violating_pairs': near,
            'allowed_pad_body_names': sorted(model.body(b).name for b in pads),
            'scope': 'Non-pad clearance only; finite pad-face/contact qualification remains required'}


class PlateRendezvous:
    def __init__(self, model, robot, raw, plan, manifest, timing, policy, output):
        from scipy.spatial import ConvexHull
        self.model, self.robot, self.raw, self.plan, self.manifest, self.timing = model, robot, raw, plan, manifest, timing
        self.output = Path(output); self.output.mkdir(exist_ok=False)
        if (policy.get('method') != 'measured_plate_rendezvous' or policy.get('source_row') != 550
                or not policy.get('evidence') or float(policy.get('max_centering_m', 0)) != .010):
            raise ValueError('Explicit evidence-bound row550 policy and 10mm correction cap required')
        self.policy = dict(policy)
        self.geometry_mode = policy.get("geometry_mode", "closing_axis")
        if self.geometry_mode not in ("closing_axis", "finite_patch_translation"):
            raise ValueError("Unknown measured geometry correction mode")
        self.machine = Rendezvous(timestep=float(model.opt.timestep), wait_limit_s=2., stable_s=.1)
        self.row = 550
        self.boundary = int(timing['source_boundary_step_index'][self.row])
        self.targets = timing['target_qpos'].copy()
        self.target_q = raw['robot_qpos'][self.row].copy()
        self.site = model.site('l_arm_tip_tcp').id
        self.plate = model.body('plate/').id
        bodies = {self.plate}
        for b in range(model.nbody):
            if int(model.body_parentid[b]) in bodies:
                bodies.add(b)
        self.plate_bodies = bodies
        self.fingers = [model.body(n).id for n in ('l_hand_distal_link', 'l_hand_distal_mimic_link')]
        self.plate_equations = []
        # Local collider equations remain valid while the freely moving body moves.
        import mujoco
        d = mujoco.MjData(model); mujoco.mj_forward(model, d)
        R, p = d.xmat[self.plate].reshape(3, 3), d.xpos[self.plate]
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) in bodies and (model.geom_contype[g] or model.geom_conaffinity[g]):
                local = (_vertices(model, d, g)-p)@R
                self.plate_equations.append((model.geom(g).name, ConvexHull(local).equations))
        self.previous_position = None
        self.admission = None
        self.guard_rows = []
        self.delta_world = np.zeros(3)
        _save(self.output/'policy.json', policy)

    def _local_geometry(self, data, matrix):
        R, p = data.xmat[self.plate].reshape(3, 3), data.xpos[self.plate]
        pts = self.pad_points@matrix[:3, :3].T+matrix[:3, 3]
        midpoint = pts.mean(0)
        axis = pts[1]-pts[0]; axis /= np.linalg.norm(axis)
        chord = choose_chord(line_intervals(self.plate_equations, R.T@(midpoint-p), R.T@axis))
        anchor = R@np.asarray(self.plan['grasp_relocation']['derived_pinch_local_m'])+p
        margins = [face_margin(v@matrix[:3, :3].T+matrix[:3, 3], anchor) for v in self.face_points]
        return chord, axis, margins

    def prepare(self, integration_data):
        """Fresh full reference IK/clearance; real integration state is read-only."""
        import mujoco
        from ..physics import initialize
        from ..feasible_pose import _pad_surfaces
        from ..geometry_clearance import RobotFixtureClearance, reachy_wheel_floor_pairs
        data = mujoco.MjData(self.model)
        size = mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION)
        state = np.empty(size)
        mujoco.mj_getState(self.model, integration_data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_setState(self.model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.model, data)
        np.savez_compressed(self.output/'rendezvous-integration-state.npz', physics_state=state,
                            qpos=integration_data.qpos, qvel=integration_data.qvel, timestamp=integration_data.time)
        nominal = float(self.plan['pad_calibration']['hands']['left']['nominal_gripper_rad'])
        initialize(self.model, data, self.robot, self.target_q, self.manifest['reachy']['mimics'],
                   {'l_hand_finger': nominal, 'r_hand_finger': 2.})
        R, p = data.site_xmat[self.site].reshape(3, 3), data.site_xpos[self.site]
        self.pad_points = np.asarray(_pad_surfaces(self.model, data, 'left')['points_tcp_m'])
        self.face_points = [(v-p)@R for v in _face_vertices(self.model, data)]
        target = self.robot.fk(self.target_q)[0].copy()
        chord, axis, margins = self._local_geometry(data, target)
        offset = (chord[1]+chord[2])/2
        self.delta_world = axis*offset
        original_margins = list(margins)
        if self.geometry_mode == "finite_patch_translation":
            plate_R = data.xmat[self.plate].reshape(3, 3)
            anchor = plate_R@np.asarray(self.plan['grasp_relocation']['derived_pinch_local_m'])+data.xpos[self.plate]
            faces = [v@target[:3, :3].T+target[:3, 3] for v in self.face_points]
            for _ in range(6):
                self.delta_world = minimum_patch_translation(faces, anchor, axis, offset, margin=.004)
                corrected = target.copy(); corrected[:3, 3] += self.delta_world
                corrected_chord, _, margins = self._local_geometry(data, corrected)
                residual = (corrected_chord[1]+corrected_chord[2])/2
                if abs(residual) < 1e-5:
                    break
                offset += residual
            if abs(residual) >= 1e-5:
                raise ValueError('Finite-patch centering did not converge within six bounded proposals')
        geometry = dict(source_row=self.row, geometry_mode=self.geometry_mode,
                        original_finite_face_margins_m=original_margins,
                        desired_face_margin_m=.004 if self.geometry_mode == "finite_patch_translation" else .003, measured_plate_pose=np.r_[data.xpos[self.plate], data.xquat[self.plate]].tolist(),
                        chord_collider=chord[0], chord_interval_m=list(chord[1:]), closing_axis_world=axis.tolist(),
                        correction_world_m=self.delta_world.tolist(), correction_m=float(np.linalg.norm(self.delta_world)),
                        finite_face_margins_m=margins, max_centering_m=.010, minimum_face_margin_m=.003,
                        robot_only=True, actual_integrator_state_writes=0)
        _save(self.output/'measured-centering.json', geometry)
        if np.linalg.norm(self.delta_world) > .010 or min(margins) < .003:
            raise ValueError('Measured finite-pad rendezvous is outside declared correction/footprint bounds')
        qpos = self.raw['robot_qpos'].copy()
        targets = self.raw['target_hand_matrix'].copy()
        times = self.timing['physical_clock'][self.timing['source_boundary_step_index']]
        x = np.clip((times-times[self.row])/.2, 0, 1)
        weights = 1-(6*x**5-15*x**4+10*x**3)
        weights[:self.row] = 0
        targets[:, 0, :3, 3] += weights[:, None]*self.delta_world
        active = self.robot.active.copy()
        try:
            self.robot.active = self.robot.arm_v[:7]
            for i in np.flatnonzero(weights):
                qpos[i], _, _ = self.robot.ik(targets[i], q=qpos[i], active_hands=(0,), iterations=120)
        finally:
            self.robot.active = active
        self.target_q = qpos[self.row].copy()
        self.target_matrix = targets[self.row, 0].copy()
        # Retain all original source intervals and interpolate corrected robot rows.
        for i, n in enumerate(self.timing['substeps_per_source_interval']):
            first, last = self.timing['source_boundary_step_index'][i:i+2]
            alpha = np.arange(1, n+1)/n
            self.targets[first:last, self.robot.arm_ids] = (qpos[i, self.robot.arm_ids]
                +alpha[:, None]*(qpos[i+1, self.robot.arm_ids]-qpos[i, self.robot.arm_ids]))
        # Approach is unchanged; the correction is commanded only while waiting.
        self.targets[:self.boundary] = self.timing['target_qpos'][:self.boundary]
        np.savez_compressed(self.output/'corrected-reference.npz', original_source_clock=self.raw['original_clock_s'],
                            original_robot_qpos=self.raw['robot_qpos'], original_hand_targets=self.raw['target_hand_matrix'],
                            robot_qpos=qpos, target_hand_matrix=targets, correction_weight=weights,
                            per_step_target_qpos=self.targets, correction_world_m=self.delta_world)
        checker = RobotFixtureClearance(self.model, active_object_bodies=('plate/',),
                                        support_pairs=reachy_wheel_floor_pairs(self.model), minimum_m=.003)
        robots = set(checker.metadata['robot_body_ids'])
        plate_j = self.model.joint('plate/').id
        addr = int(self.model.jnt_qposadr[plate_j]); object_state = data.qpos[addr:addr+7].copy()
        angles = np.unique(np.r_[np.linspace(nominal-.05, 2., 5), nominal])
        # Qualify the complete hand against the actual free plate at entry.
        # Fixture checks deliberately exclude this moving object, so they alone
        # cannot detect an impossible target that drives the palm into it.
        target_object_rows = []
        for angle in angles:
            initialize(self.model, data, self.robot, self.target_q, self.manifest['reachy']['mimics'],
                       {'l_hand_finger': float(angle), 'r_hand_finger': 2.})
            target_object_rows.append(dict(aperture_rad=float(angle), **nonpad_object_clearance(
                self.model, data, robot_bodies=robots, object_bodies=self.plate_bodies,
                allowed_pad_bodies=self.fingers)))
        object_admitted = all(row['admitted_nonpad_only'] for row in target_object_rows)
        _save(self.output/'rendezvous-nonpad-object-admission.json', dict(
            admitted_nonpad_only=object_admitted, source_row=self.row, rows=target_object_rows,
            measured_object_pose=np.r_[data.xpos[self.plate], data.xquat[self.plate]].tolist(),
            scene_sha256=self.manifest['scene_sha256'], actual_integrator_state_writes=0,
            scope='Rendezvous target/aperture envelope only; full path and finite pad guards remain required'))
        if not object_admitted:
            raise ValueError('Measured rendezvous target has non-pad robot/plate obstruction')
        rows = []
        for i, q in enumerate(qpos):
            achieved = self.robot.fk(q)
            pe = np.linalg.norm(achieved[:, :3, 3]-targets[i, :, :3, 3], axis=1)
            re = [float(np.linalg.norm(self.robot.pin.log3(targets[i, h, :3, :3]@achieved[h, :3, :3].T))) for h in (0, 1)]
            gap, depth, fixture = float(self.robot.r.geometry(q)[0]), 0., np.inf
            for angle in angles:
                initialize(self.model, data, self.robot, q, self.manifest['reachy']['mimics'],
                           {'l_hand_finger': float(angle), 'r_hand_finger': 2.})
                if not np.array_equal(data.qpos[addr:addr+7], object_state):
                    raise ValueError('Kinematic check altered fixed clone plate state')
                fixture = min(fixture, checker.inspect(data)['minimum_signed_distance_or_lower_bound_m'])
                for c in data.contact[:data.ncon]:
                    if {int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])} <= robots:
                        depth = max(depth, -float(c.dist))
            rows.append([fixture, gap, depth, *pe, *re])
            if (i+1) % 100 == 0:
                _save(self.output/'progress.json', {'checked_frames': i+1, 'source_frames': len(qpos)})
        rows = np.asarray(rows)
        passed = bool(np.isfinite(rows).all() and (rows[:, 0] >= .003).all() and (rows[:, 1] >= .009).all()
                      and (rows[:, 2] <= .002).all() and (rows[:, 3:5] <= .002).all() and (rows[:, 5:] <= .02).all())
        np.savez_compressed(self.output/'full-scene-admission.npz', metrics=rows,
                            columns=['fixture_gap_m', 'self_gap_m', 'self_depth_m', 'left_pos_m', 'right_pos_m', 'left_rot_rad', 'right_rot_rad'])
        self.admission = dict(admitted=passed, source_frames=len(rows), fixture_min_m=float(rows[:, 0].min()),
                              self_min_m=float(rows[:, 1].min()), self_depth_max_m=float(rows[:, 2].max()),
                              position_max_m=float(rows[:, 3:5].max()), rotation_max_rad=float(rows[:, 5:].max()),
                              scene_sha256=self.manifest['scene_sha256'], aperture_samples_rad=angles.tolist(),
                              checker=checker.metadata, physics_validated=False,
                              scope='All source-frame corrected two-hand references and sampled aperture envelope; active plate contacts assessed during integration; actual state collision gates remain required')
        _save(self.output/'admission.json', self.admission)
        check = np.empty(size)
        mujoco.mj_getState(self.model, integration_data, check, mujoco.mjtState.mjSTATE_INTEGRATION)
        if not np.array_equal(check, state):
            raise ValueError('Planning modified actual integration state')
        if not passed:
            raise ValueError('Corrected full reference failed IK/scene admission')
        self.machine.begin()

    def pad_forces(self, data):
        import mujoco
        forces = np.zeros(2)
        for i, c in enumerate(data.contact[:data.ncon]):
            bodies = {int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])}
            if not bodies & self.plate_bodies:
                continue
            for k, b in enumerate(self.fingers):
                if b in bodies:
                    f = np.zeros(6); mujoco.mj_contactForce(self.model, data, i, f)
                    forces[k] += max(0., f[0])
        return forces

    def update(self, data, forces):
        matrix = np.eye(4)
        matrix[:3, :3] = data.site_xmat[self.site].reshape(3, 3)
        matrix[:3, 3] = data.site_xpos[self.site]
        position = float(np.linalg.norm(matrix[:3, 3]-self.target_matrix[:3, 3]))
        rotation = float(np.linalg.norm(self.robot.pin.log3(self.target_matrix[:3, :3]@matrix[:3, :3].T)))
        speed = np.inf if self.previous_position is None else float(np.linalg.norm(matrix[:3, 3]-self.previous_position)/self.model.opt.timestep)
        self.previous_position = matrix[:3, 3].copy()
        try:
            chord, _, margins = self._local_geometry(data, matrix)
            centered = abs((chord[1]+chord[2])/2)
        except ValueError:
            centered, margins = None, [None, None]
        aligned = bool(position <= .002 and rotation <= .02 and speed <= .02 and centered is not None
                       and centered <= .003 and min(margins) >= .003)
        angle = float(data.qpos[self.model.joint('l_hand_finger').qposadr[0]])
        nominal = float(self.plan['pad_calibration']['hands']['left']['nominal_gripper_rad'])
        bilateral = bool((np.asarray(forces) >= 1.).all() and angle <= nominal+.03)
        self.guard_rows.append(dict(timestamp=float(data.time), phase=self.machine.phase, tcp_position_error_m=position,
            tcp_rotation_error_rad=rotation, tcp_speed_m_s=None if not np.isfinite(speed) else speed,
            nominal_closed_pad_chord_center_error_m=centered, finite_pad_anchor_margins_m=margins,
            actual_pad_normal_forces_n=list(map(float, forces)), actual_gripper_rad=angle,
            alignment=aligned, bilateral_force=bilateral))
        self.machine.update(alignment=aligned, bilateral_force=bilateral)

    def report(self):
        result = dict(self.machine.report(), policy=self.policy, admission=self.admission,
                      geometry_guard='Nominal closed-pad CAD proxy transformed by measured TCP; actual bilateral forces independently required',
                      source_row=self.row, correction_world_m=self.delta_world.tolist(), original_source_intent_preserved=True,
                      arm_feedforward_enabled=False, object_state_writes_after_reset=0)
        _save(self.output/'guards.json', self.guard_rows)
        _save(self.output/'result.json', result)
        return result
