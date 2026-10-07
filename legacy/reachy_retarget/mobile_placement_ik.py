"""Bounded mobile IK for inserted placement knots; no simulation stepping."""
import mujoco
import numpy as np
from scipy.optimize import minimize

from .geometry_clearance import RobotFixtureClearance, reachy_wheel_floor_pairs
from .physics import initialize


def continuous_yaw(yaw, reference):
    """Represent the same planar orientation nearest a declared angle."""
    if not np.isfinite([yaw, reference]).all():
        raise ValueError('Finite yaw and continuity reference required')
    return float(yaw + 2*np.pi*np.round((reference-yaw)/(2*np.pi)))


def transport_planar_seed(robot, seed, goal):
    """Transport a planar base with the TCP yaw/XY change, preserving joints.

    This is only an IK seed. The resulting pose receives all ordinary target
    and fixture checks. No simulation state is accessed or assigned.
    """
    current = robot.fk(seed)[1]
    relative = goal[:3, :3] @ current[:3, :3].T
    # The ZYX yaw factor is unique away from a vertical projected X axis.
    if np.linalg.norm(relative[:2, 0]) < 1e-8:
        raise ValueError('Planar orientation transport has an undefined yaw')
    yaw = float(np.arctan2(relative[1, 0], relative[0, 0]))
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    base = robot.base(seed)
    xy = goal[:2, 3] + rotation @ (base[:2] - current[:2, 3])
    transported = seed.copy()
    transported[:4] = [xy[0], xy[1], np.cos(base[2] + yaw), np.sin(base[2] + yaw)]
    return transported, dict(delta_yaw_rad=yaw, original_base=base.tolist(),
                             transported_base=[float(xy[0]), float(xy[1]), float(base[2] + yaw)])


class Solver:
    """Follow a declared TCP path with an accumulated fixture active set.

    The caller retains the original source configuration at both ends and must
    verify every interpolated row. This class certifies neither a trajectory nor
    physical success. All object states remain at the compiled reset.
    """
    def __init__(self, prepared, robot, *, base_bounds_xyyaw,
                 minimum_fixture_gap_m=.003, max_iterations=160,
                 max_active_set_updates=4, support_pairs=None, orientation_transport=False,
                 pose_tolerance_constraints=False, initial_base_yaw=None,
                 minimum_motion=False, motion_speed_caps=(.3, .3, .8, .65, .65, .65, .65, .65, .65, .65)):
        self.model, self.manifest, self.robot = prepared[0], prepared[2], robot
        bounds = np.asarray(base_bounds_xyyaw, float)
        if (bounds.shape != (3, 2) or not np.isfinite(bounds).all()
                or np.any(bounds[:, 0] >= bounds[:, 1])
                or type(max_iterations) is not int or not 1 <= max_iterations <= 500
                or type(max_active_set_updates) is not int or not 1 <= max_active_set_updates <= 8):
            raise ValueError('Explicit finite mobile placement bounds and bounded budgets required')
        self.bounds = list(map(tuple, bounds)) + list(zip(
            robot.r.model.lowerPositionLimit[robot.arm_ids[7:]] + .031,
            robot.r.model.upperPositionLimit[robot.arm_ids[7:]] - .031))
        active = self.manifest['objects'][prepared[5]['object_id']]['body']
        if support_pairs is None:
            support_pairs = reachy_wheel_floor_pairs(self.model)
        support_pairs = tuple(support_pairs)
        self._checker_options = dict(active_object_bodies=(active,), support_pairs=support_pairs)
        self.checker = RobotFixtureClearance(self.model, active_object_bodies=(active,),
            support_pairs=support_pairs, minimum_m=minimum_fixture_gap_m)
        self.data = mujoco.MjData(self.model)
        self.minimum = minimum_fixture_gap_m
        self.max_iterations, self.max_updates = max_iterations, max_active_set_updates
        if type(orientation_transport) is not bool:
            raise ValueError('orientation_transport must be an explicit boolean')
        self.orientation_transport = orientation_transport
        if type(pose_tolerance_constraints) is not bool:
            raise ValueError('pose_tolerance_constraints must be an explicit boolean')
        self.pose_tolerance_constraints = pose_tolerance_constraints
        self.motion_speed_caps = np.asarray(motion_speed_caps, float)
        if (type(minimum_motion) is not bool or (minimum_motion and not pose_tolerance_constraints)
                or self.motion_speed_caps.shape != (10,) or not np.isfinite(self.motion_speed_caps).all()
                or np.any(self.motion_speed_caps <= 0)):
            raise ValueError('Minimum-motion optimization requires hard pose constraints and ten positive speed scales')
        self.minimum_motion = minimum_motion
        if initial_base_yaw is not None and not np.isfinite(initial_base_yaw):
            raise ValueError('Finite initial base yaw required')
        self.base_yaw = None if initial_base_yaw is None else float(initial_base_yaw)
        self.pairs, self.records = [], []
        self.metadata = dict(base_bounds_xyyaw=bounds.tolist(), minimum_fixture_gap_m=minimum_fixture_gap_m,
            max_iterations=max_iterations, max_active_set_updates=max_active_set_updates,
            support_pairs=list(support_pairs), displacement_regularizer=1e-8,
            orientation_transport=orientation_transport,
            pose_tolerance_constraints=pose_tolerance_constraints,
            minimum_motion=minimum_motion, motion_speed_caps=self.motion_speed_caps.tolist(),
            motion_anchor='Previous physical reference knot before optional orientation seed transport',
            initial_base_yaw=self.base_yaw,
            base_yaw_representation='Continuous scalar through optimization and interpolation, anchored to the original release row',
            pose_tolerances_m_rad=[.002, .02], pose_constraint_numerical_margin_fraction=1e-5,
            scope='Inserted robot placement only; original source prefix/suffix and object states unchanged')

    def solve(self, goal, seed, apertures, *, phase, weight, position_interior_reserve_m=0.,
              fixture_interior_reserve_m=0., planar_constraints=None):
        robot = self.robot
        discs = None if planar_constraints is None else np.asarray(planar_constraints, float)
        if discs is not None and (discs.ndim != 2 or discs.shape[1] != 3 or not len(discs)
                or not np.isfinite(discs).all() or np.any(discs[:, 2] <= 0)):
            raise ValueError('Planar speed discs require finite XY centers and positive radii')
        if (not np.isfinite(position_interior_reserve_m)
                or not 0 <= position_interior_reserve_m <= .001
                or (position_interior_reserve_m and not self.pose_tolerance_constraints)):
            raise ValueError('A position interior reserve in [0, 1 mm] requires hard pose constraints')
        position_limit = .002 - float(position_interior_reserve_m)
        if (not np.isfinite(fixture_interior_reserve_m)
                or not 0 <= fixture_interior_reserve_m <= .001
                or self.minimum + fixture_interior_reserve_m > .05):
            raise ValueError('A fixture interior reserve in [0, 1 mm] must remain within 50 mm')
        fixture_limit = self.minimum + float(fixture_interior_reserve_m)
        # The query cap must cover the stricter reserve too, otherwise a pair
        # between the ordinary and reserved limits could never join the active set.
        checker = (RobotFixtureClearance(self.model, minimum_m=fixture_limit, **self._checker_options)
                   if fixture_interior_reserve_m else self.checker)
        goal, seed, apertures = np.asarray(goal), np.asarray(seed), np.asarray(apertures)
        if (goal.shape != (4, 4) or not np.isfinite(goal).all()
                or apertures.ndim != 1 or not len(apertures) or not np.isfinite(apertures).all()
                or np.any(apertures < -.06) or np.any(apertures > 2.)):
            raise ValueError('Finite target and declared aperture envelope required')
        transport = None
        previous = np.r_[robot.base(seed), seed[robot.arm_ids[7:]]]
        if self.base_yaw is not None:
            previous[2] = continuous_yaw(previous[2], self.base_yaw)
        if self.orientation_transport:
            seed, transport = transport_planar_seed(robot, seed, goal)
        x0 = np.r_[robot.base(seed), seed[robot.arm_ids[7:]]]
        if self.base_yaw is not None:
            expected_yaw = self.base_yaw + (transport['delta_yaw_rad'] if transport else 0.)
            x0[2] = continuous_yaw(x0[2], expected_yaw)
        x = x0.copy()

        def configuration(value):
            q = seed.copy()
            q[:4] = [value[0], value[1], np.cos(value[2]), np.sin(value[2])]
            q[robot.arm_ids[7:]] = value[3:]
            return q

        def objective(value):
            actual = robot.fk(configuration(value))[1]
            if self.minimum_motion:
                position = (goal[:3, 3]-actual[:3, 3])/.002
                rotation = robot.pin.log3(goal[:3, :3] @ actual[:3, :3].T)/.02
                return float(np.sum(((value-previous)/self.motion_speed_caps)**2)
                             + 1e-6*(position @ position+rotation @ rotation))
            error = np.r_[goal[:3, 3] - actual[:3, 3],
                          .35 * robot.pin.log3(goal[:3, :3] @ actual[:3, :3].T)]
            return float(error @ error + 1e-8 * np.sum((value - x0) ** 2))

        def synchronize(q, aperture):
            initialize(self.model, self.data, robot, q, self.manifest['mimics'],
                       {'l_hand_finger': 2., 'r_hand_finger': float(aperture)})

        def constraints(value):
            result = []
            if discs is not None:
                result.extend(1.-np.sum((value[:2]-discs[:,:2])**2,axis=1)/discs[:,2]**2)
            for aperture in np.unique([apertures.min(), apertures.max()]):
                synchronize(configuration(value), aperture)
                result.extend(mujoco.mj_geomDistance(self.model, self.data, g, h, 2., None)
                              - fixture_limit - 1e-5 for g, h in self.pairs)
            if self.pose_tolerance_constraints:
                actual = robot.fk(configuration(value))[1]
                position = goal[:3, 3] - actual[:3, 3]
                rotation = robot.pin.log3(goal[:3, :3] @ actual[:3, :3].T)
                # A tiny conservative numerical margin avoids claiming a
                # rounded boundary point passed the unchanged strict check.
                limits = np.array([position_limit, .02]) * (1. - 1e-5)
                result.extend([1. - float(position @ position) / limits[0] ** 2,
                               1. - float(rotation @ rotation) / limits[1] ** 2])
            return np.asarray(result)

        trace = []
        for attempt in range(self.max_updates):
            constraint = ([dict(type='ineq', fun=constraints)]
                          if self.pairs or self.pose_tolerance_constraints or discs is not None else [])
            optimized = minimize(objective, x, method='SLSQP', bounds=self.bounds,
                constraints=constraint, options=dict(maxiter=self.max_iterations, ftol=1e-12))
            x = optimized.x
            q = configuration(x)
            added, worst = [], None
            for aperture in apertures:
                synchronize(q, aperture)
                found = checker.inspect(self.data)
                if worst is None or found['minimum_signed_distance_or_lower_bound_m'] < worst['minimum_signed_distance_or_lower_bound_m']:
                    worst = found
                if not found['admitted']:
                    pair = tuple(self.model.geom(name).id for name in found['closest_geom_pair'])
                    if pair not in self.pairs:
                        added.append(pair)
            self.pairs.extend(dict.fromkeys(added))
            trace.append(dict(iteration=attempt, success=bool(optimized.success),
                              message=str(optimized.message), added_pairs=len(set(added))))
            if not added:
                break
        actual = robot.fk(q)[1]
        errors = [float(np.linalg.norm(goal[:3, 3] - actual[:3, 3])),
                  float(np.linalg.norm(robot.pin.log3(goal[:3, :3] @ actual[:3, :3].T)))]
        self.base_yaw = float(x[2])
        self.records.append(dict(phase=phase, weight=float(weight), base=self.reference_base(q).tolist(),
            q=q.tolist(), ik_errors=errors, fixture=worst, optimizer_trace=trace,
            orientation_transport=transport,
            position_interior_reserve_m=float(position_interior_reserve_m),
            solve_position_tolerance_m=position_limit,
            fixture_interior_reserve_m=float(fixture_interior_reserve_m),
            solve_minimum_fixture_gap_m=fixture_limit,
            planar_neighbor_discs=None if discs is None else discs.tolist()))
        return q

    def reference_base(self, q):
        """Return XY and the solver's continuous yaw for control references."""
        base = self.robot.base(q).copy()
        if self.base_yaw is not None:
            base[2] = continuous_yaw(base[2], self.base_yaw)
        return base

    def inspect(self, q, apertures):
        """Recheck an interpolated row independently of optimizer convergence."""
        worst = None
        for aperture in apertures:
            initialize(self.model, self.data, self.robot, q, self.manifest['mimics'],
                       {'l_hand_finger': 2., 'r_hand_finger': float(aperture)})
            found = self.checker.inspect(self.data)
            if worst is None or found['minimum_signed_distance_or_lower_bound_m'] < worst['minimum_signed_distance_or_lower_bound_m']:
                worst = found
        return worst
