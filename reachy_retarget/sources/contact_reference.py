"""Object-environment penetration of the source itself (``SceneRef.reference``).

MuJoCo contacts are soft: a resting object sinks into its support, and a dropped one into
whatever it lands on, by an amount set by the source's contact parameters, masses and
speeds (robomimic: the Lift cube rests 2.6 mm deep in the table, the Square nut 7.6 mm,
Transport drops reach 19 mm). Tier P gates the object-environment depth of the Reachy
rollout relative to this source value (``validate.physics``), so it is measured on the
source's own recorded states.

:class:`ObjectEnvironmentDepth` is fed the source model's ``MjData`` after each recorded
state was set and ``mj_forward`` ran (kinematics and collision; nothing is stepped). It keeps
the deepest contact (``-contact.dist``) per pair (free object, other world-child body) where
the other body is not part of the source robot; object-object contacts count for both
objects. Robot contacts (the source gripper squeezing the object, a held object pushed
against the robot) are excluded, exactly like the tier-P ``object_environment`` gate excludes
contacts with Reachy. :meth:`~ObjectEnvironmentDepth.result` drops the bodies that are
removed before validation (inactive), so the reference covers the validation scene.
"""
from __future__ import annotations

import numpy as np

METHOD = ("max over the recorded source states (state set, mj_forward, never stepped) of the "
          "penetration -contact.dist of contacts between a geom of a free object of the validation "
          "scene and any geom outside the source robot and the inactive bodies (object-object included)")


class ObjectEnvironmentDepth:
    """Accumulates the source reference depth over recorded states.

    ``objects``: {name: world-child body id} of the free objects (their whole subtree counts);
    ``robot``: (nbody,) bool, the source robot. Call :meth:`update` once per recorded state.
    """

    def __init__(self, model, objects: dict[str, int], robot):
        import mujoco

        m = self.m = model
        self._name = lambda kind, i: mujoco.mj_id2name(m, kind, int(i)) or ""
        self._geom, self._body = mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_BODY
        self.objects = dict(objects)
        root_index = {int(b): i for i, b in enumerate(self.objects.values())}
        self.obj_of = np.array([root_index.get(int(m.body_rootid[b]), -1) for b in range(m.nbody)])
        self.robot = np.asarray(robot, bool)
        self.pairs: dict[tuple[int, int], dict] = {}   # (object index, other root body) -> deepest contact
        self.frames = 0

    def update(self, d, frame: int, time_s: float | None = None):
        self.frames += 1
        n = d.ncon
        if not n or not self.objects:
            return
        g = np.asarray(d.contact.geom[:n])
        b = self.m.geom_bodyid[g]
        depth = -np.asarray(d.contact.dist[:n])
        o = self.obj_of[b]
        keep = ~self.robot[b].any(axis=1) & (depth > 0)
        for col in (0, 1):
            for i in np.flatnonzero(keep & (o[:, col] >= 0)):
                key = (int(o[i, col]), int(self.m.body_rootid[b[i, 1 - col]]))
                if key[0] == self.obj_of[b[i, 1 - col]]:
                    continue  # within one object
                if depth[i] > self.pairs.get(key, {}).get("depth_m", 0.0):
                    self.pairs[key] = {"frame": int(frame), "time_s": None if time_s is None else float(time_s),
                                       "depth_m": float(depth[i]),
                                       "geoms": [self._name(self._geom, x) or f"geom{int(x)}" for x in g[i]],
                                       "bodies": [self._name(self._body, x) for x in b[i]]}

    def result(self, inactive=()) -> dict:
        """``SceneRef.reference`` entries for the scene without the ``inactive`` object names
        (empty without objects)."""
        names = list(self.objects)
        drop_idx = {names.index(n) for n in inactive if n in names}
        drop_roots = {int(self.objects[n]) for n in inactive if n in self.objects}
        live = [k for k in range(len(names)) if k not in drop_idx]
        if not live:
            return {}
        per, worst = {names[k]: 0.0 for k in live}, None
        for (k, other), ev in self.pairs.items():
            if k in drop_idx or other in drop_roots:
                continue
            if ev["depth_m"] > per[names[k]]:
                per[names[k]] = ev["depth_m"]
            if worst is None or ev["depth_m"] > worst["depth_m"]:
                worst = {"object": names[k], **ev}
        return {"object_environment_depth_m": max(per.values()), "method": METHOD, "frames": self.frames,
                "per_object": per, "worst_contact": worst}


__all__ = ["METHOD", "ObjectEnvironmentDepth"]
