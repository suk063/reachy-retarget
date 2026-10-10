"""The episode ``scene`` section: components, mesh parts and per-frame poses (see docs/schema.md).

:func:`build_scene_components` turns a retargeted episode, its source episode (with the source
MuJoCo scene) and, when tier P ran, the physics rollout into a
:class:`~reachy_retarget.schema.episode.SceneComponents`, writing every mesh and texture it needs
to the asset library. It raises :class:`NoMeshes` when the source has no scene whose meshes can
be resolved, and :class:`NoObjectPoses` when a tracked object, articulation or scene component lacks
its pose at some step (:func:`missing_object_poses`): such episodes are not written (build records
say ``excluded: no_meshes`` / ``excluded: no_object_poses``).

Components (selection rule, ``SCENE_RULES``):

* every component of a tracked object (``ep.objects``, matched through ``geometry["body"]`` or the
  object id) and every articulated part whose joints belong to a tracked articulation
  (``ep.articulations``): always;
* every other rigid group of the scene (static scene parts, untracked free bodies such as
  distractors, untracked articulated parts): when the world AABB of one of its geoms (initial
  state) lies within ``radius`` (default 2.5 m) of the robot path, i.e. Reachy's base positions
  (on the floor), TCP and head positions and the tracked object positions over the episode;
* geoms of the world body itself (floors, walls) form the component ``worldbody`` and are kept
  per geom by the same distance rule;
* every Reachy link with geometry (``reachy/<link>``).

Poses on the episode clock (``SceneComponents.poses``):

* tracked objects: the retargeted (resampled) source tracks ``ep.objects[id].pose`` (``object_track``);
* untracked moving groups: forward kinematics of the source scene joint positions
  (``SourceEpisode.scene_qpos``, resampled onto the episode clock with the episode's
  ``source_time``: linear for hinge/slide, slerp for free joints) (``scene_state``); without
  ``scene_qpos`` the free joints of tracked objects and the tracked articulations drive the
  kinematics and the remaining joints keep the source initial state;
* static groups: constant (``static``); Reachy links: FK of ``q`` (``robot_fk``).

``physics_poses``: the same components on the tier-P clock, by forward kinematics of the rollout
``qpos`` (objects as simulated) and Reachy FK of the rollout's joint positions.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..retarget import timing
from ..robot.reachy import JOINTS
from ..schema.episode import SceneComponents
from ..schema.scene_assets import SCENE_SCHEMA
from ..schema.source import ObjectTrack
from . import reachy_links

RADIUS_M = 2.5
MAX_PATH_POINTS = 600
SCENE_RULES = {
    "components": "tracked objects and tracked articulated parts always; other rigid groups (static scene parts, "
                  "untracked free bodies, untracked articulated parts) and world-body geoms when a geom's world "
                  "AABB (initial state) is within radius_m of the robot path (Reachy base on the floor, TCP and head "
                  "positions, tracked object positions over the episode); all Reachy links",
    "rigid_group": "a body with a joint (or a world child) plus its descendants without joints",
    "visual": "visible (effective alpha > 0) geoms that do not collide or lie in one of the scene's visual render "
              "groups (groups of its visible non-colliding geoms); a component without any uses its visible "
              "colliding geoms (visual_from_collision)",
    "collision": "colliding geoms (contype or conaffinity non-zero); MuJoCo collides with mesh convex hulls",
    "meshes": "compiled MuJoCo meshes (scale, refpos/refquat and re-centring applied; vertices in the mesh frame, "
              "part pose = compiled geom pose in the component frame)",
    "textures": "recorded texture files (bytes unchanged) and builtin texture parameters; never rendered images",
}


class NoMeshes(Exception):
    """The episode's scene meshes cannot be obtained (it is excluded, not written)."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"no_meshes: {reason}")


class NoObjectPoses(Exception):
    """A tracked object or articulation lacks its state at some step (the episode is excluded, not written)."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"no_object_poses: {reason}")


def missing_object_poses(objects, articulations=None) -> str | None:
    """Why the per-step object state is incomplete, or ``None`` when it is complete: every tracked object
    needs a valid, finite pose and every articulation finite joint positions at every step."""
    gaps = []
    for oid, o in sorted(objects.items()):
        ok = np.asarray(o.valid, bool) & np.isfinite(o.pose).all(1)
        if not ok.all():
            gaps.append(f"object {oid}: pose missing at {int((~ok).sum())} of {len(ok)} steps")
    for k, a in sorted((articulations or {}).items()):
        bad = ~np.isfinite(np.asarray(a.qpos, float)).all(1)
        if bad.any():
            gaps.append(f"articulation {k}: joint positions missing at {int(bad.sum())} of {len(bad)} steps")
    if not gaps:
        return None
    return "; ".join(gaps[:5]) + (f" (+{len(gaps) - 5} more)" if len(gaps) > 5 else "")


def no_scene_reason(src) -> str:
    p = src.provenance or {}
    route = p.get("state_route")
    if p.get("missing_assets"):
        return f"source scene assets missing ({len(p['missing_assets'])} files, state_route {route})"
    if isinstance(p.get("scene"), str):
        return f"no source scene ({p['scene']})"
    return f"no source scene (state_route {route})"


def source_clock(times, source_time) -> timing.Clock:
    """The episode clock as a :class:`timing.Clock` over source rows ``times``, from the episode's
    ``source_time`` (the source time of every output row)."""
    times = np.asarray(times, float)
    s = np.clip(np.asarray(source_time, float), times[0], times[-1])
    index = np.clip(np.searchsorted(times, s, side="right") - 1, 0, len(times) - 2)
    fraction = np.clip((s - times[index]) / (times[index + 1] - times[index]), 0.0, 1.0)
    return timing.Clock(np.arange(len(s)) * 0.02, s, index, fraction, np.ones(len(s)))


def resample_qpos(columns, qpos, clock) -> np.ndarray:
    """Scene qpos rows on the output clock: free-joint blocks (``<j>/x .. /qz``) by pose
    interpolation (slerp), every other column linearly."""
    qpos = np.asarray(qpos, float)
    out = timing.linear(clock, qpos)
    cols = list(columns)
    for i, c in enumerate(cols):
        if c.endswith("/x") and cols[i:i + 7] == [c[:-1] + k for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]:
            block = qpos[:, i:i + 7]
            track = timing.object_track(clock, ObjectTrack(block, np.isfinite(block).all(1)))
            out[:, i:i + 7] = track.pose
    return out


def path_points(ep) -> np.ndarray:
    """Robot path points (n, 3): base on the floor, TCPs, head, tracked object positions."""
    base = np.c_[ep.q[:, :2], np.zeros(ep.length)]
    pts = [base, *(T[:, :3, 3] for T in ep.tcp_world.values()), ep.head_world[:, :3, 3]]
    for o in ep.objects.values():
        v = o.valid & np.isfinite(o.pose).all(1)
        pts.append(o.pose[v, :3])
    pts = np.concatenate(pts)
    step = max(1, len(pts) // MAX_PATH_POINTS)
    return pts[::step]


def _rot_error(qa, qb) -> np.ndarray:
    d = np.abs(np.einsum("...i,...i->...", qa, qb))
    return 2 * np.arccos(np.clip(d, 0, 1))


def build_scene_components(ep, src, library, *, rollout=None, radius: float = RADIUS_M) -> tuple[SceneComponents, dict]:
    """``(SceneComponents, stats)`` of a retargeted episode (see the module docstring)."""
    from ..validate.scene import MissingSceneAssets
    from .mujoco_scene import SceneExtraction

    if src.scene is None:
        raise NoMeshes(no_scene_reason(src))
    reason = missing_object_poses(ep.objects, ep.articulations)
    if reason:
        raise NoObjectPoses(reason)
    new0 = (library.new_files, library.new_bytes)
    try:
        ext = SceneExtraction(src.scene)
    except MissingSceneAssets as e:
        raise NoMeshes(f"source scene collision assets missing: {e.missing[:5]}") from e
    if ext.missing:
        raise NoMeshes("source scene visual assets missing: "
                       + ", ".join(f"{r['tag']}:{r['file']}" for r in ext.missing[:5]))
    points = path_points(ep)
    body_of = {}
    for oid, o in ep.objects.items():
        body = o.geometry.get("body") or oid
        body_of.setdefault(body, oid)
    art_of = {j: k for k, a in ep.articulations.items() for j in a.joint_names}
    cache, comps, roots, omitted, notes = {}, [], [], [], {}
    for r in sorted(ext.groups):
        geoms = ext.geoms_of(r)
        if not len(geoms):
            continue
        body = ext.group_name(r)
        oid = body_of.get(body) if r else None
        kind = ext.kind[r]
        joints = ext.group_joints(r) if r else []
        arts = sorted({art_of[j] for j in joints if j in art_of})
        tracked = oid is not None or bool(arts)
        if r == 0:
            keep = [g for g in geoms if ext.distance([g], points) <= radius]
            omitted += [{"component": body, "geom": ext.m.geom(int(g)).name, "rule": "distance"}
                        for g in geoms if g not in keep]
            geoms = np.array(keep, int)
            if not len(geoms):
                continue
        elif not tracked:
            dist = ext.distance(geoms, points)
            if dist > radius:
                omitted.append({"component": body, "distance_m": round(dist, 3), "rule": "distance"})
                continue
        visual, collision, part_notes = ext.parts(r, geoms, library, cache)
        if not visual and not collision:
            omitted.append({"component": body, "rule": "no visible or colliding geometry"})
            continue
        if part_notes.get("unsupported_geoms"):
            notes.setdefault("unsupported_geoms", {})[body] = part_notes["unsupported_geoms"]
        role = (ep.objects[oid].role if oid is not None else "manipulated" if kind == "free"
                else "articulated_part" if kind == "articulated" else "fixture")
        pose_source = "object_track" if oid is not None else "static" if kind == "static" else "scene_state"
        comps.append({"name": oid if oid is not None else body, "role": role, "source_body": body,
                      "bodies": [ext.body_names[b] for b in ext.groups[r]] if r else [],
                      "kind": kind, "task": tracked, "object_id": oid, "articulation": arts[0] if arts else None,
                      "joints": joints, "pose_source": pose_source,
                      "visual_from_collision": bool(part_notes.get("visual_from_collision")),
                      "visual": visual, "collision": collision})
        roots.append(r)
    if not any(c["visual"] for c in comps):
        raise NoMeshes("the source scene has no visual geometry near the robot path")
    names = [c["name"] for c in comps]
    for i, c in enumerate(comps):  # object ids may coincide with other bodies' names
        if names.count(c["name"]) > 1:
            c["name"] = f"{c['name']}#{c['source_body']}"
    T = ep.length

    # ---------------------------------------------------------------- poses on the episode clock
    if src.scene_qpos is not None:
        clock = source_clock(src.time, ep.source_time)
        columns = list(src.scene_qpos.joint_names)
        rows = resample_qpos(columns, src.scene_qpos.qpos, clock)
        state_source = "SourceEpisode.scene_qpos resampled onto the episode clock"
    else:
        columns, blocks = [], []
        for c, r in zip(comps, roots):
            o = ep.objects.get(c["object_id"]) if c["object_id"] else None
            if o is not None and c["kind"] == "free" and len(c["joints"]) == 1 and ext.m.body_parentid[r] == 0:
                columns += [f"{c['joints'][0]}/{k}" for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]
                blocks.append(np.where(o.valid[:, None], o.pose, np.nan))
        for a in ep.articulations.values():
            columns += list(a.joint_names)
            blocks.append(a.qpos)
        rows = np.concatenate(blocks, axis=1) if blocks else np.zeros((T, 0))
        state_source = "tracked object poses and articulations (no SourceEpisode.scene_qpos)"
    poses = ext.body_poses(columns, rows, roots)
    valid = np.ones((T, len(comps)), bool)
    checks = {}
    for i, c in enumerate(comps):
        o = ep.objects.get(c["object_id"]) if c["object_id"] else None
        if o is None:
            continue
        ok = o.valid & np.isfinite(o.pose).all(1)
        if c["kind"] != "static" and ok.any():
            dp = np.linalg.norm(poses[ok, i, :3] - o.pose[ok, :3], axis=1)
            dr = _rot_error(poses[ok, i, 3:], o.pose[ok, 3:])
            checks[c["name"]] = {"max_position_m": float(dp.max()), "max_rotation_rad": float(dr.max())}
        poses[:, i] = np.where(ok[:, None], o.pose, np.nan)
        valid[:, i] = ok
    bad = ~(valid & np.isfinite(poses).all(2))
    if bad.any():
        raise NoObjectPoses("; ".join(f"component {comps[i]['name']}: pose missing at {int(bad[:, i].sum())} of {T} "
                                      "steps" for i in np.flatnonzero(bad.any(0))[:5]))
    # Reachy links
    r_comps, r_mats = reachy_links.reachy_components(library)
    links = [c["source_body"] for c in r_comps]
    r_poses = reachy_links.link_poses(ep.q, links)

    # ---------------------------------------------------------------- physics clock
    physics_poses = None
    if rollout is not None:
        p_obj = ext.body_poses(rollout.qpos_names, rollout.qpos, roots)
        prefix = (rollout.info or {}).get("reachy_prefix", "reachy/")
        col = {n: i for i, n in enumerate(rollout.qpos_names)}
        missing = [n for n in JOINTS if f"{prefix}{n}" not in col]
        if missing:
            raise ValueError(f"physics rollout lacks Reachy joints {missing}")
        q_phys = rollout.qpos[:, [col[f"{prefix}{n}"] for n in JOINTS]]
        physics_poses = np.concatenate([p_obj, reachy_links.link_poses(q_phys, links)], axis=1)
        absent = sorted({j for c in comps for j in c["joints"]} - {n.split("/")[0] for n in rollout.qpos_names})
        if absent:
            notes["physics_joints_absent"] = absent  # kept at the source initial state

    # ---------------------------------------------------------------- materials and textures
    materials, textures = {}, {}
    for c in comps:
        for p in c["visual"] + c["collision"]:
            name = p.get("material")
            if name and name not in materials:
                materials[name] = ext.material(name)
    for m in materials.values():
        for t in [m.get("texture"), *(m.get("layers") or {}).values()]:
            if t and t not in textures:
                textures[t] = ext.texture(t, library)
    materials.update(r_mats)
    all_comps = comps + r_comps
    refs = {}
    for c in all_comps:
        for p in c["visual"] + c["collision"]:
            for k in ("mesh", "generated_mesh"):
                if p.get(k):
                    refs[p[k]["sha256"]] = p[k]["bytes"]
    for t in textures.values():
        for f in t["files"].values():
            refs[f["sha256"]] = f["bytes"]
    info = {"schema": SCENE_SCHEMA, "rules": {**SCENE_RULES, "radius_m": radius}, "state_source": state_source,
            "source": {"scene_mjcf_sha256": hashlib.sha256(src.scene.mjcf.encode()).hexdigest(),
                       "robot_prefixes": list(src.scene.robot_prefixes), "removed": ext.removed,
                       "mesh_inertia_shell": ext.shell},
            "omitted": omitted, "track_vs_scene_state": checks, "notes": notes,
            "reachy": reachy_links.provenance(),
            "physics_clock": "/physics/time" if physics_poses is not None else None}
    scene = SceneComponents(components=all_comps, poses=np.concatenate([poses, r_poses], axis=1),
                            valid=np.concatenate([valid, np.ones((T, len(r_comps)), bool)], axis=1),
                            materials=materials, textures=textures, physics_poses=physics_poses, info=info)
    stats = {"components": len(all_comps), "scene_components": len(comps), "omitted": len(omitted),
             "visual_parts": sum(len(c["visual"]) for c in all_comps),
             "collision_parts": sum(len(c["collision"]) for c in all_comps),
             "assets_referenced": len(refs), "assets_referenced_bytes": int(sum(refs.values())),
             "assets_new": library.new_files - new0[0], "assets_new_bytes": library.new_bytes - new0[1]}
    return scene, stats


__all__ = ["NoMeshes", "NoObjectPoses", "missing_object_poses", "build_scene_components", "SCENE_RULES", "RADIUS_M",
           "path_points", "source_clock", "resample_qpos", "no_scene_reason"]
