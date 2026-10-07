"""Replay BiGym lightweight demonstrations in the original BiGym environment.

BiGym demonstrations (release ``v0.9.0``) are *lightweight* recordings: 500 Hz
actions, a reset seed and environment metadata, but no states. States only exist
after replaying the actions in the BiGym environment. BiGym pins ``mujoco==3.1.5``,
``numpy==1.26.*``, dm_control 1.0.19, a Gymnasium fork and Mojo 0.1.1, which conflict
with this project's environment (numpy 2, newer MuJoCo). This file is therefore a
**standalone script** run by a separate, isolated interpreter that has BiGym installed
(see ``docs/sources.md``, section BiGym). It never imports ``reachy_retarget`` and
nothing here uses the network.

Per demonstration it writes one ``<out>/<task>/<uuid>.npz`` *replay record*:

* ``qpos``/``qvel`` of the reset state and of post-action states every ``stride``
  control steps (plus the last step), ``time`` (actual ``mjData.time``),
  ``step`` (post-action step index, ``-1`` = reset state) and ``success`` (the env's
  success check after that step);
* ``mjcf``: the complete scene MJCF **as it is after the seeded reset** (BiGym's reset
  moves furniture and props by editing compiled ``body_pos``/``body_quat``; these are
  written back into the XML and the recompiled model is checked against the live one);
* ``meta``: JSON with source identity (zip member, its SHA-256, seed, recorded package
  versions), runtime versions, the success summary (any step / final step / first
  step, the rule of ``DemoPlayer.validate_in_env``), recorded termination flags, the
  model-export check and any replay error. Failed replays are saved too.

Mesh and texture files are written once into ``<out>/assets/`` (names carry a content
hash, given by dm_control).

Replay follows ``demonstrations.demo_player.DemoPlayer.validate_in_env``: official
``Metadata``/``LightweightDemo`` loading, ``Metadata.get_env(500)``,
``env.reset(seed=demo.seed)``, then ``env.step(action, fast=True)`` for every recorded
action in order, without decimation or clipping, reading ``env.success`` after every
step. Unlike ``validate_in_env`` it does not stop at the first success.

Usage (isolated interpreter)::

    <bigym-python> reachy_retarget/sources/bigym_replay.py --zip demonstrations.zip \
        --out data/derived/bigym/replay-v1 [--tasks MovePlate,...] [--limit N] \
        [--stride 10] [--jobs 4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

# Run as a script, this file's folder is sys.path[0]; it holds this project's
# ``bigym.py`` adapter, which would shadow the BiGym package.
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) != _HERE]

RECORD_VERSION = "bigym-replay-record-v1"
CONTROL_FREQUENCY = 500
ORIENTATION_ATTRS = ("euler", "axisangle", "xyaxes", "zaxis")
QPOS0_DERIVED = {"qpos0", "qpos_spring", "body_invweight0", "dof_invweight0", "tendon_invweight0",
                 "cam_pos0", "cam_poscom0", "cam_mat0", "light_pos0", "light_poscom0", "light_dir0",
                 "body_sameframe", "actuator_length0", "actuator_acc0", "tendon_length0"}
# Texture pixels (BiGym randomizes some colors at reset) do not affect states or physics.
BODY_DERIVED = {"body_contype", "body_conaffinity", "body_margin"}
VISUAL_ONLY = {"tex_rgb", "tex_data", "mat_rgba", "geom_rgba"}


def member_versions(zip_path):
    """``(task, uuid) -> [zip members]``: every stored version of each recording."""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".safetensors") and "/lightweight/" in n]
    out = {}
    for n in sorted(names):
        parts = n.split("/")
        out.setdefault((parts[0], parts[-1].split(".")[0]), []).append(n)
    return out


def list_members(zip_path, tasks=None, limit=None):
    """Lightweight demo members, one per recording UUID.

    The release stores most recordings twice, once per action mode (``absolute`` and
    ``delta``, converted from the same recording, same UUID). The absolute version is
    replayed; a delta file is used only when no absolute file of that UUID exists.
    """
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".safetensors") and "/lightweight/" in n]
    by_uuid = {}
    for n in sorted(names):
        task, mode, _, fname = n.split("/")
        if tasks and task not in tasks:
            continue
        uuid = fname.split(".")[0]
        absolute = mode.endswith("_absolute")
        prev = by_uuid.get((task, uuid))
        if prev is None or (absolute and not prev[1]):
            by_uuid[(task, uuid)] = (n, absolute)
    out, per_task = [], {}
    for (task, _), (n, _) in sorted(by_uuid.items()):
        per_task[task] = per_task.get(task, 0) + 1
        if limit is None or per_task[task] <= limit:
            out.append(n)
    return out


def _versions():
    import importlib.metadata as md
    out = {}
    for p in ("bigym", "mujoco", "numpy", "dm-control", "gymnasium", "mojo", "safetensors"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    try:  # where the replaying BiGym was installed from (PEP 610), e.g. the pinned archive
        out["bigym_direct_url"] = json.loads(md.distribution("bigym").read_text("direct_url.json") or "null")
    except Exception:
        out["bigym_direct_url"] = None
    return out


def _elements(root):
    """Compiled body name -> MJCF element (bodies and attachment frames)."""
    els = {el.full_identifier: el for el in root.find_all("body")}
    for el in root.find_all("attachment_frame"):
        els[el.full_identifier] = el
    return els


def export_model(env):
    """Seeded-reset scene as ``(xml, assets, check)``.

    Body placements changed by the reset (compiled ``body_pos``/``body_quat``) are
    written back into the MJCF. ``check`` compares every numeric array of the live
    model with the recompiled one and the forward kinematics at the current state.
    """
    import mujoco
    import numpy as np

    m, d = env.mojo.model, env.mojo.data
    root = env.mojo.root_element.mjcf
    els = _elements(root)
    patched = []
    for b in range(1, m.nbody):
        name = m.body(b).name
        el = els.get(name)
        if el is None:
            continue
        pos, quat = np.array(m.body_pos[b]), np.array(m.body_quat[b])
        cur_pos = np.zeros(3) if el.pos is None else np.asarray(el.pos, float)
        if np.array_equal(cur_pos, pos) and el.quat is not None and np.array_equal(np.asarray(el.quat, float), quat):
            continue
        el.pos = pos.tolist()
        for attr in ORIENTATION_ATTRS:
            if getattr(el, attr, None) is not None:
                setattr(el, attr, None)
        el.quat = quat.tolist()
        patched.append(name)
    xml = root.to_xml_string()
    assets = root.get_assets()
    m2 = mujoco.MjModel.from_xml_string(xml, assets)
    # Runtime edits through physics bindings (e.g. Prop.disable(): geoms without
    # collisions, free joints with huge damping) are not in the MJCF either.
    xml, written = _write_back_bindings(m, m2, xml)
    if written:
        m2 = mujoco.MjModel.from_xml_string(xml, assets)
    diffs = {}
    for k in dir(m):
        a = getattr(m, k, None)
        if k.startswith("_") or not isinstance(a, np.ndarray) or a.dtype.kind not in "fiu":
            continue
        b = getattr(m2, k, None)
        if not isinstance(b, np.ndarray) or a.shape != b.shape:
            diffs[k] = "shape"
            continue
        if a.size:
            dv = float(np.abs(a.astype(float) - b.astype(float)).max())
            if dv > 1e-9:
                diffs[k] = dv
    # Fields derived from qpos0 (the reset pose becomes the free bodies' qpos0) and the
    # inertial frame of massless bodies do not affect kinematics or dynamics.
    massless = m.body_mass == 0
    for k in ("body_ipos", "body_iquat"):
        if k in diffs and float(np.abs(getattr(m, k) - getattr(m2, k))[~massless].max(initial=0)) <= 1e-9:
            diffs.pop(k)
    ignorable = {k: diffs.pop(k) for k in list(diffs) if k in QPOS0_DERIVED}
    # Compiler bookkeeping derived from geometry and collision flags (bounding-volume
    # hierarchies, convex-hull graphs, per-body collision summaries). The live model keeps
    # values from before runtime edits; the recompiled one is consistent with them.
    compiled = {k: diffs.pop(k) for k in list(diffs)
                if k.startswith(("bvh_", "body_bvh", "mesh_bvh", "mesh_graph")) or k in BODY_DERIVED}
    visual = {k: diffs.pop(k) for k in list(diffs) if k in VISUAL_ONLY}
    d2 = mujoco.MjData(m2)
    d2.qpos[:] = d.qpos
    mujoco.mj_kinematics(m2, d2)
    d1 = mujoco.MjData(m)
    d1.qpos[:] = d.qpos
    mujoco.mj_kinematics(m, d1)
    check = {"patched_bodies": len(patched), "written_back": written, "model_array_diffs": diffs,
             "qpos0_derived_diffs": ignorable, "compiler_derived_diffs": compiled,
             "visual_only_diffs": visual,
             "max_body_xpos_diff_m": float(np.abs(d1.xpos - d2.xpos).max()),
             "max_body_xquat_diff": float(np.abs(d1.xquat - d2.xquat).max()),
             "nq": int(m.nq), "nv": int(m.nv), "nbody": int(m.nbody)}
    check["ok"] = (not diffs and check["max_body_xpos_diff_m"] < 1e-9
                   and check["max_body_xquat_diff"] < 1e-9)
    return xml, assets, check


def _write_back_bindings(m, m2, xml_text: str):
    """Copy geom contype/conaffinity and joint damping of the live model ``m`` into
    ``xml_text`` where the compiled ``m2`` differs. A ``<freejoint>`` that needs damping
    becomes the equivalent ``<joint type="free">``. Returns ``(xml, written)``."""
    import xml.etree.ElementTree as ET

    import numpy as np

    root = ET.fromstring(xml_text)
    bodies = {el.get("name"): el for el in root.iter("body") if el.get("name")}
    written = {"geoms": 0, "joints": 0}

    def geom_el(g):
        name = m.geom(g).name
        if name:
            return next((el for el in root.iter("geom") if el.get("name") == name), None)
        b = int(m.geom_bodyid[g])
        el = bodies.get(m.body(b).name)
        if el is None:
            return None
        local = [i for i in range(m.ngeom) if m.geom_bodyid[i] == b].index(g)
        geoms = el.findall("geom")
        return geoms[local] if local < len(geoms) else None

    for g in np.flatnonzero((m.geom_contype != m2.geom_contype) | (m.geom_conaffinity != m2.geom_conaffinity)):
        el = geom_el(int(g))
        if el is None:
            raise ValueError(f"cannot map geom {g} to the MJCF")
        el.set("contype", str(int(m.geom_contype[g])))
        el.set("conaffinity", str(int(m.geom_conaffinity[g])))
        written["geoms"] += 1
    for j in range(m.njnt):
        adr = m.jnt_dofadr[j]
        if m.dof_damping[adr] == m2.dof_damping[adr]:
            continue
        name = m.joint(j).name
        el = next((e for e in root.iter() if e.tag in ("joint", "freejoint") and e.get("name") == name), None)
        if el is None:
            raise ValueError(f"cannot map joint {name!r} to the MJCF")
        if el.tag == "freejoint":
            el.tag = "joint"
            el.set("type", "free")
        el.set("damping", repr(float(m.dof_damping[adr])))
        written["joints"] += 1
    if not any(written.values()):
        return xml_text, {}
    return ET.tostring(root, encoding="unicode"), written


def _write_assets(assets, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, data in assets.items():
        if isinstance(data, str):
            data = data.encode()
        digest = hashlib.sha256(data).hexdigest()
        p = folder / name
        if p.exists():
            if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
                raise ValueError(f"asset name collision with different content: {name}")
        else:
            tmp = p.with_name(f".{name}.tmp-{os.getpid()}")
            tmp.write_bytes(data)
            os.replace(tmp, p)
        out[name] = digest
    return out


def replay_member(zip_path, member, out_root, stride=10, zip_sha256=None, versions=None, clip=False):
    """Replay one zip member and write its record. Returns the summary dict."""
    import numpy as np

    logging.disable(logging.WARNING)
    from bigym.bigym_env import CONTROL_FREQUENCY_MAX
    from demonstrations.demo import LightweightDemo
    from demonstrations.utils import Metadata

    task, mode_dir, _, fname = member.split("/")
    uuid = fname.split(".")[0]
    out = Path(out_root) / task / f"{uuid}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read(member)
    meta = {"record_version": RECORD_VERSION, "task": task, "uuid": uuid, "zip_member": member,
            "action_mode_dir": mode_dir, "member_sha256": hashlib.sha256(raw).hexdigest(),
            "member_bytes": len(raw), "zip_sha256": zip_sha256, "runtime_versions": _versions(),
            "python": sys.version.split()[0], "control_frequency_hz": CONTROL_FREQUENCY,
            "stride": stride, "actions_clipped": bool(clip), "zip_versions_of_recording": versions or [member], "replay_rule": "Metadata.get_env(500); env.reset(seed); "
            "env.step(action, fast=True) for every recorded action; env.success after each step",
            "success_rule": "DemoPlayer.validate_in_env: success if the env success check (reward > 0) "
            "holds after any replayed step", "error": None}
    arrays = {}
    rows = {"time": [], "step": [], "qpos": [], "qvel": [], "success": []}
    env = None
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / fname
            p.write_bytes(raw)
            md = Metadata.from_safetensors(p)
            demo = LightweightDemo.from_safetensors(p, md)
        env_data = md.environment_data
        meta.update(seed=int(demo.seed), env_name=md.env_name, recorded_package_versions=md.package_versions,
                    recorded_date=md.date, action_mode=env_data.action_mode_name,
                    action_mode_absolute=env_data.action_mode_absolute,
                    floating_base=env_data.floating_base, floating_dofs=list(env_data.floating_dofs),
                    robot=env_data.robot_name)
        if clip:
            # Official DemoConverter.clip_actions (BiGym's demo converter tool): clip to the
            # action space, carrying the clipped overhead into later actions. A labeled
            # variant, never the default replay.
            from demonstrations.demo_converter import DemoConverter
            orig = np.stack([s.executed_action for s in demo.timesteps])
            demo = DemoConverter.clip_actions(demo)
            diff = np.abs(np.stack([s.executed_action for s in demo.timesteps]) - orig)
            meta["clip"] = {"rule": "demonstrations.demo_converter.DemoConverter.clip_actions",
                            "changed_actions": int((diff.max(1) > 0).sum()),
                            "max_abs_change": float(diff.max()) if diff.size else 0.0,
                            "max_abs_change_per_component": diff.max(0).round(6).tolist()}
        steps = demo.timesteps
        term = [bool(s.termination) for s in steps]
        trunc = [bool(s.truncation) for s in steps]
        meta.update(n_actions=len(steps), recorded_termination_count=sum(term),
                    recorded_first_termination_step=next((i for i, t in enumerate(term) if t), None),
                    recorded_truncation_count=sum(trunc))
        env = md.get_env(CONTROL_FREQUENCY_MAX)
        doc = (type(env).__doc__ or "").strip().splitlines()
        meta["env_doc"] = doc[0].strip() if doc else None
        meta["env_class"] = f"{type(env).__module__}.{type(env).__qualname__}"
        env.reset(seed=demo.seed)
        m, d = env.mojo.model, env.mojo.data
        meta["timestep_s"] = float(m.opt.timestep)
        meta["sub_steps"] = int(env._sub_steps_count)
        xml, assets, check = export_model(env)
        meta["model_export"] = check
        meta["assets"] = _write_assets(assets, Path(out_root) / "assets")
        arrays["mjcf"] = np.frombuffer(xml.encode(), dtype=np.uint8)
        meta["mjcf_sha256"] = hashlib.sha256(xml.encode()).hexdigest()
        meta["robot"] = {"pelvis": env.robot.pelvis.mjcf.full_identifier,
                         "grippers": {side.name.lower(): {
                             "body": g.body.mjcf.full_identifier,
                             "wrist_site": g.wrist_site.mjcf.full_identifier,
                             "pinch_site": None if g._pinch_site is None else g._pinch_site.mjcf.full_identifier,
                             "actuators": [a.full_identifier for a in g.actuators],
                             "range": [float(v) for v in g.range]}
                             for side, g in env.robot.grippers.items()}}

        def keep(step, ok):
            rows["time"].append(float(d.time))
            rows["step"].append(step)
            rows["qpos"].append(d.qpos.copy())
            rows["qvel"].append(d.qvel.copy())
            rows["success"].append(ok)

        keep(-1, False)
        first, n_success, ok = None, 0, False
        n = len(steps)
        meta["replayed_actions"] = 0
        for i, st in enumerate(steps):
            env.step(st.executed_action, fast=True)
            meta["replayed_actions"] = i + 1
            ok = bool(env.success)
            if ok:
                n_success += 1
                if first is None:
                    first = i
            if (i + 1) % stride == 0 or i == n - 1:
                keep(i, ok)
        meta.update(replayed_actions=n, success_any=first is not None, success_final=ok,
                    first_success_step=first, success_step_count=n_success,
                    first_success_time_s=None if first is None else (first + 1) / CONTROL_FREQUENCY)
    except Exception as exc:  # saved, never hidden
        meta["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000],
                         "traceback": traceback.format_exc()[-4000:]}
        meta.setdefault("replayed_actions", 0)
        meta.setdefault("success_any", None)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    meta["replay_seconds"] = round(time.time() - t0, 3)
    if rows["time"]:
        arrays.update(time=np.asarray(rows["time"]), step=np.asarray(rows["step"], np.int64),
                      qpos=np.stack(rows["qpos"]), qvel=np.stack(rows["qvel"]),
                      success=np.asarray(rows["success"], bool))
    arrays["meta"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out)
    summary = {k: meta.get(k) for k in ("task", "uuid", "zip_member", "member_sha256", "seed", "n_actions",
                                          "replayed_actions", "success_any", "success_final",
                                          "first_success_step", "recorded_first_termination_step",
                                          "actions_clipped", "replay_seconds")}
    summary["error"] = None if meta["error"] is None else f"{meta['error']['type']}: {meta['error']['message'][:300]}"
    summary["model_export_ok"] = meta.get("model_export", {}).get("ok")
    summary["record"] = str(out)
    return summary


def _worker(args):
    return replay_member(*args)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tasks", default=None, help="comma-separated task names (default: all)")
    ap.add_argument("--members", default=None, help="comma-separated zip members (overrides --tasks)")
    ap.add_argument("--limit", type=int, default=None, help="max recordings per task")
    ap.add_argument("--stride", type=int, default=10, help="keep every n-th post-action state (500 Hz)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--clip-actions", action="store_true",
                    help="labeled variant: apply BiGym's DemoConverter.clip_actions before replay")
    a = ap.parse_args(argv)
    zip_sha = hashlib.sha256(a.zip.read_bytes()).hexdigest()
    if a.members:
        members = a.members.split(",")
    else:
        members = list_members(a.zip, set(a.tasks.split(",")) if a.tasks else None, a.limit)
    if a.skip_existing:
        members = [mb for mb in members
                   if not (a.out / mb.split("/")[0] / (mb.split("/")[-1].split(".")[0] + ".npz")).exists()]
    a.out.mkdir(parents=True, exist_ok=True)
    versions = member_versions(a.zip)
    jobs = [(str(a.zip), mb, str(a.out), a.stride, zip_sha,
             versions.get((mb.split("/")[0], mb.split("/")[-1].split(".")[0]), [mb]), a.clip_actions)
            for mb in members]
    log = a.out / "summary.jsonl"
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.jobs) as pool, log.open("a") as f:
            for s in pool.imap_unordered(_worker, jobs):
                f.write(json.dumps(s) + "\n")
                f.flush()
                print(json.dumps(s), flush=True)
    else:
        with log.open("a") as f:
            for j in jobs:
                s = _worker(j)
                f.write(json.dumps(s) + "\n")
                f.flush()
                print(json.dumps(s), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
