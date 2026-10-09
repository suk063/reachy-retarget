"""Lazy Viser replay of tracking episodes: both hands, head, gaze, base, grippers.

Port of reachy-control ``experiment/viewer.py`` for ``reachy-retarget-episode-v2`` files: the same
URDF meshes (:mod:`.visual`), floor grid, theme, colours (teal/blue: left/right hand targets,
orange/red: actual hands, purple: base, magenta: gaze), hand and head actual-vs-target frames, path
polylines, collision-mesh toggle, play/time/speed controls, fit camera and previous/next. Episodes
are listed from a build directory's records (``records/tracking/*.jsonl``) and filtered by cell,
regime, neck mode and tier-K result. ``--collision`` inspects the collision spheres (the sphere
model of :class:`reachy_retarget.robot.SelfCollision`) with joint sliders.

    python -m reachy_retarget.tracking.viewer --run runs/tracking/pilot [--port 8086]
    python -m reachy_retarget.tracking.viewer --collision [--port 8087]

Requires the ``viz`` extra (viser, trimesh, pycollada).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..schema.rotations import matrix_to_quat
from .visual import add_robot_visuals, prepare_episode

LEFT_GOAL, RIGHT_GOAL = (30, 165, 160), (35, 105, 210)
LEFT_ACTUAL, RIGHT_ACTUAL = (240, 150, 50), (210, 65, 60)
BASE_COLOR, GAZE_COLOR = (145, 80, 200), (190, 80, 195)


def load_catalog(run: Path):
    """[(uid, file, label dict)] of every written episode under ``run`` (final and earlier attempts)."""
    rows = []
    for p in sorted(run.glob("records/tracking/*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            sc = rec.get("scenario") or {}
            for k, a in enumerate(rec.get("attempts") or [{"file": rec.get("file"), "K": rec.get("K")}]):
                if not a.get("file"):
                    continue
                final = k == len(rec.get("attempts") or [None]) - 1
                rows.append({"file": a["file"], "cell": sc.get("cell") or rec.get("task") or "?",
                             "neck": sc.get("neck") or "source", "regime": rec.get("regime") or "?",
                             "K": bool((a.get("K") or {}).get("passed")), "final": final,
                             "name": Path(a["file"]).stem,
                             "source": rec.get("source_family") or "synthetic"})
    return rows


def _resolve(run: Path, file: str) -> Path:
    p = Path(file)
    return p if p.is_absolute() or p.exists() else run / "episodes" / p


def load_episode(path):
    from ..schema.io import read_episode
    ep = read_episode(path)
    links, positions, rotations = prepare_episode(ep)
    return ep, links, positions, rotations


def run_viewer(run, port=8086, initial=None):
    import viser
    from ..robot import Reachy
    from .neck import HEAD_TIP

    run = Path(run)
    catalog = load_catalog(run)
    if not catalog:
        raise ValueError(f"No written tracking episodes under {run}")
    by_name = {r["name"]: r for r in catalog}
    selected = initial if initial in by_name else next((r["name"] for r in catalog if r["final"]), catalog[0]["name"])
    ep, links, positions, rotations = load_episode(_resolve(run, by_name[selected]["file"]))
    robot = Reachy.load()
    server = viser.ViserServer(host="127.0.0.1", port=port, label=f"Reachy tracking · {run.name}")
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(control_layout="floating", control_width="medium", show_logo=False,
                               show_share_button=False, brand_color=(40, 100, 145))
    server.scene.add_grid("/floor", width=12, height=12, cell_size=.1, section_size=1)
    visuals = add_robot_visuals(server, links)
    collision_meshes = add_robot_visuals(server, links, "collision")
    actual_frames = [server.scene.add_frame(f"/hand/{s}/actual", axes_length=.075, axes_radius=.002)
                     for s in ("left", "right")]
    target_frames = [server.scene.add_frame(f"/hand/{s}/target", axes_length=.11, axes_radius=.003)
                     for s in ("left", "right")]
    head_actual = server.scene.add_frame("/head/actual", axes_length=.09, axes_radius=.003)
    head_target = server.scene.add_frame("/head/target", axes_length=.14, axes_radius=.003)
    base_target = server.scene.add_frame("/base/target", axes_length=.2, axes_radius=.004)
    title = server.gui.add_markdown("")
    cells = sorted({r["cell"] for r in catalog})
    cell = server.gui.add_dropdown("Cell", ("all", *cells), initial_value="all")
    regime = server.gui.add_dropdown("Regime", ("all", *sorted({r["regime"] for r in catalog})), initial_value="all")
    neck_mode = server.gui.add_dropdown("Neck", ("all", *sorted({r["neck"] for r in catalog})), initial_value="all")
    status = server.gui.add_dropdown("Tier K", ("all", "passed", "failed"), initial_value="all")
    attempts = server.gui.add_dropdown("Attempts", ("final", "all"), initial_value="final")
    names = [r["name"] for r in catalog if r["final"]]
    selector = server.gui.add_dropdown("Episode", tuple(names or [selected]), initial_value=selected)
    previous_button = server.gui.add_button("Previous")
    next_button = server.gui.add_button("Next")
    play = server.gui.add_checkbox("Play", False)
    slider = server.gui.add_slider("Time (s)", min=0., max=max(.01, ep.duration), step=.02, initial_value=0.)
    speed = server.gui.add_dropdown("Speed", ("0.25x", ".5x", "1x", "2x", "4x"), initial_value="1x")
    show_paths = server.gui.add_checkbox("Paths", True)
    show_targets = server.gui.add_checkbox("Targets", True)
    show_collision = server.gui.add_checkbox("Robot collision meshes", False)
    fit = server.gui.add_button("Fit camera")
    stats = server.gui.add_markdown("")
    server.gui.add_markdown("**Pure tracking**: no objects; the reference (targets) is tracked as given, kinematics "
                            "only (tier K).\n\nTeal/blue: left/right targets. Orange/red: actual hands. Purple: "
                            "actual base (target frame: nominal base). Magenta: gaze (head_tip +X).")
    path_handles, dynamic_handles = [], []
    pending = [None]
    state = {}

    def recompute():
        world = ep.tcp_world
        head_w = ep.head_world
        state.update(
            actual=np.stack([world[s][:, :3, 3] for s in ("left", "right")], 1),
            actual_R=np.stack([world[s][:, :3, :3] for s in ("left", "right")], 1),
            desired=np.stack([ep.reference.tcp[s][:, :3, 3] for s in ("left", "right")], 1),
            desired_R=np.stack([ep.reference.tcp[s][:, :3, :3] for s in ("left", "right")], 1),
            head_xyz=head_w[:, :3, 3], head_R=head_w[:, :3, :3],
            head_tip=(head_w @ HEAD_TIP)[:, :3, 3],
            head_ref=None if ep.reference.head is None else ep.reference.head[:, :3, :3],
            res_p=np.nan_to_num(ep.validation.get("tcp_pos_residual", np.zeros((ep.length, 2)))),
            res_r=np.nan_to_num(ep.validation.get("tcp_rot_residual", np.zeros((ep.length, 2)))),
            head_err=ep.validation.get("head_rot_residual"),
            clearance=ep.validation.get("self_clearance"))

    def camera(client):
        pts = np.concatenate((state["desired"].reshape(-1, 3), state["head_xyz"]))
        center = (pts.min(0) + pts.max(0)) / 2
        radius = max(.9, np.linalg.norm(pts.max(0) - pts.min(0)) * .6)
        client.camera.position = center + radius * np.array([2., -2.5, 1.4])
        client.camera.look_at = center
        client.camera.up_direction = (0, 0, 1)
    server.on_client_connect(camera)

    def filtered():
        return [r["name"] for r in catalog
                if (attempts.value == "all" or r["final"]) and cell.value in ("all", r["cell"])
                and regime.value in ("all", r["regime"]) and neck_mode.value in ("all", r["neck"])
                and (status.value == "all" or (status.value == "passed") == r["K"])]

    @cell.on_update
    @regime.on_update
    @neck_mode.on_update
    @status.on_update
    @attempts.on_update
    def change_filter(_):
        options = filtered()
        if not options:
            stats.content = "No episodes match these filters."
            return
        selector.options = tuple(options)
        if selector.value not in options:
            selector.value = options[0]
        pending[0] = selector.value

    @selector.on_update
    def select(_):
        pending[0] = selector.value

    @previous_button.on_click
    def previous(_):
        selector.value = selector.options[(selector.options.index(selector.value) - 1) % len(selector.options)]

    @next_button.on_click
    def next_case(_):
        selector.value = selector.options[(selector.options.index(selector.value) + 1) % len(selector.options)]

    @fit.on_click
    def fit_camera(_):
        for client in server.get_clients().values():
            camera(client)

    def segments(name, points, color, handles, thickness=.004):
        points = np.asarray(points)
        if len(points) >= 2:
            handles.append(server.scene.add_line_segments(name, np.stack((points[:-1], points[1:]), axis=1), color,
                                                          thickness=thickness))

    def setup_scene():
        for handle in path_handles:
            handle.remove()
        path_handles.clear()
        step = max(1, ep.length // 600)
        for i, (goal, actual) in enumerate(((LEFT_GOAL, LEFT_ACTUAL), (RIGHT_GOAL, RIGHT_ACTUAL))):
            segments(f"/paths/hand{i}/goal", state["desired"][::step, i], goal, path_handles)
            segments(f"/paths/hand{i}/actual", state["actual"][::step, i], actual, path_handles)
        segments("/paths/base", np.c_[ep.q[::step, :2], np.zeros(len(ep.q[::step]))], BASE_COLOR, path_handles)
        if ep.reference.base is not None:
            segments("/paths/base_target", np.c_[ep.reference.base[::step, :2], np.full(len(ep.q[::step]), .002)],
                     (120, 120, 130), path_handles, .002)
        tr = ep.extra.get("tracking", {})
        sc = tr.get("scenario") or {}
        src = tr.get("source") or {}
        k = ep.tier["K"]
        title.content = (f"## {ep.episode_id}\n\n{ep.task} · {ep.regime} · {' + '.join(ep.body_parts) or 'static'}"
                         + (f"\n\nneck {sc.get('neck')} · {sc.get('extent')} · {sc.get('speed')} · start "
                            f"{sc.get('start')}" if sc else "")
                         + (f"\n\nsource {src.get('uid')}" if src else "")
                         + f"\n\n**K {'PASS' if k['passed'] else 'FAIL'}** · {ep.length} states · {ep.duration:.1f} s")
        if not k["passed"]:
            title.content += "\n\n" + "\n\n".join(k["reasons"][:4])
        if ep.reference.head is None:
            title.content += "\n\nNeck not used · joints held · no head target"

    def update_dynamic(index):
        for handle in dynamic_handles:
            handle.remove()
        dynamic_handles.clear()
        segments("/gaze/ray", [state["head_tip"][index], state["head_tip"][index] + state["head_R"][index][:, 0]],
                 GAZE_COLOR, dynamic_handles, .003)
        if state["head_ref"] is not None:
            segments("/gaze/target_ray", [state["head_tip"][index],
                                          state["head_tip"][index] + state["head_ref"][index][:, 0]],
                     (230, 150, 230), dynamic_handles, .002)
        span = state["actual"][index]
        segments("/hands/span", span, (90, 140, 150), dynamic_handles, .003)

    recompute()
    setup_scene()
    elapsed, last_value, last_frame = 0., 0., -1
    previous_time = time.perf_counter()
    print(f"Reachy tracking viewer ready: http://127.0.0.1:{server.get_port()} ({len(catalog)} episodes)", flush=True)
    try:
        while True:
            if pending[0] is not None:
                selected = pending[0]
                pending[0] = None
                play.value = False
                ep, new_links, positions, rotations = load_episode(_resolve(run, by_name[selected]["file"]))
                if links != new_links:
                    raise ValueError("Model link list changed")
                recompute()
                slider.value = 0.
                slider.max = max(.01, ep.duration)
                elapsed = last_value = 0.
                last_frame = -1
                setup_scene()
                for client in server.get_clients().values():
                    camera(client)
            now = time.perf_counter()
            if slider.value != last_value:
                elapsed = slider.value
            elif play.value:
                elapsed += (now - previous_time) * float(speed.value.removesuffix("x"))
            previous_time = now
            end = ep.duration
            if elapsed > end:
                elapsed = 0.
            index = int(np.clip(np.searchsorted(ep.time - ep.time[0], elapsed, side="right") - 1, 0, ep.length - 1))
            with server.atomic():
                for handle in path_handles:
                    handle.visible = show_paths.value
                for handle in collision_meshes.values():
                    handle.visible = show_collision.value
                for handle in target_frames + [head_target, base_target]:
                    handle.visible = show_targets.value
                if index != last_frame:
                    for j, handle in list(visuals.items()) + list(collision_meshes.items()):
                        handle.position, handle.wxyz = positions[index, j], rotations[index, j]
                    for i in (0, 1):
                        actual_frames[i].position = state["actual"][index, i]
                        actual_frames[i].wxyz = matrix_to_quat(state["actual_R"][index, i])
                        target_frames[i].position = state["desired"][index, i]
                        target_frames[i].wxyz = matrix_to_quat(state["desired_R"][index, i])
                    head_actual.position = head_target.position = state["head_xyz"][index]
                    head_actual.wxyz = matrix_to_quat(state["head_R"][index])
                    head_target.visible = show_targets.value and state["head_ref"] is not None
                    if state["head_ref"] is not None:
                        head_target.wxyz = matrix_to_quat(state["head_ref"][index])
                    if ep.reference.base is not None:
                        b = ep.reference.base[index]
                        base_target.position = (b[0], b[1], 0.)
                        base_target.wxyz = matrix_to_quat(robot_planar(b[2]))
                    update_dynamic(index)
                    rp, rr = state["res_p"][index], state["res_r"][index]
                    stats.content = (f"**{ep.time[index] - ep.time[0]:.2f}/{end:.2f} s**\n\n"
                                     + "\n\n".join(f"{n}: {rp[i] * 1000:.2f} mm / {np.rad2deg(rr[i]):.2f}°"
                                                   for i, n in enumerate(("Left", "Right"))))
                    if state["head_err"] is not None:
                        stats.content += f"\n\nHead: {np.rad2deg(state['head_err'][index]):.2f}°"
                    if state["clearance"] is not None:
                        stats.content += f"\n\nSelf clearance: {state['clearance'][index] * 1000:.1f} mm"
                    stats.content += ("\n\nGripper opening: " + " / ".join(f"{v:.2f}" for v in ep.gripper_opening[index])
                                      + f"\n\nBase: {ep.q[index, 0]:.2f}, {ep.q[index, 1]:.2f} m, "
                                        f"{np.rad2deg(ep.q[index, 2]):.0f}°")
                    slider.value = float(ep.time[index] - ep.time[0])
                    last_frame = index
                last_value = slider.value
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    _ = robot


def robot_planar(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def run_collision_viewer(port=8087):
    """Collision spheres with joint sliders (reachy-control ``viewer.py --collision``)."""
    import trimesh
    import viser
    from ..robot import LOWER, UPPER, SelfCollision
    from ..robot.reachy import JOINTS
    from .config import TrackingConfig
    from .visual import link_poses

    sc = SelfCollision.load()
    margin = TrackingConfig().min_self_clearance
    q0 = np.zeros(22)
    links, _, _ = link_poses(q0[None])
    server = viser.ViserServer(host="127.0.0.1", port=port, label="Reachy · collision spheres")
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(control_layout="floating", control_width="medium", show_logo=False,
                               show_share_button=False, brand_color=(28, 130, 160))
    server.scene.add_grid("/floor", width=4, height=4, cell_size=.1, section_size=1)
    visuals = add_robot_visuals(server, links)
    sphere_mesh = trimesh.creation.icosphere(subdivisions=2)
    names = [sc.links[i] for i in sc.link_index]
    colors = np.array([(35, 195, 162) if n.startswith("l_") else (70, 145, 240) if n.startswith("r_") else
                       (239, 172, 62) for n in names], dtype=np.uint8)
    spheres = server.scene.add_batched_meshes_simple(
        "/spheres", sphere_mesh.vertices, sphere_mesh.faces, batched_wxyzs=np.tile([1., 0., 0., 0.], (len(names), 1)),
        batched_positions=np.zeros((len(names), 3)), batched_scales=sc.radii, batched_colors=colors, opacity=.5,
        lod="off", cast_shadow=False, receive_shadow=False)
    server.gui.add_markdown("## Reachy · collision spheres\n\nMove the joints to inspect the fitted envelope.")
    show_visual = server.gui.add_checkbox("Robot visual mesh", True)
    show_spheres = server.gui.add_checkbox("Collision spheres", True)
    opacity = server.gui.add_slider("Sphere opacity", min=.05, max=1., step=.05, initial_value=.5)
    reset = server.gui.add_button("Reset pose")
    sliders = {}
    for title, cols in (("Left arm", range(3, 10)), ("Right arm", range(10, 17)), ("Neck", range(17, 20)),
                        ("Grippers", range(20, 22))):
        with server.gui.add_folder(title + " (degrees)", expand_by_default=False):
            for c in cols:
                sliders[c] = server.gui.add_slider(JOINTS[c], min=float(np.rad2deg(LOWER[c])),
                                                   max=float(np.rad2deg(UPPER[c])), step=.1, initial_value=0.)
    status = server.gui.add_markdown("")
    server.gui.add_markdown(f"Teal/blue: arms · amber: body · yellow: below {margin * 1e3:.0f} mm (the tracking "
                            "tier-K gate) · red: intersection.")

    def camera(client):
        client.camera.position = (2.8, -3.8, 2.3)
        client.camera.look_at = (0., 0., .9)
        client.camera.up_direction = (0., 0., 1.)
    server.on_client_connect(camera)

    @reset.on_click
    def reset_pose(_):
        for s in sliders.values():
            s.value = 0.

    last = None
    try:
        while True:
            key = (tuple(s.value for s in sliders.values()), show_visual.value, show_spheres.value, opacity.value)
            if key != last:
                q = q0.copy()
                for c, s in sliders.items():
                    q[c] = np.deg2rad(s.value)
                centers = sc.sphere_centers(q)[0]
                gap = sc.distances(q)
                current = colors.copy()
                for threshold, color in ((margin, (250, 225, 45)), (0., (240, 65, 65))):
                    ids = np.unique(np.r_[sc.a[gap < threshold], sc.b[gap < threshold]])
                    current[ids] = color
                _, pos, rot = link_poses(q[None])
                with server.atomic():
                    spheres.batched_positions = centers
                    spheres.batched_colors = current
                    spheres.opacity = opacity.value
                    spheres.visible = show_spheres.value
                    for idx, handle in visuals.items():
                        handle.position, handle.wxyz = pos[0, idx], rot[0, idx]
                        handle.visible = show_visual.value
                    status.content = (f"**{len(sc.radii)} spheres**\n\nMinimum checked gap: **{gap.min() * 1000:.1f} mm**"
                                      f"\n\nIntersecting pairs: **{np.count_nonzero(gap < 0)}** · below "
                                      f"{margin * 1e3:.0f} mm: **{np.count_nonzero(gap < margin)}**")
                last = key
            time.sleep(.03)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", type=Path)
    p.add_argument("--collision", action="store_true", help="inspect collision spheres and joint poses")
    p.add_argument("--port", type=int)
    p.add_argument("--episode", help="initial episode (file stem)")
    a = p.parse_args(argv)
    if a.collision:
        run_collision_viewer(a.port or 8087)
    elif a.run:
        run_viewer(a.run, a.port or 8086, a.episode)
    else:
        p.error("give --run RUN or --collision")


if __name__ == "__main__":
    main()
