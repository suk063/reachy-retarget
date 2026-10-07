"""Local mjviser playback of retargeted states; no physics stepping or hardware."""

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import h5py
import mujoco
import numpy as np
import viser
from mjviser import ViserMujocoScene

from .physics import BASE
from .store import json_write, sha256
from .robot import CONTROL
from .viewer_assets import RobotVisuals, prepare_scene, visual_urdf


def prepare_generic_physical_scene(motion, report):
    """Use the recorded scene and its object assets without rebuilding physics."""
    scene = motion.with_name("scene.xml")
    if sha256(scene) != report["scene_sha256"]:
        raise ValueError("Recorded physical scene checksum mismatch")
    folder = motion.parent / "viewer"
    folder.mkdir(exist_ok=True)
    urdf, visual_manifest = visual_urdf(folder)
    expected = [value for key, value in report["controller_identity"].items()
                if key.endswith("/asset/reachy.urdf")]
    if expected != [visual_manifest["source_urdf_sha256"]]:
        raise ValueError("Viewer URDF differs from the retargeted source model")
    visual_manifest.update(
        source_repository="https://github.com/suk063/reachy-control",
        source_revision=subprocess.check_output(
            ["git", "-C", str(CONTROL), "rev-parse", "HEAD"], text=True).strip(),
        visualization_reference_sha256=sha256(CONTROL / "util/visualization.py"),
    )
    manifest = copy.deepcopy(report["physics"])
    for name, spec in manifest["objects"].items():
        spec["pose_channel"] = f"objects/{name}/pose"
    object_bodies = {spec["body"] for spec in manifest["objects"].values()}
    manifest.setdefault("environment", sorted(set(manifest.get("source_bodies", {})) - object_bodies))
    manifest.update(
        mode="recorded physics replay",
        recorded_scene_sha256=report["scene_sha256"],
        object_pose_policy="Replay complete simulation/qpos; no physics steps or object corrections",
        object_visual_policy="Original scene geometry and materials; renderer primitive-texture limitations apply",
    )
    json_write(folder / "scene-manifest.json", manifest)
    json_write(folder / "visual-manifest.json", visual_manifest)
    return scene, manifest, urdf, visual_manifest


def hide_redundant_display_geometry(model, urdf):
    """Change visibility groups only; retain original collision and inertial data."""
    robot_bodies = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link.get("name"))
                    for link in urdf.findall("link") if link.findall("visual")}
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] in robot_bodies:
            model.geom_group[geom] = 3
            continue
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        # robosuite often includes an exact visual duplicate of a collider.
        # Hide only exact matches so a partially decorated body stays complete.
        candidates = np.flatnonzero((model.geom_bodyid == model.geom_bodyid[geom]) &
                                    (model.geom_contype == 0) & (model.geom_conaffinity == 0))
        for visual in candidates:
            if all(np.array_equal(getattr(model, key)[geom], getattr(model, key)[visual])
                   for key in ("geom_type", "geom_dataid", "geom_size", "geom_pos", "geom_quat")):
                model.geom_group[geom] = 3
                break
    model.site_group[:] = 3


class Playback:
    def __init__(self, motion, model, manifest):
        self.model = model
        self.data = mujoco.MjData(model)
        with h5py.File(motion, "r") as f:
            self.times = f["time_s"][:]
            self.hands = f["robot/hand_pose_world"][:]
            self.targets = f["target/hand_pose_world"][:]
            self.objects = {name: f[spec["pose_channel"]][:] for name, spec in manifest["objects"].items()}
            self.object_specs = manifest["objects"]
            self.channels = {}
            for names_key, values_key in (
                ("robot/joint_names", "robot/joint_position"),
                ("robot/neck_names", "robot/neck_position"),
            ):
                if names_key in f:
                    for name, values in zip(f[names_key].asstr()[:], f[values_key][:].T):
                        self.channels[name] = values
            self.channels.update(zip(BASE, f["robot/base_pose_xyyaw"][:].T))
            for side, prefix in (("left", "l"), ("right", "r")):
                key = f"derived/gripper/{side}_position"
                if key in f:
                    self.channels[f"{prefix}_hand_finger"] = f[key][:]
        if len(self.times) < 2 or not np.all(np.diff(self.times) > 0):
            raise ValueError("Playback requires at least two strictly increasing timestamps")
        self.addresses = {}
        for name in self.channels:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0:
                raise ValueError(f"Recorded joint missing from viewer model: {name}")
            self.addresses[name] = model.jnt_qposadr[joint]
        self.mimics = manifest["mimics"]

    def set_frame(self, index):
        for name, values in self.channels.items():
            self.data.qpos[self.addresses[name]] = values[index]
        for name, (parent, multiplier, offset) in self.mimics.items():
            self.data.joint(name).qpos[0] = self.data.joint(parent).qpos[0] * multiplier + offset
        # Recorded-state playback only: never used by the physical validation runner.
        for name, poses in self.objects.items():
            self.data.joint(self.object_specs[name]["joint"]).qpos[:] = poses[index]
        self.data.time = self.times[index]
        mujoco.mj_forward(self.model, self.data)


class PhysicalPlayback:
    """Replay complete saved physics state; never replace fingers with command targets."""

    def __init__(self, path, model, manifest):
        self.model = model
        self.data = mujoco.MjData(model)
        original = mujoco.MjModel.from_xml_path(str(path.with_name("scene.xml")))
        for field in ("jnt_qposadr", "jnt_dofadr", "jnt_type", "body_parentid", "body_pos", "body_quat", "body_mass", "body_inertia"):
            np.testing.assert_allclose(getattr(model, field), getattr(original, field), atol=1e-12, rtol=0,
                                       err_msg=f"Display model changed physical {field}")
        with h5py.File(path, "r") as f:
            self.times = f["time_s"][:]
            self.qpos = f["simulation/qpos"][:]
            self.qvel = f["simulation/qvel"][:]
            self.ctrl = f["simulation/actuator_control"][:]
            self.hands = f["simulation/hand_pose_world"][:]
            self.targets = f["target/hand_pose_world"][:] if "target/hand_pose_world" in f else None
            self.generic_objects = "simulation/Can_pose" not in f
            if self.generic_objects:
                names = set(f.get("objects", {}))
                if names != set(manifest.get("objects", {})):
                    raise ValueError("Recorded objects differ from physical scene manifest")
                self.objects = {name: f[f"objects/{name}/pose"][:] for name in sorted(names)}
                self.contact = None
                self.bilateral_contact = f["metrics/bilateral_contact"][:]
                self.penetration = f["metrics/hand_object_penetration_m"][:]
            else:
                self.objects = {"Can": f["simulation/Can_pose"][:]}
                self.contact = f["simulation/contacts_hand_force_open"][:]
                self.bilateral_contact = None
                self.penetration = f["metrics/hand_can_penetration_m"][:]
            self.gripper_command = f["command/gripper_position"][:]
            self.drift = f["metrics/grasp_drift_m_rad"][:] if "metrics/grasp_drift_m_rad" in f else None
            names = (list(f["simulation/model_joint_names"].asstr()[:])
                     if "simulation/model_joint_names" in f else
                     [original.joint(i).name for i in range(original.njnt)])
            if names != [model.joint(i).name for i in range(model.njnt)]:
                raise ValueError("Physical replay joint order differs from recorded model")
            if "simulation/model_jnt_qposadr" in f:
                np.testing.assert_array_equal(f["simulation/model_jnt_qposadr"][:], model.jnt_qposadr)
        n = len(self.times)
        if (n < 2 or self.times.shape != (n,) or not np.isfinite(self.times).all()
                or not np.all(np.diff(self.times) > 0)):
            raise ValueError("Invalid physics replay shape/timestamps")
        for name, values, shape in (("qpos", self.qpos, (n, model.nq)),
                                    ("qvel", self.qvel, (n, model.nv)),
                                    ("actuator control", self.ctrl, (n, model.nu)),
                                    ("hands", self.hands, (n, 2, 7))):
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid physics replay {name} shape/values")
        for name, poses in self.objects.items():
            if poses.shape != (n, 7) or not np.isfinite(poses).all():
                raise ValueError(f"Invalid physics replay object poses: {name}")
            if not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1, atol=1e-8, rtol=0):
                raise ValueError(f"Invalid physics replay object quaternion: {name}")
        self.channels = {model.joint(i).name: self.qpos[:, model.jnt_qposadr[i]] for i in range(model.njnt)
                         if int(model.jnt_type[i]) in (int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE))}

    def set_frame(self, index):
        # Assignment is exclusively replay of a completed simulation, not a
        # validation step. Actual finger mimic deflections remain recorded.
        self.data.qpos[:] = self.qpos[index]
        self.data.qvel[:] = self.qvel[index]
        self.data.ctrl[:] = self.ctrl[index]
        self.data.time = self.times[index]
        mujoco.mj_forward(self.model, self.data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--check", action="store_true", help="Verify every replay frame without starting a server")
    parser.add_argument("--pad-markers", action="store_true", help="Show TCP, endpoint and opposing collision-surface centers")
    args = parser.parse_args()
    motion = args.motion.resolve()
    with h5py.File(motion, "r") as f:
        physical = "simulation/qpos" in f
        generic_physical = physical and "simulation/Can_pose" not in f
    report = json.loads(motion.with_name("result.json" if physical else "validation.json").read_text())
    # A separate model directory keeps the physical benchmark's fixed-neck model intact.
    model_root = motion.parent / "viewer"
    scene_path, manifest, urdf, visual_manifest = (
        prepare_generic_physical_scene(motion, report) if generic_physical else
        prepare_scene(motion, report, physical=physical))
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    if generic_physical:
        hide_redundant_display_geometry(model, urdf)
    replay = (PhysicalPlayback if physical else Playback)(motion, model, manifest)
    errors = []
    object_errors = []
    apertures = {side: [] for side, prefix in (("left", "l"), ("right", "r"))
                 if f"{prefix}_hand_finger" in replay.channels}
    for i in range(len(replay.times)):
        replay.set_frame(i)
        if not np.isfinite(replay.data.qpos).all() or not np.isfinite(replay.data.xpos).all():
            raise ValueError(f"Non-finite model state at frame {i}")
        for h, side in enumerate(("l", "r")):
            errors.append(float(np.linalg.norm(replay.data.site(f"{side}_arm_tip_tcp").xpos - replay.hands[i, h, :3])))
        for name, poses in replay.objects.items():
            body = replay.data.body(manifest["objects"][name]["body"])
            object_errors.append(float(np.linalg.norm(body.xpos - poses[i, :3])))
            if abs(float(np.dot(body.xquat, poses[i, 3:]))) < 1 - 1e-8:
                raise ValueError(f"Object quaternion differs from recorded pose at frame {i}")
        for side in apertures:
            prefix = side[0]
            tips = [replay.data.site(f"{prefix}_hand_{name}_endpoint").xpos
                    for name in ("distal_link", "distal_mimic_link")]
            apertures[side].append(float(np.linalg.norm(tips[0]-tips[1])))
    check = {
        "motion_sha256": sha256(motion), "frames": len(replay.times),
        "duration_s": float(replay.times[-1] - replay.times[0]),
        "max_viewer_fk_position_error_m": max(errors),
        "physics_validated": bool(physical and report["success"]),
        "mode": "recorded physics state" if physical else "kinematic playback; no simulation steps",
        "missing_gripper_policy": "Complete measured state required" if physical else "Model default when no derived gripper channel is available",
        "objects": list(replay.objects),
        "max_object_position_error_m": max(object_errors, default=0),
        "environment": manifest["environment"],
        "object_policy": manifest["object_pose_policy"],
        "object_visual_policy": manifest.get("object_visual_policy", "Source visual meshes and materials"),
        "missing_source_visual_assets": manifest.get("missing_visual_assets", []),
        "robot_visuals": "reachy-control URDF visual meshes/materials; visual-only support extension",
        "gripper_mapping": report.get("gripper_mapping", {}),
        "gripper_actual_ranges_rad" if physical else "gripper_target_ranges_rad": {name: [float(values.min()), float(values.max())]
                                      for name, values in replay.channels.items() if name.endswith("hand_finger")},
        "gripper_endpoint_aperture_ranges_m": {side: [min(values), max(values)] for side, values in apertures.items()},
    }
    json_write(model_root / "check.json", check)
    if max(errors) > 0.002:
        raise ValueError(f"Viewer FK differs from stored retargeted hands: {max(errors)} m")
    if max(object_errors, default=0) > 1e-8:
        raise ValueError(f"Viewer object pose differs from recorded state: {max(object_errors)} m")
    if args.check:
        print(json.dumps(check, indent=2))
        return
    server = viser.ViserServer(host="127.0.0.1", port=args.port, label="Reachy retarget replay")
    # Center the camera on the displayed workspace rather than the all-zero
    # joint model's bounding box, which includes hands below the floor.
    model.stat.center[:] = (0.25, 0., 0.85)
    scene = ViserMujocoScene(server, model, num_envs=1)
    scene.camera_tracking_enabled = False
    scene.create_visualization_gui(camera_distance=3.4, camera_azimuth=135, camera_elevation=45)
    visuals = RobotVisuals(server, model, urdf, model_root, visual_manifest)
    with server.gui.add_folder("Retarget playback"):
        mode_text = ("Actual MuJoCo states; " + ("all recorded checks passed." if report["success"] else "failed checks: " + ", ".join(report["failure_reasons"]))) if physical else "Kinematic replay; physics not validated."
        server.gui.add_markdown(
            f"**{report['source_id']}** / {report['source_sequence']}\n\n"
            f"{check['duration_s']:.2f} s · {len(replay.times)} frames · {report['status']}\n\n"
            f"Backend: {report.get('controller_backend', 'native')}. {mode_text} "
            f"Objects: {', '.join(replay.objects) or 'not recorded'}. "
            + ("Finger and object poses are actual simulated states." if physical else "Object motion is recorded playback. Missing grippers use model defaults.")
        )
        play = server.gui.add_checkbox("Play", initial_value=True)
        speed = server.gui.add_slider("Speed", min=0.25, max=4., step=0.25, initial_value=2. if physical else 1.)
        frame = server.gui.add_slider("Frame", min=0, max=len(replay.times)-1, step=1, initial_value=0)
        paths = server.gui.add_checkbox("Hand paths", initial_value=not physical)
        show_visuals = server.gui.add_checkbox("Robot visual meshes", initial_value=True)
        object_paths = server.gui.add_checkbox("Object paths", initial_value=not physical, disabled=not bool(replay.objects))
        info = server.gui.add_markdown("")
        if args.pad_markers:
            server.gui.add_markdown("Contact markers: red = TCP; green = fingertip endpoints; orange = opposing collision-face centers and midpoint. Orange marks a geometric pad proxy.")
    pad_markers = None
    if args.pad_markers:
        from .pad_alignment import pad_surfaces
        pad_markers = server.scene.add_point_cloud("/fixed_bodies/contact_alignment",points=np.zeros((6,3)),
            colors=np.array([(255,50,50),(40,220,90),(40,220,90),(255,150,20),(255,150,20),(255,150,20)],dtype=np.uint8),
            point_size=.006,point_shape="circle",precision="float32")
    @show_visuals.on_update
    def update_visuals(_):
        visuals.root.visible = show_visuals.value
    path_handles = []
    for h, color in enumerate(((60, 180, 255), (255, 160, 60))):
        points = replay.hands[:, h, :3]
        path_handles.append(server.scene.add_line_segments(
            f"/fixed_bodies/hand_path/{h}", points=np.stack((points[:-1], points[1:]), axis=1),
            colors=color, thickness=2., thickness_units="screen",
            visible=paths.value,
        ))
    @paths.on_update
    def update_paths(_):
        for handle in path_handles:
            handle.visible = paths.value
    object_path_handles = []
    for name, poses in replay.objects.items():
        object_path_handles.append(server.scene.add_line_segments(
            f"/fixed_bodies/object_path/{name}", points=np.stack((poses[:-1, :3], poses[1:, :3]), axis=1),
            colors=(170, 65, 230), thickness=2., thickness_units="screen", visible=object_paths.value))
    @object_paths.on_update
    def update_object_paths(_):
        for handle in object_path_handles:
            handle.visible = object_paths.value
    last_frame = -1
    last_tick = time.monotonic()
    cursor = 0.
    while True:
        now = time.monotonic()
        index = int(frame.value)
        if index != last_frame:
            cursor = float(replay.times[index] - replay.times[0])
        if play.value:
            cursor = (cursor + (now-last_tick) * speed.value) % check["duration_s"]
            index = min(len(replay.times)-1, int(np.searchsorted(replay.times-replay.times[0], cursor, side="right")-1))
            frame.value = index
        if index != last_frame:
            replay.set_frame(index)
            scene.update_from_mjdata(replay.data)
            visuals.update(replay.data)
            if pad_markers is not None:
                pads = pad_surfaces(model,replay.data)
                site = model.site("r_arm_tip_tcp").id
                rotation = replay.data.site_xmat[site].reshape(3,3)
                origin = replay.data.site_xpos[site]
                tips = [replay.data.site(name+"_endpoint").xpos for name in ("r_hand_distal_link","r_hand_distal_mimic_link")]
                pad_markers.points = np.vstack([origin,tips,pads["points_tcp"] @ rotation.T+origin,
                                                rotation @ pads["midpoint_tcp"]+origin]).astype(np.float32)
            gripper_text = "".join(f"  \n{side.title()} gripper {'actual' if physical else 'target'}: {replay.channels[side[0]+'_hand_finger'][index]:.2f} rad"
                                   for side in apertures)
            if physical:
                gripper_text += f"  \nClose/open command: {replay.gripper_command[index]:.2f} rad"
                if replay.generic_objects:
                    gripper_text += (f"  \nBilateral finger contact: {'yes' if replay.bilateral_contact[index] else 'no'}"
                                     f"  \nHand/object penetration: {replay.penetration[index]*1000:.3f} mm")
                else:
                    gripper_text += (f"  \nHand contact: {'yes' if replay.contact[index, 0] else 'no'}"
                                     f" · peak force: {replay.contact[index, 1]:.2f} N"
                                     f"  \nHand/can penetration: {replay.penetration[index]*1000:.3f} mm")
                if replay.drift is not None and np.isfinite(replay.drift[index]).all():
                    gripper_text += (f"  \nObject drift in grasp: {replay.drift[index, 0]*1000:.2f} mm"
                                     f" / {np.rad2deg(replay.drift[index, 1]):.2f}°")
            info.content = f"Time: {replay.times[index]:.2f} s" + gripper_text
            last_frame = index
        last_tick = now
        time.sleep(1/60)


if __name__ == "__main__":
    main()
