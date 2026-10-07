"""Offline MeLLO Vicon adapter; source markers are never robot observations.

No network, simulator, robot SDK, gap filling, or guessed sensor alignment is
performed here. A derived marker frame is not a calibrated collision-mesh pose.
"""

import csv
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


REVISION = "ea82019e6093b85311b2c5e4865760a5d0945ce6"
REPOSITORY = "https://github.com/nluttmer1/MeLLO-Data-Library"
PILOT_SEQUENCE = "Subject 1/Box/Walk/Cut VICON Data/Subject 1 - Box Walk 1.csv"
PILOT_URL = (
    "https://raw.githubusercontent.com/nluttmer1/MeLLO-Data-Library/"
    + REVISION
    + "/Data%20Library/Subject%201/Box/Walk/Cut%20VICON%20Data/Subject%201%20-%20Box%20Walk%201.csv"
)
PILOT_SHA256 = "9a65e44d910e6f2692c908a54b11c775952eecec8dba5b947369fb400e1d166f"
MISSING = [
    "calibrated_object_mesh_frame_and_collision_geometry",
    "object_mass_inertia_and_contact_parameters_for_selected_trial",
    "human_hand_orientation_and_grasp_contact_labels",
    "synchronized_load_cell_and_imu_streams",
    "retargeted_reachy_state_action_and_base_trajectory",
    "actuator_driven_physical_validation",
    "calibrated_task_object_pose",
]


def describe():
    """Return a bounded acquisition and reconstruction plan, without fetching."""
    return {
        "dataset": "mello",
        "source_revision": REVISION,
        "source_urls": [REPOSITORY, "https://pmc.ncbi.nlm.nih.gov/articles/PMC11939698/"],
        "license": "CC-BY-4.0; preserve attribution and identify derived fields",
        "native_format": "Vicon Trajectories CSV; separate IMU CSV, dSpace MAT/LFS and OpenSim MOT",
        "pilot": {"url": PILOT_URL, "path": PILOT_SEQUENCE, "bytes": 506651, "sha256": PILOT_SHA256},
        "storage_root": "/mnt/reachy-retarget",
        "pipeline": [
            {"stage": "fetch", "operation": "Fetch selected object-interaction trial and pinned provenance; exclude Gait-only trials"},
            {"stage": "inspect", "operation": "Check Vicon section, measured rate, units, labels, gaps, and original frame indices"},
            {"stage": "normalize", "operation": "Convert observed XYZ to metres; preserve gaps and frame clock; fit Box marker-frame motion with residual mask"},
            {"stage": "agent_review", "operation": "Bind calibrated mesh and marker frames, synchronize force streams, and review payload and grasp feasibility"},
            {"stage": "retarget", "operation": "Map reachable hand and object-relative contacts; preserve source motion and apply only necessary pad alignment"},
            {"stage": "validate", "operation": "Run actuator-only MuJoCo attempts with unmodified object scale/mass and retain failures"},
            {"stage": "export", "operation": "Write shared archive; emit complete Reachy-agent v5 only after required observations are produced"},
        ],
        "missing_fields": list(MISSING),
        "status": "native_marker_normalization_implemented_reconstruction_required",
        "physics_validated": False,
        "task_object_pose_coverage": {
            "scope": "observed_task_marker_groups", "all_task_object_poses_saved": False,
            "unrelated_scene_objects_required": False,
        },
        "notes": [
            "The paper describes synchronized Final Data, but the pinned public Git tree has no Final Data files.",
            "Box, luggage, briefcase, walker, cart, wheelbarrow, and door are instrumented large objects; feasibility is not assumed.",
            "A cut trial, full recording, and OpenSim derivatives must share one source group.",
        ],
    }


def _read(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if rows and rows[0] and rows[0][0].startswith("version https://git-lfs"):
        raise ValueError("Git LFS pointer is not a MeLLO payload")
    sections = [i for i, row in enumerate(rows) if row and row[0].strip() == "Trajectories"]
    if len(sections) != 1:
        raise ValueError("Expected exactly one Vicon Trajectories section")
    start = sections[0]
    if start + 5 >= len(rows):
        raise ValueError("Incomplete Vicon header")
    rate = float(rows[start + 1][0])
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("Invalid measured Vicon rate")
    labels, axes, units = rows[start + 2:start + 5]
    if axes[:2] != ["Frame", "Sub Frame"] or (len(axes) - 2) % 3:
        raise ValueError("Unrecognized Vicon frame/XYZ layout")
    nmarkers = (len(axes) - 2) // 3
    names = [labels[2 + 3 * i].strip() for i in range(nmarkers)]
    if not all(names) or len(set(names)) != nmarkers:
        raise ValueError("Missing or duplicate Vicon marker labels")
    for col in range(2, len(axes), 3):
        if axes[col:col + 3] != ["X", "Y", "Z"]:
            raise ValueError("Unrecognized Vicon marker axis order")
    if len(units) != len(axes) or set(units[2:]) not in ({"mm"}, {"m"}):
        raise ValueError("Mixed or unsupported Vicon spatial units")
    numeric = []
    for row in rows[start + 5:]:
        if not row or not any(row):
            if numeric:
                break
            continue
        if len(row) != len(axes):
            raise ValueError("Vicon data row width differs from header")
        numeric.append([float(value) if value.strip() else np.nan for value in row])
    data = np.asarray(numeric, dtype=float)
    if data.ndim != 2 or len(data) < 2:
        raise ValueError("Need at least two measured marker frames")
    frame, subframe = data[:, 0], data[:, 1]
    if not np.isfinite(data[:, :2]).all() or not np.all(subframe == 0):
        raise ValueError("Nonzero subframes require an explicit clock model")
    if not np.all(frame == np.floor(frame)) or not np.all(np.diff(frame) > 0):
        raise ValueError("Vicon frame numbers must be increasing integers")
    if np.isinf(data[:, 2:]).any():
        raise ValueError("Infinite marker coordinates")
    positions = data[:, 2:].reshape(len(data), nmarkers, 3)
    positions *= 0.001 if units[2] == "mm" else 1.0
    valid = np.isfinite(positions).all(axis=-1)
    return rate, names, frame.astype(np.int64), positions, valid


def fit_marker_frame(positions, valid, *, max_residual_m=0.005):
    """Fit observed rigid correspondences; invalid frames remain NaN.

    The reference frame uses the first fully observed marker centroid and the
    source world axes. It has no implied relationship to an object mesh/COM.
    """
    positions = np.asarray(positions, float)
    valid = np.asarray(valid, bool)
    if positions.ndim != 3 or positions.shape[2] != 3 or valid.shape != positions.shape[:2]:
        raise ValueError("Expected frame x marker x XYZ and matching valid mask")
    if max_residual_m <= 0 or not np.isfinite(max_residual_m):
        raise ValueError("Invalid marker fitting residual threshold")
    pose = np.full((len(positions), 7), np.nan)
    residual = np.full(len(positions), np.nan)
    accepted = np.zeros(len(positions), bool)
    candidates = np.flatnonzero(valid.all(axis=1))
    if not len(candidates) or positions.shape[1] < 3:
        return pose, residual, accepted, None
    first = int(candidates[0])
    template = positions[first] - positions[first].mean(axis=0)
    if np.linalg.matrix_rank(template, tol=1e-8) < 2:
        return pose, residual, accepted, None
    for i in range(len(positions)):
        keep = valid[i] & np.isfinite(positions[i]).all(axis=1)
        if keep.sum() < 3:
            continue
        source, target = template[keep], positions[i, keep]
        source_center, target_center = source.mean(axis=0), target.mean(axis=0)
        if np.linalg.matrix_rank(source - source_center, tol=1e-8) < 2:
            continue
        if np.linalg.matrix_rank(target - target_center, tol=1e-8) < 2:
            continue
        u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(vt.T @ u.T)
        rotation = vt.T @ correction @ u.T
        translation = target_center - rotation @ source_center
        residual[i] = np.sqrt(np.mean(np.sum((source @ rotation.T + translation - target) ** 2, axis=1)))
        if residual[i] <= max_residual_m:
            quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
            pose[i] = np.r_[translation, quat]
            accepted[i] = True
    return pose, residual, accepted, first


def normalize(path, output_dir=None):
    """Return shared in-memory channels; the common exporter owns persistence."""
    path = Path(path)
    rate, names, frames, positions, valid = _read(path)
    object_prefixes = {"box", "luggage", "briefcase", "walker", "shoppingcart", "cart", "wheelbarrow", "door"}
    object_names = sorted({name.split(":", 1)[0] for name in names if ":" in name
                           and name.split(":", 1)[0].lower().replace(" ", "").replace("_", "") in object_prefixes})
    if not object_names:
        raise ValueError("Robot/human-only trials are outside the object-interaction scope")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sequence = PILOT_SEQUENCE if digest == PILOT_SHA256 else path.as_posix().split("Data Library/")[-1]
    parts = sequence.split("/")
    group = "/".join(parts[:3]) if parts[0].startswith("Subject ") and len(parts) >= 4 else "unresolved-recording/" + digest
    arrays = {
        "time_s": (frames - frames[0]) / rate,
        "source/frame_index": frames,
        "source/frame_clock_s": frames / rate,
        "markers/names": np.asarray(names),
        "markers/position_m": positions,
        "markers/valid": valid,
    }
    derived = {
        "time_s": "(source frame - first source frame) / measured Vicon rate; no interpolation",
        "source/frame_clock_s": "Source frame / measured rate relative to frame index zero; absolute sensor clock origin is unknown",
        "markers/position_m": "Source XYZ converted from header units; missing values remain NaN",
    }
    masks = {"markers/position_m": "markers/valid"}
    # Only the inspected Box corner marker set is declared rigid here. Other
    # objects can contain articulated handles and require a reviewed grouping.
    box_names = ["Box:" + label for label in ("BFBR", "BFBL", "BFTL", "BFTR", "BBR", "BBL")]
    if all(name in names for name in box_names):
        indexes = [names.index(name) for name in box_names]
        pose, residual, accepted, first = fit_marker_frame(positions[:, indexes], valid[:, indexes])
        arrays.update({
            "objects/Box/marker_frame_pose_wxyz": pose,
            "objects/Box/marker_frame_valid": accepted,
            "objects/Box/marker_fit_rmse_m": residual,
        })
        derived["objects/Box/marker_frame_pose_wxyz"] = {
            "method": "Rigid SVD correspondence fit; no scale fitting or gap filling",
            "reference_frame_index": None if first is None else int(frames[first]),
            "marker_names": box_names,
            "max_accepted_rmse_m": 0.005,
            "frame": "Marker centroid with first-frame world axes; mesh/COM transform unknown",
        }
        masks.update({"objects/Box/marker_frame_pose_wxyz": "objects/Box/marker_frame_valid",
                      "objects/Box/marker_fit_rmse_m": "objects/Box/marker_frame_valid"})
    metadata = {
        "source_id": "mello", "source_revision": REVISION,
        "source_revision_verified_for_pilot": digest == PILOT_SHA256,
        "source_urls": [PILOT_URL] if digest == PILOT_SHA256 else [REPOSITORY],
        "source_sequence": sequence, "source_group": "mello/" + group,
        "source_sha256": digest, "source_rate_hz": rate,
        "coordinate_frame": "Original Vicon world frame; no scene alignment applied",
        "canonical_units": {"length": "m", "time": "s", "quaternion": "wxyz"},
        "objects": {name: {"source_marker_prefix": name, "mesh_frame_calibrated": False} for name in object_names},
        "task_object_pose_coverage": {
            "scope": "observed_task_marker_groups", "all_task_object_poses_saved": False,
            "required_task_object_ids": object_names,
            "observed_object_marker_groups": object_names,
            "derived_rigid_marker_frame_ids": ["Box"] if "objects/Box/marker_frame_pose_wxyz" in arrays else [],
            "unrelated_scene_objects_required": False,
            "all_native_marker_observations_saved": True,
            "reason": "Task marker motion is retained, but calibrated task mesh poses are unresolved; unrelated scene props are not required",
        },
        "derived_fields": derived, "validity_masks": masks, "simulation_assumptions": [],
        "missing_fields": list(MISSING), "physics_validated": False,
        "source_content": "Measured human/object markers; no Reachy state, action, sensor or camera was synthesized",
    }
    return {"arrays": arrays, "metadata": metadata, "status": "normalized_partial", "missing_fields": list(MISSING)}


def inspect(path):
    result = normalize(path)
    arrays, metadata = result["arrays"], result["metadata"]
    return {
        "dataset": "mello", "status": "inspected_not_retargeted",
        "frames": len(arrays["time_s"]), "duration_s": float(arrays["time_s"][-1]),
        "marker_count": len(arrays["markers/names"]),
        "missing_marker_observations": int((~arrays["markers/valid"]).sum()),
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
        "metadata": metadata, "missing_fields": result["missing_fields"],
        "physics_validated": False,
    }
