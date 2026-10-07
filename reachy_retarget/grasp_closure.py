"""Delay only robot gripper closure until the planned pad is central and stable.

This is a planning experiment, not evidence of actual acquisition. The caller
must supply a declared actuator response estimate and run the unchanged physical
gates. Use after any robot-clock retiming; never edit the original source file.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np

from .store import json_write, sha256


def plan(times, hands, objects, commands, pad_midpoint_tcp, center_object, world_up, *,
         response_time_s, center_tolerance_m=.003, settle_s=.05,
         max_reference_lift_m=.005):
    """Return derived intent, admission report, and numeric diagnostics.

    Exactly one contiguous negative contact-intent phase is currently supported.
    A candidate requires an already observed reference dwell inside the central
    region, remains central through estimated actuator response, and precedes
    the declared reference-lift bound and original release. No row is removed.
    """
    times, hands, objects = (np.asarray(x, dtype=float) for x in (times, hands, objects))
    commands = np.asarray(commands, dtype=float)
    midpoint, center, up = (np.asarray(x, dtype=float) for x in
                            (pad_midpoint_tcp, center_object, world_up))
    n = len(times)
    values = (times, hands, objects, commands, midpoint, center, up)
    if (times.ndim != 1 or n < 2 or not np.all(np.diff(times) > 0)
            or hands.shape != (n, 4, 4) or objects.shape != (n, 4, 4)
            or commands.shape != (n,) or any(x.shape != (3,) for x in (midpoint, center, up))
            or not all(np.isfinite(x).all() for x in values)
            or np.linalg.norm(up) < 1e-10):
        raise ValueError("Finite complete poses, intent, increasing clock and explicit world up required")
    settings = dict(response_time_s=response_time_s, center_tolerance_m=center_tolerance_m,
                    settle_s=settle_s, max_reference_lift_m=max_reference_lift_m)
    if (not all(np.isfinite(x) for x in settings.values()) or response_time_s < 0
            or center_tolerance_m <= 0 or settle_s < 0 or max_reference_lift_m < 0):
        raise ValueError("Invalid central-region, dwell, response or reference-lift bound")
    up = up / np.linalg.norm(up)
    closed = np.flatnonzero(commands < 0)
    if not len(closed) or not np.all(np.diff(closed) == 1):
        raise ValueError("Exactly one contiguous negative contact-intent phase required")
    start = int(np.flatnonzero(commands[:closed[0]+1] < 1.9999)[0])
    end = int(closed[-1])
    if start == 0:
        raise ValueError("An original open pregrasp interval is required")
    midpoint_world = np.einsum("nij,j->ni", hands[:, :3, :3], midpoint)+hands[:, :3, 3]
    midpoint_object = np.einsum("nji,nj->ni", objects[:, :3, :3], midpoint_world-objects[:, :3, 3])
    error = np.linalg.norm(midpoint_object-center, axis=1)
    central = error <= center_tolerance_m
    lift = (objects[:, :3, 3]-objects[start, :3, 3]) @ up
    candidates = []
    chosen = None
    for index in range(start, end+1):
        past_time = times[index]-settle_s
        expected_contact = times[index]+response_time_s
        past = int(np.searchsorted(times, past_time, side="right")-1)
        future = int(np.searchsorted(times, expected_contact, side="left"))
        dwell_valid = past >= start and bool(np.all(central[past:index+1]))
        response_available = future <= end
        stays_central = bool(response_available and np.all(central[index:future+1]))
        before_lift = bool(response_available and np.max(lift[index:future+1]) <= max_reference_lift_m)
        if dwell_valid and stays_central and before_lift:
            chosen = index
            candidates = [past, future]
            break
    changed = commands.copy()
    if chosen is not None:
        changed[start:chosen] = 2.
    report = dict(settings, admitted=chosen is not None, physics_validated=False,
                  original_closure_frame=start, original_closure_time_s=float(times[start]),
                  original_negative_contact_end_frame=end,
                  original_negative_contact_end_time_s=float(times[end]),
                  derived_closure_frame=chosen,
                  derived_closure_time_s=None if chosen is None else float(times[chosen]),
                  delay_s=None if chosen is None else float(times[chosen]-times[start]),
                  expected_contact_time_s=None if chosen is None else float(times[chosen]+response_time_s),
                  expected_contact_reference_lift_m=None if chosen is None else float(lift[candidates[1]]),
                  gate_error_at_closure_m=None if chosen is None else float(error[chosen]),
                  world_up=up.tolist(), center_object_m=center.tolist(), pad_midpoint_tcp_m=midpoint.tolist(),
                  changed_frames=int(np.count_nonzero(changed != commands)),
                  failure_reason=None if chosen is not None else
                    "No stable central closure and declared actuator-response window before reference lift/release",
                  semantics="Derived robot gripper intent only; planned geometry plus declared response estimate, not a measured acquisition guarantee. Complete reference poses and control clock unchanged; source intent retained separately.")
    diagnostics = dict(pad_midpoint_object_m=midpoint_object, central_error_m=error,
                       reference_lift_m=lift, central_region_mask=central)
    return changed, report, diagnostics


def apply(prepared, output, *, response_time_s, center_tolerance_m=.003, settle_s=.05,
          max_reference_lift_m=.005, response_evidence=None):
    """Apply after retiming, before rollout; preserve rejected plan evidence."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / Path(__file__).name).write_text(Path(__file__).read_text())
    details = deepcopy(prepared[5])
    pad = details["pad_alignment"]
    times, commands, hands, objects = prepared[7], prepared[9], prepared[10], prepared[11]
    changed, report, diagnostics = plan(
        times, hands, objects, commands, pad["pad"]["midpoint_tcp"],
        pad["section"]["center_object"], -np.asarray(prepared[0].opt.gravity),
        response_time_s=response_time_s, center_tolerance_m=center_tolerance_m,
        settle_s=settle_s, max_reference_lift_m=max_reference_lift_m)
    artifact = output / "closure-intent.npz"
    np.savez_compressed(artifact, control_time_s=times, original_hand_goals=hands,
                        original_object_goals=objects, original_gripper_intent=commands,
                        derived_gripper_intent=changed, **diagnostics)
    report.update(artifact=str(artifact), artifact_sha256=sha256(artifact),
                  response_evidence=deepcopy(response_evidence),
                  response_assumption="Explicit experiment input; actual response must be measured in the new rollout")
    json_write(output / "result.json", report)
    if not report["admitted"]:
        raise ValueError("Geometry-gated closure rejected; inspect " + str(output / "result.json"))
    details["grasp_closure"] = report
    details["closed_command_semantics"] = "Geometry-gated derived robot contact intent; original source intent preserved in closure-intent.npz"
    result = list(prepared)
    result[5], result[9] = details, changed
    return tuple(result)
