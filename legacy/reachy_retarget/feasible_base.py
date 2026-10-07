"""Explicit robot-only base corrections for admitted manipulation geometries.

The caller supplies each base pose on the unmodified source reference clock.
The shared solver preserves target hands, object states and the scene, records
all failures, and admits only complete IK/collision checks. Physical validation
still requires the existing actuator-only rollout and audit.
"""
import json
from pathlib import Path

from .feasible_maniskill import _prepare
from .store import json_write


def prepare(prepared, robot, output, base_xyyaw, *, right_arm_seed=None):
    task = prepared[5].get("task")
    eligibility = None
    if task != "PickPlaceCan":
        details = prepared[5]
        attachment = details.get("grasp_attachment", {})
        if (details.get("alignment_variant") != "constant_tcp_attachment"
                or attachment.get("requires_full_ik_and_collision_admission") is not True):
            raise ValueError("Generic base admission requires a calibrated constant box attachment")
        from .grasp_attachment import single_box_geometry
        obj = prepared[2]["objects"][details["object_id"]]
        eligibility = single_box_geometry(prepared[0], object_body=obj["body"],
                                          object_joint=obj["joint"])
    try:
        result = _prepare(prepared, robot, output, base_xyyaw, expected_task=task,
                          right_arm_seed=right_arm_seed,
                          gripper_admission="open_envelope" if eligibility is None else "intent_envelope")
        if eligibility is not None:
            result[5]["robot_initialization"]["shape_eligibility"] = eligibility
        return result
    finally:
        report = Path(output) / "result.json"
        if eligibility is not None and report.exists():
            value = json.loads(report.read_text())
            value["shape_eligibility"] = eligibility
            json_write(report, value)
