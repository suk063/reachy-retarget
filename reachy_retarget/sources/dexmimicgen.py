"""Adapter for DexMimicGen generated datasets (``MimicGen/dexmimicgen_datasets``).

Only the bimanual Panda tasks with parallel-jaw grippers are catalogued
(TwoArmThreading, TwoArmThreePieceAssembly, TwoArmTransport; robosuite 1.5.1,
``env_configuration = "single-arm-parallel"``). Dexterous-hand tasks (PandaDex arms,
GR-1 humanoid) are excluded because Reachy has parallel grippers; the shared
parallel-gripper model raises on them.

The robosuite reader produces one effector per arm (``gripper0_right``,
``gripper1_right``); both arms face the table side by side, so ``side_hint`` comes from
the robot base layout (see :func:`.robosuite._side_hints`). Every demo is
MimicGen-generated; the human source demos are not part of the release, so
``lineage = {"generated": True, "seed": None, ...}`` and the per-demo seed is unknown.
"""
from __future__ import annotations

from .registry import register
from .robosuite import TASK_OBJECTS, Profile, _env_key, read_robosuite_family

PARALLEL_JAW_ENVS = ("TwoArmThreading", "TwoArmThreePieceAssembly", "TwoArmTransport")


def _lineage(entry, dataset: str) -> dict:
    return {"generated": True, "seed": None, "seed_demo": None,
            "seed_note": "DexMimicGen human source demos are not published with the generated datasets",
            "generator": "DexMimicGen"}


def _task_objects(env_name: str, f):
    if env_name not in PARALLEL_JAW_ENVS:
        raise ValueError(f"DexMimicGen env {env_name!r} is not a parallel-jaw task; only "
                         f"{PARALLEL_JAW_ENVS} are supported")
    return TASK_OBJECTS.get(_env_key(env_name))


DEXMIMICGEN = Profile(task_objects=_task_objects, lineage=_lineage)


@register("dexmimicgen", select=True)
def read_dexmimicgen_hdf5(path, *, family: str, **kw):
    """DexMimicGen parallel-jaw task files (see :func:`.robosuite.read_robosuite_family`)."""
    return read_robosuite_family(path, family=family, profile=DEXMIMICGEN, **kw)


__all__ = ["DEXMIMICGEN", "PARALLEL_JAW_ENVS", "read_dexmimicgen_hdf5"]
