"""Adapter for the official LIBERO HDF5 release (``yifengzhu-hf/LIBERO-datasets``).

LIBERO files are robosuite recordings (``states`` = ``[sim time, qpos, qvel]`` and the
MJCF per demo in ``model_file``), so the shared reader of :mod:`.robosuite` does the
state replay. LIBERO specifics:

* No ``env_version`` is recorded (LIBERO was built on a robosuite 1.4 pre-release); the
  robosuite 1.4.1 wheel serves the robot assets and ``provenance["env_version_assumed"]``
  says so. Scene assets were recorded under ``chiliocosm/assets/`` (LIBERO's pre-release
  name) and resolve into ``libero/libero/assets/`` of the pinned LIBERO archive.
* ``instruction`` = ``problem_info["language_instruction"]``. The BDDL problem named by
  ``bddl_file_name`` is read from the LIBERO archive; its objects of interest, fixtures,
  movable objects and goal go to ``provenance["bddl"]``.
* Every free body is a movable BDDL object and is kept as ``manipulated`` (physics
  validation then checks that none is knocked over); articulated fixtures (cabinets,
  microwaves, stoves) appear as articulations.
* ``success``: LIBERO stores a 0/1 task-completion reward (``uint8``), so success is
  max reward >= 1 even though ``env_kwargs.reward_shaping`` is set.
* Demonstrations are human teleoperation (50 per task): ``lineage = {"generated": False}``.
"""
from __future__ import annotations

import json
import re
import zipfile

import numpy as np

from .registry import register
from .robosuite import Profile, read_robosuite_family

LIBERO_MARKER = "libero/libero/assets/"
ALIASES = {LIBERO_MARKER: ("chiliocosm/assets/",)}


def _instruction(f) -> str | None:
    raw = f["data"].attrs.get("problem_info")
    if raw is None:
        return None
    info = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    text = info.get("language_instruction")
    return " ".join(text) if isinstance(text, list) else text


def _section(text: str, name: str) -> str | None:
    """Body of the balanced ``(:name ...)`` block of a BDDL problem."""
    i = text.find(f"(:{name}")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        depth += {"(": 1, ")": -1}.get(text[j], 0)
        if depth == 0:
            return text[i + len(name) + 2:j].strip()
    return None


def parse_bddl(text: str) -> dict:
    """Objects of interest, fixtures, movable objects, language and goal of a BDDL problem."""
    typed = lambda s: re.findall(r"(\S+)\s+-\s+\S+", s or "")
    goal = _section(text, "goal")
    return {"language": (_section(text, "language") or "").strip() or None,
            "objects_of_interest": (_section(text, "obj_of_interest") or "").split(),
            "fixtures": typed(_section(text, "fixtures")),
            "objects": typed(_section(text, "objects")),
            "goal": " ".join(goal.split()) if goal else None}


def _bddl(f, ctx) -> dict:
    name = f["data"].attrs.get("bddl_file_name")
    if name is None:
        return {"bddl": None}
    name = name.decode() if isinstance(name, bytes) else str(name)
    out = {"file": name, "archive": None}
    for (path, marker, _aliases, _prefix), entry in ctx["archives"]:
        if marker != LIBERO_MARKER:
            continue
        with zipfile.ZipFile(path) as z:
            member = next((n for n in z.namelist() if n.endswith("/" + name)), None)
            if member:
                out.update(parse_bddl(z.read(member).decode()), archive=entry.id if entry else path)
    return {"bddl": out}


def _success(g, kwargs) -> bool | None:
    if "rewards" not in g:
        return None
    r = np.asarray(g["rewards"][:])
    if not set(np.unique(r).tolist()) <= {0, 1}:
        return None  # not the 0/1 completion signal this rule assumes
    return bool(r.max() >= 1)


LIBERO = Profile(
    task_objects=lambda env_name, f: None,
    instruction=_instruction,
    lineage=lambda entry, dataset: {"generated": False},
    extra_provenance=_bddl,
    marker_aliases=ALIASES,
    default_env_version="1.4.1",
    success=_success,
    success_rule="max recorded reward >= 1 (LIBERO 0/1 task-completion reward)",
)


@register("libero", select=True)
def read_libero_hdf5(path, *, family: str, **kw):
    """LIBERO task files (see :func:`.robosuite.read_robosuite_family`)."""
    return read_robosuite_family(path, family=family, profile=LIBERO, **kw)


__all__ = ["LIBERO", "parse_bddl", "read_libero_hdf5"]
