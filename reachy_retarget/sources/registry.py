"""Source-family adapter registry.

An adapter is a generator ``fn(path, *, family, **kw) -> Iterator[SourceEpisode]``
that only reads local source files. Several families may share one adapter (e.g.
robomimic and MimicGen are both robosuite recordings).

Episode selection: :func:`iter_episodes` takes ``select``, a predicate over the 0-based
position of an episode in the adapter's order. Positions it rejects are yielded as
``None`` (so callers keep counting positions) and never read. Adapters registered with
``select=True`` implement this themselves (unselected episodes are not loaded, compiled or
replayed); for the others the registry builds every episode and drops the unselected ones.

Scene meshes (``MESHES``): built episodes must store the meshes of their scene components (the
policy builds a surface feature map from them, see docs/design.md "Scene meshes"). This table is
the single place that says which families provide them; :func:`require_meshes` refuses the
others before a build reads anything.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from pathlib import Path

from ..schema.source import SourceEpisode

Adapter = Callable[..., Iterator[SourceEpisode]]
_ADAPTERS: dict[str, Adapter] = {}
_SELECTS: set[str] = set()  # families whose adapter accepts ``select`` itself

# family -> adapter module; modules are imported on first use, so one family's
# dependencies (or a broken adapter) never affect another family.
MODULES = {
    "behavior": "behavior", "bigym": "bigym", "dexmimicgen": "dexmimicgen", "libero": "libero",
    "maniskill": "maniskill", "mimicgen": "robosuite", "mobilemanibench": "mobilemanibench",
    "molmobot": "molmobot", "robocasa": "robocasa", "robomimic": "robosuite", "roboverse": "roboverse",
}


# family -> (status, note). "available": the adapter attaches a MuJoCo scene (SceneRef) whose
# meshes and textures are resolved from pinned archives (episodes whose scene cannot be resolved
# are still excluded one by one: build record ``excluded: no_meshes``); "pending": mesh extraction
# not implemented yet; "excluded": meshes cannot be obtained, the family is not built.
MESHES = {
    "robomimic": ("available", "robosuite MJCF per demo + robosuite wheel assets"),
    "mimicgen": ("available", "robosuite MJCF per demo + robosuite wheel and MimicGen assets"),
    "libero": ("available", "robosuite MJCF per demo + LIBERO assets"),
    "dexmimicgen": ("available", "robosuite MJCF per demo + robosuite/DexMimicGen assets"),
    "robocasa": ("available", "recorded kitchen MJCF + RoboCasa asset archives"),
    "bigym": ("available", "replay-record MJCF + exported BiGym assets"),
    "maniskill": ("available", "primitive scene rebuilt from the task geometry (boxes, spheres, ...)"),
    "roboverse": ("pending", "scene mesh extraction for RoboVerse sources is not implemented yet"),
    "mobilemanibench": ("pending", "scene mesh extraction for MobileManiBench is not implemented yet"),
    "behavior": ("excluded", "BEHAVIOR-1K object assets are encrypted; no meshes can be stored"),
    "molmobot": ("excluded", "MolmoBot-Data has no per-frame object poses and no scene meshes"),
}


def mesh_status(family: str) -> tuple[str, str]:
    """``(status, note)`` of a family in :data:`MESHES` (unknown families: ``("unknown", ...)``)."""
    return MESHES.get(family, ("unknown", "family not listed in sources.registry.MESHES"))


def require_meshes(family: str) -> None:
    """Raise ``ValueError`` unless ``family`` provides scene meshes (``MESHES`` status ``available``)."""
    status, note = mesh_status(family)
    if status != "available":
        raise ValueError(f"family {family!r} cannot be built: scene meshes {status} ({note}). Episodes must store "
                         "their object meshes; see sources.registry.MESHES")


def register(family: str, *more: str, select: bool = False) -> Callable[[Adapter], Adapter]:
    """Decorator registering an adapter under one or more family names.

    ``select=True``: the adapter accepts ``select`` (see the module docstring)."""
    def deco(fn: Adapter) -> Adapter:
        for name in (family, *more):
            if name in _ADAPTERS:
                raise ValueError(f"family {name!r} already registered")
            _ADAPTERS[name] = fn
            if select:
                _SELECTS.add(name)
        return fn
    return deco


def families() -> list[str]:
    return sorted(set(MODULES) | set(_ADAPTERS))


def adapter(family: str) -> Adapter:
    """The adapter of ``family``, importing its module on first use."""
    if family not in _ADAPTERS and family in MODULES:
        importlib.import_module(f"{__package__}.{MODULES[family]}")
    if family not in _ADAPTERS:
        raise KeyError(f"unknown source family {family!r}; known: {families()}")
    return _ADAPTERS[family]


def iter_episodes(family: str, path, *, select: Callable[[int], bool] | None = None,
                  **kw) -> Iterator[SourceEpisode | None]:
    """Yield the episodes of one local source file through its family adapter.

    With ``select``, episodes at rejected positions are yielded as ``None``."""
    fn = adapter(family)
    if select is None:
        yield from fn(Path(path), family=family, **kw)
    elif family in _SELECTS:
        yield from fn(Path(path), family=family, select=select, **kw)
    else:
        for i, ep in enumerate(fn(Path(path), family=family, **kw)):
            yield ep if select(i) else None
