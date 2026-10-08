"""Source-family adapter registry.

An adapter is a generator ``fn(path, *, family, **kw) -> Iterator[SourceEpisode]``
that only reads local source files. Several families may share one adapter (e.g.
robomimic and MimicGen are both robosuite recordings).
"""
from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from pathlib import Path

from ..schema.source import SourceEpisode

Adapter = Callable[..., Iterator[SourceEpisode]]
_ADAPTERS: dict[str, Adapter] = {}

# family -> adapter module; modules are imported on first use, so one family's
# dependencies (or a broken adapter) never affect another family.
MODULES = {
    "behavior": "behavior", "bigym": "bigym", "dexmimicgen": "dexmimicgen", "libero": "libero",
    "maniskill": "maniskill", "mimicgen": "robosuite", "mobilemanibench": "mobilemanibench",
    "molmobot": "molmobot", "robocasa": "robocasa", "robomimic": "robosuite", "roboverse": "roboverse",
}


def register(family: str, *more: str) -> Callable[[Adapter], Adapter]:
    """Decorator registering an adapter under one or more family names."""
    def deco(fn: Adapter) -> Adapter:
        for name in (family, *more):
            if name in _ADAPTERS:
                raise ValueError(f"family {name!r} already registered")
            _ADAPTERS[name] = fn
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


def iter_episodes(family: str, path, **kw) -> Iterator[SourceEpisode]:
    """Yield the episodes of one local source file through its family adapter."""
    yield from adapter(family)(Path(path), family=family, **kw)
