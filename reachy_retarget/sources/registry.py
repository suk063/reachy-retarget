"""Source-family adapter registry.

An adapter is a generator ``fn(path, *, family, **kw) -> Iterator[SourceEpisode]``
that only reads local source files. Several families may share one adapter (e.g.
robomimic and MimicGen are both robosuite recordings).
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

from ..schema.source import SourceEpisode

Adapter = Callable[..., Iterator[SourceEpisode]]
_ADAPTERS: dict[str, Adapter] = {}


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
    return sorted(_ADAPTERS)


def iter_episodes(family: str, path, **kw) -> Iterator[SourceEpisode]:
    """Yield the episodes of one local source file through its family adapter."""
    if family not in _ADAPTERS:
        raise KeyError(f"unknown source family {family!r}; known: {families()}")
    yield from _ADAPTERS[family](Path(path), family=family, **kw)
