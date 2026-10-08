"""Source adapters: each module registers one or more families producing SourceEpisode objects.

Adapter modules are imported lazily by :func:`iter_episodes` (see ``registry.MODULES``).
"""
from .registry import adapter, families, iter_episodes, register

__all__ = ["adapter", "families", "iter_episodes", "register"]
