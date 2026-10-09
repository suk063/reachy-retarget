"""Source adapters: each module registers one or more families producing SourceEpisode objects.

Adapter modules are imported lazily by :func:`iter_episodes` (see ``registry.MODULES``).
"""
from .registry import MESHES, adapter, families, iter_episodes, mesh_status, register, require_meshes

__all__ = ["MESHES", "adapter", "families", "iter_episodes", "mesh_status", "register", "require_meshes"]
