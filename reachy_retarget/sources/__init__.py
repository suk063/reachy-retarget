"""Source adapters: local source files -> :class:`~reachy_retarget.schema.source.SourceEpisode`."""
from . import bigym, dexmimicgen, libero, maniskill, robosuite  # noqa: F401  (registers bigym, dexmimicgen, libero, maniskill, robomimic, mimicgen)
from .registry import families, iter_episodes, register

__all__ = ["families", "iter_episodes", "register"]
