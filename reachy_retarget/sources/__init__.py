"""Source adapters: local source files -> :class:`~reachy_retarget.schema.source.SourceEpisode`."""
from . import robosuite  # noqa: F401  (registers robomimic, mimicgen)
from .registry import families, iter_episodes, register

__all__ = ["families", "iter_episodes", "register"]
