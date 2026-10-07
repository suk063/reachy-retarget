"""Source adapters: local source files -> :class:`~reachy_retarget.schema.source.SourceEpisode`."""
from . import maniskill, robosuite  # noqa: F401  (registers maniskill, robomimic, mimicgen)
from .registry import families, iter_episodes, register

__all__ = ["families", "iter_episodes", "register"]
