"""Embodiment-independent retargeting of source demonstrations onto Reachy 2 (docs/design.md)."""
from .config import RetargetConfig

__all__ = ["RetargetConfig", "RetargetResult", "retarget"]


def __getattr__(name):
    # The pipeline (grasp labels, placement, gaze, ...) is imported on first use, so modules that only
    # need the embodiment-level pieces (wbik, timing; e.g. reachy_retarget.tracking) do not load it.
    if name in ("RetargetResult", "retarget"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
