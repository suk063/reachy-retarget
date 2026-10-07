"""Embodiment-independent retargeting of source demonstrations onto Reachy 2 (docs/design.md)."""
from .config import RetargetConfig
from .pipeline import RetargetResult, retarget

__all__ = ["RetargetConfig", "RetargetResult", "retarget"]
