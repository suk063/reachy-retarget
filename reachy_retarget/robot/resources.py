"""Packaged Reachy 2 assets (copied from reachy-agent@d6d5e9f, robot/assets), parsed once."""
import functools
import json
import tomllib
from pathlib import Path

from .urdf import URDF

ASSETS = Path(__file__).resolve().parent / "assets"
URDF_FILE = ASSETS / "reachy.urdf"
URDF_SHA256 = "63a1ecab3312a72c35f19a4bb1142005d74e4d7101da87822d02fea96696d448"


@functools.cache
def urdf():
    """The calibrated Reachy 2 URDF (tripod extension baked into its origin)."""
    return URDF(URDF_FILE)


@functools.cache
def profile():
    """robot.toml of the source robot (r2-0008): base radius, postures, camera calibration."""
    return tomllib.loads((ASSETS / "robot.toml").read_text())


@functools.cache
def collision_spec():
    """collision_spheres.json: per-link spheres [x, y, z, r] and the SRDF-derived disabled pairs."""
    return json.loads((ASSETS / "collision_spheres.json").read_text())
