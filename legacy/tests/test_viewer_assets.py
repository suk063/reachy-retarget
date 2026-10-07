"""Display repairs must preserve the source model and physical geometry."""

import xml.etree.ElementTree as ET

import pytest
import numpy as np
import trimesh

pytest.importorskip("mujoco")

from reachy_retarget import viewer_assets


def test_visual_support_repair_preserves_source_joints_and_collisions(tmp_path, monkeypatch):
    control = tmp_path / "control"
    asset = control / "asset"
    asset.mkdir(parents=True)
    original = '''<robot name="fixture">
      <link name="back_bar_inner">
        <visual><origin xyz="0 0 .2" rpy="0 0 0"/><geometry><box size=".03 .03 .4"/></geometry></visual>
        <collision><origin xyz="0 0 .2"/><geometry><box size=".03 .03 .4"/></geometry></collision>
        <inertial><mass value="1"/></inertial>
      </link>
      <joint name="torso_base" type="fixed"><origin xyz="0 0 .635"/><parent link="back_bar_inner"/><child link="torso"/></joint>
      <link name="torso"/>
    </robot>'''
    source = asset / "reachy.urdf"
    source.write_text(original)
    monkeypatch.setattr(viewer_assets, "CONTROL", control)
    out = tmp_path / "viewer"
    out.mkdir()
    result, manifest = viewer_assets.visual_urdf(out)
    before = ET.fromstring(original)
    assert source.read_text() == original
    for query in ("joint", "link/collision", "link/inertial"):
        assert [ET.tostring(x) for x in result.findall(query)] == [ET.tostring(x) for x in before.findall(query)]
    assert result.find("link/visual/geometry/box").get("size") == "0.03 0.03 0.635"
    assert result.find("link/visual/origin").get("xyz") == "0 0 0.3175"
    assert manifest["source_urdf_sha256"] != manifest["visual_urdf_sha256"]


def test_front_support_reaches_torso_surface_and_preserves_lower_endpoint(tmp_path):
    torso = tmp_path / "torso.stl"
    trimesh.creation.box(extents=(.4, .4, .2)).export(torso)
    root = ET.fromstring(f'''<robot><link name="torso"><visual><geometry><mesh filename="{torso}"/></geometry></visual></link>
      <link name="left_bar_inner"><visual><origin xyz="0 0 -.2"/><geometry><box size=".03 .03 .4"/></geometry></visual></link>
      <joint name="left_bar_joint_mimic" type="revolute"><origin xyz="0 0 -.3"/>
        <axis xyz="0 1 0"/><mimic joint="tripod_joint" offset="0"/></joint></robot>''')
    joint_before = ET.tostring(root.find("joint"))
    changes = viewer_assets.extend_front_supports(root)
    visual = root.find("link[@name='left_bar_inner']/visual")
    z = np.fromstring(visual.find("origin").get("xyz"), sep=" ")[2]
    length = np.fromstring(visual.find("geometry/box").get("size"), sep=" ")[2]
    assert z-length/2 == pytest.approx(-.4)
    assert -.3+z+length/2 == pytest.approx(-.098)
    assert changes[0]["surface_distance_m"] == pytest.approx(.2)
    assert ET.tostring(root.find("joint")) == joint_before
