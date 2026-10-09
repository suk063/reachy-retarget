"""Scene meshes of retargeted episodes: a content-addressed asset library and the extraction of
per-component visual and collision parts (source MuJoCo scenes, primitive scenes, Reachy links).

* :mod:`.library`: ``<out>/assets/<sha256>.<ext>`` (meshes as MuJoCo ``.msh``, texture files as
  recorded), written once, never overwritten.
* :mod:`.mujoco_scene`: components (rigid body groups) of a ``SceneRef`` with their parts,
  materials and textures, and per-frame component poses from scene joint positions.
* :mod:`.reachy_links`: Reachy 2 link visuals (COLLADA) and link poses from canonical ``q``.
* :mod:`.components`: the episode ``scene`` section (``schema.episode.SceneComponents``).

The reader side (no MuJoCo needed for arrays) is :mod:`reachy_retarget.schema.scene_assets`.
See docs/schema.md ("Scene components and asset library") and docs/design.md.
"""
