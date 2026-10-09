# Project instructions

Goal: acquire as many and as diverse object-interaction demonstrations as possible and
retarget them onto Reachy 2 as state-only training data for manipulation policies.

* Keep source and documentation in English.
* Never import a robot SDK or connect to a robot. This project is offline with respect
  to hardware.
* Network access happens only through explicit `fetch` commands, never on import or in
  unit tests. Record source URLs, revisions, checksums, licenses and lineage.
* Stop new transfers before disk free space falls below 50 decimal GB.
* Never store images. Store robot state, object state and every control view needed
  to derive any control mode later.
* Store the scene meshes of every episode: per component (task objects, supports, receptacles,
  fixtures, articulated parts, Reachy links) its visual and collision parts with materials and
  texture files, in the content-addressed asset library `<out>/assets/`, plus per-frame component
  poses. Texture files are mesh assets, not observations; rendered images are never stored.
* Exclude datasets whose meshes cannot be obtained: `sources.registry.MESHES` is the one place
  that says which families provide meshes; the build refuses the others, and episodes whose
  scene meshes cannot be resolved are not written (`excluded: no_meshes`).
* Kinematic (K) and physics (P) validation tiers are reported separately and never
  merged. Physics rollouts never weld, teleport or overwrite moving-object state.
* Save failed retargets and failed rollouts with their reasons as well as successes.
* Do not count overlapping versions, crops, mirrors or generated variants of the same
  seed as independent demonstrations; record lineage instead.
* `legacy/` is read-only reference material from the previous project phase. Do not
  import it from new code.
* Never delete or recreate cluster pods, Deployments or nodes.
