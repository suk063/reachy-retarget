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
* Kinematic (K) and physics (P) validation tiers are reported separately and never
  merged. Physics rollouts never weld, teleport or overwrite moving-object state.
* Save failed retargets and failed rollouts with their reasons as well as successes.
* Do not count overlapping versions, crops, mirrors or generated variants of the same
  seed as independent demonstrations; record lineage instead.
* `legacy/` is read-only reference material from the previous project phase. Do not
  import it from new code.
* Never delete or recreate cluster pods, Deployments or nodes.
