# Object-inclusive data and retargeting research

Checked on **2026-10-06**. The latest parallel investigations cover **37 mobile
manipulation resource profiles**, **12 mobile human–object profiles**, and
**11 dynamics-aware retargeting methods**. They distinguish demonstration
collections, task generators, scene assets, evaluation systems and research
methods. These counts overlap existing families and releases: they must not be
added together or described as 49 independent datasets.

| Latest report | Coverage | Evidence and decisions |
| --- | --- | --- |
| [2026 retargeting update](retargeting-2026-update.md) | Recent contact and dynamics methods, including DynaRetarget and HOI-Retarget | Expanding-horizon optimization, implementation choices and differences from paper reproductions |
| [Common retargeting and integration review](common-retargeting-review.md) | Shared logic requirements and nine named resource candidates | Current code blockers, native MuJoCo priorities, adapter boundaries and acceptance criteria; review only, with no new physical results |
| [Mobile manipulation survey](mobile-manipulation-survey.md) | 37 robot-data, simulator, task and asset profiles | Base pose versus velocity/action, gripper state, object trajectories, geometry, source grasp mechanisms, release status, licenses, sizes and small acquisition options |
| [Mobile human–object survey](mobile-human-object-survey.md) | 12 profiles, plus rechecks of existing families | Carrying, pushing, locomotion, measured versus derived forces, object poses/meshes, access gaps and recording lineage |
| [Dynamics-aware retargeting methods](dynamics-aware-retargeting-methods.md) | 11 methods | Object-relative grasp selection, base placement, actuator optimization, contact objectives, release status and unassisted validation requirements |
| [Mobile robot evidence](mobile-manipulation-survey-evidence.json) | Primary URLs, code/data revisions, metadata hashes and publisher file listings | Download sizes and declared payload checksums are separated from actual acquisition and verification |
| [Human/mobile methods evidence](mobile-methods-evidence.json) | Inspected endpoints, revisions, response hashes and failed checks | Public documentation, a code release and an accessible demonstration archive remain separate claims |
| [Dynamics validation](../dynamics-validation.md) | Executed Reachy MuJoCo attempts and their limitations | Current physical results, saved failures, source-preserving scene conversion and independent actuator replay |

The surveys read primary documentation, papers, source code and public file
metadata. Research access does not establish payload download, an implemented
adapter, feasible Reachy motion or physical success. For executed results, use
the dynamics validation report and its run artifacts rather than a survey's
ranking or an upstream success rate.

## Acquisition and implementation priorities

These priorities are engineering judgments supported by the detailed reports.
Keep the Reachy gripper and each selected object's geometry and physical
properties fixed; improve grasp region, approach, closure, control and timing.

1. **Start mobile simulation work with native RoboCasa365.** Preserve the
   per-episode model, state and metadata in its native extras. Reduced training
   mirrors may omit those fields. Reconstruct base and object state from the
   saved model, then independently validate Reachy contact and carry motion.
   BiGym supplies another compact native MuJoCo route, with floating-base and
   replay-success limitations that require explicit handling.
2. **Use BEHAVIOR 2026 for broad household coverage, with its grasp mechanism
   disclosed.** Raw simulator state is more informative than reduced exports,
   but its collection and evaluation use assisted grasping. Source success is
   not a Reachy friction-contact result. Pin the corrected state revision and
   simulator version, resolve asset terms, and start with a rigid-object task.
3. **Use MoMaGen's released code as a mobile planning reference.** The public
   repository and installation documentation supersede the older forthcoming
   notice. They do not establish that a downloadable source-demonstration
   package is available. Its base reachability, visibility and object-relative
   subtask planning are useful; the OmniGibson/BEHAVIOR stack and grasp settings
   need a separate audit before transfer to MuJoCo.
4. **Inspect M3Bench for base/arm trajectories and scene geometry.** It provides
   robot and scene assets plus initial object configuration. A dense moving
   object trajectory and a friction-valid grasp are not established by its
   kinematic trajectory checks. Account for large archives before acquisition.
5. **Treat real mobile collections as state reconstruction work where needed.**
   Galaxea documents observed chassis positions, velocities and IMU; Mobile
   ALOHA's inspected writer records base velocity without a verified global
   base pose. Neither supplies general per-frame object 6D state and matched
   dynamics assets in the inspected schema. AIRoA's wrench histories add useful
   evidence, but action/base frames, freshness and release-specific terms still
   matter. Do not turn commands or perception estimates into measured labels.
6. **Prioritize MeLLO, FORCE, HIMO and object-aware KIT recordings for human
   mobility and interaction.** MeLLO contributes locomotion with measured
   interaction signals, while rigid poses and matching collision meshes still
   require work. FORCE has object motion and meshes, but its force encoding is
   derived and added resistance is not total object mass. HIMO offers explicit
   multi-object tracks behind an application process. KIT supports per-motion
   object-aware acquisition; imported body-only CMU recordings remain outside
   scope. Preserve each collection's frame, clock, access and lineage caveats.
7. **Refine actuator controls in the actual validation model.** Use the methods
   survey to select object-relative seeds, contact phases, base poses and short
   control horizons. Final evaluation must advance free-object dynamics with
   robot controls only. Saved-state playback, assisted optimization and a
   completed IK solve are distinct from a successful physical rollout.

## Earlier investigations and baseline results

The following reports remain useful and retain their original scope. Their
19 human-release and 19 robot-family profiles also overlap newer reports,
mirrors and derivative releases; the historical total of 38 profiles was never
an independent-dataset count.

| Earlier report | Recorded scope |
| --- | --- |
| [Human–object survey](human-object-survey.md) | 19 candidate releases, with object/hand semantics, contact labels, geometry, access and adapter details |
| [Robot–object survey](robot-object-survey.md) | 19 families/benchmarks, including native-state versus reduced-export checks and ranked acquisition routes |
| [Human survey evidence](human-object-survey-evidence.json) | Checked URLs, response hashes, revisions and archive-access limitations |
| [Previous coverage audit](../coverage-audit.md) | The earlier locked matrix of 29 source episodes from four dataset IDs, including normalization, kinematic checks, physical attempts and deduplication |

The original catalog snapshot contained 333 source/mirror entries. Historical
catalog acquisition totals did not establish local payload availability. The
previous validation stage explicitly fetched a small locked selection from
robomimic, MimicGen, ManiSkill and HUMOTO; that acquisition is separate from the
surveys.

### Previous physics baseline

The earlier matrix's dedicated dynamic tests focused on Can. One of three
selected Can episodes passed the unchanged tuned grasp settings; the others
exposed contact retention or actual-velocity failures. The first episode was a
tuning example, so that result was a development baseline rather than an
unbiased held-out success estimate. Other matrix tasks had selected kinematic
passes without general physical validation at that stage. These are historical
results; the [dynamics validation report](../dynamics-validation.md) records the
subsequent multi-object work and its current limitations.

The previous verification record reported 113 passing tests and matching
checksums for 23 locked downloaded files. Its resampling audit checked 59 object
tracks in 29 matrix episodes under each recorded scene placement and time
scaling, treating quaternion signs as equivalent. Those counts describe that
baseline, not the current test suite or physical success coverage.

## Scope, provenance and counting

Direct object-aware retargeting requires time-aligned finite object poses,
valid unit quaternions, explicit coordinate conventions and suitable geometry.
Names, captions, videos, base commands and static scene placements alone do not
meet that contract. A missing implemented adapter means unsupported locally;
it does not prove that an upstream collection has no object data.

A mobile claim additionally requires meaningful base displacement or a
navigation-to-manipulation transition. Measured base pose, measured velocity,
commanded displacement, reconstructed odometry and a human-derived base proxy
must remain distinguishable. Long clips or a mobile-capable robot alone do not
establish mobile task coverage.

Count original captures once across raw/processed files, crops, cameras,
annotations and mirrors. Record synthetic descendants and their seeds;
primitive actions and nested subsets are not independent complete tasks.
Scene assets, problem definitions, policy weights and generated rollouts have
different provenance from human demonstrations.

Kinematic exports preserve source-derived object trajectories under the same
scene transform and timing as the hand goals. Physical attempts initialize an
object once and then evolve it through dynamics. Never weld, teleport or
replace its state during physical validation; save failed attempts as well as
successes. Playback of recorded physical state is a visualization operation,
not a new control-replay or contact validation result.

The previously inspected robot-only Reachy numeric selection was deleted,
including its raw, normalized and derived copies. Body-only CMU, LaFAN1 and
AMASS selections are excluded from acquisition; this does not exclude separate
object-aware derivatives. Removal manifests preserve URLs, revisions, hashes
and deletion scope. Persistent exclusions survive rediscovery. Sibling
repositories and object-inclusive source payloads are preserved.

That cleanup removed 82 dataset/output/cache paths plus four standalone
robot-only previews. These included hardlinks and cached copies, so the totals
are neither independent-payload counts nor recovered disk space. Historical
machine-local evidence is recorded in `runs/coverage/final-verification.json`,
`object-resampling-audit.json`, `removals.json` and `pytest.xml`; generated data
and run artifacts remain outside Git.

For commands, see the [project README](../../README.md). The earlier reproducible
selection is [object-matrix-lock.jsonl.gz](../../configs/object-matrix-lock.jsonl.gz).
Stop new transfers before free disk space falls below 50 decimal GB, allowing
for extraction and derived outputs as well as the download itself.
