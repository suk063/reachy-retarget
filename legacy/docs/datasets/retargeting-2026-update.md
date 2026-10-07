# 2026 retargeting research and implementation decisions

The subsequent [contact-alignment review](../contact-alignment-reset.md) pauses
this broad optimization direction pending a controlled pad-offset comparison.
The literature remains relevant; implementing more of it is not evidence that
the original contact-frame problem has been solved.

Checked on 2026-10-06. This supplements the earlier
[methods survey](dynamics-aware-retargeting-methods.md). That survey missed
DynaRetarget and several recent interaction-retargeting papers. The change in
direction is to optimize an expanding physical trajectory, rather than relying
only on short receding horizons or a few grasp parameters.

Paper claims are not Reachy results. A geometric reference, a dynamic refinement,
a learned tracking policy and a hardware experiment are separate outcomes.
Publication year below distinguishes new 2026 preprints from updated older work.

| Work | Date/version checked | Mechanism and applicability |
| --- | --- | --- |
| [DynaRetarget](https://arxiv.org/html/2602.06827v3) | 2026; v3, June 10 | Progressively grows the optimization horizon and revisits earlier control knots. Uses MuJoCo native rollouts and CEM. This directly addresses irreversible early grasp mistakes in short-horizon planning. Its reported 1,024-sample, full-covariance optimizer and tracking thresholds differ substantially from our small CPU prototype. |
| [HOI-Retarget](https://arxiv.org/html/2609.34674) | September 2026 | Object-frame contact targets, palm orientation and overlapping trajectory windows improve interaction-preserving kinematic references. Useful contact representation; configurations are its decision variables, so this stage alone does not establish free-object dynamics. |
| [OTRetarget](https://arxiv.org/html/2609.36602) | September 2026 | Transfers surface interactions through optimal transport and jointly adjusts robot/object reference poses without rescaling the scene. Relevant to future contact correspondence and multi-object goals. Its optimized object references cannot be assigned to the moving object during our physical validation. |
| [TopoRetarget](https://arxiv.org/html/2606.16272) | June 2026 preprint | Uses sparse hand/object interaction graphs, directional relations and constrained Laplacian optimization with shared parameters. Supports the common-logic requirement. Its dexterous-hand references and downstream learned policies are distinct from a Reachy parallel-gripper controller. |
| [ObjRetarget](https://arxiv.org/html/2607.03828) | July 2026 preprint; project identifies IROS 2026 | Separates arm initialization/refinement from contact-aware hand geometry and synchronizes them through a temporal scheduler. Supports phase-aware control rather than copying finger angles. Its multi-finger polytope representation needs adaptation to two coupled Reachy fingers. |
| [Morphometric Imitation](https://arxiv.org/html/2609.28660) | v2, September 25, 2026 | Contact-preserving morphology optimization followed by residual RL and policy distillation. Useful distinction between geometric contact and dynamic success. A full implementation requires a training pipeline; the current optimizer does not reproduce that learning stage. |
| [SPIDER](https://arxiv.org/html/2511.09484v3) | First submitted November 2025; revised September 26, 2026 | General physics sampling across embodiments, with virtual-contact guidance in its method. Adopt physical control refinement, while keeping our attempts unassisted. The repository's September release distinguishes saved-state visualization from verified control replay. [Official code](https://github.com/facebookresearch/spider) |
| [Weave](https://arxiv.org/abs/2609.16683) | September 2026 | Contact-aware retargeting with approach completion and a learned whole-body controller. The approach must be evaluated together with the grasp; a post-grasp crop is not a complete task. This remains a method reference, not installed software. |
| [DexForge](https://arxiv.org/abs/2610.06331) | October 5, 2026 | Contact construction followed by force-aware differentiable refinement using reconstructed Gaussian object geometry. Relevant separation of initialization and dynamics, but its collision representation is not a drop-in replacement for the unchanged source meshes required here. |
| [SoftAct](https://soft-act.github.io/) | Project inspected in October 2026; publication date not independently established here | Functional, force-aware transfer to soft robots is relevant to preserving object effect across embodiments. Its compliant morphology and force representation need a separate model comparison; no soft-robot controller is imported. |

## What is implemented from these ideas

The implementation is an engineering adaptation, **not a reproduction of a
paper's algorithm or reported success rate**:

- [Source contracts](../../reachy_retarget/task_contracts.py) own dataset action
  conventions and publisher task predicates. Task names no longer select grasp
  yaw in the motion planner.
- [Interaction handling](../../reachy_retarget/interaction.py) derives a contact
  schedule, selects a collision region and grasp axis from geometry, and obtains
  an initial force from mass/friction with common bounds. Round mesh sections do
  not inherit an arbitrary compiler eigenframe. Actual support near the placement
  goal can trigger release through gripper commands.
- [Trajectory search](../../reachy_retarget/trajectory_search.py) grows the
  optimization horizon while keeping previous residual control knots adjustable.
  It uses native MuJoCo rollouts, diagonal CEM with a fixed budget, an incumbent
  and the original control sequence as alternatives. It does not implement the
  paper's full covariance, convergence-triggered horizon growth, skip cache or
  learned tracking policy.
- The common dense objective includes object translation/orientation, contact
  position in the object frame, grasp drift, measured finger force, penetration,
  actual joint speeds, margins and control regularity. The validator independently
  retains the stricter collision, drift, contact and source-goal gates.
- Every candidate starts from the same initial physical state. Only robot
  actuator commands vary. Object geometry, mass, friction and state evolution
  remain fixed inputs to the simulator. Candidate controls, physical states and
  scores are retained, including unsuccessful samples.
- [Common evaluation](../../reachy_retarget/common_benchmark.py) applies one
  frozen candidate bank and optimizer configuration to every supported episode.
  There is no task-to-force, task-to-yaw or task-to-budget table. Selecting a
  different candidate by physical outcome is part of the same algorithm.

The prototype remains limited to one active rigid object and the right gripper.
It explicitly rejects repeated grasps instead of silently taking the first one.
Passive hinge/slide fixtures can be preserved with explicit source joint states;
that does not yet provide a complete door-opening task adapter or controller.
The current ideal planar base still omits wheel traction and navigation planning.

## Evaluation requirements

Keep the current strict grasp gates. Do not adopt another paper's average object
tracking threshold as a replacement for physical grasp, collision or task success.
The local episodes have already been inspected and some previously tuned; a
shared configuration on them is a common-policy regression study, not a claim of
unseen-object generalization. Full source intervals and stationary pregrasp
windows must remain separately reported. Never count a search sample, alternate
robot embodiment or crop as another original demonstration.

For source-level implementation details used here, see MuJoCo's
[native rollout API](https://mujoco.readthedocs.io/en/stable/python.html#rollout)
and [contact sensor definition](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-contact).
Read-only sensors improve search diagnostics without attaching or driving objects.
