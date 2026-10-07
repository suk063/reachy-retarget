# ManiSkill PickCube adapter evidence

The bounded local sample is `haosulab/ManiSkill_Demonstrations` revision
`d674485bbffdd533914e52d272fdda34c0515608`,
`demos/PickCube-v1/teleop/trajectory.h5` and its `trajectory.json` companion.
The JSON identifies the simulator source commit as
`ecc579b7567c31bb3d7176539d90e86ae49296be`, with `obs_mode=none` and
`control_mode=pd_joint_pos`. Consequently the file has no observed TCP channel.
The adapter derives TCP poses from the saved simulator articulation state;
it never treats the commanded joint targets as measured poses. The original
[trajectory metadata](https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations/blob/d674485bbffdd533914e52d272fdda34c0515608/demos/PickCube-v1/teleop/trajectory.json)
and all acquired payload checksums are retained in the local ledger.

## Verified state and kinematic contracts

- Actor state is position, scalar-first quaternion, linear velocity and angular
  velocity: 13 scalars. The cube uses `env_states/actors/cube`; the goal marker
  and table remain separately preserved environment/source state. This mapping
  follows the pinned [Actor.get_state implementation](https://github.com/haosulab/ManiSkill/blob/ecc579b7567c31bb3d7176539d90e86ae49296be/mani_skill/utils/structs/actor.py#L111-L131)
  and [Pose inverse/conversion implementation](https://github.com/haosulab/ManiSkill/blob/ecc579b7567c31bb3d7176539d90e86ae49296be/mani_skill/utils/structs/pose.py#L124-L140).
- Panda articulation state contains those root-state fields followed by nine
  measured positions and nine velocities, totaling 31 scalars. The adapter
  preserves every field and derives the world TCP using the recorded root
  transform at each frame. See the pinned
  [articulation serializer](https://github.com/haosulab/ManiSkill/blob/ecc579b7567c31bb3d7176539d90e86ae49296be/mani_skill/utils/structs/articulation.py#L174-L197).
- The official [Panda agent](https://github.com/haosulab/ManiSkill/blob/ecc579b7567c31bb3d7176539d90e86ae49296be/mani_skill/agents/robots/panda/panda.py#L33-L54)
  names seven arm joints, two finger joints and `panda_hand_tcp`. Its matching
  [Panda URDF](https://github.com/haosulab/ManiSkill/blob/ecc579b7567c31bb3d7176539d90e86ae49296be/mani_skill/assets/robots/panda/panda_v2.urdf)
  is acquired and checksum-verified through the ledger. No mesh files are
  required to compute kinematics. The adapter composes named URDF joints using
  NumPy/SciPy, then checks the complete resulting transforms independently with
  Pinocchio when available. It imports neither ManiSkill nor SAPIEN.
- Simulator qpos arrays do not carry joint names. For this exact teleoperation
  export, all ten trajectories start with the documented seven-joint absolute
  command equal to the first seven measured joint positions (maximum difference
  0 radians). The adapter requires this check and records the inferred ordering;
  it rejects other exports that do not satisfy it. Finger positions are kept
  in their source order; their sum yields a measured aperture, without claiming
  a verified Reachy motor-angle mapping.
- T transitions retain T actions and T+1 complete state frames, including the
  terminal state. `source/action_time_s` records the separate action clock.
  No final action is duplicated or invented. The documented
  [trajectory layout](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html)
  distinguishes these two lengths. The pinned
  [simulation config](https://github.com/haosulab/ManiSkill/blob/ecc579b7567c31bb3d7176539d90e86ae49296be/mani_skill/utils/structs/types.py#L70-L78)
  supplies 20 Hz control unless the trajectory metadata explicitly overrides it.

The cube dimensions are the source's 20 mm half-size. The source goal is a
noncolliding visual marker; success means cube position near that goal and a
stationary robot, not necessarily releasing the cube onto a support. These
semantics are preserved from the pinned
[PickCube environment](https://github.com/haosulab/ManiSkill/blob/ecc579b7567c31bb3d7176539d90e86ae49296be/mani_skill/envs/tasks/pick_cube.py#L16-L87).
The source success flag is not Reachy success.

## Local checks and limits

The first five trajectories contain 541 complete state frames. Independent
NumPy/Pinocchio world TCP transforms agree below `9e-16` maximum element error.
Synthetic unit tests additionally rotate and translate the root, vary the
measured arm independently of its command, preserve the terminal state, and
reject a wrong source revision, truncated state sequence or unverifiable arm
ordering. Tests do not fetch any data.

The original tabletop origin places the cube near z=0.02 m. Any Reachy placement
must apply one rigid transform to both the hand and objects; the normalizer
preserves the original world. Grasp forces, Reachy contact feasibility and
release stability remain unvalidated by this adapter. A source-derived FK
channel and a Reachy kinematic pass are distinct from physical manipulation.
Machine-readable FK checks are in `runs/coverage/maniskill-fk-check.json`;
the final cross-source audit is generated by `reachy_retarget.assessment`.

Measured Panda finger positions and source gripper commands remain in the
normalized data. The current retargeter does not convert this ManiSkill
controller's gripper channel into Reachy finger targets. Its five kinematic
passes therefore cover arm/base motion with preserved object references;
they do not establish a complete Reachy pick-and-place execution.
