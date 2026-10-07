# Fixed-policy evaluation on 28 additional Lift/Stack sources

All 28 selected source episodes were executed with one predeclared policy. Nine
passed every existing manipulation gate and independent actuator replay under
the explicitly declared `global-contact-4ms` model. These are alternative-model
passes, not source-contact-model passes.

| Task | Independent source episodes | Alternative-model passes | Failures |
| --- | ---: | ---: | ---: |
| robomimic Lift | 21 | 8 | 13 |
| MimicGen Stack | 7 | 1 | 6 |
| Total | 28 | 9 | 19 |

The selection came from the previously declared 50-source baseline manifest:
Lift demos 3–23 and Stack demos 3–9. The same source episode's baseline and
follow-up count once. No unsuccessful source was removed, and no per-source
parameter or grasp geometry was selected from these outcomes.

The fixed policy was `contact005_velocity`, arm reference speed cap 0.7 rad/s,
servo target margin 0.03 rad, and global contact time-constant cap 0.004 s with
physics timestep 0.001 s. The declared velocity policy includes the existing
integral compensation and velocity feed-forward. Its gripper target uses the
already calibrated pad-contact angle minus 0.05 rad. There was no new grasp
attachment, base correction, object geometry/mass/friction change, source crop,
or gate adjustment. The contact model is explicitly a different declared
simulation protocol. Original source clocks and the derived monotone time maps
are preserved.

Before enqueueing, every parent was terminal and had a source-bound
`pad_translation` plan. All 28 were eligible. The scene, assets, normalized
source, plan clock, hand/object/gripper references and exact initial reset
inventory were checked. Immutable staging copied the numeric plan/input and
kept original assets as read-only references. The runtime was pinned to
`06930d7049d00981d9f130c655b32aa58854b021088febf6dc3aedef4b718b73`.

The final overlapping failed-gate counts were: final stability 12, no acquired
bilateral lifted grasp 11, carry contact 11, rotation drift 7, and measured arm
speed 1. A source can fail several gates. No contact-depth, self-clearance or
joint-margin gate failed this particular follow-up batch. This identifies the
remaining grasp/trajectory problems; it does not justify weakening the gates.

The batch used 24 existing worker pods, with an observed peak of 17 concurrent
claims for these 28 jobs. No pod was created, deleted or interrupted by this
evaluation. This is separate from concurrency in other simultaneously running
batches.

The [immutable policy manifest](../configs/additional-box-policy-v1.json),
[staging evidence](additional-box-policy-v1-staging.json),
[complete outcomes](additional-box-policy-v1-evidence.json), and
[archive audit](additional-box-policy-v1-archive-audit.json) preserve all source
identities, attempts and failed outcomes. Canonical state-only HDF5 files are in
`/mnt/reachy-retarget/datasets/{robomimic,mimicgen}/episodes/` with the
`additional-box-policy-v1-` prefix. Exact states, actual controls and camera
poses are retained; RGB is not stored.
