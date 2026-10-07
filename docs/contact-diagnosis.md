# Contact diagnosis of the 20 original physical attempts

The `legacy-all-v1` batch remains **0/20 complete physical passes**. Initial
admission diagnostics and controller subgates below do not remove any episode
from that denominator. New development trials are separate attempts on the same
source episodes, not additional demonstrations.

## Reproducibility and controller findings

`contact_diagnosis.diagnose()` independently replays saved actuator controls
from one recorded reset. It does not write object state after reset, modify a
model, or amend original results. Exact qpos/qvel replay was confirmed for all
nine current Can/Lift/Stack attempts, four historical passing attempts, and the
new Stack `contact_010` attempt. Maximum errors were zero in the same simulator.
All nine current initial qvel vectors and initial servo-position errors were
zero; the largest first-command discrepancy was 1.4e-17. This does not support a
recording/reset regression as the cause of their failures.

The old 6/20 experiment and current 0/20 experiment changed several factors.
Historical settings used an object-relative path, mobile planning, slower
playback, force control, and closure holds. Lift/Stack additionally started at
source time 0.25 s. Current pad-only runs preserve the complete source clock and
disable force/integral feedback. The two aggregate rates cannot isolate the
effect of one change.

| Example | Current pad-only behavior | Historical passing behavior |
| --- | --- | --- |
| Lift demo 0 | Full-close target -0.06 rad; 3.623 mm penetration; summed pad force peaks approximately 147/154 N | Force-regulated target at least 0.80 rad; 0.620 mm penetration; approximately 14.7 N peak per pad |
| Stack demo 2 versus historical demo 0 | Full-close target -0.06 rad; 3.752 mm penetration; approximately 160/155 N force peaks | Target at least 0.896 rad; 0.457 mm penetration; approximately 5.7 N force peak |

These are different attempts, not a controlled proof that force control alone
causes the entire improvement. They identify overclosure as a concrete failure
mechanism. In the subsequent Stack demo 2 `contact_010` attempt, hand penetration
decreased to 0.760 mm and the remaining failure was object/environment contact.
Its later cubeA/cubeB impact reached 5.068 mm at 3.848 s, separately from the
initial table impact.

Can requires more than a force adjustment. At source closure, actual TCP
position error was approximately 224/52/85 mm for demos 0/1/2. Demo 0 commanded
closure at 2.60 s but first touched the Can at 4.488 s; demo 2 closed at 2.95 s
and first touched at 3.802 s. Only one pad touched in those two trials; demo 1
never touched the Can. Missing drift anchors are reported as unavailable,
not interpreted as zero slip. The small translation calibration cannot by
itself compensate such large arm tracking errors.

Evidence:

- [20 original summaries and nine detailed contact replays](contact-diagnosis-audit.json)
- [Four historical exact replays](contact-diagnosis-historical.json)
- [New Stack contact replay and numerical convergence](passive-contact-convergence.json)

## Unchanged-model initial admission

`passive_admission()` holds the exact recorded initial robot actuator targets
for 0.15 s. Object poses, velocities, geometry, mass, friction, contact material,
solver options, and gates remain unchanged. This is a separate short experiment,
not a replay claim for rows beyond a stopped recording. No robot/object contact
occurred in any of these 20 probes, so the measured object impacts are isolated
from robot intervention.

| Sources | Count | Initial support condition | Peak passive object penetration | Other initial blocker |
| --- | ---: | --- | ---: | --- |
| Can | 3 | Already resting, approximately 0.297 mm overlap | 0.297 mm | None found in the short prefix |
| Lift | 3 | 10 mm above table | 3.236 mm at 0.064 s | None |
| Stack | 3 | 10 mm above table | 3.236 mm at 0.064 s | None |
| Square | 3 | 60 mm above table | 7.842 mm at 0.130 s | None |
| Threading | 3 | Handle already touching table | 0.171 mm | None found in the short prefix |
| ManiSkill PickCube | 5 | Cube already resting | 0.107 mm | Robot back-bar/table intersection: 174.79 mm |

Nine inputs therefore have an isolated passive source/model gate failure.
Eleven pass the object-only short prefix; five of these have an invalid initial
robot placement. Only six have neither diagnosed startup blocker. These are
necessary startup observations, not six task successes or a new success-rate
denominator. Robot placement can be reconsidered without changing an object;
later contact/kinematic feasibility must still be tested.

The support-gap diagnostic uses signed closest-point distance between original
collision geoms and the gravity direction. A zero-distance contact uses its
actual reset contact normal. This correctly recognizes Threading's touching
handle instead of mistaking the elevated needle shaft for the support point.
The [MuJoCo distance API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-geomdistance)
returns the signed geom distance and closest points; it does not replace the
recorded contact validator.

All per-input hashes, support pairs, initial contacts, peak pairs, times, and
unchanged original gates are in [passive-source-admission.json](passive-source-admission.json).

## Source evidence and numerical refinement

Original Lift and Stack simulator states already contain startup penetration.
Both start with a 10 mm support gap and zero recorded qvel. Original source
poses imply approximately 2.101 mm penetration at 0.05 s and 2.588 mm at 0.10 s.
The converted cube/table retain the original `solref="0.02 1"` and
`solimp="0.9 0.95 0.001 0.5 2"`, dimensions, masses, and initial poses. The
source XML, normalized source checksums, raw clock and gap values are preserved
in [passive-contact-convergence.json](passive-contact-convergence.json).

Reducing only the numerical timestep does not satisfy the unchanged 2 mm gate:

| Timestep | Lift peak | Stack peak |
| --- | ---: | ---: |
| 2 ms | 3.236 mm | 3.236 mm |
| 1 ms | 2.978 mm | 2.978 mm |
| 0.5 ms | 3.048 mm | 3.048 mm |
| 0.25 ms | 3.180 mm | 3.180 mm |

Increasing solver iterations from 100 to 500 and tightening tolerance from
1e-8 to 1e-12 changes these depths by less than 0.00014 mm. All 16 probes retain
the identical initial state and source contact properties, with no robot/object
contact in the first 0.15 s. Changing only numerical fidelity cannot cure this
soft-contact-model incompatibility. Cropping the source, moving the object,
or weakening the threshold would conceal it and is not done.

## Explicitly different global compliance sensitivity

A separate diagnostic caps **every** positive geom/contact-pair `solref` time
constant at 0.004 s, preserving damping ratios, `solimp`, friction, geometry,
mass, inertia, gravity, and initial states. It uses 1 ms and 0.5 ms timesteps
and no global contact override. This changes contact compliance; it is **not**
a source-faithful model or a promoted validation result.

| Task family | Peak with 1 ms timestep | Peak with 0.5 ms timestep |
| --- | ---: | ---: |
| Lift / Stack | 0.448 mm | 0.520 mm |
| Square | 1.121 mm | 1.297 mm |
| Can | 0.297 mm | 0.297 mm |
| Threading | 0.0071 mm | 0.0071 mm |
| ManiSkill PickCube | 0.0044 mm | 0.0044 mm |

All 40 short probes completed. The compiled model comparison checked 479 other
array fields for exact equality and saved their aggregate checksum; damping
ratios were checked separately. The saved per-geom/pair before/after `solref`
arrays make the alternative model reviewable. Initial qpos/qvel/ctrl checksums
and original scene/replay hashes are retained. The 174.79 mm ManiSkill robot
placement collision remains. These results support evaluating a separately
declared common contact model, not silently modifying the source-fidelity
results or claiming whole-task success.

Evidence: [global-contact-compliance-sensitivity.json](global-contact-compliance-sensitivity.json).

## Reproduction and tests

```python
from reachy_retarget.contact_diagnosis import (
    diagnose, passive_admission, passive_prefix_convergence,
    global_compliance_sensitivity,
)

contact_report = diagnose(attempt_path)
initial_hold = passive_admission(attempt_path)
numerical_only = passive_prefix_convergence(attempt_path)
different_contact_model = global_compliance_sensitivity(attempt_path)
```

Every function returns a report and leaves the input files unchanged. Seven
offline tests cover real actuator replay, contact classification, checksum
rejection, missing drift, constant-hold admission, touching support geometry,
numerical variants, and alternative-model invariants. No SDK or dataset network
access is used. Shared dynamics, physics, and pad-alignment code was not edited
by this audit.
