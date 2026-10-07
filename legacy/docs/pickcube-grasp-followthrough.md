# PickCube grasp follow-through and closure admission

The 36 development work items are repeated experiments on **one** source
demonstration, not 36 independent demonstrations. Center-only attachment failed
IK admission in all 12 policy variants. Axis-only and centered-plus-axis
attachments each completed 12 physical attempts; none passed every gate.
Every attempt and the original source were retained.

## Closest source-fidelity result

The centered-plus-axis attachment with `contact_010` passes all physical gates
except translation and rotation drift. Its measured drift is 6.487 mm and
17.980 degrees, hand/object penetration 0.775 mm, bilateral carry fraction 1.0,
final stable duration 1.98 s, and minimum joint margin 30.676 mrad. The original
grasp had 8.549 mm / 21.817 degrees drift under a different control/timing
policy, so that comparison is not a controlled estimate of the geometry effect.

The more directly matched geometry comparison is axis-only versus both at
`contact_010`: 27.783 versus 17.980 degrees. The local robot retiming differs
slightly because the newly solved joint paths differ. Stronger closure
`contact_020` reduces drift to 14.347 degrees but exceeds the unchanged 1 mm
penetration gate at 1.104 mm.

| Centered-plus-axis control | Drift rotation | Hand penetration | Minimum joint margin |
| --- | ---: | ---: | ---: |
| No integral or velocity compensation | 17.980 deg | 0.775 mm | 30.676 mrad |
| Integral compensation | 44.922 deg | 0.679 mm | 13.749 mrad |
| Integral plus velocity feedforward | 40.281 deg | 1.131 mm | 30.814 mrad |

The velocity variants also enable integral compensation; their effects cannot
be attributed to velocity feedforward alone. The integral variant commands
`r_wrist_roll` within 7.607 mrad of its lower limit at 4.08 s, despite a 31 mrad
reference margin. Its measured margin reaches 13.749 mrad at 4.34 s. A general
summed-target envelope with directional anti-windup, or greater planned margin,
addresses this controller problem without changing the gate.

## Contact and tracking evidence

Four independent actuator replays reproduced qpos and qvel exactly. The
centered `contact_010` attempt has 7 loaded contact points at acquisition and 8
at the final interval. MuJoCo 3.15.0 has `disableflags=0`, so native CCD and
multi-contact CCD are already enabled. The largest right-finger mimic equality
residual is 0.000852 rad, approximately 0.049 degrees. This is not evidence of a
missing multi-contact switch or a large mimic-joint disconnection.

At the 3.05 s drift anchor, both pads carry about 13.2 N of normal force.
Force-weighted centers lie approximately 17–19 mm in the cube's positive X
direction, close to its 20 mm edge. The actual collision-pad midpoint is also
offset: `[9.542, 0.018, 3.154]` mm in the object frame while its planned value
is essentially zero. TCP tracking error is 12.053 mm at acquisition and peaks
near 30 mm. Thus a central nominal reference does not establish central
measured contact.

The commanded closure begins at control time 2.05 s. First penetrating contact
is recorded at interval end 2.28 s and full bilateral contact at 2.30 s. The
planned pad midpoint enters the 1 mm central region by approximately 2.10 s,
before those contacts. Early command timing alone therefore does **not** prove
the cause of the acquired offset. Planned reference lift reaches 1 mm at
approximately 2.53 s and 5 mm at 2.67 s, leaving room to test a brief central
dwell before closure without changing the hand path or source interval.

An additional exact replay resolved the reference-frame ambiguity. At the
first robot/object force (2.278 s), the physical cube has essentially zero X
displacement; at first bilateral force (2.286 s), it has moved only 0.009 mm in
X. There are no open-intent or non-distal robot/object force events. At the
2.28 s interval end, the planned pad midpoint is 8.61 mm in X relative to the
**original reset** object frame, even though it is near zero relative to the
**moving source reference** object frame. Actual pad X is 9.24 mm and TCP
tracking error in X is only 0.58 mm. The persistent acquisition offset is
therefore predominantly source pregrasp object motion that the Reachy rollout
did not reproduce. Geometry-gated closure in the moving reference frame alone
cannot be assumed to remove it. A subsequent robot-only constant translation
can test alignment to the unchanged reset object, while retaining the original
source object trajectory for evaluation and requiring fresh full-path IK.

The full 1.9–2.5 s timeline, both object frames, first-force contacts, and exact
replay checks are in [pickcube-grasp-acquisition-audit.json](pickcube-grasp-acquisition-audit.json).
`reachy_retarget.grasp_acquisition.diagnose` generated that report with no
object assignment after reset.

Evidence: [four replay audits](pickcube-grasp-followthrough-audit.json) and
[pad midpoint comparison](pickcube-pad-midpoint-audit.json). The latter retains
its exact read-only replay script and input hashes. The reusable
`reachy_retarget.grasp_followthrough.diagnose` returns loaded contact positions,
normals, forces, mimic errors, target/actual margins, and replay agreement.

## Geometry-gated closure experiment

```python
from reachy_retarget.grasp_closure import apply

prepared = apply(
    prepared, output,                 # after existing robot-clock retiming
    response_time_s=0.25,             # explicit estimate from the prior attempt
    center_tolerance_m=0.003,
    settle_s=0.05,
    max_reference_lift_m=0.005,
    response_evidence=prior_response_evidence,
)
```

The function delays only robot gripper intent. It requires a continuous
preceding reference dwell within the selected pad-center region, continued
centrality throughout the declared actuator response window, and acquisition
before the original release and declared reference-lift bound. It currently
supports one contiguous negative contact-intent phase. The response estimate
is required explicitly and is an experiment assumption, not a guarantee of
the next physical response.

All six declared combinations passed admission against the actual corrected,
retimed plan:

| Center tolerance | Dwell | Derived command time | Delay | Estimated contact |
| --- | ---: | ---: | ---: | ---: |
| 1 mm | 20 ms | 2.12 s | 70 ms | 2.37 s |
| 1 mm | 50 ms | 2.15 s | 100 ms | 2.40 s |
| 1 mm | 100 ms | 2.20 s | 150 ms | 2.45 s |
| 3 mm | 20 ms | 2.10 s | 50 ms | 2.35 s |
| 3 mm | 50 ms | 2.13 s | 80 ms | 2.38 s |
| 3 mm | 100 ms | 2.18 s | 130 ms | 2.43 s |

Reference lift at estimated contact is 0.381–0.444 mm. The complete hand/object
references, clock, original intent, derived intent, center error, and lift
diagnostics are saved in `closure-intent.npz`. A rejected plan saves its report
and artifact before raising; it is never counted as physically tested.
Admission evidence is in [grasp-closure-admission.json](grasp-closure-admission.json)
and `/mnt/reachy-retarget/workspaces/grasp-closure-admission-v1`.

Four offline tests cover original-input preservation, positive dwell, the full
response window, insufficient prelift/release time, explicit gravity, rejected
artifact preservation, and unsupported multiple grasp intervals. Physical
results for these new closure variants are pending separately.

## Secondary contact-model questions

MuJoCo's official documentation distinguishes multi-point detection from
torsional contact friction, and explains that soft friction constraints can
creep even inside the friction cone. Native/multiple CCD is already active
here. Changes to impedance ratio or NoSlip must be evaluated as explicitly
declared solver counterfactuals with unchanged source-fidelity results; this
audit changed neither. [Multiple contacts and contact dimensions](https://mujoco.readthedocs.io/en/latest/computation/index.html#multiple-contacts),
[preventing slip](https://mujoco.readthedocs.io/en/latest/modeling.html#preventing-slip).
