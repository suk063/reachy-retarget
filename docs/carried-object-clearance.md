# Carried-object placement geometry

`CarriedObjectQuery(prepared, support_geoms=[...], minimum_m=.003)` checks
the exact compiled rigid object's collision geometry at a hypothetical TCP
target. Its fixed hand-to-object transform comes from the bound original source
release frame. Mesh compilation rotations, geom offsets and rigid child body
transforms are retained. Articulated objects are rejected by this rigid adapter.

The query owns separate, non-integrated MuJoCo data. It transforms only derived
geom positions and rotations. It never writes an object qpos, qvel, model property
or simulation clock. A test compares a rotated mesh with a rotated child body
against independent MuJoCo FK and checks exact model/state preservation.

The default query includes every compiled physical object/fixture pair, excluding
the robot because intended grip contact is assessed separately. During lowering,
`allow_support=True` permits contact only with caller-declared horizontal support
planes or thin floor boxes. Vertical bin walls cannot be exempted this way.
This represents a support-gated descent proposal; it does not prescribe object
motion or prove measured floor contact.

`prepare(prepared, output, support_geoms=[...])` requires a complete admitted robot
placement and checks every inserted traverse/lower control row. It saves full
reference arrays, per-row distances and the worst pair. Failure is retained and
raises an error. Its report explicitly leaves original source stages and physical
validation outside this new check. The existing measured support/opening guard
and all physical contact, slip and task gates remain required.

The 6a80 example exposed why hand-only clearance is insufficient. Its unchanged
desired grasp put the compiled Can 1.96444 mm through a target-bin partition
during descent. Actual recordings show that same Can/wall contact before any
floor support. A +5 mm placement-X correction inside the original task region
clears 402 sampled carried-object poses and the hand screen by 3 mm. Fresh full
robot and every-control-row carried-object admission are still required before
testing this proposal physically. The previous lower-acceleration comparison
is retained as a failure; it could not fix an obstructed carried-object target.

`measured_attachment_bound(...)` separately estimates empirical attachment error
from explicitly selected acquisition frames. It requires measured hand/object
transforms, frame indices, sample times, provenance, and per-frame evidence of
bilateral lifted grasp, no fixture contact, and pre-placement timing. It preserves
the ideal source transform and every measured `hand^-1 @ object` transform as
distinct fields. It never replaces either with a fabricated controller target.

For a conservative collision-body radius `r`, each sample's mesh displacement
bound is `norm(t_measured - t_ideal) + 2*r*sin(rotation_error/2)`. This bounds all
points in that body sphere for the recorded sample. The returned maximum is an
**observed-window** bound. Future grasp drift and robot tracking need separate
declared reserves and actual physical validation. Array checksums and selection
provenance make this distinction auditable. Tests verify surface-point coverage,
world-frame invariance, and rejection of contaminated acquisition windows.
