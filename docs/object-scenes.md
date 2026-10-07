# Source-derived dynamic object scenes

`reachy_retarget.object_scene.build_scene(root, metadata, arrays, placement)`
returns a compiled Reachy MJCF scene and a source-fidelity manifest. It performs
no network access, imports no robot SDK, and never changes a running object's
state. Recorded object poses are assigned once in the reset XML. Every moving
object keeps its free joint; object welds, servos, mocap, and gravity compensation
are rejected.

The manifest exposes `mimics` and `objects[object_id]`, including body/joint
names and IDs, position/velocity addresses, collision geometry, initial pose,
mass, and component inertias. The placement transform moves the complete source
environment and objects together and must preserve the gravity direction.

## Recorded robosuite scenes

The implementation extracts the recorded MJCF for Can, Lift, Square,
ToolHang, TwoArmTransport, MimicGen Stack, and Threading. It retains collision
geometry and task fixtures, resolves source defaults before merging with Reachy,
and copies source-compiled mass, inertia, joint dynamics, friction, contact
parameters, and environmental fluid properties. Robot defaults cannot silently
change object properties. Inertias are compared as tensors because equivalent
principal-axis frames can have different quaternion signs or axis ordering.

Missing collision meshes stop compilation and enumerate version-pinned upstream
URLs for explicit acquisition. Missing appearance-only textures are recorded.
The source robot and unrelated parked objects are omitted with explicit reasons.
ToolHang's zero-mass carrier retains the physical masses on its fixed children.

All 14 previously missing robosuite textures are now acquired from their
version-pinned upstream URLs: 11,169,253 bytes across versions 1.4.1 and 1.5.1.
All 21 robomimic/MimicGen sample scenes now have their required source textures;
`texture-acquisition.json` records URLs, hashes, sizes, and statuses. Texture
acquisition did not change object collision geometry, mass, or friction.

Each scene is compared against an isolated source environment after applying
the world placement. Source and merged body/geometry transforms, masses,
inertia tensors, contact fields, and free-joint dynamics must agree within
`1e-7`. Reachy's original world forward kinematics must remain unchanged.

## ManiSkill PickCube conversion

This path is restricted to the acquired PickCube release's
[ManiSkill revision ecc579b](https://github.com/haosulab/ManiSkill/tree/ecc579b7567c31bb3d7176539d90e86ae49296be).
It verifies local primary-evidence files against committed SHA-256 hashes before
constructing the scene. There is no recorded MJCF; the manifest labels its XML
as a generated PhysX-to-MuJoCo reconstruction.

The pinned [task implementation](https://github.com/haosulab/ManiSkill/blob/ecc579b7567c31bb3d7176539d90e86ae49296be/mani_skill/envs/tasks/pick_cube.py)
creates a red cube with half-size 0.02 m.
[The pinned setup.py](https://github.com/haosulab/ManiSkill/blob/ecc579b7567c31bb3d7176539d90e86ae49296be/setup.py)
requires SAPIEN 3.0.0.dev2, whose
[ActorBuilder](https://github.com/haosulab/SAPIEN/blob/3.0.0.dev2/python/py_package/wrapper/actor_builder.py)
defaults to density 1000 kg/m³. Thus the cube mass is 0.064 kg, and each principal
moment is approximately `1.7066667e-5 kg m²`. These values are independently
checked after MuJoCo compilation. The
[source material configuration](https://github.com/haosulab/ManiSkill/blob/ecc579b7567c31bb3d7176539d90e86ae49296be/mani_skill/utils/structs/types.py)
uses static/dynamic friction 0.3 and restitution zero.

The [source table builder](https://github.com/haosulab/ManiSkill/blob/ecc579b7567c31bb3d7176539d90e86ae49296be/mani_skill/utils/scene_builder/table/table_scene_builder.py)
uses a nonconvex GLB, scale 1.75, and a local 90-degree rotation. Its recorded
actor transform is retained and checked against the source builder. The table
is **not replaced by a plane**. The source tabletop and crossbar are convex.
The two concave legs are extruded profiles: their original cap triangles form
convex prisms without filling the open spaces. The manifest records the volume
agreement and at most approximately 0.2 micrometers of world-space adjustment
for GLB floating-point noise. Full source visual triangles and the embedded
tabletop texture remain available. The floor's infinite collision plane is
retained; its unavailable tiled appearance is listed separately.

PhysX and MuJoCo do not have equivalent contact solvers. The equal source
friction coefficients map to MuJoCo sliding friction 0.3, with no invented
torsional/rolling resistance. Restitution zero maps to a critically damped
MuJoCo contact (`solref=".02 1"`); the PhysX contact offset is not interpreted as
surface inflation. MuJoCo contact mixing and the Reachy material replace the
original Panda contact-pair behavior. These assumptions are included in every
manifest and prevent scene fidelity from being reported as numerical simulator
equivalence.

The noncolliding, source-hidden goal marker is omitted from geometry; its
recorded goal remains available in normalized arrays. PickCube's goal is a cube
within 0.025 m of the target with a stationary robot, so release is not required.

## Validation artifacts

`runs/object-matrix/runs/scene-fidelity/all-scenes.json` records compilation and
source-fidelity results for 26 acquired episodes: 15 robomimic, 6 MimicGen, and
5 ManiSkill. All compile, with a maximum recorded fidelity discrepancy of
`4.87e-11`. HUMOTO does not contain an exact source dynamics scene and is not
included in this claim.

`tests/test_object_scene.py` contains 10 offline checks for source defaults,
mass/inertia, world placement, unconstrained free fall, compound objects,
forbidden object constraints, missing assets, pinned ManiSkill evidence, and
concave-table decomposition. Run:

```sh
.venv/bin/python -m pytest -q tests/test_object_scene.py
```

Scene compilation and parameter agreement are prerequisites for dynamic trials;
they do not establish grasp, transport, placement, or task success. Separate
dynamic trials must retain their failed attempts and measure the actual object
trajectory without resetting it after simulation starts.

## PickCube planner and task adapter

`reachy_retarget.dynamics_maniskill` supplies the task-specific hooks without
changing the generic dynamic controller. It accepts only the pinned Panda
`pd_joint_pos` export. Source action column 7 is `+1=open`, `-1=close`, from the
documented normalized Panda finger target range `[-0.01, 0.04]` m. The planner's
closure convention has the opposite sign. Original T actions stay untouched;
the derived command sequence has T+1 entries and explicitly holds the last
command at the terminal measured state.

The source grasp anchor requires an actual recorded cube lift after a closed
gripper transition. The chosen Reachy grasp point is the original cube's center,
without changing its geometry. The complete scene transform initially adds
0.8 m to source world height; this moves the robot targets, table, cube, and goal
consistently.

Success uses the recorded `goal_site` target, which can differ from the final
demonstrated cube position. The publisher predicate requires Euclidean goal
error at most 0.025 m and maximum absolute non-finger Panda joint speed at most
0.2 rad/s. Its retargeted interpretation checks all Reachy arm joints at that
same angular threshold and adds explicit base stationarity limits of 0.02 m/s
planar speed and 0.2 rad/s yaw speed. Gripper release is not required. Generic
rollout contact, penetration, slip, and sustained-goal gates still apply.

`tests/test_dynamics_maniskill.py` contains 9 offline checks for action sign,
T/T+1 alignment, terminal hold, unchanged source arrays, original geometry,
goal placement, the maximum-joint-speed criterion, and unsupported schemas.
All five acquired episodes pass the adapter audit; this checks the source
contract and enables dynamic trials, without claiming their physical success.

## Reproduce source assets on a fresh root

[The matrix lock](../configs/object-matrix-lock.jsonl.gz) includes eight source
records: four demonstration sources and four supporting asset/evidence bundles.
It contains 54 downloaded raw files totaling 673,639,339 bytes, each with an
expected SHA-256 for a future direct download. Six unsuccessful exploratory
code-path probes remain in the lock as excluded failure records, so restoring
the selection does not retry them. Their original failed ledger entries remain
intact. Raw source payloads were rehashed before and after lock export and were
unchanged.

The following explicit commands restore and acquire that selection. They
require the project's configured Python environment and the read-only
`reachy-control` reference repository; they never install or import a robot SDK.

```sh
export REACHY_RETARGET_CONTROL="$(cd ../reachy-control && pwd)"
export REACHY_RETARGET_BACKEND=reference
.venv/bin/reachy-retarget --root runs/object-matrix-reproduction discover --locked configs/object-matrix-lock.jsonl.gz
.venv/bin/reachy-retarget --root runs/object-matrix-reproduction fetch --transport http
.venv/bin/reachy-retarget --root runs/object-matrix-reproduction normalize --source hf__robomimic__robomimic_datasets --source hf__amandlek__mimicgen_datasets --source humoto --limit 3
.venv/bin/reachy-retarget --root runs/object-matrix-reproduction normalize --source hf__haosulab__ManiSkill_Demonstrations --limit 5
```

Importing `object_scene` does not acquire or generate assets. An explicit
`build_scene` call rebuilds the derived Reachy meshes and ManiSkill tabletop
texture when they are absent. Source GLB geometry stays unchanged. For example,
after the commands above, build every supported source scene:

```sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np
from reachy_retarget.episodes import read_episode
from reachy_retarget.object_scene import build_scene

root = Path("runs/object-matrix-reproduction").resolve()
out = root / "runs/scene-fidelity"
out.mkdir(parents=True, exist_ok=True)
for path in sorted((root / "data/normalized").glob("*/*.h5")):
    arrays, metadata = read_episode(path)
    maniskill = metadata.get("source_format") == "ManiSkill-HDF5"
    if not metadata.get("model_xml") and not maniskill:
        continue
    placement = np.eye(4)
    placement[:3, 3] = [.5, -.1, .8 if maniskill else 0.]
    xml, manifest = build_scene(root, metadata, arrays, placement)
    stem = metadata["episode_id"]
    (out / (stem + ".xml")).write_text(xml)
    (out / (stem + ".json")).write_text(json.dumps(manifest, indent=2))
PY
```

`lock-verification.json` records an offline fresh-root check: the lock was
restored into an empty directory, verified raw bytes were linked locally, and
all 26 supported scenes compiled without preexisting generated assets. Both
Reachy meshes and the ManiSkill texture were rebuilt there. This verifies local
asset resolution and generation without claiming a second network download.
Available disk space remained approximately 103.2 decimal GB, above the
50 decimal GB reserve.
