# Reachy retarget

Acquire as many and as diverse object-interaction demonstrations as possible and
retarget them onto the Pollen Reachy 2 (two 7-DoF arms with parallel grippers, a 3-DoF
neck and a holonomic base) as **state-only** training data for manipulation policies.
No images are stored. Every episode keeps canonical robot and object state plus
precomputed views for every control mode, because the policy state/action format is
not fixed yet, and the meshes of its scene (objects, supports, fixtures, articulated parts and
Reachy's links, with materials and texture files) plus per-frame poses of every scene component,
so a policy can build a surface feature map of the scene. Datasets whose meshes cannot be
obtained are excluded.

The previous phase of this project (maximizing MuJoCo success on a few robomimic Can
sources) is kept read-only in [`legacy/`](legacy/); see `legacy/DELETED.md` for the
artifacts that were removed.

## Pipeline

```
source files ─▶ adapter ─▶ SourceEpisode ─▶ retarget ─▶ tier K ─▶ tier P ─▶ scene meshes ─▶ episode HDF5 + assets/
```

| Stage | Module | Notes |
| --- | --- | --- |
| Acquisition | `reachy_retarget.acquire` | Hash-pinned catalogs per family, one fetch interface (files, tar streams, byte ranges), 50 GB free-space reserve, image members stripped or never downloaded |
| Source adapters | `reachy_retarget.sources` | One adapter per family, imported lazily; output is the embodiment-independent `SourceEpisode` (grasp-center poses, gripper opening, base path, objects, articulations, scene) |
| Retargeting | `reachy_retarget.retarget` | Object-centric grasp selection, base placement / base assist, bounded whole-body IK with self-collision, gaze, gripper mapping, limit-respecting retiming to 50 Hz |
| Tier K | `reachy_retarget.validate.kinematic` | Every episode: tracking residuals, joint/speed limits, self-clearance, footprint, in-hand consistency |
| Tier P | `reachy_retarget.validate.physics` | Sources with a MuJoCo scene: source robot replaced by Reachy, free objects, position servos only, physical gates, actuator-only replay |
| Scene meshes | `reachy_retarget.meshes` | Per-component visual/collision parts of the source MuJoCo scene and Reachy's links, content-addressed asset library `<out>/assets/`, per-frame component poses (episode and tier-P clocks) |
| Storage | `reachy_retarget.schema` | `reachy-retarget-episode-v2` HDF5 (with `/scene`) + `index.parquet`; control-mode registry (joint, EE, head, base, gripper, reachy-agent v8); scene reader `schema.scene_assets` |
| Cluster | `cluster/` | Detached two-slot job pool over the persistent worker pods, PVC publication with hash checks |

Design: [docs/design.md](docs/design.md). Storage and control modes:
[docs/schema.md](docs/schema.md). Sources, licenses and verification:
[docs/sources.md](docs/sources.md). K and P passes are always reported separately.

## Source families

robomimic, MimicGen, LIBERO, DexMimicGen (parallel-gripper tasks), ManiSkill3 demos,
RoboCasa365 (human), BiGym, BEHAVIOR-1K 2025, MobileManiBench (G1), MolmoBot-Data,
RoboVerse (RLBench, CALVIN). Human-hand/mocap datasets and real-robot datasets without
object state are excluded by design.

Scene meshes (`reachy_retarget.sources.registry.MESHES`): robomimic, MimicGen, LIBERO,
DexMimicGen, RoboCasa, BiGym and ManiSkill (primitive scenes) provide them and are built;
RoboVerse and MobileManiBench are pending; BEHAVIOR (encrypted assets) and MolmoBot (no per-frame
object poses) are excluded and refused by the build. Episodes whose scene assets cannot be
resolved (kinematic-only source route) are not written.

## Usage

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[physics,archives,dev]'
.venv/bin/python -m pytest -q
```

```bash
.venv/bin/reachy-retarget catalog --family robomimic
.venv/bin/reachy-retarget fetch robomimic/ --root data            # explicit network access
.venv/bin/python -m reachy_retarget.evaluate --family robomimic \
    --path data/raw/robomimic/v1.5/lift/ph/low_dim_v15.hdf5 --demos 0-9 --physics
.venv/bin/python -m reachy_retarget.build --family robomimic \
    --path data/raw/robomimic/v1.5/lift/ph/low_dim_v15.hdf5 --shard 0/10 --physics --out runs/lift
.venv/bin/python -m reachy_retarget.report runs/lift
```

```python
from reachy_retarget.schema.scene_assets import load_component_meshes, read_scene, scene_mjcf
scene, library = read_scene("runs/lift/episodes/robomimic/lift/ph/demo_0.h5")   # components, poses
parts = load_component_meshes("runs/lift/episodes/robomimic/lift/ph/demo_0.h5")  # {component: [part]}
xml = scene_mjcf("runs/lift/episodes/robomimic/lift/ph/demo_0.h5", frame=0)    # compiles in MuJoCo
```

Cluster (namespace `erl-ucsd`, persistent `reachy-retarget-worker` pods; pods are never
deleted or recreated):

```bash
.venv/bin/python -m cluster.manifests build robocasa --listing runs/listings/pvc-raw-robocasa.txt
.venv/bin/python -m cluster.pool runs/build-w2-robocasa-v1.jsonl --slots 2
```

Releases are built from the committed `HEAD`, so commit before launching a batch.
