# Portable frozen parents

`frozen_parent.materialize` turns a hash-bound local-spool archive into a reusable shared parent without depending on the original worker's filesystem. It supports original `pad_translation` baselines. Corrected parents with additional external planning artifacts require a separate dependency policy and are rejected.

```python
materialize(
    archive_path,
    output,
    archive_sha256=expected_archive_sha256,
    original_spool_root=launch["spool"],
    workspace_relative=job["workspace"],
    source_relative=job["source"],
    attempt_relative=relative_attempt_path,
)
```

The original tar and selected original normalized source, metadata, motion, validation, scene, plan, replay and full replay audit remain byte-identical. The reusable workspace contains exact source and motion files, an exact plan/replay, and a derived scene/result with explicit asset relocation. Each generated mesh must be present as a regular tar member; missing files cannot silently fall back to the original pod. Existing external assets are accepted only beneath declared shared roots, verified against the original scene's checksums, and copied into the parent's content-addressed asset directory. The reused scene no longer depends on those source files.

Only XML asset file attributes and current scene-asset bindings change. Original paths, checksums and the original complete result remain available in the relocation manifest and original-inputs tree. Mesh filenames remain unchanged inside hash directories so implicit asset names remain stable. Unexpanded MJCF includes, `strippath` and unbound or ambiguous asset references fail closed.

Before publishing the final receipt, the materializer compiles the original XML through MuJoCo's in-memory asset filesystem and the relocated XML from its actual destination. It compares compiled numeric model arrays, dimensions, options, statistics and names, excluding only resource-path storage. It then applies a bounded prefix of real recorded actuator controls to both models after one initial reset. The state trajectories must match exactly and reproduce the stored prefix within the existing replay tolerance. This is relocation validation, not a new task-success claim. The original complete actuator audit remains separately bound.

The final `materialization.json` is the readiness boundary. It binds every published file and supplies `parent_workspace`, `parent_attempt` and `source` for normal frozen follow-up staging. Repeated calls verify the existing bytes and request; they do not rewrite receipts. Incomplete attempts retain their original inputs and failure report and require a fresh output label. The absolute published root must remain stable on the shared filesystem; relocating it again requires a fresh verified materialization. An empty `data/raw` directory makes no raw-download claim: frozen follow-ups use the preserved normalized source and compiled scene, while original source provenance remains unchanged.

The default 50 decimal GB reserve is checked before copies. The utility performs no network transfers or robot SDK imports. `frozen-parent-materialization-smoke-v1.json` records the first actual held-out Can baseline verification: 607 compiled fields matched exactly, 500 real recorded physics steps matched both models and stored states exactly, and the existing `frozen_plan.load` accepted the portable parent from a different worker than the original baseline pod.
