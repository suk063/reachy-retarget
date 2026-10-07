"""Apply one frozen seed bank and incremental optimizer to all source tasks."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from .dynamics import Candidate, rollout, objective
from .dynamics_audit import verify
from .episodes import read_episode
from .store import Store, json_write, sha256, now
from .task_contracts import TASK_OBJECTS
from .trajectory_search import SearchConfig, optimize


def run(root, config_path, label, *, episode_ids=None):
    store = Store(Path(root).resolve())
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if set(config) != {"schema", "candidate", "seed_bank", "optimizer"} or config["schema"] != "reachy-common-search-v1":
        raise ValueError("Common configuration must contain global settings only")
    seed = Candidate(**config["candidate"])
    search_config = SearchConfig(**config["optimizer"])
    search_config.validate()
    if not seed.automatic_grasp or seed.source_start_s != 0:
        raise ValueError("Common benchmark requires geometry inference and complete source intervals")
    out = store.root/"runs/dynamics"/label
    out.mkdir(parents=True, exist_ok=False)
    (out/"frozen-config.json").write_text(config_path.read_text())
    summary = {"created": now(), "config_sha256": sha256(config_path), "policy": config["schema"],
               "scope": "All episodes use the same seed bank, objective and search budget. Previously inspected local episodes; not an unseen-object benchmark.",
               "attempts": [], "unsupported": []}
    for row in store.rows("episodes"):
        if episode_ids and row["id"] not in episode_ids:
            continue
        _, metadata = read_episode(store.root/row["path"])
        task = metadata.get("env_args", {}).get("env_name", metadata.get("scenario"))
        record = {"source_episode": row["id"], "source_id": row["source_id"],
                  "source_sequence": row["source_sequence"], "task": task, "trials": []}
        if task not in TASK_OBJECTS:
            record.update(status="unsupported", reason="No verified source/task contract")
            summary["unsupported"].append(record)
            json_write(out/"benchmark.json", summary)
            continue
        reports = []
        for index, overrides in enumerate(config["seed_bank"]):
            if set(overrides) - {"tilt", "grasp_region_rank", "depth", "force_scale", "yaw", "calibrate_attachment"}:
                raise ValueError("Seed bank may only vary common grasp controls")
            candidate = replace(seed, **overrides)
            trial_label = f"{label}/{row['id']}/seed_{index:02d}"
            result = rollout(store, row, trial_label, candidate)
            directory = store.root/"runs/dynamics"/trial_label/row["id"]
            if (directory/"replay.h5").exists():
                audit = verify(directory)
                audited = bool(audit["actuator_replay_pass"] and result["success"])
            else:
                audited = False
            record["trials"].append({"path": str(directory.relative_to(store.root)),
                                     "kind": "common_geometry_seed", "status": result["status"],
                                     "audited_physical_pass": audited})
            reports.append((objective(result), result, candidate, directory))
            if audited:
                break
        _, selected, candidate, directory = min(reports, key=lambda item:item[0])
        if not selected["success"] and (directory/"plan.h5").exists():
            try:
                controls = optimize(directory, out/row["id"]/"optimization", search_config)
                trial_label = f"{label}/{row['id']}/optimized"
                refined = rollout(store, row, trial_label, candidate, control_path=controls)
                refined_dir = store.root/"runs/dynamics"/trial_label/row["id"]
                audit = verify(refined_dir)
                record["trials"].append({"path": str(refined_dir.relative_to(store.root)),
                                         "kind": "incremental_actuator_search", "status": refined["status"],
                                         "audited_physical_pass": bool(refined["success"] and audit["actuator_replay_pass"])})
                selected = refined if objective(refined) < objective(selected) else selected
            except Exception as exc:
                record["optimization_error"] = f"{type(exc).__name__}: {exc}"
                json_write(out/row["id"]/"optimization-error.json", record)
        record.update(status="physical_pass" if any(x["audited_physical_pass"] for x in record["trials"]) else selected["status"],
                      audited_physical_pass=any(x["audited_physical_pass"] for x in record["trials"]),
                      failure_reasons=selected["failure_reasons"])
        summary["attempts"].append(record)
        json_write(out/"benchmark.json", summary)
        print(json.dumps({"source_episode": row["id"], "task": task, "status": record["status"]}), flush=True)
    summary["finished"] = now()
    summary["counts"] = {"episodes": len(summary["attempts"]), "unsupported":len(summary["unsupported"]),
                         "audited_physical_pass":sum(r["audited_physical_pass"] for r in summary["attempts"])}
    json_write(out/"benchmark.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/common-dynamics-v1.json"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--episode", action="append")
    args = parser.parse_args()
    result = run(args.root, args.config, args.label, episode_ids=args.episode)
    print(json.dumps(result["counts"]), flush=True)


if __name__ == "__main__":
    main()
