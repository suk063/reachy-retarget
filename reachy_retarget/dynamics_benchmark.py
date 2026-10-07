"""Run a frozen dynamics configuration on development and held-out episodes.

No search or network access occurs here. Every physical attempt and independent
actuator replay audit is retained, including failures and unsupported episodes.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .dynamics import Candidate, rollout, TASK_OBJECTS
from .dynamics_audit import verify
from .episodes import read_episode
from .store import Store, json_write, sha256, now


def run(root, config_path, label):
    store = Store(Path(root).resolve())
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    out = store.root / "runs/dynamics" / label
    out.mkdir(parents=True, exist_ok=False)
    (out / "frozen-config.json").write_text(config_path.read_text())
    summary = {"created": now(), "config_sha256": sha256(config_path),
               "protocol": "reachy-object-dynamics-v1", "attempts": [], "unsupported": [],
               "scope": "Local source episodes; development settings fixed before remaining episodes. No population success-rate claim."}
    for row in store.rows("episodes"):
        _, metadata = read_episode(store.root / row["path"])
        task = metadata.get("env_args", {}).get("env_name", metadata.get("scenario"))
        record = {"source_episode": row["id"], "source_id": row["source_id"],
                  "source_sequence": row["source_sequence"], "task": task}
        if task not in config["tasks"] or task not in TASK_OBJECTS:
            record.update(status="unsupported", reason="No complete verified dynamics/task adapter in this configuration")
            summary["unsupported"].append(record)
            continue
        spec = config["tasks"][task]
        candidate = Candidate(**spec["candidate"])
        record["split"] = "development" if row["id"] == spec["development_episode"] else "held_out"
        report = rollout(store, row, label, candidate)
        attempt = out / row["id"]
        record.update(status=report["status"], success=report["success"],
                      failure_reasons=report["failure_reasons"], candidate=asdict(candidate),
                      result=str((attempt / "result.json").relative_to(store.root)),
                      result_sha256=sha256(attempt / "result.json"),
                      task_stable=report.get("gates", {}).get("task_stable_final_1s", False))
        if (attempt / "replay.h5").is_file():
            try:
                audit = verify(attempt)
                record["actuator_replay_pass"] = audit["actuator_replay_pass"]
            except Exception as exc:
                record["actuator_replay_pass"] = False
                record["audit_error"] = f"{type(exc).__name__}: {exc}"
                json_write(attempt / "actuator-replay-audit-error.json", record)
        else:
            record["actuator_replay_pass"] = None
        record["audited_physical_pass"] = bool(record["success"] and record["actuator_replay_pass"])
        summary["attempts"].append(record)
        json_write(out / "benchmark.json", summary)
    summary["finished"] = now()
    summary["counts"] = {"episodes": len(summary["attempts"]),
                         "unsupported": len(summary["unsupported"]),
                         "audited_physical_pass": sum(r["audited_physical_pass"] for r in summary["attempts"]),
                         "task_stable": sum(r["task_stable"] for r in summary["attempts"])}
    json_write(out / "benchmark.json", summary)
    print(json.dumps(summary["counts"]), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    run(args.root, args.config, args.label)


if __name__ == "__main__":
    main()
