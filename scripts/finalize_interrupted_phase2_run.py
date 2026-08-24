"""Create explicit recovered manifests after an interrupted experiment audit."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from vista_skill.skills import load_skill_artifact_record, skill_digest


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    args = _args()
    run_dir = args.run_dir
    acquisition = _jsonl(run_dir / "acquisition.jsonl")
    lineage = _jsonl(run_dir / "lineage.jsonl")
    audit = json.loads((run_dir / "update_audit.json").read_text(encoding="utf-8"))
    if not lineage:
        raise ValueError("cannot recover manifest without lineage protocol")
    protocols = {json.dumps(item["protocol"], sort_keys=True) for item in lineage}
    if len(protocols) != 1:
        raise ValueError("lineage does not contain one consistent run protocol")
    protocol = json.loads(next(iter(protocols)))

    episode_results = [
        item["payload"]
        for item in acquisition
        if item.get("event_type") == "episode_result"
    ]
    transitions = [
        item["payload"]
        for item in acquisition
        if item.get("event_type") == "transition"
    ]
    expected_budget = int(protocol["acquisition_episode_budget"])
    if len(episode_results) != expected_budget:
        raise ValueError(
            f"acquisition episode count {len(episode_results)} != protocol {expected_budget}"
        )
    frozen = load_skill_artifact_record(run_dir / "frozen_skill.json")
    attribution_counts = Counter(
        str(item.get("attribution", {}).get("target", "missing"))
        for item in transitions
    )
    reconstructed = {
        "episode_count": len(episode_results),
        "success_count": sum(bool(float(item["task_success"])) for item in episode_results),
        "mean_task_progress": mean(float(item["task_progress"]) for item in episode_results),
        "mean_environment_steps": mean(
            int(item["environment_steps"]) for item in episode_results
        ),
        "transition_count": len(transitions),
        "attribution_target_counts": dict(sorted(attribution_counts.items())),
        "proposal_attempt_count": len(lineage),
        "accepted_proposal_count": sum(bool(item["accepted"]) for item in lineage),
    }
    recovery = {
        "recovered_after_process_interruption": True,
        "scientific_rollout_coordinates_reused_from_complete_jsonl": True,
        "method_usage_unavailable": True,
        "executor_usage_unavailable": True,
        "ready_clusters_by_episode_unavailable": True,
        "reason": (
            "The original process ended after acquisition and during independent audit, "
            "before in-memory usage counters were flushed to run_manifest.json."
        ),
    }
    run_record = {
        **protocol,
        "acquisition_episode_count": len(episode_results),
        "ready_clusters_by_episode": None,
        "evolution_decisions": [bool(item["accepted"]) for item in lineage],
        "frozen_skill_sha256": skill_digest(frozen.skill),
        "method_usage": None,
        "executor_usage": None,
        "update_reliability": audit["reliability"],
        "reconstructed_summary": reconstructed,
        "recovery": recovery,
    }
    run_manifest = run_dir / "run_manifest.json"
    _write_exclusive(run_manifest, run_record)

    top_protocol = dict(protocol)
    for key in (
        "evolution_seed",
        "run_id",
        "split_rotation_index",
        "split_sha256",
        "gate_rollout_seeds",
        "acquisition_episode_budget",
    ):
        top_protocol.pop(key, None)
    run_id = str(protocol["run_id"])
    experiment_id = run_id.rsplit("_seed_", 1)[0]
    experiment_manifest = run_dir.parent / "experiment_manifest.json"
    _write_exclusive(
        experiment_manifest,
        {
            **top_protocol,
            "experiment_id": experiment_id,
            "run_count": 1,
            "runs": [run_record],
            "recovery": recovery,
        },
    )
    print(f"wrote {run_manifest}")
    print(f"wrote {experiment_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
