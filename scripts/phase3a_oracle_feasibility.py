"""Audit Phase3A state-oracle coverage and optional rollout non-interference."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labelled-events", type=Path, required=True)
    parser.add_argument("--reference-events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _key(raw: dict[str, Any]) -> str:
    args = raw["arguments"]
    return str(raw["name"]) if not args else f"{raw['name']}({','.join(args)})"


def _rollout_signature(records: list[dict[str, Any]]) -> dict[str, Any]:
    transitions = []
    episodes = []
    for record in records:
        payload = record["payload"]
        if record["event_type"] == "transition":
            audit = payload.get("metadata", {}).get("attribution_context", {})
            transitions.append(
                {
                    "episode_id": payload["episode_id"],
                    "step_id": payload["step_id"],
                    "action": payload["action"],
                    "last_action_success": payload["last_action_success"],
                    "feedback": payload["feedback"],
                    "pre_image_sha256": audit.get("pre_image_sha256"),
                    "post_image_sha256": audit.get("post_image_sha256"),
                }
            )
        elif record["event_type"] == "episode_result":
            episodes.append(
                {
                    key: payload[key]
                    for key in (
                        "episode_id",
                        "task_success",
                        "task_progress",
                        "environment_steps",
                        "invalid_actions",
                        "trajectory",
                    )
                }
            )
    return {"transitions": transitions, "episodes": episodes}


def main() -> int:
    args = _args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    records = _records(args.labelled_events)
    transitions = [r for r in records if r["event_type"] == "transition"]
    labels = [r for r in records if r["event_type"] == "state_oracle_label"]
    label_by_coordinate = {
        (str(r["payload"]["episode_id"]), int(r["payload"]["step_id"])): r["payload"]
        for r in labels
    }
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for transition in transitions:
        payload = transition["payload"]
        coordinate = (str(payload["episode_id"]), int(payload["step_id"]))
        action_type = str(payload["action"]["action_type"])
        label = label_by_coordinate.get(coordinate)
        if label is None:
            counts["missing_label"][action_type] += 1
            continue
        for item in label["post"]:
            predicate = str(item["key"]["name"])
            value = str(item["value"])
            counts["all_post"][value] += 1
            counts[f"predicate:{predicate}"][value] += 1
            counts[f"action:{action_type}"][value] += 1
    summary: dict[str, Any] = {}
    for group, values in sorted(counts.items()):
        total = sum(values.values())
        mapped = total - values["unknown"]
        summary[group] = {
            "total": total,
            "mapped": mapped,
            "coverage": mapped / total if total else 0.0,
            "values": dict(values),
        }
    raw_transition_json = json.dumps(
        [item["payload"] for item in transitions], sort_keys=True
    ).lower()
    result: dict[str, Any] = {
        "analysis_type": "phase3a_state_oracle_feasibility",
        "labelled_events": str(args.labelled_events),
        "transition_count": len(transitions),
        "oracle_label_count": len(labels),
        "coordinate_linkage_complete": len(transitions) == len(labels)
        and not counts.get("missing_label"),
        "oracle_absent_from_transition_payloads": "state_oracle" not in raw_transition_json
        and "evaluation_only_habitat_pddl_state" not in raw_transition_json,
        "mapping": summary,
        "reference_events": None,
        "rollout_non_interference": None,
    }
    if args.reference_events is not None:
        reference = _records(args.reference_events)
        labelled_signature = _rollout_signature(records)
        reference_signature = _rollout_signature(reference)
        result["reference_events"] = str(args.reference_events)
        result["rollout_non_interference"] = {
            "exact_match": labelled_signature == reference_signature,
            "labelled_transition_count": len(labelled_signature["transitions"]),
            "reference_transition_count": len(reference_signature["transitions"]),
            "labelled_episode_count": len(labelled_signature["episodes"]),
            "reference_episode_count": len(reference_signature["episodes"]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
