#!/usr/bin/env python3
"""Analyze the historical No-Skill/Static-Skill trajectory sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULTS = {
    "hab_no": "running/pilot/eval/no_skill.jsonl",
    "hab_static": "running/pilot/eval/static_shared_skill.jsonl",
    "nav_no": "running/vista_skill/nav_official/base/no_skill/events.jsonl",
    "nav_static": "running/vista_skill/nav_official/base/static_shared_skill/events.jsonl",
    "nav_no_raw": "running/eb_nav/vista_skill/evaluate/evaluate_5c1d770e3c274ddc8671b9a32e2426cb/official_test/base/s0/no_skill",
    "nav_static_raw": "running/eb_nav/vista_skill/evaluate/evaluate_7d147751609a4bd79c79d0230c7e043e/official_test/base/s0/57086f2f1f66",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_episode_results(path: Path) -> dict[str, dict[str, Any]]:
    results = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event_type") == "episode_result":
                payload = dict(record["payload"])
                results[str(payload["episode_id"])] = payload
    if not results:
        raise ValueError(f"no episode_result records in {path}")
    return results


def _mean(values: Iterable[float]) -> float:
    items = tuple(float(value) for value in values)
    return statistics.fmean(items) if items else 0.0


def _adjacent_repetition_ratio(results: Mapping[str, Mapping[str, Any]]) -> float:
    repeats = 0
    opportunities = 0
    for result in results.values():
        trajectory = tuple(str(item) for item in result.get("trajectory", ()))
        repeats += sum(left == right for left, right in zip(trajectory, trajectory[1:]))
        opportunities += max(0, len(trajectory) - 1)
    return repeats / opportunities if opportunities else 0.0


def _repeated_two_action_cycle_ratio(
    results: Mapping[str, Mapping[str, Any]],
) -> float:
    repeats = 0
    opportunities = 0
    for result in results.values():
        trajectory = tuple(str(item) for item in result.get("trajectory", ()))
        for index in range(3, len(trajectory)):
            opportunities += 1
            if trajectory[index - 3 : index - 1] == trajectory[index - 1 : index + 1]:
                repeats += 1
    return repeats / opportunities if opportunities else 0.0


def _arm_summary(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    steps = sum(int(item["environment_steps"]) for item in results.values())
    invalid = sum(int(item["invalid_actions"]) for item in results.values())
    successes = [item for item in results.values() if float(item["task_success"]) > 0]
    return {
        "episodes": len(results),
        "mean_task_success": _mean(item["task_success"] for item in results.values()),
        "mean_task_progress": _mean(item["task_progress"] for item in results.values()),
        "mean_environment_steps": _mean(item["environment_steps"] for item in results.values()),
        "mean_planner_steps": _mean(item["planner_steps"] for item in results.values()),
        "invalid_action_ratio": invalid / steps if steps else 0.0,
        "mean_steps_when_successful": _mean(item["environment_steps"] for item in successes),
        "adjacent_repetition_ratio": _adjacent_repetition_ratio(results),
        "repeated_two_action_cycle_ratio": _repeated_two_action_cycle_ratio(results),
    }


def _bootstrap_mean_delta(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    metric: str,
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    episode_ids = sorted(set(left) & set(right))
    deltas = [float(right[key][metric]) - float(left[key][metric]) for key in episode_ids]
    rng = random.Random(seed)
    boot = sorted(
        statistics.fmean(rng.choice(deltas) for _ in deltas)
        for _ in range(samples)
    )
    return {
        "mean_delta": statistics.fmean(deltas),
        "ci95_low": boot[int(0.025 * samples)],
        "ci95_high": boot[min(samples - 1, int(0.975 * samples))],
    }


def _paired_summary(
    no_skill: Mapping[str, Mapping[str, Any]],
    static: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ids = sorted(set(no_skill) & set(static))
    improved = [key for key in ids if static[key]["task_success"] > no_skill[key]["task_success"]]
    regressed = [key for key in ids if static[key]["task_success"] < no_skill[key]["task_success"]]
    progress_changed = [
        {
            "episode_id": key,
            "instruction": no_skill[key]["instruction"],
            "no_skill_success": no_skill[key]["task_success"],
            "static_success": static[key]["task_success"],
            "no_skill_progress": no_skill[key]["task_progress"],
            "static_progress": static[key]["task_progress"],
            "no_skill_steps": no_skill[key]["environment_steps"],
            "static_steps": static[key]["environment_steps"],
            "no_skill_invalid": no_skill[key]["invalid_actions"],
            "static_invalid": static[key]["invalid_actions"],
        }
        for key in ids
        if (
            no_skill[key]["task_success"],
            no_skill[key]["task_progress"],
        )
        != (static[key]["task_success"], static[key]["task_progress"])
    ]
    return {
        "paired_episodes": len(ids),
        "static_success_improved_episode_ids": improved,
        "static_success_regressed_episode_ids": regressed,
        "success_ties": len(ids) - len(improved) - len(regressed),
        "static_minus_no_skill_success": _bootstrap_mean_delta(
            no_skill, static, "task_success"
        ),
        "static_minus_no_skill_progress": _bootstrap_mean_delta(
            no_skill, static, "task_progress"
        ),
        "outcome_or_progress_changes": progress_changed,
    }


def _decode_concatenated_json(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    offset = 0
    records = []
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            break
        record, offset = decoder.raw_decode(text, offset)
        records.append(record)
    return records


def _nav_raw_summary(raw_dir: Path, results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    episodes = []
    for episode_id in sorted(results, key=lambda item: int(item.split("_")[-1])):
        index = int(episode_id.split("_")[-1]) + 1
        records = _decode_concatenated_json(raw_dir / f"episode_{index}.json")
        distances = [float(item["distance"]) for item in records]
        plan_run_lengths = []
        current_reasoning = None
        current_run = 0
        for record in records:
            reasoning = str(record.get("reasoning", ""))
            if reasoning == current_reasoning:
                current_run += 1
            else:
                if current_run:
                    plan_run_lengths.append(current_run)
                current_reasoning = reasoning
                current_run = 1
        if current_run:
            plan_run_lengths.append(current_run)
        worsening = sum(
            after > before + 1e-6
            for before, after in zip(distances, distances[1:])
        )
        stagnant = sum(
            abs(after - before) <= 1e-6
            for before, after in zip(distances, distances[1:])
        )
        minimum = min(distances)
        final = distances[-1]
        success = float(results[episode_id]["task_success"]) > 0
        episodes.append(
            {
                "episode_id": episode_id,
                "success": success,
                "minimum_distance": minimum,
                "final_distance": final,
                "worsening_steps": worsening,
                "stagnant_steps": stagnant,
                "max_open_loop_plan_actions": max(plan_run_lengths),
                "overshot_near_threshold": (
                    not success and minimum <= 1.25 and final >= minimum + 0.20
                ),
            }
        )
    return {
        "mean_minimum_distance": _mean(item["minimum_distance"] for item in episodes),
        "mean_final_distance": _mean(item["final_distance"] for item in episodes),
        "mean_worsening_steps": _mean(item["worsening_steps"] for item in episodes),
        "mean_stagnant_steps": _mean(item["stagnant_steps"] for item in episodes),
        "mean_max_open_loop_plan_actions": _mean(
            item["max_open_loop_plan_actions"] for item in episodes
        ),
        "episodes_with_plan_at_least_3_actions": sum(
            item["max_open_loop_plan_actions"] >= 3 for item in episodes
        ),
        "overshot_near_threshold_episode_ids": [
            item["episode_id"] for item in episodes if item["overshot_near_threshold"]
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", default=default)
    parser.add_argument(
        "--output",
        default="running/target_skill_oracle_v1/source_analysis.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = {name: Path(getattr(args, name)) for name in DEFAULTS}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    hab_no = _load_episode_results(paths["hab_no"])
    hab_static = _load_episode_results(paths["hab_static"])
    nav_no = _load_episode_results(paths["nav_no"])
    nav_static = _load_episode_results(paths["nav_static"])
    report = {
        "analysis_type": "posthoc_source_trajectory_analysis",
        "source_split": "official_test/base",
        "source_trajectory_contamination": True,
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256(path) if path.is_file() else None,
            }
            for name, path in paths.items()
        },
        "eb_hab": {
            "no_skill": _arm_summary(hab_no),
            "static_shared_skill": _arm_summary(hab_static),
            "paired": _paired_summary(hab_no, hab_static),
            "observed_failure_signals": {
                "dominant": "invalid Pick loops after navigation without object-nearness evidence",
                "additional": [
                    "requested-object identity substitution",
                    "Pick repeated while already holding or after failed Pick",
                    "destination/removal semantics lost after manipulation",
                    "multi-object checklist incompletion",
                ],
            },
        },
        "eb_nav": {
            "no_skill": _arm_summary(nav_no),
            "static_shared_skill": _arm_summary(nav_static),
            "paired": _paired_summary(nav_no, nav_static),
            "no_skill_distance_dynamics": _nav_raw_summary(
                paths["nav_no_raw"], nav_no
            ),
            "static_distance_dynamics": _nav_raw_summary(
                paths["nav_static_raw"], nav_static
            ),
            "observed_failure_signals": {
                "dominant": "open-loop multi-action plans overshoot or continue after distance worsens",
                "additional": [
                    "blocked translations repeated without recovery",
                    "visual closeness claims override numeric distance",
                    "rotation used despite an improving translation",
                ],
            },
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
