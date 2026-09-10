#!/usr/bin/env python3
"""Summarize matched No/Static/Target Skill validation rollouts."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from vista_skill.evaluation import composite_task_score


def _load_results(path: Path) -> dict[str, dict[str, Any]]:
    results = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event_type") == "episode_result":
                payload = dict(record["payload"])
                results[str(payload["episode_id"])] = payload
    if not results:
        raise ValueError(f"no completed episodes in {path}")
    return results


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_score(item: Mapping[str, Any]) -> float:
    return composite_task_score(
        task_success=float(item["task_success"]),
        task_progress=float(item["task_progress"]),
        invalid_action_ratio=float(item["invalid_actions"])
        / max(1, int(item["environment_steps"])),
    )


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _arm_summary(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    values = tuple(results.values())
    total_steps = sum(int(item["environment_steps"]) for item in values)
    invalid = sum(int(item["invalid_actions"]) for item in values)
    successful = tuple(item for item in values if float(item["task_success"]) > 0)
    return {
        "episodes": len(values),
        "total_environment_steps": total_steps,
        "total_planner_steps": sum(int(item["planner_steps"]) for item in values),
        "total_invalid_actions": invalid,
        "mean_task_success": _mean([float(item["task_success"]) for item in values]),
        "mean_task_progress": _mean([float(item["task_progress"]) for item in values]),
        "mean_composite_score": _mean([_episode_score(item) for item in values]),
        "invalid_action_ratio": invalid / total_steps if total_steps else 0.0,
        "mean_environment_steps": _mean(
            [float(item["environment_steps"]) for item in values]
        ),
        "mean_planner_steps": _mean([float(item["planner_steps"]) for item in values]),
        "mean_steps_when_successful": _mean(
            [float(item["environment_steps"]) for item in successful]
        ),
    }


def _bootstrap(
    deltas: Sequence[float], *, samples: int = 10_000, seed: int = 0
) -> dict[str, float]:
    rng = random.Random(seed)
    draws = sorted(
        statistics.fmean(rng.choice(deltas) for _ in deltas)
        for _ in range(samples)
    )
    return {
        "mean_delta": statistics.fmean(deltas),
        "ci95_low": draws[int(samples * 0.025)],
        "ci95_high": draws[min(samples - 1, int(samples * 0.975))],
    }


def _mcnemar_exact(wins: int, losses: int) -> float:
    discordant = wins + losses
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, index) * 0.5**discordant
        for index in range(min(wins, losses) + 1)
    )
    return min(1.0, 2.0 * tail)


def _paired(
    baseline: Mapping[str, Mapping[str, Any]],
    target: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    episode_ids = sorted(set(baseline) & set(target))
    success_deltas = [
        float(target[key]["task_success"]) - float(baseline[key]["task_success"])
        for key in episode_ids
    ]
    progress_deltas = [
        float(target[key]["task_progress"]) - float(baseline[key]["task_progress"])
        for key in episode_ids
    ]
    composite_deltas = [
        _episode_score(target[key]) - _episode_score(baseline[key])
        for key in episode_ids
    ]
    invalid_ratio_deltas = [
        float(target[key]["invalid_actions"])
        / max(1, int(target[key]["environment_steps"]))
        - float(baseline[key]["invalid_actions"])
        / max(1, int(baseline[key]["environment_steps"]))
        for key in episode_ids
    ]
    environment_step_deltas = [
        float(target[key]["environment_steps"])
        - float(baseline[key]["environment_steps"])
        for key in episode_ids
    ]
    planner_step_deltas = [
        float(target[key]["planner_steps"])
        - float(baseline[key]["planner_steps"])
        for key in episode_ids
    ]
    wins = sum(delta > 0 for delta in success_deltas)
    losses = sum(delta < 0 for delta in success_deltas)
    return {
        "paired_episodes": len(episode_ids),
        "success_wins": wins,
        "success_losses": losses,
        "success_ties": len(episode_ids) - wins - losses,
        "mcnemar_exact_p": _mcnemar_exact(wins, losses),
        "task_success": _bootstrap(success_deltas),
        "task_progress": _bootstrap(progress_deltas),
        "composite_score": _bootstrap(composite_deltas),
        "invalid_action_ratio": _bootstrap(invalid_ratio_deltas),
        "environment_steps": _bootstrap(environment_step_deltas),
        "planner_steps": _bootstrap(planner_step_deltas),
        "improved_episode_ids": [
            key for key, delta in zip(episode_ids, success_deltas) if delta > 0
        ],
        "regressed_episode_ids": [
            key for key, delta in zip(episode_ids, success_deltas) if delta < 0
        ],
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


def _nav_raw_dir(
    *,
    summary: Mapping[str, Any],
    raw_root: Path,
    subset: str,
) -> Path:
    skill_sha = summary.get("skill_sha256")
    skill_dir = str(skill_sha)[:12] if skill_sha else "no_skill"
    return (
        raw_root
        / str(summary["run_id"])
        / str(summary["stage"])
        / subset
        / "s0"
        / skill_dir
    )


def _nav_raw_summary(
    raw_dir: Path, results: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
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
        episodes.append(
            {
                "episode_id": episode_id,
                "minimum_distance": min(distances),
                "final_distance": distances[-1],
                "worsening_steps": worsening,
                "stagnant_steps": stagnant,
                "max_open_loop_plan_actions": max(plan_run_lengths),
            }
        )

    def mean(field: str) -> float:
        return _mean([float(item[field]) for item in episodes])

    return {
        "raw_trajectory_dir": str(raw_dir),
        "mean_minimum_distance": mean("minimum_distance"),
        "mean_final_distance": mean("final_distance"),
        "mean_worsening_steps": mean("worsening_steps"),
        "mean_stagnant_steps": mean("stagnant_steps"),
        "mean_max_open_loop_plan_actions": mean("max_open_loop_plan_actions"),
        "episodes_with_plan_at_least_3_actions": sum(
            item["max_open_loop_plan_actions"] >= 3 for item in episodes
        ),
        "episodes_with_exactly_single_step_plans": sum(
            item["max_open_loop_plan_actions"] == 1 for item in episodes
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="running/Phase4/target_skill_oracle_v1/validation_common_sense_seed0",
    )
    parser.add_argument(
        "--hab-run-dir",
        help="HAB result root; defaults to --run-dir.",
    )
    parser.add_argument(
        "--nav-run-dir",
        help="NAV result root; defaults to --run-dir.",
    )
    parser.add_argument(
        "--nav-raw-root",
        default="running/eb_nav/vista_skill/evaluate",
    )
    parser.add_argument("--subset", default="common_sense")
    parser.add_argument("--expected-episodes", type=int, default=20)
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    env_roots = {
        "eb_hab": Path(args.hab_run_dir) if args.hab_run_dir else run_dir,
        "eb_nav": Path(args.nav_run_dir) if args.nav_run_dir else run_dir,
    }
    output = Path(args.output) if args.output else run_dir / "analysis.json"
    if output.exists():
        raise FileExistsError(output)
    report: dict[str, Any] = {
        "protocol": "vista_target_skill_heldout_subset_analysis_v1_2026_08_27",
        "diagnostic": True,
        "source_split": "official_test/base",
        "evaluation_split": f"official_test/{args.subset}",
        "evaluation_subset_used_in_synthesis": False,
        "result_roots": {key: str(value) for key, value in env_roots.items()},
        "environments": {},
    }
    macro_success = {arm: [] for arm in ("no_skill", "static_shared_skill", "target_skill")}
    for env_name in ("eb_hab", "eb_nav"):
        env_root = env_roots[env_name]
        arms = {
            arm: _load_results(
                env_root / env_name / args.subset / arm / "events.jsonl"
            )
            for arm in ("no_skill", "static_shared_skill", "target_skill")
        }
        if any(len(results) != args.expected_episodes for results in arms.values()):
            counts = {arm: len(results) for arm, results in arms.items()}
            raise ValueError(
                f"{env_name} does not have {args.expected_episodes} episodes per arm: "
                f"{counts}"
            )
        episode_sets = {frozenset(results) for results in arms.values()}
        if len(episode_sets) != 1:
            raise ValueError(f"{env_name} arms do not contain matched episode IDs")
        summaries = {arm: _arm_summary(results) for arm, results in arms.items()}
        evaluations = {}
        for arm, summary in summaries.items():
            evaluation = _load_json(
                env_root
                / env_name
                / args.subset
                / arm
                / "events.summary.json"
            )
            evaluations[arm] = evaluation
            runtime = _load_json(
                env_root
                / env_name
                / args.subset
                / arm
                / "runtime_manifest.json"
            )
            summary["elapsed_seconds"] = float(runtime["elapsed_seconds"])
            summary["seconds_per_episode"] = float(runtime["elapsed_seconds"]) / len(
                arms[arm]
            )
            summary["run_id"] = str(evaluation["run_id"])
            summary["dataset_sha256"] = str(evaluation["dataset_sha256"])
            summary["skill_sha256"] = evaluation.get("skill_sha256")
        dataset_hashes = {
            str(evaluation["dataset_sha256"])
            for evaluation in evaluations.values()
        }
        if len(dataset_hashes) != 1:
            raise ValueError(f"{env_name} arms use different dataset artifacts")
        if env_name == "eb_nav":
            for arm, results in arms.items():
                raw_dir = _nav_raw_dir(
                    summary=evaluations[arm],
                    raw_root=Path(args.nav_raw_root),
                    subset=args.subset,
                )
                summaries[arm]["raw_navigation"] = _nav_raw_summary(raw_dir, results)
        for arm, summary in summaries.items():
            macro_success[arm].append(summary["mean_task_success"])
        report["environments"][env_name] = {
            "arms": summaries,
            "static_shared_skill_minus_no_skill": _paired(
                arms["no_skill"], arms["static_shared_skill"]
            ),
            "target_minus_no_skill": _paired(arms["no_skill"], arms["target_skill"]),
            "target_minus_static_shared_skill": _paired(
                arms["static_shared_skill"], arms["target_skill"]
            ),
        }
    report["macro_average_success"] = {
        arm: _mean(values) for arm, values in macro_success.items()
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
