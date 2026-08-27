#!/usr/bin/env python3
"""Summarize a complete stock EmbodiedBench EB-Navigation baseline run."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Sequence


SUBSETS = (
    "base",
    "common_sense",
    "complex_instruction",
    "visual_appearance",
    "long_horizon",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode_index(path: Path) -> int:
    return int(path.name.removeprefix("episode_").removesuffix("_final_res.json"))


def _load_subset(
    path: Path, expected_episodes: int
) -> tuple[list[dict[str, Any]], list[Path]]:
    files = sorted(path.glob("episode_*_final_res.json"), key=_episode_index)
    indexes = [_episode_index(item) for item in files]
    expected_indexes = list(range(1, expected_episodes + 1))
    if indexes != expected_indexes:
        raise ValueError(
            f"{path} is incomplete or non-contiguous: expected {expected_indexes}, got {indexes}"
        )
    return [json.loads(item.read_text(encoding="utf-8")) for item in files], files


def _mean(records: Sequence[dict[str, Any]], field: str) -> float:
    return statistics.fmean(float(item[field]) for item in records)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        help=(
            "Optional fresh stock run root used for visual_appearance and "
            "long_horizon after an interruption."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=60)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(
            "EmbodiedBench/embodiedbench/envs/eb_navigation/datasets"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.expected_episodes < 1:
        raise ValueError("--expected-episodes must be positive")

    subset_run_dirs = {
        subset: (
            args.resume_run_dir
            if args.resume_run_dir is not None
            and subset in {"visual_appearance", "long_horizon"}
            else args.run_dir
        )
        for subset in SUBSETS
    }
    summaries = {}
    for subset in SUBSETS:
        subset_dir = subset_run_dirs[subset] / subset
        records, files = _load_subset(
            subset_dir / "results", args.expected_episodes
        )
        config_path = subset_dir / "config.txt"
        dataset_path = args.dataset_dir / f"{subset}.json"
        if not config_path.is_file() or not dataset_path.is_file():
            raise FileNotFoundError(
                f"missing config or dataset for {subset}: {config_path}, {dataset_path}"
            )
        summaries[subset] = {
            "episodes": len(records),
            "successes": int(sum(float(item["task_success"]) for item in records)),
            "mean_task_success": _mean(records, "task_success"),
            "mean_reward": _mean(records, "reward"),
            "mean_environment_steps": _mean(records, "num_steps"),
            "mean_planner_steps": _mean(records, "planner_steps"),
            "total_planner_output_errors": int(
                sum(int(item["planner_output_error"]) for item in records)
            ),
            "mean_episode_elapsed_seconds": _mean(
                records, "episode_elapsed_seconds"
            ),
            "episode_result_sha256": {
                item.name: _sha256(item) for item in files
            },
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "dataset_path": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
        }

    report = {
        "protocol": "stock_embodiedbench_eb_nav_qwen3vl8b_full_v1_2026_08_27",
        "experiment_name": args.run_dir.name,
        "claim_scope": "stock EmbodiedBench baseline; no VISTA-Skill components",
        "subset_run_dirs": {
            subset: str(path) for subset, path in subset_run_dirs.items()
        },
        "expected_episodes_per_subset": args.expected_episodes,
        "total_episodes": sum(item["episodes"] for item in summaries.values()),
        "subsets": summaries,
        "macro_average_task_success": statistics.fmean(
            item["mean_task_success"] for item in summaries.values()
        ),
        "latex_order": {
            "columns": ["Avg.", "Base", "Com.", "Comp.", "Vis.", "Long"],
            "values": [
                statistics.fmean(
                    item["mean_task_success"] for item in summaries.values()
                ),
                *[summaries[subset]["mean_task_success"] for subset in SUBSETS],
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
