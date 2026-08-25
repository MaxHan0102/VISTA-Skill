"""Analyze the paired four-arm Phase3B executor pilot."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence


ARMS = (
    "C0_current",
    "C1_no_feedback",
    "C2_temporal_no_feedback",
    "C3_temporal_feedback",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired vectors must be non-empty and equally sized")
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        indices = [rng.randrange(len(left)) for _ in left]
        deltas.append(
            sum(left[index] - right[index] for index in indices) / len(indices)
        )
    deltas.sort()
    return {
        "mean_delta": sum(a - b for a, b in zip(left, right)) / len(left),
        "ci95_lower": deltas[int(0.025 * (len(deltas) - 1))],
        "ci95_upper": deltas[int(0.975 * (len(deltas) - 1))],
    }


def main() -> int:
    args = _args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    summaries: dict[str, dict[str, Any]] = {}
    for path in args.summary:
        row = json.loads(path.read_text())
        summaries[str(row["arm"])] = row
    if set(summaries) != set(ARMS):
        raise ValueError(f"expected summaries for {ARMS}, got {sorted(summaries)}")
    reference_ids = summaries[ARMS[0]]["episode_ids"]
    for arm in ARMS[1:]:
        if summaries[arm]["episode_ids"] != reference_ids:
            raise ValueError("executor pilot episode order is not paired")

    progress = {
        arm: [float(item["task_progress"]) for item in summaries[arm]["episodes"]]
        for arm in ARMS
    }
    c0 = summaries["C0_current"]
    c2 = summaries["C2_temporal_no_feedback"]
    invalid_limit = max(
        1.25 * float(c0["invalid_action_rate"]),
        float(c0["invalid_action_rate"]) + 0.02,
    )
    repetition_limit = max(
        1.25 * float(c0["adjacent_repetition_rate"]),
        float(c0["adjacent_repetition_rate"]) + 0.02,
    )
    checks = {
        "c2_progress_strictly_above_c1": (
            float(c2["mean_task_progress"])
            > float(summaries["C1_no_feedback"]["mean_task_progress"])
        ),
        "c2_progress_within_0_10_of_c0": (
            float(c2["mean_task_progress"])
            >= float(c0["mean_task_progress"]) - 0.10
        ),
        "c2_invalid_rate_within_limit": (
            float(c2["invalid_action_rate"]) <= invalid_limit
        ),
        "c2_repetition_rate_within_limit": (
            float(c2["adjacent_repetition_rate"]) <= repetition_limit
        ),
    }
    result = {
        "analysis_type": "phase3b_executor_feedback_history_paired_pilot",
        "episode_count": len(reference_ids),
        "episode_ids": reference_ids,
        "paired": True,
        "arm_metrics": {
            arm: {
                key: summaries[arm][key]
                for key in (
                    "mean_task_success",
                    "mean_task_progress",
                    "invalid_action_rate",
                    "adjacent_repetition_rate",
                    "triple_loop_count",
                    "planner_output_errors",
                    "executor_usage",
                    "elapsed_seconds",
                )
            }
            for arm in ARMS
        },
        "paired_progress_bootstrap": {
            "C2_minus_C1": _paired_bootstrap(
                progress["C2_temporal_no_feedback"],
                progress["C1_no_feedback"],
                samples=args.bootstrap_samples,
                seed=args.seed,
            ),
            "C2_minus_C0": _paired_bootstrap(
                progress["C2_temporal_no_feedback"],
                progress["C0_current"],
                samples=args.bootstrap_samples,
                seed=args.seed + 1,
            ),
            "C3_minus_C0": _paired_bootstrap(
                progress["C3_temporal_feedback"],
                progress["C0_current"],
                samples=args.bootstrap_samples,
                seed=args.seed + 2,
            ),
        },
        "screening_limits": {
            "invalid_action_rate": invalid_limit,
            "adjacent_repetition_rate": repetition_limit,
        },
        "go_checks": checks,
        "decision": "go" if all(checks.values()) else "no_go",
        "limitations": [
            "Thirty paired tasks are a mechanism pilot; bootstrap intervals are descriptive.",
            "Adjacent repetition includes legitimate repeated navigation actions.",
            "The pilot tests frozen S0 prompting without an evidence ledger or evolution.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
