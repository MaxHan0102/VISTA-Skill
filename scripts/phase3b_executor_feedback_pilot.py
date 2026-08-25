"""Run one paired Phase3B executor feedback/history ablation arm."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Iterable

from vista_skill.artifacts import JsonlArtifactWriter
from vista_skill.integrations.embodiedbench.cli import _make_planner
from vista_skill.integrations.embodiedbench.environment import (
    create_habitat_env,
    seed_habitat_env,
    seed_process_rngs,
)
from vista_skill.integrations.embodiedbench.runner import HabitatRolloutRunner
from vista_skill.protocol import load_experiment_manifest
from vista_skill.skills import initialize_shared_skill, skill_digest


ARMS = {
    "C0_current": (True, 0),
    "C1_no_feedback": (False, 0),
    "C2_temporal_no_feedback": (False, 3),
    "C3_temporal_feedback": (True, 3),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument(
        "--manifest", default="configs/eb_hab_train_validation_manifest.json"
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--n-shots", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: Iterable[float]) -> float:
    items = tuple(float(value) for value in values)
    return sum(items) / len(items) if items else 0.0


def _repetition_counts(trajectory: tuple[str, ...]) -> tuple[int, int, int]:
    adjacent = max(0, len(trajectory) - 1)
    repeats = sum(left == right for left, right in zip(trajectory, trajectory[1:]))
    triple_loops = sum(
        trajectory[index] == trajectory[index - 1] == trajectory[index - 2]
        for index in range(2, len(trajectory))
    )
    return repeats, adjacent, triple_loops


def main() -> int:
    args = _args()
    if args.output.exists() or args.summary.exists():
        raise FileExistsError("refusing to overwrite a Phase3B arm artifact")
    if not 1 <= args.episodes <= 60:
        raise ValueError("--episodes must be in [1, 60] for the acquisition role")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    use_feedback, multistep = ARMS[args.arm]
    manifest = load_experiment_manifest(args.manifest)
    coordinates = manifest.coordinates_for("acquisition")[: args.episodes]
    episode_ids = tuple(item.episode_id for item in coordinates)
    if len(episode_ids) != args.episodes:
        raise RuntimeError("manifest acquisition role is smaller than requested pilot")

    # RemoteModel reads this variable when the planner is constructed.
    os.environ["remote_url"] = args.base_url
    seed_process_rngs(args.seed)
    runtime_args = argparse.Namespace(
        env="eb-hab",
        model_name=args.model,
        model_type="remote",
        n_shots=args.n_shots,
        tp=1,
    )
    skill = initialize_shared_skill()
    run_id = uuid.uuid4().hex
    env = create_habitat_env(
        "train_validation",
        episode_ids=episode_ids,
        exp_name=f"phase3b_{args.arm}_{run_id}",
        resolution=args.resolution,
    )
    started = time.monotonic()
    try:
        seed_habitat_env(env, args.seed)
        planner = _make_planner(
            runtime_args,
            env,
            skill,
            engine=None,
            inject_skill=True,
            rollout_seed=args.seed,
        )
        # All arms receive the same frozen S0 and an empty, static ledger. No
        # evidence runtime is constructed, so feedback cannot leak through the
        # compact belief prompt in the no-feedback arms.
        planner.use_feedback = use_feedback
        planner.multistep = multistep
        runner = HabitatRolloutRunner(
            env,
            planner,
            engine=None,
            writer=JsonlArtifactWriter(args.output),
            expected_episode_ids=episode_ids,
            task_coordinates=coordinates,
        )
        results = runner.run(max_episodes=len(episode_ids))
        usage = dict(getattr(planner, "_vista_executor_usage", {}))
    finally:
        env.close()

    repeat_count = 0
    adjacent_count = 0
    triple_loop_count = 0
    for result in results:
        repeats, adjacent, triples = _repetition_counts(result.trajectory)
        repeat_count += repeats
        adjacent_count += adjacent
        triple_loop_count += triples
    environment_steps = sum(item.environment_steps for item in results)
    summary = {
        "analysis_type": "phase3b_executor_feedback_history_arm",
        "arm": args.arm,
        "condition": {
            "current_rgb": True,
            "action_history": True,
            "environment_feedback": use_feedback,
            "rgb_history_frames": 1 if multistep == 0 else multistep,
            "skill": "frozen_S0",
            "evidence_ledger": False,
            "evolution": False,
        },
        "model": args.model,
        "base_url": args.base_url,
        "temperature": 0.0,
        "max_completion_tokens": 4096,
        "n_shots": args.n_shots,
        "resolution": args.resolution,
        "rollout_seed": args.seed,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": manifest.digest,
        "skill_sha256": skill_digest(skill),
        "episode_ids": list(episode_ids),
        "episode_count": len(results),
        "mean_task_success": _mean(item.task_success for item in results),
        "mean_task_progress": _mean(item.task_progress for item in results),
        "total_environment_steps": environment_steps,
        "total_invalid_actions": sum(item.invalid_actions for item in results),
        "invalid_action_rate": (
            sum(item.invalid_actions for item in results) / environment_steps
            if environment_steps
            else 0.0
        ),
        "adjacent_repetition_count": repeat_count,
        "adjacent_opportunity_count": adjacent_count,
        "adjacent_repetition_rate": (
            repeat_count / adjacent_count if adjacent_count else 0.0
        ),
        "triple_loop_count": triple_loop_count,
        "planner_output_errors": sum(item.planner_output_errors for item in results),
        "executor_usage": usage,
        "elapsed_seconds": time.monotonic() - started,
        "events": str(args.output.resolve()),
        "events_sha256": _sha256(args.output),
        "episodes": [
            {
                "episode_id": item.episode_id,
                "task_success": item.task_success,
                "task_progress": item.task_progress,
                "environment_steps": item.environment_steps,
                "planner_steps": item.planner_steps,
                "invalid_actions": item.invalid_actions,
                "planner_output_errors": item.planner_output_errors,
                "trajectory": list(item.trajectory),
                "failure_reason": item.failure_reason,
            }
            for item in results
        ],
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "arm",
        "episode_count",
        "mean_task_success",
        "mean_task_progress",
        "invalid_action_rate",
        "adjacent_repetition_rate",
        "executor_usage",
        "elapsed_seconds",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
