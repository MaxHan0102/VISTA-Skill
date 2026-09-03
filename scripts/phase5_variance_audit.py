"""P5.6 paired-rollout audit for a registered temporal Skill candidate.

This is a diagnostic, not an automatic promotion decision. It can reuse
registered rollouts, run new registered coordinates, and decompose paired score
variance into between-task and within-task components.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from vista_skill.config import load_config
from vista_skill.evaluation import EpisodeCoordinate, composite_task_score
from vista_skill.integrations.embodiedbench.cli import (
    _load_verified_manifest,
    _make_audit_evaluator,
)
from vista_skill.integrations.embodiedbench.planner import ExecutorUsageTracker
from vista_skill.skills import load_skill_artifact_record, skill_digest


REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPO / "configs/phase5_p56_variance_audit.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO / value


def _episode_payload(path: Path, *, episode_id: str) -> dict[str, Any]:
    result = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event_type") == "episode_result":
                result = record.get("payload")
    if not isinstance(result, dict):
        raise ValueError(f"missing episode_result: {path}")
    if str(result.get("episode_id")) != str(episode_id):
        raise ValueError(f"episode id mismatch in {path}")
    return result


def _trajectory_signature(payload: Mapping[str, Any]) -> str:
    stable = {
        "task_success": float(payload["task_success"]),
        "task_progress": float(payload["task_progress"]),
        "environment_steps": int(payload["environment_steps"]),
        "planner_steps": int(payload["planner_steps"]),
        "invalid_actions": int(payload["invalid_actions"]),
        "planner_output_errors": int(payload["planner_output_errors"]),
        "temporal_guard_blocks": int(payload.get("temporal_guard_blocks", 0)),
        "trajectory": list(payload.get("trajectory", [])),
        "failure_reason": str(payload.get("failure_reason", "")),
    }
    canonical = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _record(
    payload: Mapping[str, Any],
    *,
    episode_id: str,
    seed: int,
    arm: str,
    path: Path,
) -> dict[str, Any]:
    steps = int(payload["environment_steps"])
    invalid = int(payload["invalid_actions"])
    success = float(payload["task_success"])
    progress = float(payload["task_progress"])
    return {
        "episode_id": str(episode_id),
        "seed": int(seed),
        "arm": arm,
        "score": composite_task_score(
            task_success=success,
            task_progress=progress,
            invalid_action_ratio=invalid / max(1, steps),
        ),
        "task_success": success,
        "task_progress": progress,
        "environment_steps": steps,
        "invalid_actions": invalid,
        "temporal_guard_blocks": int(payload.get("temporal_guard_blocks", 0)),
        "trajectory_signature": _trajectory_signature(payload),
        "artifact": str(path.relative_to(REPO)),
        "artifact_sha256": _sha256(path),
    }


def _sample_variance(values: Sequence[float]) -> float:
    return statistics.variance(values) if len(values) > 1 else 0.0


def analyze_records(
    records: Sequence[Mapping[str, Any]],
    *,
    episode_ids: Sequence[str],
    seeds: Sequence[int],
    variance_share_threshold: float,
    minimum_delta_divergent_tasks: int,
) -> dict[str, Any]:
    indexed = {
        (str(item["episode_id"]), int(item["seed"]), str(item["arm"])): item
        for item in records
    }
    expected = {
        (str(episode_id), int(seed), arm)
        for episode_id in episode_ids
        for seed in seeds
        for arm in ("parent", "candidate")
    }
    missing = sorted(expected - set(indexed))
    if missing:
        raise ValueError(f"missing registered rollout records: {missing}")

    paired_rows = []
    per_task: dict[str, Any] = {}
    for episode_id in episode_ids:
        task_deltas = []
        parent_signatures = set()
        candidate_signatures = set()
        parent_successes = []
        candidate_successes = []
        for seed in seeds:
            parent = indexed[(str(episode_id), int(seed), "parent")]
            candidate = indexed[(str(episode_id), int(seed), "candidate")]
            delta = float(candidate["score"]) - float(parent["score"])
            task_deltas.append(delta)
            parent_signatures.add(str(parent["trajectory_signature"]))
            candidate_signatures.add(str(candidate["trajectory_signature"]))
            parent_successes.append(float(parent["task_success"]))
            candidate_successes.append(float(candidate["task_success"]))
            paired_rows.append(
                {
                    "episode_id": str(episode_id),
                    "seed": int(seed),
                    "parent_score": float(parent["score"]),
                    "candidate_score": float(candidate["score"]),
                    "delta": delta,
                    "parent_success": float(parent["task_success"]),
                    "candidate_success": float(candidate["task_success"]),
                }
            )
        unique_deltas = {
            round(value, 12) for value in task_deltas
        }
        per_task[str(episode_id)] = {
            "mean_delta": statistics.fmean(task_deltas),
            "within_task_delta_variance": _sample_variance(task_deltas),
            "unique_delta_count": len(unique_deltas),
            "parent_unique_trajectory_count": len(parent_signatures),
            "candidate_unique_trajectory_count": len(candidate_signatures),
            "parent_pass_at_k": float(any(parent_successes)),
            "candidate_pass_at_k": float(any(candidate_successes)),
        }

    task_means = [float(per_task[item]["mean_delta"]) for item in episode_ids]
    within = statistics.fmean(
        float(per_task[item]["within_task_delta_variance"]) for item in episode_ids
    )
    between = _sample_variance(task_means)
    variance_share = within / (within + between) if within + between > 0.0 else 0.0
    delta_divergent = sum(
        int(per_task[item]["unique_delta_count"] > 1) for item in episode_ids
    )
    parent_trajectory_divergent = sum(
        int(per_task[item]["parent_unique_trajectory_count"] > 1)
        for item in episode_ids
    )
    candidate_trajectory_divergent = sum(
        int(per_task[item]["candidate_unique_trajectory_count"] > 1)
        for item in episode_ids
    )
    repeated_path = bool(
        variance_share >= variance_share_threshold
        and delta_divergent >= minimum_delta_divergent_tasks
    )

    arm_summary = {}
    for arm in ("parent", "candidate"):
        arm_records = [item for item in records if item["arm"] == arm]
        arm_summary[arm] = {
            "coordinate_success_rate": statistics.fmean(
                float(item["task_success"]) for item in arm_records
            ),
            "coordinate_mean_progress": statistics.fmean(
                float(item["task_progress"]) for item in arm_records
            ),
            "coordinate_mean_score": statistics.fmean(
                float(item["score"]) for item in arm_records
            ),
            "pass_at_k": statistics.fmean(
                float(per_task[item][f"{arm}_pass_at_k"]) for item in episode_ids
            ),
        }

    return {
        "episode_count": len(episode_ids),
        "rollout_seed_count": len(seeds),
        "paired_coordinate_count": len(paired_rows),
        "paired_mean_delta": statistics.fmean(
            float(item["delta"]) for item in paired_rows
        ),
        "task_first_mean_delta": statistics.fmean(task_means),
        "within_task_delta_variance": within,
        "between_task_mean_delta_variance": between,
        "within_task_variance_share": variance_share,
        "delta_divergent_tasks": delta_divergent,
        "parent_trajectory_divergent_tasks": parent_trajectory_divergent,
        "candidate_trajectory_divergent_tasks": candidate_trajectory_divergent,
        "thresholds": {
            "within_task_variance_share": variance_share_threshold,
            "minimum_delta_divergent_tasks": minimum_delta_divergent_tasks,
        },
        "decision": (
            "repeated_rollout_path" if repeated_path else "independent_task_path"
        ),
        "pass_at_k_is_promotion_eligible": False,
        "arm_summary": arm_summary,
        "per_task": per_task,
        "paired_rows": paired_rows,
    }


def _artifact_path(
    directory: Path,
    *,
    episode_id: str,
    seed: int,
    skill_sha256: str,
) -> Path:
    return directory / str(episode_id) / f"s{seed}_{skill_sha256}.jsonl"


def _load_registered_records(
    protocol: Mapping[str, Any],
    *,
    parent_sha256: str,
    candidate_sha256: str,
) -> list[dict[str, Any]]:
    source_value = protocol.get("source_seed0_rollouts")
    source = None if source_value is None else _resolve(str(source_value))
    output = _resolve(str(protocol["output_dir"])) / "update_audit_rollouts"
    reused = {int(item) for item in protocol["reused_rollout_seeds"]}
    reused_audit_sources = {
        int(seed): _resolve(str(directory))
        for seed, directory in protocol.get(
            "reused_audit_rollout_sources", {}
        ).items()
    }
    records = []
    for episode_id in protocol["episode_ids"]:
        for seed in protocol["all_rollout_seeds"]:
            for arm, digest in (
                ("parent", parent_sha256),
                ("candidate", candidate_sha256),
            ):
                if int(seed) in reused:
                    if source is None:
                        raise ValueError("reused rollout seeds require a source directory")
                    path = source / f"{episode_id}_s{seed}_{digest[:10]}.jsonl"
                elif int(seed) in reused_audit_sources:
                    path = _artifact_path(
                        reused_audit_sources[int(seed)],
                        episode_id=str(episode_id),
                        seed=int(seed),
                        skill_sha256=digest,
                    )
                else:
                    path = _artifact_path(
                        output,
                        episode_id=str(episode_id),
                        seed=int(seed),
                        skill_sha256=digest,
                    )
                payload = _episode_payload(path, episode_id=str(episode_id))
                records.append(
                    _record(
                        payload,
                        episode_id=str(episode_id),
                        seed=int(seed),
                        arm=arm,
                        path=path,
                    )
                )
    return records


def _preregistration(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    *,
    parent_artifact: Path,
    candidate_artifact: Path,
) -> dict[str, Any]:
    return {
        "protocol": protocol,
        "protocol_file": str(protocol_path.relative_to(REPO)),
        "protocol_sha256": _sha256(protocol_path),
        "driver": str(Path(__file__).resolve().relative_to(REPO)),
        "driver_sha256": _sha256(Path(__file__).resolve()),
        "parent_artifact": str(parent_artifact.relative_to(REPO)),
        "parent_artifact_sha256": _sha256(parent_artifact),
        "candidate_artifact": str(candidate_artifact.relative_to(REPO)),
        "candidate_artifact_sha256": _sha256(candidate_artifact),
        "outcomes_observed_when_written": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument(
        "--executor-base-url",
        default=os.environ.get("VISTA_METHOD_BASE_URL"),
        required=os.environ.get("VISTA_METHOD_BASE_URL") is None,
    )
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = _read_json(protocol_path)
    output = _resolve(str(protocol["output_dir"]))
    source_proposal = _resolve(str(protocol["source_proposal"]))
    parent_path = source_proposal / "parent_skill.json"
    candidate_path = source_proposal / "candidate_skill.json"
    parent = load_skill_artifact_record(parent_path).skill
    candidate = load_skill_artifact_record(candidate_path).skill
    parent_sha256 = skill_digest(parent)
    candidate_sha256 = skill_digest(candidate)

    output.mkdir(parents=True, exist_ok=True)
    prereg_path = output / "preregistration.json"
    expected_prereg = _preregistration(
        protocol_path,
        protocol,
        parent_artifact=parent_path,
        candidate_artifact=candidate_path,
    )
    if prereg_path.exists():
        if _read_json(prereg_path) != expected_prereg:
            raise ValueError("existing preregistration differs from the current protocol")
    elif args.analyze_only:
        raise FileNotFoundError("analyze-only requires an existing preregistration")
    else:
        _write_json(prereg_path, expected_prereg)

    usage = ExecutorUsageTracker()
    if not args.analyze_only:
        config = load_config(_resolve(str(protocol["config"])))
        manifest = _load_verified_manifest(str(_resolve(str(protocol["manifest"]))))
        selection_ids = {
            item.episode_id for item in manifest.coordinates_for("selection")
        }
        requested_ids = tuple(str(item) for item in protocol["episode_ids"])
        if any(item not in selection_ids for item in requested_ids):
            raise ValueError("variance audit episode is outside the registered selection role")
        indexed = {item.episode_id: item for item in manifest.tasks}
        coordinates = tuple(
            EpisodeCoordinate(
                episode_id=episode_id,
                seed=int(seed),
                subgroup=indexed[episode_id].subgroup,
                semantic_tags=(),
            )
            for episode_id in requested_ids
            for seed in protocol["new_rollout_seeds"]
        )
        executor = protocol["executor"]
        if float(executor["temperature"]) != 0.0:
            raise ValueError("this frozen diagnostic requires temperature=0")
        os.environ["remote_url"] = str(args.executor_base_url)
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        runner_args = SimpleNamespace(
            env="eb-hab",
            model_name=str(executor["model"]),
            model_type="remote",
            n_shots=int(executor["n_shots"]),
            resolution=int(executor["resolution"]),
            tp=int(executor["tensor_parallel"]),
            meta_skills="none",
            state_oracle_labels=False,
        )
        evaluator = _make_audit_evaluator(
            runner_args,
            manifest,
            coordinates,
            output,
            config,
            run_id=str(protocol["protocol_id"]),
            usage_tracker=usage,
        )
        evaluator.evaluate(
            parent,
            candidate,
            stage="audit",
            episode_budget=len(coordinates),
        )

    records = _load_registered_records(
        protocol,
        parent_sha256=parent_sha256,
        candidate_sha256=candidate_sha256,
    )
    rule = protocol["decision_rule"]
    analysis = analyze_records(
        records,
        episode_ids=tuple(str(item) for item in protocol["episode_ids"]),
        seeds=tuple(int(item) for item in protocol["all_rollout_seeds"]),
        variance_share_threshold=float(rule["within_task_variance_share_threshold"]),
        minimum_delta_divergent_tasks=int(rule["minimum_delta_divergent_tasks"]),
    )
    result = {
        "protocol_id": protocol["protocol_id"],
        "diagnostic": True,
        "claim_eligible": False,
        "parent_skill_sha256": parent_sha256,
        "candidate_skill_sha256": candidate_sha256,
        "analysis": analysis,
        "new_rollout_executor_usage": usage.payload(),
        "record_count": len(records),
    }
    _write_json(output / "records.json", {"records": records})
    _write_json(output / "analysis.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
