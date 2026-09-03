#!/usr/bin/env python3
"""Apply the frozen semantic finalist rule to a paired shadow audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from vista_skill.config import load_config
from vista_skill.evolution import (
    CandidateGate,
    GateStageResult,
    PairedEpisodeScore,
    shadow_candidate_eligible,
)
from vista_skill.integrations.embodiedbench.cli import (
    _load_verified_manifest,
    load_habitat_task_semantic_tags,
)


REPO = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_shadow_confirmation(
    stage: GateStageResult,
    *,
    protected_tolerance: float,
    subgroup_tolerance: float,
) -> str:
    if stage.passed:
        return "supportive"
    metrics = stage.metrics
    contradicted = bool(
        float(metrics.get("affected_mean_delta", float("-inf"))) <= 0.0
        or float(metrics.get("protected_mean_delta", float("-inf")))
        < -protected_tolerance
        or float(metrics.get("worst_subgroup_delta", float("-inf")))
        < -subgroup_tolerance
    )
    return "contradictory" if contradicted else "inconclusive"


def analyze(
    protocol: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    config = load_config(REPO / str(protocol["config"]))
    manifest = _load_verified_manifest(str(REPO / str(protocol["manifest"])))
    indexed = {item.episode_id: item for item in manifest.tasks}
    dataset = (
        REPO
        / "EmbodiedBench/embodiedbench/envs/eb_habitat/datasets"
        / manifest.dataset
    )
    semantic_tags = load_habitat_task_semantic_tags(dataset)
    scores = tuple(
        PairedEpisodeScore(
            episode_id=str(parent["episode_id"]),
            seed=int(parent["seed"]),
            parent_score=float(parent["score"]),
            candidate_score=float(candidate["score"]),
            subgroup=indexed[str(parent["episode_id"])].subgroup,
            parent_success=bool(parent["task_success"]),
            candidate_success=bool(candidate["task_success"]),
            semantic_tags=semantic_tags[str(parent["episode_id"])],
        )
        for parent in records
        if parent["arm"] == "parent"
        for candidate in records
        if candidate["arm"] == "candidate"
        and candidate["episode_id"] == parent["episode_id"]
        and int(candidate["seed"]) == int(parent["seed"])
    )
    expected = len(protocol["episode_ids"]) * len(protocol["all_rollout_seeds"])
    if len(scores) != expected:
        raise ValueError(f"expected {expected} paired scores, found {len(scores)}")
    gate = CandidateGate(object(), object(), object(), config.gate)  # type: ignore[arg-type]
    stage = gate._paired_stage(
        "paired_finalist",
        scores,
        config.gate.finalist_lcb_threshold,
        expected,
        tuple(str(item) for item in protocol["semantic_scope"]),
    )
    verdict = classify_shadow_confirmation(
        stage,
        protected_tolerance=config.gate.semantic_protected_regression_tolerance,
        subgroup_tolerance=config.gate.subgroup_regression_tolerance,
    )
    return {
        "verdict": verdict,
        "stage": {
            "stage": stage.stage,
            "passed": stage.passed,
            "reason": stage.reason,
            "metrics": dict(stage.metrics),
        },
        "would_remain_shadow_if_underpowered": bool(
            not stage.passed and shadow_candidate_eligible(stage, config.gate)
        ),
        "candidate_promoted": False,
        "promotion_eligible_from_this_diagnostic": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_path = Path(args.protocol).resolve()
    records_path = Path(args.records).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    records = json.loads(records_path.read_text(encoding="utf-8"))["records"]
    result = {
        "diagnostic": True,
        "claim_eligible": False,
        "protocol_sha256": _sha256(protocol_path),
        "records_sha256": _sha256(records_path),
        "analysis": analyze(protocol, records),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
