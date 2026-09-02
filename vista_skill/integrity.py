from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from vista_skill.schemas import SkillSpec


DEFAULT_EVALUATION_DATA_POLICY = (
    Path(__file__).parents[1] / "configs" / "evaluation_data_policy_v1.json"
)


@dataclass(frozen=True)
class EvaluationDataPolicy:
    policy_id: str
    schema_version: int
    digest: str
    raw: Mapping[str, Any]


def load_evaluation_data_policy(
    path: str | Path = DEFAULT_EVALUATION_DATA_POLICY,
) -> EvaluationDataPolicy:
    source = Path(path)
    raw_bytes = source.read_bytes()
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict):
        raise ValueError("evaluation data policy must be a JSON object")
    schema_version = raw.get("schema_version")
    policy_id = raw.get("policy_id")
    if schema_version != 1:
        raise ValueError("unsupported evaluation data policy schema")
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("evaluation data policy requires a non-empty policy_id")
    return EvaluationDataPolicy(
        policy_id=policy_id,
        schema_version=schema_version,
        digest=hashlib.sha256(raw_bytes).hexdigest(),
        raw=raw,
    )


def artifact_contamination_reasons(
    skill: SkillSpec,
    protocol: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return auditable reasons an artifact is ineligible for controlled claims.

    Historical diagnostic artifacts remain loadable and reproducible with an
    explicit diagnostic run.  This classifier prevents them from silently
    crossing into a controlled EmbodiedBench result.
    """

    reasons: list[str] = []
    metadata = skill.metadata
    if metadata.get("oracle_diagnostic") is True:
        reasons.append("skill metadata marks a post-hoc oracle diagnostic")
    if metadata.get("source_trajectory_contamination") is True:
        reasons.append("skill metadata marks source-trajectory contamination")
    if metadata.get("initialization") == "posthoc-human-trajectory-synthesis":
        reasons.append("skill was initialized by post-hoc human trajectory synthesis")
    source_split = str(metadata.get("source_split", "")).strip().lower()
    if source_split.startswith("official_test"):
        reasons.append("skill content was derived from an official-test split")
    if protocol.get("source_trajectory_contamination") is True:
        reasons.append("artifact protocol marks source-trajectory contamination")
    analysis_type = str(protocol.get("analysis_type", "")).strip().lower()
    if analysis_type.startswith("posthoc") or analysis_type.startswith("post-hoc"):
        reasons.append("artifact protocol marks post-hoc analysis")
    return tuple(dict.fromkeys(reasons))
