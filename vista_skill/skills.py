from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from vista_skill.schemas import (
    SkillField,
    SkillPredictionRule,
    SkillSpec,
    TerminationPolicy,
    TruthValue,
    dataclass_to_dict,
)


SKILL_ARTIFACT_SCHEMA_VERSION = "2"


@dataclass(frozen=True)
class SkillArtifact:
    """Digest-checked Skill plus the controlled protocol that produced it."""

    schema_version: str
    skill: SkillSpec
    protocol: Mapping[str, Any]
    artifact_sha256: str

    @property
    def skill_sha256(self) -> str:
        return skill_digest(self.skill)


def initialize_shared_skill() -> SkillSpec:
    """Benchmark-aware, task-agnostic S0 shared by controlled methods."""
    return SkillSpec(
        skill_id="shared_embodied_execution",
        version=0,
        activation=(
            "Use for multi-step embodied tasks that require state verification.",
        ),
        procedure=(
            "Decompose the instruction into goal predicates and track pending and completed targets.",
            "Check action preconditions and update the checklist only from new evidence.",
            "Re-observe or replan when the evidence needed for the next step is insufficient.",
        ),
        effect=(
            "Each step should make progress toward one or more unsatisfied goal predicates.",
        ),
        termination=(
            "Stop only when every required goal predicate is supported by current evidence.",
        ),
        constraint=(
            "Unknown is not false.",
            "Success of one subgoal is not evidence that the full task is complete.",
        ),
        termination_policy=TerminationPolicy.ALL_GOALS_EVIDENCE,
        prediction_rules=(
            SkillPredictionRule(
                "procedure_navigation_reaches_target",
                SkillField.PROCEDURE,
                "nav",
                "near({arg0})",
                TruthValue.TRUE,
            ),
            SkillPredictionRule(
                "effect_pick_holds_target_category",
                SkillField.EFFECT,
                "pick",
                "holding({arg0})",
                TruthValue.TRUE,
            ),
            SkillPredictionRule(
                "constraint_pick_occupies_gripper",
                SkillField.CONSTRAINT,
                "pick",
                "not_holding",
                TruthValue.FALSE,
            ),
            SkillPredictionRule(
                "effect_place_relates_held_object",
                SkillField.EFFECT,
                "place",
                "at({held},{arg0})",
                TruthValue.TRUE,
            ),
            SkillPredictionRule(
                "constraint_place_frees_gripper",
                SkillField.CONSTRAINT,
                "place",
                "not_holding",
                TruthValue.TRUE,
            ),
            SkillPredictionRule(
                "effect_open_changes_articulation",
                SkillField.EFFECT,
                "open",
                "open({arg0})",
                TruthValue.TRUE,
            ),
            SkillPredictionRule(
                "effect_close_changes_articulation",
                SkillField.EFFECT,
                "close",
                "open({arg0})",
                TruthValue.FALSE,
            ),
        ),
        metadata={"initialization": "benchmark-aware-task-agnostic"},
    )


def minimal_shared_skill() -> SkillSpec:
    """Minimal S0 variant: the weakest reasonable starting point for evolution.

    A drop-in alternative to :func:`initialize_shared_skill` for the §4.2.3
    initialization-sensitivity controlled comparison. It carries only what is
    structurally required to be a valid, runnable Skill against the fixed
    nav/pick/place/open/close action schema: a single generic activation, a
    one-line fallback procedure, a valid termination statement, and empty
    effect/constraint bodies. Compiled ``prediction_rules`` are intentionally
    empty so the engine falls back to primitive action-schema transitions only,
    giving the attribution/evolution loop the weakest prior to grow from.
    """
    return SkillSpec(
        skill_id="shared_embodied_execution",
        version=0,
        activation=(
            "Use for embodied tasks that require step-wise execution.",
        ),
        procedure=(
            "Execute the instructed primitive action and observe its result.",
        ),
        effect=(),
        termination=(
            "Stop when the instructed goal is reached.",
        ),
        constraint=(),
        termination_policy=TerminationPolicy.ALL_GOALS_EVIDENCE,
        prediction_rules=(),
        metadata={"initialization": "minimal"},
    )


def empty_shared_skill() -> SkillSpec:
    """Empty S0 variant: the absolute lower bound for init-sensitivity study.

    Every statement body is empty and there are no compiled ``prediction_rules``,
    yet the object remains a structurally valid :class:`SkillSpec` (``skill_id``,
    ``version`` 0, ``termination_policy`` present) so that
    :meth:`FixedActionSchema.compile` and :class:`VistaSkillEngine` accept it
    without raising. Represents the "no prior task knowledge" extreme of the
    §4.2.3 controlled comparison; keep teacher and evolution budget matched
    against :func:`initialize_shared_skill` and :func:`minimal_shared_skill`.
    """
    return SkillSpec(
        skill_id="shared_embodied_execution",
        version=0,
        activation=(),
        procedure=(),
        effect=(),
        termination=(),
        constraint=(),
        termination_policy=TerminationPolicy.ALL_GOALS_EVIDENCE,
        prediction_rules=(),
        metadata={"initialization": "empty"},
    )


def render_skill(skill: SkillSpec, *, max_statements_per_field: int | None = None) -> str:
    parts = [f"Skill: {skill.skill_id} (v{skill.version})"]
    for skill_field in SkillField:
        statements = skill.statements(skill_field)
        if max_statements_per_field is not None:
            statements = statements[:max_statements_per_field]
        parts.append(f"{skill_field.value.title()}:")
        parts.extend(f"- {statement}" for statement in statements)
    return "\n".join(parts)


def skill_digest(skill: SkillSpec) -> str:
    payload = json.dumps(dataclass_to_dict(skill), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def skill_artifact_digest(
    schema_version: str,
    skill: SkillSpec,
    protocol: Mapping[str, Any],
) -> str:
    payload = json.dumps(
        {
            "schema_version": str(schema_version),
            "skill": dataclass_to_dict(skill),
            "protocol": dict(protocol),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_protocol(protocol: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize protocol metadata to the exact JSON types stored on disk."""
    payload = json.dumps(dict(protocol or {}), sort_keys=True, separators=(",", ":"))
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("protocol metadata must serialize to a JSON object")
    return value


def skill_from_dict(raw: Mapping[str, Any]) -> SkillSpec:
    rules = tuple(
        SkillPredictionRule(
            rule_id=str(item["rule_id"]),
            field=SkillField(str(item["field"])),
            action_type=str(item["action_type"]),
            predicate=str(item["predicate"]),
            after=TruthValue(str(item["after"])),
            before=None
            if item.get("before") is None
            else TruthValue(str(item["before"])),
        )
        for item in raw.get("prediction_rules", ())
    )
    return SkillSpec(
        skill_id=str(raw["skill_id"]),
        version=int(raw["version"]),
        activation=tuple(str(item) for item in raw["activation"]),
        procedure=tuple(str(item) for item in raw["procedure"]),
        effect=tuple(str(item) for item in raw["effect"]),
        termination=tuple(str(item) for item in raw["termination"]),
        constraint=tuple(str(item) for item in raw["constraint"]),
        termination_policy=TerminationPolicy(str(raw["termination_policy"])),
        prediction_rules=rules,
        parent_version=None
        if raw.get("parent_version") is None
        else int(raw["parent_version"]),
        frozen=bool(raw.get("frozen", False)),
        metadata=dict(raw.get("metadata", {})),
    )


def save_skill_artifact(
    path: str | Path,
    skill: SkillSpec,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> None:
    protocol_payload = canonical_protocol(protocol)
    payload = {
        "schema_version": SKILL_ARTIFACT_SCHEMA_VERSION,
        "skill": dataclass_to_dict(skill),
        "skill_sha256": skill_digest(skill),
        "protocol": protocol_payload,
        "artifact_sha256": skill_artifact_digest(
            SKILL_ARTIFACT_SCHEMA_VERSION,
            skill,
            protocol_payload,
        ),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_content_addressed_skill(
    directory: str | Path,
    skill: SkillSpec,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> Path:
    output = Path(directory) / f"{skill_digest(skill)}.json"
    if not output.exists():
        save_skill_artifact(output, skill, protocol=protocol)
    else:
        record = load_skill_artifact_record(output)
        if record.skill != skill or dict(record.protocol) != canonical_protocol(protocol):
            raise ValueError("content-addressed Skill artifact collision")
    return output


def load_skill_artifact_record(
    path: str | Path,
    *,
    require_frozen: bool = False,
) -> SkillArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema_version = str(payload.get("schema_version", ""))
    if schema_version != SKILL_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported skill artifact schema version: {schema_version or 'missing'}"
        )
    skill = skill_from_dict(payload["skill"])
    if payload.get("skill_sha256") != skill_digest(skill):
        raise ValueError("skill artifact digest mismatch")
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("skill artifact protocol metadata must be an object")
    expected_digest = skill_artifact_digest(schema_version, skill, protocol)
    if payload.get("artifact_sha256") != expected_digest:
        raise ValueError("skill artifact envelope digest mismatch")
    if require_frozen and not skill.frozen:
        raise ValueError("evaluation requires a frozen skill artifact")
    return SkillArtifact(
        schema_version=schema_version,
        skill=skill,
        protocol=dict(protocol),
        artifact_sha256=expected_digest,
    )


def load_skill_artifact(path: str | Path, *, require_frozen: bool = False) -> SkillSpec:
    """Load only the Skill for compatibility with the original artifact API."""

    return load_skill_artifact_record(path, require_frozen=require_frozen).skill


def with_field(
    skill: SkillSpec,
    skill_field: SkillField,
    statements: Iterable[str],
    *,
    frozen: bool = False,
) -> SkillSpec:
    values = tuple(statements)
    return replace(
        skill,
        **{
            skill_field.value: values,
            "version": skill.version + 1,
            "parent_version": skill.version,
            "frozen": frozen,
        },
    )
