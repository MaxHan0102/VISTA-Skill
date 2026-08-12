from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class TruthValue(StringEnum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class SkillField(StringEnum):
    ACTIVATION = "activation"
    PROCEDURE = "procedure"
    EFFECT = "effect"
    TERMINATION = "termination"
    CONSTRAINT = "constraint"


class TerminationPolicy(StringEnum):
    ALL_GOALS_EVIDENCE = "all_goals_evidence"
    ANY_GOAL_EVIDENCE = "any_goal_evidence"
    ACTION_SUCCESS = "action_success"


class DeltaSource(StringEnum):
    ACTION_SCHEMA = "action_schema"
    SKILL = "skill"


class EvidenceSource(StringEnum):
    ENV_FEEDBACK = "env_feedback"
    VISUAL_PAIR = "visual_pair"
    ACTIVE_OBSERVATION = "active_observation"
    DERIVED_GOAL = "derived_goal"


class MismatchKind(StringEnum):
    CONTRADICTION = "contradiction"
    EXPECTED_UNSUPPORTED = "expected_unsupported"
    SUPPORTED_UNEXPECTED = "supported_unexpected"
    MISSING_PROGRESS = "missing_progress"
    TERMINATION_CONFLICT = "termination_conflict"
    IDENTITY_CONFLICT = "identity_conflict"
    TEMPORAL_CONFLICT = "temporal_conflict"
    UNCOVERED = "uncovered"


class UpdateTarget(StringEnum):
    BELIEF_REFRESH = "belief_refresh"
    SKILL_UPDATE = "skill_update"
    ABSTAIN = "abstain"
    # Reserved for the full paper formulation. P0 keeps this target disabled.
    ACTION_MODEL_UPDATE = "action_model_update"


class AbstainReason(StringEnum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    EXECUTION_LAPSE = "execution_lapse"
    STOCHASTIC_NOOP = "stochastic_noop"
    AMBIGUOUS = "ambiguous"
    ACTION_MODEL_DISABLED = "action_model_disabled"


class PatchOperation(StringEnum):
    APPEND = "append"
    INSERT_AFTER_EXACT = "insert_after_exact"
    REPLACE_EXACT = "replace_exact"
    DELETE_EXACT = "delete_exact"


def _require_confidence(value: float, name: str = "confidence") -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True, order=True)
class PredicateKey:
    name: str
    arguments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or any(ch in self.name for ch in "(),"):
            raise ValueError(f"invalid predicate name: {self.name!r}")
        if any(not argument for argument in self.arguments):
            raise ValueError("predicate arguments cannot be empty")
        if any(any(ch in argument for ch in "(),") for argument in self.arguments):
            raise ValueError("predicate arguments cannot contain delimiters")

    def render(self) -> str:
        if not self.arguments:
            return self.name
        return f"{self.name}({','.join(self.arguments)})"

    @classmethod
    def parse(cls, value: str) -> "PredicateKey":
        value = value.strip()
        if "(" not in value:
            return cls(value)
        if not value.endswith(")"):
            raise ValueError(f"invalid predicate: {value!r}")
        name, raw_arguments = value[:-1].split("(", 1)
        arguments = tuple(part.strip() for part in raw_arguments.split(","))
        return cls(name.strip(), arguments)


@dataclass(frozen=True)
class PredicateState:
    key: PredicateKey
    value: TruthValue
    confidence: float
    source: str
    evidence_ids: tuple[str, ...]
    timestamp: int
    view_id: str | None = None
    coverage: float = 1.0
    task_relevance: float = 1.0

    def __post_init__(self) -> None:
        _require_confidence(self.confidence)
        _require_confidence(self.coverage, "coverage")
        _require_confidence(self.task_relevance, "task_relevance")
        if not self.source:
            raise ValueError("state source is required")
        if not self.evidence_ids:
            raise ValueError("at least one evidence id is required")


@dataclass(frozen=True)
class PredicateEvidence:
    key: PredicateKey
    before: TruthValue
    after: TruthValue
    confidence: float
    source: EvidenceSource
    evidence_id: str
    timestamp: int
    view_id: str | None = None
    coverage: float = 1.0
    task_relevance: float = 1.0
    rationale: str = ""

    def __post_init__(self) -> None:
        _require_confidence(self.confidence)
        _require_confidence(self.coverage, "coverage")
        _require_confidence(self.task_relevance, "task_relevance")
        if not self.evidence_id:
            raise ValueError("evidence_id is required")


@dataclass(frozen=True)
class ExpectedChange:
    key: PredicateKey
    before: TruthValue
    after: TruthValue
    source: DeltaSource
    source_id: str
    skill_field: SkillField | None = None

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("expected change requires source provenance")
        if self.source is DeltaSource.SKILL and self.skill_field is None:
            raise ValueError("skill predictions must cite a skill field")
        if self.source is DeltaSource.ACTION_SCHEMA and self.skill_field is not None:
            raise ValueError("action-schema predictions cannot cite a skill field")


@dataclass(frozen=True)
class SkillPredictionRule:
    rule_id: str
    field: SkillField
    action_type: str
    predicate: str
    after: TruthValue
    before: TruthValue | None = None

@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    version: int
    activation: tuple[str, ...]
    procedure: tuple[str, ...]
    effect: tuple[str, ...]
    termination: tuple[str, ...]
    constraint: tuple[str, ...]
    termination_policy: TerminationPolicy = TerminationPolicy.ALL_GOALS_EVIDENCE
    prediction_rules: tuple[SkillPredictionRule, ...] = ()
    parent_version: int | None = None
    frozen: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.skill_id:
            raise ValueError("skill_id is required")
        if self.version < 0:
            raise ValueError("skill version cannot be negative")
        if self.parent_version is not None and self.parent_version >= self.version:
            raise ValueError("parent version must precede the current version")
        for skill_field in SkillField:
            values = getattr(self, skill_field.value)
            if any(not item.strip() for item in values):
                raise ValueError(f"{skill_field.value} contains an empty statement")

    def statements(self, skill_field: SkillField) -> tuple[str, ...]:
        return getattr(self, skill_field.value)


@dataclass(frozen=True)
class ActionCall:
    action_id: int
    action_type: str
    arguments: tuple[str, ...]
    text: str
    raw_action: str | None = None


@dataclass(frozen=True)
class EvidenceRequest:
    episode_id: str
    step_id: int
    instruction: str
    action: ActionCall
    pre_image: str
    post_image: str
    feedback: str
    last_action_success: bool | None
    pre_ledger: tuple[PredicateState, ...]
    goal_predicates: tuple[PredicateKey, ...] = ()

    def to_provider_payload(self) -> dict[str, Any]:
        """Serialize only evidence-branch inputs; predictions cannot enter here."""
        return {
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "instruction": self.instruction,
            "action": dataclass_to_dict(self.action),
            "pre_image": self.pre_image,
            "post_image": self.post_image,
            "feedback": self.feedback,
            "last_action_success": self.last_action_success,
            "pre_ledger": [dataclass_to_dict(item) for item in self.pre_ledger],
            "goal_predicates": [item.render() for item in self.goal_predicates],
        }


@dataclass(frozen=True)
class Mismatch:
    mismatch_id: str
    key: PredicateKey
    kind: MismatchKind
    expected: ExpectedChange | None
    evidence: PredicateEvidence | None
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class AttributionContext:
    executor_followed_skill: bool | None = None
    stochastic_suspected: bool = False
    identity_conflict: bool = False
    task_pattern: str = "general"
    object_context: str = "general"
    instruction: str = ""
    action_type: str = ""
    skill_obligations: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class AttributionResult:
    target: UpdateTarget
    confidence: float
    mismatch_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    rationale: str
    field: SkillField | None = None
    subreason: AbstainReason | None = None
    independent_support_count: int = 1

    def __post_init__(self) -> None:
        _require_confidence(self.confidence)
        if self.target is UpdateTarget.SKILL_UPDATE and self.field is None:
            raise ValueError("skill update requires field attribution")
        if self.target is not UpdateTarget.SKILL_UPDATE and self.field is not None:
            raise ValueError("only skill updates can cite a skill field")
        if self.target is UpdateTarget.ABSTAIN and self.subreason is None:
            raise ValueError("abstention requires a reason")


@dataclass(frozen=True)
class TransitionEvent:
    event_id: str
    episode_id: str
    task_id: str
    step_id: int
    instruction: str
    action: ActionCall
    skill_id: str
    skill_version: int
    pre_image: str
    post_image: str
    feedback: str
    last_action_success: bool | None
    pre_ledger: tuple[PredicateState, ...]
    goal_predicates: tuple[PredicateKey, ...]
    expected_delta: tuple[ExpectedChange, ...]
    evidence_delta: tuple[PredicateEvidence, ...]
    mismatches: tuple[Mismatch, ...]
    attribution: AttributionResult | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillPatch:
    patch_id: str
    skill_id: str
    parent_version: int
    field: SkillField
    operation: PatchOperation
    old: str
    new: str
    evidence_ids: tuple[str, ...]
    scope: str
    rationale: str = ""
    termination_policy: TerminationPolicy | None = None
    prediction_rules: tuple[SkillPredictionRule, ...] | None = None


def dataclass_to_dict(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, PredicateKey):
        return {"name": value.name, "arguments": list(value.arguments)}
    if hasattr(value, "__dataclass_fields__"):
        return {key: dataclass_to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): dataclass_to_dict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [dataclass_to_dict(item) for item in value]
    return value


def unique_strings(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))
