from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Sequence

from vista_skill.schemas import (
    ActionCall,
    EvidenceSource,
    PredicateEvidence,
    PredicateKey,
    PredicateState,
    TruthValue,
    unique_strings,
)


class GuardMode(str, Enum):
    STRICT = "strict"
    AUTHORITY_AWARE = "authority_aware"


@dataclass(frozen=True)
class EvidenceGuardConfig:
    mode: GuardMode = GuardMode.STRICT
    min_visual_confidence: float = 0.75
    min_visual_coverage: float = 0.50
    reject_failed_action_visual_only: bool = True
    visual_negative_requires_feedback: tuple[str, ...] = ("at",)
    require_action_local_visual_only: bool = True

    def __post_init__(self) -> None:
        for name in ("min_visual_confidence", "min_visual_coverage"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class EvidenceGuardDecision:
    key: PredicateKey
    decision: str
    output: TruthValue
    source_ids: tuple[str, ...]
    source_values: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class EvidenceGuardResult:
    evidence: tuple[PredicateEvidence, ...]
    decisions: tuple[EvidenceGuardDecision, ...]


class EvidenceReliabilityGuard:
    """Prediction-blind reliability fusion for feedback and visual evidence."""

    def __init__(self, config: EvidenceGuardConfig | None = None) -> None:
        self.config = config or EvidenceGuardConfig()

    def fuse(
        self,
        feedback: Sequence[PredicateEvidence],
        visual: Sequence[PredicateEvidence],
        pre_ledger: Sequence[PredicateState] = (),
        *,
        last_action_success: bool | None = None,
        action: ActionCall | None = None,
    ) -> EvidenceGuardResult:
        feedback = (
            *feedback,
            *_action_local_feedback_extensions(
                feedback,
                pre_ledger,
                last_action_success=last_action_success,
                action=action,
            ),
        )
        feedback_by_key = _group(feedback)
        visual_by_key = _group(visual)
        keys = tuple(sorted(set(feedback_by_key) | set(visual_by_key)))
        outputs: list[PredicateEvidence] = []
        decisions: list[EvidenceGuardDecision] = []
        for key in keys:
            result, decision = self._fuse_key(
                key,
                feedback_by_key.get(key, ()),
                visual_by_key.get(key, ()),
                last_action_success=last_action_success,
                action=action,
            )
            if result is not None:
                outputs.append(result)
            decisions.append(decision)
        return EvidenceGuardResult(tuple(outputs), tuple(decisions))

    def _fuse_key(
        self,
        key: PredicateKey,
        feedback: Sequence[PredicateEvidence],
        visual: Sequence[PredicateEvidence],
        *,
        last_action_success: bool | None,
        action: ActionCall | None,
    ) -> tuple[PredicateEvidence | None, EvidenceGuardDecision]:
        feedback_definite = _definite_values(feedback)
        eligible_visual = tuple(
            item
            for item in visual
            if item.confidence >= self.config.min_visual_confidence
            and item.coverage >= self.config.min_visual_coverage
        )
        visual_definite = _definite_values(eligible_visual)
        sources = (*feedback, *visual)

        if len(feedback_definite) > 1:
            return _unknown(
                key,
                sources,
                "feedback_internal_conflict",
                "feedback packets disagree on polarity",
            )
        if len(visual_definite) > 1:
            return _unknown(
                key,
                sources,
                "visual_internal_conflict",
                "eligible visual packets disagree on polarity",
            )

        feedback_value = next(iter(feedback_definite), None)
        visual_value = next(iter(visual_definite), None)
        if feedback_value is not None and visual_value is not None:
            if feedback_value is visual_value:
                support = tuple(
                    item
                    for item in (*feedback, *eligible_visual)
                    if item.after is feedback_value
                )
                merged = _accepted(
                    key,
                    support,
                    source=EvidenceSource.ACTIVE_OBSERVATION,
                    rationale="independent feedback and visual evidence agree",
                )
                return merged, _decision(
                    key,
                    "accepted_cross_source_agreement",
                    feedback_value,
                    support,
                    "feedback and visual polarity agree",
                )
            if self.config.mode is GuardMode.AUTHORITY_AWARE:
                support = tuple(item for item in feedback if item.after is feedback_value)
                accepted = _accepted(
                    key,
                    support,
                    source=EvidenceSource.ENV_FEEDBACK,
                    rationale="explicit action-local feedback overrides conflicting vision",
                )
                return accepted, _decision(
                    key,
                    "accepted_feedback_priority_conflict",
                    feedback_value,
                    sources,
                    "feedback and visual polarity conflict; explicit feedback retained",
                )
            return _unknown(
                key,
                sources,
                "downgraded_cross_source_conflict",
                "feedback and visual polarity conflict",
            )

        if feedback_value is not None:
            support = tuple(item for item in feedback if item.after is feedback_value)
            accepted = _accepted(
                key,
                support,
                source=EvidenceSource.ENV_FEEDBACK,
                rationale="definite structured feedback with no eligible visual contradiction",
            )
            return accepted, _decision(
                key,
                "accepted_feedback",
                feedback_value,
                sources,
                "structured feedback is definite",
            )

        if visual_value is not None:
            allowed_visual = {
                "nav": {"near"},
                "pick": {"holding", "not_holding"},
                "place": {"at", "holding", "not_holding"},
                "open": {"open"},
                "close": {"open"},
            }
            if (
                self.config.require_action_local_visual_only
                and action is not None
                and key.name not in allowed_visual.get(action.action_type, set())
            ):
                return _unknown(
                    key,
                    sources,
                    "downgraded_action_irrelevant_visual_only",
                    "visual-only predicate is not local to the executed action",
                )
            if (
                last_action_success is False
                and self.config.reject_failed_action_visual_only
            ):
                return _unknown(
                    key,
                    sources,
                    "downgraded_failed_action_visual_only",
                    (
                        "the action failed and no structured feedback supports the "
                        "visual-only state assertion"
                    ),
                )
            if (
                visual_value is TruthValue.FALSE
                and key.name in self.config.visual_negative_requires_feedback
            ):
                return _unknown(
                    key,
                    sources,
                    "downgraded_visual_negative_without_feedback",
                    (
                        "a fine-grained relation absence is not accepted from vision "
                        "without structured feedback"
                    ),
                )
            support = tuple(item for item in eligible_visual if item.after is visual_value)
            accepted = _accepted(
                key,
                support,
                source=EvidenceSource.VISUAL_PAIR,
                rationale="visual-only evidence passed confidence and coverage thresholds",
            )
            return accepted, _decision(
                key,
                "accepted_visual_only",
                visual_value,
                sources,
                "eligible visual evidence is definite and feedback is uncovered",
            )

        if sources:
            return _unknown(
                key,
                sources,
                "downgraded_insufficient_evidence",
                "no source supplied an eligible definite value",
            )
        return None, EvidenceGuardDecision(
            key,
            "uncovered",
            TruthValue.UNKNOWN,
            (),
            (),
            "no evidence packets",
        )


def _group(
    items: Sequence[PredicateEvidence],
) -> dict[PredicateKey, tuple[PredicateEvidence, ...]]:
    grouped: dict[PredicateKey, list[PredicateEvidence]] = {}
    for item in items:
        grouped.setdefault(item.key, []).append(item)
    return {key: tuple(values) for key, values in grouped.items()}


def _action_local_feedback_extensions(
    feedback: Sequence[PredicateEvidence],
    pre_ledger: Sequence[PredicateState],
    *,
    last_action_success: bool | None,
    action: ActionCall | None,
) -> tuple[PredicateEvidence, ...]:
    """Ground explicit successful-place feedback without consulting a Skill.

    EB-Habitat already reports whether the high-level action executed.  When a
    successful place follows a definite held-object ledger state, the action
    target and held object identify the resulting ``at`` relation.  This is
    fixed action-local environment evidence, not a learned/predicted delta.
    """

    timestamp = max(
        (item.timestamp for item in (*feedback, *pre_ledger)),
        default=0,
    )
    output = []
    if last_action_success is False:
        covered = {item.key for item in feedback}
        for state in pre_ledger:
            if state.key in covered or state.value is TruthValue.UNKNOWN:
                continue
            output.append(
                PredicateEvidence(
                    key=state.key,
                    before=state.value,
                    after=state.value,
                    confidence=state.confidence,
                    source=EvidenceSource.ACTIVE_OBSERVATION,
                    evidence_id=(
                        f"guard:failed_action_persistence:{state.key.render()}:"
                        f"{timestamp}"
                    ),
                    timestamp=timestamp,
                    view_id=state.view_id,
                    coverage=state.coverage,
                    task_relevance=state.task_relevance,
                    rationale=(
                        "failed high-level action did not apply; retain the definite "
                        "pre-action state when feedback supplies no replacement"
                    ),
                )
            )
        return tuple(output)
    if (
        last_action_success is not True
        or action is None
        or action.action_type != "place"
        or not action.arguments
    ):
        return ()
    target = action.arguments[-1]
    for state in pre_ledger:
        if (
            state.key.name != "holding"
            or not state.key.arguments
            or state.value is not TruthValue.TRUE
        ):
            continue
        key = PredicateKey("at", (state.key.arguments[0], target))
        output.append(
            PredicateEvidence(
                key=key,
                before=TruthValue.UNKNOWN,
                after=TruthValue.TRUE,
                confidence=0.98,
                source=EvidenceSource.ENV_FEEDBACK,
                evidence_id=f"guard:successful_place:{key.render()}:{timestamp}",
                timestamp=timestamp,
                coverage=1.0,
                rationale=(
                    "successful place feedback grounded with the definite "
                    "pre-action held object and action target"
                ),
            )
        )
    return tuple(output)


def _definite_values(items: Sequence[PredicateEvidence]) -> set[TruthValue]:
    return {
        item.after
        for item in items
        if item.after in {TruthValue.TRUE, TruthValue.FALSE}
    }


def _accepted(
    key: PredicateKey,
    support: Sequence[PredicateEvidence],
    *,
    source: EvidenceSource,
    rationale: str,
) -> PredicateEvidence:
    best = max(support, key=lambda item: (item.confidence, item.coverage))
    return replace(
        best,
        key=key,
        source=source,
        evidence_id="|".join(unique_strings(item.evidence_id for item in support)),
        confidence=max(item.confidence for item in support),
        coverage=max(item.coverage for item in support),
        rationale=rationale,
    )


def _unknown(
    key: PredicateKey,
    sources: Sequence[PredicateEvidence],
    decision: str,
    rationale: str,
) -> tuple[PredicateEvidence, EvidenceGuardDecision]:
    best = max(sources, key=lambda item: (item.confidence, item.coverage))
    item = replace(
        best,
        key=key,
        after=TruthValue.UNKNOWN,
        confidence=0.5,
        source=EvidenceSource.ACTIVE_OBSERVATION,
        evidence_id="|".join(unique_strings(item.evidence_id for item in sources)),
        coverage=max((item.coverage for item in sources), default=0.0),
        rationale=rationale,
    )
    return item, _decision(key, decision, TruthValue.UNKNOWN, sources, rationale)


def _decision(
    key: PredicateKey,
    decision: str,
    output: TruthValue,
    sources: Sequence[PredicateEvidence],
    rationale: str,
) -> EvidenceGuardDecision:
    return EvidenceGuardDecision(
        key=key,
        decision=decision,
        output=output,
        source_ids=unique_strings(item.evidence_id for item in sources),
        source_values=tuple(
            f"{item.source.value}:{item.after.value}:{item.confidence:.3f}:{item.coverage:.3f}"
            for item in sources
        ),
        rationale=rationale,
    )
