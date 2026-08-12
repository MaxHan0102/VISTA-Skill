from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from vista_skill.schemas import (
    AbstainReason,
    AttributionContext,
    AttributionResult,
    DeltaSource,
    Mismatch,
    MismatchKind,
    SkillField,
    UpdateTarget,
    unique_strings,
)


class AttributionTeacher(Protocol):
    def assign(
        self,
        mismatches: Sequence[Mismatch],
        context: AttributionContext,
    ) -> AttributionResult: ...


@dataclass(frozen=True)
class AttributionConfig:
    min_evidence_confidence: float = 0.75
    min_teacher_confidence: float = 0.70
    action_model_updates_enabled: bool = False


class CreditAssigner:
    def __init__(
        self,
        teacher: AttributionTeacher | None = None,
        config: AttributionConfig | None = None,
    ) -> None:
        self.teacher = teacher
        self.config = config or AttributionConfig()

    def assign(
        self,
        mismatches: Sequence[Mismatch],
        context: AttributionContext | None = None,
    ) -> AttributionResult:
        context = context or AttributionContext()
        mismatch_ids = tuple(item.mismatch_id for item in mismatches)
        evidence_ids = unique_strings(
            tuple(evidence_id for item in mismatches for evidence_id in item.evidence_ids)
        )

        if not mismatches:
            return self._abstain(
                mismatch_ids, evidence_ids, AbstainReason.AMBIGUOUS, "no mismatch"
            )
        if any(
            item.kind in {MismatchKind.UNCOVERED, MismatchKind.EXPECTED_UNSUPPORTED}
            or item.evidence is None
            or item.evidence.confidence < self.config.min_evidence_confidence
            for item in mismatches
        ):
            return self._abstain(
                mismatch_ids,
                evidence_ids,
                AbstainReason.INSUFFICIENT_EVIDENCE,
                "a required predicate is unknown, uncovered, or low confidence",
                confidence=0.98,
            )
        if context.executor_followed_skill is False:
            return self._abstain(
                mismatch_ids,
                evidence_ids,
                AbstainReason.EXECUTION_LAPSE,
                "the canonical skill already contains the applicable rule",
                confidence=0.95,
            )
        if context.stochastic_suspected:
            return self._abstain(
                mismatch_ids,
                evidence_ids,
                AbstainReason.STOCHASTIC_NOOP,
                "the event is marked as a possible stochastic or no-op execution",
                confidence=0.9,
            )
        if context.identity_conflict or any(
            item.kind in {MismatchKind.IDENTITY_CONFLICT, MismatchKind.TEMPORAL_CONFLICT}
            for item in mismatches
        ):
            return AttributionResult(
                target=UpdateTarget.BELIEF_REFRESH,
                confidence=0.95,
                mismatch_ids=mismatch_ids,
                evidence_ids=evidence_ids,
                rationale="the conflict is local to instance identity or episode history",
            )

        action_predictions = [
            item
            for item in mismatches
            if item.expected is not None
            and item.expected.source is DeltaSource.ACTION_SCHEMA
            and item.kind in {MismatchKind.CONTRADICTION, MismatchKind.MISSING_PROGRESS}
        ]
        if action_predictions and not self.config.action_model_updates_enabled:
            return self._abstain(
                mismatch_ids,
                evidence_ids,
                AbstainReason.ACTION_MODEL_DISABLED,
                "the mismatch implicates fixed action knowledge, so it cannot identify a persistent Skill defect",
                confidence=0.9,
            )

        skill_predictions = [
            item.expected
            for item in mismatches
            if item.expected is not None and item.expected.source is DeltaSource.SKILL
        ]
        fields = {item.skill_field for item in skill_predictions if item.skill_field is not None}
        if len(fields) == 1:
            return AttributionResult(
                target=UpdateTarget.SKILL_UPDATE,
                field=next(iter(fields)),
                confidence=min(item.evidence.confidence for item in mismatches if item.evidence),
                mismatch_ids=mismatch_ids,
                evidence_ids=evidence_ids,
                rationale="independent evidence contradicts a skill-sourced prediction",
            )

        if all(item.kind is MismatchKind.SUPPORTED_UNEXPECTED for item in mismatches):
            return AttributionResult(
                target=UpdateTarget.BELIEF_REFRESH,
                confidence=0.8,
                mismatch_ids=mismatch_ids,
                evidence_ids=evidence_ids,
                rationale="new supported state is local episode evidence",
            )

        if self.teacher is not None:
            result = self.teacher.assign(mismatches, context)
            if (
                result.confidence >= self.config.min_teacher_confidence
                and self._valid_teacher_provenance(
                    result,
                    mismatch_ids=mismatch_ids,
                    evidence_ids=evidence_ids,
                    mismatches=mismatches,
                )
            ):
                return result
        return self._abstain(
            mismatch_ids,
            evidence_ids,
            AbstainReason.AMBIGUOUS,
            "rules and constrained teacher did not establish a unique update target",
        )

    @staticmethod
    def _abstain(
        mismatch_ids: tuple[str, ...],
        evidence_ids: tuple[str, ...],
        reason: AbstainReason,
        rationale: str,
        confidence: float = 0.7,
    ) -> AttributionResult:
        return AttributionResult(
            target=UpdateTarget.ABSTAIN,
            subreason=reason,
            confidence=confidence,
            mismatch_ids=mismatch_ids,
            evidence_ids=evidence_ids,
            rationale=rationale,
        )

    @staticmethod
    def _valid_teacher_provenance(
        result: AttributionResult,
        *,
        mismatch_ids: tuple[str, ...],
        evidence_ids: tuple[str, ...],
        mismatches: Sequence[Mismatch],
    ) -> bool:
        cited_mismatches = set(result.mismatch_ids)
        cited_evidence = set(result.evidence_ids)
        if (
            not cited_mismatches
            or not cited_evidence
            or not cited_mismatches.issubset(mismatch_ids)
            or not cited_evidence.issubset(evidence_ids)
        ):
            return False
        selected = [
            mismatch
            for mismatch in mismatches
            if mismatch.mismatch_id in cited_mismatches
        ]
        selected_evidence = {
            evidence_id
            for mismatch in selected
            for evidence_id in mismatch.evidence_ids
        }
        if not cited_evidence.issubset(selected_evidence):
            return False
        if result.target is not UpdateTarget.SKILL_UPDATE:
            return result.target in {
                UpdateTarget.BELIEF_REFRESH,
                UpdateTarget.ABSTAIN,
            }
        if result.field is None:
            return False
        skill_fields = {
            mismatch.expected.skill_field
            for mismatch in selected
            if mismatch.expected is not None
            and mismatch.expected.source is DeltaSource.SKILL
            and mismatch.expected.skill_field is not None
        }
        # A direct skill-sourced diagnosis must agree with its compiled source.
        # If there is no skill-sourced prediction, the field is an explicit
        # diagnostic-teacher fallback and remains bound to the cited symptom.
        return not skill_fields or skill_fields == {result.field}
