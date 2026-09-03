from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Sequence

from vista_skill.schemas import (
    AbstainReason,
    AttributionContext,
    AttributionResult,
    DeltaSource,
    IdentifiabilityAudit,
    Mismatch,
    MismatchKind,
    SkillField,
    SkillUpdateKind,
    TruthValue,
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
    skill_discovery_enabled: bool = False
    identifiability_required: bool = True


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
        if context.executor_followed_skill is False and not any(
            item.kind in {MismatchKind.CONTRADICTION, MismatchKind.MISSING_PROGRESS}
            and item.expected is not None
            and item.expected.source is DeltaSource.SKILL
            for item in mismatches
        ):
            # The lapse veto assumes the canonical skill is correct. When a
            # covered mismatch refutes a skill-sourced expectation, the
            # executor acted on a rule the skill itself claims -- following a
            # wrong constraint looks like a lapse against the world while it
            # is evidence against the rule (E8h: 12/15 not_holding
            # contradictions of the multihold fault were masked this way).
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
        if self._completion_supported_by_evidence(mismatches, context):
            # "Skill expects not-complete, env says complete" WITH the goals
            # themselves confirmed true in this step's evidence is belief lag
            # (the ledger had not absorbed the achieved goals), not a policy
            # defect: routing it to skill_update(termination) poisons the
            # policy cluster with events no policy repair can verify.
            return AttributionResult(
                target=UpdateTarget.BELIEF_REFRESH,
                confidence=0.85,
                mismatch_ids=mismatch_ids,
                evidence_ids=evidence_ids,
                rationale=(
                    "completion goals are confirmed by current evidence but absent from "
                    "belief; refresh belief instead of blaming the termination policy"
                ),
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
            result = AttributionResult(
                target=UpdateTarget.SKILL_UPDATE,
                field=next(iter(fields)),
                confidence=min(item.evidence.confidence for item in mismatches if item.evidence),
                mismatch_ids=mismatch_ids,
                evidence_ids=evidence_ids,
                rationale="independent evidence contradicts a skill-sourced prediction",
            )
            return self._identified_update(result, mismatches, context)

        if all(item.kind is MismatchKind.SUPPORTED_UNEXPECTED for item in mismatches):
            discovery_items = tuple(
                item for item in mismatches if item.key.name != "task_complete"
            )
            if self.config.skill_discovery_enabled and discovery_items and context.action_type:
                field = (
                    SkillField.CONSTRAINT
                    if context.last_action_success is False
                    else SkillField.EFFECT
                )
                result = AttributionResult(
                    target=UpdateTarget.SKILL_UPDATE,
                    field=field,
                    update_kind=SkillUpdateKind.DISCOVERY,
                    confidence=min(
                        item.evidence.confidence
                        for item in discovery_items
                        if item.evidence is not None
                    ),
                    mismatch_ids=tuple(item.mismatch_id for item in discovery_items),
                    evidence_ids=unique_strings(
                        tuple(
                            evidence_id
                            for item in discovery_items
                            for evidence_id in item.evidence_ids
                        )
                    ),
                    rationale=(
                        "reliable action-bound outcomes are recurrent Skill "
                        "effect or precondition discovery candidates"
                    ),
                )
                return self._identified_update(result, discovery_items, context)
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
                if result.target is UpdateTarget.SKILL_UPDATE:
                    return self._identified_update(result, mismatches, context)
                return result
        return self._abstain(
            mismatch_ids,
            evidence_ids,
            AbstainReason.AMBIGUOUS,
            "rules and constrained teacher did not establish a unique update target",
        )

    def _identified_update(
        self,
        result: AttributionResult,
        mismatches: Sequence[Mismatch],
        context: AttributionContext,
    ) -> AttributionResult:
        """Attach and enforce the operational VTCA identifiability criterion.

        A persistent update is identifiable only when covered evidence has
        complete provenance, execution/stochastic/identity alternatives are
        ruled out, and the cited observations select one action-bound field.
        This turns the paper definition into an artifact-level invariant rather
        than a post-hoc description.
        """
        cited_mismatches = set(result.mismatch_ids)
        cited_evidence = set(result.evidence_ids)
        selected = tuple(
            item for item in mismatches if item.mismatch_id in cited_mismatches
        )
        evidence_sufficient = bool(selected) and all(
            item.evidence is not None
            and item.evidence.confidence >= self.config.min_evidence_confidence
            for item in selected
        )
        selected_evidence = {
            evidence_id for item in selected for evidence_id in item.evidence_ids
        }
        provenance_complete = bool(
            cited_mismatches
            and cited_evidence
            and {item.mismatch_id for item in selected} == cited_mismatches
            and cited_evidence.issubset(selected_evidence)
        )
        skill_refutation = any(
            item.expected is not None
            and item.expected.source is DeltaSource.SKILL
            and item.kind in {MismatchKind.CONTRADICTION, MismatchKind.MISSING_PROGRESS}
            for item in selected
        )
        fields = {
            item.expected.skill_field
            for item in selected
            if item.expected is not None
            and item.expected.source is DeltaSource.SKILL
            and item.expected.skill_field is not None
        }
        unique_update_target = bool(
            result.field is not None
            and (not fields or fields == {result.field})
        )
        action_bound = bool(
            context.action_type
            or any(
                item.expected is not None
                and item.expected.source is DeltaSource.SKILL
                for item in selected
            )
        )
        audit = IdentifiabilityAudit(
            evidence_sufficient=evidence_sufficient,
            provenance_complete=provenance_complete,
            executor_compliance_not_refuted=(
                context.executor_followed_skill is not False or skill_refutation
            ),
            stochasticity_ruled_out=not context.stochastic_suspected,
            identity_resolved=(
                not context.identity_conflict
                and not any(
                    item.kind
                    in {MismatchKind.IDENTITY_CONFLICT, MismatchKind.TEMPORAL_CONFLICT}
                    for item in selected
                )
            ),
            unique_update_target=unique_update_target,
            action_bound=action_bound,
        )
        if self.config.identifiability_required and not audit.identified:
            return self._abstain(
                tuple(result.mismatch_ids),
                tuple(result.evidence_ids),
                AbstainReason.AMBIGUOUS,
                "VTCA identifiability conditions did not isolate one persistent Skill update",
                confidence=result.confidence,
            )
        return replace(result, identifiability=audit)

    @staticmethod
    def _completion_supported_by_evidence(
        mismatches: Sequence[Mismatch],
        context: AttributionContext,
    ) -> bool:
        termination_affirmed = any(
            item.kind is MismatchKind.TERMINATION_CONFLICT
            and item.evidence is not None
            and item.evidence.after is TruthValue.TRUE
            for item in mismatches
        )
        if not termination_affirmed or not context.goal_predicates:
            return False
        return any(
            item.key in context.goal_predicates
            and item.evidence is not None
            and item.evidence.after is TruthValue.TRUE
            for item in mismatches
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
