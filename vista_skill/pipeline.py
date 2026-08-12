from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Mapping

from vista_skill.action_schema import FixedActionSchema
from vista_skill.attribution import CreditAssigner
from vista_skill.belief import BeliefLedger
from vista_skill.clustering import EventClusterer
from vista_skill.evidence import EvidenceExtractor
from vista_skill.emphasis import ExecutionEmphasisBuffer
from vista_skill.mismatch import compare_transitions
from vista_skill.schemas import (
    ActionCall,
    AttributionContext,
    AbstainReason,
    EvidenceRequest,
    ExpectedChange,
    PredicateState,
    PredicateKey,
    SkillField,
    SkillSpec,
    TransitionEvent,
    UpdateTarget,
)
from vista_skill.skills import skill_digest


@dataclass(frozen=True)
class PrimitiveTransition:
    episode_id: str
    task_id: str
    step_id: int
    instruction: str
    action: ActionCall
    pre_image: str
    post_image: str
    feedback: str
    last_action_success: bool | None
    goal_predicates: tuple[PredicateKey, ...] = ()
    attribution_context: AttributionContext = AttributionContext()
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class PreparedTransition:
    episode_id: str
    task_id: str
    step_id: int
    instruction: str
    action: ActionCall
    pre_image: str
    pre_ledger: tuple[PredicateState, ...]
    skill_id: str
    skill_version: int
    goal_predicates: tuple[PredicateKey, ...]
    expected_delta: tuple[ExpectedChange, ...]
    attribution_context: AttributionContext
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class FrozenSkillArtifact:
    skill: SkillSpec
    digest: str


class VistaSkillEngine:
    """Orchestrates one action-level VTCA event without owning a simulator."""

    def __init__(
        self,
        skill: SkillSpec,
        *,
        action_schema: FixedActionSchema | None = None,
        evidence_extractor: EvidenceExtractor | None = None,
        credit_assigner: CreditAssigner | None = None,
        clusterer: EventClusterer | None = None,
        ledger: BeliefLedger | None = None,
        emphasis_buffer: ExecutionEmphasisBuffer | None = None,
    ) -> None:
        self.skill = skill
        self.action_schema = action_schema or FixedActionSchema()
        self.evidence_extractor = evidence_extractor or EvidenceExtractor()
        self.credit_assigner = credit_assigner or CreditAssigner()
        self.clusterer = clusterer or EventClusterer()
        self.ledger = ledger or BeliefLedger()
        self.emphasis_buffer = emphasis_buffer or ExecutionEmphasisBuffer()
        self._frozen = skill.frozen
        self._current_step = 0

    def start_episode(self) -> None:
        """Reset episode-local adaptive state while preserving the active Skill."""
        self.ledger = BeliefLedger(self.ledger.policy)
        self.emphasis_buffer.clear()

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def current_step(self) -> int:
        """Most recently processed step id, used for emphasis decay timing."""
        return self._current_step

    def process(self, transition: PrimitiveTransition) -> TransitionEvent:
        prepared = self.prepare(
            episode_id=transition.episode_id,
            task_id=transition.task_id,
            step_id=transition.step_id,
            instruction=transition.instruction,
            action=transition.action,
            pre_image=transition.pre_image,
            goal_predicates=transition.goal_predicates,
            attribution_context=transition.attribution_context,
            metadata=transition.metadata,
        )
        return self.process_prepared(
            prepared,
            post_image=transition.post_image,
            feedback=transition.feedback,
            last_action_success=transition.last_action_success,
        )

    def prepare(
        self,
        *,
        episode_id: str,
        task_id: str,
        step_id: int,
        instruction: str,
        action: ActionCall,
        pre_image: str,
        goal_predicates: tuple[PredicateKey, ...] = (),
        attribution_context: AttributionContext | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> PreparedTransition:
        self.ledger.expire(step_id)
        expected = ()
        if not self._frozen:
            expected = self.action_schema.compile(
                action,
                self.ledger,
                self.skill,
                goal_predicates,
            )
        return PreparedTransition(
            episode_id=episode_id,
            task_id=task_id,
            step_id=step_id,
            instruction=instruction,
            action=action,
            pre_image=pre_image,
            pre_ledger=self.ledger.snapshot(),
            skill_id=self.skill.skill_id,
            skill_version=self.skill.version,
            goal_predicates=goal_predicates,
            expected_delta=expected,
            attribution_context=attribution_context or AttributionContext(),
            metadata=metadata or {},
        )

    def process_prepared(
        self,
        prepared: PreparedTransition,
        *,
        post_image: str,
        feedback: str,
        last_action_success: bool | None,
    ) -> TransitionEvent:
        if (
            prepared.skill_id != self.skill.skill_id
            or prepared.skill_version != self.skill.version
        ):
            raise RuntimeError("active skill changed after the expected transition was prepared")
        self._current_step = prepared.step_id
        expected = prepared.expected_delta
        request = EvidenceRequest(
            episode_id=prepared.episode_id,
            step_id=prepared.step_id,
            instruction=prepared.instruction,
            action=prepared.action,
            pre_image=prepared.pre_image,
            post_image=post_image,
            feedback=feedback,
            last_action_success=last_action_success,
            pre_ledger=prepared.pre_ledger,
            goal_predicates=prepared.goal_predicates,
        )
        evidence = self.evidence_extractor.extract(request)
        # Every reliable evidence packet advances belief. Routing happens later.
        self.ledger.merge(evidence)
        mismatches = ()
        attribution = None
        if not self._frozen:
            mismatches = compare_transitions(expected, evidence)
            context = replace(
                prepared.attribution_context,
                stochastic_suspected=(
                    prepared.attribution_context.stochastic_suspected
                    or (
                        last_action_success is False
                        and prepared.attribution_context.executor_followed_skill is not False
                    )
                ),
                instruction=prepared.instruction,
                action_type=prepared.action.action_type,
                skill_obligations={
                    field.value: self.skill.statements(field)
                    for field in SkillField
                },
            )
            attribution = self.credit_assigner.assign(mismatches, context)
        if (
            attribution is not None
            and
            attribution.target is UpdateTarget.ABSTAIN
            and attribution.subreason is AbstainReason.EXECUTION_LAPSE
        ):
            self.emphasis_buffer.add(
                "Follow the applicable active-skill rule and verify its result before replanning.",
                context=prepared.attribution_context.task_pattern,
                evidence_ids=attribution.evidence_ids,
                current_step=prepared.step_id,
            )
        event_id = _prepared_event_id(prepared)
        event = TransitionEvent(
            event_id=event_id,
            episode_id=prepared.episode_id,
            task_id=prepared.task_id,
            step_id=prepared.step_id,
            instruction=prepared.instruction,
            action=prepared.action,
            skill_id=self.skill.skill_id,
            skill_version=self.skill.version,
            pre_image=prepared.pre_image,
            post_image=post_image,
            feedback=feedback,
            last_action_success=last_action_success,
            pre_ledger=prepared.pre_ledger,
            goal_predicates=prepared.goal_predicates,
            expected_delta=expected,
            evidence_delta=evidence,
            mismatches=mismatches,
            attribution=attribution,
            metadata=prepared.metadata,
        )
        if attribution is not None and attribution.target is UpdateTarget.SKILL_UPDATE:
            eligible = [
                mismatch
                for mismatch in mismatches
                if mismatch.expected is not None
                and mismatch.expected.skill_field == attribution.field
            ]
            # The constrained field teacher may locate procedure/constraint faults
            # whose observable symptom originates in a fixed primitive transition.
            if not eligible and mismatches:
                eligible = [max(
                    mismatches,
                    key=lambda item: 0.0
                    if item.evidence is None
                    else item.evidence.confidence,
                )]
            for mismatch in eligible:
                self.clusterer.add(
                    event_id=event_id,
                    episode_id=prepared.episode_id,
                    skill_id=self.skill.skill_id,
                    skill_version=self.skill.version,
                    attribution=attribution,
                    mismatch=mismatch,
                    action=prepared.action,
                    pre_ledger=prepared.pre_ledger,
                    goal_predicates=prepared.goal_predicates,
                    evidence_delta=evidence,
                    task_pattern=prepared.attribution_context.task_pattern,
                    object_context=prepared.attribution_context.object_context,
                )
        return event

    def promote(self, candidate: SkillSpec) -> None:
        if self._frozen:
            raise RuntimeError("cannot promote into a frozen engine")
        if candidate.skill_id != self.skill.skill_id or candidate.parent_version != self.skill.version:
            raise ValueError("candidate is not a direct child of the active skill")
        self.skill = candidate

    def freeze(self) -> FrozenSkillArtifact:
        self.skill = replace(self.skill, frozen=True)
        self._frozen = True
        return FrozenSkillArtifact(self.skill, skill_digest(self.skill))


def _prepared_event_id(transition: PreparedTransition) -> str:
    raw = (
        f"{transition.episode_id}|{transition.task_id}|{transition.step_id}|"
        f"{transition.action.action_id}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
