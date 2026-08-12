from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

from vista_skill.clustering import ClusterItem, ClusterKey, EvidenceCluster
from vista_skill.evolution import CandidateGate, GateDecision, PatchGenerator
from vista_skill.schemas import (
    AttributionResult,
    EvidenceSource,
    Mismatch,
    MismatchKind,
    PredicateEvidence,
    PredicateKey,
    SkillField,
    SkillSpec,
    TruthValue,
    UpdateTarget,
)
from vista_skill.skills import with_field


class EmbodiSkillRoute(str, Enum):
    S_NEW = "S_NEW"
    S_BETTER = "S_BETTER"
    FAIL_SKILL = "FAIL_SKILL"
    FAIL_EXECUTION = "FAIL_EXECUTION"


@dataclass(frozen=True)
class EpisodeSummary:
    episode_id: str
    instruction: str
    success: bool
    trajectory: tuple[str, ...]
    current_skill: str
    failure_reason: str = ""


@dataclass(frozen=True)
class TrajectoryReflection:
    route: EmbodiSkillRoute
    content: str
    target_field: SkillField | None
    confidence: float
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class CommonUpdateProposal:
    target: UpdateTarget
    field: SkillField | None
    content: str
    evidence_ids: tuple[str, ...]
    source_frontend: str
    persistent: bool


class TrajectoryTeacher(Protocol):
    def reflect(self, episode: EpisodeSummary) -> TrajectoryReflection: ...


class EmbodiSkillFrontend:
    """Controlled EB-Hab reimplementation of EmbodiSkill trajectory routing."""

    def __init__(self, teacher: TrajectoryTeacher, *, common_gate: bool) -> None:
        self.teacher = teacher
        self.common_gate = common_gate

    def consume(self, episode: EpisodeSummary) -> CommonUpdateProposal:
        reflection = self.teacher.reflect(episode)
        execution_lapse = reflection.route is EmbodiSkillRoute.FAIL_EXECUTION
        persistent = not execution_lapse
        if not self.common_gate and execution_lapse:
            # Native semantics write a separate execution-note appendix.
            target = UpdateTarget.ABSTAIN
        elif execution_lapse:
            target = UpdateTarget.ABSTAIN
        else:
            target = UpdateTarget.SKILL_UPDATE
        return CommonUpdateProposal(
            target=target,
            field=reflection.target_field if persistent else None,
            content=reflection.content,
            evidence_ids=reflection.evidence_ids,
            source_frontend="embodiskill_common_gate" if self.common_gate else "embodiskill_native",
            persistent=persistent,
        )


class UnconditionalReflectionFrontend:
    """VISTA w/o VTCA control: every failed trajectory reaches the common updater."""

    def __init__(self, teacher: TrajectoryTeacher) -> None:
        self.teacher = teacher

    def consume(self, episode: EpisodeSummary) -> CommonUpdateProposal | None:
        if episode.success:
            return None
        reflection = self.teacher.reflect(episode)
        return CommonUpdateProposal(
            target=UpdateTarget.SKILL_UPDATE,
            field=reflection.target_field or SkillField.PROCEDURE,
            content=reflection.content,
            evidence_ids=reflection.evidence_ids,
            source_frontend="unconditional_trajectory_reflection",
            persistent=True,
        )


@dataclass(frozen=True)
class BaselineUpdateResult:
    skill: SkillSpec
    accepted: bool
    route: str
    decision: GateDecision | None = None


class CommonGateProposalAdapter:
    """Route a trajectory proposal through the exact VISTA candidate gate."""

    def __init__(self, generator: PatchGenerator, gate: CandidateGate) -> None:
        self.generator = generator
        self.gate = gate

    def update(
        self,
        skill: SkillSpec,
        proposal: CommonUpdateProposal,
        episode_ids: Sequence[str],
    ) -> BaselineUpdateResult:
        if (
            not proposal.persistent
            or proposal.target is not UpdateTarget.SKILL_UPDATE
            or proposal.field is None
        ):
            return BaselineUpdateResult(skill, False, proposal.source_frontend)
        cluster = proposal_cluster(skill, proposal, episode_ids)
        patch = self.generator.propose(skill, cluster)
        decision, candidate = self.gate.evaluate(skill, patch, cluster)
        return BaselineUpdateResult(
            candidate if candidate is not None else skill,
            decision.accepted,
            proposal.source_frontend,
            decision,
        )


class EmbodiSkillNativeUpdater:
    """EB-Hab adaptation of EmbodiSkill body/appendix routing semantics."""

    def __init__(self, *, max_statements_per_field: int = 12) -> None:
        self.max_statements_per_field = max_statements_per_field
        self.execution_notes: list[str] = []

    def update(
        self,
        skill: SkillSpec,
        proposal: CommonUpdateProposal,
    ) -> BaselineUpdateResult:
        if not proposal.content.strip():
            return BaselineUpdateResult(skill, False, proposal.source_frontend)
        if not proposal.persistent or proposal.target is UpdateTarget.ABSTAIN:
            self.execution_notes.append(proposal.content.strip())
            return BaselineUpdateResult(skill, False, proposal.source_frontend)
        field = proposal.field or SkillField.PROCEDURE
        statements = (*skill.statements(field), proposal.content.strip())
        if len(statements) > self.max_statements_per_field:
            statements = statements[-self.max_statements_per_field :]
        return BaselineUpdateResult(
            with_field(skill, field, statements), True, proposal.source_frontend
        )


def proposal_cluster(
    skill: SkillSpec,
    proposal: CommonUpdateProposal,
    episode_ids: Sequence[str],
) -> EvidenceCluster:
    """Represent trajectory reflection in the common updater's evidence contract."""
    if proposal.field is None or not proposal.evidence_ids:
        raise ValueError("common-gate proposal requires a field and evidence IDs")
    episodes = tuple(dict.fromkeys(str(item) for item in episode_ids))
    if len(episodes) < 2:
        raise ValueError("common-gate trajectory proposals require independent episodes")
    cluster = EvidenceCluster(
        ClusterKey(
            skill.skill_id,
            proposal.field,
            "trajectory_reflection",
            "trajectory",
            "general",
            skill.version,
        )
    )
    for index, evidence_id in enumerate(proposal.evidence_ids):
        episode_id = episodes[index % len(episodes)]
        key = PredicateKey("trajectory_failure", (episode_id,))
        observed = PredicateEvidence(
            key=key,
            before=TruthValue.UNKNOWN,
            after=TruthValue.TRUE,
            confidence=1.0,
            source=EvidenceSource.DERIVED_GOAL,
            evidence_id=evidence_id,
            timestamp=index,
            rationale=proposal.content,
        )
        mismatch = Mismatch(
            mismatch_id=f"trajectory:{index}:{evidence_id}",
            key=key,
            kind=MismatchKind.SUPPORTED_UNEXPECTED,
            expected=None,
            evidence=observed,
            evidence_ids=(evidence_id,),
        )
        attribution = AttributionResult(
            target=UpdateTarget.SKILL_UPDATE,
            field=proposal.field,
            confidence=1.0,
            mismatch_ids=(mismatch.mismatch_id,),
            evidence_ids=(evidence_id,),
            rationale=proposal.content,
        )
        cluster.items.append(
            ClusterItem(
                event_id=f"trajectory:{episode_id}:{index}",
                episode_id=episode_id,
                attribution=attribution,
                mismatch=mismatch,
                evidence_delta=(observed,),
            )
        )
    return cluster
