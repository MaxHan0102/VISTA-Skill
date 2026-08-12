from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

from vista_skill.clustering import ClusterItem, ClusterKey, EvidenceCluster
from vista_skill.evolution import (
    CandidateGate,
    GateDecision,
    GateStageResult,
    PatchGenerator,
    make_patch_id,
)
from vista_skill.schemas import (
    AttributionResult,
    EvidenceSource,
    Mismatch,
    MismatchKind,
    PatchOperation,
    PredicateEvidence,
    PredicateKey,
    SkillField,
    SkillPatch,
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
    """Outcome of a trajectory-level baseline update.

    ``parent`` is the Skill before the update and ``candidate`` is the revised
    Skill a frontend/updater produced (``None`` when no revision was proposed,
    e.g. an execution-lapse appendix note). The ``skill`` property exposes the
    resulting active Skill (candidate if accepted, else parent) so callers that
    only need "what do I run next" keep the original single-field contract.
    """

    parent: SkillSpec
    candidate: SkillSpec | None
    accepted: bool
    route: str
    patch: SkillPatch | None = None
    decision: GateDecision | None = None

    @property
    def skill(self) -> SkillSpec:
        if self.accepted and self.candidate is not None:
            return self.candidate
        return self.parent


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
            return BaselineUpdateResult(
                parent=skill,
                candidate=None,
                accepted=False,
                route=proposal.source_frontend,
            )
        cluster = proposal_cluster(skill, proposal, episode_ids)
        patch = self.generator.propose(skill, cluster)
        decision, candidate = self.gate.evaluate(skill, patch, cluster)
        return BaselineUpdateResult(
            parent=skill,
            candidate=candidate,
            accepted=decision.accepted,
            route=proposal.source_frontend,
            patch=patch,
            decision=decision,
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
        episode_ids: Sequence[str] = (),
    ) -> BaselineUpdateResult:
        content = proposal.content.strip()
        if not content:
            return BaselineUpdateResult(
                parent=skill, candidate=None, accepted=False, route=proposal.source_frontend
            )
        # Execution lapses (non-persistent / abstain) go to the appendix and never
        # mutate the Skill body; persistent proposals revise the attributed field.
        if not proposal.persistent or proposal.target is UpdateTarget.ABSTAIN:
            self.execution_notes.append(content)
            return BaselineUpdateResult(
                parent=skill, candidate=None, accepted=False, route=proposal.source_frontend
            )
        field = proposal.field or SkillField.PROCEDURE
        statements = (*skill.statements(field), content)
        if len(statements) > self.max_statements_per_field:
            statements = statements[-self.max_statements_per_field :]
        revised = with_field(skill, field, statements)
        evidence_ids = tuple(proposal.evidence_ids)
        patch = SkillPatch(
            patch_id=make_patch_id(
                skill, field, PatchOperation.APPEND, "", content, evidence_ids
            ),
            skill_id=skill.skill_id,
            parent_version=skill.version,
            field=field,
            operation=PatchOperation.APPEND,
            old="",
            new=content,
            evidence_ids=evidence_ids,
            scope="embodiskill_native_body_revision",
        )
        decision = GateDecision(
            accepted=True,
            reason="embodiskill native body revision (append without a paired gate)",
            parent_version=skill.version,
            candidate_version=revised.version,
            patch_id=patch.patch_id,
            stages=(
                GateStageResult(
                    "native_body_revision",
                    True,
                    "native semantics revise the attributed field without a paired gate",
                ),
            ),
        )
        return BaselineUpdateResult(
            parent=skill,
            candidate=revised,
            accepted=True,
            route=proposal.source_frontend,
            patch=patch,
            decision=decision,
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
