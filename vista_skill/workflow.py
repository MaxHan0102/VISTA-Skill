from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from vista_skill.baselines import (
    CommonGateProposalAdapter,
    CommonUpdateProposal,
    EmbodiSkillFrontend,
    EmbodiSkillNativeUpdater,
    EpisodeSummary,
    UnconditionalReflectionFrontend,
)
from vista_skill.evolution import (
    BoundedPatchApplier,
    CandidateGate,
    DeterministicTransitionChecker,
    EvolutionCoordinator,
    EvolutionResult,
    PairedEvaluator,
)
from vista_skill.lineage import LineageStore
from vista_skill.pipeline import VistaSkillEngine
from vista_skill.config import VistaConfig
from vista_skill.schemas import SkillField, UpdateTarget


@dataclass(frozen=True)
class WorkflowResult:
    ready_clusters: int
    evolution: tuple[EvolutionResult, ...]


def build_candidate_gate(
    engine: VistaSkillEngine,
    paired_evaluator: PairedEvaluator,
    config: VistaConfig,
) -> CandidateGate:
    """Construct the canonical VISTA paired candidate gate.

    Shared by the action-level VTCA workflow and the trajectory-level controlled
    baselines that route through the common gate (``embodiskill_star_common_gate``
    and ``vista_without_vtca``), so every method that claims
    ``update_backend: vista_common_gate`` uses the identical gate object.
    """
    return CandidateGate(
        BoundedPatchApplier(config.patch),
        DeterministicTransitionChecker(engine.action_schema),
        paired_evaluator,
        config.gate,
    )


class EvolutionWorkflow:
    """Persistent acquisition-time evolution state for one independent run."""

    def __init__(
        self,
        engine: VistaSkillEngine,
        *,
        generator,
        paired_evaluator: PairedEvaluator,
        lineage: LineageStore,
        config: VistaConfig,
        protocol: Mapping[str, object],
    ) -> None:
        if engine.frozen:
            raise RuntimeError("cannot evolve an already frozen engine")
        self.engine = engine
        gate = build_candidate_gate(engine, paired_evaluator, config)
        self.coordinator = EvolutionCoordinator(
            generator,
            gate,
            lineage,
            protocol=protocol,
        )

    def consume_episode(self, summary: EpisodeSummary) -> None:
        """Action-level VTCA ingests evidence through the engine clusterer, not
        through episode summaries. This no-op keeps the CLI loop uniform across
        action-level and trajectory-level evolution drivers."""
        return None

    def evolve_ready(self) -> tuple[int, tuple[EvolutionResult, ...]]:
        if self.engine.frozen:
            raise RuntimeError("cannot evolve an already frozen engine")
        ready = self.engine.clusterer.ready()
        active, results = self.coordinator.evolve(self.engine.skill, ready)
        if active.version != self.engine.skill.version:
            self.engine.promote(active)
        return len(ready), results


@dataclass
class _TrajectoryPendingGroup:
    field: SkillField
    episodes: set[str] = field(default_factory=set)
    proposals: list[CommonUpdateProposal] = field(default_factory=list)


class TrajectoryEvolutionWorkflow:
    """Episode-driven evolution for the controlled trajectory baselines.

    Where :class:`EvolutionWorkflow` evolves from action-level predicate
    clusters accumulated by the engine, this driver consumes whole-episode
    reflections produced by an EmbodiSkill-style frontend
    (:class:`EmbodiSkillFrontend` or :class:`UnconditionalReflectionFrontend`)
    and routes the resulting proposals through a shared updater
    (:class:`CommonGateProposalAdapter` for the common-gate variants, or
    :class:`EmbodiSkillNativeUpdater` for the native baseline).

    To keep teacher calls, token budget, and candidate count matched across
    methods (``configs/methods.json`` ``matched_controls``), reflection runs
    only on failed trajectories and a proposal fires only after
    ``min_independent_episodes`` distinct failed episodes support the same
    attributed field -- the trajectory analogue of action-level recurrence.
    """

    def __init__(
        self,
        engine: VistaSkillEngine,
        *,
        frontend: EmbodiSkillFrontend | UnconditionalReflectionFrontend,
        updater: CommonGateProposalAdapter | EmbodiSkillNativeUpdater,
        lineage: LineageStore,
        config: VistaConfig,
        protocol: Mapping[str, object],
    ) -> None:
        if engine.frozen:
            raise RuntimeError("cannot evolve an already frozen engine")
        self.engine = engine
        self.frontend = frontend
        self.updater = updater
        self.lineage = lineage
        self.protocol = protocol
        self.min_independent_episodes = config.recurrence.min_independent_episodes
        self._pending: dict[SkillField, _TrajectoryPendingGroup] = {}

    def consume_episode(self, summary: EpisodeSummary) -> None:
        if summary.success:
            return
        proposal = self.frontend.consume(summary)
        if proposal is None:
            return
        if (
            not proposal.persistent
            or proposal.target is not UpdateTarget.SKILL_UPDATE
            or proposal.field is None
        ):
            # Execution lapse / abstain: record side effects only (e.g. the
            # native appendix) with no skill mutation, lineage, or snapshot.
            self.updater.update(self.engine.skill, proposal, (summary.episode_id,))
            return
        group = self._pending.setdefault(
            proposal.field, _TrajectoryPendingGroup(proposal.field)
        )
        group.episodes.add(summary.episode_id)
        group.proposals.append(proposal)

    def evolve_ready(self) -> tuple[int, tuple[EvolutionResult, ...]]:
        if self.engine.frozen:
            raise RuntimeError("cannot evolve an already frozen engine")
        ready = [
            field
            for field, group in self._pending.items()
            if len(group.episodes) >= self.min_independent_episodes
            and group.proposals
        ]
        results: list[EvolutionResult] = []
        for skill_field in ready:
            # Remove the group once it has been evaluated so that, after a
            # promotion, fresh same-field failures accumulate against the new
            # active Skill version rather than re-firing the stale batch.
            group = self._pending.pop(skill_field)
            episode_ids = tuple(sorted(group.episodes))
            evidence_ids = tuple(
                dict.fromkeys(
                    evidence_id
                    for proposal in group.proposals
                    for evidence_id in proposal.evidence_ids
                )
            )
            if not evidence_ids:
                evidence_ids = tuple(f"{episode}:trajectory" for episode in episode_ids)
            aggregated = replace(group.proposals[-1], evidence_ids=evidence_ids)
            active = self.engine.skill
            outcome = self.updater.update(active, aggregated, episode_ids)
            if outcome.patch is None or outcome.decision is None:
                continue
            self.lineage.append(
                parent=outcome.parent,
                candidate=outcome.candidate,
                patch=outcome.patch,
                decision=outcome.decision,
                protocol=self.protocol,
            )
            results.append(
                EvolutionResult(
                    cluster_key=f"trajectory:{skill_field.value}",
                    patch=outcome.patch,
                    decision=outcome.decision,
                    parent=outcome.parent,
                    candidate=outcome.candidate,
                )
            )
            if outcome.accepted and outcome.candidate is not None:
                self.engine.promote(outcome.candidate)
        return len(ready), tuple(results)
