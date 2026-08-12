from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from vista_skill.evolution import (
    BoundedPatchApplier,
    CandidateGate,
    DeterministicTransitionChecker,
    EvolutionCoordinator,
    EvolutionResult,
    PairedEvaluator,
    PatchGenerator,
)
from vista_skill.lineage import LineageStore
from vista_skill.pipeline import FrozenSkillArtifact, VistaSkillEngine
from vista_skill.config import VistaConfig


@dataclass(frozen=True)
class WorkflowResult:
    ready_clusters: int
    evolution: tuple[EvolutionResult, ...]
    frozen: FrozenSkillArtifact


class EvolutionWorkflow:
    """Persistent acquisition-time evolution state for one independent run."""

    def __init__(
        self,
        engine: VistaSkillEngine,
        *,
        generator: PatchGenerator,
        paired_evaluator: PairedEvaluator,
        lineage: LineageStore,
        config: VistaConfig,
        protocol: Mapping[str, object],
    ) -> None:
        if engine.frozen:
            raise RuntimeError("cannot evolve an already frozen engine")
        self.engine = engine
        gate = CandidateGate(
            BoundedPatchApplier(config.patch),
            DeterministicTransitionChecker(engine.action_schema),
            paired_evaluator,
            config.gate,
        )
        self.coordinator = EvolutionCoordinator(
            generator,
            gate,
            lineage,
            protocol=protocol,
        )

    def evolve_ready(self) -> tuple[int, tuple[EvolutionResult, ...]]:
        if self.engine.frozen:
            raise RuntimeError("cannot evolve an already frozen engine")
        ready = self.engine.clusterer.ready()
        active, results = self.coordinator.evolve(self.engine.skill, ready)
        if active.version != self.engine.skill.version:
            self.engine.promote(active)
        return len(ready), results


def evolve_ready_clusters(
    engine: VistaSkillEngine,
    *,
    generator: PatchGenerator,
    paired_evaluator: PairedEvaluator,
    lineage: LineageStore,
    config: VistaConfig,
    protocol: Mapping[str, object],
) -> WorkflowResult:
    workflow = EvolutionWorkflow(
        engine,
        generator,
        paired_evaluator=paired_evaluator,
        lineage=lineage,
        config=config,
        protocol=protocol,
    )
    ready_count, results = workflow.evolve_ready()
    frozen = engine.freeze()
    return WorkflowResult(ready_count, results, frozen)
