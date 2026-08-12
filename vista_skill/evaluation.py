from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from vista_skill.evolution import PairedEpisodeScore
from vista_skill.schemas import SkillSpec


@dataclass(frozen=True)
class EpisodeCoordinate:
    episode_id: str
    seed: int
    subgroup: str


@dataclass(frozen=True)
class RolloutScore:
    score: float
    success: bool | None = None
    metrics: Mapping[str, float] | None = None


class SkillRollout(Protocol):
    def __call__(
        self,
        skill: SkillSpec,
        coordinate: EpisodeCoordinate,
        stage: str,
    ) -> RolloutScore: ...


class PairedRolloutEvaluator:
    """Run parent and candidate on exactly the same episode/seed coordinates."""

    def __init__(
        self,
        coordinates: Mapping[str, Sequence[EpisodeCoordinate]],
        rollout: SkillRollout,
    ) -> None:
        self.coordinates = {
            stage: tuple(values) for stage, values in coordinates.items()
        }
        self.rollout = rollout

    def evaluate(
        self,
        parent: SkillSpec,
        candidate: SkillSpec,
        *,
        stage: str,
        episode_budget: int,
    ) -> Sequence[PairedEpisodeScore]:
        selected = self.coordinates.get(stage, ())[:episode_budget]
        scores = []
        for coordinate in selected:
            parent_result = self.rollout(parent, coordinate, stage)
            candidate_result = self.rollout(candidate, coordinate, stage)
            scores.append(
                PairedEpisodeScore(
                    episode_id=coordinate.episode_id,
                    seed=coordinate.seed,
                    parent_score=parent_result.score,
                    candidate_score=candidate_result.score,
                    subgroup=coordinate.subgroup,
                    parent_success=parent_result.success,
                    candidate_success=candidate_result.success,
                )
            )
        return tuple(scores)


def composite_task_score(
    *,
    task_success: float,
    task_progress: float,
    invalid_action_ratio: float = 0.0,
    premature_termination: float = 0.0,
    success_weight: float = 0.7,
) -> float:
    progress_weight = 1.0 - success_weight
    penalty = 0.1 * invalid_action_ratio + 0.1 * premature_termination
    return success_weight * task_success + progress_weight * task_progress - penalty
