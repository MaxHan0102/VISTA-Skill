from __future__ import annotations

from dataclasses import dataclass, field

from vista_skill.schemas import (
    AttributionResult,
    Mismatch,
    MismatchKind,
    SkillField,
    UpdateTarget,
)
from vista_skill.schemas import (
    ActionCall,
    PredicateEvidence,
    PredicateKey,
    PredicateState,
)


@dataclass(frozen=True, order=True)
class ClusterKey:
    skill_id: str
    field: SkillField
    mismatch_kind: str
    task_pattern: str
    object_context: str
    skill_version: int = 0


@dataclass(frozen=True)
class ClusterItem:
    event_id: str
    episode_id: str
    attribution: AttributionResult
    mismatch: Mismatch
    action: ActionCall | None = None
    pre_ledger: tuple[PredicateState, ...] = ()
    goal_predicates: tuple[PredicateKey, ...] = ()
    evidence_delta: tuple[PredicateEvidence, ...] = ()


@dataclass
class EvidenceCluster:
    key: ClusterKey
    items: list[ClusterItem] = field(default_factory=list)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                evidence_id
                for item in self.items
                for evidence_id in item.attribution.evidence_ids
            )
        )

    @property
    def independent_episodes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.episode_id for item in self.items))

    @property
    def independent_support_count(self) -> int:
        return min(len(self.independent_episodes), len(self.evidence_ids))

    @property
    def mean_evidence_confidence(self) -> float:
        values = [
            item.mismatch.evidence.confidence
            for item in self.items
            if item.mismatch.evidence is not None
        ]
        return sum(values) / len(values) if values else 0.0

    @property
    def mean_attribution_confidence(self) -> float:
        if not self.items:
            return 0.0
        return sum(item.attribution.confidence for item in self.items) / len(self.items)


@dataclass(frozen=True)
class RecurrencePolicy:
    min_independent_episodes: int = 2
    min_evidence_confidence: float = 0.75
    min_attribution_confidence: float = 0.70


class EventClusterer:
    def __init__(self, policy: RecurrencePolicy | None = None) -> None:
        self.policy = policy or RecurrencePolicy()
        self._clusters: dict[ClusterKey, EvidenceCluster] = {}
        self._seen_event_evidence: set[tuple[str, str, str]] = set()

    def add(
        self,
        *,
        event_id: str,
        episode_id: str,
        skill_id: str,
        skill_version: int = 0,
        attribution: AttributionResult,
        mismatch: Mismatch,
        action: ActionCall | None = None,
        pre_ledger: tuple[PredicateState, ...] = (),
        goal_predicates: tuple[PredicateKey, ...] = (),
        evidence_delta: tuple[PredicateEvidence, ...] = (),
        task_pattern: str = "general",
        object_context: str = "general",
    ) -> EvidenceCluster | None:
        if attribution.target is not UpdateTarget.SKILL_UPDATE or attribution.field is None:
            return None
        evidence_ids = attribution.evidence_ids or mismatch.evidence_ids
        if not evidence_ids:
            return None
        if mismatch.kind is MismatchKind.TERMINATION_CONFLICT:
            # The termination policy is a global skill obligation (design 4.3):
            # its violations are object- and task-agnostic, so keying them on
            # the episode's task/object context would fragment one policy fault
            # into per-episode clusters that never reach recurrence.
            task_pattern = "any_task"
            object_context = "policy"
        unique_marker = (
            event_id,
            mismatch.mismatch_id,
            "|".join(sorted(evidence_ids)),
        )
        if unique_marker in self._seen_event_evidence:
            return None
        self._seen_event_evidence.add(unique_marker)
        key = ClusterKey(
            skill_id=skill_id,
            field=attribution.field,
            mismatch_kind=mismatch.kind.value,
            task_pattern=task_pattern,
            object_context=object_context,
            skill_version=skill_version,
        )
        cluster = self._clusters.setdefault(key, EvidenceCluster(key))
        cluster.items.append(
            ClusterItem(
                event_id,
                episode_id,
                attribution,
                mismatch,
                action,
                pre_ledger,
                goal_predicates,
                evidence_delta,
            )
        )
        return cluster

    def ready(self) -> tuple[EvidenceCluster, ...]:
        return tuple(
            cluster
            for cluster in self._clusters.values()
            if cluster.independent_support_count >= self.policy.min_independent_episodes
            and cluster.mean_evidence_confidence >= self.policy.min_evidence_confidence
            and cluster.mean_attribution_confidence >= self.policy.min_attribution_confidence
        )
