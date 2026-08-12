from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from vista_skill.schemas import (
    PredicateEvidence,
    PredicateKey,
    PredicateState,
    TruthValue,
    unique_strings,
)


@dataclass(frozen=True)
class MergePolicy:
    min_confidence: float = 0.5
    conflict_margin: float = 0.1
    stale_after_steps: int = 12


class BeliefLedger:
    """Sparse, instance-preserving three-valued episode belief."""

    def __init__(self, policy: MergePolicy | None = None) -> None:
        self.policy = policy or MergePolicy()
        self._states: dict[PredicateKey, PredicateState] = {}

    @classmethod
    def from_snapshot(
        cls,
        states: Iterable[PredicateState],
        policy: MergePolicy | None = None,
    ) -> "BeliefLedger":
        ledger = cls(policy)
        for state in states:
            if not isinstance(state, PredicateState):
                raise TypeError("snapshot contains a non-predicate state")
            ledger._states[state.key] = state
        return ledger

    def get(self, key: PredicateKey) -> PredicateState | None:
        return self._states.get(key)

    def value(self, key: PredicateKey) -> TruthValue:
        state = self.get(key)
        return TruthValue.UNKNOWN if state is None else state.value

    def snapshot(self, *, relevant_only: bool = False) -> tuple[PredicateState, ...]:
        values = self._states.values()
        if relevant_only:
            values = (item for item in values if item.task_relevance > 0.0)
        return tuple(sorted(values, key=lambda item: item.key))

    def merge(self, evidence: Iterable[PredicateEvidence]) -> tuple[PredicateState, ...]:
        """Merge evidence into belief. No prediction or attribution type is accepted."""
        changed: list[PredicateState] = []
        for item in evidence:
            if not isinstance(item, PredicateEvidence):
                raise TypeError("belief can only be updated from PredicateEvidence")
            if item.confidence < self.policy.min_confidence:
                continue
            previous = self._states.get(item.key)
            merged = self._merge_one(previous, item)
            if previous != merged:
                self._states[item.key] = merged
                changed.append(merged)
        return tuple(changed)

    def expire(self, current_step: int) -> tuple[PredicateState, ...]:
        expired: list[PredicateState] = []
        for key, state in list(self._states.items()):
            if current_step - state.timestamp <= self.policy.stale_after_steps:
                continue
            replacement = PredicateState(
                key=key,
                value=TruthValue.UNKNOWN,
                confidence=max(0.0, state.confidence * 0.5),
                source="ttl_expiration",
                evidence_ids=state.evidence_ids,
                timestamp=current_step,
                view_id=state.view_id,
                coverage=0.0,
                task_relevance=state.task_relevance,
            )
            self._states[key] = replacement
            expired.append(replacement)
        return tuple(expired)

    def _merge_one(
        self,
        previous: PredicateState | None,
        evidence: PredicateEvidence,
    ) -> PredicateState:
        evidence_ids = unique_strings(
            (*(() if previous is None else previous.evidence_ids), evidence.evidence_id)
        )
        if previous is not None and evidence.timestamp < previous.timestamp:
            return previous
        value = evidence.after
        confidence = evidence.confidence
        source = evidence.source.value

        if previous is not None and previous.value != evidence.after:
            confidence_gap = evidence.confidence - previous.confidence
            if evidence.after is TruthValue.UNKNOWN:
                value = TruthValue.UNKNOWN
            elif previous.value is TruthValue.UNKNOWN:
                value = evidence.after
            elif confidence_gap > self.policy.conflict_margin:
                value = evidence.after
            elif confidence_gap < -self.policy.conflict_margin:
                return previous
            else:
                value = TruthValue.UNKNOWN
                confidence = max(previous.confidence, evidence.confidence)
                source = "conflicting_evidence"

        return PredicateState(
            key=evidence.key,
            value=value,
            confidence=confidence,
            source=source,
            evidence_ids=evidence_ids,
            timestamp=max(evidence.timestamp, 0 if previous is None else previous.timestamp),
            view_id=evidence.view_id,
            coverage=evidence.coverage,
            task_relevance=evidence.task_relevance,
        )
