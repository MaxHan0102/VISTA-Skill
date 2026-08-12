from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def macro_f1(gold: Sequence[Hashable], predicted: Sequence[Hashable]) -> float:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted labels must have equal length")
    labels = set(gold) | set(predicted)
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        scores.append(safe_ratio(2 * precision * recall, precision + recall))
    return sum(scores) / len(scores)


@dataclass(frozen=True)
class UpdateAudit:
    patch_id: str
    promoted: bool
    overall_delta: float
    subgroup_deltas: Mapping[str, float]


def update_reliability(
    audits: Sequence[UpdateAudit],
    *,
    epsilon: float = 0.0,
) -> dict[str, float]:
    promoted = [item for item in audits if item.promoted]
    harmful = [
        item
        for item in promoted
        if item.overall_delta < -epsilon
        or any(delta < -epsilon for delta in item.subgroup_deltas.values())
    ]
    beneficial = [
        item
        for item in promoted
        if item.overall_delta > epsilon
        and all(delta >= -epsilon for delta in item.subgroup_deltas.values())
    ]
    neutral_count = len(promoted) - len(beneficial) - len(harmful)
    rejected = [item for item in audits if not item.promoted]
    missed_beneficial = [
        item
        for item in rejected
        if item.overall_delta > epsilon
        and all(delta >= -epsilon for delta in item.subgroup_deltas.values())
    ]
    return {
        "promoted_updates": float(len(promoted)),
        "beneficial_updates": float(len(beneficial)),
        "neutral_updates": float(neutral_count),
        "harmful_updates": float(len(harmful)),
        "beneficial_update_precision": safe_ratio(len(beneficial), len(promoted)),
        "neutral_update_rate": safe_ratio(neutral_count, len(promoted)),
        "harmful_update_rate": safe_ratio(len(harmful), len(promoted)),
        "update_coverage": safe_ratio(len(promoted), len(audits)),
        "missed_beneficial_update_rate": safe_ratio(
            len(missed_beneficial), len(rejected)
        ),
    }


def confusion_counts(
    gold: Sequence[Hashable], predicted: Sequence[Hashable]
) -> Mapping[tuple[Hashable, Hashable], int]:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted labels must have equal length")
    return Counter(zip(gold, predicted))


def abstention_metrics(
    gold_should_abstain: Sequence[bool], predicted_abstain: Sequence[bool]
) -> dict[str, float]:
    if len(gold_should_abstain) != len(predicted_abstain):
        raise ValueError("gold and predicted abstention labels must have equal length")
    predicted_count = sum(predicted_abstain)
    correct = sum(
        gold and predicted
        for gold, predicted in zip(gold_should_abstain, predicted_abstain)
    )
    return {
        "abstention_precision": safe_ratio(correct, predicted_count),
        "selective_coverage": safe_ratio(
            len(predicted_abstain) - predicted_count, len(predicted_abstain)
        ),
    }


def brier_score(outcomes: Sequence[bool], confidence: Sequence[float]) -> float:
    if len(outcomes) != len(confidence):
        raise ValueError("outcomes and confidence must have equal length")
    if not outcomes:
        return 0.0
    return sum(
        (probability - float(outcome)) ** 2
        for outcome, probability in zip(outcomes, confidence)
    ) / len(outcomes)


def expected_calibration_error(
    outcomes: Sequence[bool],
    confidence: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    if len(outcomes) != len(confidence):
        raise ValueError("outcomes and confidence must have equal length")
    if not outcomes:
        return 0.0
    if bins < 1:
        raise ValueError("bins must be positive")
    error = 0.0
    total = len(outcomes)
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            position
            for position, probability in enumerate(confidence)
            if lower <= probability < upper or (index == bins - 1 and probability == 1.0)
        ]
        if not members:
            continue
        accuracy = sum(float(outcomes[position]) for position in members) / len(members)
        mean_confidence = sum(confidence[position] for position in members) / len(members)
        error += len(members) / total * abs(accuracy - mean_confidence)
    return error


def predicate_transition_f1(
    gold: Sequence[Hashable], predicted: Sequence[Hashable]
) -> float:
    return macro_f1(gold, predicted)
