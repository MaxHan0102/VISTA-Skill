"""Oracle / noisy evidence providers and the calibration harness.

These components back the Phase-2 evidence-branch Go/No-Go (design doc
§4.7.4): *if visual predicate extraction itself is unreliable, VTCA cannot be
claimed*. The :class:`OracleEvidenceProvider` is the reliability ceiling (it
reads ground-truth predicate deltas, the kind Habitat goal predicates /
simulator state expose); :class:`NoisyEvidenceProvider` wraps any provider and
injects controlled, seeded noise so a researcher can sweep reliability.
:func:`evaluate_calibration`, :func:`selective_risk_curve` and
:func:`compare_providers` summarise agreement and calibration against that
ceiling.

This module is read-only with respect to the rest of the package: it only
imports from ``evidence``, ``metrics`` and ``schemas`` and never mutates them.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from vista_skill.evidence import VisualEvidenceProvider
from vista_skill.metrics import (
    brier_score,
    expected_calibration_error,
    safe_ratio,
)
from vista_skill.schemas import (
    EvidenceRequest,
    EvidenceSource,
    PredicateEvidence,
    PredicateKey,
    TruthValue,
)

_DEFINITE_VALUES: tuple[TruthValue, TruthValue] = (TruthValue.TRUE, TruthValue.FALSE)

# A ground-truth lookup maps an evidence request to the authoritative post-action
# predicate state. Keys may be ``PredicateKey`` or renderable strings; values may
# be ``TruthValue``, ``bool`` or the matching strings. ``UNKNOWN`` is permitted
# (the oracle then asserts the predicate is genuinely unset).
GroundTruthLookup = Callable[[EvidenceRequest], Mapping[Any, Any]]


# --------------------------------------------------------------------------- #
# coercion helpers
# --------------------------------------------------------------------------- #
def _coerce_key(value: Any) -> PredicateKey:
    if isinstance(value, PredicateKey):
        return value
    return PredicateKey.parse(str(value))


def _coerce_truth(value: Any) -> TruthValue:
    if isinstance(value, TruthValue):
        return value
    # ``bool`` is a subclass of ``int``; check it before falling back to string
    # parsing so ``True``/``False`` map to the obvious enum members.
    if isinstance(value, bool):
        return TruthValue.TRUE if value else TruthValue.FALSE
    return TruthValue(str(value))


def _clamp01(value: float, name: str = "value") -> float:
    coerced = float(value)
    if not 0.0 <= coerced <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return coerced


def _flip_truth(value: TruthValue) -> TruthValue:
    """Invert boolean polarity. ``UNKNOWN`` has no binary opposite and is left as-is."""
    if value is TruthValue.TRUE:
        return TruthValue.FALSE
    if value is TruthValue.FALSE:
        return TruthValue.TRUE
    return TruthValue.UNKNOWN


def _as_lookup(
    ground_truth: Callable[[EvidenceRequest], Mapping[Any, Any]] | Mapping[Any, Any],
) -> GroundTruthLookup:
    """Accept either a callable keyed on the request or a static predicate map."""
    if callable(ground_truth):
        return ground_truth
    mapping = dict(ground_truth)

    def _static(_request: EvidenceRequest) -> Mapping[Any, Any]:
        return mapping

    return _static


# --------------------------------------------------------------------------- #
# oracle provider
# --------------------------------------------------------------------------- #
class OracleEvidenceProvider:
    """Ground-truth evidence provider; the reliability ceiling.

    Unlike the VLM-backed :class:`~vista_skill.models.JsonVisualEvidenceProvider`,
    this provider does not look at images. It consults an authoritative
    post-action predicate map (the kind Habitat goal predicates / simulator
    state would expose) and emits one :class:`PredicateEvidence` per supplied
    predicate with ``confidence=1.0``. ``before`` values are recovered from the
    request's pre-action ledger, mirroring the VLM provider's contract.

    The source defaults to :attr:`EvidenceSource.DERIVED_GOAL` (the closest
    existing label for an authoritative derived signal); pass ``source=`` to
    override.
    """

    def __init__(
        self,
        ground_truth: Callable[[EvidenceRequest], Mapping[Any, Any]] | Mapping[Any, Any],
        *,
        source: EvidenceSource = EvidenceSource.DERIVED_GOAL,
        confidence: float = 1.0,
        coverage: float = 1.0,
    ) -> None:
        self._lookup = _as_lookup(ground_truth)
        self._source = source
        self._confidence = _clamp01(confidence, "confidence")
        self._coverage = _clamp01(coverage, "coverage")

    def extract(self, request: EvidenceRequest) -> tuple[PredicateEvidence, ...]:
        after_values = self._lookup(request)
        pre_values = {state.key: state.value for state in request.pre_ledger}
        items: list[PredicateEvidence] = []
        for raw_key, raw_value in after_values.items():
            key = _coerce_key(raw_key)
            after = _coerce_truth(raw_value)
            items.append(
                PredicateEvidence(
                    key=key,
                    before=pre_values.get(key, TruthValue.UNKNOWN),
                    after=after,
                    confidence=self._confidence,
                    source=self._source,
                    evidence_id=(
                        f"{request.episode_id}:s{request.step_id}:oracle:{key.render()}"
                    ),
                    timestamp=request.step_id,
                    view_id=f"{request.episode_id}:s{request.step_id}:oracle",
                    coverage=self._coverage,
                    rationale="ground-truth predicate delta",
                )
            )
        # Deterministic ordering for stable comparison downstream.
        return tuple(sorted(items, key=lambda item: item.key))


# --------------------------------------------------------------------------- #
# noisy provider
# --------------------------------------------------------------------------- #
class NoisyEvidenceProvider:
    """Controlled, seeded noise wrapper around any evidence provider.

    Used to stress-test VTCA reliability when visual predicate extraction is
    unreliable. Noise channels (any subset may be enabled):

    - ``drop_rate``: probability of dropping a true evidence item.
    - ``flip_rate``: probability of flipping a ``TruthValue`` (TRUE <-> FALSE;
      ``UNKNOWN`` is unchanged as it has no binary opposite).
    - ``false_positive_rate``: per-pool-key probability of emitting a spurious
      evidence item drawn from ``false_positive_pool``.

    Determinism contract. There is **no** module-level RNG: a
    ``random.Random(seed)`` stream is re-derived per request from a stable
    SHA-256 hash of ``(base_seed, episode_id, step_id)``. Therefore calling
    ``extract`` twice on the same request yields identical output, two provider
    instances built with the same seed agree, and runs reproduce across
    processes (no reliance on the salted ``hash()`` builtin).
    """

    def __init__(
        self,
        wrapped: VisualEvidenceProvider,
        *,
        drop_rate: float = 0.0,
        flip_rate: float = 0.0,
        false_positive_rate: float = 0.0,
        seed: int = 0,
        false_positive_pool: Sequence[PredicateKey | str] | None = None,
        false_positive_confidence: float = 0.6,
    ) -> None:
        self._wrapped = wrapped
        self._drop_rate = _clamp01(drop_rate, "drop_rate")
        self._flip_rate = _clamp01(flip_rate, "flip_rate")
        self._fp_rate = _clamp01(false_positive_rate, "false_positive_rate")
        self._base_seed = int(seed)
        self._pool: tuple[PredicateKey, ...] = tuple(
            _coerce_key(key) for key in (false_positive_pool or ())
        )
        self._fp_confidence = _clamp01(
            false_positive_confidence, "false_positive_confidence"
        )

    def extract(self, request: EvidenceRequest) -> tuple[PredicateEvidence, ...]:
        rng = random.Random(self._request_seed(request))
        result: list[PredicateEvidence] = []
        for item in self._wrapped.extract(request):
            if self._drop_rate > 0.0 and rng.random() < self._drop_rate:
                continue
            after = item.after
            if self._flip_rate > 0.0 and rng.random() < self._flip_rate:
                after = _flip_truth(after)
            if after is item.after:
                result.append(item)
            else:
                result.append(replace(item, after=after))
        if self._fp_rate > 0.0 and self._pool:
            for key in self._pool:
                if rng.random() < self._fp_rate:
                    result.append(self._false_positive(request, key, rng))
        return tuple(result)

    def _false_positive(
        self,
        request: EvidenceRequest,
        key: PredicateKey,
        rng: random.Random,
    ) -> PredicateEvidence:
        after = rng.choice(_DEFINITE_VALUES)
        return PredicateEvidence(
            key=key,
            before=TruthValue.UNKNOWN,
            after=after,
            confidence=self._fp_confidence,
            source=EvidenceSource.VISUAL_PAIR,
            evidence_id=(
                f"{request.episode_id}:s{request.step_id}:noisy:fp:{key.render()}"
            ),
            timestamp=request.step_id,
            view_id=f"{request.episode_id}:s{request.step_id}:noisy",
            coverage=1.0,
            rationale="noisy false-positive injection",
        )

    def _request_seed(self, request: EvidenceRequest) -> int:
        material = f"{self._base_seed}|{request.episode_id}|{request.step_id}".encode()
        digest = hashlib.sha256(material).digest()
        return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


# --------------------------------------------------------------------------- #
# calibration / selective-risk harness
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CalibrationReport:
    brier: float
    expected_calibration_error: float
    n_samples: int


@dataclass(frozen=True)
class SelectiveRiskPoint:
    confidence_threshold: float
    coverage: float  # fraction of decisions retained at or above the threshold
    selective_risk: float  # error rate among retained decisions (0 if none retained)
    n_retained: int


def evaluate_calibration(
    predicted: Sequence[tuple[float, bool]],
    *,
    n_bins: int = 10,
) -> CalibrationReport:
    """Compute Brier score and ECE for a list of ``(confidence, is_correct)`` pairs.

    Delegates to ``metrics.brier_score`` / ``metrics.expected_calibration_error``
    so the project has a single calibration implementation.
    """
    if not predicted:
        return CalibrationReport(brier=0.0, expected_calibration_error=0.0, n_samples=0)
    confidences = [float(confidence) for confidence, _ in predicted]
    outcomes = [bool(outcome) for _, outcome in predicted]
    return CalibrationReport(
        brier=brier_score(outcomes, confidences),
        expected_calibration_error=expected_calibration_error(
            outcomes, confidences, bins=n_bins
        ),
        n_samples=len(predicted),
    )


def selective_risk_curve(
    predicted: Sequence[tuple[float, bool]],
    thresholds: Sequence[float] | None = None,
) -> list[SelectiveRiskPoint]:
    """Sweep confidence thresholds and report coverage / selective risk.

    At each threshold ``t``: ``coverage = mean(conf >= t)`` and
    ``selective_risk = mean(not is_correct among conf >= t)`` (0 when nothing is
    retained). Defaults to the 11-point grid ``0.0, 0.1, ..., 1.0``.
    """
    if not predicted:
        return []
    if thresholds is None:
        thresholds = [index / 10.0 for index in range(11)]
    confidences = [float(confidence) for confidence, _ in predicted]
    outcomes = [bool(outcome) for _, outcome in predicted]
    total = len(predicted)
    points: list[SelectiveRiskPoint] = []
    for threshold in thresholds:
        retained = [
            (confidence, outcome)
            for confidence, outcome in zip(confidences, outcomes)
            if confidence >= threshold
        ]
        n_retained = len(retained)
        coverage = n_retained / total
        if n_retained == 0:
            selective_risk = 0.0
        else:
            selective_risk = sum(1 for _, outcome in retained if not outcome) / n_retained
        points.append(
            SelectiveRiskPoint(
                confidence_threshold=float(threshold),
                coverage=coverage,
                selective_risk=selective_risk,
                n_retained=n_retained,
            )
        )
    return points


# --------------------------------------------------------------------------- #
# provider comparison (Go/No-Go summary)
# --------------------------------------------------------------------------- #
def compare_providers(
    oracle: VisualEvidenceProvider,
    candidate: VisualEvidenceProvider,
    requests: Sequence[EvidenceRequest],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Run ``oracle`` and ``candidate`` over ``requests`` and summarise agreement.

    Predicate-level agreement treats every predicate the oracle asserts with a
    definite value (TRUE/FALSE) as a target. A candidate decision is a definite
    assertion (UNKNOWN is treated as abstention, not a decision):

    - ``predicate_recall`` = correct candidate assertions / oracle targets.
    - ``predicate_precision`` = correct candidate assertions / candidate
      assertions (a candidate assertion the oracle did not target counts as
      incorrect, i.e. a false positive).

    Candidate *decisions* (definite assertions) also feed the calibration and
    selective-risk harness: each contributes a ``(confidence, is_correct)``
    pair. ``seed`` is reserved for future reproducible resampling and is echoed
    in the result; the comparison itself is deterministic given the providers
    and requests.
    """
    oracle_asserted = 0
    candidate_asserted = 0
    candidate_correct = 0
    confidence_correct: list[tuple[float, bool]] = []
    for request in requests:
        oracle_truth: dict[PredicateKey, TruthValue] = {
            item.key: item.after
            for item in oracle.extract(request)
            if item.after in _DEFINITE_VALUES
        }
        oracle_asserted += len(oracle_truth)
        for item in candidate.extract(request):
            if item.after not in _DEFINITE_VALUES:
                continue
            candidate_asserted += 1
            if item.key in oracle_truth:
                correct = item.after is oracle_truth[item.key]
                if correct:
                    candidate_correct += 1
                confidence_correct.append((item.confidence, correct))
            else:
                # Candidate asserted a predicate the oracle never defined: a
                # hallucinated / spurious decision -> always incorrect.
                confidence_correct.append((item.confidence, False))
    precision = safe_ratio(candidate_correct, candidate_asserted)
    recall = safe_ratio(candidate_correct, oracle_asserted)
    f1 = safe_ratio(2 * precision * recall, precision + recall)
    calibration = evaluate_calibration(confidence_correct)
    curve = selective_risk_curve(confidence_correct)
    default_threshold = 0.5
    default_point = next(
        (
            point
            for point in curve
            if abs(point.confidence_threshold - default_threshold) < 1e-9
        ),
        SelectiveRiskPoint(default_threshold, 0.0, 0.0, 0),
    )
    return {
        "n_requests": len(requests),
        "n_predicate_targets": oracle_asserted,
        "n_candidate_assertions": candidate_asserted,
        "n_candidate_correct": candidate_correct,
        "predicate_precision": precision,
        "predicate_recall": recall,
        "predicate_f1": f1,
        "brier": calibration.brier,
        "expected_calibration_error": calibration.expected_calibration_error,
        "n_calibration_samples": calibration.n_samples,
        "selective_risk_curve": curve,
        "selective_risk_at_0.5": default_point,
        "seed": seed,
    }
