from __future__ import annotations

import unittest

from vista_skill.evidence_oracle import (
    CalibrationReport,
    NoisyEvidenceProvider,
    OracleEvidenceProvider,
    SelectiveRiskPoint,
    compare_providers,
    evaluate_calibration,
    selective_risk_curve,
)
from vista_skill.schemas import (
    ActionCall,
    EvidenceRequest,
    EvidenceSource,
    PredicateEvidence,
    PredicateKey,
    PredicateState,
    TruthValue,
)

# --------------------------------------------------------------------------- #
# fixture builders
# --------------------------------------------------------------------------- #
def _state(predicate: str, value: TruthValue, *, timestamp: int = 0) -> PredicateState:
    return PredicateState(
        key=PredicateKey.parse(predicate),
        value=value,
        confidence=0.9,
        source="fixture",
        evidence_ids=(f"state:{predicate}",),
        timestamp=timestamp,
    )


def _action(action_type: str = "look", arguments: tuple[str, ...] = ()) -> ActionCall:
    return ActionCall(
        action_id=0,
        action_type=action_type,
        arguments=tuple(arguments),
        text=f"{action_type} {' '.join(arguments)}".strip(),
    )


def _request(
    *,
    episode_id: str = "ep1",
    step_id: int = 1,
    pre_ledger: tuple[PredicateState, ...] = (),
    goal_predicates: tuple[PredicateKey, ...] = (),
    action: ActionCall | None = None,
) -> EvidenceRequest:
    return EvidenceRequest(
        episode_id=episode_id,
        step_id=step_id,
        instruction="inspect scene",
        action=action or _action(),
        pre_image="pre.png",
        post_image="post.png",
        feedback="",
        last_action_success=True,
        pre_ledger=pre_ledger,
        goal_predicates=goal_predicates,
    )


def _evidence(
    predicate: str,
    before: TruthValue,
    after: TruthValue,
    *,
    evidence_id: str = "ev",
    confidence: float = 0.8,
    source: EvidenceSource = EvidenceSource.VISUAL_PAIR,
) -> PredicateEvidence:
    return PredicateEvidence(
        key=PredicateKey.parse(predicate),
        before=before,
        after=after,
        confidence=confidence,
        source=source,
        evidence_id=evidence_id,
        timestamp=1,
    )


class _FixedProvider:
    """Minimal stand-in matching the VisualEvidenceProvider protocol."""

    def __init__(self, items: tuple[PredicateEvidence, ...]) -> None:
        self.items = items

    def extract(self, request: EvidenceRequest) -> tuple[PredicateEvidence, ...]:
        return self.items


# --------------------------------------------------------------------------- #
# OracleEvidenceProvider
# --------------------------------------------------------------------------- #
class OracleEvidenceProviderTests(unittest.TestCase):
    def test_emits_ground_truth_with_unit_confidence(self) -> None:
        request = _request(
            pre_ledger=(_state("holding(apple_1)", TruthValue.TRUE),),
        )
        oracle = OracleEvidenceProvider(
            {
                "holding(apple_1)": TruthValue.FALSE,
                PredicateKey("at", ("apple_1", "stand_1")): TruthValue.TRUE,
            }
        )
        items = oracle.extract(request)
        # Sorted by key -> "at" precedes "holding".
        self.assertEqual([item.key.render() for item in items],
                         ["at(apple_1,stand_1)", "holding(apple_1)"])

        by_key = {item.key.render(): item for item in items}
        holding = by_key["holding(apple_1)"]
        self.assertIs(holding.before, TruthValue.TRUE)   # recovered from pre_ledger
        self.assertIs(holding.after, TruthValue.FALSE)
        self.assertEqual(holding.confidence, 1.0)
        self.assertIs(holding.source, EvidenceSource.DERIVED_GOAL)

        at = by_key["at(apple_1,stand_1)"]
        self.assertIs(at.before, TruthValue.UNKNOWN)     # not present in pre_ledger
        self.assertIs(at.after, TruthValue.TRUE)
        self.assertEqual(at.confidence, 1.0)

    def test_callable_lookup_keyed_on_request(self) -> None:
        def lookup(req: EvidenceRequest):
            return {"near(stand_1)": "true"} if req.step_id == 1 else {}

        oracle = OracleEvidenceProvider(lookup)
        first = oracle.extract(_request(step_id=1))
        second = oracle.extract(_request(step_id=2))
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 0)

    def test_is_deterministic_across_calls(self) -> None:
        request = _request()
        oracle = OracleEvidenceProvider({"open(fridge_1)": TruthValue.TRUE})
        self.assertEqual(oracle.extract(request), oracle.extract(request))


# --------------------------------------------------------------------------- #
# NoisyEvidenceProvider
# --------------------------------------------------------------------------- #
class NoisyEvidenceProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = (
            _evidence("holding(apple_1)", TruthValue.TRUE, TruthValue.TRUE, evidence_id="e1"),
            _evidence("at(apple_1,stand_1)", TruthValue.TRUE, TruthValue.FALSE, evidence_id="e2"),
            _evidence("open(fridge_1)", TruthValue.FALSE, TruthValue.TRUE, evidence_id="e3"),
        )
        self.request = _request()

    def test_drop_rate_one_drops_everything(self) -> None:
        noisy = NoisyEvidenceProvider(
            _FixedProvider(self.base), drop_rate=1.0, seed=0
        )
        self.assertEqual(noisy.extract(self.request), ())

    def test_flip_rate_one_flips_all_definite_values(self) -> None:
        noisy = NoisyEvidenceProvider(
            _FixedProvider(self.base), flip_rate=1.0, seed=0
        )
        output = noisy.extract(self.request)
        self.assertEqual([item.after for item in output],
                         [TruthValue.FALSE, TruthValue.TRUE, TruthValue.FALSE])

    def test_zero_noise_equals_wrapped_provider(self) -> None:
        wrapped = _FixedProvider(self.base)
        noisy = NoisyEvidenceProvider(wrapped, seed=7)
        self.assertEqual(noisy.extract(self.request), wrapped.extract(self.request))

    def test_same_seed_is_identical_across_calls(self) -> None:
        noisy = NoisyEvidenceProvider(
            _FixedProvider(self.base), drop_rate=0.5, flip_rate=0.5, seed=3
        )
        self.assertEqual(noisy.extract(self.request), noisy.extract(self.request))

    def test_different_seed_can_differ(self) -> None:
        # 10 distinct predicates x drop(0.5) x flip(0.5): two independent 20-bit
        # Bernoulli streams colliding is astronomically unlikely; the run is
        # fully deterministic so this is not a flaky check.
        wide = tuple(
            _evidence(f"p({i})", TruthValue.TRUE, TruthValue.TRUE, evidence_id=f"e{i}")
            for i in range(10)
        )
        low = NoisyEvidenceProvider(_FixedProvider(wide), drop_rate=0.5, flip_rate=0.5, seed=0)
        high = NoisyEvidenceProvider(_FixedProvider(wide), drop_rate=0.5, flip_rate=0.5, seed=1)
        self.assertNotEqual(low.extract(self.request), high.extract(self.request))

    def test_false_positive_pool_emits_spurious_items(self) -> None:
        noisy = NoisyEvidenceProvider(
            _FixedProvider(self.base),
            false_positive_rate=1.0,
            false_positive_pool=("ghost(robot_1)",),
            false_positive_confidence=0.5,
            seed=0,
        )
        output = noisy.extract(self.request)
        ghost = [item for item in output if item.key.render() == "ghost(robot_1)"]
        self.assertEqual(len(ghost), 1)
        self.assertEqual(ghost[0].confidence, 0.5)
        self.assertIn(ghost[0].after, (TruthValue.TRUE, TruthValue.FALSE))


# --------------------------------------------------------------------------- #
# calibration / selective risk
# --------------------------------------------------------------------------- #
class CalibrationTests(unittest.TestCase):
    def test_report_types_and_empty(self) -> None:
        empty = evaluate_calibration([])
        self.assertIsInstance(empty, CalibrationReport)
        self.assertEqual(empty.n_samples, 0)
        self.assertEqual(empty.brier, 0.0)
        self.assertEqual(empty.expected_calibration_error, 0.0)

    def test_perfectly_calibrated_input_has_low_ece(self) -> None:
        report = evaluate_calibration(
            [(0.5, True), (0.5, False), (1.0, True), (0.0, False)]
        )
        self.assertLess(report.expected_calibration_error, 1e-9)

    def test_brier_on_known_input(self) -> None:
        # (1 - 1)^2 + (0 - 0)^2 averaged -> 0.0
        self.assertEqual(evaluate_calibration([(1.0, True), (0.0, False)]).brier, 0.0)
        # (1 - 0)^2 + (0 - 1)^2 averaged -> 1.0
        self.assertEqual(evaluate_calibration([(1.0, False), (0.0, True)]).brier, 1.0)


class SelectiveRiskCurveTests(unittest.TestCase):
    def setUp(self) -> None:
        # confidences: 0.9 (correct), 0.4 (wrong), 0.7 (correct); error = 1/3.
        self.predicted = [(0.9, True), (0.4, False), (0.7, True)]

    def test_empty_returns_empty_list(self) -> None:
        self.assertEqual(selective_risk_curve([]), [])

    def test_coverage_is_monotone_non_increasing(self) -> None:
        curve = selective_risk_curve(self.predicted)
        thresholds = [point.confidence_threshold for point in curve]
        self.assertEqual(thresholds, sorted(thresholds))
        coverages = [point.coverage for point in curve]
        for earlier, later in zip(coverages, coverages[1:]):
            self.assertGreaterEqual(earlier, later)

    def test_threshold_zero_matches_overall_error(self) -> None:
        curve = selective_risk_curve(self.predicted)
        point = next(p for p in curve if p.confidence_threshold == 0.0)
        self.assertEqual(point.coverage, 1.0)
        self.assertAlmostEqual(point.selective_risk, 1.0 / 3.0)

    def test_threshold_above_all_confidences_retains_nothing(self) -> None:
        curve = selective_risk_curve(self.predicted, thresholds=[0.95])
        self.assertEqual(len(curve), 1)
        self.assertIsInstance(curve[0], SelectiveRiskPoint)
        self.assertEqual(curve[0].coverage, 0.0)
        self.assertEqual(curve[0].selective_risk, 0.0)
        self.assertEqual(curve[0].n_retained, 0)


# --------------------------------------------------------------------------- #
# compare_providers
# --------------------------------------------------------------------------- #
class CompareProvidersTests(unittest.TestCase):
    def test_returns_expected_keys_and_sensible_values(self) -> None:
        request = _request(
            pre_ledger=(_state("holding(apple_1)", TruthValue.TRUE),),
        )
        oracle = OracleEvidenceProvider(
            {
                "holding(apple_1)": TruthValue.FALSE,
                "at(apple_1,stand_1)": TruthValue.TRUE,
            }
        )
        # Candidate: two correct assertions plus one spurious (hallucinated) one.
        candidate = _FixedProvider(
            (
                _evidence(
                    "holding(apple_1)", TruthValue.TRUE, TruthValue.FALSE,
                    evidence_id="c1", confidence=0.9,
                ),
                _evidence(
                    "at(apple_1,stand_1)", TruthValue.UNKNOWN, TruthValue.TRUE,
                    evidence_id="c2", confidence=0.7,
                ),
                _evidence(
                    "open(fridge_1)", TruthValue.FALSE, TruthValue.TRUE,
                    evidence_id="c3", confidence=0.6,
                ),
            )
        )
        report = compare_providers(oracle, candidate, [request], seed=42)
        expected_keys = {
            "n_requests", "n_predicate_targets", "n_candidate_assertions",
            "n_candidate_correct", "predicate_precision", "predicate_recall",
            "predicate_f1", "brier", "expected_calibration_error",
            "n_calibration_samples", "selective_risk_curve",
            "selective_risk_at_0.5", "seed",
        }
        self.assertEqual(set(report), expected_keys)
        self.assertEqual(report["n_requests"], 1)
        self.assertEqual(report["n_predicate_targets"], 2)
        self.assertEqual(report["n_candidate_assertions"], 3)
        self.assertEqual(report["n_candidate_correct"], 2)
        self.assertAlmostEqual(report["predicate_precision"], 2.0 / 3.0)
        self.assertEqual(report["predicate_recall"], 1.0)
        self.assertEqual(report["n_calibration_samples"], 3)
        self.assertEqual(report["seed"], 42)
        # brier = ((0.9-1)^2 + (0.7-1)^2 + (0.6-0)^2) / 3
        self.assertAlmostEqual(report["brier"], (0.01 + 0.09 + 0.36) / 3.0)
        self.assertIsInstance(report["selective_risk_at_0.5"], SelectiveRiskPoint)

    def test_empty_requests_yields_zeros(self) -> None:
        oracle = OracleEvidenceProvider({"p(x)": TruthValue.TRUE})
        report = compare_providers(oracle, _FixedProvider(()), [])
        self.assertEqual(report["n_requests"], 0)
        self.assertEqual(report["predicate_precision"], 0.0)
        self.assertEqual(report["predicate_recall"], 0.0)
        self.assertEqual(report["n_calibration_samples"], 0)


if __name__ == "__main__":
    unittest.main()
