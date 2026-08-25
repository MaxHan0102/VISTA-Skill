from __future__ import annotations

from vista_skill.evidence_guard import (
    EvidenceGuardConfig,
    EvidenceReliabilityGuard,
    GuardMode,
)
from vista_skill.schemas import (
    ActionCall,
    EvidenceSource,
    PredicateEvidence,
    PredicateKey,
    PredicateState,
    TruthValue,
)


def _evidence(
    value: TruthValue,
    *,
    source: EvidenceSource,
    confidence: float,
    coverage: float = 1.0,
    evidence_id: str,
    key: PredicateKey | None = None,
) -> PredicateEvidence:
    return PredicateEvidence(
        key=key or PredicateKey("holding", ("apple",)),
        before=TruthValue.UNKNOWN,
        after=value,
        confidence=confidence,
        coverage=coverage,
        source=source,
        evidence_id=evidence_id,
        timestamp=1,
    )


def test_strict_guard_downgrades_unequal_confidence_polarity_conflict() -> None:
    feedback = _evidence(
        TruthValue.TRUE,
        source=EvidenceSource.ENV_FEEDBACK,
        confidence=0.98,
        evidence_id="feedback",
    )
    visual = _evidence(
        TruthValue.FALSE,
        source=EvidenceSource.VISUAL_PAIR,
        confidence=0.80,
        evidence_id="visual",
    )
    result = EvidenceReliabilityGuard().fuse((feedback,), (visual,))
    assert result.evidence[0].after is TruthValue.UNKNOWN
    assert result.decisions[0].decision == "downgraded_cross_source_conflict"


def test_authority_aware_guard_retains_explicit_feedback_on_conflict() -> None:
    guard = EvidenceReliabilityGuard(
        EvidenceGuardConfig(mode=GuardMode.AUTHORITY_AWARE)
    )
    feedback = _evidence(
        TruthValue.TRUE,
        source=EvidenceSource.ENV_FEEDBACK,
        confidence=0.98,
        evidence_id="feedback",
    )
    visual = _evidence(
        TruthValue.FALSE,
        source=EvidenceSource.VISUAL_PAIR,
        confidence=0.99,
        evidence_id="visual",
    )
    result = guard.fuse((feedback,), (visual,))
    assert result.evidence[0].after is TruthValue.TRUE
    assert result.evidence[0].source is EvidenceSource.ENV_FEEDBACK
    assert result.decisions[0].decision == "accepted_feedback_priority_conflict"


def test_visual_only_must_pass_confidence_and_coverage() -> None:
    visual = _evidence(
        TruthValue.TRUE,
        source=EvidenceSource.VISUAL_PAIR,
        confidence=0.90,
        coverage=0.20,
        evidence_id="visual",
    )
    result = EvidenceReliabilityGuard().fuse((), (visual,))
    assert result.evidence[0].after is TruthValue.UNKNOWN
    assert result.decisions[0].decision == "downgraded_insufficient_evidence"


def test_cross_source_agreement_keeps_combined_provenance() -> None:
    feedback = _evidence(
        TruthValue.TRUE,
        source=EvidenceSource.ENV_FEEDBACK,
        confidence=0.98,
        evidence_id="feedback",
    )
    visual = _evidence(
        TruthValue.TRUE,
        source=EvidenceSource.VISUAL_PAIR,
        confidence=0.80,
        evidence_id="visual",
    )
    result = EvidenceReliabilityGuard().fuse((feedback,), (visual,))
    assert result.evidence[0].after is TruthValue.TRUE
    assert result.evidence[0].source is EvidenceSource.ACTIVE_OBSERVATION
    assert result.evidence[0].evidence_id == "feedback|visual"


def test_guard_audit_contains_no_prediction_or_skill_payload() -> None:
    feedback = _evidence(
        TruthValue.TRUE,
        source=EvidenceSource.ENV_FEEDBACK,
        confidence=0.98,
        evidence_id="feedback",
    )
    payload = repr(EvidenceReliabilityGuard().fuse((feedback,), ())).lower()
    for forbidden in ("expected", "skill", "mismatch", "attribution", "patch", "candidate"):
        assert forbidden not in payload


def test_guard_downgrades_visual_only_state_claim_after_failed_action() -> None:
    visual = _evidence(
        TruthValue.TRUE,
        source=EvidenceSource.VISUAL_PAIR,
        confidence=1.0,
        evidence_id="failed-visual",
    )
    result = EvidenceReliabilityGuard().fuse(
        (), (visual,), last_action_success=False
    )
    assert result.evidence[0].after is TruthValue.UNKNOWN
    assert result.decisions[0].decision == "downgraded_failed_action_visual_only"


def test_guard_requires_feedback_for_visual_only_negative_at_relation() -> None:
    visual = _evidence(
        TruthValue.FALSE,
        source=EvidenceSource.VISUAL_PAIR,
        confidence=1.0,
        evidence_id="negative-at",
        key=PredicateKey("at", ("apple", "table")),
    )
    result = EvidenceReliabilityGuard().fuse(
        (), (visual,), last_action_success=True
    )
    assert result.evidence[0].after is TruthValue.UNKNOWN
    assert (
        result.decisions[0].decision
        == "downgraded_visual_negative_without_feedback"
    )


def test_guard_grounds_successful_place_from_pre_action_holding_state() -> None:
    not_holding = _evidence(
        TruthValue.TRUE,
        source=EvidenceSource.ENV_FEEDBACK,
        confidence=0.98,
        evidence_id="place-feedback",
        key=PredicateKey("not_holding"),
    )
    held = PredicateState(
        key=PredicateKey("holding", ("apple",)),
        value=TruthValue.TRUE,
        confidence=0.98,
        source="ledger",
        evidence_ids=("pick-feedback",),
        timestamp=3,
    )
    action = ActionCall(4, "place", ("table",), "place on table")
    result = EvidenceReliabilityGuard(
        EvidenceGuardConfig(mode=GuardMode.AUTHORITY_AWARE)
    ).fuse(
        (not_holding,),
        (),
        (held,),
        last_action_success=True,
        action=action,
    )
    by_key = {item.key: item for item in result.evidence}
    assert by_key[PredicateKey("at", ("apple", "table"))].after is TruthValue.TRUE
    assert (
        by_key[PredicateKey("at", ("apple", "table"))].source
        is EvidenceSource.ENV_FEEDBACK
    )


def test_guard_retains_uncovered_pre_state_after_failed_high_level_action() -> None:
    near = PredicateState(
        key=PredicateKey("near", ("table",)),
        value=TruthValue.TRUE,
        confidence=0.9,
        source="ledger",
        evidence_ids=("nav-feedback",),
        timestamp=2,
    )
    action = ActionCall(5, "pick", ("apple",), "pick apple")
    result = EvidenceReliabilityGuard().fuse(
        (),
        (),
        (near,),
        last_action_success=False,
        action=action,
    )
    assert result.evidence[0].key == near.key
    assert result.evidence[0].before is TruthValue.TRUE
    assert result.evidence[0].after is TruthValue.TRUE
    assert "failed_action_persistence" in result.evidence[0].evidence_id


def test_guard_downgrades_visual_only_predicate_unrelated_to_action() -> None:
    near = _evidence(
        TruthValue.FALSE,
        source=EvidenceSource.VISUAL_PAIR,
        confidence=1.0,
        evidence_id="place-near-hallucination",
        key=PredicateKey("near", ("table",)),
    )
    action = ActionCall(4, "place", ("counter",), "place on counter")
    result = EvidenceReliabilityGuard().fuse(
        (),
        (near,),
        last_action_success=True,
        action=action,
    )
    assert result.evidence[0].after is TruthValue.UNKNOWN
    assert (
        result.decisions[0].decision
        == "downgraded_action_irrelevant_visual_only"
    )
