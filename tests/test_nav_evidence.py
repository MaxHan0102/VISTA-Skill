from __future__ import annotations

from vista_skill.evidence import EvidenceExtractor, _nav_feedback_strategy
from vista_skill.schemas import (
    ActionCall,
    EvidenceRequest,
    PredicateKey,
    TruthValue,
)


def _request(
    action_type: str = "move_forward",
    feedback: str = "",
    success: bool = True,
    goals: tuple = (PredicateKey("near", ("bread",)),),
) -> EvidenceRequest:
    return EvidenceRequest(
        episode_id="e0",
        step_id=1,
        instruction="navigate to the bread",
        action=ActionCall(
            action_id=0,
            action_type=action_type,
            arguments=(),
            text=action_type,
            raw_action=action_type,
        ),
        pre_image="",
        post_image="",
        feedback=feedback,
        last_action_success=success,
        pre_ledger=(),
        goal_predicates=goals,
    )


def _extract(request: EvidenceRequest):
    return EvidenceExtractor(feedback_strategy=_nav_feedback_strategy).extract(request)


def test_nav_near_target_true_within_threshold() -> None:
    evidence = _extract(_request(feedback="Target distance: 0.5m.", success=True))
    near = [e for e in evidence if e.key.name == "near"]
    assert near and near[0].after is TruthValue.TRUE


def test_nav_near_target_false_outside_threshold() -> None:
    evidence = _extract(_request(feedback="Target distance: 2.5m.", success=True))
    near = [e for e in evidence if e.key.name == "near"]
    assert near and near[0].after is TruthValue.FALSE


def test_nav_blocked_move_emits_position_changed_false() -> None:
    evidence = _extract(_request(action_type="move_forward", feedback="blocked", success=False))
    pc = [e for e in evidence if e.key.name == "position_changed"]
    assert pc and pc[0].after is TruthValue.FALSE


def test_nav_successful_move_emits_position_changed_true() -> None:
    evidence = _extract(_request(action_type="move_forward", feedback="Target distance: 1.0m.", success=True))
    assert any(
        e.key.name == "position_changed" and e.after is TruthValue.TRUE for e in evidence
    )


def test_nav_task_completion_derives_from_near_true() -> None:
    evidence = _extract(_request(feedback="Target distance: 0.3m.", success=True))
    completion = [e for e in evidence if e.key.name == "task_complete"]
    assert completion and completion[0].after is TruthValue.TRUE


def test_habitat_default_strategy_ignores_nav_distance() -> None:
    """Regression: the default extractor must not parse nav distance into near(target)."""
    evidence = EvidenceExtractor().extract(
        _request(action_type="move_forward", feedback="Target distance: 0.3m.", success=True)
    )
    assert not any(e.key.name == "near" and e.after is TruthValue.TRUE for e in evidence)
