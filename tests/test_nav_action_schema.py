from __future__ import annotations

from vista_skill.action_schema import (
    ActionSchema,
    FixedActionSchema,
    NavActionSchema,
    parse_action_call,
)
from vista_skill.belief import BeliefLedger
from vista_skill.schemas import DeltaSource, PredicateKey, TruthValue
from vista_skill.skills import initialize_nav_skill


def _action(action_type: str):
    return parse_action_call(0, (action_type, []))


def test_nav_schema_move_emits_position_changed() -> None:
    schema = NavActionSchema()
    for action_type in ("move_forward", "move_backward", "move_right", "move_left"):
        changes = schema.compile(_action(action_type), BeliefLedger(), initialize_nav_skill(), ())
        matches = [c for c in changes if c.key.name == "position_changed"]
        assert matches, action_type
        assert matches[0].after is TruthValue.TRUE
        assert matches[0].source is DeltaSource.ACTION_SCHEMA


def test_nav_schema_turn_emits_heading_changed() -> None:
    schema = NavActionSchema()
    for action_type in ("turn_right", "turn_left"):
        changes = schema.compile(_action(action_type), BeliefLedger(), initialize_nav_skill(), ())
        assert any(c.key.name == "heading_changed" for c in changes)


def test_nav_schema_look_emits_camera_tilt_changed() -> None:
    schema = NavActionSchema()
    for action_type in ("look_up", "look_down"):
        changes = schema.compile(_action(action_type), BeliefLedger(), initialize_nav_skill(), ())
        assert any(c.key.name == "camera_tilt_changed" for c in changes)


def test_nav_schema_never_predicts_near_target() -> None:
    """Geometric honesty: the schema must not predict near(target)."""
    schema = NavActionSchema()
    goals = (PredicateKey("near", ("bread",)),)
    nav_types = [
        "move_forward", "move_backward", "move_right", "move_left",
        "turn_right", "turn_left", "look_up", "look_down",
    ]
    for action_type in nav_types:
        changes = schema.compile(_action(action_type), BeliefLedger(), initialize_nav_skill(), goals)
        assert not any(c.key.name == "near" for c in changes), action_type


def test_nav_schema_precondition_checks_empty() -> None:
    schema = NavActionSchema()
    for action_type in ("move_forward", "turn_right", "look_down"):
        assert schema.precondition_checks(_action(action_type), BeliefLedger()) == []


def test_both_schemas_satisfy_protocol() -> None:
    assert isinstance(NavActionSchema(), ActionSchema)
    assert isinstance(FixedActionSchema(), ActionSchema)  # regression


def test_nav_schema_termination_reads_near_from_ledger_only() -> None:
    schema = NavActionSchema()
    goals = (PredicateKey("near", ("bread",)),)
    # Empty ledger -> near unknown -> task_complete predicted FALSE (not claimed from the move).
    changes = schema.compile(_action("move_forward"), BeliefLedger(), initialize_nav_skill(), goals)
    completion = [c for c in changes if c.key.name == "task_complete"]
    assert completion and completion[0].after is TruthValue.FALSE
