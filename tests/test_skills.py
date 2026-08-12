from __future__ import annotations

from vista_skill.action_schema import FixedActionSchema, parse_action_call
from vista_skill.belief import BeliefLedger
from vista_skill.pipeline import VistaSkillEngine
from vista_skill.schemas import SkillField, SkillSpec
from vista_skill.skills import (
    empty_shared_skill,
    initialize_shared_skill,
    minimal_shared_skill,
)


def _total_statements(skill: SkillSpec) -> int:
    return sum(len(skill.statements(field)) for field in SkillField)


def test_minimal_shared_skill_has_valid_structure() -> None:
    skill = minimal_shared_skill()
    assert isinstance(skill, SkillSpec)
    assert skill.skill_id == "shared_embodied_execution"
    assert skill.version == 0
    assert skill.parent_version is None
    assert skill.termination_policy is not None
    assert _total_statements(skill) >= 1


def test_minimal_has_strictly_fewer_effect_and_constraint_statements() -> None:
    baseline = initialize_shared_skill()
    minimal = minimal_shared_skill()
    assert len(minimal.effect) < len(baseline.effect)
    assert len(minimal.constraint) < len(baseline.constraint)


def test_minimal_is_strictly_weaker_than_spec_init() -> None:
    assert _total_statements(minimal_shared_skill()) < _total_statements(
        initialize_shared_skill()
    )


def test_minimal_has_empty_prediction_rules() -> None:
    assert minimal_shared_skill().prediction_rules == ()


def test_empty_shared_skill_has_empty_statement_bodies() -> None:
    skill = empty_shared_skill()
    assert isinstance(skill, SkillSpec)
    for field in SkillField:
        assert skill.statements(field) == ()
    assert skill.prediction_rules == ()


def test_empty_shared_skill_retains_required_identity() -> None:
    skill = empty_shared_skill()
    assert skill.skill_id == "shared_embodied_execution"
    assert skill.version == 0
    assert skill.termination_policy is not None


def test_both_variants_share_skill_id_with_spec_init() -> None:
    baseline = initialize_shared_skill()
    for skill in (minimal_shared_skill(), empty_shared_skill()):
        assert skill.skill_id == baseline.skill_id
        assert skill.version == baseline.version


def test_minimal_compiles_under_fixed_action_schema_without_error() -> None:
    schema = FixedActionSchema()
    action = parse_action_call(0, ("pick_apple", ["robot_0"]))
    changes = schema.compile(action, BeliefLedger(), minimal_shared_skill())
    # Primitive action-schema transitions are still produced even with no
    # compiled skill prediction_rules.
    assert changes
    assert any(
        item.source.name != "SKILL" or item.skill_field is not None
        for item in changes
    )


def test_empty_compiles_under_fixed_action_schema_without_error() -> None:
    schema = FixedActionSchema()
    action = parse_action_call(0, ("nav", ["stand_1"]))
    changes = schema.compile(action, BeliefLedger(), empty_shared_skill())
    assert changes


def test_minimal_compiles_with_goal_predicates_and_termination_policy() -> None:
    from vista_skill.schemas import PredicateKey

    schema = FixedActionSchema()
    ledger = BeliefLedger()
    action = parse_action_call(0, ("look", []))
    goals = (PredicateKey.parse("near(stand_1)"),)
    changes = schema.compile(
        action, ledger, minimal_shared_skill(), goals
    )
    termination = next(
        item for item in changes if item.key.name == "task_complete"
    )
    assert termination is not None


def test_empty_compiles_with_goal_predicates_and_termination_policy() -> None:
    from vista_skill.schemas import PredicateKey

    schema = FixedActionSchema()
    ledger = BeliefLedger()
    action = parse_action_call(0, ("look", []))
    goals = (PredicateKey.parse("near(stand_1)"),)
    changes = schema.compile(action, ledger, empty_shared_skill(), goals)
    assert any(item.key.name == "task_complete" for item in changes)


def test_minimal_is_accepted_by_vista_skill_engine() -> None:
    engine = VistaSkillEngine(minimal_shared_skill())
    assert engine.skill.skill_id == "shared_embodied_execution"
    assert engine.frozen is False


def test_empty_is_accepted_by_vista_skill_engine() -> None:
    engine = VistaSkillEngine(empty_shared_skill())
    assert engine.skill.skill_id == "shared_embodied_execution"
    assert engine.frozen is False


def test_minimal_is_deterministic() -> None:
    assert minimal_shared_skill() == minimal_shared_skill()


def test_empty_is_deterministic() -> None:
    assert empty_shared_skill() == empty_shared_skill()
