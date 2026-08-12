from __future__ import annotations

from vista_skill.schemas import TerminationPolicy
from vista_skill.skills import initialize_nav_skill, initialize_shared_skill


def test_nav_skill_structure() -> None:
    skill = initialize_nav_skill()
    assert skill.skill_id == "shared_navigation"
    assert skill.termination_policy is TerminationPolicy.ALL_GOALS_EVIDENCE
    assert skill.procedure and skill.effect and skill.termination and skill.constraint
    # Nav relies on NavActionSchema for structural predictions.
    assert skill.prediction_rules == ()
    assert skill.metadata.get("env") == "eb_navigation"


def test_nav_skill_is_distinct_from_habitat_seed() -> None:
    assert initialize_nav_skill().skill_id != initialize_shared_skill().skill_id


def test_nav_skill_freezes_for_evaluation() -> None:
    from dataclasses import replace

    frozen = replace(initialize_nav_skill(), frozen=True)
    assert frozen.frozen
