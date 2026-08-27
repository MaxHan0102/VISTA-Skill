from vista_skill.schemas import SkillField, TerminationPolicy
from vista_skill.skills import (
    render_skill,
    target_habitat_skill_v1,
    target_navigation_skill_v1,
)


def test_posthoc_target_skills_use_complete_five_field_schema() -> None:
    for skill in (target_habitat_skill_v1(), target_navigation_skill_v1()):
        assert skill.frozen
        assert skill.version == 1
        assert skill.parent_version == 0
        assert skill.termination_policy is TerminationPolicy.ALL_GOALS_EVIDENCE
        assert skill.metadata["oracle_diagnostic"] is True
        assert all(skill.statements(field) for field in SkillField)
        assert len(render_skill(skill).split()) <= 512


def test_habitat_target_encodes_observed_search_and_delivery_repairs() -> None:
    text = render_skill(target_habitat_skill_v1())
    assert "Navigation success" in text
    assert "do not repeat Pick" in text
    assert "Never Pick while holding" in text
    assert "remove or detach" in text
    assert target_habitat_skill_v1().prediction_rules


def test_navigation_target_requires_single_step_distance_control() -> None:
    skill = target_navigation_skill_v1()
    text = render_skill(skill)
    assert "exactly ONE primitive action" in text
    assert "distance strictly decreases" in text
    assert "Target-distance feedback overrides" not in text
    assert "Numeric target-distance feedback overrides" in text
    assert not skill.prediction_rules
