from __future__ import annotations

from vista_skill.evolution import GateStageResult

from scripts.phase5_shadow_confirmation import classify_shadow_confirmation


def _stage(*, passed=False, affected=0.1, protected=0.0, subgroup=0.0):
    return GateStageResult(
        "paired_finalist",
        passed,
        "fixture",
        {
            "affected_mean_delta": affected,
            "protected_mean_delta": protected,
            "worst_subgroup_delta": subgroup,
        },
    )


def test_shadow_confirmation_outcomes_are_mutually_exclusive() -> None:
    kwargs = {"protected_tolerance": 0.05, "subgroup_tolerance": 0.05}
    assert classify_shadow_confirmation(_stage(passed=True), **kwargs) == "supportive"
    assert classify_shadow_confirmation(_stage(), **kwargs) == "inconclusive"
    assert (
        classify_shadow_confirmation(_stage(protected=-0.2), **kwargs)
        == "contradictory"
    )
    assert (
        classify_shadow_confirmation(_stage(affected=0.0), **kwargs)
        == "contradictory"
    )
