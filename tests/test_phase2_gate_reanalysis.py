from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parents[1] / "scripts" / "phase2_gate_reanalysis.py"
_SPEC = importlib.util.spec_from_file_location("phase2_gate_reanalysis", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_contains_goal_predicate = _MODULE._contains_goal_predicate
_resolve_fault = _MODULE._resolve_fault
_task_is_affected = _MODULE._task_is_affected


def test_effect_fault_marks_direct_and_nested_on_top_goals() -> None:
    direct = {
        "goal_preds": {
            "sub_exprs": ["on_top(X, table)", "not_holding()"],
            "inputs": [{"name": "X"}],
        }
    }
    nested = {
        "goal_preds": {
            "sub_exprs": [
                "not_holding()",
                {"expr_type": "NAND", "sub_exprs": ["on_top(X, counter)"]},
            ]
        }
    }
    assert _task_is_affected(direct, "effect_pick_inversion")
    assert _task_is_affected(nested, "effect_pick_inversion")


def test_effect_fault_protects_navigation_and_articulation_goals() -> None:
    navigation = {"goal_preds": {"sub_exprs": ["robot_at(table)"]}}
    articulation = {"goal_preds": {"sub_exprs": ["closed_fridge(fridge)"]}}
    assert not _task_is_affected(navigation, "effect_pick_inversion")
    assert not _task_is_affected(articulation, "effect_pick_inversion")
    assert not _contains_goal_predicate(navigation, "on_top")


def test_multihold_fault_requires_two_top_level_object_variables() -> None:
    one = {"goal_preds": {"inputs": [{"name": "X"}]}}
    two = {"goal_preds": {"inputs": [{"name": "X"}, {"name": "Y"}]}}
    assert not _task_is_affected(one, "constraint_pick_multihold")
    assert _task_is_affected(two, "constraint_pick_multihold")


def test_auto_fault_is_inferred_and_explicit_mismatch_is_rejected() -> None:
    lineage = [{"protocol": {"skill_fault": "effect_pick_inversion"}}]
    assert _resolve_fault("auto", lineage) == "effect_pick_inversion"
    with pytest.raises(ValueError, match="conflicts with lineage"):
        _resolve_fault("constraint_pick_multihold", lineage)
