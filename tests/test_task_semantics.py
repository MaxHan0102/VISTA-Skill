from __future__ import annotations

from vista_skill.integrations.embodiedbench.task_semantics import (
    habitat_episode_semantic_tags,
    load_habitat_task_semantic_tags,
    predicate_names,
)


def test_predicate_names_ignore_expression_metadata() -> None:
    goal = {
        "expr_type": "AND",
        "inputs": [{"name": "X", "expr_type": "ball"}],
        "sub_exprs": ["on_top(X,table)", "not_holding()"],
    }
    assert predicate_names(goal) == ("on_top", "not_holding")


def test_habitat_semantic_tags_are_symbolic_and_outcome_independent() -> None:
    episode = {
        "goal_preds": {
            "inputs": [{"name": "X"}, {"name": "Y"}],
            "sub_exprs": ["on_top(X,table)", "closed_fridge(fridge)"],
        },
        "subgoals": [["robot_at(table)", "holding(ball)"]],
        # These fields must not influence semantic membership.
        "task_success": 1.0,
        "task_progress": 1.0,
        "trajectory": ["oracle-only-action"],
    }
    tags = habitat_episode_semantic_tags(episode)
    assert "action:nav" in tags
    assert "action:pick" in tags
    assert "action:place" in tags
    assert "action:open" in tags
    assert "action:close" in tags
    assert "structure:multi_entity_goal" in tags
    changed = dict(episode, task_success=0.0, task_progress=0.0, trajectory=[])
    assert habitat_episode_semantic_tags(changed) == tags


def test_train_validation_semantic_tags_cover_every_episode() -> None:
    indexed = load_habitat_task_semantic_tags(
        "EmbodiedBench/embodiedbench/envs/eb_habitat/datasets/"
        "train_validation.pickle"
    )
    assert len(indexed) == 100
    assert all(tags for tags in indexed.values())
