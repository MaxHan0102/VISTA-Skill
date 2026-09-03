from __future__ import annotations

import json

from scripts.phase5_shadow_replay import replay


def test_replay_changes_retention_but_never_promotion(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "protocol": {},
                "task_manifest": "fixture",
                "evolution_seeds": [0],
                "environment": {},
                "executor": {},
                "budgets": {
                    "patch_operations_per_round": 1,
                    "active_skill_words": 32,
                    "proxy_episodes": 4,
                    "finalist_episodes": 4,
                },
                "credit": {},
                "gate": {
                    "semantic_affected_enabled": True,
                    "semantic_min_affected_tasks": 2,
                    "semantic_min_protected_tasks": 2,
                    "shadow_candidates_enabled": True,
                },
                "final_test": {
                    "skill_frozen": True,
                    "teacher_enabled": False,
                    "attribution_enabled": False,
                    "patching_enabled": False,
                },
            }
        ),
        encoding="utf-8",
    )
    lineage = tmp_path / "lineage.jsonl"
    lineage.write_text(
        json.dumps(
            {
                "decision": {
                    "accepted": False,
                    "patch_id": "patch-1",
                    "stages": [
                        {
                            "stage": "paired_proxy",
                            "passed": False,
                            "reason": "affected benefit is underpowered",
                            "metrics": {
                                "episodes": 4,
                                "affected_mean_delta": 0.1,
                                "protected_mean_delta": 0.0,
                                "worst_subgroup_delta": 0.0,
                            },
                        }
                    ],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = replay(config, [lineage])

    assert result["replayed_counts"]["shadow"] == 1
    assert result["replayed_counts"]["promoted"] == 0
    assert result["promotion_change_count"] == 0
    assert not result["decisions"][0]["promotion_changed"]
