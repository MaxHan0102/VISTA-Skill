from scripts.phase5_variance_audit import analyze_records


def _records(deltas_by_task):
    rows = []
    for episode_id, deltas in deltas_by_task.items():
        for seed, delta in enumerate(deltas):
            rows.extend(
                (
                    {
                        "episode_id": episode_id,
                        "seed": seed,
                        "arm": "parent",
                        "score": 0.0,
                        "task_success": 0.0,
                        "task_progress": 0.0,
                        "trajectory_signature": f"p-{episode_id}-{seed}",
                    },
                    {
                        "episode_id": episode_id,
                        "seed": seed,
                        "arm": "candidate",
                        "score": delta,
                        "task_success": float(delta > 0),
                        "task_progress": float(delta > 0),
                        "trajectory_signature": f"c-{episode_id}-{seed}",
                    },
                )
            )
    return rows


def test_variance_audit_prefers_independent_tasks_for_exact_deltas():
    analysis = analyze_records(
        _records({"a": (1.0, 1.0, 1.0), "b": (-1.0, -1.0, -1.0)}),
        episode_ids=("a", "b"),
        seeds=(0, 1, 2),
        variance_share_threshold=0.2,
        minimum_delta_divergent_tasks=2,
    )

    assert analysis["within_task_delta_variance"] == 0.0
    assert analysis["delta_divergent_tasks"] == 0
    assert analysis["decision"] == "independent_task_path"
    assert analysis["pass_at_k_is_promotion_eligible"] is False


def test_variance_audit_requires_both_registered_thresholds():
    analysis = analyze_records(
        _records({"a": (-1.0, 0.0, 1.0), "b": (1.0, 0.0, -1.0)}),
        episode_ids=("a", "b"),
        seeds=(0, 1, 2),
        variance_share_threshold=0.2,
        minimum_delta_divergent_tasks=2,
    )

    assert analysis["within_task_variance_share"] >= 0.2
    assert analysis["delta_divergent_tasks"] == 2
    assert analysis["decision"] == "repeated_rollout_path"


def test_variance_audit_rejects_incomplete_registered_grid():
    records = _records({"a": (1.0, 1.0, 1.0)})
    records.pop()

    try:
        analyze_records(
            records,
            episode_ids=("a",),
            seeds=(0, 1, 2),
            variance_share_threshold=0.2,
            minimum_delta_divergent_tasks=1,
        )
    except ValueError as error:
        assert "missing registered rollout records" in str(error)
    else:
        raise AssertionError("incomplete grid must fail closed")
