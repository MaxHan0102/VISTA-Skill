from __future__ import annotations

import json

from vista_skill.evaluation import composite_task_score
from vista_skill.integrations.embodiedbench.cli import (
    _load_completed_audit_rollout,
    _next_audit_resume_artifact,
)


def _event(event_type: str, payload: dict[str, object]) -> str:
    return json.dumps({"event_type": event_type, "payload": payload}) + "\n"


def test_completed_audit_rollout_is_recovered_from_disk(tmp_path) -> None:
    artifact = tmp_path / "s101_digest.jsonl"
    artifact.write_text(
        _event("transition", {"episode_id": "80"})
        + _event(
            "episode_result",
            {
                "episode_id": "80",
                "task_success": 1.0,
                "task_progress": 0.75,
                "invalid_actions": 1,
                "environment_steps": 4,
            },
        )
    )
    score = _load_completed_audit_rollout(artifact, expected_episode_id="80")
    assert score is not None
    assert score.score == composite_task_score(
        task_success=1.0, task_progress=0.75, invalid_action_ratio=0.25
    )
    assert score.success is True


def test_incomplete_base_uses_complete_resume_without_overwrite(tmp_path) -> None:
    artifact = tmp_path / "s102_digest.jsonl"
    artifact.write_text(_event("evaluation_label", {"episode_id": "80"}))
    resume = _next_audit_resume_artifact(artifact)
    assert resume.name == "s102_digest.resume1.jsonl"
    resume.write_text(
        _event(
            "episode_result",
            {
                "episode_id": "80",
                "task_success": 0.0,
                "task_progress": 0.5,
                "invalid_actions": 0,
                "environment_steps": 2,
            },
        )
    )
    score = _load_completed_audit_rollout(artifact, expected_episode_id="80")
    assert score is not None
    assert score.success is False
    assert _next_audit_resume_artifact(artifact).name == "s102_digest.resume2.jsonl"


def test_incomplete_audit_rollout_is_not_treated_as_cached(tmp_path) -> None:
    artifact = tmp_path / "s103_digest.jsonl"
    artifact.write_text(_event("transition", {"episode_id": "80"}))
    assert _load_completed_audit_rollout(
        artifact, expected_episode_id="80"
    ) is None
