from __future__ import annotations

import json
from dataclasses import replace

from vista_skill.action_schema import parse_action_call
from vista_skill.artifacts import JsonlArtifactWriter
from vista_skill.integrations.embodiedbench.runner import HabitatRolloutRunner
from vista_skill.pipeline import VistaSkillEngine
from vista_skill.schemas import (
    EvidenceSource,
    PredicateEvidence,
    PredicateKey,
    SkillField,
    TemporalSkillRule,
    TransitionEvent,
    TruthValue,
)
from vista_skill.skills import interface_only_shared_skill
from vista_skill.temporal import TemporalRuleMonitor


def _rule() -> TemporalSkillRule:
    return TemporalSkillRule(
        rule_id="recover_failed_pick",
        field=SkillField.CONSTRAINT,
        trigger_action_type="pick",
        trigger_success=False,
        trigger_predicate="near({arg0})",
        trigger_value=TruthValue.FALSE,
        blocked_action_type="pick",
        recovery_action_types=("nav",),
    )


def _event(action, *, success: bool, evidence=(), step: int = 1) -> TransitionEvent:
    return TransitionEvent(
        event_id=f"event-{step}",
        episode_id="episode",
        task_id="task",
        step_id=step,
        instruction="pick the apple",
        action=action,
        skill_id="shared_embodied_execution",
        skill_version=1,
        pre_image="pre.png",
        post_image="post.png",
        feedback="",
        last_action_success=success,
        pre_ledger=(),
        goal_predicates=(),
        expected_delta=(),
        evidence_delta=tuple(evidence),
        mismatches=(),
    )


def test_temporal_monitor_blocks_same_retry_until_recovery() -> None:
    skill = replace(interface_only_shared_skill(), temporal_rules=(_rule(),))
    monitor = TemporalRuleMonitor()
    monitor.start_episode(skill)
    pick_apple = parse_action_call(1, ("pick", ["apple_1", "robot_0"]))
    near_false = PredicateEvidence(
        PredicateKey("near", ("apple_1",)),
        TruthValue.UNKNOWN,
        TruthValue.FALSE,
        0.95,
        EvidenceSource.ENV_FEEDBACK,
        "ev-near",
        1,
    )
    monitor.observe(_event(pick_apple, success=False, evidence=(near_false,)))

    assert not monitor.admit(pick_apple).allowed
    pick_mug = parse_action_call(2, ("pick", ["mug_1", "robot_0"]))
    assert monitor.admit(pick_mug).allowed
    assert "apple_1" in monitor.render()

    nav = parse_action_call(0, ("nav", ["counter_1"]))
    monitor.observe(_event(nav, success=True, step=2))
    assert monitor.admit(pick_apple).allowed
    assert monitor.render() == ""


class _GuardEnvironment:
    number_of_episodes = 1.0
    _current_episode_num = 0
    _current_step = 0
    _max_episode_steps = 4
    language_skill_set = ["navigate to apple", "pick up apple"]
    skill_set = [("nav", ["apple_1"]), ("pick", ["apple_1", "robot_0"])]
    episode_language_instruction = "Pick up the apple."

    class Episode:
        episode_id = "0"
        instruct_id = "task"

    def current_episode(self, all_info=True):
        return self.Episode()

    def reset(self):
        self._current_episode_num += 1
        self._current_step = 0
        return {"frame": 0}

    def step(self, action, reasoning=""):
        self._current_step += 1
        first_pick = action == 1 and self._current_step == 1
        done = action == 1 and not first_pick
        success = not first_pick
        feedback = (
            "Last action is invalid. Robot cannot pick any object that is not near the robot."
            if first_pick
            else "Last action executed successfully."
        )
        return (
            {"frame": self._current_step},
            float(success),
            done,
            {
                "env_step": self._current_step,
                "last_action_success": float(success),
                "env_feedback": feedback,
                "task_success": float(done),
                "task_progress": float(done),
                "subgoal_reward": 0.0,
            },
        )

    def save_image(self, observation):
        return f"step_{observation['frame']}.png"


class _SequencePlanner:
    planner_steps = 0
    output_json_error = 0

    def __init__(self):
        self.plans = iter(([1], [1], [0], [1]))
        self.feedback = []

    def reset(self):
        pass

    def act(self, image, instruction):
        self.planner_steps += 1
        return next(self.plans), "reason"

    def update_info(self, info):
        self.feedback.append(str(info.get("env_feedback", "")))


def test_runner_logs_guard_block_and_replans_without_environment_step(tmp_path) -> None:
    skill = replace(interface_only_shared_skill(), temporal_rules=(_rule(),))
    monitor = TemporalRuleMonitor()
    planner = _SequencePlanner()
    output = tmp_path / "events.jsonl"
    runner = HabitatRolloutRunner(
        _GuardEnvironment(),
        planner,
        VistaSkillEngine(skill),
        JsonlArtifactWriter(output),
        temporal_monitor=monitor,
    )

    result = runner.run_episode()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    guards = [item for item in records if item["event_type"] == "temporal_guard"]
    assert result.task_success == 1.0
    assert result.environment_steps == 3
    assert result.temporal_guard_blocks == 1
    assert len(guards) == 1
    assert guards[0]["payload"]["rule_ids"] == ["recover_failed_pick"]
    assert any("Sequential Skill guard blocked" in item for item in planner.feedback)
