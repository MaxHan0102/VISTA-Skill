from __future__ import annotations

import json

from vista_skill.artifacts import JsonlArtifactWriter
from vista_skill.integrations.embodiedbench.runner import HabitatRolloutRunner
from vista_skill.protocol import TaskCoordinate
from vista_skill.schemas import EvidenceSource, PredicateEvidence, PredicateKey, TruthValue
from vista_skill.pipeline import VistaSkillEngine
from vista_skill.skills import initialize_shared_skill


class FakeEnvironment:
    number_of_episodes = 1.0
    _current_episode_num = 0
    _current_step = 0
    _max_episode_steps = 4
    language_skill_set = ["navigate to stand", "pick up apple"]
    skill_set = [("nav", ["tvstand_1"]), ("pick", ["apple_1", "robot_0"])]
    episode_language_instruction = "Pick up the apple."

    class Episode:
        episode_id = "0"
        instruct_id = "fake-task"

    def current_episode(self, all_info=True):
        return self.Episode()

    def reset(self):
        self._current_episode_num += 1
        self._current_step = 0
        return {"frame": 0}

    def step(self, action, reasoning=""):
        self._current_step += 1
        done = self._current_step == 2
        feedback = (
            "Last action executed successfully."
            if action == 0
            else "Last action executed successfully and you are holding apple."
        )
        return (
            {"frame": self._current_step},
            1.0,
            done,
            {
                "env_step": self._current_step,
                "last_action_success": 1.0,
                "env_feedback": feedback,
                "task_success": float(done),
                "task_progress": self._current_step / 2,
                "subgoal_reward": 0.5,
            },
        )

    def save_image(self, observation):
        return f"step_{observation['frame']}.png"

    def close(self):
        pass


class FakePlanner:
    planner_steps = 0
    output_json_error = 0

    def reset(self):
        pass

    def act(self, image, instruction):
        self.planner_steps += 1
        return [0, 1], "{}"

    def update_info(self, info):
        pass


def test_multi_action_plan_writes_one_transition_per_primitive(tmp_path) -> None:
    output = tmp_path / "events.jsonl"
    runner = HabitatRolloutRunner(
        FakeEnvironment(),
        FakePlanner(),
        VistaSkillEngine(initialize_shared_skill()),
        JsonlArtifactWriter(output),
    )
    result = runner.run_episode()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    transitions = [item for item in records if item["event_type"] == "transition"]
    labels = [item for item in records if item["event_type"] == "evaluation_label"]
    assert len(transitions) == 2
    assert len(labels) == 2
    assert transitions[0]["payload"]["pre_image"] == "step_0.png"
    assert transitions[1]["payload"]["pre_image"] == "step_1.png"
    assert "task_progress" not in transitions[0]["payload"]
    assert result.task_success == 1.0


def test_runner_resets_belief_between_episodes(tmp_path) -> None:
    class TwoEpisodeEnvironment(FakeEnvironment):
        number_of_episodes = 2.0

        class Episode:
            episode_id = "1"
            instruct_id = "fake-task-2"

    environment = TwoEpisodeEnvironment()
    engine = VistaSkillEngine(initialize_shared_skill())
    runner = HabitatRolloutRunner(
        environment,
        FakePlanner(),
        engine,
        JsonlArtifactWriter(tmp_path / "events.jsonl"),
    )
    runner.run_episode()
    engine.ledger.merge(())
    runner.run_episode()
    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    transitions = [item for item in records if item["event_type"] == "transition"]
    assert transitions[2]["payload"]["pre_ledger"] == []


def test_runner_fails_closed_on_manifest_order_mismatch(tmp_path) -> None:
    runner = HabitatRolloutRunner(
        FakeEnvironment(),
        FakePlanner(),
        None,
        JsonlArtifactWriter(tmp_path / "events.jsonl"),
    )
    import pytest

    with pytest.raises(RuntimeError, match="episode order mismatch"):
        runner.run_episode(expected_episode_id="different")


def test_runner_supplies_online_attribution_context_and_cluster_scope(tmp_path) -> None:
    class SuccessfulContradictionProvider:
        def extract(self, request):
            key = PredicateKey("holding", (request.action.arguments[0],))
            return (
                PredicateEvidence(
                    key=key,
                    before=TruthValue.UNKNOWN,
                    after=TruthValue.FALSE,
                    confidence=0.99,
                    source=EvidenceSource.VISUAL_PAIR,
                    evidence_id=f"{request.episode_id}:vision",
                    timestamp=request.step_id,
                    view_id="post",
                    coverage=1.0,
                ),
            )

    from vista_skill.evidence import EvidenceExtractor, EvidenceExtractorConfig

    environment = FakeEnvironment()
    planner = FakePlanner()
    planner.act = lambda image, instruction: ([1], "{}")
    engine = VistaSkillEngine(
        initialize_shared_skill(),
        evidence_extractor=EvidenceExtractor(
            SuccessfulContradictionProvider(),
            EvidenceExtractorConfig(visual_action_types=("pick",)),
        ),
    )
    output = tmp_path / "events.jsonl"
    runner = HabitatRolloutRunner(
        environment,
        planner,
        engine,
        JsonlArtifactWriter(output),
        task_coordinates=(TaskCoordinate("0", "task-0", "multi-target", 0),),
    )
    runner.run_episode(expected_episode_id="0")
    transition = next(
        item["payload"]
        for item in map(json.loads, output.read_text().splitlines())
        if item["event_type"] == "transition"
    )
    context = transition["metadata"]["attribution_context"]
    assert context["executor_followed_skill"] is False
    assert context["task_pattern"] == "multi-target"
    assert context["object_context"] == "pick:apple"
    assert transition["attribution"]["subreason"] == "execution_lapse"
    assert not engine.clusterer.ready()


def test_runner_separates_cluster_keys_by_task_and_object_context() -> None:
    from vista_skill.integrations.embodiedbench.runner import _object_context
    from vista_skill.action_schema import parse_action_call

    apple = parse_action_call(0, ("pick_apple", ["apple_1", "robot_0"]))
    mug = parse_action_call(1, ("pick_mug", ["mug_2", "robot_0"]))
    assert _object_context(apple, ()) == "pick:apple"
    assert _object_context(mug, ()) == "pick:mug"
