from __future__ import annotations

import json
from types import SimpleNamespace

from vista_skill.artifacts import JsonlArtifactWriter
from vista_skill.integrations.embodiedbench.runner import HabitatRolloutRunner
from vista_skill.integrations.embodiedbench.state_oracle import (
    HabitatStateOracle,
    _entity_matches,
    mapped_coverage,
    oracle_query_keys,
)
from vista_skill.pipeline import VistaSkillEngine
from vista_skill.schemas import ActionCall, PredicateKey, TruthValue
from vista_skill.skills import initialize_shared_skill


class _FakeEnvironment:
    number_of_episodes = 1.0
    _current_episode_num = 0
    _current_step = 0
    _max_episode_steps = 4
    language_skill_set = ["navigate to stand", "pick up apple"]
    skill_set = [("nav", ["table_01"]), ("pick", ["apple", "robot_0"])]
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


class _FakePlanner:
    planner_steps = 0
    output_json_error = 0

    def reset(self):
        pass

    def act(self, image, instruction):
        self.planner_steps += 1
        return [0, 1], "{}"

    def update_info(self, info):
        pass


class _Predicate:
    def __init__(self, compact_str: str, truth: bool, arg_types=()) -> None:
        self.compact_str = compact_str
        self.truth = truth
        self._arg_values = tuple(
            SimpleNamespace(expr_type=SimpleNamespace(name=name)) for name in arg_types
        )

    def is_true(self, _sim_info):
        return self.truth


class _Pddl:
    sim_info = object()

    def __init__(self, predicates):
        self.predicates = predicates

    def get_possible_predicates(self):
        return self.predicates

    def get_true_predicates(self):
        return [item for item in self.predicates if item.truth]


class _Task:
    def __init__(self):
        self.pddl_problem = _Pddl(
            [
                _Predicate("holding(ball_0)", True),
                _Predicate("holding(apple_0)", False),
                _Predicate("not_holding()", False),
                _Predicate("on_top(ball_0,receptacle_aabb_sofa)", False),
                _Predicate("robot_at(receptacle_aabb_table_01)", True),
                _Predicate("opened_fridge(fridge_push_point)", True),
            ]
        )

    def is_goal_satisfied(self):
        return False


def test_oracle_maps_vista_categories_to_pddl_predicates() -> None:
    env = SimpleNamespace(env=SimpleNamespace(_env=SimpleNamespace(task=_Task())))
    keys = (
        PredicateKey("holding", ("ball",)),
        PredicateKey("not_holding"),
        PredicateKey("at", ("ball", "receptacle_aabb_sofa")),
        PredicateKey("near", ("receptacle_aabb_table_01",)),
        PredicateKey("open", ("fridge_push_point",)),
        PredicateKey("task_complete"),
    )
    observed = HabitatStateOracle().observe(env, keys)
    assert [item.value for item in observed] == [
        TruthValue.TRUE,
        TruthValue.FALSE,
        TruthValue.FALSE,
        TruthValue.TRUE,
        TruthValue.TRUE,
        TruthValue.FALSE,
    ]
    assert mapped_coverage(observed) == 1.0


def test_oracle_query_keys_use_only_evidence_side_inputs() -> None:
    action = ActionCall(0, "pick", ("apple",), "pick apple")
    keys = oracle_query_keys(action, (), (PredicateKey("at", ("apple", "table")),))
    assert PredicateKey("holding", ("apple",)) in keys
    assert PredicateKey("not_holding") in keys
    assert PredicateKey("task_complete") in keys


def test_oracle_entity_match_preserves_categories_and_instances() -> None:
    assert _entity_matches("ball", "ball_0")
    assert _entity_matches("receptacle_aabb_sofa", "receptacle_aabb_sofa")
    assert not _entity_matches("apple", "ball_0")


def test_oracle_can_match_public_category_through_pddl_entity_type() -> None:
    task = _Task()
    task.pddl_problem.predicates.append(
        _Predicate(
            "on_top(056_tennis_ball_:0000,receptacle_aabb_sofa)",
            True,
            ("ball", "place_receptacle"),
        )
    )
    env = SimpleNamespace(task=task)
    observed = HabitatStateOracle().observe(
        env, (PredicateKey("at", ("ball", "receptacle_aabb_sofa")),)
    )
    assert observed[0].value is TruthValue.TRUE


def test_oracle_uses_resolvable_domain_as_closed_world_for_missing_permutation() -> None:
    class DomainOnlyPddl:
        sim_info = object()
        predicates = {"on_top": object()}
        all_entities = {
            "ball": SimpleNamespace(
                name="056_tennis_ball_:0000",
                expr_type=SimpleNamespace(name="ball"),
            ),
            "sofa": SimpleNamespace(
                name="receptacle_aabb_sofa",
                expr_type=SimpleNamespace(name="place_receptacle"),
            ),
        }

        def get_possible_predicates(self):
            return []

        def get_true_predicates(self):
            return []

    task = _Task()
    task.pddl_problem = DomainOnlyPddl()
    observed = HabitatStateOracle().observe(
        SimpleNamespace(task=task),
        (PredicateKey("at", ("ball", "receptacle_aabb_sofa")),),
    )
    assert observed[0].value is TruthValue.FALSE
    assert observed[0].mapped_predicates[0].startswith("domain_closed_world:")


def test_oracle_resets_pddl_truth_cache_before_observing() -> None:
    calls = []

    class SimInfo:
        def reset_pred_truth_cache(self):
            calls.append("reset")

    class CachedPddl(_Pddl):
        sim_info = SimInfo()

        def get_true_predicates(self):
            assert calls == ["reset"]
            return super().get_true_predicates()

    task = _Task()
    task.pddl_problem = CachedPddl([_Predicate("not_holding()", True)])
    observed = HabitatStateOracle().observe(
        SimpleNamespace(task=task), (PredicateKey("not_holding"),)
    )
    assert observed[0].value is TruthValue.TRUE
    assert calls == ["reset"]


def test_runner_writes_oracle_as_separate_label_without_transition_leakage(tmp_path) -> None:
    class OracleEnvironment(_FakeEnvironment):
        def __init__(self):
            self.task = _Task()

    environment = OracleEnvironment()
    planner = _FakePlanner()
    output = tmp_path / "events.jsonl"
    runner = HabitatRolloutRunner(
        environment,
        planner,
        VistaSkillEngine(initialize_shared_skill()),
        JsonlArtifactWriter(output),
        state_oracle=HabitatStateOracle(),
    )
    runner.run_episode()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    labels = [item for item in records if item["event_type"] == "state_oracle_label"]
    transitions = [item for item in records if item["event_type"] == "transition"]
    assert len(labels) == len(transitions) == 2
    assert "state_oracle" not in json.dumps(transitions)
    assert labels[0]["payload"]["source"] == "evaluation_only_habitat_pddl_state"
