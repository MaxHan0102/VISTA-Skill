from __future__ import annotations

import pytest

from vista_skill.integrations.embodiedbench.environment import (
    NAV_SKILL_SET,
    _NavEnvAdapter,
    _nav_index_from_id,
    nav_goal_predicates,
    seed_nav_env,
)


class _FakeNavEnv:
    """Minimal stand-in for EBNavigationEnv (no AI2-THOR)."""

    def __init__(self) -> None:
        self.number_of_episodes = 5
        self._current_episode_num = 1  # 1-based after reset
        self._current_step = 0
        self._max_episode_steps = 20
        self.language_skill_set = ["Move forward by 0.25"] * 8
        self.episode_language_instruction = "navigate to the bread"
        self.episode_data = {"scene": "FloorPlan11"}
        self.selected_indexes: list[int] = []
        self.last_step = None

    def reset(self, **kwargs):
        return {"head_rgb": "frame"}

    def step(self, action, reasoning, i_flag):
        self.last_step = (action, reasoning, i_flag)
        info = {
            "distance": 0.5,
            "env_feedback": "ok",
            "task_success": 0.0,
            "last_action_success": 1,
            "env_step": 1,
        }
        return {"head_rgb": "frame"}, 0.0, False, info

    def save_image(self, *args, **kwargs):
        return "/tmp/nav_frame.png"

    def close(self):
        return None


def test_adapter_synthesizes_skill_set_tuples() -> None:
    env = _NavEnvAdapter(_FakeNavEnv())
    assert env.skill_set == NAV_SKILL_SET
    assert len(env.skill_set) == 8
    assert all(isinstance(e, tuple) and len(e) == 2 and e[1] == [] for e in env.skill_set)


def test_adapter_step_injects_i_flag_and_missing_keys() -> None:
    fake = _FakeNavEnv()
    env = _NavEnvAdapter(fake)
    _, _, _, info = env.step(0, reasoning="go")
    assert fake.last_step == (0, "go", 1)  # i_flag=1 injected
    assert "task_progress" in info and "subgoal_reward" in info and "action_id" in info


def test_adapter_step_enriches_feedback_with_distance() -> None:
    env = _NavEnvAdapter(_FakeNavEnv())
    _, _, _, info = env.step(0, reasoning="go")
    assert "Target distance: 0.500m." in info["env_feedback"]


def test_adapter_current_episode_returns_positional_id() -> None:
    env = _NavEnvAdapter(_FakeNavEnv())
    episode = env.current_episode()
    assert episode.episode_id == "nav_0"
    assert episode.instruct_id == "nav_0"
    assert episode.scene_id == "FloorPlan11"


def test_adapter_save_image_returns_str() -> None:
    env = _NavEnvAdapter(_FakeNavEnv())
    assert isinstance(env.save_image({"head_rgb": "x"}), str)


def test_adapter_delegates_protocol_attributes() -> None:
    env = _NavEnvAdapter(_FakeNavEnv())
    assert env.number_of_episodes == 5
    assert env._max_episode_steps == 20
    assert env.episode_language_instruction == "navigate to the bread"
    assert env.language_skill_set[0] == "Move forward by 0.25"


def test_nav_index_from_id_parses() -> None:
    assert _nav_index_from_id("nav_7") == 7
    assert _nav_index_from_id("3") == 3


def test_nav_goal_predicates_extracts_target() -> None:
    goals = nav_goal_predicates("navigate to the Bread in the room", None, None)
    assert goals and goals[0].name == "near"
    assert goals[0].arguments == ("bread",)


def test_nav_goal_predicates_falls_back_when_no_object() -> None:
    goals = nav_goal_predicates("explore the kitchen", None, None)
    assert goals and goals[0].arguments == ("target",)


class _TypoSeedNavEnv(_FakeNavEnv):
    """Stock EBNavigationEnv.seed() against ai2thor 5.0.0: typo'd call dies."""

    def seed(self, seed=None):
        raise AttributeError(
            "'Controller' object has no attribute 'random_initilize'"
        )


class _BrokenSeedNavEnv(_FakeNavEnv):
    def seed(self, seed=None):
        raise AttributeError("'Controller' object has no attribute 'startup'")


def test_seed_nav_env_swallows_stock_typo_seed() -> None:
    # Process RNGs must still be seeded; the dead stock call must not abort.
    seed_nav_env(_NavEnvAdapter(_TypoSeedNavEnv()), 7)


def test_seed_nav_env_reraises_unrelated_seed_failure() -> None:
    with pytest.raises(AttributeError):
        seed_nav_env(_NavEnvAdapter(_BrokenSeedNavEnv()), 7)


def test_seed_nav_env_requires_seed_method() -> None:
    with pytest.raises(RuntimeError):
        seed_nav_env(_FakeNavEnv(), 7)
