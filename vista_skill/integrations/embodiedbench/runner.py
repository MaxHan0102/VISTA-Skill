from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, replace
import re
from typing import Any, Callable, Mapping, Protocol, Sequence

from vista_skill.action_schema import parse_action_call
from vista_skill.artifacts import JsonlArtifactWriter
from vista_skill.pipeline import VistaSkillEngine
from vista_skill.protocol import TaskCoordinate
from vista_skill.schemas import (
    ActionCall,
    AttributionContext,
    PredicateKey,
    TruthValue,
)


class HabitatEnvironment(Protocol):
    number_of_episodes: float
    _current_episode_num: int
    _current_step: int
    _max_episode_steps: int
    language_skill_set: list[str]
    skill_set: list[tuple[str, list[str]]]
    episode_language_instruction: str

    def reset(self) -> Any: ...
    def step(self, action: int, reasoning: str = "") -> tuple[Any, float, bool, dict[str, Any]]: ...
    def save_image(self, observation: Any) -> str: ...
    def close(self) -> None: ...


class HabitatPlanner(Protocol):
    planner_steps: int
    output_json_error: int

    def reset(self) -> None: ...
    def act(self, observation: str, instruction: str) -> tuple[Any, str]: ...
    def update_info(self, info: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True)
class RunnerConfig:
    max_planner_retries: int = 2
    stop_on_planner_error: bool = True


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    instruction: str
    task_success: float
    task_progress: float
    reward_mean: float
    environment_steps: int
    planner_steps: int
    invalid_actions: int
    planner_output_errors: int
    elapsed_seconds: float


class HabitatRolloutRunner:
    """External evaluator with one VTCA hook per primitive environment action."""

    def __init__(
        self,
        env: HabitatEnvironment,
        planner: HabitatPlanner,
        engine: VistaSkillEngine | None,
        writer: JsonlArtifactWriter,
        *,
        task_id_provider: Callable[[HabitatEnvironment], str] | None = None,
        goal_predicate_provider: Callable[
            [str, str, list[tuple[str, list[str]]]], tuple[PredicateKey, ...]
        ]
        | None = None,
        config: RunnerConfig | None = None,
        expected_episode_ids: tuple[str, ...] | None = None,
        task_coordinates: Sequence[TaskCoordinate] = (),
    ) -> None:
        self.env = env
        self.planner = planner
        self.engine = engine
        self.writer = writer
        self.task_id_provider = task_id_provider or _default_task_id
        self.goal_predicate_provider = goal_predicate_provider or (
            lambda instruction, image, actions: ()
        )
        self.config = config or RunnerConfig()
        self.expected_episode_ids = expected_episode_ids
        self.task_coordinates = {
            item.episode_id: item for item in task_coordinates
        }

    def run(self, *, max_episodes: int | None = None) -> tuple[EpisodeResult, ...]:
        available = int(self.env.number_of_episodes) - self.env._current_episode_num
        if self.expected_episode_ids is not None:
            available = min(available, len(self.expected_episode_ids))
        count = available if max_episodes is None else min(available, max_episodes)
        results = []
        for index in range(max(0, count)):
            expected = None if self.expected_episode_ids is None else self.expected_episode_ids[index]
            results.append(self.run_episode(expected_episode_id=expected))
        return tuple(results)

    def run_episode(self, *, expected_episode_id: str | None = None) -> EpisodeResult:
        started = time.monotonic()
        observation = self.env.reset()
        pre_image = self.env.save_image(observation)
        instruction = self.env.episode_language_instruction
        episode_id = _current_episode_id(self.env)
        if expected_episode_id is not None and episode_id != expected_episode_id:
            raise RuntimeError(
                f"episode order mismatch: expected {expected_episode_id}, got {episode_id}"
            )
        coordinate = self.task_coordinates.get(episode_id)
        task_id = (
            coordinate.task_id if coordinate is not None else self.task_id_provider(self.env)
        )
        task_pattern = (
            coordinate.subgroup
            if coordinate is not None
            else _default_task_pattern(self.env)
        )
        goal_predicates = self.goal_predicate_provider(
            instruction, pre_image, self.env.skill_set
        )
        self.planner.reset()
        if self.engine is not None:
            self.engine.start_episode()
        rewards: list[float] = []
        invalid_actions = 0
        done = False
        last_info: dict[str, Any] = {
            "task_success": 0.0,
            "task_progress": 0.0,
            "env_step": 0,
        }

        while not done:
            plan, reasoning = self._act_with_retry(pre_image, instruction)
            if plan == -1 or plan == -2:
                self.writer.append(
                    "planner_output",
                    {"episode_id": episode_id, "code": plan, "raw_response": reasoning},
                )
                if self.config.stop_on_planner_error or plan == -2:
                    break
                continue
            actions = plan if isinstance(plan, list) else [plan]
            remaining = self.env._max_episode_steps - self.env._current_step
            for action_id in actions[:remaining]:
                raw_action = self.env.skill_set[action_id]
                action_text = self.env.language_skill_set[action_id]
                action_call = parse_action_call(action_id, raw_action, action_text)
                prepared = None
                if self.engine is not None:
                    attribution_context, context_audit = _online_attribution_context(
                        self.engine,
                        action_call,
                        instruction=instruction,
                        task_pattern=task_pattern,
                        goal_predicates=goal_predicates,
                    )
                    context_audit["pre_image_sha256"] = _file_sha256(pre_image)
                    prepared = self.engine.prepare(
                        episode_id=episode_id,
                        task_id=task_id,
                        step_id=self.env._current_step + 1,
                        instruction=instruction,
                        action=action_call,
                        pre_image=pre_image,
                        goal_predicates=goal_predicates,
                        attribution_context=attribution_context,
                        metadata={"attribution_context": context_audit},
                    )
                observation, reward, done, info = self.env.step(action_id, reasoning=reasoning)
                post_image = self.env.save_image(observation)
                self.planner.update_info(info)
                rewards.append(float(reward))
                invalid_actions += int(info.get("last_action_success", 0) == 0)
                last_info = info

                if self.engine is not None and prepared is not None:
                    metadata = dict(prepared.metadata)
                    context_audit = dict(metadata.get("attribution_context", {}))
                    context_audit["post_image_sha256"] = _file_sha256(post_image)
                    metadata["attribution_context"] = context_audit
                    prepared = replace(prepared, metadata=metadata)
                    event = self.engine.process_prepared(
                        prepared,
                        post_image=post_image,
                        feedback=str(info.get("env_feedback", "")),
                        last_action_success=bool(info.get("last_action_success", 0)),
                    )
                    self.writer.append("transition", event)
                self.writer.append(
                    "evaluation_label",
                    {
                        "episode_id": episode_id,
                        "step_id": info.get("env_step"),
                        "task_success": info.get("task_success"),
                        "task_progress": info.get("task_progress"),
                        "subgoal_reward": info.get("subgoal_reward"),
                    },
                )
                pre_image = post_image
                if done or info.get("last_action_success", 0) == 0:
                    break

        result = EpisodeResult(
            episode_id=episode_id,
            instruction=instruction,
            task_success=float(last_info.get("task_success", 0.0)),
            task_progress=float(last_info.get("task_progress", 0.0)),
            reward_mean=sum(rewards) / len(rewards) if rewards else 0.0,
            environment_steps=int(last_info.get("env_step", self.env._current_step)),
            planner_steps=self.planner.planner_steps,
            invalid_actions=invalid_actions,
            planner_output_errors=self.planner.output_json_error,
            elapsed_seconds=time.monotonic() - started,
        )
        self.writer.append("episode_result", result)
        return result

    def _act_with_retry(self, image: str, instruction: str) -> tuple[Any, str]:
        last_error: Exception | None = None
        for _ in range(self.config.max_planner_retries + 1):
            try:
                return self.planner.act(image, instruction)
            except Exception as error:  # model/network boundary
                last_error = error
        raise RuntimeError("planner retry budget exhausted") from last_error


def _default_task_id(env: HabitatEnvironment) -> str:
    current_episode = getattr(env, "current_episode", None)
    if callable(current_episode):
        try:
            active = current_episode(all_info=True)
        except TypeError:
            active = current_episode()
        for name in ("instruct_id", "episode_id", "scene_id"):
            value = getattr(active, name, None)
            if value is not None:
                return str(value)
            if isinstance(active, Mapping) and name in active:
                return str(active[name])
    episode_data = getattr(env, "episode_data", None)
    for name in ("instruct_id", "episode_id", "scene_id"):
        value = getattr(episode_data, name, None)
        if value is not None:
            return str(value)
        if isinstance(episode_data, Mapping) and name in episode_data:
            return str(episode_data[name])
    return f"episode_{env._current_episode_num}"


def _default_task_pattern(env: HabitatEnvironment) -> str:
    current_episode = getattr(env, "current_episode", None)
    if callable(current_episode):
        try:
            active = current_episode(all_info=True)
        except TypeError:
            active = current_episode()
        value = getattr(active, "instruct_id", None)
        if value is None and isinstance(active, Mapping):
            value = active.get("instruct_id")
        if value is not None:
            return str(value)
    return _default_task_id(env)


def _online_attribution_context(
    engine: VistaSkillEngine,
    action: ActionCall,
    *,
    instruction: str,
    task_pattern: str,
    goal_predicates: tuple[PredicateKey, ...],
) -> tuple[AttributionContext, dict[str, Any]]:
    """Build attribution inputs only from the active Skill and online state."""
    checks = _necessary_precondition_checks(engine, action)
    followed = all(item["satisfied"] for item in checks) if checks else True
    object_context = _object_context(action, goal_predicates)
    context = AttributionContext(
        executor_followed_skill=followed,
        task_pattern=task_pattern,
        object_context=object_context,
        instruction=instruction,
        action_type=action.action_type,
    )
    return context, {
        "source": "manifest_or_episode_id+action+episode_local_ledger",
        "executor_followed_skill": followed,
        "task_pattern": task_pattern,
        "object_context": object_context,
        "necessary_preconditions": checks,
    }


def _necessary_precondition_checks(
    engine: VistaSkillEngine,
    action: ActionCall,
) -> list[dict[str, Any]]:
    required: list[tuple[PredicateKey, TruthValue]] = []
    if action.action_type == "pick":
        required.append((PredicateKey("not_holding"), TruthValue.TRUE))
    elif action.action_type == "place" and action.arguments:
        required.extend(
            (
                (PredicateKey("not_holding"), TruthValue.FALSE),
                (PredicateKey("near", (action.arguments[0],)), TruthValue.TRUE),
            )
        )
    elif action.action_type in {"open", "close"} and action.arguments:
        required.append((PredicateKey("near", (action.arguments[0],)), TruthValue.TRUE))
    checks = []
    for key, expected in required:
        observed = engine.ledger.value(key)
        checks.append(
            {
                "predicate": key.render(),
                "required": expected.value,
                "observed": observed.value,
                "satisfied": observed is expected,
            }
        )
    return checks


def _object_context(
    action: ActionCall,
    goal_predicates: tuple[PredicateKey, ...],
) -> str:
    entities = [_entity_category(item) for item in action.arguments]
    entities.extend(
        _entity_category(argument)
        for predicate in goal_predicates
        for argument in predicate.arguments
    )
    stable = tuple(dict.fromkeys(item for item in entities if item and item != "robot"))
    suffix = "+".join(stable) if stable else "none"
    return f"{action.action_type}:{suffix}"


def _entity_category(value: str) -> str:
    normalized = value.lower().strip()
    normalized = re.sub(r"_\d+$", "", normalized)
    return re.sub(r"[^a-z0-9_]+", "_", normalized).strip("_")


def _file_sha256(path: str) -> str | None:
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


def _current_episode_id(env: HabitatEnvironment) -> str:
    current_episode = getattr(env, "current_episode", None)
    if callable(current_episode):
        try:
            active = current_episode(all_info=True)
        except TypeError:
            active = current_episode()
        value = getattr(active, "episode_id", None)
        if value is None and isinstance(active, Mapping):
            value = active.get("episode_id")
        if value is not None:
            return str(value)
    return f"episode_{env._current_episode_num}"
