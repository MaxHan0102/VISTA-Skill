from __future__ import annotations

import importlib
import random
import re
import threading
from types import SimpleNamespace
from typing import Any, Sequence

from vista_skill.schemas import PredicateKey


_VALID_SET_LOCK = threading.Lock()


def create_habitat_env(
    eval_set: str,
    *,
    episode_ids: Sequence[str] | None = None,
    **kwargs: Any,
):
    """Construct stock EBHabEnv, allowing its shipped train_validation split.

    Stock EmbodiedBench ships the dataset but omits it from ValidEvalSets. The
    adapter extends the runtime allow-list only during construction and restores
    it immediately; no source file or environment behavior is modified.
    """
    env_module = importlib.import_module("embodiedbench.envs.eb_habitat.EBHabEnv")

    if eval_set in env_module.ValidEvalSets:
        env = env_module.EBHabEnv(eval_set=eval_set, **kwargs)
    else:
        if eval_set != "train_validation":
            raise ValueError(f"unsupported EB-Habitat split: {eval_set}")
        with _VALID_SET_LOCK:
            env_module.ValidEvalSets.append(eval_set)
            try:
                env = env_module.EBHabEnv(eval_set=eval_set, **kwargs)
            finally:
                env_module.ValidEvalSets.remove(eval_set)
    if episode_ids is not None:
        _select_ordered_episodes(env, episode_ids)
    return env


def _select_ordered_episodes(env: Any, episode_ids: Sequence[str]) -> None:
    requested = tuple(str(item) for item in episode_ids)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("episode selection must be non-empty and unique")
    indexed = {str(item.episode_id): item for item in env.dataset.episodes}
    missing = [item for item in requested if item not in indexed]
    if missing:
        raise ValueError(f"manifest episodes are absent from the dataset: {missing}")
    selected = [indexed[item] for item in requested]
    iterator = env.config.habitat.environment.iterator_options
    iterator.shuffle = False
    iterator.group_by_scene = False
    iterator.cycle = False
    env.dataset.episodes = selected
    env.env.episodes = selected
    env.number_of_episodes = len(selected)
    env.down_sample_ratio = 1.0
    env._current_episode_num = 0
    # The underlying habitat Env serves episodes from an internal iterator
    # (CustomEpisodeIterator) built at construction from the full dataset; simply
    # reassigning ``.episodes`` above is ignored, so reset() would still draw the
    # dataset's original first episode (the dataset is not episode_id-ordered).
    # Re-point that Env's dataset + iterator at the selected list so reset()
    # serves the requested episodes in the requested, non-cycling order.
    _repoint_habitat_episode_iterator(env.env, selected)


def _repoint_habitat_episode_iterator(root: Any, selected: Sequence[Any]) -> None:
    """Drive the deepest habitat ``Env``'s episode iterator from ``selected``."""
    habitat_env = _find_habitat_env(root)
    if habitat_env is None:
        return
    dataset = getattr(habitat_env, "_dataset", None)
    if dataset is not None:
        dataset.episodes = list(selected)
    habitat_env._episode_iterator = iter(selected)
    if hasattr(habitat_env, "episode_iterator"):
        habitat_env.episode_iterator = habitat_env._episode_iterator
    if selected:
        habitat_env._current_episode = selected[0]


def _find_habitat_env(obj: Any) -> Any:
    """Walk the GymHabitatEnv -> ... -> habitat.Env wrapper chain."""
    seen: set[int] = set()
    current = obj
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "_episode_iterator"):
            return current
        current = getattr(current, "env", None) or getattr(current, "_env", None)
    return None


def seed_habitat_env(env: Any, seed: int) -> None:
    """Apply the paired coordinate seed to stock Habitat and its RNGs."""
    seed = seed_process_rngs(seed)
    core = getattr(env, "env", None)
    if core is None or not hasattr(core, "seed"):
        raise RuntimeError("stock Habitat environment does not expose seed()")
    core.seed(seed)


def seed_process_rngs(seed: int) -> int:
    """Seed process RNGs before simulator or planner construction."""
    seed = int(seed)
    random.seed(seed)
    numpy = importlib.import_module("numpy")
    numpy.random.seed(seed)
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        # Torch is only required for real Habitat rollouts; in torch-less
        # environments (unit tests) there is no torch RNG to seed.
        return seed
    torch.manual_seed(seed)
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.manual_seed_all(seed)
    return seed


# --- EB-Navigation adapter --------------------------------------------------
#
# EBNavEnv exposes a Gym-style surface close to the runner's HabitatEnvironment
# protocol but with four gaps the adapter must bridge (see _NavEnvAdapter docstring).
# All motion primitives are parameterless, so skill_set tuples carry empty args.

NAV_SKILL_SET = [
    ("move_forward", []),
    ("move_backward", []),
    ("move_right", []),
    ("move_left", []),
    ("turn_right", []),
    ("turn_left", []),
    ("look_up", []),
    ("look_down", []),
]


class _NavEnvAdapter:
    """Bridge EBNavigationEnv to the runner's HabitatEnvironment protocol.

    Bridges four verified interface gaps without modifying EmbodiedBench:
      1. ``skill_set``: nav only sets ``language_skill_set`` -> synthesize tuples.
      2. ``step`` signature: nav requires ``(action, reasoning, i_flag)`` -> inject i_flag=1.
      3. ``info`` dict: nav omits ``task_progress``/``subgoal_reward``/``action_id`` -> inject.
         Also enriches ``env_feedback`` with the target distance so the nav evidence
         strategy can derive ``near(target)``.
      4. Episode identity: nav episodes are positional with no id -> expose
         ``current_episode()`` returning ``nav_<index>`` aligned with _official_episode_ids.
    """

    def __init__(self, env: Any) -> None:
        self._env = env
        self.skill_set = list(NAV_SKILL_SET)

    def __getattr__(self, name: str) -> Any:
        # Delegate protocol fields (number_of_episodes, _current_step,
        # language_skill_set, episode_language_instruction, episode_data, ...).
        return getattr(self._env, name)

    def current_episode(self, all_info: bool = False) -> SimpleNamespace:
        positional = max(0, self._env._current_episode_num - 1)  # 1-based after reset
        selected = getattr(self._env, "selected_indexes", []) or []
        original = selected[positional] if len(selected) else positional
        episode_id = f"nav_{original}"
        data = getattr(self._env, "episode_data", None) or {}
        scene = str(data.get("scene", "")) if isinstance(data, dict) else ""
        return SimpleNamespace(
            episode_id=episode_id, instruct_id=episode_id, scene_id=scene
        )

    def reset(self, **kwargs: Any) -> Any:
        return self._env.reset(**kwargs)

    def step(self, action: int, reasoning: str = "", **kwargs: Any) -> tuple:
        obs, reward, done, info = self._env.step(action, reasoning, 1)  # i_flag=1
        info.setdefault("task_progress", 0.0)
        info.setdefault("subgoal_reward", 0.0)
        info.setdefault("action_id", action)
        distance = info.get("distance")
        if distance is not None:
            try:
                distance = float(distance)
            except (TypeError, ValueError):
                distance = None
        if distance is not None:
            base = str(info.get("env_feedback", "") or "")
            info["env_feedback"] = f"{base} Target distance: {distance:.3f}m.".strip()
        return obs, reward, done, info

    def save_image(self, observation: Any = None) -> str:
        # Default-mode construction guarantees a single path string is returned.
        return self._env.save_image()

    def close(self) -> None:
        self._env.close()


def _nav_index_from_id(episode_id: str) -> int:
    """Convert a nav episode id ``nav_<i>`` to its positional dataset index."""
    marker = "nav_"
    if isinstance(episode_id, str) and episode_id.startswith(marker):
        return int(episode_id[len(marker):])
    return int(episode_id)


def create_nav_env(
    eval_set: str,
    *,
    episode_ids: Sequence[str] | None = None,
    resolution: int = 500,
    exp_name: str = "nav_eval",
) -> _NavEnvAdapter:
    """Construct a stock EBNavigationEnv wrapped for the VISTA runner.

    ``episode_ids`` (``nav_<i>``) select episodes positionally via the env's
    ``selected_indexes``; ``None``/empty uses the full split. Rendering is forced
    to default mode so ``save_image`` returns a single path string.
    """
    module = importlib.import_module("embodiedbench.envs.eb_navigation.EBNavEnv")
    if episode_ids:
        selected_indexes = [_nav_index_from_id(eid) for eid in episode_ids]
    else:
        selected_indexes = []
    env = module.EBNavigationEnv(
        eval_set=eval_set,
        exp_name=exp_name,
        resolution=resolution,
        multiview=False,
        boundingbox=False,
        multistep=False,
        selected_indexes=selected_indexes,
    )
    return _NavEnvAdapter(env)


def seed_nav_env(env: Any, seed: int) -> None:
    """Apply the paired coordinate seed to nav and its process RNGs.

    Best-effort: AI2-THOR determinism is weaker than Habitat's (see plan risks).
    Stock ``EBNavigationEnv.seed`` calls a typo'd ``random_initilize`` that does
    not exist on ai2thor 5.0.0 (whose real ``random_initialize`` was itself
    removed and always raises), so the process-RNG seeding below is the only
    effective part; the broken stock call is swallowed instead of editing
    EmbodiedBench in place.
    """
    seed = seed_process_rngs(seed)
    target = env._env if isinstance(env, _NavEnvAdapter) else env
    seed_method = getattr(target, "seed", None)
    if not callable(seed_method):
        raise RuntimeError("stock EBNavigationEnv does not expose seed()")
    try:
        seed_method(seed)
    except AttributeError as exc:
        if "random_initilize" not in str(exc):
            raise


def nav_goal_predicates(instruction: str, image: Any, actions: Any) -> tuple[PredicateKey, ...]:
    """Rule-based goal grounding for nav: derive ``near(target)`` without a model.

    The exact target token is not load-bearing for the VTCA loop -- nav reports a
    single per-episode target distance, so the evidence strategy emits
    ``near(<this target>)`` from that distance regardless of the string. We still
    parse the object word from the instruction for readability.
    """
    text = instruction or ""
    match = re.search(r"navigate to (?:the |a |an )?([A-Za-z]+)", text, re.IGNORECASE)
    target = match.group(1).lower() if match else "target"
    target = re.sub(r"[^a-z0-9_]+", "_", target).strip("_") or "target"
    return (PredicateKey("near", (target,)),)
