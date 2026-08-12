from __future__ import annotations

import importlib
import random
import threading
from typing import Any, Sequence


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
    torch = importlib.import_module("torch")
    torch.manual_seed(seed)
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.manual_seed_all(seed)
    return seed
