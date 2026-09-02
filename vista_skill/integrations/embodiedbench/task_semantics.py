from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Mapping


_PREDICATE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_HABITAT_ACTIONS_BY_PREDICATE = {
    "robot_at": ("nav",),
    "holding": ("pick",),
    "on_top": ("pick", "place"),
    # A closed-fridge goal can either be a direct close task or the final state
    # of an open/place/close task. Including both articulation primitives is a
    # conservative scope; it never reads task outcomes.
    "closed_fridge": ("open", "close"),
}


def predicate_names(value: Any) -> tuple[str, ...]:
    """Extract predicate names from a nested symbolic goal/subgoal object."""

    names: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, str):
            match = _PREDICATE.match(node)
            if match:
                names.append(match.group(1).lower())
            return
        if isinstance(node, Mapping):
            for child in node.values():
                visit(child)
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(value)
    return tuple(dict.fromkeys(names))


def habitat_episode_semantic_tags(episode: Mapping[str, Any]) -> tuple[str, ...]:
    """Precompute outcome-independent action/predicate tags for gate selection.

    This adapter translates Habitat's symbolic goal surface into the primitive
    action vocabulary consumed by the environment-neutral candidate gate. It
    uses no reward, success, task progress, trajectory, or model output.
    """

    names = set(predicate_names(episode.get("goal_preds", {})))
    names.update(predicate_names(episode.get("subgoals", ())))
    tags = {f"predicate:{name}" for name in names}
    for name in names:
        tags.update(
            f"action:{action_type}"
            for action_type in _HABITAT_ACTIONS_BY_PREDICATE.get(name, ())
        )
    goal = episode.get("goal_preds", {})
    inputs = goal.get("inputs", ()) if isinstance(goal, Mapping) else ()
    if isinstance(inputs, (list, tuple)) and len(inputs) >= 2:
        tags.add("structure:multi_entity_goal")
    return tuple(sorted(tags))


def load_habitat_task_semantic_tags(
    dataset: str | Path,
) -> dict[str, tuple[str, ...]]:
    """Load trusted local EB-HAB development data and index semantic tags."""

    with Path(dataset).open("rb") as handle:
        payload = pickle.load(handle)
    episodes = payload.get("all_eps") if isinstance(payload, Mapping) else None
    if not isinstance(episodes, list):
        raise ValueError("Habitat semantic tagging requires an all_eps list")
    indexed: dict[str, tuple[str, ...]] = {}
    for episode in episodes:
        if not isinstance(episode, Mapping) or "episode_id" not in episode:
            raise ValueError("Habitat semantic tagging found an invalid episode")
        episode_id = str(episode["episode_id"])
        if episode_id in indexed:
            raise ValueError(f"duplicate Habitat episode ID: {episode_id}")
        indexed[episode_id] = habitat_episode_semantic_tags(episode)
    return indexed
