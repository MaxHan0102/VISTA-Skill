from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from vista_skill.schemas import (
    ActionCall,
    PredicateKey,
    PredicateState,
    TruthValue,
)


@dataclass(frozen=True)
class OraclePredicateObservation:
    key: PredicateKey
    value: TruthValue
    mapped_predicates: tuple[str, ...] = ()
    rationale: str = ""


@dataclass(frozen=True)
class StateOracleTransitionLabel:
    episode_id: str
    step_id: int
    query_keys: tuple[PredicateKey, ...]
    pre: tuple[OraclePredicateObservation, ...]
    post: tuple[OraclePredicateObservation, ...]
    source: str = "evaluation_only_habitat_pddl_state"


def oracle_query_keys(
    action: ActionCall,
    pre_ledger: Sequence[PredicateState],
    goal_predicates: Sequence[PredicateKey],
) -> tuple[PredicateKey, ...]:
    """Build evaluation queries without consulting Skill predictions.

    The set mirrors the evidence-side query vocabulary and adds the direct
    action-local facts emitted by the deterministic feedback parser.  It is
    safe to construct before the action because every input is already part of
    ``EvidenceRequest``.
    """

    keys = [*goal_predicates, *(item.key for item in pre_ledger)]
    if action.action_type == "pick" and action.arguments:
        keys.extend(
            (
                PredicateKey("holding", (action.arguments[0],)),
                PredicateKey("not_holding"),
            )
        )
    elif action.action_type == "place":
        keys.append(PredicateKey("not_holding"))
        for item in pre_ledger:
            if (
                item.key.name == "holding"
                and item.key.arguments
                and item.value is TruthValue.TRUE
            ):
                keys.append(item.key)
                if action.arguments:
                    keys.append(
                        PredicateKey("at", (item.key.arguments[0], action.arguments[0]))
                    )
    elif action.action_type in {"open", "close"} and action.arguments:
        keys.append(PredicateKey("open", (action.arguments[0],)))
    elif action.action_type == "nav" and action.arguments:
        keys.append(PredicateKey("near", (action.arguments[0],)))
    keys.append(PredicateKey("task_complete"))
    return tuple(dict.fromkeys(keys))


class HabitatStateOracle:
    """Read-only EB-Habitat PDDL state adapter used only for evaluation labels."""

    def observe(
        self,
        env: Any,
        keys: Sequence[PredicateKey],
    ) -> tuple[OraclePredicateObservation, ...]:
        task = _find_predicate_task(env)
        if task is None:
            return tuple(
                OraclePredicateObservation(
                    key,
                    TruthValue.UNKNOWN,
                    rationale="Habitat predicate task is unavailable",
                )
                for key in keys
            )
        pddl = getattr(task, "pddl_problem", None) or getattr(task, "pddl", None)
        possible = ()
        true_predicates = ()
        # EB-Habitat clears this cache at the beginning of ``task.step`` but
        # PDDL action precondition checks populate it again before applying
        # the action.  Without a post-action reset, an evaluation-only oracle
        # can therefore read pre-action truth values after the simulator has
        # changed (for example, ``not_holding()`` stays cached as true after a
        # successful pick).  Clearing only the predicate evaluation cache is
        # read-only with respect to simulator state and forces a coherent
        # snapshot for every oracle observation.
        sim_info = getattr(pddl, "sim_info", None) if pddl is not None else None
        reset_truth_cache = getattr(sim_info, "reset_pred_truth_cache", None)
        if callable(reset_truth_cache):
            reset_truth_cache()
        if pddl is not None and hasattr(pddl, "get_possible_predicates"):
            possible = tuple(pddl.get_possible_predicates())
        if pddl is not None and hasattr(pddl, "get_true_predicates"):
            true_predicates = tuple(pddl.get_true_predicates())
        return tuple(
            self._observe_key(task, possible, true_predicates, key) for key in keys
        )

    def _observe_key(
        self,
        task: Any,
        possible: Sequence[Any],
        true_predicates: Sequence[Any],
        key: PredicateKey,
    ) -> OraclePredicateObservation:
        if key.name == "task_complete":
            method = getattr(task, "is_goal_satisfied", None)
            if callable(method):
                try:
                    value = TruthValue.TRUE if bool(method()) else TruthValue.FALSE
                    return OraclePredicateObservation(
                        key,
                        value,
                        ("goal_expr",),
                        "evaluated from Habitat PDDL goal expression",
                    )
                except Exception as error:
                    return OraclePredicateObservation(
                        key,
                        TruthValue.UNKNOWN,
                        rationale=f"goal evaluation failed: {type(error).__name__}",
                    )
        true_matched = [
            predicate
            for predicate in true_predicates
            if _predicate_matches_key(_predicate_key(predicate), key, predicate)
        ]
        if true_matched:
            names = tuple(_predicate_compact_str(predicate) for predicate in true_matched)
            return OraclePredicateObservation(
                key,
                TruthValue.TRUE,
                names,
                "matched Habitat get_true_predicates state",
            )
        matched = [
            predicate
            for predicate in possible
            if _predicate_matches_key(_predicate_key(predicate), key, predicate)
        ]
        if not matched:
            aliases = {
                "holding": ("holding",),
                "not_holding": ("not_holding",),
                "at": ("at", "on_top", "in"),
                "near": ("robot_at", "robot_at_closest", "robot_at_obj"),
                "open": ("opened_cab", "opened_fridge"),
            }.get(key.name, (key.name,))
            candidates = tuple(
                _predicate_compact_str(predicate)
                for predicate in possible
                if (
                    (parsed := _predicate_key(predicate)) is not None
                    and parsed.name in aliases
                )
            )
            detail = ", ".join(candidates[:8])
            pddl = getattr(task, "pddl_problem", None) or getattr(task, "pddl", None)
            if _domain_query_resolvable(pddl, key, aliases):
                return OraclePredicateObservation(
                    key,
                    TruthValue.FALSE,
                    (f"domain_closed_world:{key.render()}",),
                    (
                        "predicate family and query entities resolve in Habitat PDDL "
                        "domain but no matching true predicate exists"
                    ),
                )
            return OraclePredicateObservation(
                key,
                TruthValue.UNKNOWN,
                rationale=(
                    "no compatible Habitat PDDL predicate"
                    + (f"; alias candidates: {detail}" if detail else "")
                ),
            )
        names = tuple(_predicate_compact_str(predicate) for predicate in matched)
        sim_info = getattr(
            getattr(task, "pddl_problem", None) or getattr(task, "pddl", None),
            "sim_info",
            None,
        )
        try:
            truths = [bool(predicate.is_true(sim_info)) for predicate in matched]
        except Exception as error:
            return OraclePredicateObservation(
                key,
                TruthValue.UNKNOWN,
                names,
                f"predicate evaluation failed: {type(error).__name__}",
            )
        return OraclePredicateObservation(
            key,
            TruthValue.TRUE if any(truths) else TruthValue.FALSE,
            names,
            "existential match over compatible Habitat PDDL entities",
        )


def _find_predicate_task(root: Any) -> Any | None:
    queue = [root]
    seen: set[int] = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if (
            hasattr(current, "is_goal_satisfied")
            and (hasattr(current, "pddl_problem") or hasattr(current, "pddl"))
        ):
            return current
        task = getattr(current, "task", None)
        if task is not None:
            queue.append(task)
        for name in ("env", "_env"):
            child = getattr(current, name, None)
            if child is not None:
                queue.append(child)
    return None


def _predicate_compact_str(predicate: Any) -> str:
    value = getattr(predicate, "compact_str", None)
    if callable(value):
        value = value()
    return str(value if value is not None else predicate).replace(" ", "")


def _predicate_key(predicate: Any) -> PredicateKey | None:
    rendered = _predicate_compact_str(predicate)
    # Habitat renders zero-arity predicates as ``not_holding()`` whereas the
    # shared VISTA syntax uses ``not_holding``.  PredicateKey.parse correctly
    # rejects an empty entity, so normalize only this explicit zero-arity form.
    if rendered.endswith("()"):
        rendered = rendered[:-2]
    try:
        return PredicateKey.parse(rendered)
    except ValueError:
        return None


def _predicate_matches_key(
    candidate: PredicateKey | None,
    query: PredicateKey,
    raw_predicate: Any | None = None,
) -> bool:
    if candidate is None:
        return False
    aliases: dict[str, tuple[str, ...]] = {
        "holding": ("holding",),
        "not_holding": ("not_holding",),
        "at": ("at", "on_top", "in"),
        "near": ("robot_at", "robot_at_closest", "robot_at_obj"),
        "open": ("opened_cab", "opened_fridge"),
    }
    allowed = aliases.get(query.name, (query.name,))
    if candidate.name not in allowed:
        return False
    candidate_args = candidate.arguments
    query_args = query.arguments
    raw_args = tuple(getattr(raw_predicate, "_arg_values", ()) or ())

    def matches(index: int, wanted: str, actual: str) -> bool:
        if _entity_matches(wanted, actual):
            return True
        if index >= len(raw_args):
            return False
        return _raw_entity_matches(wanted, raw_args[index])

    if query.name == "near":
        return bool(query_args and candidate_args) and matches(
            len(candidate_args) - 1, query_args[-1], candidate_args[-1]
        )
    if query.name == "open":
        return bool(query_args and candidate_args) and matches(
            0, query_args[0], candidate_args[0]
        )
    if len(candidate_args) != len(query_args):
        return False
    return all(
        matches(index, wanted, actual)
        for index, (wanted, actual) in enumerate(zip(query_args, candidate_args))
    )


def _domain_query_resolvable(
    pddl: Any,
    query: PredicateKey,
    aliases: Sequence[str],
) -> bool:
    if pddl is None:
        return False
    predicates = getattr(pddl, "predicates", {})
    predicate_names = set(predicates) if isinstance(predicates, dict) else {
        str(getattr(item, "_name", "")) for item in predicates
    }
    if not predicate_names.intersection(aliases):
        return False
    if not query.arguments:
        return True
    entities = getattr(pddl, "all_entities", {})
    values = tuple(entities.values()) if isinstance(entities, dict) else tuple(entities)
    return all(
        any(_raw_entity_matches(argument, entity) for entity in values)
        for argument in query.arguments
    )


def _raw_entity_matches(wanted: str, entity: Any) -> bool:
    name = getattr(entity, "name", None)
    if name is not None and _entity_matches(wanted, str(name)):
        return True
    entity_type = getattr(entity, "expr_type", None)
    type_name = getattr(entity_type, "name", entity_type)
    return type_name is not None and _entity_matches(wanted, str(type_name))


def _entity_matches(wanted: str, actual: str) -> bool:
    left = _entity_tokens(wanted)
    right = _entity_tokens(actual)
    if wanted.lower() == actual.lower():
        return True
    if not left or not right:
        return False
    # A goal-grounded category such as ``ball`` denotes any concrete ball_i.
    if len(left) == 1:
        return left[0] in right
    if len(right) == 1:
        return right[0] in left
    return left == right or all(token in right for token in left)


def _entity_tokens(value: str) -> tuple[str, ...]:
    normalized = value.lower().strip()
    normalized = normalized.replace("refrigerator", "fridge")
    normalized = normalized.replace("fridge_type", "fridge")
    normalized = re.sub(r"^receptacle_aabb_", "", normalized)
    tokens = []
    ignored = re.compile(
        r"^(?:\d+|top\d*|tbl\d*|frl|apartment|push|point|robot|entity)$"
    )
    for token in re.split(r"[^a-z0-9]+", normalized):
        if not token or ignored.match(token):
            continue
        if token.isdigit():
            continue
        tokens.append(token)
    return tuple(tokens)


def mapped_coverage(items: Iterable[OraclePredicateObservation]) -> float:
    observations = tuple(items)
    if not observations:
        return 0.0
    mapped = sum(item.value is not TruthValue.UNKNOWN for item in observations)
    return mapped / len(observations)
