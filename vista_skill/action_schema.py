from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from vista_skill.belief import BeliefLedger
from vista_skill.schemas import (
    ActionCall,
    DeltaSource,
    ExpectedChange,
    PredicateKey,
    SkillField,
    SkillPredictionRule,
    SkillSpec,
    TerminationPolicy,
    TruthValue,
)


_ACTION_ALIASES = {
    "navigate": "nav",
    "navigate_to": "nav",
    "pick_up": "pick",
    "pickup": "pick",
    "open": "open",
    "close": "close",
}


def normalize_entity(value: str) -> str:
    """Normalize separators without stripping instance-identifying digits."""
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def normalize_action_type(value: str) -> str:
    value = normalize_entity(value)
    if value.startswith("pick_"):
        return "pick"
    if value.startswith("open_"):
        return "open"
    if value.startswith("close_"):
        return "close"
    return _ACTION_ALIASES.get(value, value)


def parse_action_call(
    action_id: int,
    raw_action: str | tuple[str, Iterable[str]],
    text: str | None = None,
) -> ActionCall:
    if isinstance(raw_action, tuple):
        name, raw_arguments = raw_action
        arguments = tuple(normalize_entity(item) for item in raw_arguments)
        normalized_name = normalize_entity(name)
        if normalized_name.startswith("pick_"):
            category = normalized_name.removeprefix("pick_")
            arguments = (category, *tuple(item for item in arguments if not item.startswith("robot_")))
        raw = f"{name}({','.join(raw_arguments)})"
    else:
        raw = raw_action.strip()
        if "(" in raw and raw.endswith(")"):
            name, argument_text = raw[:-1].split("(", 1)
            arguments = tuple(
                normalize_entity(item) for item in argument_text.split(",") if item.strip()
            )
        else:
            name = raw
            arguments = ()
    return ActionCall(
        action_id=action_id,
        action_type=normalize_action_type(name),
        arguments=arguments,
        text=text or raw,
        raw_action=raw,
    )


@dataclass(frozen=True)
class FixedActionSchema:
    schema_id: str = "eb_habitat_pddl_v1"

    def compile(
        self,
        action: ActionCall,
        ledger: BeliefLedger,
        skill: SkillSpec,
        goal_predicates: tuple[PredicateKey, ...] = (),
    ) -> tuple[ExpectedChange, ...]:
        changes = list(self._primitive_changes(action, ledger))
        changes.extend(self._skill_changes(action, ledger, skill.prediction_rules, skill))
        completion = self._termination_change(
            ledger, skill, action, changes, goal_predicates
        )
        if completion is not None:
            changes.append(completion)
        deduplicated: dict[tuple[PredicateKey, DeltaSource, str], ExpectedChange] = {}
        for change in changes:
            deduplicated[(change.key, change.source, change.source_id)] = change
        return tuple(deduplicated.values())

    @staticmethod
    def _termination_change(
        ledger: BeliefLedger,
        skill: SkillSpec,
        action: ActionCall,
        changes: list[ExpectedChange],
        goal_predicates: tuple[PredicateKey, ...],
    ) -> ExpectedChange | None:
        if not goal_predicates:
            return None
        post_values = {item.key: item.after for item in changes}
        satisfied = [
            post_values.get(goal, ledger.value(goal)) is TruthValue.TRUE
            for goal in goal_predicates
        ]
        if skill.termination_policy is TerminationPolicy.ALL_GOALS_EVIDENCE:
            after = all(satisfied)
        elif skill.termination_policy is TerminationPolicy.ANY_GOAL_EVIDENCE:
            after = any(satisfied)
        else:
            after = action.action_type != "noop"
        key = PredicateKey("task_complete")
        return ExpectedChange(
            key=key,
            before=ledger.value(key),
            after=TruthValue.TRUE if after else TruthValue.FALSE,
            source=DeltaSource.SKILL,
            source_id=f"{skill.skill_id}:v{skill.version}:termination_policy",
            skill_field=SkillField.TERMINATION,
        )

    def _primitive_changes(
        self,
        action: ActionCall,
        ledger: BeliefLedger,
    ) -> Iterable[ExpectedChange]:
        action_type = action.action_type
        args = action.arguments
        source_id = f"{self.schema_id}:{action_type}"

        if action_type == "nav" and args:
            yield self._change(ledger, PredicateKey("near", (args[0],)), True, source_id)
        elif action_type == "pick" and args:
            obj = args[0]
            yield self._change(ledger, PredicateKey("holding", (obj,)), True, source_id)
            yield self._change(ledger, PredicateKey("not_holding"), False, source_id)
        elif action_type == "place" and args:
            receptacle = args[0]
            held = [
                state.key.arguments[0]
                for state in ledger.snapshot()
                if state.key.name == "holding"
                and state.key.arguments
                and state.value is TruthValue.TRUE
            ]
            yield self._change(ledger, PredicateKey("not_holding"), True, source_id)
            for obj in held:
                yield self._change(ledger, PredicateKey("holding", (obj,)), False, source_id)
                yield self._change(
                    ledger,
                    PredicateKey("at", (obj, receptacle)),
                    True,
                    source_id,
                )
        elif action_type in {"open", "close"} and args:
            target = args[0]
            opened = action_type == "open"
            yield self._change(ledger, PredicateKey("open", (target,)), opened, source_id)

    def _skill_changes(
        self,
        action: ActionCall,
        ledger: BeliefLedger,
        rules: tuple[SkillPredictionRule, ...],
        skill: SkillSpec,
    ) -> Iterable[ExpectedChange]:
        bindings = {f"arg{index}": value for index, value in enumerate(action.arguments)}
        held = next(
            (
                state.key.arguments[0]
                for state in ledger.snapshot()
                if state.key.name == "holding"
                and state.key.arguments
                and state.value is TruthValue.TRUE
            ),
            "held_object",
        )
        bindings["held"] = held

        for rule in rules:
            normalized_rule_action = (
                "*" if rule.action_type == "*" else normalize_action_type(rule.action_type)
            )
            if normalized_rule_action not in {"*", action.action_type}:
                continue
            rendered = rule.predicate
            for key, value in bindings.items():
                rendered = rendered.replace("{" + key + "}", value)
            predicate = PredicateKey.parse(rendered)
            before = rule.before or ledger.value(predicate)
            yield ExpectedChange(
                key=predicate,
                before=before,
                after=rule.after,
                source=DeltaSource.SKILL,
                source_id=f"{skill.skill_id}:v{skill.version}:{rule.rule_id}",
                skill_field=rule.field,
            )

    @staticmethod
    def _change(
        ledger: BeliefLedger,
        key: PredicateKey,
        after: bool,
        source_id: str,
    ) -> ExpectedChange:
        return ExpectedChange(
            key=key,
            before=ledger.value(key),
            after=TruthValue.TRUE if after else TruthValue.FALSE,
            source=DeltaSource.ACTION_SCHEMA,
            source_id=source_id,
        )
