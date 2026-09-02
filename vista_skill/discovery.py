from __future__ import annotations

import hashlib
from dataclasses import dataclass

from vista_skill.schemas import (
    ActionCall,
    Mismatch,
    MismatchKind,
    PredicateState,
    SkillField,
    SkillPredictionRule,
    TruthValue,
)


_DISCOVERABLE_ACTIONS = frozenset({"nav", "pick", "place", "open", "close"})


@dataclass(frozen=True)
class GeneralizedDiscovery:
    action_type: str
    predicate: str
    after: TruthValue

    @property
    def signature(self) -> str:
        return f"{self.action_type}|{self.predicate}|{self.after.value}"

    def prediction_rule(self) -> SkillPredictionRule:
        suffix = hashlib.sha256(self.signature.encode("utf-8")).hexdigest()[:10]
        return SkillPredictionRule(
            rule_id=f"discovered_{self.action_type}_{suffix}",
            field=SkillField.EFFECT,
            action_type=self.action_type,
            predicate=self.predicate,
            before=None,
            after=self.after,
        )

    def executor_statement(self) -> str:
        """Render a concise, instance-free instruction from the grounded rule."""
        known = {
            ("nav", "near({arg0})", TruthValue.TRUE): (
                "After a successful navigation, verify that the selected target is "
                "near before continuing."
            ),
            ("pick", "holding({arg0})", TruthValue.TRUE): (
                "After a successful pick, verify that the selected object is held "
                "before continuing."
            ),
            ("pick", "not_holding", TruthValue.FALSE): (
                "After a successful pick, treat the gripper as occupied and verify "
                "holding before continuing."
            ),
            ("place", "at({held},{arg0})", TruthValue.TRUE): (
                "After a successful place, verify that the previously held object is "
                "at the selected receptacle."
            ),
            ("place", "holding({held})", TruthValue.FALSE): (
                "After a successful place, verify that the placed object is no longer held."
            ),
            ("place", "not_holding", TruthValue.TRUE): (
                "After a successful place, verify that the gripper is free."
            ),
            ("open", "open({arg0})", TruthValue.TRUE): (
                "After a successful open action, verify that the selected object is open."
            ),
            ("close", "open({arg0})", TruthValue.FALSE): (
                "After a successful close action, verify that the selected object is closed."
            ),
        }
        statement = known.get((self.action_type, self.predicate, self.after))
        if statement is not None:
            return statement
        return (
            f"After a successful {self.action_type} action, verify the generalized "
            f"effect {self.predicate}={self.after.value} before continuing."
        )


def generalize_supported_transition(
    action: ActionCall | None,
    mismatch: Mismatch,
    pre_ledger: tuple[PredicateState, ...] = (),
) -> GeneralizedDiscovery | None:
    """Turn one grounded observed change into an instance-free rule.

    Every predicate argument must bind to an action argument or the object held
    before the action. This fail-closed rule prevents a single episode's object
    identifier from leaking into persistent memory.
    """
    if (
        action is None
        or action.action_type not in _DISCOVERABLE_ACTIONS
        or mismatch.kind is not MismatchKind.SUPPORTED_UNEXPECTED
        or mismatch.evidence is None
        or mismatch.evidence.after is TruthValue.UNKNOWN
        or mismatch.key.name == "task_complete"
    ):
        return None

    bindings: dict[str, str] = {
        value: f"{{arg{index}}}" for index, value in enumerate(action.arguments)
    }
    held = next(
        (
            state.key.arguments[0]
            for state in pre_ledger
            if state.key.name == "holding"
            and state.key.arguments
            and state.value is TruthValue.TRUE
        ),
        None,
    )
    if held is not None and held not in bindings:
        bindings[held] = "{held}"

    rendered_arguments = []
    for argument in mismatch.key.arguments:
        placeholder = bindings.get(argument)
        if placeholder is None:
            return None
        rendered_arguments.append(placeholder)
    predicate = mismatch.key.name
    if rendered_arguments:
        predicate += f"({','.join(rendered_arguments)})"
    return GeneralizedDiscovery(
        action_type=action.action_type,
        predicate=predicate,
        after=mismatch.evidence.after,
    )
