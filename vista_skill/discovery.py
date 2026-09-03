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
    TemporalSkillRule,
    TruthValue,
)


_DISCOVERABLE_ACTIONS = frozenset({"nav", "pick", "place", "open", "close"})


@dataclass(frozen=True)
class GeneralizedDiscovery:
    action_type: str
    predicate: str
    after: TruthValue
    field: SkillField = SkillField.EFFECT

    @property
    def signature(self) -> str:
        return f"{self.action_type}|{self.predicate}|{self.after.value}"

    def prediction_rule(self) -> SkillPredictionRule:
        rule_identity = f"{self.field.value}|{self.signature}"
        suffix = hashlib.sha256(rule_identity.encode("utf-8")).hexdigest()[:10]
        return SkillPredictionRule(
            rule_id=f"discovered_{self.action_type}_{suffix}",
            field=self.field,
            action_type=self.action_type,
            predicate=self.predicate,
            before=self.after if self.field is SkillField.CONSTRAINT else None,
            after=self.after,
        )

    def temporal_rule(self) -> TemporalSkillRule | None:
        """Compile supported recovery semantics for discoveries that imply order."""
        if (
            self.field is not SkillField.CONSTRAINT
            or self.action_type != "pick"
            or self.predicate != "near({arg0})"
            or self.after is not TruthValue.FALSE
        ):
            return None
        suffix = hashlib.sha256(
            f"temporal|{self.field.value}|{self.signature}".encode("utf-8")
        ).hexdigest()[:10]
        return TemporalSkillRule(
            rule_id=f"recover_pick_{suffix}",
            field=self.field,
            trigger_action_type="pick",
            trigger_success=False,
            trigger_predicate="near({arg0})",
            trigger_value=TruthValue.FALSE,
            blocked_action_type="pick",
            recovery_action_types=("nav",),
            argument_index=0,
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
        constraint_known = {
            ("pick", "near({arg0})", TruthValue.FALSE): (
                "Before picking, require evidence that the selected object is near; "
                "after a not-near failure, do not repeat the same pick from an "
                "unchanged state: navigate successfully and re-evaluate before retrying."
            ),
            ("pick", "not_holding", TruthValue.FALSE): (
                "Before picking, require evidence that the gripper is free."
            ),
            ("place", "not_holding", TruthValue.TRUE): (
                "Before placing, require evidence that an object is held."
            ),
            ("open", "near({arg0})", TruthValue.FALSE): (
                "Before opening, require evidence that the selected object is near."
            ),
            ("close", "near({arg0})", TruthValue.FALSE): (
                "Before closing, require evidence that the selected object is near."
            ),
        }
        if self.field is SkillField.CONSTRAINT:
            statement = constraint_known.get(
                (self.action_type, self.predicate, self.after)
            )
            if statement is not None:
                return statement
            return (
                f"Avoid {self.action_type} when {self.predicate}="
                f"{self.after.value}; first satisfy its missing precondition."
            )
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
    field: SkillField = SkillField.EFFECT,
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
        field=field,
    )
