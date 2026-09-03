from __future__ import annotations

from dataclasses import dataclass

from vista_skill.schemas import (
    ActionCall,
    PredicateKey,
    PredicateState,
    SkillSpec,
    TemporalSkillRule,
    TransitionEvent,
    TruthValue,
)


@dataclass(frozen=True)
class TemporalAdmission:
    allowed: bool
    rule_ids: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class ActiveTemporalObligation:
    rule: TemporalSkillRule
    target: str
    trigger_step: int


class TemporalRuleMonitor:
    """Deterministic episode-local monitor for compiled temporal Skill rules."""

    def __init__(self) -> None:
        self._rules: tuple[TemporalSkillRule, ...] = ()
        self._active: dict[tuple[str, str], ActiveTemporalObligation] = {}

    def start_episode(self, skill: SkillSpec) -> None:
        self._rules = tuple(skill.temporal_rules)
        self._active.clear()

    @property
    def active(self) -> tuple[ActiveTemporalObligation, ...]:
        return tuple(
            self._active[key]
            for key in sorted(self._active)
        )

    def admit(self, action: ActionCall) -> TemporalAdmission:
        blocked = []
        for obligation in self.active:
            rule = obligation.rule
            if action.action_type != rule.blocked_action_type:
                continue
            argument = _action_argument(action, rule.argument_index)
            if argument == obligation.target:
                blocked.append(rule.rule_id)
        if not blocked:
            return TemporalAdmission(True)
        target = _action_argument(action, 0) or "the same target"
        return TemporalAdmission(
            False,
            tuple(blocked),
            (
                f"Sequential Skill guard blocked repeating {action.action_type} on "
                f"{target} from an unchanged state. Execute a permitted recovery "
                "action and re-evaluate visual/environment evidence first."
            ),
        )

    def observe(self, event: TransitionEvent) -> None:
        # A successful recovery action changes the state before any retry. A
        # contrary observation of the triggering predicate is an equivalent
        # evidence-based release condition.
        for key, obligation in tuple(self._active.items()):
            rule = obligation.rule
            recovered = bool(
                event.last_action_success
                and event.action.action_type in rule.recovery_action_types
            )
            predicate = _bind_predicate(
                rule.trigger_predicate,
                event.action,
                event.pre_ledger,
                target_override=obligation.target,
            )
            evidence_released = predicate is not None and any(
                item.key == predicate
                and item.after not in {TruthValue.UNKNOWN, rule.trigger_value}
                for item in event.evidence_delta
            )
            if recovered or evidence_released:
                self._active.pop(key, None)

        for rule in self._rules:
            if (
                event.action.action_type != rule.trigger_action_type
                or event.last_action_success is None
                or bool(event.last_action_success) is not rule.trigger_success
            ):
                continue
            predicate = _bind_predicate(
                rule.trigger_predicate,
                event.action,
                event.pre_ledger,
            )
            if predicate is None or not any(
                item.key == predicate and item.after is rule.trigger_value
                for item in event.evidence_delta
            ):
                continue
            target = _action_argument(event.action, rule.argument_index)
            if target is None:
                continue
            self._active[(rule.rule_id, target)] = ActiveTemporalObligation(
                rule=rule,
                target=target,
                trigger_step=event.step_id,
            )

    def render(self) -> str:
        if not self._active:
            return ""
        lines = []
        for obligation in self.active:
            recovery = "/".join(obligation.rule.recovery_action_types)
            lines.append(
                f"- Rule {obligation.rule.rule_id}: do not repeat "
                f"{obligation.rule.blocked_action_type} on {obligation.target}; "
                f"first complete a successful {recovery} recovery and re-evaluate."
            )
        return "\n".join(lines)


def _action_argument(action: ActionCall, index: int) -> str | None:
    if index >= len(action.arguments):
        return None
    return action.arguments[index]


def _bind_predicate(
    template: str,
    action: ActionCall,
    pre_ledger: tuple[PredicateState, ...],
    *,
    target_override: str | None = None,
) -> PredicateKey | None:
    bindings = {
        f"arg{index}": value for index, value in enumerate(action.arguments)
    }
    if target_override is not None:
        bindings["arg0"] = target_override
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
    if held is not None:
        bindings["held"] = held
    rendered = template
    for placeholder, value in bindings.items():
        rendered = rendered.replace(f"{{{placeholder}}}", value)
    if "{" in rendered or "}" in rendered:
        return None
    try:
        return PredicateKey.parse(rendered)
    except ValueError:
        return None
