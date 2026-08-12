from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from vista_skill.belief import BeliefLedger
from vista_skill.schemas import (
    AbstainReason,
    AttributionContext,
    EvidenceSource,
    PredicateEvidence,
    PredicateKey,
    SkillField,
    SkillPredictionRule,
    SkillSpec,
    TerminationPolicy,
    TruthValue,
    UpdateTarget,
)


class FaultType(str, Enum):
    STALE_BELIEF = "stale_belief"
    UNKNOWN_AS_FALSE = "unknown_as_false"
    INSTANCE_IDENTITY = "instance_identity"
    ACTIVATION = "activation"
    PROCEDURE = "procedure"
    EFFECT = "effect"
    TERMINATION = "termination"
    CONSTRAINT = "constraint"
    EXECUTION_LAPSE = "execution_lapse"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    STOCHASTIC_NOOP = "stochastic_noop"


@dataclass(frozen=True)
class FaultCase:
    fault_type: FaultType
    gold_target: UpdateTarget
    gold_field: SkillField | None
    gold_abstain_reason: AbstainReason | None
    context: AttributionContext


def inject_skill_fault(skill: SkillSpec, fault_type: FaultType) -> SkillSpec:
    if fault_type is FaultType.TERMINATION:
        return replace(
            skill,
            termination=("Stop after any one required goal is supported.",),
            termination_policy=TerminationPolicy.ANY_GOAL_EVIDENCE,
        )
    field_map = {
        FaultType.ACTIVATION: SkillField.ACTIVATION,
        FaultType.PROCEDURE: SkillField.PROCEDURE,
        FaultType.EFFECT: SkillField.EFFECT,
        FaultType.CONSTRAINT: SkillField.CONSTRAINT,
    }
    skill_field = field_map.get(fault_type)
    if skill_field is None:
        raise ValueError(f"{fault_type.value} is not a skill fault")
    statements = list(skill.statements(skill_field))
    statements.append(f"Injected contradictory {skill_field.value} rule.")
    prediction = SkillPredictionRule(
        rule_id=f"fault_{skill_field.value}",
        field=skill_field,
        action_type="*",
        predicate=f"{skill_field.value}_satisfied",
        before=TruthValue.FALSE,
        after=TruthValue.TRUE,
    )
    return replace(
        skill,
        **{
            skill_field.value: tuple(statements),
            "prediction_rules": (*skill.prediction_rules, prediction),
        },
    )


def inject_belief_fault(
    ledger: BeliefLedger,
    key: PredicateKey,
    fault_type: FaultType,
    *,
    timestamp: int,
) -> None:
    if fault_type not in {
        FaultType.STALE_BELIEF,
        FaultType.UNKNOWN_AS_FALSE,
        FaultType.INSTANCE_IDENTITY,
    }:
        raise ValueError(f"{fault_type.value} is not a belief fault")
    after = TruthValue.FALSE
    evidence = PredicateEvidence(
        key=key,
        before=ledger.value(key),
        after=after,
        confidence=0.99,
        source=EvidenceSource.ACTIVE_OBSERVATION,
        evidence_id=f"fault:{fault_type.value}:{key.render()}",
        timestamp=timestamp,
        rationale="simulator-assisted diagnostic injection",
    )
    ledger.merge((evidence,))


def diagnostic_case(fault_type: FaultType) -> FaultCase:
    if fault_type in {
        FaultType.STALE_BELIEF,
        FaultType.UNKNOWN_AS_FALSE,
        FaultType.INSTANCE_IDENTITY,
    }:
        return FaultCase(
            fault_type,
            UpdateTarget.BELIEF_REFRESH,
            None,
            None,
            AttributionContext(identity_conflict=fault_type is FaultType.INSTANCE_IDENTITY),
        )
    if fault_type in {
        FaultType.ACTIVATION,
        FaultType.PROCEDURE,
        FaultType.EFFECT,
        FaultType.TERMINATION,
        FaultType.CONSTRAINT,
    }:
        return FaultCase(
            fault_type,
            UpdateTarget.SKILL_UPDATE,
            SkillField(fault_type.value),
            None,
            AttributionContext(),
        )
    reason_map = {
        FaultType.EXECUTION_LAPSE: AbstainReason.EXECUTION_LAPSE,
        FaultType.INSUFFICIENT_EVIDENCE: AbstainReason.INSUFFICIENT_EVIDENCE,
        FaultType.STOCHASTIC_NOOP: AbstainReason.STOCHASTIC_NOOP,
    }
    reason = reason_map[fault_type]
    return FaultCase(
        fault_type,
        UpdateTarget.ABSTAIN,
        None,
        reason,
        AttributionContext(
            executor_followed_skill=False if fault_type is FaultType.EXECUTION_LAPSE else None,
            stochastic_suspected=fault_type is FaultType.STOCHASTIC_NOOP,
        ),
    )
