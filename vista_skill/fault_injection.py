from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Sequence

from vista_skill.action_schema import ActionSchema
from vista_skill.attribution import CreditAssigner
from vista_skill.belief import BeliefLedger
from vista_skill.metrics import (
    abstention_metrics,
    confusion_counts,
    macro_f1,
    safe_ratio,
)
from vista_skill.mismatch import compare_transitions
from vista_skill.schemas import (
    AbstainReason,
    ActionCall,
    AttributionContext,
    DeltaSource,
    EvidenceSource,
    ExpectedChange,
    Mismatch,
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
    # Vocabulary-aligned skill fault: inverts S0's compiled pick effect
    # (holding after pick) so mismatches are COVERED contradictions on a
    # predicate the visual evidence layer actually observes -- unlike the
    # synthetic <field>_satisfied predicates, which can only ever produce
    # uncovered/unsupported mismatches that attribution abstains on.
    EFFECT_PICK_INVERSION = "effect_pick_inversion"
    # Driver-only marker: a no-fault episode step. Used by build_fault_cases
    # to verify that the assigner does not invent Skill defects on clean input.
    CLEAN = "clean"


@dataclass(frozen=True)
class FaultCase:
    fault_type: FaultType
    gold_target: UpdateTarget
    gold_field: SkillField | None
    gold_abstain_reason: AbstainReason | None
    context: AttributionContext
    mismatches: tuple[Mismatch, ...] = ()


def inject_skill_fault(skill: SkillSpec, fault_type: FaultType) -> SkillSpec:
    if fault_type is FaultType.EFFECT_PICK_INVERSION:
        inverted = tuple(
            replace(rule, after=TruthValue.FALSE)
            if rule.rule_id == "effect_pick_holds_target_category"
            else rule
            for rule in skill.prediction_rules
        )
        if inverted == skill.prediction_rules:
            raise ValueError("S0 pick effect rule not found; injection target moved")
        return replace(
            skill,
            effect=("Picking up an object leaves the gripper empty.",),
            prediction_rules=inverted,
        )
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
    if fault_type is FaultType.CLEAN:
        return FaultCase(
            fault_type,
            UpdateTarget.ABSTAIN,
            None,
            AbstainReason.AMBIGUOUS,
            AttributionContext(),
        )
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


# ---------------------------------------------------------------------------
# Fault-injection attribution-quality driver (RQ2 diagnostic).
#
# This block ties the existing inject_skill_fault / inject_belief_fault
# primitives together with CreditAssigner and the metrics module so a caller
# can measure attribution quality: does the assigner route each injected
# fault to its gold target (skill_update / belief_refresh / abstain)?
#
# The default diagnostic path is rule-only -- a CreditAssigner constructed
# without an AttributionTeacher -- so the driver never calls a live model.
# Pass an assigner that carries an AttributionTeacher to measure the
# teacher-augmented path; the driver itself remains teacher-agnostic.
# ---------------------------------------------------------------------------


_SKILL_FAULT_FIELDS: tuple[FaultType, ...] = (
    FaultType.ACTIVATION,
    FaultType.PROCEDURE,
    FaultType.EFFECT,
    FaultType.CONSTRAINT,
)

# Fault types covered by the default diagnostic set. Order is fixed so the
# emitted report is deterministic.
_DIAGNOSTIC_FAULTS: tuple[FaultType, ...] = (
    FaultType.CLEAN,
    FaultType.PROCEDURE,
    FaultType.EFFECT,
    FaultType.CONSTRAINT,
    FaultType.ACTIVATION,
    FaultType.TERMINATION,
    FaultType.STALE_BELIEF,
    FaultType.UNKNOWN_AS_FALSE,
    FaultType.INSTANCE_IDENTITY,
    FaultType.INSUFFICIENT_EVIDENCE,
    FaultType.EXECUTION_LAPSE,
    FaultType.STOCHASTIC_NOOP,
)


def _look_action(case_index: int) -> ActionCall:
    """A neutral primitive action for materializing ``*``-bound skill rules."""

    return ActionCall(
        action_id=case_index,
        action_type="look",
        arguments=(),
        text="look",
        raw_action="look",
    )


def _contradict(
    expected_change: ExpectedChange,
    *,
    fault_type: FaultType,
    case_index: int,
    confidence: float = 0.92,
) -> tuple[Mismatch, ...]:
    """Build the mismatch produced when evidence refutes a skill prediction."""

    evidence = PredicateEvidence(
        key=expected_change.key,
        before=expected_change.before,
        after=expected_change.before,
        confidence=confidence,
        source=EvidenceSource.ACTIVE_OBSERVATION,
        evidence_id=f"fault_ev:{fault_type.value}:{case_index}",
        timestamp=case_index,
    )
    return compare_transitions((expected_change,), (evidence,))


def _skill_fault_expected(
    faulty_skill: SkillSpec,
    field: SkillField,
    action_schema: ActionSchema,
    case_index: int,
) -> ExpectedChange:
    """Compile a faulty Skill and isolate the injected ``fault_<field>`` rule."""

    compiled = action_schema.compile(
        _look_action(case_index), BeliefLedger(), faulty_skill
    )
    needle = f"fault_{field.value}"
    for change in compiled:
        if (
            change.source is DeltaSource.SKILL
            and change.skill_field is field
            and needle in change.source_id
        ):
            return change
    raise AssertionError(
        f"inject_skill_fault({field.value}) did not compile to a skill rule "
        f"matching {needle!r}; check that the rule uses action_type='*'"
    )


def _build_fault_mismatches(
    fault_type: FaultType,
    base_skill: SkillSpec,
    action_schema: ActionSchema,
    *,
    case_index: int = 0,
) -> tuple[Mismatch, ...]:
    """Reconstruct the mismatch a fault produces using real pipeline kinds.

    The emitted kinds (contradiction / missing_progress / supported_unexpected
    / expected_unsupported / termination_conflict) are exactly what
    :func:`vista_skill.mismatch.compare_transitions` yields in the live
    evaluator loop, so CreditAssigner sees realistic inputs.
    """

    if fault_type is FaultType.CLEAN:
        # Benign no-fault step. Even offsets carry no signal (correctly
        # abstain); odd offsets carry a harmless new observation whose correct
        # attribution is belief_refresh, never skill_update.
        if case_index % 2 == 0:
            return ()
        key = PredicateKey("observed", (f"clean_{case_index}",))
        evidence = PredicateEvidence(
            key=key,
            before=TruthValue.UNKNOWN,
            after=TruthValue.TRUE,
            confidence=0.9,
            source=EvidenceSource.ACTIVE_OBSERVATION,
            evidence_id=f"clean_ev:{case_index}",
            timestamp=case_index,
        )
        return compare_transitions((), (evidence,))

    if fault_type in _SKILL_FAULT_FIELDS:
        faulty = inject_skill_fault(base_skill, fault_type)
        field = SkillField(fault_type.value)
        expected_change = _skill_fault_expected(
            faulty, field, action_schema, case_index
        )
        return _contradict(
            expected_change, fault_type=fault_type, case_index=case_index
        )

    if fault_type is FaultType.TERMINATION:
        faulty = inject_skill_fault(base_skill, fault_type)
        expected_change = ExpectedChange(
            key=PredicateKey("task_complete"),
            before=TruthValue.FALSE,
            after=TruthValue.TRUE,
            source=DeltaSource.SKILL,
            source_id=f"{faulty.skill_id}:v{faulty.version}:termination_policy",
            skill_field=SkillField.TERMINATION,
        )
        return _contradict(
            expected_change, fault_type=fault_type, case_index=case_index
        )

    if fault_type is FaultType.INSTANCE_IDENTITY:
        # A genuine skill-sourced contradiction that the identity-conflict
        # flag must override: instance-identity errors are episode-local and
        # must never mutate the persistent Skill.
        faulty = inject_skill_fault(base_skill, FaultType.PROCEDURE)
        expected_change = _skill_fault_expected(
            faulty, SkillField.PROCEDURE, action_schema, case_index
        )
        return _contradict(
            expected_change, fault_type=fault_type, case_index=case_index
        )

    if fault_type in {FaultType.STALE_BELIEF, FaultType.UNKNOWN_AS_FALSE}:
        # Belief correction: exercise inject_belief_fault on a seeded ledger so
        # the case reflects the primitive's real effect, then surface the
        # resulting transition as supported_unexpected evidence the Skill never
        # predicted. The assigner must route this to belief_refresh.
        key = PredicateKey("belief_corrected", (fault_type.value,))
        ledger = BeliefLedger()
        ledger.merge(
            (
                PredicateEvidence(
                    key=key,
                    before=TruthValue.UNKNOWN,
                    after=TruthValue.TRUE,
                    confidence=0.6,
                    source=EvidenceSource.DERIVED_GOAL,
                    evidence_id=f"seed:{fault_type.value}:{case_index}",
                    timestamp=0,
                ),
            )
        )
        inject_belief_fault(ledger, key, fault_type, timestamp=case_index + 1)
        evidence = PredicateEvidence(
            key=key,
            before=TruthValue.TRUE,
            after=ledger.value(key),
            confidence=0.9,
            source=EvidenceSource.ACTIVE_OBSERVATION,
            evidence_id=f"fault_ev:{fault_type.value}:{case_index}",
            timestamp=case_index,
        )
        return compare_transitions((), (evidence,))

    if fault_type is FaultType.INSUFFICIENT_EVIDENCE:
        expected_change = ExpectedChange(
            key=PredicateKey("unconfirmed_state"),
            before=TruthValue.FALSE,
            after=TruthValue.TRUE,
            source=DeltaSource.SKILL,
            source_id=f"diag:v0:insufficient:{case_index}",
            skill_field=SkillField.PROCEDURE,
        )
        evidence = PredicateEvidence(
            key=expected_change.key,
            before=TruthValue.UNKNOWN,
            after=TruthValue.UNKNOWN,
            confidence=0.5,
            source=EvidenceSource.ACTIVE_OBSERVATION,
            evidence_id=f"fault_ev:insufficient:{case_index}",
            timestamp=case_index,
        )
        return compare_transitions((expected_change,), (evidence,))

    if fault_type is FaultType.EXECUTION_LAPSE:
        # Skill-sourced contradiction, but the executor deviated from the
        # canonical Skill, so the canonical rule is not the defect.
        faulty = inject_skill_fault(base_skill, FaultType.PROCEDURE)
        expected_change = _skill_fault_expected(
            faulty, SkillField.PROCEDURE, action_schema, case_index
        )
        return _contradict(
            expected_change, fault_type=fault_type, case_index=case_index
        )

    if fault_type is FaultType.STOCHASTIC_NOOP:
        key = PredicateKey("noop_state", (str(case_index),))
        evidence = PredicateEvidence(
            key=key,
            before=TruthValue.FALSE,
            after=TruthValue.TRUE,
            confidence=0.9,
            source=EvidenceSource.ACTIVE_OBSERVATION,
            evidence_id=f"fault_ev:noop:{case_index}",
            timestamp=case_index,
        )
        return compare_transitions((), (evidence,))

    return ()


def _field_label(value: SkillField | None) -> str:
    return "none" if value is None else value.value


@dataclass(frozen=True)
class FaultInjectionReport:
    n_cases: int
    target_macro_f1: float
    field_macro_f1: float
    abstention_precision: float
    abstention_recall: float
    confusion: dict[tuple[str, str], int]
    per_case: tuple[dict[str, object], ...]


def build_fault_cases(
    skill: SkillSpec,
    action_schema: ActionSchema,
    *,
    per_kind: int = 2,
    fault_types: Sequence[FaultType] | None = None,
) -> list[FaultCase]:
    """Build a balanced, deterministic fault-injection diagnostic set.

    Each enabled FaultType is materialized ``per_kind`` times. Determinism is
    guaranteed by the fixed iteration order and a monotonic per-case index --
    no random number generation is involved. Skill faults are produced via
    :func:`inject_skill_fault` (compiled through ``action_schema`` so the
    expected changes mirror the live evaluator) and belief faults reflect the
    transition that :func:`inject_belief_fault` would merge, so every case
    exercises the real injection primitives.

    The default ``fault_types`` tuple covers: a clean/no-fault step,
    skill-procedure / -effect / -constraint / -activation / -termination
    faults, belief-ledger faults (stale, unknown-as-false, instance-identity),
    and the three abstention motifs (insufficient evidence, execution lapse,
    stochastic no-op).
    """

    selected = tuple(fault_types) if fault_types is not None else _DIAGNOSTIC_FAULTS
    cases: list[FaultCase] = []
    for fault_type in selected:
        for offset in range(per_kind):
            mismatches = _build_fault_mismatches(
                fault_type, skill, action_schema, case_index=offset
            )
            base_case = diagnostic_case(fault_type)
            gold_target = base_case.gold_target
            gold_field = base_case.gold_field
            gold_reason = base_case.gold_abstain_reason
            # Clean odd-offset steps carry a benign supported-unexpected
            # signal whose correct attribution is belief_refresh.
            if fault_type is FaultType.CLEAN and offset % 2 == 1:
                gold_target = UpdateTarget.BELIEF_REFRESH
                gold_reason = None
            cases.append(
                replace(
                    base_case,
                    gold_target=gold_target,
                    gold_abstain_reason=gold_reason,
                    mismatches=mismatches,
                )
            )
    return cases


def run_fault_injection_evaluation(
    assigner: CreditAssigner,
    skill: SkillSpec,
    cases: Sequence[FaultCase],
    *,
    action_schema: ActionSchema,
) -> FaultInjectionReport:
    """Evaluate attribution quality on a fault-injection diagnostic set.

    For each FaultCase the driver reconstructs the mismatch the fault produces
    and invokes :meth:`CreditAssigner.assign` with the case's
    AttributionContext. Each case built by :func:`build_fault_cases` already
    carries its reconstructed mismatches; a case supplied without mismatches is
    rebuilt on the fly from its fault type via ``action_schema`` (and the base
    ``skill``), so externally authored cases still work.

    The predicted target/field is compared against the case's gold target/field
    and aggregated via :mod:`vista_skill.metrics`:

    * :func:`macro_f1` -> ``target_macro_f1`` and ``field_macro_f1``
      (``None`` fields are mapped to the ``"none"`` label so non-skill rows
      contribute to the field score).
    * :func:`confusion_counts` -> ``confusion`` keyed by
      ``(gold_target, pred_target)`` string values.
    * :func:`abstention_metrics` -> ``abstention_precision``; abstention recall
      is computed against the gold-abstain subset.

    Returns a :class:`FaultInjectionReport` carrying the aggregate metrics plus
    a ``per_case`` record tuple for subgroup and error analysis.
    """

    gold_targets: list[str] = []
    pred_targets: list[str] = []
    gold_fields: list[str] = []
    pred_fields: list[str] = []
    gold_should_abstain: list[bool] = []
    predicted_abstain: list[bool] = []
    per_case: list[dict[str, object]] = []

    for case in cases:
        mismatches = case.mismatches
        if not mismatches and case.fault_type is not FaultType.CLEAN:
            # Externally authored case without pre-built mismatches: rebuild.
            mismatches = _build_fault_mismatches(
                case.fault_type, skill, action_schema
            )
        result = assigner.assign(mismatches, case.context)

        gold_target = case.gold_target
        pred_target = result.target
        gold_field_label = _field_label(case.gold_field)
        pred_field_label = _field_label(result.field)

        gold_targets.append(gold_target.value)
        pred_targets.append(pred_target.value)
        gold_fields.append(gold_field_label)
        pred_fields.append(pred_field_label)
        gold_should_abstain.append(gold_target is UpdateTarget.ABSTAIN)
        predicted_abstain.append(pred_target is UpdateTarget.ABSTAIN)

        per_case.append(
            {
                "fault_type": case.fault_type.value,
                "gold_target": gold_target.value,
                "pred_target": pred_target.value,
                "gold_field": gold_field_label,
                "pred_field": pred_field_label,
                "gold_abstain_reason": (
                    case.gold_abstain_reason.value
                    if case.gold_abstain_reason is not None
                    else None
                ),
                "pred_abstain_reason": (
                    result.subreason.value
                    if result.subreason is not None
                    else None
                ),
                "target_correct": gold_target is pred_target,
                "field_correct": gold_field_label == pred_field_label,
                "confidence": result.confidence,
                "rationale": result.rationale,
            }
        )

    confusion = {
        (str(gold), str(pred)): count
        for (gold, pred), count in confusion_counts(
            gold_targets, pred_targets
        ).items()
    }
    abstention = abstention_metrics(gold_should_abstain, predicted_abstain)
    gold_abstain_count = sum(gold_should_abstain)
    correct_abstain = sum(
        gold and pred
        for gold, pred in zip(gold_should_abstain, predicted_abstain)
    )

    return FaultInjectionReport(
        n_cases=len(cases),
        target_macro_f1=macro_f1(gold_targets, pred_targets),
        field_macro_f1=macro_f1(gold_fields, pred_fields),
        abstention_precision=abstention["abstention_precision"],
        abstention_recall=safe_ratio(correct_abstain, gold_abstain_count),
        confusion=confusion,
        per_case=tuple(per_case),
    )
