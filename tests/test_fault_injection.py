from __future__ import annotations

from dataclasses import replace

from vista_skill.action_schema import FixedActionSchema, parse_action_call
from vista_skill.attribution import AttributionConfig, CreditAssigner
from vista_skill.belief import BeliefLedger
from vista_skill.fault_injection import (
    FaultInjectionReport,
    FaultType,
    build_fault_cases,
    diagnostic_case,
    inject_skill_fault,
    run_fault_injection_evaluation,
)
from vista_skill.mismatch import compare_transitions
from vista_skill.schemas import (
    AbstainReason,
    AttributionContext,
    DeltaSource,
    EvidenceSource,
    PredicateEvidence,
    SkillField,
    TruthValue,
    UpdateTarget,
)
from vista_skill.skills import initialize_shared_skill


def _build_report(per_kind: int = 2):
    skill = initialize_shared_skill()
    schema = FixedActionSchema()
    cases = build_fault_cases(skill, schema, per_kind=per_kind)
    report = run_fault_injection_evaluation(
        CreditAssigner(), skill, cases, action_schema=schema
    )
    return cases, report


def _records(report: FaultInjectionReport, fault_type: FaultType):
    return [
        record
        for record in report.per_case
        if record["fault_type"] == fault_type.value
    ]


def test_clean_skill_does_not_produce_false_skill_updates() -> None:
    cases, report = _build_report()
    clean = _records(report, FaultType.CLEAN)
    assert clean, "expected at least one clean diagnostic case"
    for record in clean:
        assert record["pred_target"] != UpdateTarget.SKILL_UPDATE.value
        assert record["pred_field"] == "none"


def test_skill_procedure_fault_attributed_to_procedure() -> None:
    cases, report = _build_report()
    procedure = _records(report, FaultType.PROCEDURE)
    assert procedure, "expected at least one procedure fault case"
    for record in procedure:
        assert record["pred_target"] == UpdateTarget.SKILL_UPDATE.value
        assert record["pred_field"] == SkillField.PROCEDURE.value
        assert record["pred_target"] == record["gold_target"]


def test_skill_effect_and_constraint_faults_name_their_field() -> None:
    cases, report = _build_report()
    for fault_type, field in (
        (FaultType.EFFECT, SkillField.EFFECT),
        (FaultType.CONSTRAINT, SkillField.CONSTRAINT),
    ):
        records = _records(report, fault_type)
        assert records, f"expected at least one {fault_type.value} fault case"
        for record in records:
            assert record["pred_target"] == UpdateTarget.SKILL_UPDATE.value
            assert record["pred_field"] == field.value


def test_belief_fault_attributed_to_belief_refresh_not_skill_update() -> None:
    # Key correctness property: evidence/ledger errors must never mutate the
    # persistent Skill. This includes instance-identity conflicts, which carry
    # a skill-sourced contradiction that the identity flag must override.
    cases, report = _build_report()
    belief_types = {
        FaultType.STALE_BELIEF,
        FaultType.UNKNOWN_AS_FALSE,
        FaultType.INSTANCE_IDENTITY,
    }
    belief_records = [
        record
        for record in report.per_case
        if record["fault_type"] in {item.value for item in belief_types}
    ]
    assert belief_records, "expected at least one belief fault case"
    for record in belief_records:
        assert record["pred_target"] == UpdateTarget.BELIEF_REFRESH.value
        assert record["pred_target"] != UpdateTarget.SKILL_UPDATE.value
        assert record["pred_field"] == "none"


def test_report_confusion_sums_to_n_cases() -> None:
    cases, report = _build_report()
    assert report.n_cases == len(cases)
    assert sum(report.confusion.values()) == report.n_cases
    # Every confusion key is a (gold_target, pred_target) string pair.
    for gold_target, pred_target in report.confusion:
        assert isinstance(gold_target, str)
        assert isinstance(pred_target, str)
    # The on-diagonal counts equal the per-fault-type correctness tally.
    diagonal = sum(
        count for (gold, pred), count in report.confusion.items() if gold == pred
    )
    assert diagonal == report.n_cases


def test_report_target_macro_f1_populated_and_self_consistent() -> None:
    cases, report = _build_report()
    assert 0.0 <= report.target_macro_f1 <= 1.0
    assert 0.0 <= report.field_macro_f1 <= 1.0
    assert report.target_macro_f1 > 0.0
    assert report.abstention_precision >= 0.0
    assert report.abstention_recall >= 0.0
    # The rule-based assigner perfectly separates these diagnostic cases.
    assert report.target_macro_f1 == 1.0
    assert report.field_macro_f1 == 1.0


def test_evaluation_is_deterministic() -> None:
    skill = initialize_shared_skill()
    schema = FixedActionSchema()
    cases_a = build_fault_cases(skill, schema, per_kind=3)
    cases_b = build_fault_cases(skill, schema, per_kind=3)
    assigner = CreditAssigner()
    first = run_fault_injection_evaluation(assigner, skill, cases_a, action_schema=schema)
    second = run_fault_injection_evaluation(assigner, skill, cases_b, action_schema=schema)
    assert first == second
    assert cases_a == cases_b


def test_externally_authored_case_is_rebuilt_by_the_driver() -> None:
    # A FaultCase built without pre-populated mismatches should be reconstructed
    # on the fly by the driver from its fault_type + base skill + action_schema.
    skill = initialize_shared_skill()
    schema = FixedActionSchema()
    gold = diagnostic_case(FaultType.PROCEDURE)
    case = replace(gold, mismatches=())
    report = run_fault_injection_evaluation(
        CreditAssigner(), skill, [case], action_schema=schema
    )
    assert report.n_cases == 1
    record = report.per_case[0]
    assert record["pred_target"] == UpdateTarget.SKILL_UPDATE.value
    assert record["pred_field"] == SkillField.PROCEDURE.value


def test_abstention_motifs_route_to_documented_reason() -> None:
    cases, report = _build_report()
    expected_reason = {
        FaultType.INSUFFICIENT_EVIDENCE: AbstainReason.INSUFFICIENT_EVIDENCE,
        FaultType.EXECUTION_LAPSE: AbstainReason.EXECUTION_LAPSE,
        FaultType.STOCHASTIC_NOOP: AbstainReason.STOCHASTIC_NOOP,
    }
    for fault_type, reason in expected_reason.items():
        records = _records(report, fault_type)
        assert records, f"expected at least one {fault_type.value} case"
        for record in records:
            assert record["pred_target"] == UpdateTarget.ABSTAIN.value
            assert record["pred_abstain_reason"] == reason.value


def test_teacher_is_optional_and_rule_only_is_default() -> None:
    # The driver must work with a teacher-less assigner and must not require a
    # live model. A teacher supplied through the assigner is exercised by the
    # provenance rules in test_core_method; here we only assert the default path.
    skill = initialize_shared_skill()
    schema = FixedActionSchema()
    cases = build_fault_cases(skill, schema, per_kind=1)
    assigner = CreditAssigner(config=AttributionConfig())
    assert assigner.teacher is None
    report = run_fault_injection_evaluation(assigner, skill, cases, action_schema=schema)
    assert report.target_macro_f1 == 1.0


def test_clean_context_is_neutral_when_fault_is_absent() -> None:
    # A diagnostic_case for a clean step must not accidentally carry an
    # execution-lapse or stochastic flag that would bias the assigner.
    case = diagnostic_case(FaultType.CLEAN)
    assert case.context is not None
    assert case.context.executor_followed_skill is None
    assert case.context.stochastic_suspected is False
    assert case.context.identity_conflict is False
    assert case.gold_target is UpdateTarget.ABSTAIN
    assert case.gold_field is None
    assert case.gold_abstain_reason is AbstainReason.AMBIGUOUS


def test_attribution_context_overrides_take_priority_over_skill_field() -> None:
    # Directly verify the priority the driver relies on for INSTANCE_IDENTITY:
    # a skill-sourced contradiction combined with identity_conflict must yield
    # belief_refresh, not skill_update.
    skill = initialize_shared_skill()
    schema = FixedActionSchema()
    faulty = inject_skill_fault(skill, FaultType.PROCEDURE)
    expected = schema.compile(
        parse_action_call(0, ("look", [])), BeliefLedger(), faulty
    )
    change = next(
        item
        for item in expected
        if item.source is DeltaSource.SKILL
        and item.skill_field is SkillField.PROCEDURE
    )
    evidence = PredicateEvidence(
        key=change.key,
        before=change.before,
        after=change.before,
        confidence=0.92,
        source=EvidenceSource.ACTIVE_OBSERVATION,
        evidence_id="identity_ev",
        timestamp=1,
    )
    mismatches = compare_transitions((change,), (evidence,))
    result = CreditAssigner().assign(
        mismatches, AttributionContext(identity_conflict=True)
    )
    assert result.target is UpdateTarget.BELIEF_REFRESH
