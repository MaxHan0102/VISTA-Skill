from __future__ import annotations

from dataclasses import replace

import pytest

from vista_skill.action_schema import FixedActionSchema, normalize_entity, parse_action_call
from vista_skill.attribution import CreditAssigner
from vista_skill.belief import BeliefLedger
from vista_skill.clustering import EventClusterer
from vista_skill.evidence import EvidenceExtractor, EvidenceExtractorConfig
from vista_skill.mismatch import compare_transitions
from vista_skill.pipeline import PrimitiveTransition, VistaSkillEngine
from vista_skill.schemas import (
    AbstainReason,
    AttributionContext,
    AttributionResult,
    DeltaSource,
    EvidenceRequest,
    EvidenceSource,
    ExpectedChange,
    MismatchKind,
    PredicateEvidence,
    PredicateKey,
    PredicateState,
    SkillField,
    SkillPredictionRule,
    TerminationPolicy,
    TruthValue,
    UpdateTarget,
)
from vista_skill.skills import initialize_shared_skill


def state(predicate: str, value: TruthValue, *, timestamp: int = 0) -> PredicateState:
    return PredicateState(
        key=PredicateKey.parse(predicate),
        value=value,
        confidence=0.9,
        source="fixture",
        evidence_ids=(f"state:{predicate}",),
        timestamp=timestamp,
    )


def evidence(
    predicate: str,
    before: TruthValue,
    after: TruthValue,
    *,
    evidence_id: str = "ev1",
    confidence: float = 0.9,
    coverage: float = 1.0,
) -> PredicateEvidence:
    return PredicateEvidence(
        key=PredicateKey.parse(predicate),
        before=before,
        after=after,
        confidence=confidence,
        source=EvidenceSource.VISUAL_PAIR,
        evidence_id=evidence_id,
        timestamp=1,
        coverage=coverage,
    )


def test_instance_identity_is_preserved() -> None:
    assert normalize_entity("Apple 12") == "apple_12"
    assert PredicateKey.parse("on(apple_1,tv_stand_2)").render() == "on(apple_1,tv_stand_2)"


def test_habitat_pick_category_is_recovered_from_action_name() -> None:
    action = parse_action_call(7, ("pick_apple", ["robot_0"]))
    assert action.action_type == "pick"
    assert action.arguments == ("apple",)


def test_evidence_request_has_no_prediction_or_skill_fields() -> None:
    request = EvidenceRequest(
        episode_id="ep1",
        step_id=1,
        instruction="place the apple",
        action=parse_action_call(1, ("place", ["tvstand_1"])),
        pre_image="pre.png",
        post_image="post.png",
        feedback="success",
        last_action_success=True,
        pre_ledger=(state("holding(apple_1)", TruthValue.TRUE),),
    )
    serialized = str(request.to_provider_payload()).lower()
    for forbidden in ("expected", "skill", "mismatch", "attribution", "patch", "candidate"):
        assert forbidden not in serialized


def test_belief_only_accepts_evidence_and_stale_input_cannot_overwrite() -> None:
    ledger = BeliefLedger()
    fresh = evidence("holding(apple_1)", TruthValue.UNKNOWN, TruthValue.TRUE)
    ledger.merge((replace(fresh, timestamp=5),))
    ledger.merge((replace(fresh, timestamp=4, after=TruthValue.FALSE, confidence=1.0),))
    assert ledger.value(PredicateKey.parse("holding(apple_1)")) is TruthValue.TRUE
    with pytest.raises(TypeError):
        ledger.merge((object(),))  # type: ignore[arg-type]


def test_unknown_is_unsupported_not_contradiction() -> None:
    prediction = ExpectedChange(
        key=PredicateKey.parse("at(apple_1,tvstand_1)"),
        before=TruthValue.FALSE,
        after=TruthValue.TRUE,
        source=DeltaSource.SKILL,
        source_id="skill:v0:effect",
        skill_field=SkillField.EFFECT,
    )
    mismatches = compare_transitions(
        (prediction,),
        (evidence("at(apple_1,tvstand_1)", TruthValue.FALSE, TruthValue.UNKNOWN),),
    )
    assert mismatches[0].kind is MismatchKind.EXPECTED_UNSUPPORTED


def test_expected_compiler_cites_skill_field_and_action_rule() -> None:
    skill = replace(
        initialize_shared_skill(),
        prediction_rules=(
            SkillPredictionRule(
                "complete_after_place",
                SkillField.TERMINATION,
                "place",
                "task_complete",
                TruthValue.TRUE,
            ),
        ),
    )
    ledger = BeliefLedger()
    ledger.merge((evidence("holding(apple_1)", TruthValue.UNKNOWN, TruthValue.TRUE),))
    changes = FixedActionSchema().compile(
        parse_action_call(3, ("place", ["tvstand_1"])), ledger, skill
    )
    assert any(item.source is DeltaSource.ACTION_SCHEMA for item in changes)
    termination = next(item for item in changes if item.key.name == "task_complete")
    assert termination.source is DeltaSource.SKILL
    assert termination.skill_field is SkillField.TERMINATION
    assert skill.skill_id in termination.source_id


def test_initial_skill_compiles_observable_nontermination_obligations() -> None:
    skill = initialize_shared_skill()
    changes = FixedActionSchema().compile(
        parse_action_call(0, ("pick_apple", ["robot_0"])),
        BeliefLedger(),
        skill,
    )
    fields = {
        item.skill_field
        for item in changes
        if item.source is DeltaSource.SKILL
    }
    assert fields == {SkillField.EFFECT, SkillField.CONSTRAINT}


def test_termination_policy_is_executable_in_expected_branch() -> None:
    ledger = BeliefLedger()
    ledger.merge(
        (
            evidence("at(apple_1,stand_1)", TruthValue.UNKNOWN, TruthValue.TRUE),
            evidence("at(apple_2,stand_1)", TruthValue.UNKNOWN, TruthValue.FALSE, evidence_id="ev2"),
        )
    )
    goals = (
        PredicateKey.parse("at(apple_1,stand_1)"),
        PredicateKey.parse("at(apple_2,stand_1)"),
    )
    action = parse_action_call(0, ("look", []))
    all_goals = FixedActionSchema().compile(
        action, ledger, initialize_shared_skill(), goals
    )
    assert next(item for item in all_goals if item.key.name == "task_complete").after is TruthValue.FALSE
    buggy_skill = replace(
        initialize_shared_skill(), termination_policy=TerminationPolicy.ANY_GOAL_EVIDENCE
    )
    any_goal = FixedActionSchema().compile(action, ledger, buggy_skill, goals)
    assert next(item for item in any_goal if item.key.name == "task_complete").after is TruthValue.TRUE


def test_successful_place_feedback_does_not_claim_visual_goal_relation() -> None:
    extractor = EvidenceExtractor()
    request = EvidenceRequest(
        episode_id="ep1",
        step_id=2,
        instruction="place apple on stand",
        action=parse_action_call(3, ("place", ["tvstand_1"])),
        pre_image="pre.png",
        post_image="post.png",
        feedback="Last action executed successfully and you are holding nothing.",
        last_action_success=True,
        pre_ledger=(state("holding(apple_1)", TruthValue.TRUE),),
    )
    result = extractor.extract(request)
    assert {item.key.name for item in result} == {"holding", "not_holding"}


def test_rule_first_router_abstains_on_unknown_and_routes_skill_field() -> None:
    skill_prediction = ExpectedChange(
        PredicateKey.parse("task_complete"),
        TruthValue.FALSE,
        TruthValue.TRUE,
        DeltaSource.SKILL,
        "skill:v0:termination",
        SkillField.TERMINATION,
    )
    unsupported = compare_transitions(
        (skill_prediction,),
        (evidence("task_complete", TruthValue.FALSE, TruthValue.UNKNOWN),),
    )
    result = CreditAssigner().assign(unsupported)
    assert result.target is UpdateTarget.ABSTAIN
    assert result.subreason is AbstainReason.INSUFFICIENT_EVIDENCE

    contradicted = compare_transitions(
        (skill_prediction,),
        (evidence("task_complete", TruthValue.FALSE, TruthValue.FALSE),),
    )
    result = CreditAssigner().assign(contradicted)
    assert result.target is UpdateTarget.SKILL_UPDATE
    assert result.field is SkillField.TERMINATION


def test_executor_lapse_takes_priority_over_skill_patch() -> None:
    prediction = ExpectedChange(
        PredicateKey.parse("task_complete"),
        TruthValue.FALSE,
        TruthValue.TRUE,
        DeltaSource.SKILL,
        "skill:v0:termination",
        SkillField.TERMINATION,
    )
    mismatches = compare_transitions(
        (prediction,),
        (evidence("task_complete", TruthValue.FALSE, TruthValue.FALSE),),
    )
    result = CreditAssigner().assign(
        mismatches, AttributionContext(executor_followed_skill=False)
    )
    assert result.target is UpdateTarget.ABSTAIN
    assert result.subreason is AbstainReason.EXECUTION_LAPSE


def test_fixed_action_schema_failure_cannot_become_a_skill_update() -> None:
    changes = FixedActionSchema().compile(
        parse_action_call(0, ("nav", ["stand_1"])),
        BeliefLedger(),
        initialize_shared_skill(),
    )
    mismatches = compare_transitions(
        changes,
        (evidence("near(stand_1)", TruthValue.UNKNOWN, TruthValue.FALSE),),
    )
    result = CreditAssigner().assign(mismatches)
    assert {
        item.expected.source for item in mismatches if item.expected is not None
    } == {DeltaSource.ACTION_SCHEMA, DeltaSource.SKILL}
    assert result.target is UpdateTarget.ABSTAIN
    assert result.subreason is AbstainReason.ACTION_MODEL_DISABLED


def test_failed_primitive_is_treated_as_stochastic_before_persistent_routing() -> None:
    engine = VistaSkillEngine(
        initialize_shared_skill(),
        evidence_extractor=EvidenceExtractor(
            FixedVisualProvider(
                evidence("near(stand_1)", TruthValue.UNKNOWN, TruthValue.FALSE)
            ),
            # Force a visual check for navigation in this diagnostic fixture.
            config=EvidenceExtractorConfig(visual_action_types=("nav",)),
        ),
    )
    event = engine.process(
        PrimitiveTransition(
            episode_id="ep1",
            task_id="task1",
            step_id=1,
            instruction="navigate to the stand",
            action=parse_action_call(0, ("nav", ["stand_1"])),
            pre_image="pre.png",
            post_image="post.png",
            feedback="The action failed.",
            last_action_success=False,
            attribution_context=AttributionContext(executor_followed_skill=True),
        )
    )
    assert event.attribution is not None
    assert event.attribution.target is UpdateTarget.ABSTAIN
    assert event.attribution.subreason is AbstainReason.STOCHASTIC_NOOP
    assert engine.clusterer.ready() == ()


class HallucinatingAttributionTeacher:
    def assign(self, mismatches, context):
        return AttributionResult(
            target=UpdateTarget.SKILL_UPDATE,
            field=SkillField.PROCEDURE,
            confidence=0.99,
            mismatch_ids=("invented-mismatch",),
            evidence_ids=("invented-evidence",),
            rationale="unsupported provenance",
        )


def test_teacher_cannot_invent_attribution_provenance() -> None:
    predictions = (
        ExpectedChange(
            PredicateKey("procedure_ok"), TruthValue.TRUE, TruthValue.FALSE,
            DeltaSource.SKILL, "skill:v0:procedure", SkillField.PROCEDURE,
        ),
        ExpectedChange(
            PredicateKey("constraint_ok"), TruthValue.TRUE, TruthValue.FALSE,
            DeltaSource.SKILL, "skill:v0:constraint", SkillField.CONSTRAINT,
        ),
    )
    observed = (
        evidence("procedure_ok", TruthValue.TRUE, TruthValue.TRUE, evidence_id="ev-procedure"),
        evidence("constraint_ok", TruthValue.TRUE, TruthValue.TRUE, evidence_id="ev-constraint"),
    )
    result = CreditAssigner(HallucinatingAttributionTeacher()).assign(
        compare_transitions(predictions, observed)
    )
    assert result.target is UpdateTarget.ABSTAIN
    assert result.subreason is AbstainReason.AMBIGUOUS


def test_recurrence_requires_independent_episodes_and_deduplicates_event() -> None:
    prediction = ExpectedChange(
        PredicateKey.parse("task_complete"),
        TruthValue.FALSE,
        TruthValue.TRUE,
        DeltaSource.SKILL,
        "skill:v0:termination",
        SkillField.TERMINATION,
    )
    mismatch_one = compare_transitions(
        (prediction,),
        (evidence("task_complete", TruthValue.FALSE, TruthValue.FALSE, evidence_id="ev1"),),
    )[0]
    attribution = AttributionResult(
        UpdateTarget.SKILL_UPDATE,
        0.9,
        (mismatch_one.mismatch_id,),
        ("ev1",),
        "termination defect",
        field=SkillField.TERMINATION,
    )
    clusterer = EventClusterer()
    clusterer.add(
        event_id="event1", episode_id="ep1", skill_id="skill", attribution=attribution, mismatch=mismatch_one
    )
    clusterer.add(
        event_id="event1", episode_id="ep1", skill_id="skill", attribution=attribution, mismatch=mismatch_one
    )
    assert clusterer.ready() == ()

    mismatch_two = replace(
        mismatch_one,
        mismatch_id="m2",
        evidence=replace(mismatch_one.evidence, evidence_id="ev2"),  # type: ignore[arg-type]
        evidence_ids=("ev2",),
    )
    attribution_two = replace(attribution, mismatch_ids=("m2",), evidence_ids=("ev2",))
    clusterer.add(
        event_id="event2", episode_id="ep2", skill_id="skill", attribution=attribution_two, mismatch=mismatch_two
    )
    assert len(clusterer.ready()) == 1


def test_recurrence_does_not_double_count_same_evidence_across_episodes() -> None:
    prediction = ExpectedChange(
        PredicateKey("task_complete"),
        TruthValue.FALSE,
        TruthValue.TRUE,
        DeltaSource.SKILL,
        "skill:v0:termination",
        SkillField.TERMINATION,
    )
    mismatch = compare_transitions(
        (prediction,),
        (evidence("task_complete", TruthValue.FALSE, TruthValue.FALSE, evidence_id="shared"),),
    )[0]
    attribution = AttributionResult(
        UpdateTarget.SKILL_UPDATE,
        0.9,
        (mismatch.mismatch_id,),
        ("shared",),
        "termination defect",
        field=SkillField.TERMINATION,
    )
    clusterer = EventClusterer()
    for event_id, episode_id in (("e1", "ep1"), ("e2", "ep2")):
        clusterer.add(
            event_id=event_id,
            episode_id=episode_id,
            skill_id="skill",
            attribution=attribution,
            mismatch=mismatch,
        )
    assert clusterer.ready() == ()


class FixedVisualProvider:
    def __init__(self, item: PredicateEvidence) -> None:
        self.item = item
        self.requests = []

    def extract(self, request: EvidenceRequest):
        self.requests.append(request)
        return (self.item,)


def test_engine_advances_belief_and_freeze_disables_attribution_not_local_belief() -> None:
    visual = FixedVisualProvider(
        evidence("visible(apple_1)", TruthValue.UNKNOWN, TruthValue.TRUE)
    )
    engine = VistaSkillEngine(
        initialize_shared_skill(), evidence_extractor=EvidenceExtractor(visual)
    )
    event = engine.process(
        PrimitiveTransition(
            episode_id="ep1",
            task_id="t1",
            step_id=1,
            instruction="inspect apple",
            action=parse_action_call(0, ("look", [])),
            pre_image="pre.png",
            post_image="post.png",
            feedback="",
            last_action_success=True,
            goal_predicates=(PredicateKey.parse("visible(apple_1)"),),
        )
    )
    assert event.evidence_delta
    assert engine.ledger.value(PredicateKey.parse("visible(apple_1)")) is TruthValue.TRUE
    artifact = engine.freeze()
    assert artifact.skill.frozen
    frozen_event = engine.process(
        PrimitiveTransition(
            "ep1", "t1", 2, "inspect apple", parse_action_call(0, ("look", [])),
            "pre.png", "post.png", "", True
        )
    )
    assert frozen_event.expected_delta == ()
    assert frozen_event.mismatches == ()
    assert frozen_event.attribution is None


def test_prepare_materializes_expected_before_post_action() -> None:
    engine = VistaSkillEngine(initialize_shared_skill())
    action = parse_action_call(0, ("nav", ["stand_1"]))
    prepared = engine.prepare(
        episode_id="ep1",
        task_id="t1",
        step_id=1,
        instruction="navigate to stand",
        action=action,
        pre_image="pre.png",
    )
    assert prepared.expected_delta
    assert not hasattr(prepared, "post_image")


def test_prepare_expires_stale_belief_to_unknown() -> None:
    ledger = BeliefLedger()
    ledger.merge((evidence("near(stand_1)", TruthValue.UNKNOWN, TruthValue.TRUE),))
    engine = VistaSkillEngine(initialize_shared_skill(), ledger=ledger)
    prepared = engine.prepare(
        episode_id="ep1",
        task_id="t1",
        step_id=20,
        instruction="inspect the stand",
        action=parse_action_call(0, ("look", [])),
        pre_image="pre.png",
    )
    assert next(item for item in prepared.pre_ledger if item.key.name == "near").value is TruthValue.UNKNOWN


def test_start_episode_clears_episode_local_belief() -> None:
    engine = VistaSkillEngine(initialize_shared_skill())
    engine.ledger.merge(
        (evidence("holding(apple_1)", TruthValue.UNKNOWN, TruthValue.TRUE),)
    )
    engine.start_episode()
    prepared = engine.prepare(
        episode_id="ep2",
        task_id="t2",
        step_id=1,
        instruction="open the fridge",
        action=parse_action_call(0, ("open", ["fridge_1"])),
        pre_image="pre.png",
    )
    assert prepared.pre_ledger == ()
