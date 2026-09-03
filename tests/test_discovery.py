from __future__ import annotations

from vista_skill.action_schema import SkillOnlyActionSchema, parse_action_call
from vista_skill.attribution import AttributionConfig, CreditAssigner
from vista_skill.belief import BeliefLedger
from vista_skill.clustering import EventClusterer, RecurrencePolicy
from vista_skill.evolution import (
    BoundedPatchApplier,
    DeterministicTransitionChecker,
    _cluster_fingerprint,
    _cluster_priority,
)
from vista_skill.evidence import EvidenceExtractor
from vista_skill.mismatch import compare_transitions
from vista_skill.models import JsonBoundedPatchGenerator
from vista_skill.pipeline import PrimitiveTransition, VistaSkillEngine
from vista_skill.schemas import (
    AttributionContext,
    EvidenceSource,
    PredicateEvidence,
    PredicateKey,
    SkillField,
    SkillUpdateKind,
    TruthValue,
    UpdateTarget,
)
from vista_skill.skills import interface_only_shared_skill


def _observed(
    predicate: str,
    *,
    evidence_id: str,
    before: TruthValue = TruthValue.UNKNOWN,
    after: TruthValue = TruthValue.TRUE,
) -> PredicateEvidence:
    return PredicateEvidence(
        key=PredicateKey.parse(predicate),
        before=before,
        after=after,
        confidence=0.95,
        source=EvidenceSource.ENV_FEEDBACK,
        evidence_id=evidence_id,
        timestamp=1,
        coverage=1.0,
    )


def _discovery_attribution(
    mismatch,
    action_type: str = "pick",
    *,
    last_action_success: bool | None = None,
):
    return CreditAssigner(
        config=AttributionConfig(skill_discovery_enabled=True)
    ).assign(
        (mismatch,),
        AttributionContext(
            action_type=action_type,
            executor_followed_skill=True,
            last_action_success=last_action_success,
        ),
    )


def test_supported_unexpected_is_discovery_only_when_enabled() -> None:
    mismatch = compare_transitions(
        (), (_observed("holding(apple_1)", evidence_id="ev1"),)
    )[0]
    legacy = CreditAssigner().assign(
        (mismatch,), AttributionContext(action_type="pick")
    )
    discovered = _discovery_attribution(mismatch)
    assert legacy.target is UpdateTarget.BELIEF_REFRESH
    assert discovered.target is UpdateTarget.SKILL_UPDATE
    assert discovered.field is SkillField.EFFECT
    assert discovered.update_kind is SkillUpdateKind.DISCOVERY


def test_task_completion_is_not_mined_as_an_action_effect() -> None:
    mismatch = compare_transitions(
        (), (_observed("task_complete", evidence_id="ev-goal"),)
    )[0]
    result = _discovery_attribution(mismatch)
    assert result.target is UpdateTarget.BELIEF_REFRESH


def test_discovery_recurrence_generalizes_across_object_instances() -> None:
    clusterer = EventClusterer(RecurrencePolicy(min_independent_episodes=2))
    for index, obj in enumerate(("apple_1", "mug_2"), start=1):
        action = parse_action_call(index, (f"pick_{obj}", ["robot_0"]))
        mismatch = compare_transitions(
            (), (_observed(f"holding({obj})", evidence_id=f"ev{index}"),)
        )[0]
        attribution = _discovery_attribution(mismatch)
        clusterer.add(
            event_id=f"event{index}",
            episode_id=f"episode{index}",
            skill_id="shared_embodied_execution",
            attribution=attribution,
            mismatch=mismatch,
            action=action,
        )
    ready = clusterer.ready()
    assert len(ready) == 1
    assert ready[0].key.object_context == "pick|holding({arg0})|true"


def test_discovery_rejects_unbound_instance_specific_predicate() -> None:
    action = parse_action_call(1, ("pick_apple_1", ["robot_0"]))
    mismatch = compare_transitions(
        (), (_observed("at(apple_1,table_9)", evidence_id="ev1"),)
    )[0]
    cluster = EventClusterer().add(
        event_id="event1",
        episode_id="episode1",
        skill_id="shared_embodied_execution",
        attribution=_discovery_attribution(mismatch),
        mismatch=mismatch,
        action=action,
    )
    assert cluster is None


def test_discovery_patch_adds_grounded_compiled_rule_and_repairs_cache() -> None:
    skill = interface_only_shared_skill()
    schema = SkillOnlyActionSchema()
    clusterer = EventClusterer(RecurrencePolicy(min_independent_episodes=2))
    for index, obj in enumerate(("apple_1", "mug_2"), start=1):
        action = parse_action_call(index, (f"pick_{obj}", ["robot_0"]))
        observed = _observed(f"holding({obj})", evidence_id=f"ev{index}")
        mismatch = compare_transitions((), (observed,))[0]
        clusterer.add(
            event_id=f"event{index}",
            episode_id=f"episode{index}",
            skill_id=skill.skill_id,
            attribution=_discovery_attribution(mismatch),
            mismatch=mismatch,
            action=action,
            evidence_delta=(observed,),
        )
    cluster = clusterer.ready()[0]

    class _TextAuthor:
        def complete_json(self, *, system, content, schema, purpose):
            raise AssertionError("grounded discovery must not call the teacher")

    patch = JsonBoundedPatchGenerator(_TextAuthor()).propose(skill, cluster)
    assert patch.operation.value == "append"
    assert patch.old == ""
    assert len(patch.prediction_rules or ()) == 1
    assert patch.prediction_rules[0].predicate == "holding({arg0})"
    candidate = BoundedPatchApplier().apply(skill, patch)
    checks = DeterministicTransitionChecker(schema).check(skill, candidate, cluster)
    assert checks and all(item.repaired for item in checks)


def test_explicit_failed_pick_routes_to_constraint_discovery_not_stochastic() -> None:
    engine = VistaSkillEngine(
        interface_only_shared_skill(),
        action_schema=SkillOnlyActionSchema(),
        evidence_extractor=EvidenceExtractor(),
        credit_assigner=CreditAssigner(
            config=AttributionConfig(skill_discovery_enabled=True)
        ),
    )
    event = engine.process(
        PrimitiveTransition(
            episode_id="episode1",
            task_id="task1",
            step_id=1,
            instruction="pick the sponge",
            action=parse_action_call(1, ("pick_sponge", ["robot_0"])),
            pre_image="pre.png",
            post_image="post.png",
            feedback=(
                "Last action is invalid. Robot cannot pick any object that is "
                "not near the robot."
            ),
            last_action_success=False,
            attribution_context=AttributionContext(
                executor_followed_skill=True,
                action_type="pick",
            ),
        )
    )
    assert event.attribution is not None
    assert event.attribution.target is UpdateTarget.SKILL_UPDATE
    assert event.attribution.field is SkillField.CONSTRAINT
    assert event.attribution.update_kind is SkillUpdateKind.DISCOVERY


def test_failed_pick_constraint_generalizes_and_explains_cached_failures() -> None:
    skill = interface_only_shared_skill()
    schema = SkillOnlyActionSchema()
    clusterer = EventClusterer(RecurrencePolicy(min_independent_episodes=2))
    for index, obj in enumerate(("sponge_1", "spoon_2"), start=1):
        action = parse_action_call(index, (f"pick_{obj}", ["robot_0"]))
        observed = _observed(
            f"near({obj})",
            evidence_id=f"failure{index}",
            after=TruthValue.FALSE,
        )
        mismatch = compare_transitions((), (observed,))[0]
        clusterer.add(
            event_id=f"event{index}",
            episode_id=f"episode{index}",
            skill_id=skill.skill_id,
            attribution=_discovery_attribution(
                mismatch,
                last_action_success=False,
            ),
            mismatch=mismatch,
            action=action,
            evidence_delta=(observed,),
        )
    cluster = clusterer.ready()[0]
    assert cluster.key.field is SkillField.CONSTRAINT
    assert cluster.key.object_context == "pick|near({arg0})|false"

    class _NoTeacher:
        def complete_json(self, **kwargs):
            raise AssertionError("grounded constraint discovery is deterministic")

    patch = JsonBoundedPatchGenerator(_NoTeacher()).propose(skill, cluster)
    assert patch.field is SkillField.CONSTRAINT
    assert "require evidence" in patch.new
    assert patch.prediction_rules[0].before is TruthValue.FALSE
    assert patch.prediction_rules[0].after is TruthValue.FALSE
    candidate = BoundedPatchApplier().apply(skill, patch)
    checks = DeterministicTransitionChecker(schema).check(skill, candidate, cluster)
    assert checks and all(item.repaired for item in checks)
    preconditions = schema.precondition_checks(
        cluster.items[0].action,
        BeliefLedger(),
        candidate,
    )
    assert preconditions[0]["predicate"] == "near(sponge_1)"
    assert preconditions[0]["required"] == "true"
    assert preconditions[0]["satisfied"] is False


def test_discovery_cluster_is_not_reproposed_when_support_grows() -> None:
    clusterer = EventClusterer(RecurrencePolicy(min_independent_episodes=2))
    cluster = None
    for index, obj in enumerate(("apple_1", "mug_2"), start=1):
        action = parse_action_call(index, (f"pick_{obj}", ["robot_0"]))
        mismatch = compare_transitions(
            (), (_observed(f"holding({obj})", evidence_id=f"ev{index}"),)
        )[0]
        cluster = clusterer.add(
            event_id=f"event{index}",
            episode_id=f"episode{index}",
            skill_id="shared_embodied_execution",
            attribution=_discovery_attribution(mismatch),
            mismatch=mismatch,
            action=action,
        )
    assert cluster is not None
    fingerprint = _cluster_fingerprint(cluster)
    action = parse_action_call(3, ("pick_plate_3", ["robot_0"]))
    mismatch = compare_transitions(
        (), (_observed("holding(plate_3)", evidence_id="ev3"),)
    )[0]
    grown = clusterer.add(
        event_id="event3",
        episode_id="episode3",
        skill_id="shared_embodied_execution",
        attribution=_discovery_attribution(mismatch),
        mismatch=mismatch,
        action=action,
    )
    assert grown is cluster
    assert _cluster_fingerprint(grown) == fingerprint


def test_constraint_candidates_are_prioritized_over_effect_candidates() -> None:
    clusterer = EventClusterer(RecurrencePolicy(min_independent_episodes=1))
    action = parse_action_call(1, ("pick_apple_1", ["robot_0"]))
    effect_mismatch = compare_transitions(
        (), (_observed("holding(apple_1)", evidence_id="effect"),)
    )[0]
    constraint_mismatch = compare_transitions(
        (),
        (
            _observed(
                "near(apple_1)",
                evidence_id="constraint",
                after=TruthValue.FALSE,
            ),
        ),
    )[0]
    effect = clusterer.add(
        event_id="effect",
        episode_id="ep-effect",
        skill_id="shared_embodied_execution",
        attribution=_discovery_attribution(effect_mismatch),
        mismatch=effect_mismatch,
        action=action,
    )
    constraint = clusterer.add(
        event_id="constraint",
        episode_id="ep-constraint",
        skill_id="shared_embodied_execution",
        attribution=_discovery_attribution(
            constraint_mismatch,
            last_action_success=False,
        ),
        mismatch=constraint_mismatch,
        action=action,
    )
    assert effect is not None and constraint is not None
    assert _cluster_priority(constraint) < _cluster_priority(effect)
