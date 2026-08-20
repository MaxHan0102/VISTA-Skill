"""P1-1 goal–evidence alignment (E7 follow-up).

E7 showed the repair loop cannot close because goal predicates were semantic
labels (``pick_plate(robot_0)``) that never resolve against the primitive
evidence vocabulary (``holding(plate)``), so the ANY/ALL termination policies
compiled identically and termination repairs were replay-invisible. These
tests pin the aligned behaviour: goals land in the observable vocabulary, the
policies diverge on partially-satisfied multi-goal events, and a termination
policy repair becomes replay-verifiable through the real gate checker.
"""
from __future__ import annotations

from dataclasses import replace

from vista_skill.action_schema import FixedActionSchema, parse_action_call
from vista_skill.belief import BeliefLedger, PredicateState
from vista_skill.clustering import ClusterItem, ClusterKey, EvidenceCluster
from vista_skill.evolution import DeterministicTransitionChecker
from vista_skill.fault_injection import FaultType, inject_skill_fault
from vista_skill.mismatch import MismatchKind, compare_transitions
from vista_skill.models import JsonGoalGrounder
from vista_skill.schemas import (
    ActionCall,
    AttributionResult,
    DeltaSource,
    EvidenceSource,
    ExpectedChange,
    Mismatch,
    PredicateEvidence,
    PredicateKey,
    PredicateState as State,
    SkillField,
    TerminationPolicy,
    TruthValue,
    UpdateTarget,
)
from vista_skill.skills import initialize_shared_skill


class _FakeJsonModel:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def complete_json(self, *, system, content, schema, purpose) -> dict:
        return dict(self.payload)


def _image(tmp_path):
    path = tmp_path / "initial.png"
    path.write_bytes(b"png")
    return str(path)


def test_goal_grounder_keeps_only_observable_vocabulary(tmp_path) -> None:
    model = _FakeJsonModel(
        {
            "goal_predicates": [
                "at(apple_1, fridge_1)",
                "pick_apple_1",
                "place_apple_on_fridge",
                "task_complete()",
                "near(sofa)",
                "holding(apple_1, extra)",
                "open(fridge_1)",
            ]
        }
    )
    grounded = JsonGoalGrounder(model).ground(
        "put the apple in the fridge and end at the sofa",
        _image(tmp_path),
        (("pick", ("apple_1",)),),
    )
    assert grounded == (
        PredicateKey("at", ("apple_1", "fridge_1")),
        PredicateKey("near", ("sofa",)),
        PredicateKey("open", ("fridge_1",)),
    )


def test_goal_grounder_empty_when_nothing_representable(tmp_path) -> None:
    model = _FakeJsonModel({"goal_predicates": ["close_fridge", "pick_ball"]})
    grounded = JsonGoalGrounder(model).ground(
        "close the fridge", _image(tmp_path), (("close", ("fridge_1",)),)
    )
    assert grounded == ()


def test_goal_grounder_normalizes_argument_casing(tmp_path) -> None:
    """Goal keys must match action/evidence keys exactly.

    E8b: the model copied instruction casing ("Sofa") while habitat entities
    are lowercase ("sofa"); the un-normalized goal key never satisfied, so
    ANY/ALL stayed indistinguishable on real events.
    """
    model = _FakeJsonModel(
        {"goal_predicates": ["at(ball, receptacle_aabb_Sofa_frl_apartment_sofa)"]}
    )
    grounded = JsonGoalGrounder(model).ground(
        "put the ball on the sofa", _image(tmp_path), (("place", ("sofa",)),)
    )
    assert grounded == (
        PredicateKey("at", ("ball", "receptacle_aabb_sofa_frl_apartment_sofa")),
    )


def _ledger():
    return BeliefLedger.from_snapshot(
        (
            PredicateState(
                key=PredicateKey("holding", ("ball_1",)),
                value=TruthValue.TRUE,
                confidence=0.95,
                source="visual_pair",
                evidence_ids=("ev0",),
                timestamp=1,
            ),
        )
    )


def _goals():
    return (
        PredicateKey("at", ("ball_1", "sofa")),
        PredicateKey("at", ("apple_1", "sofa")),
    )


def _place_action():
    return parse_action_call(6, ("place", ("sofa",)), "place sofa")


def _evidence(with_task_complete_value=TruthValue.FALSE):
    return (
        PredicateEvidence(
            key=PredicateKey("not_holding"),
            before=TruthValue.FALSE,
            after=TruthValue.TRUE,
            confidence=0.95,
            source=EvidenceSource.VISUAL_PAIR,
            evidence_id="ep1:s6:not_holding",
            timestamp=6,
            view_id="ep1:s6",
            coverage=0.9,
            rationale="gripper open",
        ),
        PredicateEvidence(
            key=PredicateKey("holding", ("ball_1",)),
            before=TruthValue.TRUE,
            after=TruthValue.FALSE,
            confidence=0.95,
            source=EvidenceSource.VISUAL_PAIR,
            evidence_id="ep1:s6:holding",
            timestamp=6,
            view_id="ep1:s6",
            coverage=0.9,
            rationale="ball released",
        ),
        PredicateEvidence(
            key=PredicateKey("at", ("ball_1", "sofa")),
            before=TruthValue.UNKNOWN,
            after=TruthValue.TRUE,
            confidence=0.95,
            source=EvidenceSource.VISUAL_PAIR,
            evidence_id="ep1:s6:at",
            timestamp=6,
            view_id="ep1:s6",
            coverage=0.9,
            rationale="ball on sofa",
        ),
        PredicateEvidence(
            key=PredicateKey("task_complete"),
            before=TruthValue.UNKNOWN,
            after=with_task_complete_value,
            confidence=0.9,
            source=EvidenceSource.ACTIVE_OBSERVATION,
            evidence_id="ep1:s6:complete",
            timestamp=6,
            view_id="ep1:s6",
            coverage=0.9,
            rationale="public feedback",
        ),
    )


def test_termination_policies_diverge_on_partially_satisfied_goals() -> None:
    """Mid-episode on a two-goal task, ANY claims completion, ALL does not.

    This divergence is exactly what the injected ANY fault should express and
    what E7 lost to vocabulary misalignment.
    """
    schema = FixedActionSchema()
    parent = inject_skill_fault(initialize_shared_skill(), FaultType.TERMINATION)
    assert parent.termination_policy is TerminationPolicy.ANY_GOAL_EVIDENCE
    candidate = replace(parent, termination_policy=TerminationPolicy.ALL_GOALS_EVIDENCE)

    parent_expected = schema.compile(_place_action(), _ledger(), parent, _goals())
    candidate_expected = schema.compile(_place_action(), _ledger(), candidate, _goals())

    def _task_complete_after(changes):
        for change in changes:
            if change.key == PredicateKey("task_complete"):
                return change.after
        return None

    # ball placed (one of two goals) but apple still missing: env not done.
    assert _task_complete_after(parent_expected) is TruthValue.TRUE
    assert _task_complete_after(candidate_expected) is TruthValue.FALSE


def test_termination_policy_repair_is_replay_verifiable() -> None:
    """The E7 blocker as a regression test: the policy repair must repair.

    Cached event: place(ball_1) on a two-goal task, task_complete evidence
    FALSE. Parent (ANY) expects TRUE -> contradiction; candidate (ALL) expects
    FALSE -> contradiction gone -> DeterministicTransitionChecker reports the
    target repaired without introducing new conflicts.
    """
    schema = FixedActionSchema()
    parent = inject_skill_fault(initialize_shared_skill(), FaultType.TERMINATION)
    candidate = replace(parent, termination_policy=TerminationPolicy.ALL_GOALS_EVIDENCE)

    ledger = _ledger()
    action = _place_action()
    evidence = _evidence()
    parent_expected = schema.compile(action, ledger, parent, _goals())
    parent_mismatches = compare_transitions(parent_expected, evidence)
    target = next(
        m for m in parent_mismatches if m.key == PredicateKey("task_complete")
    )
    assert target.kind is MismatchKind.TERMINATION_CONFLICT

    cluster = EvidenceCluster(
        ClusterKey(
            parent.skill_id,
            SkillField.TERMINATION,
            "contradiction",
            "task_complete",
            "general",
            parent.version,
        )
    )
    cluster.items.append(
        ClusterItem(
            event_id="ep1:s6",
            episode_id="ep1",
            attribution=AttributionResult(
                target=UpdateTarget.SKILL_UPDATE,
                field=SkillField.TERMINATION,
                confidence=0.9,
                mismatch_ids=(target.mismatch_id,),
                evidence_ids=("ep1:s6:complete",),
                rationale="independent evidence contradicts termination policy",
            ),
            mismatch=target,
            action=action,
            pre_ledger=tuple(ledger.snapshot()),
            goal_predicates=_goals(),
            evidence_delta=evidence,
        )
    )
    checks = DeterministicTransitionChecker().check(parent, candidate, cluster)
    assert len(checks) == 1
    assert checks[0].repaired, "policy repair must shrink the task_complete mismatch"
    assert not checks[0].introduced_conflict
    assert checks[0].executable


# --- P1-3: recurrence-key de-fragmentation -------------------------------


def test_task_type_pattern_maps_instructions() -> None:
    from vista_skill.integrations.embodiedbench.runner import _task_type_pattern

    assert (
        _task_type_pattern("Transport all plates and put them on the right counter.")
        == "transport_all_to_receptacle"
    )
    assert _task_type_pattern("pick up the apple from the counter") == "pick_object"
    assert _task_type_pattern("put the knife in the drawer") == "place_object_at_receptacle"
    assert _task_type_pattern("navigate to the sofa") == "navigate_to_target"
    assert _task_type_pattern("open the fridge") == "articulation_open"
    assert _task_type_pattern("close the cabinet 7") == "articulation_close"
    assert _task_type_pattern("") == ""


def test_entity_category_reduces_to_semantic_heads() -> None:
    from vista_skill.integrations.embodiedbench.runner import _entity_category

    assert (
        _entity_category("receptacle_aabb_tbl2_top1_frl_apartment_table_02")
        == "table"
    )
    assert _entity_category("receptacle_aabb_sofa_frl_apartment_sofa") == "sofa"
    assert (
        _entity_category("receptacle_aabb_counter_right_kitchen_counter")
        == "counter"
    )
    assert _entity_category("apple_1") == "apple"


def _termination_conflict_event(
    clusterer, *, event_id, episode_id, task_pattern, object_context, evidence_id
):
    mismatch = Mismatch(
        mismatch_id=f"{event_id}:m",
        key=PredicateKey("task_complete"),
        kind=MismatchKind.TERMINATION_CONFLICT,
        expected=ExpectedChange(
            key=PredicateKey("task_complete"),
            before=TruthValue.UNKNOWN,
            after=TruthValue.FALSE,
            source=DeltaSource.SKILL,
            source_id="skill:v0:termination_policy",
            skill_field=SkillField.TERMINATION,
        ),
        evidence=replace(_evidence()[3], evidence_id=evidence_id),
        evidence_ids=(evidence_id,),
    )
    return clusterer.add(
        event_id=event_id,
        episode_id=episode_id,
        skill_id="shared_embodied_execution",
        attribution=AttributionResult(
            target=UpdateTarget.SKILL_UPDATE,
            field=SkillField.TERMINATION,
            confidence=1.0,
            mismatch_ids=(mismatch.mismatch_id,),
            evidence_ids=(evidence_id,),
            rationale="policy contradicts evidence",
        ),
        mismatch=mismatch,
        task_pattern=task_pattern,
        object_context=object_context,
    )


def test_termination_conflicts_cluster_across_tasks_and_objects() -> None:
    """One policy fault, two different tasks/objects -> one ready cluster.

    Before P1-3 the key fragmented on (task hash, full AABB names), so policy
    violations never reached min_independent_episodes.
    """
    from vista_skill.clustering import EventClusterer

    clusterer = EventClusterer()
    first = _termination_conflict_event(
        clusterer,
        event_id="ep1:s5",
        episode_id="ep1",
        task_pattern="transport_all_to_receptacle",
        object_context="place:table+plate",
        evidence_id="ep1:s5:complete",
    )
    second = _termination_conflict_event(
        clusterer,
        event_id="ep2:s3",
        episode_id="ep2",
        task_pattern="place_object_at_receptacle",
        object_context="nav:sofa+ball",
        evidence_id="ep2:s3:complete",
    )
    assert first is not None and second is not None
    assert first is second  # same cluster object
    assert first.key.task_pattern == "any_task"
    assert first.key.object_context == "policy"
    assert first.independent_support_count == 2
    ready = clusterer.ready()
    assert len(ready) == 1 and ready[0] is first


# --- P1-5: evidence-aware termination routing ----------------------------


def _ev(key, after, evidence_id="ev"):
    return PredicateEvidence(
        key=key,
        before=TruthValue.UNKNOWN,
        after=after,
        confidence=0.95,
        source=EvidenceSource.VISUAL_PAIR,
        evidence_id=evidence_id,
        timestamp=1,
        view_id="v",
        coverage=0.9,
        rationale="seen",
    )


def _termination_mismatch(task_complete_after):
    return Mismatch(
        mismatch_id="m0",
        key=PredicateKey("task_complete"),
        kind=MismatchKind.TERMINATION_CONFLICT,
        expected=ExpectedChange(
            key=PredicateKey("task_complete"),
            before=TruthValue.UNKNOWN,
            after=TruthValue.FALSE,
            source=DeltaSource.SKILL,
            source_id="skill:v0:termination_policy",
            skill_field=SkillField.TERMINATION,
        ),
        evidence=_ev(PredicateKey("task_complete"), task_complete_after, "ev:tc"),
        evidence_ids=("ev:tc",),
    )


def _goal_mismatch(goal, after):
    return Mismatch(
        mismatch_id="m1",
        key=goal,
        kind=MismatchKind.SUPPORTED_UNEXPECTED,
        expected=None,
        evidence=_ev(goal, after, "ev:goal"),
        evidence_ids=("ev:goal",),
    )


def test_env_complete_with_confirmed_goals_routes_to_belief_refresh() -> None:
    """E8c ep6/ep8/ep17 shape: goals confirmed true in the same evidence and
    the env says complete -- belief lag, not a termination-policy defect."""
    from vista_skill.attribution import CreditAssigner
    from vista_skill.schemas import AttributionContext

    goal = PredicateKey("at", ("ball_1", "sofa"))
    result = CreditAssigner().assign(
        (
            _termination_mismatch(TruthValue.TRUE),
            _goal_mismatch(goal, TruthValue.TRUE),
        ),
        AttributionContext(goal_predicates=(goal,)),
    )
    assert result.target is UpdateTarget.BELIEF_REFRESH


def test_env_incomplete_keeps_termination_routing() -> None:
    """E8c ep15 shape: one goal achieved, env says NOT complete -- the genuine
    policy-fault signal must still reach skill_update(termination)."""
    from vista_skill.attribution import CreditAssigner
    from vista_skill.schemas import AttributionContext

    achieved = PredicateKey("at", ("wrench", "sink"))
    pending = PredicateKey("at", ("sponge", "sink"))
    result = CreditAssigner().assign(
        (
            _termination_mismatch(TruthValue.FALSE),
            _goal_mismatch(achieved, TruthValue.TRUE),
            _goal_mismatch(pending, TruthValue.FALSE),
        ),
        AttributionContext(goal_predicates=(achieved, pending)),
    )
    assert result.target is UpdateTarget.SKILL_UPDATE
    assert result.field is SkillField.TERMINATION


def test_completion_routing_requires_goal_context() -> None:
    """Without goal context the rule must not fire (routing unchanged)."""
    from vista_skill.attribution import CreditAssigner
    from vista_skill.schemas import AttributionContext

    goal = PredicateKey("at", ("ball_1", "sofa"))
    result = CreditAssigner().assign(
        (
            _termination_mismatch(TruthValue.TRUE),
            _goal_mismatch(goal, TruthValue.TRUE),
        ),
        AttributionContext(),
    )
    assert result.target is UpdateTarget.SKILL_UPDATE


def test_patch_generator_retries_policy_echo() -> None:
    """E8e/E8f: the teacher restated the implicated policy under shifted
    contexts (twice). For the discriminating conflict shape the policy enum is
    derived from the cached evidence, not model choice."""
    from vista_skill.clustering import ClusterItem, ClusterKey, EvidenceCluster
    from vista_skill.models import JsonBoundedPatchGenerator

    faulty = inject_skill_fault(initialize_shared_skill(), FaultType.TERMINATION)
    cluster = EvidenceCluster(
        ClusterKey(
            faulty.skill_id,
            SkillField.TERMINATION,
            "termination_conflict",
            "any_task",
            "policy",
            faulty.version,
        )
    )
    cluster.items.append(
        ClusterItem(
            event_id="ep15:s4",
            episode_id="ep15",
            attribution=AttributionResult(
                target=UpdateTarget.SKILL_UPDATE,
                field=SkillField.TERMINATION,
                confidence=1.0,
                mismatch_ids=("m0",),
                evidence_ids=("ev:tc",),
                rationale="policy contradicts evidence",
            ),
            mismatch=_termination_mismatch(TruthValue.FALSE),
        )
    )

    class _EchoModel:
        def complete_json(self, *, system, content, schema, purpose) -> dict:
            return {
                "operation": "replace_exact",
                "old": "Stop after any one required goal is supported.",
                "new": "Stop only after every goal is supported.",
                "scope": "termination",
                "rationale": "repair",
                "termination_policy": "any_goal_evidence",
                "prediction_rules": [],
            }

    patch = JsonBoundedPatchGenerator(_EchoModel()).propose(faulty, cluster)
    # The enum comes from the evidence semantics even though the model echoed.
    assert patch.termination_policy is TerminationPolicy.ALL_GOALS_EVIDENCE
    # The model still authors the textual statement.
    assert patch.new == "Stop only after every goal is supported."


def test_non_termination_patch_drops_model_policy_echo() -> None:
    """E8i: 6/6 constraint patches failed static because the model parroted
    the exposed current termination_policy onto a non-termination field. The
    generator strips it -- the applier forbids policy changes outside the
    termination field anyway."""
    from vista_skill.clustering import ClusterItem, ClusterKey, EvidenceCluster
    from vista_skill.fault_injection import FaultType as FT
    from vista_skill.models import JsonBoundedPatchGenerator

    faulty = inject_skill_fault(initialize_shared_skill(), FT.CONSTRAINT_PICK_MULTIHOLD)
    cluster = EvidenceCluster(
        ClusterKey(
            faulty.skill_id,
            SkillField.CONSTRAINT,
            "missing_progress",
            "pick_object",
            "pick:apple",
            faulty.version,
        )
    )
    mismatch = Mismatch(
        mismatch_id="m0",
        key=PredicateKey("not_holding"),
        kind=MismatchKind.MISSING_PROGRESS,
        expected=ExpectedChange(
            key=PredicateKey("not_holding"),
            before=TruthValue.TRUE,
            after=TruthValue.TRUE,
            source=DeltaSource.SKILL,
            source_id="skill:v0:constraint_pick_occupies_gripper",
            skill_field=SkillField.CONSTRAINT,
        ),
        evidence=_ev(PredicateKey("not_holding"), TruthValue.FALSE, "ev:nh"),
        evidence_ids=("ev:nh",),
    )
    cluster.items.append(
        ClusterItem(
            event_id="ep1:s2",
            episode_id="ep1",
            attribution=AttributionResult(
                target=UpdateTarget.SKILL_UPDATE,
                field=SkillField.CONSTRAINT,
                confidence=1.0,
                mismatch_ids=("m0",),
                evidence_ids=("ev:nh",),
                rationale="constraint refuted",
            ),
            mismatch=mismatch,
        )
    )

    class _PolicyParrotingModel:
        def complete_json(self, *, system, content, schema, purpose) -> dict:
            return {
                "operation": "replace_exact",
                "old": "The gripper can hold multiple objects.",
                "new": "The gripper holds one object at a time.",
                "scope": "constraint",
                "rationale": "repair",
                "termination_policy": "any_goal_evidence",
                "prediction_rules": [],
            }

    patch = JsonBoundedPatchGenerator(_PolicyParrotingModel()).propose(faulty, cluster)
    assert patch.field is SkillField.CONSTRAINT
    assert patch.termination_policy is None
    # P1-10: the refuted compiled rule's value is derived from the evidence
    # (not_holding stays FALSE after pick), not restated from the faulty skill.
    assert patch.prediction_rules
    # The patch carries ONLY the attributed field's rules: the applier retains
    # other fields' rules itself, so including them would duplicate rule IDs
    # (E8k crashed the whole experiment this way).
    assert all(r.field is SkillField.CONSTRAINT for r in patch.prediction_rules)
    rule = next(r for r in patch.prediction_rules if r.predicate == "not_holding")
    assert rule.after is TruthValue.FALSE
    assert rule.rule_id == "constraint_pick_occupies_gripper"
