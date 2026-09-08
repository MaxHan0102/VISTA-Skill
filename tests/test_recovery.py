from __future__ import annotations

from dataclasses import replace

from vista_skill.action_schema import parse_action_call
from vista_skill.evolution import BoundedPatchApplier
from vista_skill.evidence import EvidenceExtractor, EvidenceExtractorConfig
from vista_skill.recovery import propose_recovery, recovery_chains
from vista_skill.schemas import (
    EvidenceRequest, EvidenceSource, PredicateEvidence, PredicateKey, PredicateState, TruthValue,
)
from vista_skill.skills import interface_only_shared_skill, load_skill_artifact_record, save_skill_artifact, skill_digest
from vista_skill.temporal import TemporalRuleMonitor
from test_temporal import _event, _rule


def _chain(episode="a", target="apple"):
    def event(step, action, success, evidence=()):
        return {
            "event_id": f"{episode}:{step}", "episode_id": episode, "step_id": step,
            "action": {"action_type": action, "arguments": [target if action == "pick" else "table"]},
            "last_action_success": success, "evidence_delta": list(evidence),
            "attribution": {"target": "skill_update", "identifiability": {"identified": True}},
        }
    def evidence(step):
        return {"key": {"name": "near", "arguments": [target]}, "after": "false",
                "confidence": 0.9, "coverage": 1.0, "timestamp": step,
                "evidence_id": f"{episode}:{step}:near"}
    return [event(1, "pick", False, [evidence(1)]), event(2, "nav", True),
            event(3, "pick", False, [evidence(3)])]


def test_discovery_requires_independent_identified_chains():
    parent = interface_only_shared_skill()
    patch, _ = propose_recovery(parent, _chain() * 3)
    assert patch is None
    events = _chain() + _chain("b", "mug")
    patch, audit = propose_recovery(parent, events)
    assert patch is not None and len(audit["independent_episodes"]) == 2
    candidate = BoundedPatchApplier().apply(parent, patch)
    assert candidate.procedure and not parent.procedure
    assert candidate.temporal_rules[0].recovery_release == "target_evidence"
    assert "apple" not in patch.new and "mug" not in patch.new
    events[0]["attribution"]["identifiability"]["identified"] = False
    assert propose_recovery(parent, events)[0] is None


def test_resolved_evidence_and_other_target_do_not_support_chain():
    events = _chain()
    resolved = dict(events[0]["evidence_delta"][0], after="true", timestamp=2)
    events[1]["evidence_delta"] = [resolved]
    assert not recovery_chains(events)
    events = _chain()
    events[2]["action"]["arguments"] = ["mug"]
    assert not recovery_chains(events)


def test_target_recovery_ignores_unrelated_navigation_and_weak_or_stale_evidence():
    rule = replace(_rule(), recovery_release="target_evidence")
    monitor = TemporalRuleMonitor()
    monitor.start_episode(replace(interface_only_shared_skill(), temporal_rules=(rule,)))
    pick = parse_action_call(1, ("pick", ["apple_1", "robot_0"]))
    nav = parse_action_call(0, ("nav", ["table_1"]))
    negative = PredicateEvidence(PredicateKey("near", ("apple_1",)), TruthValue.UNKNOWN,
                                 TruthValue.FALSE, 0.9, EvidenceSource.ENV_FEEDBACK, "failure", 1)
    monitor.observe(_event(pick, success=False, evidence=(negative,)))
    monitor.observe(_event(nav, success=True, step=2))
    assert not monitor.admit(pick).allowed
    for item in (
        replace(negative, after=TruthValue.UNKNOWN, timestamp=3),
        replace(negative, after=TruthValue.TRUE, confidence=0.5, timestamp=3),
        replace(negative, after=TruthValue.TRUE, timestamp=1),
        replace(negative, key=PredicateKey("near", ("mug_1",)), after=TruthValue.TRUE, timestamp=3),
    ):
        monitor.observe(_event(nav, success=True, evidence=(item,), step=3))
        assert not monitor.admit(pick).allowed
    monitor.observe(_event(nav, success=True, evidence=(
        replace(negative, after=TruthValue.TRUE, timestamp=4, evidence_id="fresh"),
    ), step=4))
    assert monitor.admit(pick).allowed


def test_new_temporal_artifact_roundtrip_and_legacy_digest(tmp_path):
    parent = replace(interface_only_shared_skill(), temporal_rules=(_rule(),))
    path = tmp_path / "old.json"
    save_skill_artifact(path, parent)
    assert "recovery_release" not in path.read_text()
    assert skill_digest(load_skill_artifact_record(path).skill) == skill_digest(parent)
    candidate = replace(parent, temporal_rules=(replace(_rule(), recovery_release="target_evidence"),))
    save_skill_artifact(path, candidate)
    assert load_skill_artifact_record(path).skill == candidate
    assert skill_digest(candidate) != skill_digest(parent)


def test_failed_precondition_visual_trigger_is_prediction_blind_and_bounded():
    class Provider:
        def __init__(self):
            self.requests = []

        def extract(self, request):
            self.requests.append(request)
            assert not hasattr(request, "skill") and not hasattr(request, "expected_delta")
            return ()

    provider = Provider()
    extractor = EvidenceExtractor(provider, EvidenceExtractorConfig(
        visual_action_types=(), visual_on_unresolved_goals=False,
        visual_on_failed_preconditions=True, max_visual_calls_per_episode=1,
    ))
    state = PredicateState(PredicateKey("near", ("apple",)), TruthValue.FALSE, 0.9, "feedback", ("ev",), 1)
    request = EvidenceRequest(
        episode_id="a", step_id=2, instruction="move apple", pre_image="pre", post_image="post",
        action=parse_action_call(0, ("nav", ["table"])), feedback="success",
        last_action_success=True, pre_ledger=(state,), goal_predicates=(),
    )
    extractor.extract(replace(request, pre_ledger=()))
    assert not provider.requests
    extractor.extract(request)
    extractor.extract(replace(request, step_id=3))
    assert len(provider.requests) == 1
    extractor.extract(replace(request, episode_id="b"))
    assert len(provider.requests) == 2


def test_unknown_precondition_keeps_visual_recovery_active_until_fresh_resolution():
    key = PredicateKey("near", ("apple",))

    class Provider:
        calls = 0

        def extract(self, request):
            self.calls += 1
            return (PredicateEvidence(key, TruthValue.UNKNOWN,
                TruthValue.UNKNOWN if self.calls == 1 else TruthValue.TRUE,
                0.9, EvidenceSource.VISUAL_PAIR, f"visual-{self.calls}", request.step_id),)

    provider = Provider()
    extractor = EvidenceExtractor(provider, EvidenceExtractorConfig(
        visual_action_types=(), visual_on_unresolved_goals=False,
        visual_on_failed_preconditions=True, max_visual_calls_per_episode=4,
    ))
    state = PredicateState(key, TruthValue.FALSE, 0.9, "feedback", ("failure",), 1)
    request = EvidenceRequest("a", 2, "move apple", parse_action_call(0, ("nav", ["table"])),
                              "pre", "post", "success", True, (state,))
    extractor.extract(request)
    unknown = replace(state, value=TruthValue.UNKNOWN, timestamp=2)
    extractor.extract(replace(request, step_id=3, pre_ledger=(unknown,)))
    assert provider.calls == 2  # UNKNOWN does not disable the observation channel.
    resolved = replace(state, value=TruthValue.TRUE, timestamp=3)
    extractor.extract(replace(request, step_id=4, pre_ledger=(resolved,)))
    assert provider.calls == 2
    extractor.extract(replace(request, episode_id="new", pre_ledger=()))
    assert provider.calls == 2  # Pending evidence does not cross episode boundaries.
