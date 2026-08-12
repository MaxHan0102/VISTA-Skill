from __future__ import annotations

import json
from dataclasses import replace

import pytest

from vista_skill.clustering import ClusterItem, ClusterKey, EvidenceCluster
from vista_skill.evolution import (
    BoundedPatchApplier,
    CachedTransitionCheck,
    CandidateGate,
    GateConfig,
    PairedEpisodeScore,
    PatchValidationError,
)
from vista_skill.protocol import DataSplit
from vista_skill.config import load_config
from vista_skill.schemas import (
    AttributionResult,
    DeltaSource,
    EvidenceSource,
    ExpectedChange,
    Mismatch,
    MismatchKind,
    PatchOperation,
    PredicateEvidence,
    PredicateKey,
    SkillField,
    SkillPatch,
    SkillPredictionRule,
    TerminationPolicy,
    TruthValue,
    UpdateTarget,
)
from vista_skill.skills import (
    initialize_shared_skill,
    load_skill_artifact,
    load_skill_artifact_record,
    save_skill_artifact,
)


def patch_for(skill, *, old: str, new: str, operation=PatchOperation.REPLACE_EXACT):
    return SkillPatch(
        patch_id="patch1",
        skill_id=skill.skill_id,
        parent_version=skill.version,
        field=SkillField.TERMINATION,
        operation=operation,
        old=old,
        new=new,
        evidence_ids=("ev1", "ev2"),
        scope="multi-instance tasks",
    )


def test_patch_is_exact_target_and_single_field() -> None:
    skill = initialize_shared_skill()
    old = skill.termination[0]
    candidate = BoundedPatchApplier().apply(
        skill,
        patch_for(skill, old=old, new="Stop only after every bound target is verified."),
    )
    assert candidate.termination == ("Stop only after every bound target is verified.",)
    assert candidate.procedure == skill.procedure
    with pytest.raises(PatchValidationError):
        BoundedPatchApplier().apply(skill, patch_for(skill, old="missing", new="replacement"))


def test_termination_patch_updates_compiled_policy() -> None:
    skill = replace(
        initialize_shared_skill(), termination_policy=TerminationPolicy.ANY_GOAL_EVIDENCE
    )
    patch = replace(
        patch_for(
            skill,
            old=skill.termination[0],
            new="Stop only after every goal is evidence verified.",
        ),
        termination_policy=TerminationPolicy.ALL_GOALS_EVIDENCE,
    )
    candidate = BoundedPatchApplier().apply(skill, patch)
    assert candidate.termination_policy is TerminationPolicy.ALL_GOALS_EVIDENCE


def test_patch_rejects_instance_specific_rule() -> None:
    skill = initialize_shared_skill()
    with pytest.raises(PatchValidationError):
        BoundedPatchApplier().apply(
            skill,
            patch_for(skill, old=skill.termination[0], new="Verify apple_1 before stopping."),
        )


@pytest.mark.parametrize(
    "rules",
    [
        (
            SkillPredictionRule(
                "bad_action", SkillField.TERMINATION, "teleport", "task_complete", TruthValue.TRUE
            ),
        ),
        (
            SkillPredictionRule(
                "bad_placeholder", SkillField.TERMINATION, "place", "at({arg2},{arg0})", TruthValue.TRUE
            ),
        ),
        (
            SkillPredictionRule(
                "unbound_placeholder", SkillField.TERMINATION, "nav", "near({arg1})", TruthValue.TRUE
            ),
        ),
        (
            SkillPredictionRule(
                "instance_bound", SkillField.TERMINATION, "place", "at(apple_1,{arg0})", TruthValue.TRUE
            ),
        ),
        (
            SkillPredictionRule(
                "duplicate", SkillField.TERMINATION, "place", "task_complete", TruthValue.TRUE
            ),
            SkillPredictionRule(
                "duplicate", SkillField.TERMINATION, "place", "task_complete", TruthValue.TRUE
            ),
        ),
    ],
)
def test_patch_rejects_unsafe_compiled_rules(rules) -> None:
    skill = initialize_shared_skill()
    patch = replace(
        patch_for(
            skill,
            old=skill.termination[0],
            new="Stop only after every target is verified.",
        ),
        prediction_rules=rules,
    )
    with pytest.raises(PatchValidationError):
        BoundedPatchApplier().apply(skill, patch)


class PassingTransitionChecker:
    def check(self, parent, candidate, cluster):
        return (CachedTransitionCheck("event", repaired=True),)


class ScoreEvaluator:
    def __init__(self, proxy, finalist):
        self.proxy = proxy
        self.finalist = finalist

    def evaluate(self, parent, candidate, *, stage, episode_budget):
        values = self.proxy if stage == "proxy" else self.finalist
        return tuple(
            PairedEpisodeScore(f"ep{i}", 0, 0.0, score, subgroup)
            for i, (score, subgroup) in enumerate(values)
        )


def empty_cluster(skill) -> EvidenceCluster:
    key = PredicateKey("task_complete")
    expected = ExpectedChange(
        key,
        TruthValue.FALSE,
        TruthValue.TRUE,
        DeltaSource.SKILL,
        f"{skill.skill_id}:v{skill.version}:termination",
        SkillField.TERMINATION,
    )
    observed = PredicateEvidence(
        key,
        TruthValue.FALSE,
        TruthValue.FALSE,
        0.9,
        EvidenceSource.VISUAL_PAIR,
        "ev1",
        1,
    )
    mismatch = Mismatch(
        "m1", key, MismatchKind.TERMINATION_CONFLICT, expected, observed, ("ev1",)
    )
    attribution = AttributionResult(
        UpdateTarget.SKILL_UPDATE,
        0.9,
        ("m1",),
        ("ev1", "ev2"),
        "termination defect",
        field=SkillField.TERMINATION,
    )
    cluster = EvidenceCluster(
        ClusterKey(skill.skill_id, SkillField.TERMINATION, "termination_conflict", "all", "objects")
    )
    cluster.items.append(ClusterItem("event", "episode", attribution, mismatch))
    return cluster


def test_gate_accepts_positive_paired_candidate() -> None:
    skill = initialize_shared_skill()
    scores = [(0.2, "base"), (0.3, "long"), (0.1, "base"), (0.4, "long")]
    gate = CandidateGate(
        BoundedPatchApplier(),
        PassingTransitionChecker(),
        ScoreEvaluator(scores, scores),
        GateConfig(bootstrap_samples=200, proxy_episode_budget=4, finalist_episode_budget=4),
    )
    decision, candidate = gate.evaluate(
        skill,
        patch_for(skill, old=skill.termination[0], new="Stop only after all targets are verified."),
        empty_cluster(skill),
    )
    assert decision.accepted
    assert candidate is not None


def test_gate_rejects_protected_subgroup_regression() -> None:
    skill = initialize_shared_skill()
    proxy = [(0.2, "base"), (0.2, "long"), (0.2, "base"), (0.2, "long")]
    finalist = [(0.5, "base"), (-0.2, "long"), (0.5, "base"), (-0.2, "long")]
    gate = CandidateGate(
        BoundedPatchApplier(),
        PassingTransitionChecker(),
        ScoreEvaluator(proxy, finalist),
        GateConfig(bootstrap_samples=200, proxy_episode_budget=4, finalist_episode_budget=4),
    )
    decision, candidate = gate.evaluate(
        skill,
        patch_for(skill, old=skill.termination[0], new="Stop only after all targets are verified."),
        empty_cluster(skill),
    )
    assert not decision.accepted
    assert candidate is None
    assert decision.stages[-1].metrics["worst_subgroup_delta"] == pytest.approx(-0.2)


def test_gate_fails_closed_when_paired_budget_is_incomplete() -> None:
    skill = initialize_shared_skill()
    one_score = [(0.3, "base")]
    gate = CandidateGate(
        BoundedPatchApplier(),
        PassingTransitionChecker(),
        ScoreEvaluator(one_score, one_score),
        GateConfig(bootstrap_samples=10, proxy_episode_budget=2, finalist_episode_budget=2),
    )
    decision, candidate = gate.evaluate(
        skill,
        patch_for(skill, old=skill.termination[0], new="Stop only after all targets are verified."),
        empty_cluster(skill),
    )
    assert not decision.accepted
    assert candidate is None
    assert "required episodes" in decision.reason


def test_split_integrity_rejects_leakage() -> None:
    with pytest.raises(ValueError, match="leakage"):
        DataSplit(("a",), ("b",), ("a",), ("c",))


def test_protocol_config_drives_runtime_thresholds() -> None:
    config = load_config("configs/vista_p0.json")
    assert config.recurrence.min_independent_episodes == 2
    assert config.gate.proxy_episode_budget == 10
    assert config.gate.finalist_episode_budget == 30
    assert config.raw["evolution_seeds"] == [0, 1, 2]
    assert len(config.digest) == 64


def test_frozen_skill_artifact_round_trip_and_digest(tmp_path) -> None:
    skill = replace(initialize_shared_skill(), frozen=True)
    output = tmp_path / "skill.json"
    save_skill_artifact(output, skill, protocol={"split_hash": "abc"})
    assert load_skill_artifact(output, require_frozen=True) == skill
    record = load_skill_artifact_record(output, require_frozen=True)
    assert record.skill == skill
    assert record.protocol == {"split_hash": "abc"}
    assert len(record.artifact_sha256) == 64
    payload = output.read_text(encoding="utf-8").replace(
        "Unknown is not false.", "Unknown is false."
    )
    output.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_skill_artifact(output)


def test_skill_artifact_digest_covers_protocol_metadata(tmp_path) -> None:
    skill = replace(initialize_shared_skill(), frozen=True)
    output = tmp_path / "skill.json"
    save_skill_artifact(
        output,
        skill,
        protocol={"config_sha256": "config-a", "manifest_sha256": "manifest-a"},
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["protocol"]["manifest_sha256"] = "manifest-b"
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="envelope digest mismatch"):
        load_skill_artifact_record(output)


def test_protocol_tuple_round_trip_uses_canonical_json_types(tmp_path) -> None:
    skill = replace(initialize_shared_skill(), frozen=True)
    output = tmp_path / "skill.json"
    save_skill_artifact(output, skill, protocol={"seeds": (0, 1, 2)})
    assert load_skill_artifact_record(output).protocol == {"seeds": [0, 1, 2]}
