from __future__ import annotations

import pytest

from vista_skill.baselines import (
    CommonGateProposalAdapter,
    CommonUpdateProposal,
    EmbodiSkillNativeUpdater,
    proposal_cluster,
)
from vista_skill.evolution import (
    BoundedPatchApplier,
    CandidateGate,
    DeterministicTransitionChecker,
    GateConfig,
    PairedEpisodeScore,
)
from vista_skill.schemas import PatchOperation, SkillPatch
from vista_skill.schemas import SkillField, UpdateTarget
from vista_skill.skills import initialize_shared_skill


def test_native_execution_lapse_writes_appendix_without_skill_mutation() -> None:
    skill = initialize_shared_skill()
    updater = EmbodiSkillNativeUpdater()
    proposal = CommonUpdateProposal(
        target=UpdateTarget.ABSTAIN,
        field=None,
        content="Check the held-object state before another pick.",
        evidence_ids=("ev1",),
        source_frontend="embodiskill_native",
        persistent=False,
    )
    result = updater.update(skill, proposal)
    assert result.skill == skill
    assert updater.execution_notes == [proposal.content]


def test_common_proposal_cluster_preserves_independent_support() -> None:
    skill = initialize_shared_skill()
    proposal = CommonUpdateProposal(
        target=UpdateTarget.SKILL_UPDATE,
        field=SkillField.PROCEDURE,
        content="Verify the target before acting.",
        evidence_ids=("ev1", "ev2"),
        source_frontend="unconditional_trajectory_reflection",
        persistent=True,
    )
    cluster = proposal_cluster(skill, proposal, ("ep1", "ep2"))
    assert cluster.independent_support_count == 2
    assert cluster.key.skill_version == skill.version


def test_proposal_cluster_floor_follows_configured_threshold() -> None:
    skill = initialize_shared_skill()
    proposal = CommonUpdateProposal(
        target=UpdateTarget.SKILL_UPDATE,
        field=SkillField.PROCEDURE,
        content="Verify the target before acting.",
        evidence_ids=("ev1",),
        source_frontend="unconditional_trajectory_reflection",
        persistent=True,
    )
    # Pilot configs lower min_independent_episodes to 1; a single-episode
    # proposal must build a cluster instead of crashing the gate.
    cluster = proposal_cluster(skill, proposal, ("ep1",), min_episodes=1)
    assert cluster.independent_support_count == 1
    with pytest.raises(ValueError):
        proposal_cluster(skill, proposal, ("ep1",))


def test_common_gate_adapter_accepts_min_episodes_override() -> None:
    skill = initialize_shared_skill()
    proposal = CommonUpdateProposal(
        UpdateTarget.SKILL_UPDATE,
        SkillField.PROCEDURE,
        "Trajectory omitted target verification.",
        ("ev1",),
        "unconditional_trajectory_reflection",
        True,
    )
    gate = CandidateGate(
        BoundedPatchApplier(),
        DeterministicTransitionChecker(),
        PositivePairedEvaluator(),
        GateConfig(
            bootstrap_samples=20,
            proxy_episode_budget=2,
            finalist_episode_budget=2,
        ),
    )
    result = CommonGateProposalAdapter(
        ProcedurePatchGenerator(), gate, min_episodes=1
    ).update(skill, proposal, ("ep1",))
    assert result.accepted
    assert result.skill.version == 1


class ProcedurePatchGenerator:
    def propose(self, skill, cluster):
        return SkillPatch(
            patch_id="baseline-patch",
            skill_id=skill.skill_id,
            parent_version=skill.version,
            field=cluster.key.field,
            operation=PatchOperation.APPEND,
            old="",
            new="Verify the target before acting.",
            evidence_ids=cluster.evidence_ids,
            scope="trajectory failures",
        )


class PositivePairedEvaluator:
    def evaluate(self, parent, candidate, *, stage, episode_budget):
        return tuple(
            PairedEpisodeScore(f"ep{index}", index, 0.0, 1.0, "all")
            for index in range(episode_budget)
        )


def test_common_gate_adapter_can_reach_paired_gate() -> None:
    skill = initialize_shared_skill()
    proposal = CommonUpdateProposal(
        UpdateTarget.SKILL_UPDATE,
        SkillField.PROCEDURE,
        "Repeated trajectories omit target verification.",
        ("ev1", "ev2"),
        "embodiskill_common_gate",
        True,
    )
    gate = CandidateGate(
        BoundedPatchApplier(),
        DeterministicTransitionChecker(),
        PositivePairedEvaluator(),
        GateConfig(
            bootstrap_samples=20,
            proxy_episode_budget=2,
            finalist_episode_budget=2,
        ),
    )
    result = CommonGateProposalAdapter(ProcedurePatchGenerator(), gate).update(
        skill, proposal, ("ep1", "ep2")
    )
    assert result.accepted
    assert result.skill.version == 1
