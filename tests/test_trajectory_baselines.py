from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vista_skill.baselines import (
    CommonGateProposalAdapter,
    EmbodiSkillFrontend,
    EmbodiSkillNativeUpdater,
    EmbodiSkillRoute,
    EpisodeSummary,
    TrajectoryReflection,
    UnconditionalReflectionFrontend,
)
from vista_skill.config import load_config
from vista_skill.evolution import (
    BoundedPatchApplier,
    CandidateGate,
    DeterministicTransitionChecker,
    GateConfig,
    PairedEpisodeScore,
)
from vista_skill.integrations.embodiedbench import cli
from vista_skill.integrations.embodiedbench.cli import parse_args
from vista_skill.integrations.embodiedbench.runner import EpisodeResult
from vista_skill.lineage import LineageStore
from vista_skill.models import JsonTrajectoryTeacher
from vista_skill.pipeline import VistaSkillEngine
from vista_skill.schemas import SkillField
from vista_skill.skills import initialize_shared_skill
from vista_skill.workflow import TrajectoryEvolutionWorkflow


CONFIG = load_config("configs/vista_p0.json")


def _failed_summary(episode_id: str, *, instruction: str = "pick the apple") -> EpisodeSummary:
    return EpisodeSummary(
        episode_id=episode_id,
        instruction=instruction,
        success=False,
        trajectory=("nav apple", "pick apple"),
        current_skill="skill-v0",
        failure_reason="target object not reached",
    )


class _FakeJsonModel:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.purposes: list[str] = []

    def complete_json(self, *, system, content, schema, purpose) -> dict:
        self.purposes.append(purpose)
        return dict(self.payload)


class _FakeTeacher:
    """Direct TrajectoryTeacher stand-in returning a fixed reflection."""

    def __init__(
        self,
        route: EmbodiSkillRoute = EmbodiSkillRoute.FAIL_SKILL,
        field: SkillField = SkillField.PROCEDURE,
    ) -> None:
        self.route = route
        self.field = field

    def reflect(self, episode: EpisodeSummary) -> TrajectoryReflection:
        return TrajectoryReflection(
            route=self.route,
            content="Verify the target object identity before acting.",
            target_field=self.field,
            confidence=0.9,
            evidence_ids=(f"{episode.episode_id}:ev0",),
        )


class _ProcedurePatchGenerator:
    def propose(self, skill, cluster):
        from vista_skill.schemas import PatchOperation, SkillPatch

        return SkillPatch(
            patch_id="traj-baseline-patch",
            skill_id=skill.skill_id,
            parent_version=skill.version,
            field=cluster.key.field,
            operation=PatchOperation.APPEND,
            old="",
            new="Verify the target object identity before acting.",
            evidence_ids=cluster.evidence_ids,
            scope="trajectory failures",
        )


class _PositivePairedEvaluator:
    def evaluate(self, parent, candidate, *, stage, episode_budget):
        return tuple(
            PairedEpisodeScore(f"ep{index}", index, 0.0, 1.0, "all")
            for index in range(episode_budget)
        )


def _episode_result(episode_id: str) -> EpisodeResult:
    return EpisodeResult(
        episode_id=episode_id,
        instruction="pick the apple",
        task_success=0.0,
        task_progress=0.3,
        reward_mean=0.0,
        environment_steps=2,
        planner_steps=2,
        invalid_actions=0,
        planner_output_errors=0,
        elapsed_seconds=0.0,
        trajectory=("nav apple", "pick apple"),
        failure_reason="target object not reached",
    )


# --------------------------------------------------------------------------- #
# JsonTrajectoryTeacher
# --------------------------------------------------------------------------- #


def test_trajectory_teacher_parses_model_reflection() -> None:
    payload = {
        "route": "FAIL_SKILL",
        "content": "Re-confirm the target receptacle before placing.",
        "target_field": "procedure",
        "confidence": 0.88,
        "evidence_ids": ["ep1:ev0", "ep1:ev1"],
    }
    model = _FakeJsonModel(payload)
    teacher = JsonTrajectoryTeacher(model)
    reflection = teacher.reflect(_failed_summary("ep1"))
    assert reflection.route is EmbodiSkillRoute.FAIL_SKILL
    assert reflection.target_field is SkillField.PROCEDURE
    assert reflection.confidence == pytest.approx(0.88)
    assert reflection.evidence_ids == ("ep1:ev0", "ep1:ev1")
    assert model.purposes == ["trajectory_reflection"]


def test_trajectory_teacher_drops_duplicate_evidence_ids() -> None:
    payload = {
        "route": "FAIL_EXECUTION",
        "content": "Check gripper state.",
        "target_field": None,
        "confidence": 0.5,
        "evidence_ids": ["ev", "ev", "ev2"],
    }
    teacher = JsonTrajectoryTeacher(_FakeJsonModel(payload))
    reflection = teacher.reflect(_failed_summary("ep2"))
    assert reflection.route is EmbodiSkillRoute.FAIL_EXECUTION
    assert reflection.target_field is None
    assert reflection.evidence_ids == ("ev", "ev2")


# --------------------------------------------------------------------------- #
# TrajectoryEvolutionWorkflow
# --------------------------------------------------------------------------- #


def test_native_workflow_revises_after_independent_failures(tmp_path) -> None:
    engine = VistaSkillEngine(initialize_shared_skill())
    lineage = LineageStore(tmp_path / "lineage.jsonl")
    workflow = TrajectoryEvolutionWorkflow(
        engine,
        frontend=EmbodiSkillFrontend(_FakeTeacher(), common_gate=False),
        updater=EmbodiSkillNativeUpdater(),
        lineage=lineage,
        config=CONFIG,
        protocol={"run_id": "native"},
    )
    workflow.consume_episode(_failed_summary("ep1"))
    assert workflow.evolve_ready() == (0, ())
    workflow.consume_episode(_failed_summary("ep2"))
    ready, results = workflow.evolve_ready()

    assert ready == 1
    assert len(results) == 1
    assert engine.skill.version == 1
    assert results[0].decision.accepted is True
    assert len(lineage.records()) == 1
    assert lineage.records()[0]["accepted"] is True


def test_native_workflow_requires_independent_episodes(tmp_path) -> None:
    engine = VistaSkillEngine(initialize_shared_skill())
    workflow = TrajectoryEvolutionWorkflow(
        engine,
        frontend=EmbodiSkillFrontend(_FakeTeacher(), common_gate=False),
        updater=EmbodiSkillNativeUpdater(),
        lineage=LineageStore(tmp_path / "lineage.jsonl"),
        config=CONFIG,
        protocol={},
    )
    # Same episode reflected twice must not count as independent support.
    workflow.consume_episode(_failed_summary("ep1"))
    workflow.consume_episode(_failed_summary("ep1"))
    assert workflow.evolve_ready() == (0, ())
    assert engine.skill.version == 0


def test_native_workflow_execution_lapse_does_not_mutate_skill(tmp_path) -> None:
    engine = VistaSkillEngine(initialize_shared_skill())
    updater = EmbodiSkillNativeUpdater()
    workflow = TrajectoryEvolutionWorkflow(
        engine,
        frontend=EmbodiSkillFrontend(
            _FakeTeacher(route=EmbodiSkillRoute.FAIL_EXECUTION, field=SkillField.PROCEDURE),
            common_gate=False,
        ),
        updater=updater,
        lineage=LineageStore(tmp_path / "lineage.jsonl"),
        config=CONFIG,
        protocol={},
    )
    for episode_id in ("ep1", "ep2"):
        workflow.consume_episode(_failed_summary(episode_id))
        workflow.evolve_ready()
    assert engine.skill.version == 0
    assert updater.execution_notes  # appendix populated
    assert engine.skill.procedure == initialize_shared_skill().procedure


def test_common_gate_workflow_promotes_through_paired_gate(tmp_path) -> None:
    engine = VistaSkillEngine(initialize_shared_skill())
    lineage = LineageStore(tmp_path / "lineage.jsonl")
    gate = CandidateGate(
        BoundedPatchApplier(),
        DeterministicTransitionChecker(engine.action_schema),
        _PositivePairedEvaluator(),
        GateConfig(
            bootstrap_samples=20,
            proxy_episode_budget=2,
            finalist_episode_budget=2,
        ),
    )
    updater = CommonGateProposalAdapter(_ProcedurePatchGenerator(), gate)
    workflow = TrajectoryEvolutionWorkflow(
        engine,
        frontend=EmbodiSkillFrontend(_FakeTeacher(), common_gate=True),
        updater=updater,
        lineage=lineage,
        config=CONFIG,
        protocol={"run_id": "common_gate"},
    )
    for episode_id in ("ep1", "ep2"):
        workflow.consume_episode(_failed_summary(episode_id))
    ready, results = workflow.evolve_ready()

    assert ready == 1
    assert results[0].decision.accepted is True
    assert engine.skill.version == 1
    # The common gate must record the staged decision lineage.
    record = lineage.records()[0]
    assert record["accepted"] is True
    assert any(
        stage["stage"] == "paired_finalist" for stage in record["decision"]["stages"]
    )


def test_unconditional_frontend_skips_successful_episodes(tmp_path) -> None:
    engine = VistaSkillEngine(initialize_shared_skill())
    workflow = TrajectoryEvolutionWorkflow(
        engine,
        frontend=UnconditionalReflectionFrontend(_FakeTeacher()),
        updater=EmbodiSkillNativeUpdater(),
        lineage=LineageStore(tmp_path / "lineage.jsonl"),
        config=CONFIG,
        protocol={},
    )
    success = EpisodeSummary(
        episode_id="ep_ok",
        instruction="pick the apple",
        success=True,
        trajectory=("nav apple", "pick apple"),
        current_skill="skill-v0",
        failure_reason="",
    )
    workflow.consume_episode(success)
    assert workflow.evolve_ready() == (0, ())
    # Two failures are still required for the unconditional proposal to fire.
    workflow.consume_episode(_failed_summary("ep1"))
    assert workflow.evolve_ready() == (0, ())
    workflow.consume_episode(_failed_summary("ep2"))
    ready, _ = workflow.evolve_ready()
    assert ready == 1
    assert engine.skill.version == 1


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #


def test_parse_args_accepts_trajectory_baseline_methods() -> None:
    for method in (
        "embodiskill_star_native",
        "embodiskill_star_common_gate",
        "vista_without_vtca",
    ):
        parsed = parse_args(["experiment", "--method", method, "--method-model", "m"])
        assert parsed.method == method


def test_trajectory_experiment_requires_method_model() -> None:
    args = parse_args(
        ["experiment", "--method", "embodiskill_star_native", "--diagnostic"]
    )
    with pytest.raises(ValueError, match="requires --method-model"):
        cli._run_experiment(args)


class _FakeEnv:
    def close(self) -> None:
        pass


class _FakeRunner:
    def run_episode(self, *, expected_episode_id):
        return _episode_result(expected_episode_id)


class _FakeMethodModel:
    def __init__(self) -> None:
        self.usage: dict = {}

    def complete_json(self, *, system, content, schema, purpose) -> dict:
        return {
            "route": "FAIL_SKILL",
            "content": "Verify the target object identity before picking.",
            "target_field": "procedure",
            "confidence": 0.9,
            "evidence_ids": ["ev0"],
        }


def test_native_experiment_runs_three_rotated_seeds(tmp_path, monkeypatch) -> None:
    output = tmp_path / "experiment"
    args = parse_args(
        [
            "experiment",
            "--method",
            "embodiskill_star_native",
            "--diagnostic",
            "--max-acquisition-episodes",
            "3",
            "--output-dir",
            str(output),
            "--method-model",
            "fake-teacher",
        ]
    )

    monkeypatch.setattr(cli, "create_habitat_env", lambda *a, **k: _FakeEnv())
    monkeypatch.setattr(cli, "seed_habitat_env", lambda env, seed: None)
    monkeypatch.setattr(cli, "_make_runner", lambda *a, **k: _FakeRunner())
    monkeypatch.setattr(cli, "_make_method_model", lambda args, seed=None: _FakeMethodModel())
    cli._run_experiment(args)

    manifest = json.loads((output / "experiment_manifest.json").read_text())
    assert manifest["run_count"] == 3
    assert manifest["runs"][0]["executor_usage"] is None  # non-env fake runner
    for seed in (0, 1, 2):
        run_dir = output / f"seed_{seed}"
        assert (run_dir / "frozen_skill.json").exists()
        # A native body revision must have fired (>=2 failures of 3 episodes)
        # and produced lineage + a post-hoc update audit.
        lineage = [json.loads(line) for line in (run_dir / "lineage.jsonl").read_text().splitlines() if line.strip()]
        assert lineage and lineage[0]["accepted"] is True
        assert (run_dir / "update_audit.json").exists()
