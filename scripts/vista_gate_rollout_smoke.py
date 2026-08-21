"""Deterministic full-gate-rollover smoke against live vLLM + real Habitat.

Purpose: exercise the EXPENSIVE, unit-test-uncovered gate machinery end-to-end so
the real experiment is unlikely to error there:
  - _make_runner rollout closure invoked per (parent/candidate, coordinate, stage)
  - CandidateGate proxy + finalist stages (real Habitat episodes via vLLM)
  - bootstrap LCB + worst-subgroup regression on real rollout scores
  - LineageStore.append + AcceptedUpdateSnapshotStore snapshot
  - run_rotated_update_audit (post-hoc parent-vs-candidate re-evaluation)

The acquisition recurrence gate (>=2 independent skill_update episodes) is
bypassed by feeding a ready EvidenceCluster straight to the gate. The real
DeterministicTransitionChecker is unit-tested separately (tests/test_evolution.py);
a PassingTransitionChecker is used here ONLY to guarantee the proxy/finalist
rollout stages actually execute regardless of the synthetic cluster's shape.
Gate budgets are shrunk (proxy=2, finalist=3, bootstrap=200) for speed.

Run with the vLLM server up on :8001 and the embench env (Habitat).
"""
from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

# must be set before importing cli (it reads env at import)
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

from vista_skill.clustering import ClusterItem, ClusterKey, EvidenceCluster
from vista_skill.config import load_config
from vista_skill.evolution import (
    BoundedPatchApplier,
    CachedTransitionCheck,
    CandidateGate,
    GateConfig,
)
from vista_skill.integrations.embodiedbench import cli
from vista_skill.integrations.embodiedbench.environment import (
    create_habitat_env,
    seed_habitat_env,
    seed_process_rngs,
)
from vista_skill.lineage import LineageStore
from vista_skill.models import (
    JsonBoundedPatchGenerator,
    OpenAICompatibleJsonModel,
)
from vista_skill.evaluation import RolloutScore, composite_task_score, EpisodeCoordinate
from vista_skill.evolution import PairedEpisodeScore
from vista_skill.pipeline import VistaSkillEngine
from vista_skill.belief import BeliefLedger
from vista_skill.evidence import EvidenceExtractor
from vista_skill.protocol import load_experiment_manifest
from vista_skill.schemas import (
    DeltaSource,
    ExpectedChange,
    Mismatch,
    MismatchKind,
    PredicateEvidence,
    PredicateKey,
    SkillField,
    TruthValue,
    UpdateTarget,
)
from vista_skill.schemas import AttributionResult
from vista_skill.skills import initialize_shared_skill, skill_digest
from vista_skill.update_audit import make_rotated_audit_plan, run_rotated_update_audit

BASE_URL = os.environ.get("VISTA_METHOD_BASE_URL", "http://127.0.0.1:8001/v1")
MODEL = os.environ.get("VISTA_METHOD_MODEL", "Qwen/Qwen3-VL-8B-Instruct")


class _PassingTransitionChecker:
    """Always-repaired transition checker, to force proxy/finalist to run."""

    def check(self, parent, candidate, cluster):
        return tuple(
            CachedTransitionCheck(item.event_id, repaired=True) for item in cluster.items
        )


def _termination_cluster(skill):
    key = PredicateKey("task_complete")
    expected = ExpectedChange(
        key, TruthValue.FALSE, TruthValue.TRUE, DeltaSource.SKILL,
        f"{skill.skill_id}:v{skill.version}:termination", SkillField.TERMINATION,
    )
    observed = PredicateEvidence(
        key, TruthValue.FALSE, TruthValue.FALSE, 0.9, "visual_pair", "ev1", 1,
    )
    mismatch = Mismatch(
        "m1", key, MismatchKind.TERMINATION_CONFLICT, expected, observed, ("ev1", "ev2"),
    )
    attribution = AttributionResult(
        target=UpdateTarget.SKILL_UPDATE, confidence=0.9, mismatch_ids=("m1",),
        evidence_ids=("ev1", "ev2"), rationale="termination defect",
        field=SkillField.TERMINATION,
    )
    cluster = EvidenceCluster(
        ClusterKey(skill.skill_id, SkillField.TERMINATION, "termination_conflict", "all", "objects",
                   skill.version),
    )
    cluster.items.append(ClusterItem("event", "episode", attribution, mismatch))
    return cluster


def _make_rollout(args, manifest, run_dir, run_id):
    """Real Habitat rollout closure: parent vs candidate on one coordinate."""
    indexed = {t.episode_id: t for t in manifest.tasks}

    def rollout(skill, coordinate, stage):
        seed_process_rngs(coordinate.seed)
        env = create_habitat_env(
            "train_validation",
            episode_ids=(coordinate.episode_id,),
            exp_name=cli._habitat_exp_name("gate_smoke", run_id, stage, coordinate.episode_id,
                                           f"s{coordinate.seed}", skill_digest(skill)[:10]),
            resolution=args.resolution,
        )
        artifact = (run_dir / "gate_rollouts" / stage /
                    f"{coordinate.episode_id}_s{coordinate.seed}_{skill_digest(skill)[:10]}.jsonl")
        try:
            seed_habitat_env(env, coordinate.seed)
            frozen = replace(skill, frozen=True)
            runtime = VistaSkillEngine(frozen, evidence_extractor=EvidenceExtractor(),
                                       ledger=BeliefLedger())
            runner = cli._make_runner(
                args, env, frozen, artifact, engine=runtime, goal_grounder=None,
                expected_episode_ids=(coordinate.episode_id,),
                task_coordinates=(indexed[coordinate.episode_id],),
                rollout_seed=coordinate.seed,
            )
            result = runner.run_episode(expected_episode_id=coordinate.episode_id)
        finally:
            env.close()
        return RolloutScore(
            score=composite_task_score(
                task_success=result.task_success, task_progress=result.task_progress,
                invalid_action_ratio=result.invalid_actions / max(1, result.environment_steps),
            ),
            success=bool(result.task_success),
        )

    return rollout


def main() -> int:
    run_dir = Path(tempfile.mkdtemp(prefix="vista_gate_smoke_"))
    config = load_config("configs/vista_p0.json")
    manifest = cli._load_verified_manifest("configs/eb_hab_train_validation_manifest.json")
    run_manifest = manifest.rotate_split(0)
    run_id = "gate_smoke"
    method_model = OpenAICompatibleJsonModel(
        MODEL, base_url=BASE_URL, api_key="EMPTY", temperature=0.0, max_tokens=512, seed=0,
    )
    # Stock RemoteModel reads `remote_url` at import time (cli.main sets it).
    os.environ["remote_url"] = BASE_URL

    args = cli.parse_args([
        "experiment", "--method", "full", "--diagnostic",
        "--executor-base-url", BASE_URL, "--method-model", MODEL,
    ])

    parent = initialize_shared_skill()
    cluster = _termination_cluster(parent)

    gate_seeds = (0, 1, 2)
    selection = run_manifest.coordinates_for("selection")
    coords = cli._paired_selection_coordinates(
        selection, gate_seeds, proxy_budget=2, finalist_budget=3,
    )
    from vista_skill.evaluation import PairedRolloutEvaluator
    evaluator = PairedRolloutEvaluator(coords, _make_rollout(args, manifest, run_dir, run_id))

    gate = CandidateGate(
        BoundedPatchApplier(config.patch),
        _PassingTransitionChecker(),
        evaluator,
        GateConfig(bootstrap_samples=200, alpha=0.05, proxy_episode_budget=2,
                   finalist_episode_budget=3),
    )
    lineage = LineageStore(run_dir / "lineage.jsonl")

    steps = []

    # 1. real patch from vLLM
    patch = JsonBoundedPatchGenerator(method_model).propose(parent, cluster)
    static_errors = gate.applier.validate(parent, patch)
    print(f"[{'PASS' if not static_errors else 'FAIL'}] patch.generate: op={patch.operation.value} "
          f"field={patch.field.value} static_errors={static_errors}")
    steps.append(not static_errors)

    proposed_candidate = None
    if not static_errors:
        proposed_candidate = gate.applier.apply(parent, patch)

    # 2. full gate evaluate (real proxy [+finalist] Habitat rollouts)
    try:
        decision, candidate = gate.evaluate(parent, patch, cluster)
        stages_run = [s.stage for s in decision.stages]
        proxy_ok = any("proxy" in s for s in stages_run)
        print(f"[{'PASS' if proxy_ok else 'FAIL'}] gate.evaluate: accepted={decision.accepted} "
              f"stages={stages_run}")
        steps.append(proxy_ok)
    except Exception as exc:
        print(f"[FAIL] gate.evaluate raised: {exc!r}")
        decision, candidate = None, None
        steps.append(False)

    # 3. lineage + snapshot
    try:
        if decision is not None:
            lineage.append(parent=parent, candidate=proposed_candidate, patch=patch,
                           decision=decision, protocol={
                               "run_id": run_id,
                               "split_rotation_index": 0,
                               "split_sha256": run_manifest.split.digest(),
                               "manifest_sha256": manifest.digest,
                           })
        n_snap = len(lineage.accepted_snapshots.load_all())
        print(f"[PASS] lineage.append: records={len(lineage.records())} snapshots={n_snap}")
        steps.append(True)
    except Exception as exc:
        print(f"[FAIL] lineage.append raised: {exc!r}")
        steps.append(False)

    # 4. post-hoc audit (real Habitat re-evaluation of the snapshot)
    try:
        from vista_skill.update_audit import RotatedAuditPlan
        full = make_rotated_audit_plan(manifest, rotation_index=0, rollout_seeds=(101,))
        audit_coords = full.coordinates[:2]
        # Tiny but COMPLETE plan (all metadata fields kept, only coordinates
        # shrunk to 2) so the audit coverage check passes -- the real CLI uses
        # the full 20-task x N-seed plan.
        audit_plan = RotatedAuditPlan(
            rotation_index=full.rotation_index,
            manifest_sha256=full.manifest_sha256,
            split_sha256=full.split_sha256,
            rollout_seeds=full.rollout_seeds,
            coordinates=audit_coords,
            task_ids=full.task_ids,
        )
        audit_evaluator = cli._make_audit_evaluator(
            args, manifest, audit_coords, run_dir, run_id=run_id
        )
        audit = run_rotated_update_audit(
            lineage.accepted_snapshots, audit_evaluator,
            plan=audit_plan, output_path=run_dir / "update_audit.json",
        )
        print(f"[PASS] update_audit: reliability={dict(audit.reliability)}")
        steps.append(True)
    except Exception as exc:
        print(f"[FAIL] update_audit raised: {exc!r}")
        steps.append(False)

    print(f"\nteacher billing: { {k: vars(v) for k, v in method_model.usage.items()} }")
    print(f"artifacts under: {run_dir}")
    print(f"{sum(steps)}/{len(steps)} gate-rollover checks passed")
    return 0 if all(steps) else 1


if __name__ == "__main__":
    sys.exit(main())
