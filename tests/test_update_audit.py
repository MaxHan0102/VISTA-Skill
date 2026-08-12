from __future__ import annotations

import json

import pytest

from vista_skill.evaluation import PairedRolloutEvaluator, RolloutScore
from vista_skill.evolution import GateDecision
from vista_skill.lineage import LineageStore
from vista_skill.metrics import UpdateAudit, update_reliability
from vista_skill.protocol import DataSplit, ExperimentManifest, TaskCoordinate
from vista_skill.schemas import PatchOperation, SkillField, SkillPatch
from vista_skill.skills import initialize_shared_skill, with_field
from vista_skill.update_audit import (
    AcceptedUpdateSnapshotStore,
    make_rotated_audit_plan,
    run_rotated_update_audit,
)


def _manifest() -> ExperimentManifest:
    tasks = tuple(
        TaskCoordinate(
            episode_id=f"episode-{index}",
            task_id=f"task-{index}",
            subgroup="protected" if index % 2 else "base",
            dataset_index=index,
        )
        for index in range(6)
    )
    return ExperimentManifest(
        manifest_id="audit-fixture",
        dataset="fixture.pickle",
        dataset_sha256="0" * 64,
        tasks=tasks,
        split=DataSplit(
            acquisition=("task-0", "task-1"),
            selection=("task-2", "task-3"),
            audit=("task-4", "task-5"),
            final_test=(),
        ),
    )


def _accepted_update():
    parent = initialize_shared_skill()
    candidate = with_field(
        parent,
        SkillField.PROCEDURE,
        (*parent.procedure, "Verify the bound target before acting."),
    )
    patch = SkillPatch(
        patch_id="patch-audit-1",
        skill_id=parent.skill_id,
        parent_version=parent.version,
        field=SkillField.PROCEDURE,
        operation=PatchOperation.APPEND,
        old="",
        new="Verify the bound target before acting.",
        evidence_ids=("evidence-1", "evidence-2"),
        scope="multi-instance tasks",
    )
    return parent, candidate, patch


def _store_snapshot(tmp_path, manifest, plan):
    parent, candidate, patch = _accepted_update()
    store = AcceptedUpdateSnapshotStore(tmp_path / "accepted_updates")
    snapshot = store.save(
        parent=parent,
        candidate=candidate,
        patch=patch,
        protocol={
            "manifest_sha256": manifest.digest,
            "split_rotation_index": plan.rotation_index,
            "split_sha256": plan.split_sha256,
        },
    )
    return store, snapshot


def test_accepted_update_snapshot_round_trip_and_tamper_detection(tmp_path) -> None:
    manifest = _manifest()
    plan = make_rotated_audit_plan(
        manifest, rotation_index=1, rollout_seeds=(101, 102)
    )
    store, snapshot = _store_snapshot(tmp_path, manifest, plan)

    restored = store.load_all()
    assert restored == (snapshot,)
    assert restored[0].parent.version == 0
    assert restored[0].candidate.version == 1

    snapshot_path = store.root / snapshot.snapshot_id / "snapshot.json"
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    payload["field"] = "termination"
    snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot digest mismatch"):
        store.load(snapshot.snapshot_id)


def test_lineage_persists_auditable_accepted_and_rejected_proposals(tmp_path) -> None:
    parent, candidate, patch = _accepted_update()
    lineage = LineageStore(tmp_path / "lineage.jsonl")
    accepted = GateDecision(
        accepted=True,
        reason="passed",
        parent_version=parent.version,
        candidate_version=candidate.version,
        patch_id=patch.patch_id,
        stages=(),
    )
    record = lineage.append(
        parent=parent,
        candidate=candidate,
        patch=patch,
        decision=accepted,
        protocol={"run": "fixture"},
    )
    assert record.accepted_snapshot_id is not None
    assert len(lineage.accepted_snapshots.load_all()) == 1

    rejected = GateDecision(
        accepted=False,
        reason="rejected",
        parent_version=candidate.version,
        candidate_version=None,
        patch_id="patch-rejected",
        stages=(),
    )
    rejected_patch = SkillPatch(
        patch_id="patch-rejected",
        skill_id=candidate.skill_id,
        parent_version=candidate.version,
        field=SkillField.PROCEDURE,
        operation=PatchOperation.APPEND,
        old="",
        new="Rejected statement.",
        evidence_ids=("evidence-3", "evidence-4"),
        scope="fixture",
    )
    rejected_candidate = with_field(
        candidate,
        SkillField.PROCEDURE,
        (*candidate.procedure, "Rejected statement."),
    )
    rejected_record = lineage.append(
        parent=candidate,
        candidate=rejected_candidate,
        patch=rejected_patch,
        decision=rejected,
    )
    assert rejected_record.accepted_snapshot_id is None
    assert rejected_record.proposal_snapshot_id is not None
    snapshots = lineage.accepted_snapshots.load_all()
    assert len(snapshots) == 2
    assert snapshots[0].promoted
    assert not snapshots[1].promoted


def test_rotated_audit_runs_exact_pairs_and_aggregates_task_first(tmp_path) -> None:
    manifest = _manifest()
    plan = make_rotated_audit_plan(
        manifest, rotation_index=1, rollout_seeds=(101, 102)
    )
    store, snapshot = _store_snapshot(tmp_path, manifest, plan)
    candidate_deltas = {
        ("episode-0", 101): 0.4,
        ("episode-0", 102): 0.2,
        ("episode-1", 101): 0.1,
        ("episode-1", 102): 0.1,
    }

    def rollout(skill, coordinate, stage):
        assert stage == "audit"
        delta = candidate_deltas[(coordinate.episode_id, coordinate.seed)]
        return RolloutScore(
            score=0.5 if skill.version == 0 else 0.5 + delta,
            success=skill.version > 0,
        )

    evaluator = PairedRolloutEvaluator({"audit": plan.coordinates}, rollout)
    output = tmp_path / "update_audit.json"
    report = run_rotated_update_audit(
        store, evaluator, plan=plan, output_path=output, epsilon=0.01
    )

    assert report.update_count == 1
    assert report.pair_count == 4
    assert report.audits[0].snapshot_id == snapshot.snapshot_id
    assert report.audits[0].classification == "beneficial"
    assert report.audits[0].task_deltas == pytest.approx(
        {"task-0": 0.3, "task-1": 0.1}
    )
    assert report.audits[0].overall_delta == pytest.approx(0.2)
    assert report.reliability["beneficial_update_precision"] == 1.0
    assert json.loads(output.read_text(encoding="utf-8"))["pair_count"] == 4


class IncompleteEvaluator:
    def evaluate(self, parent, candidate, *, stage, episode_budget):
        return ()


def test_rotated_audit_fails_closed_on_incomplete_coordinates(tmp_path) -> None:
    manifest = _manifest()
    plan = make_rotated_audit_plan(manifest, rotation_index=0, rollout_seeds=(7,))
    store, _ = _store_snapshot(tmp_path, manifest, plan)
    with pytest.raises(ValueError, match="coordinate mismatch"):
        run_rotated_update_audit(store, IncompleteEvaluator(), plan=plan)


def test_rotated_audit_rejects_snapshot_from_another_split(tmp_path) -> None:
    manifest = _manifest()
    first = make_rotated_audit_plan(manifest, rotation_index=0, rollout_seeds=(7,))
    second = make_rotated_audit_plan(manifest, rotation_index=1, rollout_seeds=(7,))
    store, _ = _store_snapshot(tmp_path, manifest, first)
    with pytest.raises(ValueError, match="audit protocol mismatch"):
        run_rotated_update_audit(store, IncompleteEvaluator(), plan=second)


def test_update_reliability_does_not_call_subgroup_regression_beneficial() -> None:
    metrics = update_reliability(
        (
            UpdateAudit("patch-harmful", True, 0.2, {"base": 0.4, "protected": -0.1}),
            UpdateAudit("patch-beneficial", True, 0.1, {"base": 0.1, "protected": 0.1}),
        )
    )
    assert metrics["beneficial_updates"] == 1.0
    assert metrics["harmful_updates"] == 1.0
    assert metrics["beneficial_update_precision"] == 0.5
    assert metrics["harmful_update_rate"] == 0.5


def test_update_reliability_reports_missed_beneficial_rejections() -> None:
    metrics = update_reliability(
        (
            UpdateAudit("accepted", True, 0.1, {"base": 0.1}),
            UpdateAudit("missed", False, 0.2, {"base": 0.2}),
            UpdateAudit("good-reject", False, -0.1, {"base": -0.1}),
        )
    )
    assert metrics["missed_beneficial_update_rate"] == 0.5
