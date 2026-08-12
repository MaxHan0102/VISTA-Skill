from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from vista_skill.evaluation import EpisodeCoordinate
from vista_skill.evolution import PairedEpisodeScore, PairedEvaluator
from vista_skill.metrics import UpdateAudit, update_reliability
from vista_skill.protocol import ExperimentManifest
from vista_skill.schemas import SkillPatch, SkillSpec, dataclass_to_dict
from vista_skill.skills import (
    SkillArtifact,
    canonical_protocol,
    load_skill_artifact_record,
    save_skill_artifact,
    skill_digest,
)


SNAPSHOT_SCHEMA_VERSION = "1"
UPDATE_AUDIT_SCHEMA_VERSION = "1"
_SAFE_SNAPSHOT_ID = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class AcceptedUpdateSnapshot:
    snapshot_id: str
    patch_id: str
    field: str
    evidence_ids: tuple[str, ...]
    promoted: bool
    parent: SkillSpec
    candidate: SkillSpec
    parent_artifact_sha256: str
    candidate_artifact_sha256: str
    protocol: Mapping[str, Any]


class AcceptedUpdateSnapshotStore:
    """Digest-checked parent/candidate versions for every auditable proposal."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.index_path = self.root / "index.jsonl"

    def save(
        self,
        *,
        parent: SkillSpec,
        candidate: SkillSpec,
        patch: SkillPatch,
        promoted: bool = True,
        protocol: Mapping[str, Any] | None = None,
    ) -> AcceptedUpdateSnapshot:
        if patch.parent_version != parent.version:
            raise ValueError("snapshot patch does not match parent version")
        if candidate.parent_version != parent.version:
            raise ValueError("snapshot candidate does not descend from parent")
        if candidate.skill_id != parent.skill_id or patch.skill_id != parent.skill_id:
            raise ValueError("snapshot skill IDs do not match")
        if candidate.version <= parent.version:
            raise ValueError("snapshot candidate version must advance the parent")

        snapshot_id = _snapshot_id(parent, candidate, patch.patch_id)
        if snapshot_id in self.snapshot_ids():
            raise FileExistsError(f"accepted update snapshot already exists: {snapshot_id}")
        snapshot_dir = self.root / snapshot_id
        if snapshot_dir.exists():
            raise FileExistsError(f"accepted update snapshot directory exists: {snapshot_dir}")
        snapshot_dir.mkdir(parents=True)

        source_protocol = canonical_protocol(protocol)
        parent_path = snapshot_dir / "parent_skill.json"
        candidate_path = snapshot_dir / "candidate_skill.json"
        save_skill_artifact(
            parent_path,
            parent,
            protocol={
                "snapshot_id": snapshot_id,
                "snapshot_role": "parent",
                "source_protocol": source_protocol,
            },
        )
        save_skill_artifact(
            candidate_path,
            candidate,
            protocol={
                "snapshot_id": snapshot_id,
                "snapshot_role": "candidate",
                "source_protocol": source_protocol,
            },
        )
        parent_artifact = load_skill_artifact_record(parent_path)
        candidate_artifact = load_skill_artifact_record(candidate_path)
        manifest = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "patch_id": patch.patch_id,
            "field": patch.field.value,
            "evidence_ids": list(patch.evidence_ids),
            "promoted": bool(promoted),
            "parent_version": parent.version,
            "candidate_version": candidate.version,
            "parent_skill_sha256": skill_digest(parent),
            "candidate_skill_sha256": skill_digest(candidate),
            "parent_artifact_sha256": parent_artifact.artifact_sha256,
            "candidate_artifact_sha256": candidate_artifact.artifact_sha256,
            "protocol": source_protocol,
        }
        manifest["snapshot_sha256"] = _mapping_digest(manifest)
        _write_json_exclusive(snapshot_dir / "snapshot.json", manifest)
        self.root.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "snapshot_id": snapshot_id,
                        "patch_id": patch.patch_id,
                        "parent_version": parent.version,
                        "candidate_version": candidate.version,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        return self.load(snapshot_id)

    def snapshot_ids(self) -> tuple[str, ...]:
        if not self.index_path.exists():
            return ()
        identifiers: list[str] = []
        with self.index_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                snapshot_id = str(record["snapshot_id"])
                _validate_snapshot_id(snapshot_id)
                identifiers.append(snapshot_id)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("accepted update snapshot index contains duplicate IDs")
        return tuple(identifiers)

    def load_all(self) -> tuple[AcceptedUpdateSnapshot, ...]:
        return tuple(self.load(snapshot_id) for snapshot_id in self.snapshot_ids())

    def load(self, snapshot_id: str) -> AcceptedUpdateSnapshot:
        _validate_snapshot_id(snapshot_id)
        snapshot_dir = self.root / snapshot_id
        manifest = json.loads((snapshot_dir / "snapshot.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported accepted update snapshot schema")
        supplied_digest = manifest.get("snapshot_sha256")
        unsigned = dict(manifest)
        unsigned.pop("snapshot_sha256", None)
        if supplied_digest != _mapping_digest(unsigned):
            raise ValueError("accepted update snapshot digest mismatch")
        if manifest.get("snapshot_id") != snapshot_id:
            raise ValueError("accepted update snapshot ID mismatch")

        parent_artifact = load_skill_artifact_record(snapshot_dir / "parent_skill.json")
        candidate_artifact = load_skill_artifact_record(snapshot_dir / "candidate_skill.json")
        protocol = manifest.get("protocol")
        if not isinstance(protocol, Mapping):
            raise ValueError("accepted update snapshot protocol must be an object")
        self._validate_artifact(
            parent_artifact, manifest, snapshot_id, "parent", protocol
        )
        self._validate_artifact(
            candidate_artifact, manifest, snapshot_id, "candidate", protocol
        )
        parent = parent_artifact.skill
        candidate = candidate_artifact.skill
        if candidate.parent_version != parent.version:
            raise ValueError("accepted update snapshot lineage mismatch")
        if parent.skill_id != candidate.skill_id:
            raise ValueError("accepted update snapshot skill ID mismatch")
        return AcceptedUpdateSnapshot(
            snapshot_id=snapshot_id,
            patch_id=str(manifest["patch_id"]),
            field=str(manifest["field"]),
            evidence_ids=tuple(str(item) for item in manifest["evidence_ids"]),
            promoted=bool(manifest["promoted"]),
            parent=parent,
            candidate=candidate,
            parent_artifact_sha256=parent_artifact.artifact_sha256,
            candidate_artifact_sha256=candidate_artifact.artifact_sha256,
            protocol=dict(protocol),
        )

    @staticmethod
    def _validate_artifact(
        artifact: SkillArtifact,
        manifest: Mapping[str, Any],
        snapshot_id: str,
        role: str,
        protocol: Mapping[str, Any],
    ) -> None:
        expected_protocol = {
            "snapshot_id": snapshot_id,
            "snapshot_role": role,
            "source_protocol": dict(protocol),
        }
        if artifact.protocol != expected_protocol:
            raise ValueError(f"accepted update {role} artifact protocol mismatch")
        if artifact.artifact_sha256 != manifest.get(f"{role}_artifact_sha256"):
            raise ValueError(f"accepted update {role} artifact digest mismatch")
        if skill_digest(artifact.skill) != manifest.get(f"{role}_skill_sha256"):
            raise ValueError(f"accepted update {role} skill digest mismatch")
        if artifact.skill.version != int(manifest[f"{role}_version"]):
            raise ValueError(f"accepted update {role} version mismatch")


@dataclass(frozen=True)
class RotatedAuditPlan:
    rotation_index: int
    manifest_sha256: str
    split_sha256: str
    rollout_seeds: tuple[int, ...]
    coordinates: tuple[EpisodeCoordinate, ...]
    task_ids: Mapping[str, str]


def make_rotated_audit_plan(
    manifest: ExperimentManifest,
    *,
    rotation_index: int,
    rollout_seeds: Sequence[int],
) -> RotatedAuditPlan:
    seeds = tuple(int(seed) for seed in rollout_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("audit rollout seeds must be non-empty and unique")
    rotated = manifest.rotate_split(rotation_index)
    audit_tasks = rotated.coordinates_for("audit")
    if not audit_tasks:
        raise ValueError("rotated audit split is empty")
    coordinates = tuple(
        EpisodeCoordinate(item.episode_id, seed, item.subgroup)
        for item in audit_tasks
        for seed in seeds
    )
    return RotatedAuditPlan(
        rotation_index=rotation_index,
        manifest_sha256=manifest.digest,
        split_sha256=rotated.split.digest(),
        rollout_seeds=seeds,
        coordinates=coordinates,
        task_ids={item.episode_id: item.task_id for item in audit_tasks},
    )


@dataclass(frozen=True)
class AuditPairDelta:
    task_id: str
    episode_id: str
    seed: int
    subgroup: str
    parent_score: float
    candidate_score: float
    delta: float
    parent_success: bool | None
    candidate_success: bool | None


@dataclass(frozen=True)
class UpdateAuditRecord:
    snapshot_id: str
    patch_id: str
    field: str
    parent_version: int
    candidate_version: int
    parent_skill_sha256: str
    candidate_skill_sha256: str
    promoted: bool
    classification: str
    overall_delta: float
    subgroup_deltas: Mapping[str, float]
    task_deltas: Mapping[str, float]
    pairs: tuple[AuditPairDelta, ...]

    def metric(self) -> UpdateAudit:
        return UpdateAudit(
            patch_id=self.patch_id,
            promoted=self.promoted,
            overall_delta=self.overall_delta,
            subgroup_deltas=self.subgroup_deltas,
        )


@dataclass(frozen=True)
class UpdateAuditReport:
    schema_version: str
    rotation_index: int
    manifest_sha256: str
    split_sha256: str
    rollout_seeds: tuple[int, ...]
    epsilon: float
    update_count: int
    pair_count: int
    audits: tuple[UpdateAuditRecord, ...]
    reliability: Mapping[str, float]


def run_rotated_update_audit(
    snapshots: AcceptedUpdateSnapshotStore | str | os.PathLike[str],
    evaluator: PairedEvaluator,
    *,
    plan: RotatedAuditPlan,
    output_path: str | os.PathLike[str] | None = None,
    epsilon: float = 0.0,
) -> UpdateAuditReport:
    if epsilon < 0.0:
        raise ValueError("update audit epsilon must be non-negative")
    store = (
        snapshots
        if isinstance(snapshots, AcceptedUpdateSnapshotStore)
        else AcceptedUpdateSnapshotStore(snapshots)
    )
    accepted = store.load_all()
    audits = tuple(
        _audit_snapshot(snapshot, evaluator, plan, epsilon) for snapshot in accepted
    )
    report = UpdateAuditReport(
        schema_version=UPDATE_AUDIT_SCHEMA_VERSION,
        rotation_index=plan.rotation_index,
        manifest_sha256=plan.manifest_sha256,
        split_sha256=plan.split_sha256,
        rollout_seeds=plan.rollout_seeds,
        epsilon=epsilon,
        update_count=len(audits),
        pair_count=sum(len(item.pairs) for item in audits),
        audits=audits,
        reliability=update_reliability(
            tuple(item.metric() for item in audits), epsilon=epsilon
        ),
    )
    if output_path is not None:
        _write_json_exclusive(Path(output_path), dataclass_to_dict(report))
    return report


def _audit_snapshot(
    snapshot: AcceptedUpdateSnapshot,
    evaluator: PairedEvaluator,
    plan: RotatedAuditPlan,
    epsilon: float,
) -> UpdateAuditRecord:
    _validate_snapshot_protocol(snapshot, plan)
    scores = tuple(
        evaluator.evaluate(
            snapshot.parent,
            snapshot.candidate,
            stage="audit",
            episode_budget=len(plan.coordinates),
        )
    )
    expected = {
        (coordinate.episode_id, coordinate.seed): coordinate
        for coordinate in plan.coordinates
    }
    actual = {(item.episode_id, item.seed): item for item in scores}
    if len(actual) != len(scores):
        raise ValueError("update audit returned duplicate task/seed pairs")
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(
            f"update audit coordinate mismatch; missing={missing}, extra={extra}"
        )
    ordered_scores = tuple(actual[key] for key in expected)
    pairs: list[AuditPairDelta] = []
    for score in ordered_scores:
        coordinate = expected[(score.episode_id, score.seed)]
        if score.subgroup != coordinate.subgroup:
            raise ValueError("update audit subgroup differs from frozen manifest")
        pairs.append(
            AuditPairDelta(
                task_id=plan.task_ids[score.episode_id],
                episode_id=score.episode_id,
                seed=score.seed,
                subgroup=score.subgroup,
                parent_score=score.parent_score,
                candidate_score=score.candidate_score,
                delta=score.candidate_score - score.parent_score,
                parent_success=score.parent_success,
                candidate_success=score.candidate_success,
            )
        )
    task_deltas = _mean_by(pairs, key=lambda item: item.task_id)
    subgroup_deltas = _subgroup_task_means(pairs)
    overall_delta = sum(task_deltas.values()) / len(task_deltas)
    classification = _classify(overall_delta, subgroup_deltas, epsilon)
    return UpdateAuditRecord(
        snapshot_id=snapshot.snapshot_id,
        patch_id=snapshot.patch_id,
        field=snapshot.field,
        parent_version=snapshot.parent.version,
        candidate_version=snapshot.candidate.version,
        parent_skill_sha256=skill_digest(snapshot.parent),
        candidate_skill_sha256=skill_digest(snapshot.candidate),
        promoted=snapshot.promoted,
        classification=classification,
        overall_delta=overall_delta,
        subgroup_deltas=subgroup_deltas,
        task_deltas=task_deltas,
        pairs=tuple(pairs),
    )


def _validate_snapshot_protocol(
    snapshot: AcceptedUpdateSnapshot, plan: RotatedAuditPlan
) -> None:
    required = {
        "split_rotation_index": plan.rotation_index,
        "split_sha256": plan.split_sha256,
        "manifest_sha256": plan.manifest_sha256,
    }
    mismatches = [
        key for key, expected in required.items()
        if snapshot.protocol.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "accepted update snapshot audit protocol mismatch: "
            + ", ".join(mismatches)
        )


def _subgroup_task_means(pairs: Sequence[AuditPairDelta]) -> dict[str, float]:
    grouped: dict[str, list[AuditPairDelta]] = {}
    for item in pairs:
        grouped.setdefault(item.subgroup, []).append(item)
    values: dict[str, float] = {}
    for subgroup, items in grouped.items():
        task_means = _mean_by(items, key=lambda item: item.task_id)
        values[subgroup] = sum(task_means.values()) / len(task_means)
    return values


def _mean_by(items: Sequence[AuditPairDelta], *, key) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for item in items:
        grouped.setdefault(str(key(item)), []).append(item.delta)
    return {
        name: sum(values) / len(values) for name, values in grouped.items()
    }


def _classify(
    overall_delta: float,
    subgroup_deltas: Mapping[str, float],
    epsilon: float,
) -> str:
    if overall_delta < -epsilon or any(
        delta < -epsilon for delta in subgroup_deltas.values()
    ):
        return "harmful"
    if overall_delta > epsilon:
        return "beneficial"
    return "neutral"


def _snapshot_id(parent: SkillSpec, candidate: SkillSpec, patch_id: str) -> str:
    patch_digest = hashlib.sha256(patch_id.encode("utf-8")).hexdigest()[:12]
    return (
        f"v{parent.version}-v{candidate.version}-{patch_digest}-"
        f"{skill_digest(candidate)[:12]}"
    )


def _validate_snapshot_id(snapshot_id: str) -> None:
    if not snapshot_id or not _SAFE_SNAPSHOT_ID.fullmatch(snapshot_id):
        raise ValueError("invalid accepted update snapshot ID")


def _mapping_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
