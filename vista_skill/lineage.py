from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from vista_skill.evolution import GateDecision
from vista_skill.schemas import SkillPatch, SkillSpec, dataclass_to_dict
from vista_skill.skills import skill_digest


@dataclass(frozen=True)
class LineageRecord:
    timestamp: str
    accepted: bool
    parent_version: int
    candidate_version: int | None
    parent_hash: str
    candidate_hash: str | None
    patch: SkillPatch
    decision: GateDecision
    protocol: Mapping[str, Any]
    accepted_snapshot_id: str | None = None
    proposal_snapshot_id: str | None = None


class LineageStore:
    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        accepted_snapshot_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.path = Path(path)
        from vista_skill.update_audit import AcceptedUpdateSnapshotStore

        snapshot_dir = (
            Path(accepted_snapshot_dir)
            if accepted_snapshot_dir is not None
            else self.path.parent / "update_proposals"
        )
        self.accepted_snapshots = AcceptedUpdateSnapshotStore(snapshot_dir)

    def append(
        self,
        *,
        parent: SkillSpec,
        candidate: SkillSpec | None,
        patch: SkillPatch,
        decision: GateDecision,
        protocol: Mapping[str, Any] | None = None,
    ) -> LineageRecord:
        if decision.accepted and candidate is None:
            raise ValueError("accepted lineage decisions require a candidate snapshot")
        snapshot_id = None
        if candidate is not None:
            snapshot = self.accepted_snapshots.save(
                parent=parent,
                candidate=candidate,
                patch=patch,
                promoted=decision.accepted,
                protocol=protocol or {},
            )
            snapshot_id = snapshot.snapshot_id
        record = LineageRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            accepted=decision.accepted,
            parent_version=parent.version,
            candidate_version=None if candidate is None else candidate.version,
            parent_hash=skill_digest(parent),
            candidate_hash=None if candidate is None else skill_digest(candidate),
            patch=patch,
            decision=decision,
            protocol=protocol or {},
            accepted_snapshot_id=snapshot_id if decision.accepted else None,
            proposal_snapshot_id=snapshot_id,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dataclass_to_dict(record), sort_keys=True) + "\n")
        return record

    def records(self) -> tuple[dict[str, Any], ...]:
        if not self.path.exists():
            return ()
        records: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return tuple(records)
