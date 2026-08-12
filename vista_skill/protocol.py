from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class DataSplit:
    acquisition: tuple[str, ...]
    selection: tuple[str, ...]
    audit: tuple[str, ...]
    final_test: tuple[str, ...]

    def __post_init__(self) -> None:
        roles = {
            "acquisition": set(self.acquisition),
            "selection": set(self.selection),
            "audit": set(self.audit),
            "final_test": set(self.final_test),
        }
        names = tuple(roles)
        duplicates = [
            name
            for name in names
            if len(getattr(self, name)) != len(roles[name])
        ]
        if duplicates:
            raise ValueError(
                "duplicate task IDs within split roles: " + ", ".join(duplicates)
            )
        overlaps = []
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                shared = roles[left] & roles[right]
                if shared:
                    overlaps.append(f"{left}/{right}: {sorted(shared)}")
        if overlaps:
            raise ValueError("data split leakage: " + "; ".join(overlaps))

    def digest(self) -> str:
        payload = json.dumps(
            {
                "acquisition": self.acquisition,
                "selection": self.selection,
                "audit": self.audit,
                "final_test": self.final_test,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def ids_for(self, stage: str) -> tuple[str, ...]:
        try:
            return getattr(self, stage)
        except AttributeError as error:
            raise ValueError(f"unknown experiment stage: {stage}") from error


@dataclass(frozen=True)
class TaskCoordinate:
    episode_id: str
    task_id: str
    subgroup: str
    dataset_index: int


@dataclass(frozen=True)
class ExperimentManifest:
    manifest_id: str
    dataset: str
    dataset_sha256: str
    tasks: tuple[TaskCoordinate, ...]
    split: DataSplit

    def __post_init__(self) -> None:
        task_ids = tuple(item.task_id for item in self.tasks)
        episode_ids = tuple(item.episode_id for item in self.tasks)
        if len(set(task_ids)) != len(task_ids) or len(set(episode_ids)) != len(episode_ids):
            raise ValueError("manifest task and episode IDs must be unique")
        known = set(task_ids)
        assigned = set(
            (*self.split.acquisition, *self.split.selection, *self.split.audit, *self.split.final_test)
        )
        if known != assigned:
            raise ValueError("manifest tasks and split assignments differ")

    def coordinates_for(self, stage: str) -> tuple[TaskCoordinate, ...]:
        ordered_ids = self.split.ids_for(stage)
        indexed = {item.task_id: item for item in self.tasks}
        return tuple(indexed[task_id] for task_id in ordered_ids)

    def rotate_split(self, rotation_index: int) -> "ExperimentManifest":
        """Rotate 60/20/20 roles by one 20-task block per evolution run."""
        if rotation_index < 0:
            raise ValueError("rotation_index must be non-negative")
        acquisition_size = len(self.split.acquisition)
        selection_size = len(self.split.selection)
        audit_size = len(self.split.audit)
        if selection_size < 1 or selection_size != audit_size:
            raise ValueError("split rotation requires equal non-empty selection and audit roles")
        ordered_ids = tuple(item.task_id for item in self.tasks)
        assigned_count = acquisition_size + selection_size + audit_size
        if assigned_count != len(ordered_ids) or self.split.final_test:
            raise ValueError("split rotation requires all manifest tasks in acquisition/selection/audit")
        offset = (rotation_index * selection_size) % len(ordered_ids)
        rotated_ids = ordered_ids[offset:] + ordered_ids[:offset]
        split = deterministic_split(
            rotated_ids,
            acquisition_size=acquisition_size,
            selection_size=selection_size,
            audit_size=audit_size,
        )
        return ExperimentManifest(
            manifest_id=f"{self.manifest_id}:rotation:{rotation_index}",
            dataset=self.dataset,
            dataset_sha256=self.dataset_sha256,
            tasks=self.tasks,
            split=split,
        )

    @property
    def digest(self) -> str:
        return manifest_digest(
            {
                "manifest_id": self.manifest_id,
                "dataset": self.dataset,
                "dataset_sha256": self.dataset_sha256,
                "tasks": [vars(item) for item in self.tasks],
                "split": {
                    "acquisition": self.split.acquisition,
                    "selection": self.split.selection,
                    "audit": self.split.audit,
                    "final_test": self.split.final_test,
                },
            }
        )


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    split = raw["split"]
    return ExperimentManifest(
        manifest_id=str(raw["manifest_id"]),
        dataset=str(raw["dataset"]),
        dataset_sha256=str(raw["dataset_sha256"]),
        tasks=tuple(
            TaskCoordinate(
                episode_id=str(item["episode_id"]),
                task_id=str(item["task_id"]),
                subgroup=str(item["subgroup"]),
                dataset_index=int(item["dataset_index"]),
            )
            for item in raw["tasks"]
        ),
        split=DataSplit(
            acquisition=tuple(str(item) for item in split["acquisition"]),
            selection=tuple(str(item) for item in split["selection"]),
            audit=tuple(str(item) for item in split["audit"]),
            final_test=tuple(str(item) for item in split.get("final_test", [])),
        ),
    )


def deterministic_split(
    task_ids: Sequence[str],
    *,
    acquisition_size: int = 60,
    selection_size: int = 20,
    audit_size: int = 20,
) -> DataSplit:
    required = acquisition_size + selection_size + audit_size
    if len(task_ids) < required:
        raise ValueError(f"need at least {required} task ids, got {len(task_ids)}")
    unique = tuple(dict.fromkeys(task_ids))
    if len(unique) != len(task_ids):
        raise ValueError("task ids must be unique before splitting")
    first = acquisition_size
    second = first + selection_size
    third = second + audit_size
    return DataSplit(
        acquisition=unique[:first],
        selection=unique[first:second],
        audit=unique[second:third],
        final_test=unique[third:],
    )


def manifest_digest(values: Mapping[str, object] | Iterable[object]) -> str:
    payload = json.dumps(values, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
