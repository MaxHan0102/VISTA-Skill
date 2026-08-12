from __future__ import annotations

import json

import pytest

from vista_skill.integrations.embodiedbench.manifest import build_manifest
from vista_skill.protocol import DataSplit, load_experiment_manifest


def test_stock_manifest_has_fixed_60_20_20_split(tmp_path) -> None:
    dataset = "EmbodiedBench/embodiedbench/envs/eb_habitat/datasets/train_validation.pickle"
    payload = build_manifest(dataset)
    output = tmp_path / "manifest.json"
    output.write_text(json.dumps(payload), encoding="utf-8")
    manifest = load_experiment_manifest(output)
    assert len(manifest.tasks) == 100
    assert len(manifest.split.acquisition) == 60
    assert len(manifest.split.selection) == 20
    assert len(manifest.split.audit) == 20
    assert manifest.coordinates_for("selection")[0].episode_id == "60"


def test_manifest_split_rotation_is_deterministic_and_disjoint() -> None:
    manifest = load_experiment_manifest(
        "configs/eb_hab_train_validation_manifest.json"
    )
    first = manifest.rotate_split(0)
    second = manifest.rotate_split(1)
    third = manifest.rotate_split(2)
    assert first.split == manifest.split
    assert second.split.acquisition[:3] == ("20", "21", "22")
    assert second.split.selection[:3] == ("80", "81", "82")
    assert second.split.audit[:3] == ("0", "1", "2")
    assert third.split.acquisition[:3] == ("40", "41", "42")
    assert len({item.split.digest() for item in (first, second, third)}) == 3


def test_split_rejects_duplicates_within_a_role() -> None:
    with pytest.raises(ValueError, match="duplicate task IDs"):
        DataSplit(("a", "a"), ("b",), ("c",), ())


def test_manifest_execution_preserves_signed_split_order() -> None:
    manifest = load_experiment_manifest(
        "configs/eb_hab_train_validation_manifest.json"
    )
    reversed_selection = tuple(reversed(manifest.split.selection))
    changed = type(manifest)(
        manifest_id=manifest.manifest_id,
        dataset=manifest.dataset,
        dataset_sha256=manifest.dataset_sha256,
        tasks=manifest.tasks,
        split=DataSplit(
            manifest.split.acquisition,
            reversed_selection,
            manifest.split.audit,
            manifest.split.final_test,
        ),
    )
    assert tuple(
        item.task_id for item in changed.coordinates_for("selection")
    ) == reversed_selection
