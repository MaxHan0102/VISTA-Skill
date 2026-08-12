from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

from vista_skill.protocol import deterministic_split, manifest_digest


def build_manifest(
    dataset_path: str | Path,
    *,
    split_sizes: tuple[int, int, int] | None = (60, 20, 20),
) -> dict[str, Any]:
    path = Path(dataset_path)
    dataset_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    episodes = raw.get("all_eps", ())
    tasks = []
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise ValueError(f"dataset episode {index} is not a mapping")
        episode_id = str(episode["episode_id"])
        tasks.append(
            {
                "episode_id": episode_id,
                "task_id": episode_id,
                "subgroup": str(episode.get("instruct_id", "unknown")),
                "dataset_index": index,
            }
        )
    task_ids = tuple(item["task_id"] for item in tasks)
    if split_sizes is None:
        split_payload = {
            "acquisition": [],
            "selection": [],
            "audit": [],
            "final_test": list(task_ids),
        }
    else:
        split = deterministic_split(
            task_ids,
            acquisition_size=split_sizes[0],
            selection_size=split_sizes[1],
            audit_size=split_sizes[2],
        )
        split_payload = {
            "acquisition": list(split.acquisition),
            "selection": list(split.selection),
            "audit": list(split.audit),
            "final_test": list(split.final_test),
        }
    payload = {
        "manifest_id": "eb_hab_train_validation_60_20_20_v1",
        "dataset": path.name,
        "dataset_sha256": dataset_hash,
        "tasks": tasks,
        "split": split_payload,
    }
    payload["manifest_sha256"] = manifest_digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic EB-Hab task manifest.")
    parser.add_argument("dataset")
    parser.add_argument("output")
    args = parser.parse_args()
    payload = build_manifest(args.dataset)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
