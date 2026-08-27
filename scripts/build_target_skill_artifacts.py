#!/usr/bin/env python3
"""Build digest-checked post-hoc target Skills for the oracle diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from vista_skill.skills import (
    load_skill_artifact_record,
    render_skill,
    save_skill_artifact,
    skill_digest,
    target_habitat_skill_v1,
    target_navigation_skill_v1,
)


SOURCE_PATHS = {
    "eb-hab": (
        "running/pilot/eval/no_skill.jsonl",
        "running/pilot/eval/static_shared_skill.jsonl",
    ),
    "eb-nav": (
        "running/vista_skill/nav_official/base/no_skill/events.jsonl",
        "running/vista_skill/nav_official/base/static_shared_skill/events.jsonl",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="running/target_skill_oracle_v1/artifacts",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    builders = {
        "eb-hab": target_habitat_skill_v1,
        "eb-nav": target_navigation_skill_v1,
    }
    artifacts: dict[str, object] = {}
    for env_name, builder in builders.items():
        skill = builder()
        source_records = []
        for raw_path in SOURCE_PATHS[env_name]:
            path = Path(raw_path)
            if not path.is_file():
                raise FileNotFoundError(f"missing trajectory source: {path}")
            source_records.append(
                {"path": str(path), "sha256": _sha256(path)}
            )
        protocol = {
            "protocol": "vista_target_skill_posthoc_oracle_v1_2026_08_27",
            "method": "human_trajectory_synthesis_skip_8b_evolution",
            "env": env_name,
            "executor_model": "Qwen/Qwen3-VL-8B-Instruct",
            "executor_model_type": "remote",
            "executor_temperature": 0.0,
            "tensor_parallel": 1,
            "resolution": 500,
            "frozen": True,
            "diagnostic": True,
            "source_split": "official_test/base",
            "source_arms": ["no_skill", "static_shared_skill"],
            "source_trajectory_contamination": True,
            "automatic_evolution_calls": 0,
            "teacher_calls": 0,
            "trajectory_sources": source_records,
            "claim_scope": (
                "post-hoc target/reference only; not a controlled or held-out "
                "evolution result"
            ),
        }
        filename = (
            "target_habitat_skill_v1.json"
            if env_name == "eb-hab"
            else "target_navigation_skill_v1.json"
        )
        path = output_dir / filename
        save_skill_artifact(path, skill, protocol=protocol)
        record = load_skill_artifact_record(path, require_frozen=True)
        artifacts[env_name] = {
            "path": str(path),
            "skill_id": skill.skill_id,
            "skill_sha256": skill_digest(skill),
            "artifact_sha256": record.artifact_sha256,
            "rendered_word_count": len(render_skill(skill).split()),
            "rendered_skill": render_skill(skill),
            "trajectory_sources": source_records,
        }

    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": "vista_target_skill_posthoc_oracle_v1_2026_08_27",
                "artifacts": artifacts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifacts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
