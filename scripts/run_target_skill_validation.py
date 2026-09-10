#!/usr/bin/env python3
"""Run matched frozen No/Static/Target Skill validation arms."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from vista_skill.skills import load_skill_artifact_record


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--output-dir",
        default="running/Phase4/target_skill_oracle_v1/validation_common_sense_seed0",
    )
    parser.add_argument("--subset", default="common_sense")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hab-python",
        default=sys.executable,
        help="Python executable containing the EB-Habitat dependencies.",
    )
    parser.add_argument(
        "--nav-python",
        default="/root/miniconda3/envs/max_embench_nav/bin/python",
        help="Python executable containing AI2-THOR/EB-Navigation dependencies.",
    )
    parser.add_argument(
        "--hab-skill",
        default="running/Phase4/target_skill_oracle_v1/artifacts/target_habitat_skill_v1.json",
    )
    parser.add_argument(
        "--nav-skill",
        default="running/Phase4/target_skill_oracle_v1/artifacts/target_navigation_skill_v1.json",
    )
    parser.add_argument(
        "--envs",
        default="eb-hab,eb-nav",
        help="Comma-separated subset of eb-hab,eb-nav.",
    )
    parser.add_argument(
        "--arms",
        default="no_skill,static_shared_skill,target_skill",
        help="Comma-separated subset of no_skill,static_shared_skill,target_skill.",
    )
    return parser.parse_args(argv)


def _run_arm(
    *,
    env_name: str,
    arm: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, object]:
    config = "configs/vista_nav.json" if env_name == "eb-nav" else "configs/vista_p0.json"
    skill_path = Path(args.nav_skill if env_name == "eb-nav" else args.hab_skill)
    mode = arm if arm != "target_skill" else "frozen_skill"
    arm_dir = output_dir / env_name.replace("-", "_") / args.subset / arm
    arm_dir.mkdir(parents=True, exist_ok=False)
    event_path = arm_dir / "events.jsonl"
    environment_python = args.nav_python if env_name == "eb-nav" else args.hab_python
    command = [
        environment_python,
        "-m",
        "vista_skill.integrations.embodiedbench.cli",
        "evaluate",
        "--env",
        env_name,
        "--eval-set",
        args.subset,
        "--stage",
        "official_test",
        "--mode",
        mode,
        "--config",
        config,
        "--model-name",
        "Qwen/Qwen3-VL-8B-Instruct",
        "--model-type",
        "remote",
        "--executor-base-url",
        args.base_url,
        "--resolution",
        "500",
        "--tp",
        "1",
        "--seed",
        str(args.seed),
        "--diagnostic",
        "--max-episodes",
        str(args.episodes),
        "--output",
        str(event_path),
    ]
    if arm == "target_skill":
        command.extend(["--skill", str(skill_path)])
    env = dict(os.environ)
    env["PYTHONPATH"] = "EmbodiedBench:."
    env["OPENAI_API_KEY"] = "EMPTY"
    started = time.time()
    log_path = arm_dir / "console.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND " + json.dumps(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record = {
        "env": env_name,
        "arm": arm,
        "command": command,
        "started_unix": started,
        "finished_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "exit_code": completed.returncode,
        "event_path": str(event_path),
        "summary_path": str(event_path.with_suffix(".summary.json")),
        "console_log": str(log_path),
    }
    (arm_dir / "runtime_manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[{env_name}/{arm}] exit={completed.returncode} "
        f"elapsed={record['elapsed_seconds']:.1f}s",
        flush=True,
    )
    if completed.returncode:
        raise RuntimeError(f"validation arm failed; inspect {log_path}")
    return record


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    allowed_envs = {"eb-hab", "eb-nav"}
    allowed_arms = {"no_skill", "static_shared_skill", "target_skill"}
    envs = tuple(item.strip() for item in args.envs.split(",") if item.strip())
    arms = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    if not envs or not set(envs) <= allowed_envs:
        raise ValueError(f"unsupported --envs: {envs}")
    if not arms or not set(arms) <= allowed_arms:
        raise ValueError(f"unsupported --arms: {arms}")
    for skill_path in (Path(args.hab_skill), Path(args.nav_skill)):
        load_skill_artifact_record(skill_path, require_frozen=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    records = []
    for env_name in envs:
        for arm in arms:
            records.append(
                _run_arm(
                    env_name=env_name,
                    arm=arm,
                    args=args,
                    output_dir=output_dir,
                )
            )
    manifest = {
        "protocol": "vista_target_skill_heldout_subset_validation_v1_2026_08_27",
        "diagnostic": True,
        "source_split": "official_test/base",
        "evaluation_split": f"official_test/{args.subset}",
        "source_trajectory_contamination": True,
        "evaluation_subset_used_in_synthesis": False,
        "note": (
            "Evaluation subset is disjoint from the base trajectories used to "
            "write the target Skills, but the artifacts remain post-hoc diagnostics."
        ),
        "base_url": args.base_url,
        "episodes_per_arm": args.episodes,
        "seed": args.seed,
        "started_unix": started,
        "finished_unix": time.time(),
        "arms": records,
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
