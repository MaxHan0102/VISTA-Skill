"""Collect fixed primitive-action stress transitions with Habitat state labels."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from vista_skill.action_schema import parse_action_call
from vista_skill.artifacts import JsonlArtifactWriter
from vista_skill.belief import BeliefLedger
from vista_skill.evidence import EvidenceExtractor
from vista_skill.integrations.embodiedbench.environment import (
    create_habitat_env,
    seed_habitat_env,
    seed_process_rngs,
)
from vista_skill.integrations.embodiedbench.runner import HabitatRolloutRunner
from vista_skill.integrations.embodiedbench.state_oracle import HabitatStateOracle
from vista_skill.pipeline import VistaSkillEngine
from vista_skill.protocol import load_experiment_manifest
from vista_skill.skills import initialize_shared_skill


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="configs/eb_hab_train_validation_manifest.json"
    )
    parser.add_argument("--start-index", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--exp-name", required=True)
    return parser.parse_args()


class ScriptedStressPlanner:
    """Prediction-blind fixed action schedule over the public action catalog."""

    output_json_error = 0

    def __init__(self, env: Any) -> None:
        self.env = env
        self.planner_steps = 0
        self._episode_index = -1
        self._position = 0
        self._sequences = self._build_sequences()

    def _build_sequences(self) -> tuple[tuple[int, ...], ...]:
        calls = [
            parse_action_call(index, raw, self.env.language_skill_set[index])
            for index, raw in enumerate(self.env.skill_set)
        ]
        by_type: dict[str, list[Any]] = {}
        for call in calls:
            by_type.setdefault(call.action_type, []).append(call)
        if not all(by_type.get(name) for name in ("nav", "pick", "place", "open", "close")):
            raise RuntimeError("public action catalog lacks a required stress action type")
        sequences = []
        for opened in by_type["open"]:
            if not opened.arguments:
                continue
            target = opened.arguments[0]
            nav = next(
                (item for item in by_type["nav"] if item.arguments and item.arguments[0] == target),
                None,
            )
            closed = next(
                (item for item in by_type["close"] if item.arguments and item.arguments[0] == target),
                None,
            )
            if nav is None or closed is None:
                continue
            # Fixed order: two precondition probes, open before navigation,
            # then successful/repeated container operations.
            sequences.append(
                (
                    by_type["place"][0].action_id,
                    by_type["pick"][0].action_id,
                    opened.action_id,
                    nav.action_id,
                    opened.action_id,
                    opened.action_id,
                    closed.action_id,
                    closed.action_id,
                )
            )
        if not sequences:
            raise RuntimeError("could not pair nav/open/close actions by public target")
        return tuple(sequences)

    def reset(self) -> None:
        self._episode_index += 1
        self._position = 0
        self.planner_steps = 0

    def act(self, observation: str, instruction: str):
        del observation, instruction
        sequence = self._sequences[self._episode_index % len(self._sequences)]
        if self._position >= len(sequence):
            return -2, "fixed Phase3A stress schedule exhausted"
        action = sequence[self._position]
        self._position += 1
        self.planner_steps += 1
        return action, "fixed prediction-blind Phase3A primitive stress schedule"

    def update_info(self, info):
        del info


def main() -> int:
    args = _args()
    for target in (args.output, args.run_manifest):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
    manifest = load_experiment_manifest(args.manifest)
    acquisition = manifest.coordinates_for("acquisition")
    selected = acquisition[args.start_index : args.start_index + args.episodes]
    if len(selected) != args.episodes:
        raise ValueError("requested stress episode range exceeds acquisition split")
    episode_ids = tuple(item.episode_id for item in selected)
    seed_process_rngs(args.seed)
    env = create_habitat_env(
        "train_validation",
        episode_ids=episode_ids,
        exp_name=args.exp_name,
        resolution=args.resolution,
    )
    started = time.time()
    try:
        seed_habitat_env(env, args.seed)
        engine = VistaSkillEngine(
            initialize_shared_skill(),
            evidence_extractor=EvidenceExtractor(),
            ledger=BeliefLedger(),
        )
        runner = HabitatRolloutRunner(
            env,
            ScriptedStressPlanner(env),
            engine,
            JsonlArtifactWriter(args.output),
            expected_episode_ids=episode_ids,
            task_coordinates=selected,
            state_oracle=HabitatStateOracle(),
        )
        results = runner.run()
    finally:
        env.close()
    action_counts: dict[str, int] = {}
    transition_count = 0
    for line in args.output.read_text().splitlines():
        record = json.loads(line)
        if record["event_type"] == "transition":
            transition_count += 1
            action = str(record["payload"]["action"]["action_type"])
            action_counts[action] = action_counts.get(action, 0) + 1
    payload = {
        "analysis_type": "phase3a_scripted_stress_collection",
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": manifest.digest,
        "seed": args.seed,
        "episode_ids": episode_ids,
        "episode_count": len(results),
        "transition_count": transition_count,
        "action_counts": action_counts,
        "task_successes": sum(item.task_success > 0 for item in results),
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "output": str(args.output),
        "policy": (
            "fixed public-catalog schedule: invalid place/pick/open probes, then "
            "nav/open/reopen/close/reclose; no state-adaptive action selection"
        ),
    }
    args.run_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.run_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
