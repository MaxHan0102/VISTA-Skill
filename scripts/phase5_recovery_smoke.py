"""Real Habitat admission smoke with a scripted planner; no learning claim."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path

from vista_skill.artifacts import JsonlArtifactWriter
from vista_skill.integrations.embodiedbench.environment import create_habitat_development_env, seed_habitat_env, seed_process_rngs
from vista_skill.integrations.embodiedbench.runner import HabitatRolloutRunner
from vista_skill.pipeline import VistaSkillEngine
from vista_skill.schemas import SkillField, TemporalSkillRule, TruthValue
from vista_skill.skills import interface_only_shared_skill
from vista_skill.temporal import TemporalRuleMonitor


REPO = Path(__file__).resolve().parents[1]


class ScriptedPlanner:
    planner_steps = 0
    output_json_error = 0

    def __init__(self, actions):
        self.actions = iter(actions)

    def reset(self):
        pass

    def act(self, image, instruction):
        self.planner_steps += 1
        return next(self.actions, -2), "predeclared action-admission smoke; no model"

    def update_info(self, info):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = REPO / "running/Phase5/phase5_p57_recovery_20260907/development.pickle"
    os.chdir(REPO / "EmbodiedBench")
    results = {}
    for release in ("action_or_evidence", "target_evidence"):
        artifact = output / f"{release}.jsonl"
        if artifact.exists():
            raise FileExistsError(artifact)
        seed_process_rngs(570907)
        env = create_habitat_development_env(dataset, episode_ids=("p57_acquisition_00",),
            resolution=500, exp_name=f"p57_scripted_smoke/{release}")
        try:
            seed_habitat_env(env, 570907)
            nav = env.skill_set.index(("nav", ["fridge_push_point"]))
            pick = env.skill_set.index(("pick_ball", ["robot_0"]))
            rule = TemporalSkillRule("smoke_rule", SkillField.PROCEDURE, "pick", False,
                "near({arg0})", TruthValue.FALSE, "pick", ("nav",), recovery_release=release)
            skill = replace(interface_only_shared_skill(), temporal_rules=(rule,))
            runner = HabitatRolloutRunner(env, ScriptedPlanner([nav, pick, nav, pick]),
                VistaSkillEngine(skill), JsonlArtifactWriter(artifact), temporal_monitor=TemporalRuleMonitor())
            results[release] = asdict(runner.run_episode(expected_episode_id="p57_acquisition_00"))
        finally:
            env.close()
    payload = {"scripted_mechanism_only": True, "model_calls": 0,
               "natural_discovery": False, "claim_eligible": False, "results": results}
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
