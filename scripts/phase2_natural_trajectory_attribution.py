"""Episode-level Qwen reflection audit on natural fault-injected rollouts."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from vista_skill.baselines import EmbodiSkillRoute, EpisodeSummary
from vista_skill.fault_injection import FaultType, inject_skill_fault
from vista_skill.metrics import macro_f1
from vista_skill.models import JsonTrajectoryTeacher, OpenAICompatibleJsonModel
from vista_skill.skills import initialize_shared_skill, render_skill


FAULT_PROFILES = {
    "constraint_pick_multihold": {
        "fault_type": FaultType.CONSTRAINT_PICK_MULTIHOLD,
        "rule_suffix": ":constraint_pick_occupies_gripper",
        "evidence_after": "false",
        "field": "constraint",
    },
    "effect_pick_inversion": {
        "fault_type": FaultType.EFFECT_PICK_INVERSION,
        "rule_suffix": ":effect_pick_holds_target_category",
        "evidence_after": "true",
        "field": "effect",
    },
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--fault", choices=tuple(FAULT_PROFILES), default="constraint_pick_multihold"
    )
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_episodes(path: Path, fault: str) -> list[dict[str, object]]:
    profile = FAULT_PROFILES[fault]
    transitions: dict[str, list[dict[str, object]]] = defaultdict(list)
    results: dict[str, dict[str, object]] = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        payload = record.get("payload", {})
        episode_id = str(payload.get("episode_id", ""))
        if record.get("event_type") == "transition":
            transitions[episode_id].append(payload)
        elif record.get("event_type") == "episode_result":
            results[episode_id] = payload

    episodes = []
    for episode_id, result in sorted(results.items(), key=lambda item: int(item[0])):
        positive = any(
            mismatch.get("kind") == "contradiction"
            and (mismatch.get("expected") or {}).get("source_id", "").endswith(
                str(profile["rule_suffix"])
            )
            and (mismatch.get("evidence") or {}).get("after")
            == profile["evidence_after"]
            and float((mismatch.get("evidence") or {}).get("confidence", 0.0)) >= 0.75
            for event in transitions[episode_id]
            for mismatch in event.get("mismatches", [])
        )
        episodes.append(
            {
                "episode_id": episode_id,
                "result": result,
                "gold_target": "skill_update" if positive else "abstain",
                "gold_field": profile["field"] if positive else "none",
            }
        )
    return episodes


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    gold = [str(row["gold_target"]) for row in rows]
    predicted = [str(row["predicted_target"]) for row in rows]
    tp = sum(g == p == "skill_update" for g, p in zip(gold, predicted))
    fp = sum(g == "abstain" and p == "skill_update" for g, p in zip(gold, predicted))
    fn = sum(g == "skill_update" and p != "skill_update" for g, p in zip(gold, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "target_macro_f1": macro_f1(gold, predicted),
        "field_macro_f1": macro_f1(
            [str(row["gold_field"]) for row in rows],
            [str(row["predicted_field"]) for row in rows],
        ),
        "skill_update_precision": precision,
        "skill_update_recall": recall,
        "skill_update_f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "confusion": dict(Counter(f"{g}->{p}" for g, p in zip(gold, predicted))),
    }


def main() -> int:
    args = _args()
    episodes = _load_episodes(args.events, args.fault)
    if not episodes:
        raise ValueError("no completed episodes in event artifact")
    profile = FAULT_PROFILES[args.fault]
    faulty = inject_skill_fault(initialize_shared_skill(), profile["fault_type"])
    seed_reports = []
    for seed in range(args.seeds):
        model = OpenAICompatibleJsonModel(
            args.model,
            base_url=args.base_url,
            api_key="EMPTY",
            temperature=0.0,
            max_tokens=512,
            seed=seed,
        )
        teacher = JsonTrajectoryTeacher(model)
        rows = []
        for index, case in enumerate(episodes):
            result = case["result"]
            assert isinstance(result, dict)
            error = None
            try:
                reflection = teacher.reflect(
                    EpisodeSummary(
                        episode_id=str(case["episode_id"]),
                        instruction=str(result["instruction"]),
                        success=bool(float(result["task_success"])),
                        trajectory=tuple(str(x) for x in result.get("trajectory", [])),
                        current_skill=render_skill(faulty),
                        failure_reason=str(result.get("failure_reason", "")),
                    )
                )
                persistent = reflection.route is not EmbodiSkillRoute.FAIL_EXECUTION
                predicted_target = "skill_update" if persistent else "abstain"
                predicted_field = (
                    reflection.target_field.value
                    if persistent and reflection.target_field is not None
                    else "none"
                )
                route = reflection.route.value
                confidence = reflection.confidence
            except Exception as exc:
                predicted_target = "invalid"
                predicted_field = "invalid"
                route = "invalid"
                confidence = 0.0
                error = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "episode_id": case["episode_id"],
                    "gold_target": case["gold_target"],
                    "gold_field": case["gold_field"],
                    "predicted_target": predicted_target,
                    "predicted_field": predicted_field,
                    "route": route,
                    "confidence": confidence,
                    "error": error,
                }
            )
            print(
                f"seed={seed} episode={index + 1}/{len(episodes)} "
                f"gold={case['gold_target']} pred={predicted_target}/{predicted_field}"
                + (f" error={error}" if error else "")
            )
        seed_reports.append(
            {
                "seed": seed,
                "metrics": _summarize(rows),
                "rows": rows,
                "usage": {
                    purpose: {
                        "calls": usage.calls,
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                    }
                    for purpose, usage in model.usage.items()
                },
            }
        )
    output = {
        "analysis_type": "natural_fault_injection_trajectory_reflection_weak_audit",
        "source_events": str(args.events),
        "fault": args.fault,
        "gold_positive_rule_suffix": profile["rule_suffix"],
        "model": args.model,
        "base_url": args.base_url,
        "episode_count": len(episodes),
        "gold_positive_count": sum(
            case["gold_target"] == "skill_update" for case in episodes
        ),
        "seed_reports": seed_reports,
        "limitation": (
            "The injected fault defines positive episodes before outcomes, but controls "
            "are not independently annotated for unrelated natural Skill defects."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    for report in seed_reports:
        print(f"seed={report['seed']} metrics={report['metrics']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
