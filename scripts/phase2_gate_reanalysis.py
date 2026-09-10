"""Offline Phase-2 reanalysis of gate and independent audit artifacts.

This is a diagnostic, not a threshold-selection tool.  Affected groups are
defined from the injected fault before looking at outcomes.  Every remaining
audited task is treated as protected.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

from vista_skill.evolution import bootstrap_lcb


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("running/Phase1/fault_repair_e8p_constraint/full/seed_0"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "EmbodiedBench/embodiedbench/envs/eb_habitat/datasets/"
            "train_validation.pickle"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("running/Phase2/phase2_gate_reanalysis.json")
    )
    parser.add_argument(
        "--fault",
        choices=("auto", "constraint_pick_multihold", "effect_pick_inversion"),
        default="auto",
        help="Fault-specific affected-task definition; auto reads lineage protocol.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--protected-margin", type=float, default=0.05)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _contains_goal_predicate(node: Any, predicate: str) -> bool:
    if isinstance(node, str):
        return node.strip().startswith(f"{predicate}(")
    if isinstance(node, dict):
        return any(_contains_goal_predicate(value, predicate) for value in node.values())
    if isinstance(node, (list, tuple)):
        return any(_contains_goal_predicate(value, predicate) for value in node)
    return False


def _task_is_affected(episode: dict[str, Any], fault: str) -> bool:
    """Return membership in the outcome-independent, fault-specific group."""
    goal = episode.get("goal_preds", {})
    if fault == "constraint_pick_multihold":
        object_variables = goal.get("inputs", []) if isinstance(goal, dict) else []
        return len(object_variables) >= 2
    if fault == "effect_pick_inversion":
        # Relocation and removal goals both contain on_top, possibly nested
        # under NAND.  They require pick; fridge and robot_at tasks do not.
        return _contains_goal_predicate(goal, "on_top")
    raise ValueError(f"unsupported fault profile: {fault}")


def _affected_definition(fault: str) -> str:
    if fault == "constraint_pick_multihold":
        return "goal has at least two object variables"
    if fault == "effect_pick_inversion":
        return "goal expression recursively contains on_top(...) and requires pick"
    raise ValueError(f"unsupported fault profile: {fault}")


def _resolve_fault(requested: str, lineage: list[dict[str, Any]]) -> str:
    recorded = {
        record.get("protocol", {}).get("skill_fault")
        for record in lineage
        if record.get("protocol", {}).get("skill_fault")
    }
    if requested != "auto":
        if recorded and recorded != {requested}:
            raise ValueError(
                f"requested fault {requested!r} conflicts with lineage: {sorted(recorded)}"
            )
        return requested
    if len(recorded) != 1:
        raise ValueError(f"cannot infer one fault from lineage: {sorted(recorded)}")
    fault = next(iter(recorded))
    if fault not in {"constraint_pick_multihold", "effect_pick_inversion"}:
        raise ValueError(f"unsupported inferred fault profile: {fault}")
    return fault


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def _lcb(
    values: Iterable[float], *, alpha: float, samples: int, seed: int
) -> float | None:
    items = list(values)
    if not items:
        return None
    return bootstrap_lcb(items, alpha=alpha, samples=samples, seed=seed)


def _stage_metrics(record: dict[str, Any], name: str) -> dict[str, Any] | None:
    for stage in record["decision"]["stages"]:
        if stage["stage"] == name:
            return {"passed": stage["passed"], **stage.get("metrics", {})}
    return None


def _rule_after(record: dict[str, Any], rule_id: str) -> str | None:
    for rule in record["patch"].get("prediction_rules", []):
        if rule["rule_id"] == rule_id:
            return str(rule["after"])
    return None


def _candidate_report(
    lineage: dict[str, Any],
    audit: dict[str, Any],
    affected_tasks: set[str],
    *,
    alpha: float,
    samples: int,
    protected_margin: float,
    seed: int,
    fault: str,
    affected_definition: str,
) -> dict[str, Any]:
    task_deltas = {str(k): float(v) for k, v in audit["task_deltas"].items()}
    affected = [delta for task, delta in task_deltas.items() if task in affected_tasks]
    protected = [delta for task, delta in task_deltas.items() if task not in affected_tasks]
    global_values = list(task_deltas.values())
    global_lcb = _lcb(global_values, alpha=alpha, samples=samples, seed=seed)
    affected_lcb = _lcb(affected, alpha=alpha, samples=samples, seed=seed + 1000)
    protected_lcb = _lcb(protected, alpha=alpha, samples=samples, seed=seed + 2000)
    semantic_policy_pass = bool(
        affected_lcb is not None
        and protected_lcb is not None
        and affected_lcb > 0.0
        and protected_lcb > -protected_margin
    )
    place_rule_after = _rule_after(lineage, "constraint_place_frees_gripper")
    return {
        "candidate_sha256": audit["candidate_skill_sha256"],
        "patch_id": audit["patch_id"],
        "recorded_gate_accepted": bool(lineage["accepted"]),
        "selection_proxy": _stage_metrics(lineage, "paired_proxy"),
        "selection_finalist": _stage_metrics(lineage, "paired_finalist"),
        "audit_classification_epsilon_0": audit["classification"],
        "audit_global": {
            "n_tasks": len(global_values),
            "mean_delta": _mean(global_values),
            "bootstrap_lcb": global_lcb,
        },
        "audit_affected": {
            "definition": affected_definition,
            "n_tasks": len(affected),
            "mean_delta": _mean(affected),
            "bootstrap_lcb": affected_lcb,
        },
        "audit_protected": {
            "definition": "all remaining tasks",
            "n_tasks": len(protected),
            "mean_delta": _mean(protected),
            "bootstrap_lcb": protected_lcb,
            "noninferiority_margin": -protected_margin,
        },
        "diagnostic_affected_plus_protected_policy_pass": semantic_policy_pass,
        "compiled_rule_check": {
            "target_fault": fault,
            "pick_rule_after": _rule_after(
                lineage, "constraint_pick_occupies_gripper"
            ),
            "effect_pick_rule_after": _rule_after(
                lineage, "effect_pick_holds_target_category"
            ),
            "place_rule_after": place_rule_after,
            "unrelated_place_rule_corrupted": (
                place_rule_after is not None and place_rule_after != "true"
            ),
        },
    }


def main() -> int:
    args = _args()
    lineage = _load_jsonl(args.run_dir / "lineage.jsonl")
    audit = json.loads((args.run_dir / "update_audit.json").read_text())
    fault = _resolve_fault(args.fault, lineage)
    affected_definition = _affected_definition(fault)
    with args.dataset.open("rb") as handle:
        episodes = pickle.load(handle)["all_eps"]
    affected_tasks = {
        str(episode["episode_id"])
        for episode in episodes
        if _task_is_affected(episode, fault)
    }

    audits_by_hash = {
        item["candidate_skill_sha256"]: item for item in audit["audits"]
    }
    candidates = []
    non_materialized = []
    for index, record in enumerate(lineage):
        candidate_hash = record.get("candidate_hash")
        if candidate_hash is None:
            non_materialized.append(
                {
                    "patch_id": record["patch"]["patch_id"],
                    "field": record["patch"]["field"],
                    "reason": record["decision"]["reason"],
                    "recorded_gate_accepted": bool(record["accepted"]),
                }
            )
            continue
        if candidate_hash not in audits_by_hash:
            raise ValueError(f"lineage candidate lacks audit: {candidate_hash}")
        candidates.append(
            _candidate_report(
                record,
                audits_by_hash[candidate_hash],
                affected_tasks,
                alpha=args.alpha,
                samples=args.bootstrap_samples,
                protected_margin=args.protected_margin,
                seed=index,
                fault=fault,
                affected_definition=affected_definition,
            )
        )

    result = {
        "analysis_type": "posthoc_diagnostic_not_threshold_selection",
        "source_run": str(args.run_dir),
        "fault": fault,
        "alpha": args.alpha,
        "bootstrap_samples": args.bootstrap_samples,
        "affected_group_rule": affected_definition,
        "audit_affected_task_ids": sorted(
            {
                task
                for candidate in candidates
                for task in audits_by_hash[candidate["candidate_sha256"]]["task_deltas"]
                if task in affected_tasks
            },
            key=int,
        ),
        "proposal_attempt_count": len(lineage),
        "materialized_candidate_count": len(candidates),
        "unique_candidate_hash_count": len(
            {candidate["candidate_sha256"] for candidate in candidates}
        ),
        "non_materialized_candidate_count": len(non_materialized),
        "non_materialized_candidates": non_materialized,
        "unrelated_rule_corruption_count": sum(
            candidate["compiled_rule_check"]["unrelated_place_rule_corrupted"]
            for candidate in candidates
        ),
        "semantic_policy_pass_count": sum(
            candidate["diagnostic_affected_plus_protected_policy_pass"]
            for candidate in candidates
        ),
        "candidates": candidates,
        "interpretation_guardrail": (
            "Audit outcomes were already observed. Use this result to formulate and "
            "power a preregistered policy, never to select its final threshold."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(
        f"proposals: {len(lineage)}; materialized candidates: {len(candidates)}; "
        f"unique hashes: {result['unique_candidate_hash_count']}; "
        f"non-materialized: {len(non_materialized)}"
    )
    print(
        "unrelated place-rule corruption: "
        f"{result['unrelated_rule_corruption_count']}/{len(candidates)}"
    )
    print(
        "posthoc affected/protected policy passes: "
        f"{result['semantic_policy_pass_count']}/{len(candidates)}"
    )
    for candidate in candidates:
        print(
            candidate["candidate_sha256"][:12],
            f"audit={candidate['audit_classification_epsilon_0']}",
            f"global={candidate['audit_global']['mean_delta']:+.4f}",
            f"affected={candidate['audit_affected']['mean_delta']:+.4f}",
            f"protected={candidate['audit_protected']['mean_delta']:+.4f}",
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
