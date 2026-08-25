"""Gated dev/selection analysis for strict no-feedback temporal evidence."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from scripts.phase3a_evidence_guard_audit import (
    _aggregate_contradiction,
    _attribution_metrics,
    _cache,
    _context,
    _deduplicate,
    _evaluate_arm,
    _evidence,
    _expected,
    _key,
    _oracle_evidence,
    _predicate_metrics,
    _request,
)
from vista_skill.attribution import CreditAssigner
from vista_skill.config import load_config
from vista_skill.mismatch import compare_transitions
from vista_skill.schemas import PredicateEvidence, TruthValue


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pair-strict", type=Path, required=True)
    parser.add_argument("--temporal-strict", type=Path, required=True)
    parser.add_argument("--images-feedback", type=Path, required=True)
    parser.add_argument("--images-only", type=Path, required=True)
    parser.add_argument(
        "--config", default="configs/vista_fault_repair_fullsel_p10.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _strict_cache(path: Path) -> dict[str, dict[str, Any]]:
    output = _cache(path)
    for sample_id, row in output.items():
        if not all(not value for value in row["isolation_audit"].values()):
            raise ValueError(f"strict isolation audit failed for {sample_id}")
    return output


def _evaluate_strict(
    arm: str,
    samples: Sequence[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    assigner: CreditAssigner,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predicate_rows = []
    rows = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        request = _request(sample)
        evidence = _deduplicate(
            tuple(_evidence(item) for item in cache[sample_id].get("evidence", []))
        )
        routing_evidence = tuple(
            item for item in evidence if item.key.name != "task_complete"
        )
        oracle = {
            _key(item["key"]): TruthValue(str(item["value"]))
            for item in sample["oracle"]["post"]
        }
        expected = tuple(
            item
            for item in (_expected(raw) for raw in sample["transition"]["expected_delta"])
            if item.key.name != "task_complete"
        )
        oracle_items = tuple(
            item
            for item in _oracle_evidence(sample, request)
            if item.key.name != "task_complete"
        )
        base = {
            "sample_id": sample_id,
            "action_type": request.action.action_type,
            "oracle": oracle,
            "evidence": evidence,
        }
        predicate_rows.append(base)
        expected_cases = [("clean", expected)]
        if request.action.action_type == "pick":
            for case_name, marker, faulty_value in (
                (
                    "counterfactual_effect_pick_inversion",
                    "effect_pick_holds_target_category",
                    TruthValue.FALSE,
                ),
                (
                    "counterfactual_constraint_pick_multihold",
                    "constraint_pick_occupies_gripper",
                    TruthValue.TRUE,
                ),
            ):
                changed = tuple(
                    replace(item, after=faulty_value) if marker in item.source_id else item
                    for item in expected
                )
                if changed != expected:
                    expected_cases.append((case_name, changed))
        for case_name, case_expected in expected_cases:
            mismatches = compare_transitions(case_expected, routing_evidence)
            attribution = assigner.assign(mismatches, _context(sample))
            gold_mismatches = compare_transitions(case_expected, oracle_items)
            gold_attribution = assigner.assign(gold_mismatches, _context(sample))
            rows.append(
                {
                    **base,
                    "sample_id": f"{sample_id}:{case_name}",
                    "source_sample_id": sample_id,
                    "case_type": case_name,
                    "expected": case_expected,
                    "mismatches": mismatches,
                    "gold_target": gold_attribution.target.value,
                    "gold_field": (
                        "none"
                        if gold_attribution.field is None
                        else gold_attribution.field.value
                    ),
                    "pred_target": attribution.target.value,
                    "pred_field": (
                        "none" if attribution.field is None else attribution.field.value
                    ),
                }
            )
    return (
        {
            "arm": arm,
            "sample_count": len(predicate_rows),
            "propagation_case_count": len(rows),
            "predicate_state": _predicate_metrics(
                predicate_rows, include_task_complete=False
            ),
            "predicate_all_including_task_complete": _predicate_metrics(
                predicate_rows, include_task_complete=True
            ),
            "contradiction": _aggregate_contradiction(rows),
            "attribution_vs_oracle_routing": _attribution_metrics(rows),
        },
        rows,
    )


def _cost(samples: Iterable[dict[str, Any]], cache: dict[str, dict[str, Any]]) -> dict[str, int]:
    return {
        key: sum(int(cache[str(sample["sample_id"])]["usage"][key]) for sample in samples)
        for key in ("calls", "prompt_tokens", "completion_tokens")
    }


def _checks(
    temporal: dict[str, Any],
    pair: dict[str, Any],
    e0: dict[str, Any],
    temporal_cost: dict[str, int],
    pair_cost: dict[str, int],
) -> dict[str, bool]:
    predicate = temporal["predicate_state"]
    pair_predicate = pair["predicate_state"]
    contradiction = temporal["contradiction"]
    attribution = temporal["attribution_vs_oracle_routing"]
    e0_attr = e0["attribution_vs_oracle_routing"]
    temporal_tokens = temporal_cost["prompt_tokens"] + temporal_cost["completion_tokens"]
    pair_tokens = pair_cost["prompt_tokens"] + pair_cost["completion_tokens"]
    return {
        "predicate_macro_f1_gain_ge_0_10": (
            predicate["macro_f1"] >= pair_predicate["macro_f1"] + 0.10
        ),
        "accepted_predicate_precision_ge_0_90": predicate["precision"] >= 0.90,
        "query_coverage_ge_0_50": predicate["coverage"] >= 0.50,
        "contradiction_recall_ge_0_50": contradiction["recall"] >= 0.50,
        "clean_false_skill_update_rate_lt_0_05": (
            attribution["clean_false_skill_update_rate"] < 0.05
        ),
        "target_macro_f1_within_0_05_of_e0": (
            attribution["target_macro_f1"] >= e0_attr["target_macro_f1"] - 0.05
        ),
        "tokens_le_1_5x_pair": temporal_tokens <= 1.5 * pair_tokens,
    }


def _tier_counts(
    samples: Iterable[dict[str, Any]], cache: dict[str, dict[str, Any]]
) -> dict[str, int]:
    output: dict[str, int] = {}
    for sample in samples:
        for observation in cache[str(sample["sample_id"])]["observations"]:
            tier = str(observation.get("tier", "malformed"))
            output[tier] = output.get(tier, 0) + 1
    return dict(sorted(output.items()))


def main() -> int:
    args = _args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    dataset = json.loads(args.dataset.read_text())
    pair_cache = _strict_cache(args.pair_strict)
    temporal_cache = _strict_cache(args.temporal_strict)
    fb_cache = _cache(args.images_feedback)
    images_only_cache = _cache(args.images_only)
    selected_samples = [
        row for row in dataset["samples"] if row["split"] in {"dev", "selection"}
    ]
    selected_ids = {str(row["sample_id"]) for row in selected_samples}
    if set(pair_cache) != selected_ids or set(temporal_cache) != selected_ids:
        raise ValueError("strict cache does not exactly cover frozen dev+selection")
    config = load_config(args.config)
    assigner = CreditAssigner(config=config.attribution)
    by_split = {
        split: [row for row in selected_samples if row["split"] == split]
        for split in ("dev", "selection")
    }

    dev_pair, _ = _evaluate_strict("T0_pair_strict", by_split["dev"], pair_cache, assigner)
    dev_temporal, _ = _evaluate_strict(
        "T1_temporal_strict", by_split["dev"], temporal_cache, assigner
    )
    dev_e0, _ = _evaluate_arm(
        "E0_current", by_split["dev"], fb_cache, images_only_cache, assigner
    )
    dev_cost = {
        "T0_pair_strict": _cost(by_split["dev"], pair_cache),
        "T1_temporal_strict": _cost(by_split["dev"], temporal_cache),
    }
    dev_checks = _checks(
        dev_temporal,
        dev_pair,
        dev_e0,
        dev_cost["T1_temporal_strict"],
        dev_cost["T0_pair_strict"],
    )
    dev_pass = all(dev_checks.values())

    selection_opened = dev_pass
    selection_metrics = None
    selection_cost = None
    selection_checks = None
    if selection_opened:
        selection_pair, _ = _evaluate_strict(
            "T0_pair_strict", by_split["selection"], pair_cache, assigner
        )
        selection_temporal, _ = _evaluate_strict(
            "T1_temporal_strict", by_split["selection"], temporal_cache, assigner
        )
        selection_e0, _ = _evaluate_arm(
            "E0_current",
            by_split["selection"],
            fb_cache,
            images_only_cache,
            assigner,
        )
        selection_cost = {
            "T0_pair_strict": _cost(by_split["selection"], pair_cache),
            "T1_temporal_strict": _cost(by_split["selection"], temporal_cache),
        }
        selection_checks = _checks(
            selection_temporal,
            selection_pair,
            selection_e0,
            selection_cost["T1_temporal_strict"],
            selection_cost["T0_pair_strict"],
        )
        selection_metrics = {
            "T0_pair_strict": selection_pair,
            "T1_temporal_strict": selection_temporal,
            "E0_current": selection_e0,
        }
    decision = (
        "go"
        if selection_checks is not None and all(selection_checks.values())
        else "no_go_selection"
        if selection_opened
        else "no_go_dev"
    )
    result = {
        "analysis_type": "phase3b_strict_temporal_evidence_gated_audit",
        "dataset": str(args.dataset.resolve()),
        "dataset_id": dataset["dataset_id"],
        "audit_samples_read": 0,
        "dev_metrics": {
            "T0_pair_strict": dev_pair,
            "T1_temporal_strict": dev_temporal,
            "E0_current": dev_e0,
        },
        "dev_cost": dev_cost,
        "dev_tier_counts": _tier_counts(by_split["dev"], temporal_cache),
        "dev_go_checks": dev_checks,
        "selection_opened": selection_opened,
        "selection_metrics": selection_metrics,
        "selection_cost": selection_cost,
        "selection_tier_counts": (
            _tier_counts(by_split["selection"], temporal_cache)
            if selection_opened
            else None
        ),
        "selection_go_checks": selection_checks,
        "decision": decision,
        "information_isolation": {
            "raw_feedback": False,
            "last_action_success": False,
            "pre_ledger": False,
            "expected_transition": False,
            "oracle": False,
            "skill": False,
            "action_prior_authorizes_persistent_write": False,
        },
        "limitations": [
            "Oracle labels define action-local query truth, not human visual-inferability labels.",
            "Counterfactual faults modify only expected transitions; observed RGB and oracle state stay fixed.",
            "Phase3A audit is not read; a new scene/task-disjoint audit is conditional on both Phase3B gates.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "decision": decision,
                "dev_go_checks": dev_checks,
                "selection_opened": selection_opened,
                "selection_go_checks": selection_checks,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
