"""Frozen multi-arm Phase3A evidence/propagation audit against Habitat state."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from vista_skill.attribution import CreditAssigner
from vista_skill.config import load_config
from vista_skill.evidence import EvidenceExtractor
from vista_skill.evidence_guard import (
    EvidenceGuardConfig,
    EvidenceReliabilityGuard,
    GuardMode,
)
from vista_skill.evidence_oracle import (
    NoisyEvidenceProvider,
    evaluate_calibration,
    selective_risk_curve,
)
from vista_skill.metrics import macro_f1, safe_ratio
from vista_skill.mismatch import compare_transitions
from vista_skill.schemas import (
    ActionCall,
    AttributionContext,
    DeltaSource,
    EvidenceRequest,
    EvidenceSource,
    ExpectedChange,
    PredicateEvidence,
    PredicateKey,
    PredicateState,
    SkillField,
    TruthValue,
)


THRESHOLDS = (0.50, 0.60, 0.75, 0.90)
COVERAGES = (0.00, 0.25, 0.50, 0.75)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images-feedback", type=Path, required=True)
    parser.add_argument("--images-only", type=Path, required=True)
    parser.add_argument("--config", default="configs/vista_fault_repair_fullsel_p10.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _key(raw: dict[str, Any]) -> PredicateKey:
    return PredicateKey(str(raw["name"]), tuple(str(x) for x in raw["arguments"]))


def _action(raw: dict[str, Any]) -> ActionCall:
    return ActionCall(
        int(raw["action_id"]),
        str(raw["action_type"]),
        tuple(str(x) for x in raw["arguments"]),
        str(raw["text"]),
        raw.get("raw_action"),
    )


def _state(raw: dict[str, Any]) -> PredicateState:
    return PredicateState(
        _key(raw["key"]),
        TruthValue(str(raw["value"])),
        float(raw["confidence"]),
        str(raw["source"]),
        tuple(str(x) for x in raw["evidence_ids"]),
        int(raw["timestamp"]),
        raw.get("view_id"),
        float(raw.get("coverage", 1.0)),
        float(raw.get("task_relevance", 1.0)),
    )


def _evidence(raw: dict[str, Any]) -> PredicateEvidence:
    return PredicateEvidence(
        _key(raw["key"]),
        TruthValue(str(raw["before"])),
        TruthValue(str(raw["after"])),
        float(raw["confidence"]),
        EvidenceSource(str(raw["source"])),
        str(raw["evidence_id"]),
        int(raw["timestamp"]),
        raw.get("view_id"),
        float(raw.get("coverage", 1.0)),
        float(raw.get("task_relevance", 1.0)),
        str(raw.get("rationale", "")),
    )


def _expected(raw: dict[str, Any]) -> ExpectedChange:
    field = raw.get("skill_field")
    return ExpectedChange(
        key=_key(raw["key"]),
        before=TruthValue(str(raw["before"])),
        after=TruthValue(str(raw["after"])),
        source=DeltaSource(str(raw["source"])),
        source_id=str(raw["source_id"]),
        skill_field=None if field is None else SkillField(str(field)),
    )


def _request(sample: dict[str, Any]) -> EvidenceRequest:
    raw = sample["transition"]
    return EvidenceRequest(
        episode_id=str(raw["episode_id"]),
        step_id=int(raw["step_id"]),
        instruction=str(raw["instruction"]),
        action=_action(raw["action"]),
        pre_image=str(raw["pre_image"]),
        post_image=str(raw["post_image"]),
        feedback=str(raw["feedback"]),
        last_action_success=raw.get("last_action_success"),
        pre_ledger=tuple(_state(x) for x in raw.get("pre_ledger", [])),
        goal_predicates=tuple(_key(x) for x in raw.get("goal_predicates", [])),
    )


def _context(sample: dict[str, Any]) -> AttributionContext:
    transition = sample["transition"]
    raw = transition.get("metadata", {}).get("attribution_context", {})
    return AttributionContext(
        executor_followed_skill=raw.get("executor_followed_skill"),
        stochastic_suspected=bool(raw.get("stochastic_suspected", False)),
        identity_conflict=bool(raw.get("identity_conflict", False)),
        task_pattern=str(raw.get("task_pattern", "general")),
        object_context=str(raw.get("object_context", "general")),
        instruction=str(transition["instruction"]),
        action_type=str(transition["action"]["action_type"]),
        skill_obligations={
            str(key): tuple(str(x) for x in value)
            for key, value in raw.get("skill_obligations", {}).items()
        },
        goal_predicates=tuple(_key(x) for x in transition.get("goal_predicates", [])),
    )


def _cache(path: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            output[str(row["sample_id"])] = row
    return output


class _FixedProvider:
    def __init__(self, items: Sequence[PredicateEvidence]) -> None:
        self.items = tuple(items)

    def extract(self, request: EvidenceRequest) -> tuple[PredicateEvidence, ...]:
        del request
        return self.items


def _oracle_evidence(sample: dict[str, Any], request: EvidenceRequest) -> tuple[PredicateEvidence, ...]:
    pre = {_key(x["key"]): TruthValue(str(x["value"])) for x in sample["oracle"]["pre"]}
    output = []
    for item in sample["oracle"]["post"]:
        key = _key(item["key"])
        value = TruthValue(str(item["value"]))
        if value is TruthValue.UNKNOWN:
            continue
        output.append(
            PredicateEvidence(
                key=key,
                before=pre.get(key, TruthValue.UNKNOWN),
                after=value,
                confidence=1.0,
                source=EvidenceSource.DERIVED_GOAL,
                evidence_id=f"{sample['sample_id']}:state_oracle:{key.render()}",
                timestamp=request.step_id,
                coverage=1.0,
                rationale="evaluation-only Habitat state",
            )
        )
    return tuple(output)


def _deduplicate(items: Iterable[PredicateEvidence]) -> tuple[PredicateEvidence, ...]:
    return EvidenceExtractor._deduplicate(tuple(items))


def _arm_evidence(
    arm: str,
    sample: dict[str, Any],
    request: EvidenceRequest,
    feedback: tuple[PredicateEvidence, ...],
    images_feedback: tuple[PredicateEvidence, ...],
    images_only: tuple[PredicateEvidence, ...],
) -> tuple[PredicateEvidence, ...]:
    if arm == "E0_current":
        return _deduplicate(
            (*feedback, *(x for x in images_feedback if x.confidence >= 0.50))
        )
    if arm == "E1_feedback_only":
        return _deduplicate(feedback)
    if arm == "E5_oracle":
        return _oracle_evidence(sample, request)
    family, confidence, coverage = arm.split(":")
    confidence_f = float(confidence)
    coverage_f = float(coverage)
    if family == "E2_threshold":
        visual = tuple(
            x
            for x in images_feedback
            if x.confidence >= confidence_f and x.coverage >= coverage_f
        )
        return _deduplicate((*feedback, *visual))
    mode = GuardMode.STRICT if family == "E3_strict" else GuardMode.AUTHORITY_AWARE
    guard = EvidenceReliabilityGuard(
        EvidenceGuardConfig(
            mode=mode,
            min_visual_confidence=confidence_f,
            min_visual_coverage=coverage_f,
        )
    )
    # Guard v2 keeps the one-call feedback-conditioned visual extractor used
    # by E0, then applies deterministic parser/semantic reliability rules to
    # its packets.  The separately cached images-only condition remains the
    # independence ablation that motivated this revision on dev.
    return guard.fuse(
        feedback,
        images_feedback,
        request.pre_ledger,
        last_action_success=request.last_action_success,
        action=request.action,
    ).evidence


def _macro_binary(gold: Sequence[str], predicted: Sequence[str]) -> float:
    labels = ("true", "false")
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        scores.append(safe_ratio(2 * precision * recall, precision + recall))
    return sum(scores) / len(scores)


def _predicate_metrics(rows: Sequence[dict[str, Any]], *, include_task_complete: bool) -> dict[str, Any]:
    gold: list[str] = []
    predicted: list[str] = []
    asserted_correct: list[tuple[float, bool]] = []
    probabilities: list[float] = []
    binary_outcomes: list[bool] = []
    by_action: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        pred = {item.key: item for item in row["evidence"]}
        for key, value in row["oracle"].items():
            if value is TruthValue.UNKNOWN:
                continue
            if not include_task_complete and key.name == "task_complete":
                continue
            item = pred.get(key)
            output = TruthValue.UNKNOWN if item is None else item.after
            gold.append(value.value)
            predicted.append(output.value)
            by_action[row["action_type"]].append((value.value, output.value))
            if item is not None and output is not TruthValue.UNKNOWN:
                asserted_correct.append((item.confidence, output is value))
            if item is None or output is TruthValue.UNKNOWN:
                probability = 0.5
            elif output is TruthValue.TRUE:
                probability = item.confidence
            else:
                probability = 1.0 - item.confidence
            probabilities.append(probability)
            binary_outcomes.append(value is TruthValue.TRUE)
    calibration = evaluate_calibration(asserted_correct)
    selective = selective_risk_curve(asserted_correct, thresholds=(0.5, 0.75, 0.9))
    asserted = sum(value != "unknown" for value in predicted)
    correct = sum(g == p for g, p in zip(gold, predicted))
    return {
        "targets": len(gold),
        "asserted": asserted,
        "correct": correct,
        "coverage": safe_ratio(asserted, len(gold)),
        "precision": safe_ratio(correct, asserted),
        "recall": safe_ratio(correct, len(gold)),
        "macro_f1": _macro_binary(gold, predicted),
        "brier": (
            sum((p - float(y)) ** 2 for p, y in zip(probabilities, binary_outcomes))
            / len(probabilities)
            if probabilities
            else 0.0
        ),
        "ece_asserted": calibration.expected_calibration_error,
        "selective_risk": [asdict(item) for item in selective],
        "by_action": {
            action: {
                "targets": len(items),
                "coverage": safe_ratio(sum(p != "unknown" for _, p in items), len(items)),
                "macro_f1": _macro_binary(
                    [g for g, _ in items], [p for _, p in items]
                ),
            }
            for action, items in sorted(by_action.items())
        },
    }


def _contradiction_counts(row: dict[str, Any]) -> dict[str, int]:
    gold = row["oracle"]
    pred = {item.key: item for item in row["evidence"]}
    counts = Counter()
    for expected in row["expected"]:
        truth = gold.get(expected.key, TruthValue.UNKNOWN)
        if truth is TruthValue.UNKNOWN:
            continue
        output = pred.get(expected.key)
        gold_contradiction = truth is not expected.after
        pred_contradiction = (
            output is not None
            and output.after is not TruthValue.UNKNOWN
            and output.after is not expected.after
        )
        if gold_contradiction and pred_contradiction:
            counts["tp"] += 1
        elif gold_contradiction:
            counts["fn"] += 1
        elif pred_contradiction:
            counts["fp"] += 1
        else:
            counts["tn"] += 1
    return dict(counts)


def _aggregate_contradiction(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    counts = Counter()
    for row in rows:
        counts.update(_contradiction_counts(row))
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    return {
        **{key: float(counts[key]) for key in ("tp", "fp", "fn", "tn")},
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
        "false_contradiction_rate": safe_ratio(fp, fp + tn),
    }


def _attribution_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gold_targets = [row["gold_target"] for row in rows]
    pred_targets = [row["pred_target"] for row in rows]
    gold_fields = [row["gold_field"] for row in rows]
    pred_fields = [row["pred_field"] for row in rows]
    controls = [row for row in rows if row["gold_target"] != "skill_update"]
    positives = [row for row in rows if row["gold_target"] == "skill_update"]
    false_updates = [row for row in controls if row["pred_target"] == "skill_update"]
    recovered = [row for row in positives if row["pred_target"] == "skill_update"]
    recurrence = {
        (
            row["action_type"],
            row["pred_field"],
            tuple(sorted(item.key.render() for item in row["mismatches"])),
        )
        for row in false_updates
    }
    return {
        "target_macro_f1": macro_f1(gold_targets, pred_targets),
        "field_macro_f1": macro_f1(gold_fields, pred_fields),
        "clean_control_count": len(controls),
        "false_skill_updates": len(false_updates),
        "clean_false_skill_update_rate": safe_ratio(len(false_updates), len(controls)),
        "skill_update_gold_count": len(positives),
        "skill_update_recall": safe_ratio(len(recovered), len(positives)),
        "false_recurrence_cluster_proxy_count": len(recurrence),
    }


def _evaluate_arm(
    arm: str,
    samples: Sequence[dict[str, Any]],
    fb_cache: dict[str, dict[str, Any]],
    image_cache: dict[str, dict[str, Any]],
    assigner: CreditAssigner,
    *,
    noise: dict[str, float] | None = None,
    noise_seed: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    extractor = EvidenceExtractor()
    predicate_rows = []
    rows = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        request = _request(sample)
        feedback = extractor.extract_feedback(request)
        images_feedback = tuple(
            _evidence(x) for x in fb_cache[sample_id].get("evidence", [])
        )
        images_only = tuple(
            _evidence(x) for x in image_cache[sample_id].get("evidence", [])
        )
        if noise is not None:
            feedback = NoisyEvidenceProvider(
                _FixedProvider(feedback),
                flip_rate=noise.get("feedback_flip", 0.0),
                drop_rate=noise.get("feedback_drop", 0.0),
                seed=noise_seed + 101,
            ).extract(request)
            images_only = NoisyEvidenceProvider(
                _FixedProvider(images_only),
                flip_rate=noise.get("visual_flip", 0.0),
                drop_rate=noise.get("visual_drop", 0.0),
                false_positive_rate=noise.get("false_positive", 0.0),
                false_positive_pool=(PredicateKey("spurious", ("phase3a",)),),
                seed=noise_seed + 202,
            ).extract(request)
            images_feedback = NoisyEvidenceProvider(
                _FixedProvider(images_feedback),
                flip_rate=noise.get("visual_flip", 0.0),
                drop_rate=noise.get("visual_drop", 0.0),
                false_positive_rate=noise.get("false_positive", 0.0),
                false_positive_pool=(PredicateKey("spurious", ("phase3a",)),),
                seed=noise_seed + 303,
            ).extract(request)
        evidence = _arm_evidence(
            arm, sample, request, feedback, images_feedback, images_only
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
            for item in (
                _expected(x) for x in sample["transition"]["expected_delta"]
            )
            if item.key.name != "task_complete"
        )
        oracle_items = _oracle_evidence(sample, request)
        routing_oracle_items = tuple(
            item for item in oracle_items if item.key.name != "task_complete"
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
            for case_name, rule_marker, faulty_value in (
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
                    replace(item, after=faulty_value)
                    if rule_marker in item.source_id
                    else item
                    for item in expected
                )
                if changed != expected:
                    expected_cases.append((case_name, changed))
        for case_name, case_expected in expected_cases:
            mismatches = compare_transitions(case_expected, routing_evidence)
            attribution = assigner.assign(mismatches, _context(sample))
            gold_mismatches = compare_transitions(
                case_expected, routing_oracle_items
            )
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
                    "gold_field": "none" if gold_attribution.field is None else gold_attribution.field.value,
                    "pred_target": attribution.target.value,
                    "pred_field": "none" if attribution.field is None else attribution.field.value,
                }
            )
    metrics = {
        "arm": arm,
        "sample_count": len(predicate_rows),
        "propagation_case_count": len(rows),
        "predicate_state": _predicate_metrics(predicate_rows, include_task_complete=False),
        "predicate_all_including_task_complete": _predicate_metrics(
            predicate_rows, include_task_complete=True
        ),
        "contradiction": _aggregate_contradiction(rows),
        "attribution_vs_oracle_routing": _attribution_metrics(rows),
    }
    return metrics, rows


def _passes(metrics: dict[str, Any], e0: dict[str, Any]) -> bool:
    predicate = metrics["predicate_state"]
    contradiction = metrics["contradiction"]
    attribution = metrics["attribution_vs_oracle_routing"]
    e0_attr = e0["attribution_vs_oracle_routing"]
    return (
        predicate["coverage"] >= 0.50
        and contradiction["precision"] >= 0.90
        and attribution["clean_false_skill_update_rate"] < 0.05
        and attribution["skill_update_recall"]
        >= e0_attr["skill_update_recall"] - 0.10
        and attribution["target_macro_f1"] >= e0_attr["target_macro_f1"]
        and attribution["field_macro_f1"] >= e0_attr["field_macro_f1"]
    )


def _bootstrap_false_contradiction(
    e0_rows: Sequence[dict[str, Any]],
    guard_rows: Sequence[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if len(e0_rows) != len(guard_rows):
        raise ValueError("paired bootstrap rows differ")
    rng = random.Random(seed)
    e0_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    guard_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in e0_rows:
        e0_by_source[str(row["source_sample_id"])].append(row)
    for row in guard_rows:
        guard_by_source[str(row["source_sample_id"])].append(row)
    if set(e0_by_source) != set(guard_by_source):
        raise ValueError("paired bootstrap source coordinates differ")
    sources = tuple(sorted(e0_by_source))

    def rate(
        grouped: dict[str, list[dict[str, Any]]], draws: Sequence[str]
    ) -> float:
        counts = Counter()
        for source in draws:
            for row in grouped[source]:
                counts.update(_contradiction_counts(row))
        return safe_ratio(counts["fp"], counts["fp"] + counts["tn"])

    e0_rate = rate(e0_by_source, sources)
    guard_rate = rate(guard_by_source, sources)
    deltas = []
    for _ in range(samples):
        draw = [sources[rng.randrange(len(sources))] for _ in sources]
        deltas.append(rate(e0_by_source, draw) - rate(guard_by_source, draw))
    deltas.sort()
    lower = deltas[int(0.025 * (len(deltas) - 1))] if deltas else 0.0
    upper = deltas[int(0.975 * (len(deltas) - 1))] if deltas else 0.0
    return {
        "e0_false_contradiction_rate": e0_rate,
        "guard_false_contradiction_rate": guard_rate,
        "absolute_reduction": e0_rate - guard_rate,
        "relative_reduction": safe_ratio(e0_rate - guard_rate, e0_rate),
        "paired_bootstrap_95_ci_lower": lower,
        "paired_bootstrap_95_ci_upper": upper,
    }


def _candidate_arms() -> list[str]:
    return [
        f"{family}:{confidence:.2f}:{coverage:.2f}"
        for family in ("E2_threshold", "E3_strict", "E4_authority")
        for confidence in THRESHOLDS
        for coverage in COVERAGES
    ]


def _guard_candidate_arms() -> list[str]:
    return [
        arm
        for arm in _candidate_arms()
        if arm.startswith(("E3_strict:", "E4_authority:"))
    ]


def _matched_threshold_arm(guard_arm: str) -> str:
    _, confidence, coverage = guard_arm.split(":")
    return f"E2_threshold:{confidence}:{coverage}"


def _rank(metrics: dict[str, Any]) -> tuple[float, ...]:
    contradiction = metrics["contradiction"]
    predicate = metrics["predicate_state"]
    attribution = metrics["attribution_vs_oracle_routing"]
    return (
        -contradiction["false_contradiction_rate"],
        contradiction["precision"],
        predicate["coverage"],
        attribution["target_macro_f1"],
    )


def _cost(
    samples: Sequence[dict[str, Any]], cache: dict[str, dict[str, Any]]
) -> dict[str, int]:
    return {
        key: sum(int(cache[str(row["sample_id"])]["usage"][key]) for row in samples)
        for key in ("calls", "prompt_tokens", "completion_tokens")
    }


def main() -> int:
    args = _args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    dataset = json.loads(args.dataset.read_text())
    fb_cache = _cache(args.images_feedback)
    image_cache = _cache(args.images_only)
    sample_ids = {str(x["sample_id"]) for x in dataset["samples"]}
    if set(fb_cache) != sample_ids or set(image_cache) != sample_ids:
        raise ValueError("visual caches do not exactly cover the frozen dataset")
    config = load_config(args.config)
    assigner = CreditAssigner(config=config.attribution)
    by_split = {
        split: [row for row in dataset["samples"] if row["split"] == split]
        for split in ("dev", "selection", "audit")
    }
    fixed_arms = ("E0_current", "E1_feedback_only", "E5_oracle")
    dev_metrics: dict[str, dict[str, Any]] = {}
    dev_rows: dict[str, list[dict[str, Any]]] = {}
    for arm in (*fixed_arms, *_candidate_arms()):
        dev_metrics[arm], dev_rows[arm] = _evaluate_arm(
            arm, by_split["dev"], fb_cache, image_cache, assigner
        )
    dev_e0 = dev_metrics["E0_current"]
    eligible = [
        arm for arm in _guard_candidate_arms() if _passes(dev_metrics[arm], dev_e0)
    ]
    # Selection sees only candidates that pass all dev safety/coverage gates.
    selection_metrics: dict[str, dict[str, Any]] = {}
    selection_rows: dict[str, list[dict[str, Any]]] = {}
    for arm in ("E0_current", *eligible):
        selection_metrics[arm], selection_rows[arm] = _evaluate_arm(
            arm, by_split["selection"], fb_cache, image_cache, assigner
        )
    selection_e0 = selection_metrics["E0_current"]
    confirmed = [
        arm for arm in eligible if _passes(selection_metrics[arm], selection_e0)
    ]
    frozen_arm = max(confirmed, key=lambda arm: _rank(selection_metrics[arm])) if confirmed else None
    matched_threshold_arm = (
        None if frozen_arm is None else _matched_threshold_arm(frozen_arm)
    )
    audit_metrics: dict[str, dict[str, Any]] = {}
    audit_rows: dict[str, list[dict[str, Any]]] = {}
    audit_arms = [] if frozen_arm is None else [*fixed_arms, frozen_arm]
    if matched_threshold_arm is not None and matched_threshold_arm not in audit_arms:
        audit_arms.append(matched_threshold_arm)
    for arm in audit_arms:
        audit_metrics[arm], audit_rows[arm] = _evaluate_arm(
            arm, by_split["audit"], fb_cache, image_cache, assigner
        )
    bootstrap = None
    go_checks = None
    decision = "no_go_no_candidate_passed_dev_and_selection"
    noise_results = {}
    if frozen_arm is not None:
        bootstrap = _bootstrap_false_contradiction(
            audit_rows["E0_current"],
            audit_rows[frozen_arm],
            samples=args.bootstrap_samples,
            seed=args.seed,
        )
        current_cost = _cost(by_split["audit"], fb_cache)
        guard_cost = _cost(by_split["audit"], fb_cache)
        guard_metrics = audit_metrics[frozen_arm]
        e0_metrics = audit_metrics["E0_current"]
        threshold_metrics = audit_metrics[matched_threshold_arm]
        attr = guard_metrics["attribution_vs_oracle_routing"]
        e0_attr = e0_metrics["attribution_vs_oracle_routing"]
        go_checks = {
            "false_contradiction_relative_reduction_ge_0_50": bootstrap["relative_reduction"] >= 0.50,
            "paired_ci_excludes_zero": bootstrap["paired_bootstrap_95_ci_lower"] > 0.0,
            "contradiction_precision_ge_0_90": guard_metrics["contradiction"]["precision"] >= 0.90,
            "coverage_ge_0_50": guard_metrics["predicate_state"]["coverage"] >= 0.50,
            "clean_false_skill_update_lt_0_05": attr["clean_false_skill_update_rate"] < 0.05,
            "skill_update_recall_loss_le_0_10": attr["skill_update_recall"] >= e0_attr["skill_update_recall"] - 0.10,
            "target_macro_f1_not_below_e0": attr["target_macro_f1"] >= e0_attr["target_macro_f1"],
            "field_macro_f1_not_below_e0": attr["field_macro_f1"] >= e0_attr["field_macro_f1"],
            "visual_calls_not_above_e0": guard_cost["calls"] <= current_cost["calls"],
            "tokens_le_1_2x_e0": (
                guard_cost["prompt_tokens"] + guard_cost["completion_tokens"]
                <= 1.2 * (current_cost["prompt_tokens"] + current_cost["completion_tokens"])
            ),
            "guard_gain_not_explained_by_matched_threshold": (
                guard_metrics["contradiction"]["false_contradiction_rate"]
                < threshold_metrics["contradiction"]["false_contradiction_rate"]
                and guard_metrics["predicate_state"]["coverage"]
                >= threshold_metrics["predicate_state"]["coverage"] - 0.10
            ),
        }
        decision = "go" if all(go_checks.values()) else "no_go"
        for name, noise in {
            "visual_flip_10": {"visual_flip": 0.10},
            "visual_drop_20": {"visual_drop": 0.20},
            "feedback_flip_10": {"feedback_flip": 0.10},
            "visual_feedback_flip_10": {"visual_flip": 0.10, "feedback_flip": 0.10},
            "false_positive_10": {"false_positive": 0.10},
        }.items():
            metrics, _ = _evaluate_arm(
                frozen_arm,
                by_split["audit"],
                fb_cache,
                image_cache,
                assigner,
                noise=noise,
                noise_seed=args.seed,
            )
            noise_results[name] = metrics
    result = {
        "analysis_type": "phase3a_evidence_reliability_guard_frozen_audit",
        "dataset": str(args.dataset.resolve()),
        "dataset_id": dataset["dataset_id"],
        "config": args.config,
        "selection_protocol": {
            "dev_candidate_count": len(_candidate_arms()),
            "deployable_guard_candidate_count": len(_guard_candidate_arms()),
            "dev_eligible_arms": eligible,
            "selection_confirmed_arms": confirmed,
            "frozen_arm": frozen_arm,
            "matched_threshold_arm": matched_threshold_arm,
            "audit_opened_once": frozen_arm is not None,
        },
        "dev_metrics": dev_metrics,
        "selection_metrics": selection_metrics,
        "audit_metrics": audit_metrics,
        "audit_cost": {
            "E0_current": (
                None if frozen_arm is None else _cost(by_split["audit"], fb_cache)
            ),
            "guard": None if frozen_arm is None else _cost(by_split["audit"], fb_cache),
        },
        "paired_bootstrap": bootstrap,
        "go_checks": go_checks,
        "decision": decision,
        "noise_stress_audit": noise_results,
        "limitations": [
            "Attribution gold is rule routing over independent simulator-state evidence, not human intent labels.",
            "Offline recurrence count is a unique-cluster proxy; proposal/gate outcomes require the preregistered live pilot.",
            "task_complete is reported separately because rule-only collection has no learned goal grounder.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "frozen_arm": frozen_arm,
                "decision": decision,
                "go_checks": go_checks,
                "bootstrap": bootstrap,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
