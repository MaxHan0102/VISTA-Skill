"""Weak counterfactual attribution audit on natural fault-injected rollouts.

The injected fault fixes the positive target before outcomes are inspected.
Other feedback-observable, skill-mismatch events are controls and should not
update that fault.  This remains weak gold because it does not independently
annotate unrelated natural defects.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from vista_skill.attribution import CreditAssigner
from vista_skill.metrics import macro_f1
from vista_skill.models import JsonAttributionTeacher, OpenAICompatibleJsonModel
from vista_skill.schemas import (
    AttributionContext,
    DeltaSource,
    EvidenceSource,
    ExpectedChange,
    Mismatch,
    MismatchKind,
    PredicateEvidence,
    PredicateKey,
    SkillField,
    TruthValue,
)


FAULT_PROFILES = {
    "constraint_pick_multihold": {
        "rule_suffix": ":constraint_pick_occupies_gripper",
        "evidence_after": TruthValue.FALSE,
        "field": "constraint",
    },
    "effect_pick_inversion": {
        "rule_suffix": ":effect_pick_holds_target_category",
        "evidence_after": TruthValue.TRUE,
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _key(raw: dict[str, Any]) -> PredicateKey:
    return PredicateKey(str(raw["name"]), tuple(str(x) for x in raw["arguments"]))


def _expected(raw: dict[str, Any] | None) -> ExpectedChange | None:
    if raw is None:
        return None
    field = raw.get("skill_field")
    return ExpectedChange(
        key=_key(raw["key"]),
        before=TruthValue(str(raw["before"])),
        after=TruthValue(str(raw["after"])),
        source=DeltaSource(str(raw["source"])),
        source_id=str(raw["source_id"]),
        skill_field=None if field is None else SkillField(str(field)),
    )


def _evidence(raw: dict[str, Any] | None) -> PredicateEvidence | None:
    if raw is None:
        return None
    return PredicateEvidence(
        key=_key(raw["key"]),
        before=TruthValue(str(raw["before"])),
        after=TruthValue(str(raw["after"])),
        confidence=float(raw["confidence"]),
        source=EvidenceSource(str(raw["source"])),
        evidence_id=str(raw["evidence_id"]),
        timestamp=int(raw["timestamp"]),
        view_id=None if raw.get("view_id") is None else str(raw["view_id"]),
        coverage=float(raw.get("coverage", 1.0)),
        task_relevance=float(raw.get("task_relevance", 1.0)),
        rationale=str(raw.get("rationale", "")),
    )


def _mismatch(raw: dict[str, Any]) -> Mismatch:
    return Mismatch(
        mismatch_id=str(raw["mismatch_id"]),
        key=_key(raw["key"]),
        kind=MismatchKind(str(raw["kind"])),
        expected=_expected(raw.get("expected")),
        evidence=_evidence(raw.get("evidence")),
        evidence_ids=tuple(str(x) for x in raw.get("evidence_ids", [])),
    )


def _context(payload: dict[str, Any]) -> AttributionContext:
    raw = payload.get("metadata", {}).get("attribution_context", {})
    return AttributionContext(
        executor_followed_skill=raw.get("executor_followed_skill"),
        stochastic_suspected=bool(raw.get("stochastic_suspected", False)),
        identity_conflict=bool(raw.get("identity_conflict", False)),
        task_pattern=str(raw.get("task_pattern", "general")),
        object_context=str(raw.get("object_context", "general")),
        instruction=str(payload.get("instruction", "")),
        action_type=str(payload.get("action", {}).get("action_type", "")),
        goal_predicates=tuple(_key(item) for item in payload.get("goal_predicates", [])),
    )


def _is_fault_positive(mismatch: Mismatch, fault: str) -> bool:
    profile = FAULT_PROFILES[fault]
    expected = mismatch.expected
    return bool(
        expected is not None
        and expected.source is DeltaSource.SKILL
        and expected.source_id.endswith(str(profile["rule_suffix"]))
        and mismatch.kind is MismatchKind.CONTRADICTION
        and mismatch.evidence is not None
        and mismatch.evidence.after is profile["evidence_after"]
        and mismatch.evidence.confidence >= 0.75
    )


def _load_cases(path: Path, fault: str) -> list[dict[str, Any]]:
    profile = FAULT_PROFILES[fault]
    cases = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("event_type") != "transition":
            continue
        payload = record["payload"]
        mismatches = tuple(_mismatch(item) for item in payload.get("mismatches", []))
        has_skill_mismatch = any(
            item.expected is not None and item.expected.source is DeltaSource.SKILL
            for item in mismatches
        )
        has_feedback_evidence = any(
            item.get("source") == EvidenceSource.ENV_FEEDBACK.value
            for item in payload.get("evidence_delta", [])
        )
        if not has_skill_mismatch or not has_feedback_evidence:
            continue
        positive = any(_is_fault_positive(item, fault) for item in mismatches)
        cases.append(
            {
                "payload": payload,
                "mismatches": mismatches,
                "context": _context(payload),
                "gold_target": "skill_update" if positive else "abstain",
                "gold_field": profile["field"] if positive else "none",
            }
        )
    return cases


def _field(value: SkillField | None) -> str:
    return "none" if value is None else value.value


def _summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    gold = [row["gold_target"] for row in rows]
    predicted = [row["predicted_target"] for row in rows]
    true_positive = sum(g == p == "skill_update" for g, p in zip(gold, predicted))
    false_positive = sum(g != "skill_update" and p == "skill_update" for g, p in zip(gold, predicted))
    false_negative = sum(g == "skill_update" and p != "skill_update" for g, p in zip(gold, predicted))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "cases": len(rows),
        "target_macro_f1": macro_f1(gold, predicted),
        "field_macro_f1": macro_f1(
            [row["gold_field"] for row in rows],
            [row["predicted_field"] for row in rows],
        ),
        "skill_update_precision": precision,
        "skill_update_recall": recall,
        "skill_update_f1": f1,
        "confusion": dict(Counter(f"{g}->{p}" for g, p in zip(gold, predicted))),
    }


def _assign_decisive_skill_partition(
    assigner: CreditAssigner,
    mismatches: tuple[Mismatch, ...],
    context: AttributionContext,
):
    """Test a general provenance-aware dominance rule without fault identity."""
    decisive = tuple(
        item
        for item in mismatches
        if item.expected is not None
        and item.expected.source is DeltaSource.SKILL
        and item.expected.skill_field is not None
        and item.kind in {MismatchKind.CONTRADICTION, MismatchKind.MISSING_PROGRESS}
        and item.evidence is not None
        and item.evidence.confidence >= assigner.config.min_evidence_confidence
    )
    fields = {item.expected.skill_field for item in decisive if item.expected}
    conflicting_action_schema = any(
        item.expected is not None
        and item.expected.source is DeltaSource.ACTION_SCHEMA
        and item.kind in {MismatchKind.CONTRADICTION, MismatchKind.MISSING_PROGRESS}
        for item in mismatches
    )
    protected_context = (
        context.identity_conflict
        or context.stochastic_suspected
        or conflicting_action_schema
        or (
            context.executor_followed_skill is False
            and any(item.kind is MismatchKind.MISSING_PROGRESS for item in decisive)
        )
        or any(
            item.kind
            in {
                MismatchKind.IDENTITY_CONFLICT,
                MismatchKind.TEMPORAL_CONFLICT,
                MismatchKind.TERMINATION_CONFLICT,
            }
            for item in mismatches
        )
    )
    if len(fields) == 1 and not protected_context:
        return assigner.assign(decisive, context)
    return assigner.assign(mismatches, context)


def main() -> int:
    args = _args()
    cases = _load_cases(args.events, args.fault)
    if not cases:
        raise ValueError("no feedback-observable skill mismatch events")
    model = OpenAICompatibleJsonModel(
        args.model,
        base_url=args.base_url,
        api_key="EMPTY",
        temperature=0.0,
        max_tokens=512,
        seed=args.seed,
    )
    teacher = JsonAttributionTeacher(model)
    relevant_assigner = CreditAssigner()
    conditions: dict[str, list[dict[str, Any]]] = {
        "recorded_rule_first": [],
        "decisive_skill_partition_rule_first": [],
        "fault_relevant_oracle_filter": [],
        "direct_qwen_teacher": [],
    }
    for index, case in enumerate(cases):
        payload = case["payload"]
        recorded = payload["attribution"]
        relevant = tuple(
            item for item in case["mismatches"] if _is_fault_positive(item, args.fault)
        )
        filtered_result = relevant_assigner.assign(relevant, case["context"])
        partitioned_result = _assign_decisive_skill_partition(
            relevant_assigner, case["mismatches"], case["context"]
        )
        error = None
        try:
            teacher_result = teacher.assign(case["mismatches"], case["context"])
            teacher_target = teacher_result.target.value
            teacher_field = _field(teacher_result.field)
        except Exception as exc:
            teacher_target = "invalid"
            teacher_field = "invalid"
            error = f"{type(exc).__name__}: {exc}"
        common = {
            "case_index": index,
            "episode_id": str(payload["episode_id"]),
            "step_id": int(payload["step_id"]),
            "action_type": str(payload["action"]["action_type"]),
            "gold_target": case["gold_target"],
            "gold_field": case["gold_field"],
        }
        conditions["recorded_rule_first"].append(
            {
                **common,
                "predicted_target": str(recorded["target"]),
                "predicted_field": "none" if recorded.get("field") is None else str(recorded["field"]),
            }
        )
        conditions["fault_relevant_oracle_filter"].append(
            {
                **common,
                "predicted_target": filtered_result.target.value,
                "predicted_field": _field(filtered_result.field),
            }
        )
        conditions["decisive_skill_partition_rule_first"].append(
            {
                **common,
                "predicted_target": partitioned_result.target.value,
                "predicted_field": _field(partitioned_result.field),
            }
        )
        conditions["direct_qwen_teacher"].append(
            {
                **common,
                "predicted_target": teacher_target,
                "predicted_field": teacher_field,
                "error": error,
            }
        )
        print(
            f"case={index + 1}/{len(cases)} gold={case['gold_target']} "
            f"recorded={recorded['target']} filtered={filtered_result.target.value} "
            f"partitioned={partitioned_result.target.value} teacher={teacher_target}"
        )
    output = {
        "analysis_type": "natural_fault_injection_attribution_weak_audit",
        "source_events": str(args.events),
        "fault": args.fault,
        "base_url": args.base_url,
        "model": args.model,
        "seed": args.seed,
        "gold_positive_rule_suffix": FAULT_PROFILES[args.fault]["rule_suffix"],
        "gold_positive_count": sum(case["gold_target"] == "skill_update" for case in cases),
        "conditions": {
            name: {"metrics": _summarize(rows), "rows": rows}
            for name, rows in conditions.items()
        },
        "teacher_usage": {
            purpose: {
                "calls": usage.calls,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            }
            for purpose, usage in model.usage.items()
        },
        "limitation": (
            "The injected fault rule defines positive weak gold before outcomes. "
            "Controls are not independently annotated for unrelated natural Skill defects; "
            "the oracle filter is a mechanism upper bound, not a deployable method."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    for name, payload in output["conditions"].items():
        print(name, payload["metrics"])
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
