"""Weak-label audit of Qwen evidence on recorded natural Habitat transitions.

Simulator-derived public feedback predicates are used as weak gold.  Each event
is replayed with images+feedback, images only, and feedback only to separate the
incremental value of the visual observations from public feedback.  This does
not replace human/simulator-state gold annotation.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from vista_skill.evidence_oracle import evaluate_calibration, selective_risk_curve
from vista_skill.models import (
    JsonVisualEvidenceProvider,
    OpenAICompatibleJsonModel,
    _evidence_queries,
    _evidence_schema,
)
from vista_skill.schemas import (
    ActionCall,
    EvidenceRequest,
    EvidenceSource,
    PredicateEvidence,
    PredicateKey,
    TruthValue,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--request-seed", type=int, default=0)
    return parser.parse_args()


def _key(raw: dict[str, Any]) -> PredicateKey:
    return PredicateKey(str(raw["name"]), tuple(str(x) for x in raw["arguments"]))


def _action(raw: dict[str, Any]) -> ActionCall:
    return ActionCall(
        action_id=int(raw["action_id"]),
        action_type=str(raw["action_type"]),
        arguments=tuple(str(x) for x in raw["arguments"]),
        text=str(raw["text"]),
        raw_action=None if raw.get("raw_action") is None else str(raw["raw_action"]),
    )


def _load_events(path: Path, limit: int | None) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("event_type") != "transition":
            continue
        payload = record["payload"]
        feedback_gold = [
            item
            for item in payload.get("evidence_delta", [])
            if item.get("source") == "env_feedback"
            and item.get("after") in {"true", "false"}
        ]
        if not feedback_gold:
            continue
        if not Path(payload["pre_image"]).is_file() or not Path(
            payload["post_image"]
        ).is_file():
            continue
        events.append({"payload": payload, "gold": feedback_gold})
        if limit is not None and len(events) >= limit:
            break
    return events


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    targets = len(rows)
    asserted = [row for row in rows if row["asserted"]]
    correct = [row for row in rows if row["correct"]]
    precision = _safe_ratio(len(correct), len(asserted))
    recall = _safe_ratio(len(correct), targets)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    calibration_input = [
        (float(row["confidence"]), bool(row["correct"])) for row in asserted
    ]
    calibration = evaluate_calibration(calibration_input)
    curve = selective_risk_curve(calibration_input)
    at_08 = next(
        (point for point in curve if abs(point.confidence_threshold - 0.8) < 1e-9),
        None,
    )
    return {
        "targets": targets,
        "asserted": len(asserted),
        "correct": len(correct),
        "coverage": _safe_ratio(len(asserted), targets),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": calibration.brier,
        "ece": calibration.expected_calibration_error,
        "selective_risk_at_0_8": None if at_08 is None else asdict(at_08),
    }


def _run_condition(
    events: list[dict[str, Any]],
    *,
    base_url: str,
    model_name: str,
    condition: str,
    request_seed: int,
) -> dict[str, Any]:
    if condition not in {"images_and_feedback", "images_only", "feedback_only"}:
        raise ValueError(f"unknown condition: {condition}")
    model = OpenAICompatibleJsonModel(
        model_name,
        base_url=base_url,
        api_key="EMPTY",
        temperature=0.0,
        max_tokens=1024,
        seed=request_seed,
    )
    provider = JsonVisualEvidenceProvider(model)
    rows: list[dict[str, Any]] = []
    failures = []
    for index, event in enumerate(events):
        payload = event["payload"]
        gold = {_key(item["key"]): TruthValue(str(item["after"])) for item in event["gold"]}
        request = EvidenceRequest(
            episode_id=str(payload["episode_id"]),
            step_id=int(payload["step_id"]),
            instruction=str(payload["instruction"]),
            action=_action(payload["action"]),
            pre_image=str(payload["pre_image"]),
            post_image=str(payload["post_image"]),
            feedback=(
                str(payload["feedback"])
                if condition in {"images_and_feedback", "feedback_only"}
                else ""
            ),
            last_action_success=payload.get("last_action_success"),
            pre_ledger=(),
            goal_predicates=tuple(gold),
        )
        try:
            if condition == "feedback_only":
                predicted = {
                    item.key: item for item in _extract_feedback_only(model, request)
                }
            else:
                predicted = {item.key: item for item in provider.extract(request)}
        except Exception as error:  # retain model/API failures as zero coverage
            failures.append(
                {
                    "episode_id": request.episode_id,
                    "step_id": request.step_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            predicted = {}
        for key, value in gold.items():
            item = predicted.get(key)
            asserted = item is not None and item.after is not TruthValue.UNKNOWN
            rows.append(
                {
                    "episode_id": request.episode_id,
                    "step_id": request.step_id,
                    "action_type": request.action.action_type,
                    "predicate": key.render(),
                    "gold": value.value,
                    "predicted": None if item is None else item.after.value,
                    "confidence": 0.0 if item is None else item.confidence,
                    "coverage_score": 0.0 if item is None else item.coverage,
                    "asserted": asserted,
                    "correct": asserted and item.after is value,
                }
            )
        print(
            f"condition={condition} event={index + 1}/{len(events)} "
            f"episode={request.episode_id} step={request.step_id}"
        )
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_action[row["action_type"]].append(row)
    return {
        "condition": condition,
        "include_images": condition != "feedback_only",
        "include_public_feedback": condition != "images_only",
        "metrics": _summarize(rows),
        "metrics_by_action": {
            action: _summarize(items) for action, items in sorted(by_action.items())
        },
        "api_failures": failures,
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


def _extract_feedback_only(
    model: OpenAICompatibleJsonModel,
    request: EvidenceRequest,
) -> list[PredicateEvidence]:
    """Run the same structured evidence query without either image payload."""
    queries = _evidence_queries(request)
    prompt = {
        "instruction": request.instruction,
        "executed_action": {
            "type": request.action.action_type,
            "arguments": request.action.arguments,
            "text": request.action.text,
        },
        "public_environment_feedback": request.feedback,
        "pre_action_belief": [],
        "query_predicates": [key.render() for key in queries],
        "rule": (
            "Use only explicit public feedback. Return unknown when the feedback "
            "does not establish a queried predicate."
        ),
    }
    result = model.complete_json(
        system=(
            "Extract evidence supported only by public environment feedback. "
            "Do not infer desired or predicted effects. Return unknown when coverage "
            "is insufficient."
        ),
        content=[{"type": "text", "text": json.dumps(prompt, sort_keys=True)}],
        schema=_evidence_schema(),
        purpose="vista_feedback_only_evidence",
    )
    items = []
    for index, observation in enumerate(result.get("observations", [])):
        try:
            key = PredicateKey.parse(str(observation["predicate"]))
        except ValueError:
            continue
        items.append(
            PredicateEvidence(
                key=key,
                before=TruthValue.UNKNOWN,
                after=TruthValue(str(observation["value"])),
                confidence=float(observation["confidence"]),
                source=EvidenceSource.ENV_FEEDBACK,
                evidence_id=(
                    f"{request.episode_id}:s{request.step_id}:"
                    f"feedback_only:{index}:{key.render()}"
                ),
                timestamp=request.step_id,
                coverage=float(observation["coverage"]),
                rationale=str(observation["evidence"]),
            )
        )
    return items


def main() -> int:
    args = _args()
    events = _load_events(args.events, args.max_events)
    if not events:
        raise ValueError("no feedback-labelled transitions with available images")
    condition_names = ("images_and_feedback", "images_only", "feedback_only")
    conditions = [
        _run_condition(
            events,
            base_url=args.base_url,
            model_name=args.model,
            condition=condition,
            request_seed=args.request_seed,
        )
        for condition in condition_names
    ]
    result = {
        "analysis_type": "feedback_weak_gold_natural_transition_audit",
        "source_events": str(args.events),
        "event_count": len(events),
        "predicate_target_count": sum(len(event["gold"]) for event in events),
        "model": args.model,
        "base_url": args.base_url,
        "request_seed": args.request_seed,
        "conditions": conditions,
        "limitation": (
            "Gold labels are derived from public simulator feedback and cover only "
            "feedback-observable predicates. They are not independent human labels; "
            "the feedback-enabled condition also exposes the weak gold text to the model."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for condition in conditions:
        metrics = condition["metrics"]
        print(
            f"condition={condition['condition']} "
            f"coverage={metrics['coverage']:.3f} precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f}"
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
