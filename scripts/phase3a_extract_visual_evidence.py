"""Cache one VLM evidence condition for the frozen Phase3A dataset."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from vista_skill.models import JsonVisualEvidenceProvider, OpenAICompatibleJsonModel
from vista_skill.schemas import (
    ActionCall,
    EvidenceRequest,
    PredicateKey,
    PredicateState,
    TruthValue,
    dataclass_to_dict,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--condition",
        choices=("images_and_feedback", "images_only"),
        required=True,
    )
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--request-seed", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
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


def _state(raw: dict[str, Any]) -> PredicateState:
    return PredicateState(
        key=_key(raw["key"]),
        value=TruthValue(str(raw["value"])),
        confidence=float(raw["confidence"]),
        source=str(raw["source"]),
        evidence_ids=tuple(str(x) for x in raw["evidence_ids"]),
        timestamp=int(raw["timestamp"]),
        view_id=raw.get("view_id"),
        coverage=float(raw.get("coverage", 1.0)),
        task_relevance=float(raw.get("task_relevance", 1.0)),
    )


def _request(sample: dict[str, Any]) -> EvidenceRequest:
    payload = sample["transition"]
    return EvidenceRequest(
        episode_id=str(payload["episode_id"]),
        step_id=int(payload["step_id"]),
        instruction=str(payload["instruction"]),
        action=_action(payload["action"]),
        pre_image=str(payload["pre_image"]),
        post_image=str(payload["post_image"]),
        feedback=str(payload["feedback"]),
        last_action_success=payload.get("last_action_success"),
        pre_ledger=tuple(_state(item) for item in payload.get("pre_ledger", [])),
        goal_predicates=tuple(_key(item) for item in payload.get("goal_predicates", [])),
    )


def _usage(model: OpenAICompatibleJsonModel) -> dict[str, int]:
    total = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    for counter in model.usage.values():
        for key in total:
            total[key] += int(getattr(counter, key))
    return total


def _difference(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: after[key] - before[key] for key in after}


def main() -> int:
    args = _args()
    if args.summary.exists():
        raise FileExistsError(f"refusing to overwrite completed summary {args.summary}")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}; use --resume")
    dataset = json.loads(args.dataset.read_text())
    processed: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                processed[str(row["sample_id"])] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    models = {
        base_url: OpenAICompatibleJsonModel(
            args.model,
            base_url=base_url,
            api_key="EMPTY",
            temperature=0.0,
            max_tokens=4096,
            seed=args.request_seed,
        )
        for base_url in args.base_url
    }
    providers = {
        base_url: JsonVisualEvidenceProvider(
            model, include_feedback=args.condition == "images_and_feedback"
        )
        for base_url, model in models.items()
    }
    started = time.monotonic()
    mode = "a" if args.output.exists() else "x"
    with args.output.open(mode, encoding="utf-8", buffering=1) as handle:
        for index, sample in enumerate(dataset["samples"], 1):
            sample_id = str(sample["sample_id"])
            if sample_id in processed:
                continue
            request = _request(sample)
            condition_offset = 0 if args.condition == "images_and_feedback" else 1
            base_url = args.base_url[(index - 1 + condition_offset) % len(args.base_url)]
            model = models[base_url]
            provider = providers[base_url]
            before = _usage(model)
            evidence = ()
            error = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    evidence = tuple(provider.extract(request))
                    error = None
                    break
                except Exception as exc:  # network/model boundary; checkpoint failure
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < args.max_retries:
                        time.sleep(min(2.0 * attempt, 5.0))
            after = _usage(model)
            row = {
                "sample_id": sample_id,
                "condition": args.condition,
                "evidence": dataclass_to_dict(evidence),
                "error": error,
                "attempts": attempt,
                "base_url": base_url,
                "usage": _difference(after, before),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            processed[sample_id] = row
            print(
                f"condition={args.condition} sample={index}/{len(dataset['samples'])} "
                f"id={sample_id} evidence={len(evidence)} error={row['error']}",
                flush=True,
            )
    rows = list(processed.values())
    usage = {
        key: sum(int(row["usage"][key]) for row in rows)
        for key in ("calls", "prompt_tokens", "completion_tokens")
    }
    summary = {
        "analysis_type": "phase3a_visual_evidence_cache",
        "dataset": str(args.dataset.resolve()),
        "dataset_id": dataset["dataset_id"],
        "condition": args.condition,
        "model": args.model,
        "base_urls": list(args.base_url),
        "endpoint_assignment": (
            "round_robin by frozen dataset index; images_only uses +1 crossover offset"
        ),
        "request_seed": args.request_seed,
        "sample_count": len(rows),
        "error_count": sum(row["error"] is not None for row in rows),
        "usage": usage,
        "usage_by_endpoint": {
            base_url: {
                key: sum(
                    int(row["usage"][key])
                    for row in rows
                    if row.get("base_url") == base_url
                )
                for key in ("calls", "prompt_tokens", "completion_tokens")
            }
            for base_url in args.base_url
        },
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "output": str(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if len(rows) == len(dataset["samples"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
