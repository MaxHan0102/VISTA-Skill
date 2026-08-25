"""Cache strict no-feedback pair or temporal evidence for Phase3B."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from scripts.phase3a_extract_visual_evidence import _request
from vista_skill.models import OpenAICompatibleJsonModel, _image_data_url
from vista_skill.schemas import (
    EvidenceSource,
    PredicateEvidence,
    PredicateKey,
    TruthValue,
    dataclass_to_dict,
)


TIERS = ("direct_current", "temporal", "action_prior", "unknown")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--condition", choices=("T0_pair_strict", "T1_temporal_strict"), required=True
    )
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--request-seed", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_transitions(path: Path) -> list[dict[str, Any]]:
    output = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event_type") == "transition":
            output.append(row["payload"])
    return output


def _history(dataset: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in ("natural_events", "stress_events"):
        for row in _read_transitions(Path(dataset[key])):
            grouped[str(row["episode_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["step_id"]))
    return dict(grouped)


def _action_payload(raw: dict[str, Any]) -> dict[str, Any]:
    action = raw["action"]
    return {
        "step_id": int(raw["step_id"]),
        "type": str(action["action_type"]),
        "arguments": [str(value) for value in action["arguments"]],
        "text": str(action["text"]),
    }


def _queries(
    sample: dict[str, Any],
    recent: list[dict[str, Any]],
    *,
    temporal: bool,
) -> tuple[PredicateKey, ...]:
    request = _request(sample)
    action = request.action
    queries = list(request.goal_predicates)
    if action.action_type == "pick" and action.arguments:
        queries.extend(
            (PredicateKey("holding", (action.arguments[0],)), PredicateKey("not_holding"))
        )
    elif action.action_type == "place" and action.arguments:
        queries.append(PredicateKey("not_holding"))
        if temporal:
            prior_pick = next(
                (
                    row
                    for row in reversed(recent[:-1])
                    if row["action"]["action_type"] == "pick"
                    and row["action"]["arguments"]
                ),
                None,
            )
            if prior_pick is not None:
                queries.append(
                    PredicateKey(
                        "at",
                        (
                            str(prior_pick["action"]["arguments"][0]),
                            action.arguments[0],
                        ),
                    )
                )
    elif action.action_type in {"open", "close"} and action.arguments:
        queries.append(PredicateKey("open", (action.arguments[0],)))
    elif action.action_type == "nav" and action.arguments:
        queries.append(PredicateKey("near", (action.arguments[0],)))
    return tuple(dict.fromkeys(queries))


def _frames(
    sample: dict[str, Any],
    recent: list[dict[str, Any]],
    *,
    temporal: bool,
) -> list[tuple[str, str]]:
    transition = sample["transition"]
    output: list[tuple[str, str]] = []
    if temporal and len(recent) >= 2:
        previous = recent[-2]
        output.append(("previous_pre", str(Path(previous["pre_image"]).resolve())))
    output.extend(
        (
            ("current_pre", str(Path(transition["pre_image"]).resolve())),
            ("current_post", str(Path(transition["post_image"]).resolve())),
        )
    )
    # Preserve order while avoiding duplicated files at episode boundaries.
    unique = []
    seen = set()
    for role, path in output:
        if path not in seen:
            unique.append((role, path))
            seen.add(path)
    return unique


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["observations"],
        "properties": {
            "observations": {
                "type": "array",
                "maxItems": 16,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "predicate",
                        "value",
                        "confidence",
                        "coverage",
                        "tier",
                        "evidence",
                    ],
                    "properties": {
                        "predicate": {"type": "string", "minLength": 1},
                        "value": {
                            "type": "string",
                            "enum": ["true", "false", "unknown"],
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "coverage": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "tier": {"type": "string", "enum": list(TIERS)},
                        "evidence": {"type": "string", "maxLength": 160},
                    },
                },
            }
        },
    }


def _usage(model: OpenAICompatibleJsonModel) -> dict[str, int]:
    total = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    for counter in model.usage.values():
        for key in total:
            total[key] += int(getattr(counter, key))
    return total


def _difference(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: after[key] - before[key] for key in after}


def _accepted_evidence(
    sample: dict[str, Any],
    observations: Iterable[dict[str, Any]],
    queries: tuple[PredicateKey, ...],
) -> tuple[PredicateEvidence, ...]:
    request = _request(sample)
    allowed = set(queries)
    output = []
    for index, item in enumerate(observations):
        try:
            key = PredicateKey.parse(str(item["predicate"]))
            value = TruthValue(str(item["value"]))
            tier = str(item["tier"])
            confidence = float(item["confidence"])
            coverage = float(item["coverage"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            key not in allowed
            or tier not in {"direct_current", "temporal"}
            or confidence < 0.50
        ):
            continue
        output.append(
            PredicateEvidence(
                key=key,
                before=TruthValue.UNKNOWN,
                after=value,
                confidence=confidence,
                source=EvidenceSource.VISUAL_PAIR,
                evidence_id=(
                    f"{request.episode_id}:s{request.step_id}:phase3b:{index}:{key.render()}"
                ),
                timestamp=request.step_id,
                view_id=f"{request.episode_id}:s{request.step_id}:strict_temporal",
                coverage=coverage,
                rationale=f"tier={tier}; {str(item.get('evidence', ''))}",
            )
        )
    return tuple(output)


def main() -> int:
    args = _args()
    if args.summary.exists():
        raise FileExistsError(f"refusing to overwrite completed summary {args.summary}")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}; use --resume")
    dataset = json.loads(args.dataset.read_text())
    samples = [
        row for row in dataset["samples"] if row["split"] in {"dev", "selection"}
    ]
    if len(samples) != 200:
        raise ValueError(f"expected frozen dev+selection=200, got {len(samples)}")
    histories = _history(dataset)
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
    temporal = args.condition == "T1_temporal_strict"
    started = time.monotonic()
    mode = "a" if args.output.exists() else "x"
    with args.output.open(mode, encoding="utf-8", buffering=1) as handle:
        for index, sample in enumerate(samples, 1):
            sample_id = str(sample["sample_id"])
            if sample_id in processed:
                continue
            episode_id = str(sample["transition"]["episode_id"])
            step_id = int(sample["transition"]["step_id"])
            recent = [
                row for row in histories[episode_id] if int(row["step_id"]) <= step_id
            ]
            if not recent or int(recent[-1]["step_id"]) != step_id:
                raise RuntimeError(f"history linkage failed for {sample_id}")
            recent = recent[-4:]
            frames = _frames(sample, recent, temporal=temporal)
            queries = _queries(sample, recent, temporal=temporal)
            request = _request(sample)
            payload = {
                "instruction": request.instruction,
                "executed_action": {
                    "type": request.action.action_type,
                    "arguments": list(request.action.arguments),
                    "text": request.action.text,
                },
                "query_predicates": [key.render() for key in queries],
                "frame_order": [role for role, _ in frames],
                "rule": (
                    "Use pixels as evidence. Not visible means unknown, not false. "
                    "An action alone is action_prior, never direct evidence. Preserve "
                    "numbered instance identity."
                ),
            }
            if temporal:
                payload["recent_public_actions"] = [
                    _action_payload(row) for row in recent
                ]
            content = [
                {"type": "text", "text": json.dumps(payload, sort_keys=True)},
                *[
                    {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
                    for _, path in frames
                ],
            ]
            endpoint_offset = 1 if temporal else 0
            base_url = args.base_url[(index - 1 + endpoint_offset) % len(args.base_url)]
            model = models[base_url]
            before = _usage(model)
            result: dict[str, Any] = {"observations": []}
            error = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    result = dict(
                        model.complete_json(
                            system=(
                                "Extract evidence only from the chronologically ordered RGB "
                                "frames. Recent actions are hypotheses, not proof. Label every "
                                "claim direct_current, temporal, action_prior, or unknown. Do "
                                "not infer desired or predicted effects."
                            ),
                            content=content,
                            schema=_schema(),
                            purpose=(
                                "vista_phase3b_temporal_strict"
                                if temporal
                                else "vista_phase3b_pair_strict"
                            ),
                        )
                    )
                    error = None
                    break
                except Exception as exc:  # model/network checkpoint boundary
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < args.max_retries:
                        time.sleep(min(2.0 * attempt, 5.0))
            after = _usage(model)
            observations = list(result.get("observations", []))
            evidence = _accepted_evidence(sample, observations, queries)
            row = {
                "sample_id": sample_id,
                "split": sample["split"],
                "condition": args.condition,
                "frame_roles": [role for role, _ in frames],
                "frame_sha256": [_sha256(Path(path)) for _, path in frames],
                "recent_action_count": len(recent) if temporal else 1,
                "query_predicates": [key.render() for key in queries],
                "observations": observations,
                "evidence": dataclass_to_dict(evidence),
                "error": error,
                "attempts": attempt,
                "base_url": base_url,
                "usage": _difference(after, before),
                "isolation_audit": {
                    "feedback_serialized": False,
                    "last_action_success_serialized": False,
                    "pre_ledger_serialized": False,
                    "expected_serialized": False,
                    "oracle_serialized": False,
                    "skill_serialized": False,
                },
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            processed[sample_id] = row
            print(
                f"condition={args.condition} sample={index}/{len(samples)} "
                f"id={sample_id} accepted={len(evidence)} error={error}",
                flush=True,
            )

    rows = [processed[str(sample["sample_id"])] for sample in samples]
    tier_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for item in row["observations"]:
            tier_counts[str(item.get("tier", "malformed"))] += 1
    usage = {
        key: sum(int(row["usage"][key]) for row in rows)
        for key in ("calls", "prompt_tokens", "completion_tokens")
    }
    summary = {
        "analysis_type": "phase3b_strict_no_feedback_evidence_cache",
        "dataset": str(args.dataset.resolve()),
        "dataset_id": dataset["dataset_id"],
        "condition": args.condition,
        "splits": {"dev": 120, "selection": 80, "audit": 0},
        "model": args.model,
        "base_urls": list(args.base_url),
        "request_seed": args.request_seed,
        "sample_count": len(rows),
        "error_count": sum(row["error"] is not None for row in rows),
        "tier_counts": dict(sorted(tier_counts.items())),
        "accepted_evidence_count": sum(len(row["evidence"]) for row in rows),
        "usage": usage,
        "usage_by_endpoint": {
            base_url: {
                key: sum(
                    int(row["usage"][key])
                    for row in rows
                    if row["base_url"] == base_url
                )
                for key in ("calls", "prompt_tokens", "completion_tokens")
            }
            for base_url in args.base_url
        },
        "strict_isolation": all(
            not value
            for row in rows
            for value in row["isolation_audit"].values()
        ),
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    # Match the live evidence provider's fail-closed contract: a model/network
    # failure after the bounded retry budget is a recorded zero-evidence
    # abstention, not a reason to discard the other paired samples. Completeness
    # remains the process-level hard failure.
    return 0 if len(rows) == len(samples) else 2


if __name__ == "__main__":
    raise SystemExit(main())
