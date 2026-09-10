"""Validate Phase3B artifacts and write the immutable final decision record."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ARMS = (
    "C0_current",
    "C1_no_feedback",
    "C2_temporal_no_feedback",
    "C3_temporal_feedback",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("running/Phase3/phase3b"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl_ids(path: Path) -> list[str]:
    return [str(json.loads(line)["sample_id"]) for line in path.read_text().splitlines()]


def main() -> int:
    args = _args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    executor = _read(args.root / "phase3b_executor_paired_analysis_20260825.json")
    evidence = _read(
        args.root / "phase3b_temporal_evidence_gated_analysis_20260825.json"
    )
    arm_summaries = {
        arm: _read(args.root / f"executor_{arm}" / "summary.json") for arm in ARMS
    }
    episode_ids = arm_summaries[ARMS[0]]["episode_ids"]
    if any(arm_summaries[arm]["episode_ids"] != episode_ids for arm in ARMS[1:]):
        raise ValueError("executor arm coordinates are not exactly paired")
    if len(episode_ids) != 30:
        raise ValueError("executor pilot is not complete")
    for arm, summary in arm_summaries.items():
        events = Path(summary["events"])
        if _sha256(events) != summary["events_sha256"]:
            raise ValueError(f"executor events hash mismatch: {arm}")

    cache_summaries = {
        name: _read(args.root / f"{name}.summary.json")
        for name in ("T0_pair_strict", "T1_temporal_strict")
    }
    cache_ids = {}
    for name, summary in cache_summaries.items():
        cache = args.root / f"{name}.jsonl"
        if _sha256(cache) != summary["output_sha256"]:
            raise ValueError(f"evidence cache hash mismatch: {name}")
        cache_ids[name] = _jsonl_ids(cache)
        if len(cache_ids[name]) != 200 or len(set(cache_ids[name])) != 200:
            raise ValueError(f"evidence cache coordinates invalid: {name}")
        if not summary["strict_isolation"]:
            raise ValueError(f"information isolation failed: {name}")
    if cache_ids["T0_pair_strict"] != cache_ids["T1_temporal_strict"]:
        raise ValueError("strict evidence caches are not paired in the same order")
    if executor["decision"] != "no_go":
        raise ValueError("unexpected P3B-1 decision")
    if evidence["decision"] != "no_go_dev" or evidence["selection_opened"]:
        raise ValueError("unexpected P3B-2 gate state")
    if int(evidence["audit_samples_read"]) != 0:
        raise ValueError("Phase3B evidence analyzer consumed an audit sample")

    result = {
        "analysis_type": "phase3b_final_decision",
        "decision": "no_go",
        "phase3b_1_executor": {
            "decision": executor["decision"],
            "episode_count": executor["episode_count"],
            "go_checks": executor["go_checks"],
            "arm_metrics": executor["arm_metrics"],
            "paired_progress_bootstrap": executor["paired_progress_bootstrap"],
        },
        "phase3b_2_temporal_evidence": {
            "decision": evidence["decision"],
            "dev_go_checks": evidence["dev_go_checks"],
            "selection_opened": evidence["selection_opened"],
            "audit_samples_read": evidence["audit_samples_read"],
            "dev_metrics": evidence["dev_metrics"],
            "dev_cost": evidence["dev_cost"],
            "dev_tier_counts": evidence["dev_tier_counts"],
        },
        "conditional_stages": {
            "fresh_episode_60_79_collection": "canceled_by_preregistered_no_go",
            "fresh_temporal_audit": "canceled_by_preregistered_no_go",
            "late_feedback_fusion": "canceled_by_preregistered_no_go",
        },
        "artifact_validation": {
            "executor_episode_ids_exactly_paired": True,
            "executor_event_hashes_match": True,
            "strict_cache_ids_exactly_paired": True,
            "strict_cache_hashes_match": True,
            "strict_information_isolation": True,
            "phase3a_audit_samples_read": 0,
            "T0_error_count": cache_summaries["T0_pair_strict"]["error_count"],
            "T1_error_count": cache_summaries["T1_temporal_strict"]["error_count"],
        },
        "artifact_sha256": {
            "executor_analysis": _sha256(
                args.root / "phase3b_executor_paired_analysis_20260825.json"
            ),
            "temporal_evidence_analysis": _sha256(
                args.root / "phase3b_temporal_evidence_gated_analysis_20260825.json"
            ),
            "T0_cache": cache_summaries["T0_pair_strict"]["output_sha256"],
            "T1_cache": cache_summaries["T1_temporal_strict"]["output_sha256"],
        },
        "interpretation_boundary": [
            "Temporal RGB recovered mean executor progress in this 30-task pilot but failed the preregistered repetition burden gate.",
            "Strict no-feedback temporal evidence was precise when it asserted, but coverage and contradiction recall were far below persistent-write requirements.",
            "The frozen 8B model did not use the extra frame as a reliable substitute for structured feedback; this does not establish a limit for larger or trained world-aware models.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "phase3b_1": executor["decision"],
                "phase3b_2": evidence["decision"],
                "conditional_stages": result["conditional_stages"],
                "artifact_validation": result["artifact_validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
