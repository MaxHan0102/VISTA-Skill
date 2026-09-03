#!/usr/bin/env python3
"""Replay frozen lineage decisions through the retention-only shadow policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from vista_skill.config import load_config
from vista_skill.evolution import GateStageResult, shadow_candidate_eligible


REPO = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_lineage(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def replay(config_path: Path, lineage_paths: list[Path]) -> dict[str, Any]:
    config = load_config(config_path)
    decisions = []
    for path in lineage_paths:
        for index, record in enumerate(_load_lineage(path)):
            decision = record["decision"]
            stages = decision.get("stages", [])
            failed = None
            if stages and not bool(stages[-1]["passed"]):
                payload = stages[-1]
                failed = GateStageResult(
                    stage=str(payload["stage"]),
                    passed=False,
                    reason=str(payload["reason"]),
                    metrics={
                        str(key): float(value)
                        for key, value in payload.get("metrics", {}).items()
                    },
                )
            accepted = bool(decision["accepted"])
            eligible = bool(
                not accepted
                and failed is not None
                and shadow_candidate_eligible(failed, config.gate)
            )
            replayed = "promoted" if accepted else "shadow" if eligible else "rejected"
            decisions.append(
                {
                    "lineage": _display_path(path),
                    "lineage_sha256": _sha256(path),
                    "record_index": index,
                    "patch_id": decision["patch_id"],
                    "current_disposition": "promoted" if accepted else "rejected",
                    "replayed_disposition": replayed,
                    "promotion_changed": False,
                    "failed_stage": None if failed is None else failed.stage,
                    "failed_stage_metrics": {}
                    if failed is None
                    else dict(failed.metrics),
                }
            )
    return {
        "diagnostic": True,
        "claim_eligible": False,
        "retention_only": True,
        "config": _display_path(config_path),
        "config_sha256": _sha256(config_path),
        "decision_count": len(decisions),
        "current_counts": {
            state: sum(item["current_disposition"] == state for item in decisions)
            for state in ("rejected", "shadow", "promoted")
        },
        "replayed_counts": {
            state: sum(item["replayed_disposition"] == state for item in decisions)
            for state in ("rejected", "shadow", "promoted")
        },
        "promotion_change_count": 0,
        "decisions": decisions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--lineage", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    lineage_paths = [Path(item).resolve() for item in args.lineage]
    result = replay(config_path, lineage_paths)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
