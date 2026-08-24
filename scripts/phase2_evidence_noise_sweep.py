"""Synthetic oracle/noisy-evidence sensitivity sweep for Phase 2.

The sweep validates the calibration harness and estimates how evidence drop,
polarity flips and false positives propagate into rule-first VTCA.  It is not
a substitute for the planned human-labelled natural-event evidence audit.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

from vista_skill.action_schema import FixedActionSchema
from vista_skill.attribution import CreditAssigner
from vista_skill.config import load_config
from vista_skill.evidence_oracle import (
    NoisyEvidenceProvider,
    OracleEvidenceProvider,
    compare_providers,
)
from vista_skill.fault_injection import build_fault_cases
from vista_skill.metrics import macro_f1
from vista_skill.mismatch import compare_transitions
from vista_skill.schemas import (
    ActionCall,
    EvidenceRequest,
    Mismatch,
    PredicateEvidence,
    PredicateKey,
    PredicateState,
    SkillField,
    TruthValue,
)
from vista_skill.skills import initialize_shared_skill


class _FixedProvider:
    def __init__(self, items: Sequence[PredicateEvidence]) -> None:
        self.items = tuple(items)

    def extract(self, _request: EvidenceRequest) -> tuple[PredicateEvidence, ...]:
        return self.items


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("running/phase2_evidence_noise.json")
    )
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--fault-cases-per-kind", type=int, default=10)
    return parser.parse_args()


def _request(index: int) -> EvidenceRequest:
    target = f"object_{index % 7}"
    receptacle = f"receptacle_{index % 5}"
    holding = PredicateKey("holding", (target,))
    near = PredicateKey("near", (receptacle,))
    opened = PredicateKey("open", (f"container_{index % 3}",))
    pre = (
        PredicateState(
            holding,
            TruthValue.FALSE,
            1.0,
            "synthetic_oracle_fixture",
            (f"pre:{index}:holding",),
            index,
        ),
        PredicateState(
            near,
            TruthValue.FALSE,
            1.0,
            "synthetic_oracle_fixture",
            (f"pre:{index}:near",),
            index,
        ),
        PredicateState(
            opened,
            TruthValue.FALSE,
            1.0,
            "synthetic_oracle_fixture",
            (f"pre:{index}:open",),
            index,
        ),
    )
    action_types = ("pick", "nav", "open")
    action_type = action_types[index % len(action_types)]
    argument = target if action_type == "pick" else (
        receptacle if action_type == "nav" else f"container_{index % 3}"
    )
    return EvidenceRequest(
        episode_id=f"noise_{index}",
        step_id=index + 1,
        instruction="synthetic evidence calibration",
        action=ActionCall(index, action_type, (argument,), f"{action_type} {argument}"),
        pre_image="synthetic_pre.png",
        post_image="synthetic_post.png",
        feedback="",
        last_action_success=True,
        pre_ledger=pre,
    )


def _truth(request: EvidenceRequest) -> dict[PredicateKey, TruthValue]:
    return {
        state.key: (
            TruthValue.TRUE
            if (request.step_id + offset) % 3
            else TruthValue.FALSE
        )
        for offset, state in enumerate(request.pre_ledger)
    }


def _noise_grid() -> list[dict[str, float]]:
    rows = [{"drop": 0.0, "flip": 0.0, "false_positive": 0.0}]
    for rate in (0.05, 0.10, 0.20, 0.30):
        rows.extend(
            (
                {"drop": rate, "flip": 0.0, "false_positive": 0.0},
                {"drop": 0.0, "flip": rate, "false_positive": 0.0},
                {"drop": 0.0, "flip": 0.0, "false_positive": rate},
                {"drop": rate, "flip": rate, "false_positive": rate},
            )
        )
    return rows


def _aggregate(rows: Sequence[dict[str, float]], key: str) -> dict[str, float]:
    values = [float(row[key]) for row in rows]
    return {"mean": mean(values), "std": pstdev(values)}


def _field_label(field: SkillField | None) -> str:
    return "none" if field is None else field.value


def _noisy_mismatches(
    mismatches: Sequence[Mismatch],
    request: EvidenceRequest,
    *,
    noise: dict[str, float],
    seed: int,
) -> tuple[Mismatch, ...]:
    expected = tuple(item.expected for item in mismatches if item.expected is not None)
    evidence = tuple(item.evidence for item in mismatches if item.evidence is not None)
    provider = NoisyEvidenceProvider(
        _FixedProvider(evidence),
        drop_rate=noise["drop"],
        flip_rate=noise["flip"],
        false_positive_rate=noise["false_positive"],
        false_positive_pool=(PredicateKey("spurious", (request.episode_id,)),),
        seed=seed,
    )
    return compare_transitions(expected, provider.extract(request))


def _attribution_metrics(
    noise: dict[str, float], *, seed: int, per_kind: int
) -> dict[str, float]:
    config = load_config("configs/vista_p0.json")
    skill = initialize_shared_skill()
    cases = build_fault_cases(skill, FixedActionSchema(), per_kind=per_kind)
    assigner = CreditAssigner(config=config.attribution)
    gold_targets: list[str] = []
    predicted_targets: list[str] = []
    gold_fields: list[str] = []
    predicted_fields: list[str] = []
    false_skill_updates = 0
    non_skill_cases = 0
    for index, case in enumerate(cases):
        request = _request(10000 + index)
        mismatches = _noisy_mismatches(
            case.mismatches, request, noise=noise, seed=seed
        )
        result = assigner.assign(mismatches, case.context)
        gold_targets.append(case.gold_target.value)
        predicted_targets.append(result.target.value)
        gold_fields.append(_field_label(case.gold_field))
        predicted_fields.append(_field_label(result.field))
        if case.gold_target.value != "skill_update":
            non_skill_cases += 1
            false_skill_updates += int(result.target.value == "skill_update")
    return {
        "target_macro_f1": macro_f1(gold_targets, predicted_targets),
        "field_macro_f1": macro_f1(gold_fields, predicted_fields),
        "non_skill_false_update_rate": false_skill_updates / non_skill_cases,
    }


def main() -> int:
    args = _args()
    requests = [_request(index) for index in range(args.requests)]
    oracle = OracleEvidenceProvider(_truth)
    false_positive_pool = (
        PredicateKey("spurious", ("left",)),
        PredicateKey("spurious", ("right",)),
    )
    conditions: list[dict[str, Any]] = []
    for condition_index, noise in enumerate(_noise_grid()):
        replicates = []
        for replicate in range(args.replicates):
            seed = condition_index * 1000 + replicate
            provider = NoisyEvidenceProvider(
                oracle,
                drop_rate=noise["drop"],
                flip_rate=noise["flip"],
                false_positive_rate=noise["false_positive"],
                false_positive_pool=false_positive_pool,
                seed=seed,
            )
            evidence = compare_providers(oracle, provider, requests, seed=seed)
            attribution = _attribution_metrics(
                noise, seed=seed, per_kind=args.fault_cases_per_kind
            )
            replicates.append(
                {
                    "predicate_precision": evidence["predicate_precision"],
                    "predicate_recall": evidence["predicate_recall"],
                    "predicate_f1": evidence["predicate_f1"],
                    "brier": evidence["brier"],
                    "ece": evidence["expected_calibration_error"],
                    "selective_risk_at_0_5": asdict(
                        evidence["selective_risk_at_0.5"]
                    )["selective_risk"],
                    **attribution,
                }
            )
        metric_names = tuple(replicates[0])
        conditions.append(
            {
                "noise": noise,
                "replicates": args.replicates,
                "metrics": {
                    metric: _aggregate(replicates, metric) for metric in metric_names
                },
            }
        )

    result = {
        "analysis_type": "synthetic_oracle_noise_sensitivity",
        "n_requests": args.requests,
        "replicates": args.replicates,
        "fault_cases_per_kind": args.fault_cases_per_kind,
        "conditions": conditions,
        "limitation": (
            "Synthetic oracle/noise results validate sensitivity and the metrics "
            "harness; they do not estimate Qwen visual-evidence accuracy on natural "
            "Habitat events. Human/simulator gold annotation remains required."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for condition in conditions:
        noise = condition["noise"]
        metrics = condition["metrics"]
        print(
            f"drop={noise['drop']:.2f} flip={noise['flip']:.2f} "
            f"fp={noise['false_positive']:.2f} "
            f"F1={metrics['predicate_f1']['mean']:.3f} "
            f"target={metrics['target_macro_f1']['mean']:.3f} "
            f"field={metrics['field_macro_f1']['mean']:.3f}"
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
