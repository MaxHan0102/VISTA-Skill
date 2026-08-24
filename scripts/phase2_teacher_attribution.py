"""Direct-Qwen attribution baseline on the controlled fault diagnostic set."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

from vista_skill.action_schema import FixedActionSchema
from vista_skill.fault_injection import build_fault_cases
from vista_skill.metrics import macro_f1
from vista_skill.models import JsonAttributionTeacher, OpenAICompatibleJsonModel
from vista_skill.skills import initialize_shared_skill


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _field(value) -> str:
    return "none" if value is None else value.value


def main() -> int:
    args = _args()
    skill = initialize_shared_skill()
    cases = build_fault_cases(skill, FixedActionSchema())
    seed_reports = []
    for seed in range(args.seeds):
        model = OpenAICompatibleJsonModel(
            args.model,
            base_url=args.base_url,
            api_key="EMPTY",
            temperature=0.0,
            max_tokens=512,
            seed=seed,
        )
        teacher = JsonAttributionTeacher(model)
        rows = []
        for index, case in enumerate(cases):
            try:
                result = teacher.assign(case.mismatches, case.context)
                predicted_target = result.target.value
                predicted_field = _field(result.field)
                error = None
                confidence = result.confidence
            except Exception as exc:  # count malformed teacher decisions as invalid
                predicted_target = "invalid"
                predicted_field = "invalid"
                error = f"{type(exc).__name__}: {exc}"
                confidence = 0.0
            rows.append(
                {
                    "case_index": index,
                    "fault_type": case.fault_type.value,
                    "gold_target": case.gold_target.value,
                    "predicted_target": predicted_target,
                    "gold_field": _field(case.gold_field),
                    "predicted_field": predicted_field,
                    "confidence": confidence,
                    "error": error,
                }
            )
            print(
                f"seed={seed} case={index + 1}/{len(cases)} "
                f"gold={case.gold_target.value} pred={predicted_target}"
            )
        target_f1 = macro_f1(
            [row["gold_target"] for row in rows],
            [row["predicted_target"] for row in rows],
        )
        field_f1 = macro_f1(
            [row["gold_field"] for row in rows],
            [row["predicted_field"] for row in rows],
        )
        seed_reports.append(
            {
                "seed": seed,
                "target_macro_f1": target_f1,
                "field_macro_f1": field_f1,
                "invalid_count": sum(row["error"] is not None for row in rows),
                "confusion": dict(
                    Counter(
                        f"{row['gold_target']}->{row['predicted_target']}"
                        for row in rows
                    )
                ),
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
        )
        print(f"seed={seed} target={target_f1:.3f} field={field_f1:.3f}")
    target_values = [row["target_macro_f1"] for row in seed_reports]
    field_values = [row["field_macro_f1"] for row in seed_reports]
    result = {
        "analysis_type": "direct_teacher_attribution_without_rule_first",
        "base_url": args.base_url,
        "model": args.model,
        "n_cases": len(cases),
        "seeds": args.seeds,
        "target_macro_f1": {
            "mean": mean(target_values),
            "std": pstdev(target_values),
        },
        "field_macro_f1": {
            "mean": mean(field_values),
            "std": pstdev(field_values),
        },
        "seed_reports": seed_reports,
        "limitation": (
            "The controlled cases are synthetic and generated from the same fault "
            "taxonomy as the method; results are mechanism diagnostics, not natural-event accuracy."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"target={result['target_macro_f1']} field={result['field_macro_f1']} "
        f"wrote {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
