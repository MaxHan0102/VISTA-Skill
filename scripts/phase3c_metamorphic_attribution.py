"""Label-preserving metamorphic audit for the frozen attribution Meta-Skill."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from vista_skill.action_schema import FixedActionSchema
from vista_skill.fault_injection import build_fault_cases
from vista_skill.meta_skills import frozen_meta_skills
from vista_skill.models import JsonAttributionTeacher, OpenAICompatibleJsonModel
from vista_skill.schemas import Mismatch, PredicateKey
from vista_skill.skills import initialize_shared_skill


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _rename_key(key: PredicateKey, names: dict[str, str]) -> PredicateKey:
    return PredicateKey(key.name, tuple(names.get(item, item) for item in key.arguments))


def _rename_mismatch(item: Mismatch, index: int) -> Mismatch:
    names = {
        value: f"entity_{chr(ord('a') + offset)}"
        for offset, value in enumerate(dict.fromkeys(item.key.arguments))
    }
    key = _rename_key(item.key, names)
    expected = (
        None if item.expected is None else replace(item.expected, key=key)
    )
    evidence = (
        None
        if item.evidence is None
        else replace(
            item.evidence,
            key=key,
            evidence_id=f"evidence_renamed_{index}",
        )
    )
    return replace(
        item,
        mismatch_id=f"mismatch_renamed_{index}",
        key=key,
        expected=expected,
        evidence=evidence,
        evidence_ids=(f"evidence_renamed_{index}",) if evidence is not None else (),
    )


def _id_only(item: Mismatch, index: int) -> Mismatch:
    evidence = (
        None
        if item.evidence is None
        else replace(item.evidence, evidence_id=f"citation_alpha_{index}")
    )
    return replace(
        item,
        mismatch_id=f"difference_alpha_{index}",
        evidence=evidence,
        evidence_ids=(f"citation_alpha_{index}",) if evidence is not None else (),
    )


def _predict(teacher, mismatches, context):
    result = teacher.assign(mismatches, context)
    return {
        "target": result.target.value,
        "field": "none" if result.field is None else result.field.value,
    }


def main() -> int:
    args = _args()
    bundle = frozen_meta_skills()
    model = OpenAICompatibleJsonModel(
        args.model,
        base_url=args.base_url,
        api_key="EMPTY",
        temperature=0.0,
        max_tokens=512,
        seed=args.seed,
    )
    teacher = JsonAttributionTeacher(model, bundle.attribute_and_scope)
    cases = build_fault_cases(initialize_shared_skill(), FixedActionSchema())
    rows = []
    agreements = []
    invalid = 0
    for case_index, case in enumerate(cases):
        variants = {
            "identifier_rename": tuple(
                _id_only(item, index) for index, item in enumerate(case.mismatches)
            ),
            "entity_rename": tuple(
                _rename_mismatch(item, index)
                for index, item in enumerate(case.mismatches)
            ),
            "input_order": tuple(reversed(case.mismatches)),
        }
        record = {"case_index": case_index, "fault_type": case.fault_type.value}
        try:
            canonical = _predict(teacher, case.mismatches, case.context)
            record["canonical"] = canonical
        except Exception as exc:
            canonical = None
            record["canonical_error"] = f"{type(exc).__name__}: {exc}"
            invalid += 1
        variant_records = {}
        for name, mismatches in variants.items():
            try:
                prediction = _predict(teacher, mismatches, case.context)
                agrees = canonical is not None and prediction == canonical
                agreements.append(agrees)
                variant_records[name] = {**prediction, "agrees": agrees}
            except Exception as exc:
                invalid += 1
                agreements.append(False)
                variant_records[name] = {
                    "agrees": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        record["variants"] = variant_records
        rows.append(record)
        print(
            f"case={case_index + 1}/{len(cases)} agreement="
            f"{sum(bool(x) for x in agreements)}/{len(agreements)}"
        )
    result = {
        "analysis_type": "phase3c_label_preserving_metamorphic_attribution",
        "model": args.model,
        "base_url": args.base_url,
        "seed": args.seed,
        "meta_skill_sha256": bundle.sha256,
        "cases": len(cases),
        "variant_predictions": len(agreements),
        "agreement": sum(agreements) / len(agreements),
        "invalid_count": invalid,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"agreement={result['agreement']:.3f}; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
