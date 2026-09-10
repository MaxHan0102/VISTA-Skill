"""Apply the frozen Phase3C offline Go/No-Go criteria without model calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from vista_skill.meta_skills import frozen_meta_skills


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("running/Phase3/phase3c/offline"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path):
    return json.loads(path.read_text())


def _false_updates(metrics: dict) -> int:
    return sum(
        int(count)
        for key, count in metrics["confusion"].items()
        if not key.startswith("skill_update->") and key.endswith("->skill_update")
    )


def _natural_report(paths, condition):
    metrics = [_read(path)["conditions"][condition]["metrics"] for path in paths]
    return {
        "files": [str(path) for path in paths],
        "target_macro_f1_mean": mean(item["target_macro_f1"] for item in metrics),
        "field_macro_f1_mean": mean(item["field_macro_f1"] for item in metrics),
        "skill_update_f1_mean": mean(item["skill_update_f1"] for item in metrics),
        "false_updates_total": sum(_false_updates(item) for item in metrics),
        "cases_total": sum(int(item["cases"]) for item in metrics),
    }


def main() -> int:
    args = _args()
    root = args.root
    bundle = frozen_meta_skills()
    synthetic_meta = _read(root / "synthetic_meta.json")
    synthetic_current = _read(Path("running/Phase2/phase2_teacher_attribution_173_20260824.json"))
    synthetic = {
        "current_target_macro_f1": synthetic_current["target_macro_f1"]["mean"],
        "meta_target_macro_f1": synthetic_meta["target_macro_f1"]["mean"],
        "current_field_macro_f1": synthetic_current["field_macro_f1"]["mean"],
        "meta_field_macro_f1": synthetic_meta["field_macro_f1"]["mean"],
        "invalid_count": sum(item["invalid_count"] for item in synthetic_meta["seed_reports"]),
    }
    synthetic["target_delta"] = (
        synthetic["meta_target_macro_f1"] - synthetic["current_target_macro_f1"]
    )
    synthetic["field_delta"] = (
        synthetic["meta_field_macro_f1"] - synthetic["current_field_macro_f1"]
    )
    synthetic["pass"] = bool(
        synthetic["target_delta"] >= 0.10
        and synthetic["field_delta"] >= -0.02
        and synthetic["invalid_count"] == 0
    )

    current_paths = {
        "constraint_pick_multihold": [
            Path("running/Phase2/phase2_natural_attribution_weak_audit_173_20260824.json"),
            *[
                Path(f"running/Phase2/phase2_natural_attribution_weak_audit_seed{seed}_173_20260824.json")
                for seed in range(1, 5)
            ],
        ],
        "effect_pick_inversion": [
            Path(f"running/Phase2/phase2_effect_natural_attribution_audit_seed{seed}_173_20260824.json")
            for seed in range(5)
        ],
    }
    natural = {}
    merged_current_false = 0
    merged_meta_false = 0
    for fault, current_files in current_paths.items():
        meta_files = [root / f"natural_{fault}_seed{seed}.json" for seed in range(5)]
        current_direct = _natural_report(current_files, "direct_qwen_teacher")
        meta_direct = _natural_report(meta_files, "direct_qwen_teacher")
        current_rule = _natural_report(current_files, "recorded_rule_first")
        meta_rule = _natural_report(meta_files, "rule_first_with_teacher")
        direct_pass = meta_direct["skill_update_f1_mean"] >= current_direct["skill_update_f1_mean"]
        rule_pass = bool(
            meta_rule["target_macro_f1_mean"] >= current_rule["target_macro_f1_mean"]
            and meta_rule["field_macro_f1_mean"] >= current_rule["field_macro_f1_mean"]
        )
        natural[fault] = {
            "current_direct": current_direct,
            "meta_direct": meta_direct,
            "current_rule_first": current_rule,
            "meta_rule_first": meta_rule,
            "direct_skill_update_f1_noninferior": direct_pass,
            "rule_first_target_and_field_noninferior": rule_pass,
        }
        merged_current_false += current_rule["false_updates_total"]
        merged_meta_false += meta_rule["false_updates_total"]
    natural_false_update_pass = merged_meta_false <= merged_current_false
    natural_pass = bool(
        all(
            item["direct_skill_update_f1_noninferior"]
            and item["rule_first_target_and_field_noninferior"]
            for item in natural.values()
        )
        and natural_false_update_pass
    )

    patches = {}
    for fault in ("constraint_pick_multihold", "effect_pick_inversion"):
        payload = _read(root / f"patch_{fault}_meta.json")
        neutral = sum(bool(item.get("environment_neutral")) for item in payload["trials"])
        patches[fault] = {
            "passed": payload["passed"],
            "total": payload["total"],
            "environment_neutral": neutral,
            "test_intent_count": sum(
                bool(item.get("has_test_intent")) for item in payload["trials"]
            ),
            "pass": payload["passed"] >= 9 and neutral == payload["total"],
        }
    patch_pass = all(item["pass"] for item in patches.values())

    metamorphic_raw = _read(root / "metamorphic_meta.json")
    metamorphic = {
        "agreement": metamorphic_raw["agreement"],
        "invalid_count": metamorphic_raw["invalid_count"],
        "pass": bool(
            metamorphic_raw["agreement"] >= 0.90
            and metamorphic_raw["invalid_count"] == 0
        ),
    }
    snapshot = _read(Path("configs/phase3c_meta_skills_frozen_v1.json"))
    access_and_freeze = {
        "version": bundle.version,
        "code_sha256": bundle.sha256,
        "snapshot_sha256": snapshot["sha256"],
        "whitespace_token_proxy": bundle.whitespace_tokens,
        "pass": bool(
            snapshot["sha256"] == bundle.sha256
            and snapshot["skills"] == bundle.canonical_payload()["skills"]
            and bundle.whitespace_tokens <= 1000
        ),
    }
    criteria = {
        "synthetic": synthetic["pass"],
        "natural": natural_pass,
        "patch": patch_pass,
        "metamorphic": metamorphic["pass"],
        "access_and_freeze": access_and_freeze["pass"],
    }
    decision = "go" if all(criteria.values()) else "no_go"
    result = {
        "analysis_type": "phase3c_frozen_offline_gate",
        "meta_skill_version": bundle.version,
        "meta_skill_sha256": bundle.sha256,
        "criteria": criteria,
        "decision": decision,
        "synthetic": synthetic,
        "natural": natural,
        "merged_rule_first_false_updates": {
            "current": merged_current_false,
            "meta": merged_meta_false,
            "pass": natural_false_update_pass,
        },
        "patches": patches,
        "metamorphic": metamorphic,
        "access_and_freeze": access_and_freeze,
        "conditional_branches": {
            "executor_pilot": "eligible" if decision == "go" else "cancelled_by_offline_gate",
            "eb_hab_live_evolution": "pending_executor_gate" if decision == "go" else "cancelled_by_offline_gate",
            "eb_nav_zero_shot": "pending_executor_gate" if decision == "go" else "cancelled_by_offline_gate",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "criteria": criteria}, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
