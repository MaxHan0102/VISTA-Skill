"""Live-Qwen stability check for vocabulary-aligned pick-rule repairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vista_skill.clustering import ClusterItem, ClusterKey, EvidenceCluster
from vista_skill.evolution import BoundedPatchApplier
from vista_skill.fault_injection import FaultType, inject_skill_fault
from vista_skill.models import JsonBoundedPatchGenerator, OpenAICompatibleJsonModel
from vista_skill.schemas import (
    AttributionResult,
    DeltaSource,
    ExpectedChange,
    Mismatch,
    MismatchKind,
    PredicateEvidence,
    PredicateKey,
    SkillField,
    TruthValue,
    UpdateTarget,
)
from vista_skill.skills import initialize_shared_skill, skill_digest


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--fault",
        choices=("constraint_pick_multihold", "effect_pick_inversion"),
        default="constraint_pick_multihold",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _repair_case(skill, fault: str):
    if fault == "constraint_pick_multihold":
        return (
            SkillField.CONSTRAINT,
            "constraint_pick_occupies_gripper",
            PredicateKey("not_holding"),
            TruthValue.TRUE,
            TruthValue.FALSE,
            MismatchKind.MISSING_PROGRESS,
        )
    return (
        SkillField.EFFECT,
        "effect_pick_holds_target_category",
        PredicateKey("holding", ("object",)),
        TruthValue.FALSE,
        TruthValue.TRUE,
        MismatchKind.CONTRADICTION,
    )


def _cluster(skill, fault: str) -> tuple[EvidenceCluster, str, TruthValue]:
    field, rule_id, key, expected_after, evidence_after, mismatch_kind = (
        _repair_case(skill, fault)
    )
    evidence = PredicateEvidence(
        key=key,
        before=TruthValue.UNKNOWN,
        after=evidence_after,
        confidence=0.98,
        source="env_feedback",
        evidence_id=f"phase2:{fault}:{key.render()}",
        timestamp=2,
    )
    mismatch = Mismatch(
        mismatch_id=f"phase2-{fault}",
        key=key,
        kind=mismatch_kind,
        expected=ExpectedChange(
            key=key,
            before=TruthValue.UNKNOWN,
            after=expected_after,
            source=DeltaSource.SKILL,
            source_id=(
                f"{skill.skill_id}:v{skill.version}:"
                f"{rule_id}"
            ),
            skill_field=field,
        ),
        evidence=evidence,
        evidence_ids=(evidence.evidence_id,),
    )
    attribution = AttributionResult(
        target=UpdateTarget.SKILL_UPDATE,
        confidence=0.98,
        mismatch_ids=(mismatch.mismatch_id,),
        evidence_ids=(evidence.evidence_id,),
        rationale=f"the injected {fault} pick rule is contradicted",
        field=field,
    )
    cluster = EvidenceCluster(
        ClusterKey(
            skill.skill_id,
            field,
            mismatch_kind.value,
            "place_object_at_receptacle",
            "pick:object",
            skill.version,
        )
    )
    cluster.items.append(ClusterItem("event", "episode", attribution, mismatch))
    return cluster, rule_id, evidence_after


def main() -> int:
    args = _args()
    fault_type = (
        FaultType.CONSTRAINT_PICK_MULTIHOLD
        if args.fault == "constraint_pick_multihold"
        else FaultType.EFFECT_PICK_INVERSION
    )
    parent = inject_skill_fault(initialize_shared_skill(), fault_type)
    cluster, target_rule_id, corrected_value = _cluster(parent, args.fault)
    trials = []
    for seed in range(args.trials):
        model = OpenAICompatibleJsonModel(
            args.model,
            base_url=args.base_url,
            api_key="EMPTY",
            temperature=0.0,
            max_tokens=1024,
            seed=seed,
        )
        record = {"seed": seed}
        try:
            patch = JsonBoundedPatchGenerator(model).propose(parent, cluster)
            candidate = BoundedPatchApplier().apply(parent, patch)
            rules = {rule.rule_id: rule for rule in candidate.prediction_rules}
            target_repaired = rules[target_rule_id].after is corrected_value
            unrelated_preserved = all(
                rule.after is rules[rule.rule_id].after
                for rule in parent.prediction_rules
                if rule.field is patch.field and rule.rule_id != target_rule_id
            )
            record.update(
                {
                    "passed": target_repaired and unrelated_preserved,
                    "patch_id": patch.patch_id,
                    "operation": patch.operation.value,
                    "old": patch.old,
                    "new": patch.new,
                    "target_rule_id": target_rule_id,
                    "target_rule_after": rules[target_rule_id].after.value,
                    "unrelated_same_field_rules_preserved": unrelated_preserved,
                    "candidate_sha256": skill_digest(candidate),
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
        except Exception as error:  # model boundary: retain every failed trial
            record.update(
                {"passed": False, "error": f"{type(error).__name__}: {error}"}
            )
        trials.append(record)
        print(f"seed={seed} passed={record['passed']}")

    result = {
        "base_url": args.base_url,
        "model": args.model,
        "fault": args.fault,
        "trials": trials,
        "passed": sum(item["passed"] for item in trials),
        "total": len(trials),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"passed {result['passed']}/{result['total']}; wrote {args.output}")
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
