"""Exercise the EVOLUTION teacher path against a live local vLLM endpoint.

The acquisition smoke already validated vista_goal_grounding + vista_visual_evidence
on real Habitat data, and rule-first attribution (which needs no teacher). This
script closes the remaining teacher purposes that only fire during evolution:

  1. vista_attribution  -- JsonAttributionTeacher.assign (the constrained fallback)
  2. vista_bounded_patch -- JsonBoundedPatchGenerator.propose (the patch teacher)
  3. BoundedPatchApplier.apply -- confirm the real vLLM patch applies cleanly

No simulator needed; uses a synthetic skill-sourced contradiction cluster.
"""
from __future__ import annotations

import os
import sys

from vista_skill.clustering import ClusterItem, ClusterKey, EvidenceCluster
from vista_skill.evolution import BoundedPatchApplier
from vista_skill.models import (
    JsonAttributionTeacher,
    JsonBoundedPatchGenerator,
    OpenAICompatibleJsonModel,
)
from vista_skill.schemas import (
    AttributionContext,
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
from vista_skill.skills import initialize_shared_skill

BASE_URL = os.environ.get("VISTA_METHOD_BASE_URL", "http://127.0.0.1:8001/v1")
MODEL = os.environ.get("VISTA_METHOD_MODEL", "Qwen/Qwen3-VL-8B-Instruct")


def main() -> int:
    skill = initialize_shared_skill()
    model = OpenAICompatibleJsonModel(
        MODEL, base_url=BASE_URL, api_key="EMPTY", temperature=0.0, max_tokens=512, seed=0
    )
    fails = 0

    # 1. attribution teacher (the constrained fallback path) ----------------
    try:
        key = PredicateKey("holding", ("ball_1",))
        expected = ExpectedChange(
            key, TruthValue.FALSE, TruthValue.TRUE, DeltaSource.SKILL,
            f"{skill.skill_id}:v{skill.version}:effect", SkillField.EFFECT,
        )
        evidence = PredicateEvidence(
            key, TruthValue.FALSE, TruthValue.FALSE, 0.92,
            "active_observation", "ev1", 1,
        )
        mismatch = Mismatch(
            "m1", key, MismatchKind.CONTRADICTION, expected, evidence, ("ev1",)
        )
        context = AttributionContext(
            executor_followed_skill=True,
            instruction="pick up the ball",
            action_type="pick",
            skill_obligations={f.value: skill.statements(f) for f in SkillField},
        )
        result = JsonAttributionTeacher(model).assign([mismatch], context)
        ok = (
            isinstance(result, AttributionResult)
            and result.target.value in {"belief_refresh", "skill_update", "abstain"}
        )
        billed = model.usage.get("vista_attribution")
        print(f"[{'PASS' if ok else 'FAIL'}] attribution.assign: target={result.target.value} "
              f"field={result.field} conf={result.confidence} | billed={bool(billed)}")
        if not (ok and billed):
            fails += 1
    except Exception as exc:  # noqa: BLE001
        fails += 1
        print(f"[FAIL] attribution.assign: {exc!r}")

    # 2. bounded patch generator -------------------------------------------
    try:
        cluster = _termination_cluster(skill)
        patch = JsonBoundedPatchGenerator(model).propose(skill, cluster)
        ok_field = patch.field is cluster.key.field
        billed = model.usage.get("vista_bounded_patch")
        print(f"[{'PASS' if ok_field else 'FAIL'}] patch.propose: op={patch.operation.value} "
              f"field={patch.field.value} new={patch.new[:45]!r} | billed={bool(billed)}")
        if not (ok_field and billed):
            fails += 1
    except Exception as exc:  # noqa: BLE001
        fails += 1
        print(f"[FAIL] patch.propose: {exc!r}")

    # 3. the real vLLM patch must apply cleanly to the active skill ---------
    try:
        candidate = BoundedPatchApplier().apply(skill, patch)
        ok = candidate.version == skill.version + 1
        print(f"[{'PASS' if ok else 'FAIL'}] patch.apply: v{skill.version} -> v{candidate.version} "
              f"field '{patch.field.value}' now {len(candidate.statements(patch.field))} stmts")
        if not ok:
            fails += 1
    except Exception as exc:  # noqa: BLE001
        fails += 1
        print(f"[FAIL] patch.apply: {exc!r}")

    print(f"\nteacher billing: { {k: vars(v) for k, v in model.usage.items()} }")
    print(f"{3 - fails}/3 evolution-teacher checks passed")
    return 0 if fails == 0 else 1


def _termination_cluster(skill) -> EvidenceCluster:
    key = PredicateKey("task_complete")
    expected = ExpectedChange(
        key, TruthValue.FALSE, TruthValue.TRUE, DeltaSource.SKILL,
        f"{skill.skill_id}:v{skill.version}:termination", SkillField.TERMINATION,
    )
    observed = PredicateEvidence(
        key, TruthValue.FALSE, TruthValue.FALSE, 0.9,
        "visual_pair", "ev1", 1,
    )
    mismatch = Mismatch(
        "m1", key, MismatchKind.TERMINATION_CONFLICT, expected, observed, ("ev1", "ev2")
    )
    attribution = AttributionResult(
        target=UpdateTarget.SKILL_UPDATE,
        confidence=0.9,
        mismatch_ids=("m1",),
        evidence_ids=("ev1", "ev2"),
        rationale="termination defect",
        field=SkillField.TERMINATION,
    )
    cluster = EvidenceCluster(
        ClusterKey(skill.skill_id, SkillField.TERMINATION, "termination_conflict", "all", "objects")
    )
    cluster.items.append(ClusterItem("event", "episode", attribution, mismatch))
    return cluster


if __name__ == "__main__":
    sys.exit(main())
