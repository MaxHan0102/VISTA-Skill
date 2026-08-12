from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from vista_skill.attribution import AttributionConfig
from vista_skill.belief import MergePolicy
from vista_skill.clustering import RecurrencePolicy
from vista_skill.evolution import GateConfig, PatchPolicy
from vista_skill.protocol import manifest_digest


@dataclass(frozen=True)
class VistaConfig:
    raw: Mapping[str, Any]
    digest: str
    belief: MergePolicy
    attribution: AttributionConfig
    recurrence: RecurrencePolicy
    patch: PatchPolicy
    gate: GateConfig


def load_config(path: str | Path) -> VistaConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    required_sections = {
        "protocol",
        "task_manifest",
        "evolution_seeds",
        "environment",
        "executor",
        "budgets",
        "credit",
        "gate",
        "final_test",
    }
    missing = sorted(required_sections - raw.keys())
    if missing:
        raise ValueError(f"missing required config sections: {missing}")
    credit = raw.get("credit", {})
    budgets = raw.get("budgets", {})
    gate = raw.get("gate", {})
    config = VistaConfig(
        raw=raw,
        digest=manifest_digest(raw),
        belief=MergePolicy(),
        attribution=AttributionConfig(
            min_evidence_confidence=float(credit.get("min_evidence_confidence", 0.75)),
            min_teacher_confidence=float(credit.get("min_attribution_confidence", 0.70)),
            action_model_updates_enabled=bool(
                credit.get("action_model_updates_enabled", False)
            ),
        ),
        recurrence=RecurrencePolicy(
            min_independent_episodes=int(credit.get("min_independent_episodes", 2)),
            min_evidence_confidence=float(credit.get("min_evidence_confidence", 0.75)),
            min_attribution_confidence=float(
                credit.get("min_attribution_confidence", 0.70)
            ),
        ),
        patch=PatchPolicy(
            max_operations=int(budgets.get("patch_operations_per_round", 1)),
            max_active_skill_words=int(budgets.get("active_skill_words", 512)),
        ),
        gate=GateConfig(
            bootstrap_samples=int(gate.get("bootstrap_samples", 2000)),
            alpha=float(gate.get("alpha", 0.05)),
            proxy_episode_budget=int(budgets.get("proxy_episodes", 10)),
            finalist_episode_budget=int(budgets.get("finalist_episodes", 30)),
            proxy_lcb_threshold=float(gate.get("proxy_lcb_threshold", 0.0)),
            finalist_lcb_threshold=float(gate.get("finalist_lcb_threshold", 0.0)),
            subgroup_regression_tolerance=float(
                gate.get("subgroup_regression_tolerance", 0.05)
            ),
        ),
    )
    if config.recurrence.min_independent_episodes < 1:
        raise ValueError("min_independent_episodes must be positive")
    if config.patch.max_operations != 1:
        raise ValueError("P0 requires exactly one atomic patch operation per round")
    if config.gate.proxy_episode_budget < 1 or config.gate.finalist_episode_budget < 1:
        raise ValueError("paired gate budgets must be positive")
    if config.gate.proxy_episode_budget > config.gate.finalist_episode_budget:
        raise ValueError("proxy budget cannot exceed finalist budget")
    evolution_seeds = tuple(int(item) for item in raw["evolution_seeds"])
    if not evolution_seeds or len(set(evolution_seeds)) != len(evolution_seeds):
        raise ValueError("evolution_seeds must be non-empty and unique")
    if credit.get("action_model_updates_enabled", False):
        raise ValueError("persistent action-model updates are disabled in VISTA-Skill P0")
    final_test = raw["final_test"]
    if not final_test.get("skill_frozen", False) or any(
        final_test.get(key, True)
        for key in ("teacher_enabled", "attribution_enabled", "patching_enabled")
    ):
        raise ValueError("final-test protocol must freeze skill and disable evolution")
    return config
