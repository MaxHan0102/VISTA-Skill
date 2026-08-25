from __future__ import annotations

import json
from pathlib import Path

from scripts.phase2_patch_stability import _cluster
from vista_skill.action_schema import FixedActionSchema
from vista_skill.fault_injection import FaultType, build_fault_cases, inject_skill_fault
from vista_skill.meta_skills import frozen_meta_skills
from vista_skill.models import JsonAttributionTeacher, JsonBoundedPatchGenerator
from vista_skill.skills import initialize_shared_skill


class _CapturingModel:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_frozen_meta_skill_snapshot_matches_code_and_budget() -> None:
    bundle = frozen_meta_skills()
    snapshot = json.loads(
        Path("configs/phase3c_meta_skills_frozen_v1.json").read_text()
    )
    assert snapshot["version"] == bundle.version
    assert snapshot["sha256"] == bundle.sha256
    assert snapshot["whitespace_token_proxy"] == bundle.whitespace_tokens
    assert snapshot["skills"] == bundle.canonical_payload()["skills"]
    assert bundle.whitespace_tokens <= 1000


def test_attribution_meta_skill_is_added_without_changing_payload_contract() -> None:
    case = build_fault_cases(initialize_shared_skill(), FixedActionSchema())[0]
    model = _CapturingModel(
        {
            "target": "abstain",
            "field": None,
            "subreason": "ambiguous",
            "confidence": 0.5,
            "mismatch_ids": [item.mismatch_id for item in case.mismatches],
            "evidence_ids": [],
            "rationale": "insufficient causal separation",
        }
    )
    skill = frozen_meta_skills().attribute_and_scope
    JsonAttributionTeacher(model, skill).assign(case.mismatches, case.context)
    call = model.calls[0]
    assert skill.instruction in call["system"]
    assert call["purpose"] == "vista_meta_attribution"
    payload = json.loads(call["content"][0]["text"])
    assert set(payload) == {"mismatches", "context", "allowed_targets", "allowed_fields"}


def test_patch_meta_skill_is_added_and_keeps_bounded_schema() -> None:
    parent = inject_skill_fault(
        initialize_shared_skill(), FaultType.CONSTRAINT_PICK_MULTIHOLD
    )
    cluster, _, _ = _cluster(parent, "constraint_pick_multihold")
    old = parent.statements(cluster.key.field)[0]
    model = _CapturingModel(
        {
            "operation": "replace_exact",
            "old": old,
            "new": "Verify the gripper state before picking another object.",
            "scope": "pick actions with an occupied gripper",
            "rationale": "Replay the cited condition and compare affected and protected behavior.",
            "termination_policy": None,
            "prediction_rules": [],
        }
    )
    skill = frozen_meta_skills().patch_and_test
    patch = JsonBoundedPatchGenerator(model, skill).propose(parent, cluster)
    call = model.calls[0]
    assert skill.instruction in call["system"]
    assert call["purpose"] == "vista_meta_bounded_patch"
    assert patch.field is cluster.key.field
    assert patch.evidence_ids == cluster.evidence_ids
