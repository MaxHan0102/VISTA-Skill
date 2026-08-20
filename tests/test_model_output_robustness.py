"""Model-returned junk predicates must degrade, never abort, the method loop.

Live-server runs hit model strings (e.g. ``""``) that ``PredicateKey.parse``
rejects; both the goal grounder and the visual evidence provider must drop the
offending entry and keep the rollout alive (same contract as the existing
JSONDecodeError degradation in ``JsonVisualEvidenceProvider.extract``).
"""
from __future__ import annotations

import json

from vista_skill.models import (
    JsonBoundedPatchGenerator,
    JsonGoalGrounder,
    JsonVisualEvidenceProvider,
)
from vista_skill.schemas import ActionCall, EvidenceRequest, PredicateKey


class _FakeJsonModel:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.schemas: list[dict] = []

    def complete_json(self, *, system, content, schema, purpose) -> dict:
        self.schemas.append(schema)
        return dict(self.payload)


def _request(tmp_path) -> EvidenceRequest:
    pre = tmp_path / "pre.png"
    post = tmp_path / "post.png"
    pre.write_bytes(b"png")
    post.write_bytes(b"png")
    return EvidenceRequest(
        episode_id="ep1",
        step_id=1,
        instruction="pick the apple",
        action=ActionCall(0, "pick", ("apple_1",), "pick apple_1"),
        pre_image=str(pre),
        post_image=str(post),
        feedback="object picked up",
        last_action_success=True,
        pre_ledger=(),
        goal_predicates=(),
    )


def _observation(predicate: str) -> dict:
    return {
        "predicate": predicate,
        "value": "true",
        "confidence": 0.9,
        "coverage": 0.9,
        "evidence": "visible in frame",
    }


def _image(tmp_path):
    path = tmp_path / "initial.png"
    path.write_bytes(b"png")
    return str(path)


def test_goal_grounder_drops_unparseable_predicates(tmp_path) -> None:
    model = _FakeJsonModel(
        {"goal_predicates": ["", "()", "at(apple_1, fridge_1)", "holding(apple_1)"]}
    )
    grounded = JsonGoalGrounder(model).ground(
        "pick the apple", _image(tmp_path), (("pick", ("apple_1",)),)
    )
    assert grounded == (
        PredicateKey("at", ("apple_1", "fridge_1")),
        PredicateKey("holding", ("apple_1",)),
    )


def test_goal_grounder_all_junk_returns_empty(tmp_path) -> None:
    model = _FakeJsonModel({"goal_predicates": ["", "()"]})
    grounded = JsonGoalGrounder(model).ground(
        "pick the apple", _image(tmp_path), (("pick", ("apple_1",)),)
    )
    assert grounded == ()


def test_goal_grounder_schema_forbids_empty_strings(tmp_path) -> None:
    model = _FakeJsonModel({"goal_predicates": ["at(apple_1)"]})
    JsonGoalGrounder(model).ground(
        "pick the apple", _image(tmp_path), (("pick", ("apple_1",)),)
    )
    items = model.schemas[0]["properties"]["goal_predicates"]
    assert items["minItems"] == 1
    assert items["items"]["minLength"] == 1


def test_evidence_provider_skips_unparseable_predicates(tmp_path) -> None:
    model = _FakeJsonModel(
        {
            "observations": [
                _observation(""),
                _observation("holding(apple_1)"),
                _observation("()"),
            ]
        }
    )
    evidence = JsonVisualEvidenceProvider(model).extract(_request(tmp_path))
    assert len(evidence) == 1
    assert evidence[0].key == PredicateKey("holding", ("apple_1",))


def test_evidence_schema_forbids_empty_predicate_strings(tmp_path) -> None:
    model = _FakeJsonModel({"observations": []})
    JsonVisualEvidenceProvider(model).extract(_request(tmp_path))
    observation = model.schemas[0]["properties"]["observations"]["items"]
    assert observation["properties"]["predicate"]["minLength"] == 1


def test_patch_generator_payload_exposes_compiled_context() -> None:
    """The teacher must see the compiled view to emit a repairing patch.

    A text-only patch leaves compiled predictions identical, so the gate
    rejects it ("target was not repaired"). The generator therefore has to
    show the current termination policy / field rules and the compiled
    contract (E5/E6 rejections traced to this information gap).
    """
    from vista_skill.baselines import CommonUpdateProposal, proposal_cluster
    from vista_skill.fault_injection import FaultType, inject_skill_fault
    from vista_skill.schemas import SkillField, TerminationPolicy, UpdateTarget
    from vista_skill.skills import initialize_shared_skill

    skill = inject_skill_fault(initialize_shared_skill(), FaultType.TERMINATION)
    proposal = CommonUpdateProposal(
        UpdateTarget.SKILL_UPDATE,
        SkillField.TERMINATION,
        "Executor stops after a single goal is satisfied.",
        ("ev1", "ev2"),
        "unconditional_trajectory_reflection",
        True,
    )
    cluster = proposal_cluster(skill, proposal, ("ep1", "ep2"))
    captured: dict = {}

    class _CapturingModel:
        def complete_json(self, *, system, content, schema, purpose) -> dict:
            captured["payload"] = json.loads(content[0]["text"])
            return {
                "operation": "replace_exact",
                "old": "Stop after any one required goal is supported.",
                "new": "Stop only when every goal predicate is supported.",
                "scope": "termination",
                "rationale": "restore all-goals termination",
                "termination_policy": "all_goals_evidence",
                "prediction_rules": [],
            }

    patch = JsonBoundedPatchGenerator(_CapturingModel()).propose(skill, cluster)
    payload = captured["payload"]
    assert payload["compiled_termination_policy"] == "any_goal_evidence"
    assert "termination_policy" in payload["compiled_contract"]
    assert isinstance(payload["field_prediction_rules"], list)
    assert patch.termination_policy is TerminationPolicy.ALL_GOALS_EVIDENCE
    assert patch.prediction_rules == ()
