"""Frozen, environment-neutral procedures for safe Skill evolution.

These are instructions, not learned components.  Keeping the text in one
content-addressed bundle makes a Phase3C arm reproducible and prevents a
different prompt from being silently used after outcomes are inspected.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass


META_SKILL_VERSION = "phase3c_frozen_v1"


@dataclass(frozen=True)
class EvolutionMetaSkill:
    name: str
    stage: str
    allowed_inputs: tuple[str, ...]
    instruction: str


@dataclass(frozen=True)
class FrozenEvolutionMetaSkills:
    observe_and_recover: EvolutionMetaSkill
    attribute_and_scope: EvolutionMetaSkill
    patch_and_test: EvolutionMetaSkill
    version: str = META_SKILL_VERSION

    def canonical_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "skills": [
                {
                    "name": item.name,
                    "stage": item.stage,
                    "allowed_inputs": list(item.allowed_inputs),
                    "instruction": item.instruction,
                }
                for item in (
                    self.observe_and_recover,
                    self.attribute_and_scope,
                    self.patch_and_test,
                )
            ],
        }

    @property
    def sha256(self) -> str:
        raw = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @property
    def whitespace_tokens(self) -> int:
        return sum(
            len(item.instruction.split())
            for item in (
                self.observe_and_recover,
                self.attribute_and_scope,
                self.patch_and_test,
            )
        )


_OBSERVE_AND_RECOVER = """Observation-and-recovery procedure:
1. Before an irreversible action, or before repeating a failed action, classify every decision-critical fact as supported, unknown, or conflicted.
2. If an unknown fact could change the next action and a legal low-cost observation is available, take one targeted observation that can resolve it. Do not treat absence from one view as false.
3. Treat action feedback as evidence, not as an infallible fact. Compare it with the next observation and the local action history.
4. When sources conflict, avoid repeating the unchanged action. Change viewpoint, re-observe the affected entity, or verify a prerequisite. Use the cheapest legal recovery that can distinguish the competing explanations.
5. Proceed when the required facts are supported. If they remain unobservable after the available checks, preserve them as unknown and choose a reversible alternative."""


_ATTRIBUTE_AND_SCOPE = """Causal attribution procedure for persistent memory:
1. Work only from cited transition evidence. If visibility, coverage, identity, or cause is ambiguous, abstain from a persistent update.
2. First exclude an execution lapse, an unmet prerequisite, and a suspected stochastic outcome. These do not justify changing a persistent procedure.
3. Use belief_refresh when the contradiction concerns stale local state, unknown treated as false, object identity, or an expectation that did not originate in a persistent procedure field.
4. Use skill_update only when the executor followed the relevant procedure, supported evidence contradicts a procedure-sourced expectation, and the cited expectation identifies one field. Prefer recurring or independently supported evidence; otherwise abstain.
5. For skill_update, map only the implicated contract: activation for when it applies; procedure for action order; effect for predicted results; constraint for prerequisites or prohibitions; termination for completion. Never infer a field from the desired goal alone.
6. Cite only supplied mismatch and evidence identifiers. A field is legal only with skill_update."""


_PATCH_AND_TEST = """Bounded patch-and-test procedure:
1. Change only the attributed field and the smallest exact statement needed to resolve the cited recurring contradiction. Do not rewrite the whole procedure.
2. When the field has a compiled contract, repair the compiled value or rule as well as its text. Preserve every unrelated statement and compiled rule exactly.
3. Generalize over action type, predicate, and role; never encode an episode, scene, numbered object instance, benchmark, or observed outcome as a special case.
4. Bind the patch to supplied evidence identifiers. Do not add unsupported facts.
5. In the rationale, state a falsifiable test intent: replay the cited condition, compare parent and candidate on the affected transition, and check unrelated protected behavior for regression. The deterministic gate, not this procedure, makes the acceptance decision."""


def frozen_meta_skills() -> FrozenEvolutionMetaSkills:
    """Return the immutable Phase3C instruction bundle."""
    bundle = FrozenEvolutionMetaSkills(
        observe_and_recover=EvolutionMetaSkill(
            name="Observe-and-Recover",
            stage="executor_observation",
            allowed_inputs=(
                "current_action_surface",
                "legal_observation_capabilities",
                "unresolved_local_facts",
                "public_feedback",
                "local_action_history",
            ),
            instruction=_OBSERVE_AND_RECOVER,
        ),
        attribute_and_scope=EvolutionMetaSkill(
            name="Attribute-and-Scope",
            stage="transition_attribution",
            allowed_inputs=("mismatches", "attribution_context"),
            instruction=_ATTRIBUTE_AND_SCOPE,
        ),
        patch_and_test=EvolutionMetaSkill(
            name="Patch-and-Test",
            stage="bounded_patch_proposal",
            allowed_inputs=(
                "attributed_field",
                "current_field_statements",
                "recurring_evidence_cluster",
                "compiled_field_contract",
            ),
            instruction=_PATCH_AND_TEST,
        ),
    )
    lint_meta_skills(bundle)
    return bundle


def lint_meta_skills(bundle: FrozenEvolutionMetaSkills) -> None:
    """Fail closed on the frozen generality and budget contract."""
    if bundle.version != META_SKILL_VERSION:
        raise ValueError("unexpected frozen Meta-Skill version")
    if bundle.whitespace_tokens > 1000:
        raise ValueError("Meta-Skill bundle exceeds the 1000-token proxy budget")
    text = "\n".join(
        item.instruction
        for item in (
            bundle.observe_and_recover,
            bundle.attribute_and_scope,
            bundle.patch_and_test,
        )
    )
    banned = ("EmbodiedBench", "EB-HAB", "EB-NAV")
    for value in banned:
        if value.casefold() in text.casefold():
            raise ValueError(f"environment-specific Meta-Skill wording: {value}")
    # Numbered entity identities are a common route to instance overfitting.
    if re.search(r"\b[a-z][a-z_]*_[0-9]+\b", text, flags=re.IGNORECASE):
        raise ValueError("instance-specific identifier in Meta-Skill text")
