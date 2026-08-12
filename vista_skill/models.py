from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from vista_skill.baselines import (
    EmbodiSkillRoute,
    EpisodeSummary,
    TrajectoryReflection,
)
from vista_skill.evolution import PatchGenerator, make_patch_id
from vista_skill.clustering import EvidenceCluster
from vista_skill.schemas import (
    AbstainReason,
    AttributionContext,
    AttributionResult,
    EvidenceRequest,
    EvidenceSource,
    Mismatch,
    PatchOperation,
    PredicateEvidence,
    PredicateKey,
    SkillField,
    SkillPatch,
    SkillPredictionRule,
    SkillSpec,
    TerminationPolicy,
    TruthValue,
    UpdateTarget,
)


class JsonModel(Protocol):
    def complete_json(
        self,
        *,
        system: str,
        content: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any],
        purpose: str,
    ) -> Mapping[str, Any]: ...


@dataclass
class UsageCounter:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OpenAICompatibleJsonModel:
    """Optional OpenAI-compatible backend with per-purpose usage accounting."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError("install the optional openai package to use this backend") from error
            kwargs = {"base_url": base_url, "api_key": api_key}
            client = OpenAI(**{key: value for key, value in kwargs.items() if value is not None})
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = None if seed is None else int(seed)
        self.usage: dict[str, UsageCounter] = {}

    def complete_json(
        self,
        *,
        system: str,
        content: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any],
        purpose: str,
    ) -> Mapping[str, Any]:
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": list(content)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": purpose, "strict": True, "schema": schema},
            },
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            request["seed"] = self.seed
        response = self.client.chat.completions.create(
            **request,
        )
        counter = self.usage.setdefault(purpose, UsageCounter())
        counter.calls += 1
        usage = getattr(response, "usage", None)
        counter.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        counter.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        raw = response.choices[0].message.content
        return json.loads(raw)


class JsonVisualEvidenceProvider:
    """Pre/post visual evidence provider with a leakage-safe request contract."""

    def __init__(self, model: JsonModel) -> None:
        self.model = model

    def extract(self, request: EvidenceRequest) -> Sequence[PredicateEvidence]:
        queries = _evidence_queries(request)
        prompt = {
            "instruction": request.instruction,
            "executed_action": {
                "type": request.action.action_type,
                "arguments": request.action.arguments,
                "text": request.action.text,
            },
            "public_environment_feedback": request.feedback,
            "pre_action_belief": [
                {"predicate": item.key.render(), "value": item.value.value}
                for item in request.pre_ledger
            ],
            "query_predicates": [key.render() for key in queries],
            "rule": "Not visible means unknown, not false. Preserve numbered instance identities.",
        }
        result = self.model.complete_json(
            system=(
                "Extract only visual evidence supported by the two images and public feedback. "
                "Do not infer desired or predicted effects. Return unknown when coverage is insufficient."
            ),
            content=[
                {"type": "image_url", "image_url": {"url": _image_data_url(request.pre_image)}},
                {"type": "image_url", "image_url": {"url": _image_data_url(request.post_image)}},
                {"type": "text", "text": json.dumps(prompt, sort_keys=True)},
            ],
            schema=_evidence_schema(),
            purpose="vista_visual_evidence",
        )
        items = []
        pre_values = {item.key: item.value for item in request.pre_ledger}
        for index, observation in enumerate(result.get("observations", [])):
            key = PredicateKey.parse(str(observation["predicate"]))
            items.append(
                PredicateEvidence(
                    key=key,
                    before=pre_values.get(key, TruthValue.UNKNOWN),
                    after=TruthValue(str(observation["value"])),
                    confidence=float(observation["confidence"]),
                    source=EvidenceSource.VISUAL_PAIR,
                    evidence_id=f"{request.episode_id}:s{request.step_id}:visual:{index}:{key.render()}",
                    timestamp=request.step_id,
                    view_id=f"{request.episode_id}:s{request.step_id}:prepost",
                    coverage=float(observation["coverage"]),
                    rationale=str(observation["evidence"]),
                )
            )
        return items


class JsonGoalGrounder:
    """Ground instruction goals without simulator predicates or expected effects."""

    def __init__(self, model: JsonModel) -> None:
        self.model = model

    def ground(
        self,
        instruction: str,
        initial_image: str,
        action_catalog: Sequence[tuple[str, Sequence[str]]],
    ) -> tuple[PredicateKey, ...]:
        public_actions = [
            {"action_type": name, "arguments": list(arguments)}
            for name, arguments in action_catalog
        ]
        result = self.model.complete_json(
            system=(
                "Ground the instruction into task-completion predicates using only the initial image "
                "and public action catalog. Preserve numbered instance IDs. Do not invent hidden state."
            ),
            content=[
                {"type": "image_url", "image_url": {"url": _image_data_url(initial_image)}},
                {
                    "type": "text",
                    "text": json.dumps(
                        {"instruction": instruction, "public_actions": public_actions},
                        sort_keys=True,
                    ),
                },
            ],
            schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["goal_predicates"],
                "properties": {
                    "goal_predicates": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
            },
            purpose="vista_goal_grounding",
        )
        return tuple(
            dict.fromkeys(
                PredicateKey.parse(str(value))
                for value in result.get("goal_predicates", [])
            )
        )


class JsonAttributionTeacher:
    def __init__(self, model: JsonModel) -> None:
        self.model = model

    def assign(
        self,
        mismatches: Sequence[Mismatch],
        context: AttributionContext,
    ) -> AttributionResult:
        payload = {
            "mismatches": [_mismatch_payload(item) for item in mismatches],
            "context": {
                "executor_followed_skill": context.executor_followed_skill,
                "stochastic_suspected": context.stochastic_suspected,
                "identity_conflict": context.identity_conflict,
                "instruction": context.instruction,
                "executed_action_type": context.action_type,
                "active_skill_obligations": context.skill_obligations,
            },
            "allowed_targets": ["belief_refresh", "skill_update", "abstain"],
            "allowed_fields": [field.value for field in SkillField],
        }
        result = self.model.complete_json(
            system=(
                "Assign a persistent-memory update target from cited transition evidence. "
                "Prefer abstention when evidence cannot distinguish causes. A field is legal only for skill_update."
            ),
            content=[{"type": "text", "text": json.dumps(payload, sort_keys=True)}],
            schema=_attribution_schema(),
            purpose="vista_attribution",
        )
        target = UpdateTarget(str(result["target"]))
        field_value = result.get("field")
        reason_value = result.get("subreason")
        return AttributionResult(
            target=target,
            field=SkillField(field_value) if target is UpdateTarget.SKILL_UPDATE else None,
            subreason=AbstainReason(reason_value) if target is UpdateTarget.ABSTAIN else None,
            confidence=float(result["confidence"]),
            mismatch_ids=tuple(str(item) for item in result["mismatch_ids"]),
            evidence_ids=tuple(str(item) for item in result["evidence_ids"]),
            rationale=str(result["rationale"]),
        )


class JsonTrajectoryTeacher:
    """Model-backed trajectory reflection for the controlled baseline frontends.

    This is the trajectory-level analogue of :class:`JsonAttributionTeacher`.
    The full VISTA method attributes predicate-level mismatches per primitive
    action; the EmbodiSkill-style baselines instead reflect once over a whole
    completed episode and decide whether a persistent skill revision is
    warranted. Sharing the same OpenAI-compatible backend (and therefore the
    same ``--method-model``) keeps teacher model, token budget, and call count
    matched across the controlled comparison, as required by ``methods.json``.
    """

    def __init__(self, model: JsonModel) -> None:
        self.model = model

    def reflect(self, episode: EpisodeSummary) -> TrajectoryReflection:
        payload = {
            "instruction": episode.instruction,
            "episode_succeeded": bool(episode.success),
            "trajectory": list(episode.trajectory),
            "current_skill": episode.current_skill,
            "failure_reason": episode.failure_reason,
            "allowed_routes": [route.value for route in EmbodiSkillRoute],
            "allowed_fields": [field.value for field in SkillField],
            "rule": (
                "Route as FAIL_EXECUTION when the failure is an execution lapse "
                "(unsatisfied preconditions, wrong object, not a skill gap). "
                "Otherwise pick a skill field and propose one concrete, boundable "
                "statement that revises it. Preserve numbered instance identities; "
                "do not invent hidden state."
            ),
        }
        result = self.model.complete_json(
            system=(
                "Reflect on one completed embodied-agent trajectory and decide "
                "whether a persistent procedural-skill revision is warranted."
            ),
            content=[{"type": "text", "text": json.dumps(payload, sort_keys=True)}],
            schema=_trajectory_reflection_schema(),
            purpose="trajectory_reflection",
        )
        route = EmbodiSkillRoute(str(result["route"]))
        field_value = result.get("target_field")
        evidence_ids = tuple(
            dict.fromkeys(str(item) for item in result.get("evidence_ids", []))
        )
        return TrajectoryReflection(
            route=route,
            content=str(result["content"]),
            target_field=SkillField(field_value) if field_value is not None else None,
            confidence=float(result["confidence"]),
            evidence_ids=evidence_ids,
        )


class JsonBoundedPatchGenerator(PatchGenerator):
    def __init__(self, model: JsonModel) -> None:
        self.model = model

    def propose(self, skill: SkillSpec, cluster: EvidenceCluster) -> SkillPatch:
        field = cluster.key.field
        evidence_ids = cluster.evidence_ids
        payload = {
            "skill_id": skill.skill_id,
            "parent_version": skill.version,
            "attributed_field": field.value,
            "current_statements": skill.statements(field),
            "cluster": [
                {
                    "kind": item.mismatch.kind.value,
                    "predicate": item.mismatch.key.render(),
                    "expected": None if item.mismatch.expected is None else item.mismatch.expected.after.value,
                    "evidence": None if item.mismatch.evidence is None else item.mismatch.evidence.after.value,
                    "evidence_ids": item.attribution.evidence_ids,
                }
                for item in cluster.items
            ],
            "allowed_operations": [item.value for item in PatchOperation],
            "constraints": [
                "one atomic operation in the attributed field only",
                "old must exactly match an existing statement when required",
                "do not introduce object instance identifiers or unsupported facts",
            ],
        }
        result = self.model.complete_json(
            system="Generate one evidence-bound, exact-target field patch. Never rewrite the full skill.",
            content=[{"type": "text", "text": json.dumps(payload, sort_keys=True)}],
            schema=_patch_schema(),
            purpose="vista_bounded_patch",
        )
        operation = PatchOperation(str(result["operation"]))
        old = str(result["old"])
        new = str(result["new"])
        termination_policy = (
            TerminationPolicy(str(result["termination_policy"]))
            if result.get("termination_policy") is not None
            else None
        )
        prediction_rules = tuple(
            SkillPredictionRule(
                rule_id=str(item["rule_id"]),
                field=field,
                action_type=str(item["action_type"]),
                predicate=str(item["predicate"]),
                before=None if item["before"] is None else TruthValue(str(item["before"])),
                after=TruthValue(str(item["after"])),
            )
            for item in result.get("prediction_rules", [])
        )
        return SkillPatch(
            patch_id=make_patch_id(
                skill,
                field,
                operation,
                old,
                new,
                evidence_ids,
                termination_policy,
                prediction_rules,
            ),
            skill_id=skill.skill_id,
            parent_version=skill.version,
            field=field,
            operation=operation,
            old=old,
            new=new,
            evidence_ids=evidence_ids,
            scope=str(result["scope"]),
            rationale=str(result["rationale"]),
            termination_policy=termination_policy,
            prediction_rules=prediction_rules,
        )


def _image_data_url(path: str) -> str:
    image_path = Path(path)
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _evidence_queries(request: EvidenceRequest) -> tuple[PredicateKey, ...]:
    """Build queries only from evidence-side inputs, never from predictions."""
    queries = [*request.goal_predicates, *(item.key for item in request.pre_ledger)]
    action = request.action
    if action.action_type == "place" and action.arguments:
        receptacle = action.arguments[0]
        for item in request.pre_ledger:
            if (
                item.key.name == "holding"
                and item.key.arguments
                and item.value is TruthValue.TRUE
            ):
                queries.append(PredicateKey("at", (item.key.arguments[0], receptacle)))
    elif action.action_type in {"open", "close"} and action.arguments:
        queries.append(PredicateKey("open", (action.arguments[0],)))
    elif action.action_type == "nav" and action.arguments:
        queries.append(PredicateKey("near", (action.arguments[0],)))
    return tuple(dict.fromkeys(queries))


def _mismatch_payload(item: Mismatch) -> dict[str, Any]:
    return {
        "mismatch_id": item.mismatch_id,
        "predicate": item.key.render(),
        "kind": item.kind.value,
        "expected": None
        if item.expected is None
        else {
            "before": item.expected.before.value,
            "after": item.expected.after.value,
            "source": item.expected.source.value,
            "source_id": item.expected.source_id,
            "skill_field": None if item.expected.skill_field is None else item.expected.skill_field.value,
        },
        "evidence": None
        if item.evidence is None
        else {
            "before": item.evidence.before.value,
            "after": item.evidence.after.value,
            "confidence": item.evidence.confidence,
            "coverage": item.evidence.coverage,
            "evidence_id": item.evidence.evidence_id,
        },
    }


def _evidence_schema() -> dict[str, Any]:
    observation = {
        "type": "object",
        "additionalProperties": False,
        "required": ["predicate", "value", "confidence", "coverage", "evidence"],
        "properties": {
            "predicate": {"type": "string"},
            "value": {"type": "string", "enum": ["true", "false", "unknown"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "coverage": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": "string"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["observations"],
        "properties": {"observations": {"type": "array", "items": observation}},
    }


def _attribution_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["target", "field", "subreason", "confidence", "mismatch_ids", "evidence_ids", "rationale"],
        "properties": {
            "target": {"type": "string", "enum": ["belief_refresh", "skill_update", "abstain"]},
            "field": {"type": ["string", "null"], "enum": [None, *[item.value for item in SkillField]]},
            "subreason": {"type": ["string", "null"], "enum": [None, *[item.value for item in AbstainReason]]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "mismatch_ids": {"type": "array", "items": {"type": "string"}},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
    }


def _trajectory_reflection_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["route", "content", "target_field", "confidence", "evidence_ids"],
        "properties": {
            "route": {
                "type": "string",
                "enum": [item.value for item in EmbodiSkillRoute],
            },
            "content": {"type": "string"},
            "target_field": {
                "type": ["string", "null"],
                "enum": [None, *[item.value for item in SkillField]],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
        },
    }


def _patch_schema() -> dict[str, Any]:
    rule_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["rule_id", "action_type", "predicate", "before", "after"],
        "properties": {
            "rule_id": {"type": "string"},
            "action_type": {"type": "string"},
            "predicate": {"type": "string"},
            "before": {"type": ["string", "null"], "enum": [None, "true", "false", "unknown"]},
            "after": {"type": "string", "enum": ["true", "false", "unknown"]},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["operation", "old", "new", "scope", "rationale", "termination_policy", "prediction_rules"],
        "properties": {
            "operation": {"type": "string", "enum": [item.value for item in PatchOperation]},
            "old": {"type": "string"},
            "new": {"type": "string"},
            "scope": {"type": "string"},
            "rationale": {"type": "string"},
            "termination_policy": {
                "type": ["string", "null"],
                "enum": [None, *[item.value for item in TerminationPolicy]],
            },
            "prediction_rules": {"type": "array", "items": rule_schema},
        },
    }
