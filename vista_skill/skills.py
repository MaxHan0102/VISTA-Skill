from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from vista_skill.schemas import (
    SkillField,
    SkillPredictionRule,
    SkillSpec,
    TemporalSkillRule,
    TerminationPolicy,
    TruthValue,
    dataclass_to_dict,
)


SKILL_ARTIFACT_SCHEMA_VERSION = "2"


@dataclass(frozen=True)
class SkillArtifact:
    """Digest-checked Skill plus the controlled protocol that produced it."""

    schema_version: str
    skill: SkillSpec
    protocol: Mapping[str, Any]
    artifact_sha256: str

    @property
    def skill_sha256(self) -> str:
        return skill_digest(self.skill)


def initialize_shared_skill() -> SkillSpec:
    """Benchmark-aware, task-agnostic S0 shared by controlled methods."""
    return SkillSpec(
        skill_id="shared_embodied_execution",
        version=0,
        activation=(
            "Use for multi-step embodied tasks that require state verification.",
        ),
        procedure=(
            "Decompose the instruction into goal predicates and track pending and completed targets.",
            "Check action preconditions and update the checklist only from new evidence.",
            "Re-observe or replan when the evidence needed for the next step is insufficient.",
        ),
        effect=(
            "Each step should make progress toward one or more unsatisfied goal predicates.",
        ),
        termination=(
            "Stop only when every required goal predicate is supported by current evidence.",
        ),
        constraint=(
            "Unknown is not false.",
            "Success of one subgoal is not evidence that the full task is complete.",
        ),
        termination_policy=TerminationPolicy.ALL_GOALS_EVIDENCE,
        prediction_rules=(
            SkillPredictionRule(
                "procedure_navigation_reaches_target",
                SkillField.PROCEDURE,
                "nav",
                "near({arg0})",
                TruthValue.TRUE,
            ),
            SkillPredictionRule(
                "effect_pick_holds_target_category",
                SkillField.EFFECT,
                "pick",
                "holding({arg0})",
                TruthValue.TRUE,
            ),
            SkillPredictionRule(
                "constraint_pick_occupies_gripper",
                SkillField.CONSTRAINT,
                "pick",
                "not_holding",
                TruthValue.FALSE,
            ),
            SkillPredictionRule(
                "effect_place_relates_held_object",
                SkillField.EFFECT,
                "place",
                "at({held},{arg0})",
                TruthValue.TRUE,
            ),
            SkillPredictionRule(
                "constraint_place_frees_gripper",
                SkillField.CONSTRAINT,
                "place",
                "not_holding",
                TruthValue.TRUE,
            ),
            SkillPredictionRule(
                "effect_open_changes_articulation",
                SkillField.EFFECT,
                "open",
                "open({arg0})",
                TruthValue.TRUE,
            ),
            SkillPredictionRule(
                "effect_close_changes_articulation",
                SkillField.EFFECT,
                "close",
                "open({arg0})",
                TruthValue.FALSE,
            ),
        ),
        metadata={"initialization": "benchmark-aware-task-agnostic"},
    )


def initialize_nav_skill() -> SkillSpec:
    """Task-agnostic S0 for EB-Navigation (egocentric point/goal navigation).

    Navigation action effects are geometric, so compiled ``prediction_rules``
    are intentionally empty: :class:`NavActionSchema` owns the structural
    primitive predictions (position/heading/camera-tilt changed) and
    ``near(target)`` is observed only through evidence (env distance). The Skill
    contributes execution guidance via its statements and a goal-evidence
    termination policy, mirroring the EB-Habitat S0's role without over-claiming
    geometric effects it cannot deterministically predict.
    """
    return SkillSpec(
        skill_id="shared_navigation",
        version=0,
        activation=(
            "Use for egocentric navigation tasks that require approaching a target.",
        ),
        procedure=(
            "Identify the target object from the instruction and track distance to it.",
            "Move toward the target and confirm progress from environment feedback.",
            "Re-observe or replan when a move is blocked or the target leaves view.",
        ),
        effect=(
            "Each movement should reduce the distance to the target object.",
        ),
        termination=(
            "Stop only when near(target) is supported by current evidence.",
        ),
        constraint=(
            "Unknown is not false.",
            "A blocked move is not evidence that the target has been reached.",
        ),
        termination_policy=TerminationPolicy.ALL_GOALS_EVIDENCE,
        prediction_rules=(),
        metadata={"initialization": "nav-task-agnostic", "env": "eb_navigation"},
    )


def target_habitat_skill_v1() -> SkillSpec:
    """Post-hoc oracle target for the trajectory-synthesis diagnostic.

    This is deliberately *not* a controlled-protocol initialization.  Its text
    was synthesized after inspecting the historical official-test/base
    No-Skill and Static-Skill trajectories, so it must never be presented as a
    held-out or automatically evolved Skill.  It is kept as a normal
    five-field :class:`SkillSpec` so the frozen executor surface is identical to
    any Skill that the VISTA evolution loop could eventually produce.
    """
    return SkillSpec(
        skill_id="target_habitat_rearrangement_oracle_v1",
        version=1,
        parent_version=0,
        frozen=True,
        activation=(
            "Use for embodied search, delivery, and removal tasks with named movable objects and receptacles.",
            "Bind exact object categories, quantities, source clues, and destination before acting; keep those bindings unchanged.",
        ),
        procedure=(
            "Translate the instruction into a checklist of exact target objects, remaining quantities, source clues, and the exact destination. For remove or detach from X, the goal is not-at-X: after picking, place the object on a different valid receptacle.",
            "When not holding, search systematically across untried plausible receptacles. Navigate to one and inspect the new image. Navigation success means only that the robot reached that receptacle; it never proves that the requested object is present or near.",
            "Pick only when the exact requested object category is visibly present at the current location or feedback explicitly supports nearness. Do not replace the requested noun with a similar object or synonym. If a Pick fails as not near, mark that location tried, navigate to a different untried location, and do not repeat Pick until new location or visibility evidence exists.",
            "For a possible closed container, navigate to it, open it at most once when closed, then inspect before picking. Do not repeat Open after success or without new state evidence.",
            "After feedback confirms Pick or holding the target, immediately enter delivery mode: navigate to the exact destination and Place there. Never Pick while holding or place back on the source or at a convenient receptacle.",
            "After a successful Place, verify the exact object-destination goal. For multiple objects, mark only that object complete and repeat the search-pick-deliver cycle for the remaining checklist. After an uncertain search, Pick, Open, or Place, output only the next evidence-gathering action; chain actions only when every intermediate precondition is already supported.",
        ),
        effect=(
            "A successful Navigation establishes nearness to its receptacle only, not nearness to any movable object guessed to be there.",
            "A successful Pick of the requested object establishes holding that object; a failed action changes no task predicate. A successful Place at the bound destination establishes the object-destination goal and frees the gripper.",
            "Progress counts only for the requested category, quantity, and destination; interacting with another object is not progress.",
        ),
        termination=(
            "For delivery, stop only after every requested object instance is evidence-verified at the exact destination and the gripper is free.",
            "For remove or detach tasks, stop only after every requested object is no longer at the named source and has been placed on a different valid receptacle. Holding, reaching a receptacle, or completing one of several subgoals is not completion.",
        ),
        constraint=(
            "Never repeat the same failed interaction at the same location without an intervening state-changing or evidence-gathering action; switch to a different untried location after not-near feedback.",
            "Preserve object identity and destination fidelity. Never Pick while already holding, Place while empty, or infer that an unseen object is absent.",
            "Use only listed action IDs and exact action semantics. Treat explicit environment feedback about success, holding, nearness, open state, and invalid preconditions as stronger evidence than a visual guess.",
        ),
        termination_policy=TerminationPolicy.ALL_GOALS_EVIDENCE,
        prediction_rules=initialize_shared_skill().prediction_rules,
        metadata={
            "initialization": "posthoc-human-trajectory-synthesis",
            "oracle_diagnostic": True,
            "source_environment": "eb-hab",
            "source_split": "official_test/base",
            "source_arms": ["no_skill", "static_shared_skill"],
        },
    )


def target_navigation_skill_v1() -> SkillSpec:
    """Post-hoc EB-Navigation oracle target built from historical base traces.

    As with :func:`target_habitat_skill_v1`, this artifact is contaminated by
    official-test/base observations and is only an upper-bound/reference target
    for future Skill evolution work.
    """
    return SkillSpec(
        skill_id="target_feedback_navigation_oracle_v1",
        version=1,
        parent_version=0,
        frozen=True,
        activation=(
            "Use for egocentric navigation to one named target object when each step reports action success and target distance.",
            "Bind the exact target category from the instruction and optimize only its reported distance.",
        ),
        procedure=(
            "Maintain the last and best target distance from feedback. On every planner call output exactly ONE primitive action, despite any generic request for a 5-6 action plan, then re-observe the image and new distance before choosing again.",
            "Probe a plausible forward, left, or right translation. Continue that direction only while the reported distance strictly decreases; a successful motion is not progress when distance stays equal or increases.",
            "If distance increases, do not continue the stale plan: choose the inverse lateral direction or backtrack once, then re-observe. When already close to the best point, use single-step probes and never execute a long run that can overshoot it.",
            "If an action fails or distance is unchanged, treat the path as blocked. Do not repeat the same translation; choose a different lateral move, or one 90-degree rotation followed by a separately planned translation. Use rotation or camera tilt only for recovery when the target is lost or translations are blocked.",
            "Prefer the direction with demonstrated numeric improvement over a visual guess. If two consecutive choices fail to beat the best distance, return toward the last improving direction and test a different single-step alternative.",
        ),
        effect=(
            "A movement makes goal progress only when the numeric target distance decreases. Rotation and camera tilt change viewpoint but do not by themselves reduce distance.",
            "A failed or zero-displacement action leaves position unchanged and invalidates repeating that same local move.",
        ),
        termination=(
            "The task is complete only when feedback supports near(target), operationally target distance at or below 1.0 meter; otherwise continue single-step closed-loop control.",
        ),
        constraint=(
            "Numeric target-distance feedback overrides claims that the object merely looks visible, close, or reachable.",
            "Never emit more than one action in executable_plan. Never repeat a blocked action, continue a direction after distance worsens, or rotate while a translation is reliably reducing distance.",
            "Unknown is not false, action success is not goal success, and the target identity must remain the exact instructed category.",
        ),
        termination_policy=TerminationPolicy.ALL_GOALS_EVIDENCE,
        prediction_rules=(),
        metadata={
            "initialization": "posthoc-human-trajectory-synthesis",
            "oracle_diagnostic": True,
            "source_environment": "eb-nav",
            "source_split": "official_test/base",
            "source_arms": ["no_skill", "static_shared_skill"],
        },
    )


def minimal_shared_skill() -> SkillSpec:
    """Minimal S0 variant: the weakest reasonable starting point for evolution.

    A drop-in alternative to :func:`initialize_shared_skill` for the §4.2.3
    initialization-sensitivity controlled comparison. It carries only what is
    structurally required to be a valid, runnable Skill against the fixed
    nav/pick/place/open/close action schema: a single generic activation, a
    one-line fallback procedure, a valid termination statement, and empty
    effect/constraint bodies. Compiled ``prediction_rules`` are intentionally
    empty so the engine falls back to primitive action-schema transitions only,
    giving the attribution/evolution loop the weakest prior to grow from.
    """
    return SkillSpec(
        skill_id="shared_embodied_execution",
        version=0,
        activation=(
            "Use for embodied tasks that require step-wise execution.",
        ),
        procedure=(
            "Execute the instructed primitive action and observe its result.",
        ),
        effect=(),
        termination=(
            "Stop when the instructed goal is reached.",
        ),
        constraint=(),
        termination_policy=TerminationPolicy.ALL_GOALS_EVIDENCE,
        prediction_rules=(),
        metadata={"initialization": "minimal"},
    )


def empty_shared_skill() -> SkillSpec:
    """Empty S0 variant: the absolute lower bound for init-sensitivity study.

    Every statement body is empty and there are no compiled ``prediction_rules``,
    yet the object remains a structurally valid :class:`SkillSpec` (``skill_id``,
    ``version`` 0, ``termination_policy`` present) so that
    :meth:`FixedActionSchema.compile` and :class:`VistaSkillEngine` accept it
    without raising. Represents the "no prior task knowledge" extreme of the
    §4.2.3 controlled comparison; keep teacher and evolution budget matched
    against :func:`initialize_shared_skill` and :func:`minimal_shared_skill`.
    """
    return SkillSpec(
        skill_id="shared_embodied_execution",
        version=0,
        activation=(),
        procedure=(),
        effect=(),
        termination=(),
        constraint=(),
        termination_policy=TerminationPolicy.ALL_GOALS_EVIDENCE,
        prediction_rules=(),
        metadata={"initialization": "empty"},
    )


def interface_only_shared_skill() -> SkillSpec:
    """Primary Phase-5 S0: structure and action interface, no task rules.

    The benchmark supplies the executor's legal action interface. The Skill
    supplies only its five typed fields and persistent identity; every field is
    initially empty and no transition or termination rule is pre-installed.
    This makes acquired content attributable to the agent's own interactions.
    """
    return SkillSpec(
        skill_id="shared_embodied_execution",
        version=0,
        activation=(),
        procedure=(),
        effect=(),
        termination=(),
        constraint=(),
        termination_policy=TerminationPolicy.ALL_GOALS_EVIDENCE,
        prediction_rules=(),
        metadata={
            "initialization": "interface-only",
            "prior_task_rules": 0,
            "prior_transition_rules": 0,
        },
    )


def render_skill(skill: SkillSpec, *, max_statements_per_field: int | None = None) -> str:
    parts = [f"Skill: {skill.skill_id} (v{skill.version})"]
    for skill_field in SkillField:
        statements = skill.statements(skill_field)
        if max_statements_per_field is not None:
            statements = statements[:max_statements_per_field]
        parts.append(f"{skill_field.value.title()}:")
        parts.extend(f"- {statement}" for statement in statements)
    return "\n".join(parts)


def skill_digest(skill: SkillSpec) -> str:
    payload = json.dumps(_skill_payload(skill), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def skill_artifact_digest(
    schema_version: str,
    skill: SkillSpec,
    protocol: Mapping[str, Any],
) -> str:
    payload = json.dumps(
        {
            "schema_version": str(schema_version),
            "skill": _skill_payload(skill),
            "protocol": dict(protocol),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_protocol(protocol: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize protocol metadata to the exact JSON types stored on disk."""
    payload = json.dumps(dict(protocol or {}), sort_keys=True, separators=(",", ":"))
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("protocol metadata must serialize to a JSON object")
    return value


def _skill_payload(skill: SkillSpec) -> dict[str, Any]:
    """Serialize optional compiled extensions without invalidating legacy v2 Skills."""
    payload = dataclass_to_dict(skill)
    if not skill.temporal_rules:
        payload.pop("temporal_rules", None)
    return payload


def skill_from_dict(raw: Mapping[str, Any]) -> SkillSpec:
    rules = tuple(
        SkillPredictionRule(
            rule_id=str(item["rule_id"]),
            field=SkillField(str(item["field"])),
            action_type=str(item["action_type"]),
            predicate=str(item["predicate"]),
            after=TruthValue(str(item["after"])),
            before=None
            if item.get("before") is None
            else TruthValue(str(item["before"])),
        )
        for item in raw.get("prediction_rules", ())
    )
    temporal_rules = tuple(
        TemporalSkillRule(
            rule_id=str(item["rule_id"]),
            field=SkillField(str(item["field"])),
            trigger_action_type=str(item["trigger_action_type"]),
            trigger_success=bool(item["trigger_success"]),
            trigger_predicate=str(item["trigger_predicate"]),
            trigger_value=TruthValue(str(item["trigger_value"])),
            blocked_action_type=str(item["blocked_action_type"]),
            recovery_action_types=tuple(
                str(value) for value in item["recovery_action_types"]
            ),
            argument_index=int(item.get("argument_index", 0)),
        )
        for item in raw.get("temporal_rules", ())
    )
    return SkillSpec(
        skill_id=str(raw["skill_id"]),
        version=int(raw["version"]),
        activation=tuple(str(item) for item in raw["activation"]),
        procedure=tuple(str(item) for item in raw["procedure"]),
        effect=tuple(str(item) for item in raw["effect"]),
        termination=tuple(str(item) for item in raw["termination"]),
        constraint=tuple(str(item) for item in raw["constraint"]),
        termination_policy=TerminationPolicy(str(raw["termination_policy"])),
        prediction_rules=rules,
        temporal_rules=temporal_rules,
        parent_version=None
        if raw.get("parent_version") is None
        else int(raw["parent_version"]),
        frozen=bool(raw.get("frozen", False)),
        metadata=dict(raw.get("metadata", {})),
    )


def save_skill_artifact(
    path: str | Path,
    skill: SkillSpec,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> None:
    protocol_payload = canonical_protocol(protocol)
    payload = {
        "schema_version": SKILL_ARTIFACT_SCHEMA_VERSION,
        "skill": _skill_payload(skill),
        "skill_sha256": skill_digest(skill),
        "protocol": protocol_payload,
        "artifact_sha256": skill_artifact_digest(
            SKILL_ARTIFACT_SCHEMA_VERSION,
            skill,
            protocol_payload,
        ),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_content_addressed_skill(
    directory: str | Path,
    skill: SkillSpec,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> Path:
    output = Path(directory) / f"{skill_digest(skill)}.json"
    if not output.exists():
        save_skill_artifact(output, skill, protocol=protocol)
    else:
        record = load_skill_artifact_record(output)
        if record.skill != skill or dict(record.protocol) != canonical_protocol(protocol):
            raise ValueError("content-addressed Skill artifact collision")
    return output


def load_skill_artifact_record(
    path: str | Path,
    *,
    require_frozen: bool = False,
) -> SkillArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema_version = str(payload.get("schema_version", ""))
    if schema_version != SKILL_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported skill artifact schema version: {schema_version or 'missing'}"
        )
    skill = skill_from_dict(payload["skill"])
    if payload.get("skill_sha256") != skill_digest(skill):
        raise ValueError("skill artifact digest mismatch")
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("skill artifact protocol metadata must be an object")
    expected_digest = skill_artifact_digest(schema_version, skill, protocol)
    if payload.get("artifact_sha256") != expected_digest:
        raise ValueError("skill artifact envelope digest mismatch")
    if require_frozen and not skill.frozen:
        raise ValueError("evaluation requires a frozen skill artifact")
    return SkillArtifact(
        schema_version=schema_version,
        skill=skill,
        protocol=dict(protocol),
        artifact_sha256=expected_digest,
    )


def load_skill_artifact(path: str | Path, *, require_frozen: bool = False) -> SkillSpec:
    """Load only the Skill for compatibility with the original artifact API."""

    return load_skill_artifact_record(path, require_frozen=require_frozen).skill


def with_field(
    skill: SkillSpec,
    skill_field: SkillField,
    statements: Iterable[str],
    *,
    frozen: bool = False,
) -> SkillSpec:
    values = tuple(statements)
    return replace(
        skill,
        **{
            skill_field.value: values,
            "version": skill.version + 1,
            "parent_version": skill.version,
            "frozen": frozen,
        },
    )
