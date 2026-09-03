from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass, field, replace
from typing import Mapping, Protocol, Sequence

from vista_skill.clustering import EvidenceCluster
from vista_skill.action_schema import ActionSchema, FixedActionSchema
from vista_skill.belief import BeliefLedger
from vista_skill.mismatch import compare_transitions
from vista_skill.schemas import (
    PatchOperation,
    SkillField,
    SkillPatch,
    SkillPredictionRule,
    SkillSpec,
    TemporalSkillRule,
    TerminationPolicy,
    TruthValue,
    SkillUpdateKind,
)
from vista_skill.skills import with_field


_ALLOWED_ACTION_TYPES = {"*", "nav", "pick", "place", "open", "close"}
_ALLOWED_PLACEHOLDERS = {"arg0", "arg1", "held"}
_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")
_ACTION_PLACEHOLDERS = {
    "*": {"held"},
    "nav": {"arg0", "held"},
    "pick": {"arg0", "held"},
    "place": {"arg0", "held"},
    "open": {"arg0", "held"},
    "close": {"arg0", "held"},
}


class PatchGenerator(Protocol):
    def propose(self, skill: SkillSpec, cluster: EvidenceCluster) -> SkillPatch: ...


@dataclass(frozen=True)
class PatchPolicy:
    max_operations: int = 1
    max_statements_per_field: int = 12
    max_active_skill_words: int = 512
    reject_instance_specific_rules: bool = True


class PatchValidationError(ValueError):
    pass


class BoundedPatchApplier:
    """Fail-closed one-field patch application."""

    def __init__(self, policy: PatchPolicy | None = None) -> None:
        self.policy = policy or PatchPolicy()

    def validate(self, skill: SkillSpec, patch: SkillPatch) -> tuple[str, ...]:
        errors: list[str] = []
        if skill.frozen:
            errors.append("skill is frozen")
        if patch.skill_id != skill.skill_id or patch.parent_version != skill.version:
            errors.append("patch parent does not match the active skill")
        if not patch.evidence_ids:
            errors.append("patch has no evidence binding")
        statements = skill.statements(patch.field)
        if patch.operation in {
            PatchOperation.REPLACE_EXACT,
            PatchOperation.DELETE_EXACT,
            PatchOperation.INSERT_AFTER_EXACT,
        } and patch.old not in statements:
            errors.append("exact target is absent from the attributed field")
        if patch.operation in {PatchOperation.APPEND, PatchOperation.REPLACE_EXACT, PatchOperation.INSERT_AFTER_EXACT} and not patch.new.strip():
            errors.append("patch adds an empty statement")
        if patch.operation is PatchOperation.DELETE_EXACT and patch.new:
            errors.append("delete patch must not contain replacement text")
        if self.policy.reject_instance_specific_rules and _contains_instance_identifier(patch.new):
            errors.append("candidate contains an instance-specific identifier")
        if patch.termination_policy is not None and patch.field is not SkillField.TERMINATION:
            errors.append("termination policy can only change with the termination field")
        if patch.prediction_rules is not None:
            if any(rule.field is not patch.field for rule in patch.prediction_rules):
                errors.append("compiled prediction crosses the attributed field")
            errors.extend(_validate_prediction_rules(patch.prediction_rules))
        if patch.temporal_rules is not None:
            if any(rule.field is not patch.field for rule in patch.temporal_rules):
                errors.append("compiled temporal rule crosses the attributed field")
            errors.extend(_validate_temporal_rules(patch.temporal_rules))
        return tuple(errors)

    def apply(self, skill: SkillSpec, patch: SkillPatch) -> SkillSpec:
        errors = self.validate(skill, patch)
        if errors:
            raise PatchValidationError("; ".join(errors))
        statements = list(skill.statements(patch.field))
        if patch.operation is PatchOperation.APPEND:
            statements.append(patch.new.strip())
        elif patch.operation is PatchOperation.INSERT_AFTER_EXACT:
            index = statements.index(patch.old)
            statements.insert(index + 1, patch.new.strip())
        elif patch.operation is PatchOperation.REPLACE_EXACT:
            statements[statements.index(patch.old)] = patch.new.strip()
        elif patch.operation is PatchOperation.DELETE_EXACT:
            statements.remove(patch.old)
        else:
            raise PatchValidationError(f"unsupported operation: {patch.operation}")

        if len(statements) > self.policy.max_statements_per_field:
            raise PatchValidationError("field statement budget exceeded")
        candidate = with_field(skill, patch.field, statements)
        if patch.termination_policy is not None:
            candidate = replace(candidate, termination_policy=patch.termination_policy)
        if patch.prediction_rules is not None:
            retained = tuple(
                rule for rule in candidate.prediction_rules if rule.field is not patch.field
            )
            candidate = replace(candidate, prediction_rules=(*retained, *patch.prediction_rules))
        if patch.temporal_rules is not None:
            retained_temporal = tuple(
                rule for rule in candidate.temporal_rules if rule.field is not patch.field
            )
            candidate = replace(
                candidate,
                temporal_rules=(*retained_temporal, *patch.temporal_rules),
            )
        rule_errors = _validate_prediction_rules(candidate.prediction_rules)
        rule_errors = (*rule_errors, *_validate_temporal_rules(candidate.temporal_rules))
        if rule_errors:
            raise PatchValidationError("; ".join(rule_errors))
        word_count = sum(
            len(statement.split())
            for skill_field in SkillField
            for statement in candidate.statements(skill_field)
        )
        if word_count > self.policy.max_active_skill_words:
            raise PatchValidationError("active skill budget exceeded")
        return candidate


def _contains_instance_identifier(value: str) -> bool:
    return bool(re.search(r"\b[a-zA-Z]+(?:_[a-zA-Z]+)*_\d+\b", value))


def _validate_prediction_rules(
    rules: Sequence[SkillPredictionRule],
) -> tuple[str, ...]:
    errors: list[str] = []
    rule_ids: set[str] = set()
    signatures: dict[tuple[str, str, TruthValue | None], TruthValue] = {}
    for rule in rules:
        if not rule.rule_id.strip() or rule.rule_id in rule_ids:
            errors.append("compiled prediction rule IDs must be non-empty and unique")
        rule_ids.add(rule.rule_id)
        if rule.action_type not in _ALLOWED_ACTION_TYPES:
            errors.append(f"unsupported compiled action type: {rule.action_type}")
        placeholders = set(_PLACEHOLDER.findall(rule.predicate))
        if placeholders - _ALLOWED_PLACEHOLDERS:
            errors.append("compiled predicate contains an unsupported placeholder")
        elif placeholders - _ACTION_PLACEHOLDERS.get(rule.action_type, set()):
            errors.append("compiled predicate placeholder is unavailable for its action type")
        if "{" in _PLACEHOLDER.sub("", rule.predicate) or "}" in _PLACEHOLDER.sub("", rule.predicate):
            errors.append("compiled predicate contains malformed placeholders")
        parseable = _PLACEHOLDER.sub("placeholder", rule.predicate)
        try:
            from vista_skill.schemas import PredicateKey

            PredicateKey.parse(parseable)
        except (TypeError, ValueError):
            errors.append(f"compiled predicate is invalid: {rule.predicate}")
        if _contains_instance_identifier(rule.action_type) or _contains_instance_identifier(
            _PLACEHOLDER.sub("", rule.predicate)
        ):
            errors.append("compiled prediction contains an instance-specific identifier")
        signature = (rule.action_type, rule.predicate, rule.before)
        previous = signatures.get(signature)
        if previous is not None:
            if previous is rule.after:
                errors.append("compiled prediction contains a duplicate rule")
            else:
                errors.append("compiled prediction contains conflicting rules")
        signatures[signature] = rule.after
    return tuple(dict.fromkeys(errors))


def _validate_temporal_rules(
    rules: Sequence[TemporalSkillRule],
) -> tuple[str, ...]:
    errors: list[str] = []
    rule_ids: set[str] = set()
    signatures: set[tuple[str, bool, str, TruthValue, str, int]] = set()
    for rule in rules:
        if not rule.rule_id.strip() or rule.rule_id in rule_ids:
            errors.append("compiled temporal rule IDs must be non-empty and unique")
        rule_ids.add(rule.rule_id)
        if rule.field not in {SkillField.PROCEDURE, SkillField.CONSTRAINT}:
            errors.append("compiled temporal rules require a procedure or constraint field")
        if rule.trigger_action_type not in _ALLOWED_ACTION_TYPES - {"*"}:
            errors.append(f"unsupported temporal trigger action: {rule.trigger_action_type}")
        if rule.blocked_action_type not in _ALLOWED_ACTION_TYPES - {"*"}:
            errors.append(f"unsupported temporal blocked action: {rule.blocked_action_type}")
        if not rule.recovery_action_types or any(
            action not in _ALLOWED_ACTION_TYPES - {"*"}
            for action in rule.recovery_action_types
        ):
            errors.append("temporal rule requires supported recovery actions")
        if rule.argument_index < 0:
            errors.append("temporal rule argument index cannot be negative")
        placeholders = set(_PLACEHOLDER.findall(rule.trigger_predicate))
        if placeholders - _ALLOWED_PLACEHOLDERS:
            errors.append("temporal predicate contains an unsupported placeholder")
        elif placeholders - _ACTION_PLACEHOLDERS.get(rule.trigger_action_type, set()):
            errors.append("temporal predicate placeholder is unavailable for its trigger")
        parseable = _PLACEHOLDER.sub("placeholder", rule.trigger_predicate)
        try:
            from vista_skill.schemas import PredicateKey

            PredicateKey.parse(parseable)
        except (TypeError, ValueError):
            errors.append(f"compiled temporal predicate is invalid: {rule.trigger_predicate}")
        if _contains_instance_identifier(_PLACEHOLDER.sub("", rule.trigger_predicate)):
            errors.append("compiled temporal rule contains an instance-specific identifier")
        signature = (
            rule.trigger_action_type,
            rule.trigger_success,
            rule.trigger_predicate,
            rule.trigger_value,
            rule.blocked_action_type,
            rule.argument_index,
        )
        if signature in signatures:
            errors.append("compiled temporal rule contains a duplicate signature")
        signatures.add(signature)
    return tuple(dict.fromkeys(errors))


@dataclass(frozen=True)
class CachedTransitionCheck:
    event_id: str
    repaired: bool
    introduced_conflict: bool = False
    executable: bool = True


class TransitionChecker(Protocol):
    def check(
        self,
        parent: SkillSpec,
        candidate: SkillSpec,
        cluster: EvidenceCluster,
    ) -> Sequence[CachedTransitionCheck]: ...


class DeterministicTransitionChecker:
    """Replay cached evidence against parent and candidate structured Skills."""

    def __init__(self, action_schema: ActionSchema | None = None) -> None:
        self.action_schema = action_schema or FixedActionSchema()

    def check(
        self,
        parent: SkillSpec,
        candidate: SkillSpec,
        cluster: EvidenceCluster,
    ) -> Sequence[CachedTransitionCheck]:
        checks: list[CachedTransitionCheck] = []
        for item in cluster.items:
            if cluster.key.mismatch_kind == "trajectory_reflection":
                checks.append(
                    CachedTransitionCheck(
                        item.event_id,
                        repaired=(
                            parent.statements(cluster.key.field)
                            != candidate.statements(cluster.key.field)
                            and bool(item.evidence_delta)
                        ),
                        executable=bool(candidate.statements(cluster.key.field)),
                    )
                )
                continue
            if item.action is None or not item.evidence_delta:
                checks.append(
                    CachedTransitionCheck(
                        item.event_id,
                        repaired=False,
                        executable=False,
                    )
                )
                continue
            ledger = BeliefLedger.from_snapshot(item.pre_ledger)
            if cluster.key.field is SkillField.CONSTRAINT:
                parent_checks = self.action_schema.precondition_checks(
                    item.action, ledger, parent
                )
                candidate_checks = self.action_schema.precondition_checks(
                    item.action, ledger, candidate
                )
                target = item.mismatch.key.render()
                parent_explains = any(
                    check["predicate"] == target and not check["satisfied"]
                    for check in parent_checks
                )
                candidate_explains = any(
                    check["predicate"] == target and not check["satisfied"]
                    for check in candidate_checks
                )
                checks.append(
                    CachedTransitionCheck(
                        event_id=item.event_id,
                        repaired=candidate_explains and not parent_explains,
                        introduced_conflict=False,
                        executable=bool(candidate.statements(cluster.key.field)),
                    )
                )
                continue
            parent_expected = self.action_schema.compile(
                item.action, ledger, parent, item.goal_predicates
            )
            candidate_expected = self.action_schema.compile(
                item.action, ledger, candidate, item.goal_predicates
            )
            parent_mismatches = compare_transitions(parent_expected, item.evidence_delta)
            candidate_mismatches = compare_transitions(candidate_expected, item.evidence_delta)
            target_key = item.mismatch.key
            parent_target = {
                (mismatch.key, mismatch.kind) for mismatch in parent_mismatches
                if mismatch.key == target_key
            }
            candidate_target = {
                (mismatch.key, mismatch.kind) for mismatch in candidate_mismatches
                if mismatch.key == target_key
            }
            parent_signatures = {
                (mismatch.key, mismatch.kind) for mismatch in parent_mismatches
            }
            candidate_signatures = {
                (mismatch.key, mismatch.kind) for mismatch in candidate_mismatches
            }
            checks.append(
                CachedTransitionCheck(
                    event_id=item.event_id,
                    repaired=bool(parent_target) and len(candidate_target) < len(parent_target),
                    introduced_conflict=bool(candidate_signatures - parent_signatures),
                    executable=bool(candidate.statements(cluster.key.field)),
                )
            )
        return tuple(checks)


@dataclass(frozen=True)
class PairedEpisodeScore:
    episode_id: str
    seed: int
    parent_score: float
    candidate_score: float
    subgroup: str
    parent_success: bool | None = None
    candidate_success: bool | None = None
    semantic_tags: tuple[str, ...] = ()


class PairedEvaluator(Protocol):
    def evaluate(
        self,
        parent: SkillSpec,
        candidate: SkillSpec,
        *,
        stage: str,
        episode_budget: int,
    ) -> Sequence[PairedEpisodeScore]: ...


@dataclass(frozen=True)
class GateConfig:
    bootstrap_samples: int = 2000
    alpha: float = 0.05
    proxy_episode_budget: int = 10
    finalist_episode_budget: int = 30
    proxy_rollout_repeats: int = 1
    proxy_lcb_threshold: float = 0.0
    finalist_lcb_threshold: float = 0.0
    subgroup_regression_tolerance: float = 0.05
    semantic_affected_enabled: bool = False
    semantic_affected_lcb_threshold: float = 0.0
    semantic_protected_regression_tolerance: float = 0.05
    semantic_min_affected_tasks: int = 2
    semantic_min_protected_tasks: int = 2
    sequential_enabled: bool = False
    sequential_batch_size: int = 2
    sequential_min_episodes: int = 4
    sequential_max_abs_delta: float = 1.1
    shadow_candidates_enabled: bool = False
    shadow_min_mean_delta: float = 0.0
    random_seed: int = 0


@dataclass(frozen=True)
class GateStageResult:
    stage: str
    passed: bool
    reason: str
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason: str
    parent_version: int
    candidate_version: int | None
    patch_id: str
    stages: tuple[GateStageResult, ...]
    disposition: str | None = None

    def __post_init__(self) -> None:
        disposition = self.disposition or (
            "promoted" if self.accepted else "rejected"
        )
        if disposition not in {"rejected", "shadow", "promoted"}:
            raise ValueError(f"unsupported gate disposition: {disposition}")
        if self.accepted != (disposition == "promoted"):
            raise ValueError("only promoted Gate decisions may be accepted")
        if disposition in {"shadow", "promoted"} and self.candidate_version is None:
            raise ValueError(
                "shadow and promoted Gate decisions require a candidate version"
            )
        object.__setattr__(self, "disposition", disposition)


def shadow_candidate_eligible(
    failed: GateStageResult,
    config: GateConfig,
) -> bool:
    """Return whether a non-promoted result is safe to retain for more evidence.

    The predicate changes retention only. It never changes the full promotion
    checks in :class:`CandidateGate`.
    """

    if failed.stage not in {"paired_proxy", "paired_finalist"}:
        return False
    metrics = failed.metrics
    if float(metrics.get("sequential_early_stop", 0.0)) != 0.0:
        return False
    if float(metrics.get("episodes", 0.0)) <= 0.0:
        return False
    if config.semantic_affected_enabled and "affected_mean_delta" in metrics:
        if "protected_mean_delta" not in metrics:
            return False
        return bool(
            float(metrics["affected_mean_delta"]) > config.shadow_min_mean_delta
            and float(metrics["protected_mean_delta"])
            >= -config.semantic_protected_regression_tolerance
            and float(metrics.get("worst_subgroup_delta", -math.inf))
            >= -config.subgroup_regression_tolerance
        )
    return bool(
        float(metrics.get("mean_delta", -math.inf)) > config.shadow_min_mean_delta
        and float(metrics.get("worst_subgroup_delta", -math.inf))
        >= -config.subgroup_regression_tolerance
    )


class CandidateGate:
    def __init__(
        self,
        applier: BoundedPatchApplier,
        transition_checker: TransitionChecker,
        paired_evaluator: PairedEvaluator,
        config: GateConfig | None = None,
    ) -> None:
        self.applier = applier
        self.transition_checker = transition_checker
        self.paired_evaluator = paired_evaluator
        self.config = config or GateConfig()

    def evaluate(
        self,
        parent: SkillSpec,
        patch: SkillPatch,
        cluster: EvidenceCluster,
    ) -> tuple[GateDecision, SkillSpec | None]:
        stages: list[GateStageResult] = []
        errors = self.applier.validate(parent, patch)
        if cluster.key.skill_version != parent.version:
            errors = (*errors, "evidence cluster belongs to a different skill version")
        if patch.field is not cluster.key.field:
            errors = (*errors, "patch field does not match the evidence cluster")
        if not set(patch.evidence_ids).issubset(cluster.evidence_ids):
            errors = (*errors, "patch cites evidence outside the target cluster")
        if errors:
            stages.append(GateStageResult("static", False, "; ".join(errors)))
            return self._reject(parent, patch, stages), None
        candidate = self.applier.apply(parent, patch)
        stages.append(GateStageResult("static", True, "schema, scope, and evidence checks passed"))
        semantic_scope = _candidate_semantic_scope(cluster)

        transition_checks = tuple(self.transition_checker.check(parent, candidate, cluster))
        transition_passed = bool(transition_checks) and all(
            item.repaired and not item.introduced_conflict and item.executable
            for item in transition_checks
        )
        stages.append(
            GateStageResult(
                "transition_consistency",
                transition_passed,
                "cached transition checks passed" if transition_passed else "target was not repaired or a new conflict was introduced",
                {"checked_events": float(len(transition_checks))},
            )
        )
        if not transition_passed:
            return self._reject(parent, patch, stages), None

        proxy_stage = self._evaluate_paired_stage(
            parent,
            candidate,
            "paired_proxy",
            "proxy",
            self.config.proxy_lcb_threshold,
            self.config.proxy_episode_budget,
            semantic_scope,
        )
        stages.append(proxy_stage)
        if not proxy_stage.passed:
            decision = self._nonpromoted(parent, candidate, patch, stages)
            if decision.disposition != "shadow":
                return decision, None
            shadow_finalist = self._evaluate_paired_stage(
                parent,
                candidate,
                "paired_finalist",
                "finalist",
                self.config.finalist_lcb_threshold,
                self.config.finalist_episode_budget,
                semantic_scope,
            )
            stages.append(shadow_finalist)
            if shadow_finalist.passed or shadow_candidate_eligible(
                shadow_finalist, self.config
            ):
                return self._shadow(parent, candidate, patch, stages), candidate
            return self._reject(parent, patch, stages), None

        finalist_stage = self._evaluate_paired_stage(
            parent,
            candidate,
            "paired_finalist",
            "finalist",
            self.config.finalist_lcb_threshold,
            self.config.finalist_episode_budget,
            semantic_scope,
        )
        stages.append(finalist_stage)
        if not finalist_stage.passed:
            decision = self._nonpromoted(parent, candidate, patch, stages)
            return decision, candidate if decision.disposition == "shadow" else None
        return (
            GateDecision(
                accepted=True,
                reason="candidate passed all selection stages",
                parent_version=parent.version,
                candidate_version=candidate.version,
                patch_id=patch.patch_id,
                stages=tuple(stages),
            ),
            candidate,
        )

    def _evaluate_paired_stage(
        self,
        parent: SkillSpec,
        candidate: SkillSpec,
        result_stage: str,
        evaluator_stage: str,
        threshold: float,
        required_budget: int,
        semantic_scope: tuple[str, ...],
    ) -> GateStageResult:
        if not self.config.sequential_enabled:
            scores = tuple(
                self.paired_evaluator.evaluate(
                    parent,
                    candidate,
                    stage=evaluator_stage,
                    episode_budget=required_budget,
                )
            )
            return self._paired_stage(
                result_stage, scores, threshold, required_budget, semantic_scope
            )

        first = min(required_budget, self.config.sequential_min_episodes)
        budgets = list(range(first, required_budget + 1, self.config.sequential_batch_size))
        if not budgets or budgets[-1] != required_budget:
            budgets.append(required_budget)
        max_looks = len(budgets)
        for look, budget in enumerate(budgets, start=1):
            scores = tuple(
                self.paired_evaluator.evaluate(
                    parent,
                    candidate,
                    stage=evaluator_stage,
                    episode_budget=budget,
                )
            )
            if budget < required_budget:
                stopped = self._sequential_futility_stage(
                    result_stage,
                    scores,
                    threshold,
                    required_budget,
                    semantic_scope,
                    look=look,
                    max_looks=max_looks,
                )
                if stopped is not None:
                    return stopped
                continue
            result = self._paired_stage(
                result_stage, scores, threshold, required_budget, semantic_scope
            )
            metrics = dict(result.metrics)
            metrics.update(
                {
                    "sequential_looks": float(look),
                    "sequential_stop_budget": float(len(scores)),
                    "sequential_early_stop": 0.0,
                }
            )
            return GateStageResult(result.stage, result.passed, result.reason, metrics)
        raise RuntimeError("sequential gate produced no analysis look")

    def _sequential_futility_stage(
        self,
        stage: str,
        scores: Sequence[PairedEpisodeScore],
        threshold: float,
        required_budget: int,
        semantic_scope: tuple[str, ...],
        *,
        look: int,
        max_looks: int,
    ) -> GateStageResult | None:
        if len(scores) < self.config.sequential_min_episodes:
            return None
        pairs = {(item.episode_id, item.seed) for item in scores}
        if len(pairs) != len(scores):
            return GateStageResult(stage, False, "paired episode/seed keys are not unique")
        differences = [item.candidate_score - item.parent_score for item in scores]
        bound = self.config.sequential_max_abs_delta
        if any(not math.isfinite(value) or abs(value) > bound for value in differences):
            return GateStageResult(
                stage,
                False,
                "paired score difference violates the registered sequential bound",
            )
        task_differences = _task_mean_differences(scores)
        # Interim confidence sequences require independent task units. The
        # finalist pool repeats each task across seeds, so it remains a fixed-
        # budget task-first bootstrap unless/until complete task blocks are
        # exposed by the evaluator.
        if len(task_differences) != len(scores):
            return None
        semantic_available = bool(
            self.config.semantic_affected_enabled and semantic_scope
        )
        stream_count = 2 if semantic_available else 1
        spent_alpha = self.config.alpha / (max_looks * stream_count)
        global_lcb, global_ucb = bounded_mean_interval(
            task_differences,
            alpha=spent_alpha,
            max_abs_value=bound,
        )
        metrics = {
            "mean_delta": sum(differences) / len(differences),
            "episodes": float(len(scores)),
            "independent_tasks": float(len(task_differences)),
            "sequential_look": float(look),
            "sequential_looks": float(look),
            "sequential_stop_budget": float(len(scores)),
            "sequential_early_stop": 1.0,
            "sequential_alpha_per_bound": spent_alpha,
            "sequential_global_lcb": global_lcb,
            "sequential_global_ucb": global_ucb,
        }
        if semantic_available:
            affected = tuple(
                item
                for item in scores
                if set(item.semantic_tags).intersection(semantic_scope)
            )
            protected = tuple(item for item in scores if item not in affected)
            affected_tasks = _task_mean_differences(affected)
            protected_tasks = _task_mean_differences(protected)
            metrics.update(
                {
                    "affected_independent_tasks": float(len(affected_tasks)),
                    "protected_independent_tasks": float(len(protected_tasks)),
                }
            )
            if len(affected_tasks) >= self.config.semantic_min_affected_tasks:
                affected_lcb, affected_ucb = bounded_mean_interval(
                    affected_tasks,
                    alpha=spent_alpha,
                    max_abs_value=bound,
                )
                metrics.update(
                    {
                        "sequential_affected_lcb": affected_lcb,
                        "sequential_affected_ucb": affected_ucb,
                    }
                )
                if affected_ucb <= self.config.semantic_affected_lcb_threshold:
                    return GateStageResult(
                        stage,
                        False,
                        "sequential safe gate stopped: affected benefit is futile",
                        metrics,
                    )
            if len(protected_tasks) >= self.config.semantic_min_protected_tasks:
                protected_lcb, protected_ucb = bounded_mean_interval(
                    protected_tasks,
                    alpha=spent_alpha,
                    max_abs_value=bound,
                )
                metrics.update(
                    {
                        "sequential_protected_lcb": protected_lcb,
                        "sequential_protected_ucb": protected_ucb,
                    }
                )
                if protected_ucb < -self.config.semantic_protected_regression_tolerance:
                    return GateStageResult(
                        stage,
                        False,
                        "sequential safe gate stopped: protected regression is identified",
                        metrics,
                    )
            return None

        remaining = required_budget - len(scores)
        deterministic_best_final = (
            sum(differences) + remaining * bound
        ) / required_budget
        metrics["sequential_best_possible_final_mean"] = deterministic_best_final
        if global_ucb <= threshold or deterministic_best_final <= threshold:
            return GateStageResult(
                stage,
                False,
                "sequential safe gate stopped: global benefit is futile",
                metrics,
            )
        return None

    def _paired_stage(
        self,
        stage: str,
        scores: Sequence[PairedEpisodeScore],
        threshold: float,
        required_budget: int,
        semantic_scope: tuple[str, ...] = (),
    ) -> GateStageResult:
        if not scores:
            return GateStageResult(stage, False, "paired evaluator returned no episodes")
        if len(scores) < required_budget:
            return GateStageResult(
                stage,
                False,
                f"paired evaluator returned {len(scores)} of {required_budget} required episodes",
            )
        pairs = {(item.episode_id, item.seed) for item in scores}
        if len(pairs) != len(scores):
            return GateStageResult(stage, False, "paired episode/seed keys are not unique")
        differences = [item.candidate_score - item.parent_score for item in scores]
        task_differences = _task_mean_differences(scores)
        lcb = bootstrap_lcb(
            task_differences,
            alpha=self.config.alpha,
            samples=self.config.bootstrap_samples,
            seed=self.config.random_seed,
        )
        subgroup_deltas = _subgroup_deltas(scores)
        worst_group = min(subgroup_deltas.values())
        task_rollout_counts: dict[str, int] = {}
        for item in scores:
            task_rollout_counts[item.episode_id] = (
                task_rollout_counts.get(item.episode_id, 0) + 1
            )
        metrics = {
            "mean_delta": sum(differences) / len(differences),
            "lcb": lcb,
            "worst_subgroup_delta": worst_group,
            "episodes": float(len(scores)),
            "independent_tasks": float(len(task_differences)),
            "repeated_tasks": float(
                sum(count > 1 for count in task_rollout_counts.values())
            ),
            "max_rollouts_per_task": float(max(task_rollout_counts.values())),
            "semantic_scope_tag_count": float(len(semantic_scope)),
        }
        if all(
            item.parent_success is not None and item.candidate_success is not None
            for item in scores
        ):
            task_ids = tuple(task_rollout_counts)
            metrics.update(
                {
                    "parent_pass_at_k": sum(
                        any(
                            bool(item.parent_success)
                            for item in scores
                            if item.episode_id == task_id
                        )
                        for task_id in task_ids
                    )
                    / len(task_ids),
                    "candidate_pass_at_k": sum(
                        any(
                            bool(item.candidate_success)
                            for item in scores
                            if item.episode_id == task_id
                        )
                        for task_id in task_ids
                    )
                    / len(task_ids),
                }
            )
        semantic_available = bool(
            self.config.semantic_affected_enabled and semantic_scope
        )
        if semantic_available:
            affected = tuple(
                item
                for item in scores
                if set(item.semantic_tags).intersection(semantic_scope)
            )
            protected = tuple(item for item in scores if item not in affected)
            affected_tasks = _task_mean_differences(affected)
            protected_tasks = _task_mean_differences(protected)
            affected_lcb = bootstrap_lcb(
                affected_tasks,
                alpha=self.config.alpha,
                samples=self.config.bootstrap_samples,
                seed=self.config.random_seed + 1000,
            )
            protected_lcb = bootstrap_lcb(
                protected_tasks,
                alpha=self.config.alpha,
                samples=self.config.bootstrap_samples,
                seed=self.config.random_seed + 2000,
            )
            metrics.update(
                {
                    "affected_independent_tasks": float(len(affected_tasks)),
                    "protected_independent_tasks": float(len(protected_tasks)),
                }
            )
            if affected_tasks:
                metrics.update(
                    {
                        "affected_mean_delta": sum(affected_tasks)
                        / len(affected_tasks),
                        "affected_lcb": affected_lcb,
                    }
                )
            if protected_tasks:
                metrics.update(
                    {
                        "protected_mean_delta": sum(protected_tasks)
                        / len(protected_tasks),
                        "protected_lcb": protected_lcb,
                    }
                )
            coverage_ok = bool(
                len(affected_tasks) >= self.config.semantic_min_affected_tasks
                and len(protected_tasks) >= self.config.semantic_min_protected_tasks
            )
            passed = bool(
                coverage_ok
                and affected_lcb > self.config.semantic_affected_lcb_threshold
                and protected_lcb
                >= -self.config.semantic_protected_regression_tolerance
            )
            if not coverage_ok:
                reason = "semantic affected/protected task coverage is incomplete"
            elif passed:
                reason = "semantic affected benefit and protected non-inferiority checks passed"
            else:
                reason = "affected-task benefit is unproven or protected tasks regressed"
        else:
            passed = bool(
                lcb > threshold
                and worst_group >= -self.config.subgroup_regression_tolerance
            )
            reason = (
                "paired lower bound and subgroup checks passed"
                if passed
                else "paired lower bound is non-positive or a protected subgroup regressed"
            )
        return GateStageResult(
            stage,
            passed,
            reason,
            metrics,
        )

    @staticmethod
    def _reject(
        parent: SkillSpec,
        patch: SkillPatch,
        stages: list[GateStageResult],
    ) -> GateDecision:
        return GateDecision(
            accepted=False,
            reason=stages[-1].reason,
            parent_version=parent.version,
            candidate_version=None,
            patch_id=patch.patch_id,
            stages=tuple(stages),
        )

    def _nonpromoted(
        self,
        parent: SkillSpec,
        candidate: SkillSpec,
        patch: SkillPatch,
        stages: list[GateStageResult],
    ) -> GateDecision:
        failed = stages[-1]
        if self.config.shadow_candidates_enabled and shadow_candidate_eligible(
            failed, self.config
        ):
            return self._shadow(parent, candidate, patch, stages)
        return self._reject(parent, patch, stages)

    @staticmethod
    def _shadow(
        parent: SkillSpec,
        candidate: SkillSpec,
        patch: SkillPatch,
        stages: list[GateStageResult],
    ) -> GateDecision:
        latest = stages[-1]
        return GateDecision(
            accepted=False,
            reason=f"candidate retained in shadow: {latest.reason}",
            parent_version=parent.version,
            candidate_version=candidate.version,
            patch_id=patch.patch_id,
            stages=tuple(stages),
            disposition="shadow",
        )


class LineageWriter(Protocol):
    def append(
        self,
        *,
        parent: SkillSpec,
        candidate: SkillSpec | None,
        patch: SkillPatch,
        decision: GateDecision,
        protocol: Mapping[str, object] | None = None,
    ) -> object: ...


@dataclass(frozen=True)
class EvolutionResult:
    cluster_key: str
    patch: SkillPatch
    decision: GateDecision
    parent: SkillSpec
    candidate: SkillSpec | None


class EvolutionCoordinator:
    """Common updater backend shared by VTCA and controlled frontends."""

    def __init__(
        self,
        generator: PatchGenerator,
        gate: CandidateGate,
        lineage: LineageWriter,
        *,
        protocol: Mapping[str, object] | None = None,
        max_proposals_per_round: int | None = None,
    ) -> None:
        self.generator = generator
        self.gate = gate
        self.lineage = lineage
        self.protocol = protocol or {}
        if max_proposals_per_round is not None and max_proposals_per_round < 1:
            raise ValueError("max_proposals_per_round must be positive")
        self.max_proposals_per_round = max_proposals_per_round
        self._processed: set[str] = set()

    def evolve(
        self,
        parent: SkillSpec,
        clusters: Sequence[EvidenceCluster],
    ) -> tuple[SkillSpec, tuple[EvolutionResult, ...]]:
        active = parent
        results: list[EvolutionResult] = []
        proposed = 0
        for cluster in sorted(clusters, key=_cluster_priority):
            if (
                self.max_proposals_per_round is not None
                and proposed >= self.max_proposals_per_round
            ):
                break
            if (
                cluster.key.skill_id != active.skill_id
                or cluster.key.skill_version != active.version
            ):
                continue
            fingerprint = _cluster_fingerprint(cluster)
            if fingerprint in self._processed:
                continue
            self._processed.add(fingerprint)
            proposed += 1
            patch = self.generator.propose(active, cluster)
            proposed_candidate = None
            if not self.gate.applier.validate(active, patch):
                proposed_candidate = self.gate.applier.apply(active, patch)
            decision, accepted_candidate = self.gate.evaluate(active, patch, cluster)
            self.lineage.append(
                parent=active,
                candidate=proposed_candidate,
                patch=patch,
                decision=decision,
                protocol=self.protocol,
            )
            results.append(
                EvolutionResult(
                    str(cluster.key), patch, decision, active, proposed_candidate
                )
            )
            if decision.accepted and accepted_candidate is not None:
                active = accepted_candidate
        return active, tuple(results)

def bootstrap_lcb(
    differences: Sequence[float],
    *,
    alpha: float,
    samples: int,
    seed: int,
) -> float:
    if not differences:
        return -math.inf
    if len(differences) == 1 or samples <= 1:
        return float(differences[0])
    rng = random.Random(seed)
    means = []
    size = len(differences)
    for _ in range(samples):
        means.append(sum(rng.choice(differences) for _ in range(size)) / size)
    means.sort()
    index = max(0, min(len(means) - 1, math.floor(alpha * len(means))))
    return means[index]


def bounded_mean_interval(
    values: Sequence[float],
    *,
    alpha: float,
    max_abs_value: float,
) -> tuple[float, float]:
    """Finite-look Hoeffding interval for bounded paired task differences.

    ``CandidateGate`` divides alpha across all registered looks/streams before
    calling this helper. The resulting union bound remains valid when an
    interim futility decision stops rollout collection early.
    """
    if not values:
        return -math.inf, math.inf
    if not 0.0 < alpha < 1.0:
        raise ValueError("sequential alpha must be in (0, 1)")
    if max_abs_value <= 0.0:
        raise ValueError("sequential paired-difference bound must be positive")
    if any(not math.isfinite(value) or abs(value) > max_abs_value for value in values):
        raise ValueError("value violates the registered paired-difference bound")
    mean = sum(values) / len(values)
    radius = max_abs_value * math.sqrt(
        2.0 * math.log(2.0 / alpha) / len(values)
    )
    return max(-max_abs_value, mean - radius), min(max_abs_value, mean + radius)


def _subgroup_deltas(scores: Sequence[PairedEpisodeScore]) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for item in scores:
        groups.setdefault(item.subgroup, []).append(item.candidate_score - item.parent_score)
    return {name: sum(values) / len(values) for name, values in groups.items()}


def _task_mean_differences(scores: Sequence[PairedEpisodeScore]) -> list[float]:
    """Aggregate within-task seeds before the nonparametric bootstrap."""
    tasks: dict[str, list[float]] = {}
    for item in scores:
        tasks.setdefault(item.episode_id, []).append(
            item.candidate_score - item.parent_score
        )
    return [sum(values) / len(values) for values in tasks.values()]


def _candidate_semantic_scope(cluster: EvidenceCluster) -> tuple[str, ...]:
    """Derive an outcome-independent task scope from structured evidence.

    Action-local procedure/effect/constraint revisions are evaluated on tasks
    whose precomputed goal semantics require the triggering primitive. Global
    activation/termination changes retain the conservative global gate.
    """

    if cluster.key.field not in {
        SkillField.PROCEDURE,
        SkillField.EFFECT,
        SkillField.CONSTRAINT,
    }:
        return ()
    action_types = {
        item.action.action_type.strip().lower()
        for item in cluster.items
        if item.action is not None and item.action.action_type.strip()
    }
    action_types.discard("*")
    return tuple(sorted(f"action:{action_type}" for action_type in action_types))


def make_patch_id(
    skill: SkillSpec,
    field: SkillField,
    operation: PatchOperation,
    old: str,
    new: str,
    evidence_ids: Sequence[str],
    termination_policy: TerminationPolicy | None = None,
    prediction_rules: Sequence[SkillPredictionRule] = (),
    temporal_rules: Sequence[TemporalSkillRule] = (),
) -> str:
    raw = "|".join(
        (
            skill.skill_id,
            str(skill.version),
            field.value,
            operation.value,
            old,
            new,
            "" if termination_policy is None else termination_policy.value,
            *(f"{rule.rule_id}:{rule.predicate}:{rule.after.value}" for rule in prediction_rules),
            *(
                f"{rule.rule_id}:{rule.trigger_action_type}:{rule.trigger_success}:"
                f"{rule.trigger_predicate}:{rule.trigger_value.value}:"
                f"{rule.blocked_action_type}:{','.join(rule.recovery_action_types)}"
                for rule in temporal_rules
            ),
            *evidence_ids,
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _cluster_fingerprint(cluster: EvidenceCluster) -> str:
    is_discovery = bool(
        cluster.items
        and all(
            item.attribution.update_kind is SkillUpdateKind.DISCOVERY
            for item in cluster.items
        )
    )
    # A deterministic Discovery patch is uniquely determined by its
    # generalized cluster key and parent Skill version. Additional supporting
    # evidence strengthens the same candidate but must not re-spend a paired
    # gate on every acquisition episode.
    raw = (
        str(cluster.key)
        if is_discovery
        else "|".join((str(cluster.key), *sorted(cluster.evidence_ids)))
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cluster_priority(cluster: EvidenceCluster) -> tuple[int, str]:
    priority = {
        SkillField.CONSTRAINT: 0,
        SkillField.PROCEDURE: 1,
        SkillField.TERMINATION: 2,
        SkillField.ACTIVATION: 3,
        SkillField.EFFECT: 4,
    }
    return priority[cluster.key.field], str(cluster.key)
