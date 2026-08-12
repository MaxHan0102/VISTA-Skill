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
    TerminationPolicy,
    TruthValue,
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
        rule_errors = _validate_prediction_rules(candidate.prediction_rules)
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
    proxy_lcb_threshold: float = 0.0
    finalist_lcb_threshold: float = 0.0
    subgroup_regression_tolerance: float = 0.05
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

        proxy = tuple(
            self.paired_evaluator.evaluate(
                parent,
                candidate,
                stage="proxy",
                episode_budget=self.config.proxy_episode_budget,
            )
        )
        proxy_stage = self._paired_stage(
            "paired_proxy",
            proxy,
            self.config.proxy_lcb_threshold,
            self.config.proxy_episode_budget,
        )
        stages.append(proxy_stage)
        if not proxy_stage.passed:
            return self._reject(parent, patch, stages), None

        finalist = tuple(
            self.paired_evaluator.evaluate(
                parent,
                candidate,
                stage="finalist",
                episode_budget=self.config.finalist_episode_budget,
            )
        )
        finalist_stage = self._paired_stage(
            "paired_finalist",
            finalist,
            self.config.finalist_lcb_threshold,
            self.config.finalist_episode_budget,
        )
        stages.append(finalist_stage)
        if not finalist_stage.passed:
            return self._reject(parent, patch, stages), None
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

    def _paired_stage(
        self,
        stage: str,
        scores: Sequence[PairedEpisodeScore],
        threshold: float,
        required_budget: int,
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
        passed = lcb > threshold and worst_group >= -self.config.subgroup_regression_tolerance
        reason = (
            "paired lower bound and subgroup checks passed"
            if passed
            else "paired lower bound is non-positive or a protected subgroup regressed"
        )
        return GateStageResult(
            stage,
            passed,
            reason,
            {
                "mean_delta": sum(differences) / len(differences),
                "lcb": lcb,
                "worst_subgroup_delta": worst_group,
                "episodes": float(len(scores)),
                "independent_tasks": float(len(task_differences)),
            },
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
    ) -> None:
        self.generator = generator
        self.gate = gate
        self.lineage = lineage
        self.protocol = protocol or {}
        self._processed: set[str] = set()

    def evolve(
        self,
        parent: SkillSpec,
        clusters: Sequence[EvidenceCluster],
    ) -> tuple[SkillSpec, tuple[EvolutionResult, ...]]:
        active = parent
        results: list[EvolutionResult] = []
        for cluster in clusters:
            if (
                cluster.key.skill_id != active.skill_id
                or cluster.key.skill_version != active.version
            ):
                continue
            fingerprint = _cluster_fingerprint(cluster)
            if fingerprint in self._processed:
                continue
            self._processed.add(fingerprint)
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


def make_patch_id(
    skill: SkillSpec,
    field: SkillField,
    operation: PatchOperation,
    old: str,
    new: str,
    evidence_ids: Sequence[str],
    termination_policy: TerminationPolicy | None = None,
    prediction_rules: Sequence[SkillPredictionRule] = (),
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
            *evidence_ids,
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _cluster_fingerprint(cluster: EvidenceCluster) -> str:
    raw = "|".join((str(cluster.key), *sorted(cluster.evidence_ids)))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
