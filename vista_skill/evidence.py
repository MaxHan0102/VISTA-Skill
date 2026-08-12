from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from vista_skill.schemas import (
    EvidenceRequest,
    EvidenceSource,
    PredicateEvidence,
    PredicateKey,
    TruthValue,
    unique_strings,
)


class VisualEvidenceProvider(Protocol):
    def extract(self, request: EvidenceRequest) -> Sequence[PredicateEvidence]: ...


@dataclass(frozen=True)
class EvidenceExtractorConfig:
    rule_confidence: float = 0.98
    visual_action_types: tuple[str, ...] = ("place",)
    min_visual_confidence: float = 0.5


# A feedback strategy turns environment feedback into predicate evidence via the
# supplied ``add(key, after, rationale, confidence=None)`` closure. The default
# reproduces EB-Habitat semantics; EB-Navigation supplies its own.
FeedbackEvidenceStrategy = Callable[[EvidenceRequest, Callable[..., None]], None]

# EB-Navigation: SUCCESS_THRESHOLD in EBNavEnv is 1.0m from the target.
_NAV_NEAR_THRESHOLD = 1.0
_NAV_DISTANCE_RE = re.compile(r"target distance:\s*([0-9.]+)\s*m", re.IGNORECASE)
_NAV_STRUCTURAL_PREDICATE = {
    "move_forward": "position_changed",
    "move_backward": "position_changed",
    "move_right": "position_changed",
    "move_left": "position_changed",
    "turn_right": "heading_changed",
    "turn_left": "heading_changed",
    "look_up": "camera_tilt_changed",
    "look_down": "camera_tilt_changed",
}


class EvidenceExtractor:
    """Information-isolated rule-first evidence branch."""

    def __init__(
        self,
        visual_provider: VisualEvidenceProvider | None = None,
        config: EvidenceExtractorConfig | None = None,
        feedback_strategy: FeedbackEvidenceStrategy | None = None,
    ) -> None:
        self.visual_provider = visual_provider
        self.config = config or EvidenceExtractorConfig()
        self.feedback_strategy = feedback_strategy or _habitat_feedback_strategy

    def extract(self, request: EvidenceRequest) -> tuple[PredicateEvidence, ...]:
        evidence = list(self._from_feedback(request))
        covered_keys = {item.key for item in evidence if item.after is not TruthValue.UNKNOWN}
        unresolved_goals = any(key not in covered_keys for key in request.goal_predicates)
        needs_visual = (
            request.action.action_type in self.config.visual_action_types
            or unresolved_goals
        )
        if self.visual_provider is not None and needs_visual:
            visual_items = self.visual_provider.extract(request)
            evidence.extend(
                item
                for item in visual_items
                if item.confidence >= self.config.min_visual_confidence
            )
        completion = self._derive_task_completion(request, evidence)
        if completion is not None:
            evidence.append(completion)
        return self._deduplicate(evidence)

    @staticmethod
    def _derive_task_completion(
        request: EvidenceRequest,
        evidence: Sequence[PredicateEvidence],
    ) -> PredicateEvidence | None:
        if not request.goal_predicates:
            return None
        post_values = {item.key: item.after for item in evidence}
        pre_values = {item.key: item.value for item in request.pre_ledger}
        values = [
            post_values.get(key, pre_values.get(key, TruthValue.UNKNOWN))
            for key in request.goal_predicates
        ]
        if all(value is TruthValue.TRUE for value in values):
            after = TruthValue.TRUE
            confidence = min(
                (item.confidence for item in evidence if item.key in request.goal_predicates),
                default=0.9,
            )
        elif any(value is TruthValue.FALSE for value in values):
            after = TruthValue.FALSE
            confidence = min(
                (item.confidence for item in evidence if item.key in request.goal_predicates),
                default=0.8,
            )
        else:
            after = TruthValue.UNKNOWN
            confidence = 0.5
        key = PredicateKey("task_complete")
        before = pre_values.get(key, TruthValue.UNKNOWN)
        support_ids = sorted(
            item.evidence_id for item in evidence if item.key in request.goal_predicates
        )
        evidence_id = (
            f"{request.episode_id}:s{request.step_id}:goal:"
            + ("+".join(support_ids) if support_ids else "ledger")
        )
        return PredicateEvidence(
            key=key,
            before=before,
            after=after,
            confidence=confidence,
            source=EvidenceSource.DERIVED_GOAL,
            evidence_id=evidence_id,
            timestamp=request.step_id,
            view_id=f"{request.episode_id}:s{request.step_id}:goal",
            coverage=sum(value is not TruthValue.UNKNOWN for value in values) / len(values),
            rationale="derived from independently grounded goal predicates",
        )

    def _from_feedback(self, request: EvidenceRequest) -> tuple[PredicateEvidence, ...]:
        items: list[PredicateEvidence] = []

        def add(
            key: PredicateKey,
            after: TruthValue,
            rationale: str,
            confidence: float | None = None,
        ) -> None:
            before_state = next((x for x in request.pre_ledger if x.key == key), None)
            items.append(
                PredicateEvidence(
                    key=key,
                    before=TruthValue.UNKNOWN if before_state is None else before_state.value,
                    after=after,
                    confidence=self.config.rule_confidence if confidence is None else confidence,
                    source=EvidenceSource.ENV_FEEDBACK,
                    evidence_id=(
                        f"{request.episode_id}:s{request.step_id}:feedback:{key.render()}"
                    ),
                    timestamp=request.step_id,
                    view_id=f"{request.episode_id}:s{request.step_id}:post",
                    coverage=1.0,
                    rationale=rationale,
                )
            )

        self.feedback_strategy(request, add)
        return tuple(items)

    @staticmethod
    def _deduplicate(items: Sequence[PredicateEvidence]) -> tuple[PredicateEvidence, ...]:
        grouped: dict[PredicateKey, PredicateEvidence] = {}
        for item in items:
            previous = grouped.get(item.key)
            if previous is None or item.confidence > previous.confidence:
                grouped[item.key] = item
            elif item.confidence == previous.confidence and item.after != previous.after:
                grouped[item.key] = PredicateEvidence(
                    key=item.key,
                    before=item.before,
                    after=TruthValue.UNKNOWN,
                    confidence=item.confidence,
                    source=item.source,
                    evidence_id="|".join(unique_strings((previous.evidence_id, item.evidence_id))),
                    timestamp=max(previous.timestamp, item.timestamp),
                    view_id=item.view_id,
                    coverage=max(previous.coverage, item.coverage),
                    task_relevance=max(previous.task_relevance, item.task_relevance),
                    rationale="conflicting evidence",
                )
        return tuple(grouped.values())


def _habitat_feedback_strategy(request: EvidenceRequest, add: Callable[..., None]) -> None:
    """Default EB-Habitat feedback -> evidence mapping (pick/place/nav/open/close)."""
    action = request.action
    feedback = request.feedback.lower()
    success = request.last_action_success

    if success is True and action.action_type == "pick" and action.arguments:
        add(PredicateKey("holding", (action.arguments[0],)), TruthValue.TRUE, feedback)
        add(PredicateKey("not_holding"), TruthValue.FALSE, feedback)
    elif success is True and action.action_type == "place":
        add(PredicateKey("not_holding"), TruthValue.TRUE, feedback)
        for state in request.pre_ledger:
            if state.key.name == "holding" and state.value is TruthValue.TRUE:
                add(state.key, TruthValue.FALSE, feedback)
    elif success is True and action.action_type == "nav" and action.arguments:
        add(PredicateKey("near", (action.arguments[0],)), TruthValue.TRUE, feedback, 0.9)
    elif success is True and action.action_type in {"open", "close"} and action.arguments:
        add(
            PredicateKey("open", (action.arguments[0],)),
            TruthValue.TRUE if action.action_type == "open" else TruthValue.FALSE,
            feedback,
        )

    if success is False and "not near" in feedback and action.arguments:
        target = action.arguments[-1] if action.action_type == "place" else action.arguments[0]
        add(PredicateKey("near", (target,)), TruthValue.FALSE, feedback, 0.9)
    if success is False and "when holding something" in feedback:
        add(PredicateKey("not_holding"), TruthValue.FALSE, feedback, 0.95)
    if success is False and "when not holding" in feedback:
        add(PredicateKey("not_holding"), TruthValue.TRUE, feedback, 0.95)

    opened_match = re.search(r"now (?:the )?(.+?) is open", feedback)
    closed_match = re.search(r"now (?:the )?(.+?) is closed", feedback)
    if opened_match:
        add(PredicateKey("open", (_feedback_entity(opened_match.group(1)),)), TruthValue.TRUE, feedback)
    if closed_match:
        add(PredicateKey("open", (_feedback_entity(closed_match.group(1)),)), TruthValue.FALSE, feedback)


def _nav_distance(feedback: str) -> float | None:
    match = _NAV_DISTANCE_RE.search(feedback or "")
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _nav_feedback_strategy(request: EvidenceRequest, add: Callable[..., None]) -> None:
    """EB-Navigation feedback -> evidence mapping.

    Goal predicate ``near(target)`` is derived from the distance the env adapter
    appends to ``env_feedback``; structural transient predicates are
    confirmed/refuted by ``last_action_success``. ``near(target)`` is NEVER
    predicted by the action schema, only observed here.
    """
    action = request.action
    feedback = request.feedback
    success = request.last_action_success

    distance = _nav_distance(feedback)
    if distance is not None:
        for goal in request.goal_predicates:
            if goal.name == "near" and goal.arguments:
                add(
                    PredicateKey("near", goal.arguments),
                    TruthValue.TRUE if distance <= _NAV_NEAR_THRESHOLD else TruthValue.FALSE,
                    f"target distance {distance:.3f}m",
                    0.9,
                )

    structural = _NAV_STRUCTURAL_PREDICATE.get(action.action_type)
    if structural is not None:
        if success is True:
            add(PredicateKey(structural), TruthValue.TRUE, feedback)
        elif success is False:
            add(PredicateKey(structural), TruthValue.FALSE, feedback)


def _feedback_entity(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
