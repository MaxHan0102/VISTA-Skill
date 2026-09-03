from __future__ import annotations

import threading
from typing import Callable

from vista_skill.belief import BeliefLedger
from vista_skill.schemas import SkillSpec
from vista_skill.skills import render_skill


_USAGE_KEYS = ("calls", "prompt_tokens", "completion_tokens")


class ExecutorUsageTracker:
    """Run-scoped executor usage, aggregated across short-lived rollout runners."""

    def __init__(self) -> None:
        self._by_phase: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def activate(self, phase: str) -> None:
        with self._lock:
            self._by_phase.setdefault(phase, {key: 0 for key in _USAGE_KEYS})

    def record(self, phase: str, usage) -> None:  # type: ignore[no-untyped-def]
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        with self._lock:
            counter = self._by_phase.setdefault(
                phase, {key: 0 for key in _USAGE_KEYS}
            )
            counter["calls"] += 1
            counter["prompt_tokens"] += prompt_tokens
            counter["completion_tokens"] += completion_tokens

    def phase_payload(self, phase: str) -> dict[str, int] | None:
        with self._lock:
            counter = self._by_phase.get(phase)
            return None if counter is None else dict(counter)

    def payload(self) -> dict[str, object] | None:
        with self._lock:
            if not self._by_phase:
                return None
            by_phase = {
                phase: dict(counter)
                for phase, counter in sorted(self._by_phase.items())
            }
        total = {
            key: sum(counter[key] for counter in by_phase.values())
            for key in _USAGE_KEYS
        }
        return {**total, "by_phase": by_phase}


class _SeededCompletions:
    def __init__(
        self,
        completions,
        seed: int,
        usage_tracker: ExecutorUsageTracker,
        usage_phase: str,
    ) -> None:  # type: ignore[no-untyped-def]
        self._completions = completions
        self._seed = int(seed)
        self._usage_tracker = usage_tracker
        self._usage_phase = usage_phase
        # The seed wrapper is the single chokepoint every executor request passes
        # through, so it is the natural place to count executor calls/tokens for
        # §6.14 cost reporting without modifying EmbodiedBench.
        self._usage_tracker.activate(usage_phase)

    @property
    def usage(self) -> dict[str, int]:
        return self._usage_tracker.phase_payload(self._usage_phase) or {
            key: 0 for key in _USAGE_KEYS
        }

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self._completions, name)

    def create(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        supplied = kwargs.get("seed")
        if supplied is not None and int(supplied) != self._seed:
            raise ValueError("executor request seed differs from paired rollout seed")
        kwargs["seed"] = self._seed
        response = self._completions.create(*args, **kwargs)
        self._usage_tracker.record(
            self._usage_phase, getattr(response, "usage", None)
        )
        return response


class _SeededChat:
    def __init__(
        self,
        chat,
        seed: int,
        usage_tracker: ExecutorUsageTracker,
        usage_phase: str,
    ) -> None:  # type: ignore[no-untyped-def]
        self._chat = chat
        self.completions = _SeededCompletions(
            chat.completions, seed, usage_tracker, usage_phase
        )

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self._chat, name)


class _SeededOpenAIClient:
    def __init__(
        self,
        client,
        seed: int,
        usage_tracker: ExecutorUsageTracker,
        usage_phase: str,
    ) -> None:  # type: ignore[no-untyped-def]
        self._client = client
        self.chat = _SeededChat(client.chat, seed, usage_tracker, usage_phase)

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self._client, name)


def configure_planner_inference_seed(
    planner,
    seed: int,
    *,
    usage_tracker: ExecutorUsageTracker | None = None,
    usage_phase: str = "executor",
) -> None:  # type: ignore[no-untyped-def]
    """Inject the paired rollout seed without modifying EmbodiedBench."""
    remote_model = getattr(planner, "model", None)
    client = getattr(remote_model, "model", None)
    chat = getattr(client, "chat", None)
    if chat is None or getattr(chat, "completions", None) is None:
        raise RuntimeError("remote executor does not expose OpenAI-compatible chat completions")
    tracker = usage_tracker or ExecutorUsageTracker()
    seeded_client = _SeededOpenAIClient(
        client, seed, tracker, usage_phase
    )
    remote_model.model = seeded_client
    # Expose the shared tracker and phase on the planner so each completed
    # rollout artifact can record an independently auditable episode delta.
    planner._vista_executor_usage_tracker = tracker
    planner._vista_executor_usage_phase = usage_phase


def compact_ledger(ledger: BeliefLedger, *, max_items: int = 24) -> str:
    states = sorted(
        ledger.snapshot(relevant_only=True),
        key=lambda item: (item.task_relevance, item.timestamp, item.confidence),
        reverse=True,
    )[:max_items]
    if not states:
        return "- No task-relevant predicate has reliable evidence yet."
    return "\n".join(
        f"- {item.key.render()} = {item.value.value} "
        f"(confidence={item.confidence:.2f}, source={item.source})"
        for item in states
    )


class SkillPromptMixin:
    """Mixin for EmbodiedBench VLMPlanner; override only prompt construction."""

    _vista_skill_provider: Callable[[], SkillSpec]
    _vista_ledger_provider: Callable[[], BeliefLedger]
    _vista_emphasis_provider: Callable[[], str]
    _vista_observation_meta_skill_provider: Callable[[], str]
    _vista_temporal_rule_provider: Callable[[], str]

    def configure_vista_prompt(
        self,
        skill_provider: Callable[[], SkillSpec],
        ledger_provider: Callable[[], BeliefLedger],
        emphasis_provider: Callable[[], str] | None = None,
        observation_meta_skill_provider: Callable[[], str] | None = None,
        temporal_rule_provider: Callable[[], str] | None = None,
    ) -> None:
        self._vista_skill_provider = skill_provider
        self._vista_ledger_provider = ledger_provider
        self._vista_emphasis_provider = emphasis_provider or (lambda: "")
        self._vista_observation_meta_skill_provider = (
            observation_meta_skill_provider or (lambda: "")
        )
        self._vista_temporal_rule_provider = temporal_rule_provider or (lambda: "")

    def process_prompt(self, user_instruction, prev_act_feedback=()):  # type: ignore[no-untyped-def]
        prompt = super().process_prompt(user_instruction, prev_act_feedback)  # type: ignore[misc]
        if not hasattr(self, "_vista_skill_provider"):
            return prompt
        skill = self._vista_skill_provider()
        ledger = self._vista_ledger_provider()
        emphasis = self._vista_emphasis_provider()
        observation_meta_skill = self._vista_observation_meta_skill_provider()
        temporal_rules = self._vista_temporal_rule_provider()
        emphasis_section = (
            "\n\n## Temporary execution emphasis\n" + emphasis if emphasis else ""
        )
        observation_section = (
            "\n\n## Frozen observation-and-recovery skill\n" + observation_meta_skill
            if observation_meta_skill
            else ""
        )
        temporal_section = (
            "\n\n## Active sequential Skill obligations\n" + temporal_rules
            if temporal_rules
            else ""
        )
        return (
            prompt
            + "\n\n## Active procedural skill\n"
            + render_skill(skill)
            + "\n\n## Evidence-supported local belief\n"
            + compact_ledger(ledger)
            + emphasis_section
            + observation_section
            + temporal_section
            + "\nUse unknown predicates as a reason to observe or replan, never as false facts."
        )


def make_skill_aware_planner(base_planner_class):  # type: ignore[no-untyped-def]
    """Create a planner subclass without importing EmbodiedBench at package import time."""
    return type("SkillAwareVLMPlanner", (SkillPromptMixin, base_planner_class), {})
