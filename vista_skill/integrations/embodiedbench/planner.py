from __future__ import annotations

from typing import Callable

from vista_skill.belief import BeliefLedger
from vista_skill.schemas import SkillSpec
from vista_skill.skills import render_skill


class _SeededCompletions:
    def __init__(self, completions, seed: int) -> None:  # type: ignore[no-untyped-def]
        self._completions = completions
        self._seed = int(seed)
        # The seed wrapper is the single chokepoint every executor request passes
        # through, so it is the natural place to count executor calls/tokens for
        # §6.14 cost reporting without modifying EmbodiedBench.
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self._completions, name)

    def create(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        supplied = kwargs.get("seed")
        if supplied is not None and int(supplied) != self._seed:
            raise ValueError("executor request seed differs from paired rollout seed")
        kwargs["seed"] = self._seed
        response = self._completions.create(*args, **kwargs)
        usage = getattr(response, "usage", None)
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
        return response


class _SeededChat:
    def __init__(self, chat, seed: int) -> None:  # type: ignore[no-untyped-def]
        self._chat = chat
        self.completions = _SeededCompletions(chat.completions, seed)

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self._chat, name)


class _SeededOpenAIClient:
    def __init__(self, client, seed: int) -> None:  # type: ignore[no-untyped-def]
        self._client = client
        self.chat = _SeededChat(client.chat, seed)

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self._client, name)


def configure_planner_inference_seed(planner, seed: int) -> None:  # type: ignore[no-untyped-def]
    """Inject the paired rollout seed without modifying EmbodiedBench."""
    remote_model = getattr(planner, "model", None)
    client = getattr(remote_model, "model", None)
    chat = getattr(client, "chat", None)
    if chat is None or getattr(chat, "completions", None) is None:
        raise RuntimeError("remote executor does not expose OpenAI-compatible chat completions")
    seeded_client = _SeededOpenAIClient(client, seed)
    remote_model.model = seeded_client
    # Expose the executor token-usage accumulator on the planner so cost
    # reporting can read it later without modifying EmbodiedBench source.
    planner._vista_executor_usage = seeded_client.chat.completions.usage


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

    def configure_vista_prompt(
        self,
        skill_provider: Callable[[], SkillSpec],
        ledger_provider: Callable[[], BeliefLedger],
        emphasis_provider: Callable[[], str] | None = None,
    ) -> None:
        self._vista_skill_provider = skill_provider
        self._vista_ledger_provider = ledger_provider
        self._vista_emphasis_provider = emphasis_provider or (lambda: "")

    def process_prompt(self, user_instruction, prev_act_feedback=()):  # type: ignore[no-untyped-def]
        prompt = super().process_prompt(user_instruction, prev_act_feedback)  # type: ignore[misc]
        if not hasattr(self, "_vista_skill_provider"):
            return prompt
        skill = self._vista_skill_provider()
        ledger = self._vista_ledger_provider()
        emphasis = self._vista_emphasis_provider()
        emphasis_section = (
            "\n\n## Temporary execution emphasis\n" + emphasis if emphasis else ""
        )
        return (
            prompt
            + "\n\n## Active procedural skill\n"
            + render_skill(skill)
            + "\n\n## Evidence-supported local belief\n"
            + compact_ledger(ledger)
            + emphasis_section
            + "\nUse unknown predicates as a reason to observe or replan, never as false facts."
        )


def make_skill_aware_planner(base_planner_class):  # type: ignore[no-untyped-def]
    """Create a planner subclass without importing EmbodiedBench at package import time."""
    return type("SkillAwareVLMPlanner", (SkillPromptMixin, base_planner_class), {})
