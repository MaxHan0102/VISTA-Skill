from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionEmphasis:
    text: str
    context: str
    evidence_ids: tuple[str, ...]
    expires_at_step: int


class ExecutionEmphasisBuffer:
    """Decaying reminders for execution lapses; never part of canonical Skill."""

    def __init__(self, *, ttl_steps: int = 6, max_items: int = 4) -> None:
        self.ttl_steps = ttl_steps
        self.max_items = max_items
        self._items: list[ExecutionEmphasis] = []

    def add(
        self,
        text: str,
        *,
        context: str,
        evidence_ids: tuple[str, ...],
        current_step: int,
    ) -> None:
        item = ExecutionEmphasis(
            text=text,
            context=context,
            evidence_ids=evidence_ids,
            expires_at_step=current_step + self.ttl_steps,
        )
        self._items = [
            existing
            for existing in self._items
            if not (existing.text == text and existing.context == context)
        ]
        self._items.append(item)
        self._items = self._items[-self.max_items :]

    def active(self, current_step: int) -> tuple[ExecutionEmphasis, ...]:
        self._items = [item for item in self._items if item.expires_at_step >= current_step]
        return tuple(self._items)

    def render(self, current_step: int) -> str:
        return "\n".join(f"- {item.text}" for item in self.active(current_step))

    def clear(self) -> None:
        self._items.clear()
