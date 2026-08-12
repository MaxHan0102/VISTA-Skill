"""VISTA-Skill-layer smoke against a live local vLLM endpoint (no simulator).

Exercises the REAL method-model code paths -- not just raw HTTP -- against the
served Qwen3-VL model:
  1. OpenAICompatibleJsonModel.complete_json (usage accounting + JSON parse)
  2. JsonTrajectoryTeacher.reflect            (baseline teacher path)
  3. JsonVisualEvidenceProvider.extract        (multimodal pre/post image path)
If these pass, the VISTA-Skill method layer can talk to vLLM correctly; the only
remaining unknown for a real experiment is the Habitat simulator (EGL/CUDA).

Usage (server already running on :8000):
    python scripts/vista_real_model_smoke.py
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

from vista_skill.baselines import EpisodeSummary
from vista_skill.models import (
    JsonTrajectoryTeacher,
    JsonVisualEvidenceProvider,
    OpenAICompatibleJsonModel,
)
from vista_skill.schemas import EvidenceRequest, ActionCall


BASE_URL = os.environ.get("VISTA_METHOD_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("VISTA_METHOD_MODEL", "Qwen/Qwen3-VL-8B-Instruct")


def _write_tiny_png(path: Path, color: tuple[int, int, int]) -> None:
    try:
        from PIL import Image

        Image.new("RGB", (16, 16), color=color).save(path, format="PNG")
    except ImportError:
        # 1x1 fallback PNG (red)
        import base64

        path.write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAE"
                "hQGAuK9dQwAAAABJRU5ErkJggg=="
            )
        )


def main() -> int:
    model = OpenAICompatibleJsonModel(
        MODEL,
        base_url=BASE_URL,
        api_key="EMPTY",
        temperature=0.0,
        max_tokens=512,
        seed=0,
    )
    failures = 0

    # 1. raw complete_json + usage accounting
    try:
        out = model.complete_json(
            system="Return JSON.",
            content=[{"type": "text", "text": '{"ask": "say ready"}'}],
            schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["reply"],
                "properties": {"reply": {"type": "string"}},
            },
            purpose="smoke_raw",
        )
        assert "reply" in out, out
        usage = model.usage.get("smoke_raw")
        assert usage and usage.calls == 1 and usage.prompt_tokens > 0, usage
        print(f"[PASS] complete_json: reply={out['reply']!r} usage={usage}")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"[FAIL] complete_json: {exc!r}")

    # 2. trajectory teacher reflection (real baseline code path)
    try:
        teacher = JsonTrajectoryTeacher(model)
        summary = EpisodeSummary(
            episode_id="smoke_ep1",
            instruction="pick the apple on the counter",
            success=False,
            trajectory=("nav apple", "pick apple"),
            current_skill="shared_embodied_execution v0",
            failure_reason="gripper was already holding another object",
        )
        reflection = teacher.reflect(summary)
        assert reflection.route.value in {"S_NEW", "S_BETTER", "FAIL_SKILL", "FAIL_EXECUTION"}
        assert reflection.content.strip()
        assert "trajectory_reflection" in model.usage
        print(f"[PASS] teacher.reflect: route={reflection.route.value} field={reflection.target_field}")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"[FAIL] teacher.reflect: {exc!r}")

    # 3. visual evidence provider (multimodal pre/post image path)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pre = Path(tmp) / "pre.png"
            post = Path(tmp) / "post.png"
            _write_tiny_png(pre, (40, 40, 40))
            _write_tiny_png(post, (200, 60, 60))
            provider = JsonVisualEvidenceProvider(model)
            request = EvidenceRequest(
                episode_id="smoke_ep1",
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
            evidence = provider.extract(request)
            assert isinstance(evidence, list)
            print(f"[PASS] evidence.extract: returned {len(evidence)} predicate observations")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"[FAIL] evidence.extract: {exc!r}")

    print(f"\nusage accounting: { {k: vars(v) for k, v in model.usage.items()} }")
    print(f"{3 - failures}/3 VISTA-layer checks passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
