"""Probe a local vLLM endpoint for the features VISTA-Skill requires.

Run AFTER the vLLM OpenAI-compatible server is up, e.g.:
    python scripts/probe_vllm_endpoint.py --base-url http://127.0.0.1:8000/v1

Validates, against the served Qwen3-VL model, the exact contract that
OpenAICompatibleJsonModel (method/teacher) and EmbodiedBench RemoteModel
(executor) depend on:
  1. /v1/models lists the expected served name
  2. plain chat completion returns content + usage
  3. same seed -> identical output (reproducibility)
  4. response_format json_schema (strict) returns parseable JSON
  5. multimodal image_url (base64 data URL) works
Each check prints PASS/FAIL; exits non-zero if any FAIL.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys

from openai import OpenAI

from vista_skill.models import (
    _attribution_schema,
    _trajectory_reflection_schema,
)

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"  # must match vLLM --served-model-name
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}: {detail}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    # 1. models list
    try:
        models = [m.id for m in client.models.list().data]
        record(
            "models_list",
            args.model in models,
            f"served models={models} (expect '{args.model}')",
        )
    except Exception as exc:  # noqa: BLE001
        record("models_list", False, f"exception: {exc!r}")
        return 1

    # 2. plain completion + usage
    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "Reply with one short sentence."},
                {"role": "user", "content": "Say hello."},
            ],
            temperature=0.0,
            max_tokens=64,
            seed=args.seed,
        )
        content = resp.choices[0].message.content
        usage = getattr(resp, "usage", None)
        ok = bool(content) and getattr(usage, "prompt_tokens", 0) > 0
        record("plain_completion", ok, f"content={content!r} usage_pt={getattr(usage,'prompt_tokens',None)}")
    except Exception as exc:  # noqa: BLE001
        record("plain_completion", False, f"exception: {exc!r}")

    # 3. seed determinism (same prompt+seed twice -> identical text)
    try:
        r1 = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            temperature=0.0,
            max_tokens=64,
            seed=args.seed,
        )
        r2 = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            temperature=0.0,
            max_tokens=64,
            seed=args.seed,
        )
        a, b = r1.choices[0].message.content, r2.choices[0].message.content
        record("seed_determinism", a == b, f"identical={a == b} (len {len(a)}/{len(b)})")
    except Exception as exc:  # noqa: BLE001
        record("seed_determinism", False, f"exception: {exc!r}")

    # 4. structured output: json_schema strict (attribution contract)
    try:
        schema = _attribution_schema()
        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "Return JSON matching the schema."},
                {
                    "role": "user",
                    "content": "An action failed to achieve its expected effect. "
                    "Attribute it as skill_update to the procedure field.",
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "vista_attribution", "strict": True, "schema": schema},
            },
            temperature=0.0,
            max_tokens=256,
            seed=args.seed,
        )
        raw = resp.choices[0].message.content
        parsed = json.loads(raw)
        required = set(schema["required"])
        ok = required.issubset(parsed.keys()) and parsed.get("target") in {
            "belief_refresh",
            "skill_update",
            "abstain",
        }
        record("json_schema_strict", ok, f"target={parsed.get('target')} keys={sorted(parsed.keys())}")
    except Exception as exc:  # noqa: BLE001
        record("json_schema_strict", False, f"exception: {exc!r}")

    # 5. trajectory reflection schema (used by the baseline teacher)
    try:
        schema = _trajectory_reflection_schema()
        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "Return JSON matching the schema."},
                {
                    "role": "user",
                    "content": "A pick failed because the gripper was already holding "
                    "another object. Reflect on this failed trajectory.",
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "trajectory_reflection", "strict": True, "schema": schema},
            },
            temperature=0.0,
            max_tokens=256,
            seed=args.seed,
        )
        parsed = json.loads(resp.choices[0].message.content)
        ok = parsed.get("route") in {
            "S_NEW",
            "S_BETTER",
            "FAIL_SKILL",
            "FAIL_EXECUTION",
        }
        record("trajectory_schema", ok, f"route={parsed.get('route')} field={parsed.get('target_field')}")
    except Exception as exc:  # noqa: BLE001
        record("trajectory_schema", False, f"exception: {exc!r}")

    # 6. multimodal image_url (base64 data URL) -- executor & evidence provider path
    try:
        png = _tiny_png()
        data_url = f"data:image/png;base64,{base64.b64encode(png).decode()}"
        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": "Describe this image in one sentence."},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=64,
            seed=args.seed,
        )
        content = resp.choices[0].message.content
        record("multimodal_image", bool(content), f"content={content!r}")
    except Exception as exc:  # noqa: BLE001
        record("multimodal_image", False, f"exception: {exc!r}")

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{passed}/{len(results)} checks passed", flush=True)
    return 0 if passed == len(results) else 2


def _tiny_png() -> bytes:
    """Return a minimal valid 8x8 PNG without third-party deps."""
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (8, 8), color=(123, 45, 67)).save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        # Fallback: a tiny pre-baked 1x1 PNG (red pixel).
        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAE"
            "hQGAuK9dQwAAAABJRU5ErkJggg=="
        )


if __name__ == "__main__":
    sys.exit(main())
