"""Stock EmbodiedBench evaluation with non-invasive feedback/API adapters.

Default settings, planning, parsing, scoring and artifact formats come from the
checked-in benchmark. No Skill runtime or independent rollout loop is used.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import inspect
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "EmbodiedBench"
sys.path.insert(0, str(REPO))
SETS = {
    "eb-hab": ("base", "common_sense", "complex_instruction", "spatial_relationship", "visual_appearance", "long_horizon"),
    "eb-nav": ("base", "common_sense", "complex_instruction", "visual_appearance", "long_horizon"),
}
ENDPOINTS = {"openai": ("OPENAI_API_KEY", "https://api.openlux.ai/v1"),
             "gemini": ("GEMINI_API_KEY", "https://api.openlux.ai/v1"),
             "qwen": ("QWEN_API_KEY", None)}
SCHEMA_COMPATIBILITY = "closed_objects_v1"


def jsonable(value):
    if isinstance(value, type) and hasattr(value, "model_json_schema"):
        return {"schema_class": value.__name__, "schema": value.model_json_schema()}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=jsonable) + "\n")


def append_json(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=jsonable) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def allocate_output(requested, timestamp, *, create):
    candidate, attempt = requested, 0
    while True:
        if create:
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            except FileExistsError:
                pass
        elif not candidate.exists() and not candidate.is_symlink():
            return candidate
        suffix = f"_{timestamp}" + (f"_{attempt:03d}" if attempt else "")
        candidate = requested.with_name(requested.name + suffix)
        attempt += 1


def adapted_function(original, replacements):
    """Apply checked, minimal in-memory edits; fail closed on upstream drift."""
    source = textwrap.dedent(inspect.getsource(original))
    for before, after, count in replacements:
        if source.count(before) != count:
            raise RuntimeError(f"stock source changed: cannot adapt {original.__qualname__}")
        source = source.replace(before, after)
    namespace = dict(original.__globals__)
    exec(compile(source, f"<feedback-adapter:{original.__qualname__}>", "exec"), namespace)
    return namespace[original.__name__]


def feedback_evaluate(original, env_name):
    if env_name == "eb-hab":
        edits = [("if done or info['last_action_success'] == 0:", "if done:", 1)]
    else:
        edits = [("if info['last_action_success'] == 0:", "if False:  # outcome hidden from policy control flow", 1)]
    return adapted_function(original, edits), edits


def feedback_planner(stock, env_name):
    class FeedbackPlanner(stock):
        def update_info(self, info):
            # Keep only the attempted action. Raw info remains untouched for the
            # stock evaluator, metrics and env.save_episode_log().
            return super().update_info({"action_id": info["action_id"], "env_feedback": ""})

    if env_name == "eb-hab":
        def init(self, *args, **kwargs):
            kwargs["use_feedback"] = False
            stock.__init__(self, *args, **kwargs)
        FeedbackPlanner.__init__ = init
    else:
        # NAV has no use_feedback switch. Remove only the two history feedback
        # interpolations; keep all official instructions and action history.
        FeedbackPlanner.process_prompt = adapted_function(stock.process_prompt, [
            (", env feedback: {}", "", 2),
            (", action_feedback[1])", ")", 2),
        ])
    return FeedbackPlanner


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        # Libraries such as colorama can retain this stream until atexit,
        # after main() has already closed terminal.log.
        if not getattr(self.stream, "closed", False):
            self.stream.write(text)
        if not self.log.closed:
            self.log.write(text)
            self.log.flush()
        return len(text)

    def flush(self):
        if not getattr(self.stream, "closed", False):
            self.stream.flush()
        if not self.log.closed:
            self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class FatalAPIError(BaseException):
    """Do not let stock retry loops retry invalid credentials forever."""

    def __init__(self, message, status_code=None, provider_error=None):
        super().__init__(message)
        self.status_code = status_code
        self.provider_error = provider_error or {}


def permanent_schema_error(details):
    message = str(details.get("message", "")).lower()
    return (details.get("code") == "invalid_json_schema"
            or "invalid schema" in message or "invalid json schema" in message
            or ("response_schema" in message and "unknown name" in message))


def inline_schema_refs(schema):
    """Expand local references without dropping field constraints or definitions."""
    def expand(node, visiting=()):
        if isinstance(node, list):
            return [expand(value, visiting) for value in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            ref = node["$ref"]
            if not ref.startswith("#/") or ref in visiting:
                raise ValueError(f"Unsupported recursive/external schema reference: {ref}")
            target = schema
            for part in ref[2:].split("/"):
                target = target[part.replace("~1", "/").replace("~0", "~")]
            return {**expand(target, (*visiting, ref)),
                    **expand({k: v for k, v in node.items() if k != "$ref"}, visiting)}
        return {key: expand(value, visiting) for key, value in node.items() if key not in {"$defs", "definitions"}}
    return expand(schema)


def schema_revision(model):
    return "gemini_inline_refs_v1" if model in {"gemini-3-flash-preview", "gemini-3.5-flash"} else SCHEMA_COMPATIBILITY


def compatible_request(kwargs):
    """Close JSON-schema objects on a copy; never mutate stock schema globals."""
    response_format = kwargs.get("response_format")
    if schema_revision(kwargs.get("model")) == "gemini_inline_refs_v1" and isinstance(response_format, type) and hasattr(response_format, "model_json_schema"):
        original = response_format
        # Keep the SDK's parse path and original Pydantic validation. Only the
        # generated wire schema changes; all fields and constraints are retained.
        def model_json_schema(cls, *a, **kw):
            return inline_schema_refs(original.model_json_schema(*a, **kw))
        adapted = type(original.__name__, (original,), {"model_json_schema": classmethod(model_json_schema)})
        return {**kwargs, "response_format": adapted}
    if not isinstance(response_format, dict) or response_format.get("type") != "json_schema":
        return kwargs  # Pydantic parse() schemas stay with the SDK converter.
    definition = response_format.get("json_schema", {})
    if not isinstance(definition, dict) or not isinstance(definition.get("schema"), dict):
        return kwargs
    response_format = deepcopy(response_format)

    def close_objects(node):
        if not isinstance(node, dict):
            return
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
        for key in ("properties", "$defs", "definitions", "patternProperties"):
            for child in node.get(key, {}).values():
                close_objects(child)
        for key in ("items", "anyOf", "allOf", "oneOf", "prefixItems"):
            child = node.get(key)
            for item in child if isinstance(child, list) else [child]:
                close_objects(item)

    close_objects(response_format["json_schema"]["schema"])
    return {**kwargs, "response_format": response_format}


def api_error_details(exc):
    """Keep structured provider diagnostics, not raw requests or credentials."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        body = body["error"]
    details = {}
    if isinstance(body, dict):
        for field in ("message", "type", "code", "param"):
            value = body.get(field)
            if not isinstance(value, (str, int, float)):
                continue
            text = str(value)
            for name, secret in os.environ.items():
                if secret and len(secret) >= 8 and any(marker in name.upper() for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                    text = text.replace(secret, "[REDACTED]")
            text = re.sub(r"\b(?:sk-|ghp_)[A-Za-z0-9_-]+", "[REDACTED]", text)
            text = re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", text)
            text = re.sub(r"data:image/[^\s\"']+", "[IMAGE OMITTED]", text)
            details[field] = text[:2000]
    return details


class AuditedClient:
    """Audit the SDK boundary with a minimal object-schema compatibility fix."""
    def __init__(self, client, output, *, smoke=False, api_retries=0, retry_delay=5):
        self.client, self.output, self.smoke = client, output, smoke
        self.api_retries, self.retry_delay = api_retries, retry_delay
        self.calls = 0
        completions = SimpleNamespace(create=self.create, parse=self.parse)
        self.chat = SimpleNamespace(completions=completions)
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    def pack(self, value):
        if isinstance(value, str) and value.startswith("data:image/"):
            header, encoded = value.split(",", 1)
            raw = base64.b64decode(encoded)
            sha = hashlib.sha256(raw).hexdigest()
            folder = self.output / "input_images"
            folder.mkdir(exist_ok=True)
            path = folder / sha
            if not path.exists():
                path.write_bytes(raw)
            return {"data_url_header": header, "path": str(path), "sha256": sha}
        if isinstance(value, list):
            return [self.pack(x) for x in value]
        if isinstance(value, dict):
            return {k: self.pack(v) for k, v in value.items()}
        return value

    def call(self, method, kwargs):
        kwargs = compatible_request(kwargs)
        for attempt in range(self.api_retries + 1):
            try:
                return self._call_once(method, kwargs)
            except (Exception, FatalAPIError) as exc:
                status = getattr(exc, "status_code", None)
                transient = status in {400, 408, 409, 429} or (status is not None and status >= 500)
                transient = transient or type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}
                if permanent_schema_error(getattr(exc, "provider_error", None) or api_error_details(exc)):
                    transient = False
                if not self.api_retries or not transient:
                    raise
                if attempt == self.api_retries:
                    raise FatalAPIError(f"API retries exhausted ({type(exc).__name__}, HTTP {status})", status,
                                        getattr(exc, "provider_error", None) or api_error_details(exc)) from None
                delay = min(self.retry_delay * 2 ** attempt, 30)
                print(f"API retry {attempt + 1}/{self.api_retries} in {delay}s (HTTP {status}); request unchanged", flush=True)
                time.sleep(delay)

    def _call_once(self, method, kwargs):
        self.calls += 1
        append_json(self.output / "api_requests.jsonl", {"call": self.calls, "method": method,
                    "smoke_policy": self.smoke, "schema_compatibility": schema_revision(kwargs.get("model")),
                    "request": self.pack(kwargs)})
        if self.smoke:
            # Two-action plans exercise the official batch loop, not an imposed
            # single-action policy. This is only a simulator plumbing check.
            payload = {"visual_state_description": "smoke", "reasoning_and_reflection": "fixed policy",
                       "language_plan": "execute action 0 twice", "executable_plan": [
                           {"action_id": 0, "action_name": "smoke"}, {"action_id": 0, "action_name": "smoke"}]}
            response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps(payload), parsed=SimpleNamespace(model_dump_json=lambda: json.dumps(payload))))],
                usage=SimpleNamespace(prompt_tokens=0))
            append_json(self.output / "api_responses.jsonl", {"call": self.calls, "smoke_policy": True, "response": payload})
            return response
        try:
            target = self.client.chat.completions if method == "create" else self.client.beta.chat.completions
            response = getattr(target, method)(**kwargs)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            details = api_error_details(exc)
            append_json(self.output / "api_errors.jsonl", {"call": self.calls,
                        "error_type": type(exc).__name__, "http_status": status, "provider_error": details})
            if status in {400, 401, 403, 404, 422}:
                detail = details.get("message") or "check gateway key, model ID, request compatibility and permissions"
                raise FatalAPIError(f"API HTTP {status}: {detail}", status, details) from None
            raise
        append_json(self.output / "api_responses.jsonl", {"call": self.calls, "response": response})
        return response

    def create(self, **kwargs):
        return self.call("create", kwargs)

    def parse(self, **kwargs):
        return self.call("parse", kwargs)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=SETS, default="eb-hab")
    parser.add_argument("--provider", choices=ENDPOINTS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--track", choices=("rgb_only", "full"), default="rgb_only")
    parser.add_argument("--eval-sets", nargs="+", default=["base"])
    parser.add_argument("--episodes", type=int, default=2, help="per split; 0 = all remaining")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="process/environment seed, once per split")
    parser.add_argument("--frames", type=int, default=1, help="stock single frame by default; explicit temporal ablation")
    parser.add_argument("--n-shots", type=int, help="otherwise stock YAML default")
    parser.add_argument("--resolution", type=int, help="otherwise stock YAML default")
    parser.add_argument("--max-steps", type=int, help="explicit smoke budget; otherwise stock limit")
    parser.add_argument("--max-tokens", type=int, help="otherwise stock RemoteModel default")
    parser.add_argument("--temperature", type=float, help="override stock global; stock GPT-5 omits temperature")
    parser.add_argument("--api-retries", type=int, default=0, help="bounded retries of the identical request; resume launcher enables this")
    parser.add_argument("--retry-delay", type=float, default=5)
    parser.add_argument("--expected-episodes", type=Path, help="resume-only episode identity check before model calls")
    parser.add_argument("--artifact-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--resume-existing", action="store_true", help="fill missing tasks in an existing output directory")
    parser.add_argument("--api-key-env")
    parser.add_argument("--base-url", default=None, help="default https://api.openlux.ai/v1")
    parser.add_argument("--output", type=Path, required=True, help="under running/; timestamp yyyymmddhhmm on collision")
    parser.add_argument("--log-level", choices=("INFO", "DEBUG"), default="INFO")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-policy", action="store_true", help="no API; fixed two-action plans")
    args = parser.parse_args(argv)
    if args.artifact_root:
        args.artifact_root = args.artifact_root.resolve()
        if REPO / "running" not in args.artifact_root.parents or not (args.artifact_root / "config.json").is_file():
            parser.error("artifact root must be an existing registered run under running/")
        if args.resume_existing:
            parser.error("worker artifact root cannot be combined with --resume-existing")
    args.api_key_env = args.api_key_env or ENDPOINTS[args.provider][0]
    args.base_url = args.base_url or ENDPOINTS[args.provider][1]
    if not args.base_url:
        parser.error("--base-url is required for the Qwen inference server")
    from urllib.parse import urlsplit
    url = urlsplit(args.base_url)
    if url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password or url.query or url.fragment:
        parser.error("base URL must not contain credentials, a query or fragment")
    if args.eval_sets == ["all"]:
        args.eval_sets = list(SETS[args.env])
    if len(set(args.eval_sets)) != len(args.eval_sets) or any(s not in SETS[args.env] for s in args.eval_sets):
        parser.error("invalid or duplicate evaluation split")
    if args.episodes < 0 or args.start_index < 0 or args.frames < 1 or args.api_retries < 0 or args.retry_delay < 0 or any(
        x is not None and x < 1 for x in (args.max_steps, args.max_tokens, args.resolution)
    ) or (args.n_shots is not None and args.n_shots < 0):
        parser.error("invalid numeric budget")
    if args.env == "eb-nav" and args.frames not in {1, 3}:
        parser.error("stock NAV supports 1 or 3 frames")
    if (args.provider == "openai" and "gpt" not in args.model) or (args.provider == "gemini" and "gemini" not in args.model):
        parser.error("model ID must match the stock GPT/Gemini routing for the selected provider")
    if args.provider == "qwen" and ("Qwen3-VL" not in args.model or "8B-Instruct" not in args.model):
        parser.error("Qwen provider requires a Qwen3-VL-8B-Instruct served model ID")
    args.output = args.output.resolve()
    if REPO / "running" not in args.output.parents:
        parser.error("output must be a subdirectory of repository running/")
    return args


def official_config(args):
    import yaml
    config = yaml.safe_load((BENCH / f"embodiedbench/configs/{args.env}.yaml").read_text())
    config.update(model_name=args.model, eval_sets=args.eval_sets, exp_name="feedback_comparison")
    for name in ("n_shots", "resolution"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    if args.frames != 1:
        config["multistep"] = args.frames if args.env == "eb-hab" else True
    if args.env == "eb-hab":
        config["env_feedback"] = args.track == "full"
        config["start_epi_index"] = args.start_index
    return config


def run_official(args, config, metadata):
    from vista_skill.integrations.embodiedbench.environment import seed_process_rngs, seed_habitat_env, seed_nav_env
    sys.path.insert(0, str(BENCH))
    module_name = "eb_habitat_evaluator" if args.env == "eb-hab" else "eb_navigation_evaluator"
    module = importlib.import_module(f"embodiedbench.evaluator.{module_name}")
    remote = importlib.import_module("embodiedbench.planner.remote_model")
    evaluator_class = module.EB_HabitatEvaluator if args.env == "eb-hab" else module.EB_NavigationEvaluator
    env_symbol, planner_symbol = ("EBHabEnv", "VLMPlanner") if args.env == "eb-hab" else ("EBNavigationEnv", "EBNavigationPlanner")
    stock_env, stock_planner = getattr(module, env_symbol), getattr(module, planner_symbol)
    audit = args.output / "audit"
    audit.mkdir()
    client = None
    if not args.smoke_policy:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get(args.api_key_env) or ("EMPTY" if args.provider == "qwen" else None), base_url=args.base_url)
    audited = AuditedClient(client, audit, smoke=args.smoke_policy,
                            api_retries=args.api_retries, retry_delay=args.retry_delay)
    expected_episodes = json.loads(args.expected_episodes.read_text()) if args.expected_episodes else {}
    metadata["source_hashes"] = {str(Path(p).relative_to(REPO)): digest(p) for p in (
        inspect.getfile(stock_env), inspect.getfile(stock_planner), module.__file__, remote.__file__,
        str(BENCH / f"embodiedbench/configs/{args.env}.yaml"))}
    metadata["runtime_changes"] = []
    artifact_root = args.artifact_root or args.output
    evaluator = evaluator_class(config)
    if args.track == "rgb_only":
        method, edits = feedback_evaluate(evaluator_class.evaluate, args.env)
        evaluator.evaluate = method.__get__(evaluator)
        metadata["runtime_changes"] = edits
    if args.artifact_root:
        from scripts.resume_closed_source_wo_feedback import compatible
        compatible(json.loads((artifact_root / "config.json").read_text()), metadata)
        # Keep native episode files at the canonical location. Save per-session
        # config.txt separately so it cannot overwrite the original full config.
        edits = [("os.path.join(self.env.log_path, 'config.txt')",
                  f"os.path.join({str(args.output)!r}, self.eval_set, 'config.txt')", 1)]
        evaluator.evaluate_main = adapted_function(evaluator_class.evaluate_main, edits).__get__(evaluator)
        evaluator.save_episode_metric = adapted_function(evaluator_class.save_episode_metric, [
            ("'w', encoding='utf-8'", "'x', encoding='utf-8'", 1),
        ]).__get__(evaluator)
        metadata["storage_runtime_changes"] = edits + [("result open mode", "exclusive creation")]
    planner_class = feedback_planner(stock_planner, args.env) if args.track == "rgb_only" else stock_planner

    def env_factory(**kwargs):
        split = kwargs["eval_set"]
        seed_process_rngs(args.seed)
        if args.env == "eb-nav":
            dataset = BENCH / f"embodiedbench/envs/eb_navigation/datasets/{split}.json"
            total = len(json.loads(dataset.read_text())["tasks"])
            end = total if not args.episodes else args.start_index + args.episodes
            if not 0 <= args.start_index < end <= total:
                raise ValueError("requested episode range exceeds dataset")
            kwargs["selected_indexes"] = list(range(args.start_index, end))
        else:
            dataset = BENCH / f"embodiedbench/envs/eb_habitat/datasets/{split}.pickle"
        env = stock_env(**kwargs)
        try:
            if args.env == "eb-hab":
                total = int(env.number_of_episodes)
                end = total if not args.episodes else args.start_index + args.episodes
                if not 0 <= args.start_index < end <= total:
                    raise ValueError("requested episode range exceeds dataset")
                # Preserve the official iterator, order and start-index behavior.
                env.number_of_episodes = end
                seed_habitat_env(env, args.seed)
            else:
                seed_nav_env(env, args.seed)
            env.log_path = str(artifact_root / split)
            Path(env.log_path).mkdir(exist_ok=bool(args.artifact_root))
            if args.max_steps:
                env._max_episode_steps = min(env._max_episode_steps, args.max_steps)
            (args.output / split).mkdir(exist_ok=True)
            write_json(args.output / split / "selection.json", {"start_index": args.start_index,
                       "episodes": end - args.start_index, "seed": args.seed,
                       "dataset_sha256": digest(dataset), "order": "stock dataset/iterator"})
            if args.artifact_root:
                registered = json.loads((artifact_root / "config.json").read_text())
                selection = Path(env.log_path) / "selection.json"
                if not selection.exists():
                    write_json(selection, {"start_index": registered["start_index"],
                        "episodes": registered["episodes"] or total - registered["start_index"],
                        "seed": args.seed, "dataset_sha256": digest(dataset), "order": "stock dataset/iterator"})
                canonical_config = Path(env.log_path) / "config.txt"
                if not canonical_config.exists():
                    canonical_config.write_text(str(registered["official_config"]))
            # Read-only observation of resets for real IDs; no reordering or reseeding.
            original_reset = env.reset
            def reset(*a, **kw):
                if args.artifact_root:
                    next_ordinal = (env._current_episode_num + 1 if args.env == "eb-hab"
                                    else env.selected_indexes[env._current_episode_num] + 1)
                    result = Path(env.log_path) / "results" / f"episode_{next_ordinal}_final_res.json"
                    if result.exists():
                        raise FatalAPIError(f"Result already exists: {result}; restart controller to rescan and skip")
                obs = original_reset(*a, **kw)
                active = env.current_episode() if args.env == "eb-hab" else f"nav_{env.selected_indexes[env._current_episode_num - 1]}"
                episode_id = getattr(active, "episode_id", active)
                ordinal = env._current_episode_num if args.env == "eb-hab" else env.selected_indexes[env._current_episode_num - 1] + 1
                expected_id = expected_episodes.get(split, {}).get(str(ordinal))
                if expected_id is not None and str(episode_id) != expected_id:
                    raise FatalAPIError(f"Resume episode identity mismatch at {split}/{ordinal}: {episode_id} != {expected_id}")
                append_json(audit / "episodes.jsonl", {"split": split, "episode_number": env._current_episode_num,
                            "episode_id": str(episode_id), "instruction": env.episode_language_instruction,
                            "api_call_start": audited.calls + 1})
                return obs
            env.reset = reset
            return env
        except BaseException:
            env.close()
            raise

    logger = module.logger
    old_level = logger.level
    handlers = [(h, h.stream) for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    logger.setLevel(getattr(logging, args.log_level))
    for handler, _ in handlers:
        handler.setStream(sys.stderr)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(module, env_symbol, env_factory))
            stack.enter_context(patch.object(module, planner_symbol, planner_class))
            stack.enter_context(patch.object(remote, "OpenAI", lambda **kw: audited))
            if args.artifact_root:
                stock_average = module.average_json_values
                def average_results(*a, **kw):
                    # Stock glob '*.json' would include an older summary in the
                    # next average. Only final episode files are observations.
                    kw["target_file"] = "episode_*_final_res.json"
                    return stock_average(*a, **kw)
                stack.enter_context(patch.object(module, "average_json_values", average_results))
                # adapted_function snapshots globals. Rebind its three patched
                # dependencies too, or it would instantiate the unwrapped env.
                evaluator.evaluate_main.__func__.__globals__.update({
                    env_symbol: env_factory, planner_symbol: planner_class,
                    "average_json_values": average_results,
                })
            for arg, symbol in ((args.max_tokens, "max_completion_tokens"), (args.temperature, "temperature")):
                if arg is not None:
                    stack.enter_context(patch.object(remote, symbol, arg))
            metadata["remote_defaults"] = {"temperature": remote.temperature, "max_completion_tokens": remote.max_completion_tokens,
                                           "gpt5_temperature": "omitted by stock RemoteModel"}
            write_json(args.output / "config.json", metadata)
            evaluator.check_config_valid()
            evaluator.evaluate_main()
    finally:
        if evaluator.env is not None:
            evaluator.env.close()
        if client is not None:
            client.close()
        for handler, stream in handlers:
            handler.setStream(stream)
        logger.setLevel(old_level)
    return audited.calls


def main(argv=None):
    args = arguments(argv)
    config = official_config(args)
    started = datetime.now(timezone(timedelta(hours=8)))
    requested = args.output
    stamp = started.strftime("%Y%m%d%H%M")
    if args.resume_existing and (requested / "config.json").exists():
        from scripts.resume_closed_source_wo_feedback import compatible, resume
        saved = json.loads((requested / "config.json").read_text())
        compatible(saved, {**vars(args), "protocol": "stock_feedback_v2", "official_config": config})
        if any(saved[k] != vars(args)[k] for k in ("eval_sets", "episodes", "start_index")):
            raise SystemExit("Existing run has different task ranges; use its resume launcher or a new output")
        resume(requested, SimpleNamespace(dry_run=args.dry_run, api_retries=args.api_retries or 3,
                                         max_retries=3, retry_delay=30))
        return 0
    args.output = allocate_output(requested, stamp, create=False)
    metadata = {**vars(args), "output": str(args.output), "requested_output": str(requested),
                "started_at": started.isoformat(), "protocol": "stock_feedback_v2", "official_config": config,
                "schema_compatibility": schema_revision(args.model),
                "script_sha256": digest(__file__), "skill_or_memory": False}
    if args.dry_run:
        print(json.dumps(metadata, indent=2, default=jsonable))
        return 0
    if not args.smoke_policy and args.provider != "qwen" and not os.environ.get(args.api_key_env):
        raise SystemExit(f"Missing {args.api_key_env}")
    args.output = allocate_output(requested, stamp, create=True)
    metadata["output"] = str(args.output)
    metadata["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    metadata["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True))
    write_json(args.output / "config.json", metadata)
    old_cwd = Path.cwd()
    try:
        os.chdir(BENCH)
        with (args.output / "terminal.log").open("w") as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                print(f"Output directory: {args.output}", flush=True)
                calls = run_official(args, config, metadata)
        categories = {}
        for split in args.eval_sets:
            expected = json.loads((args.output / split / "selection.json").read_text())["episodes"]
            folder = (args.artifact_root or args.output) / split / "results"
            paths = [folder / f"episode_{i}_final_res.json" for i in range(args.start_index + 1, args.start_index + expected + 1)]
            rows = [json.loads(p.read_text()) for p in paths if p.exists()]
            if len(rows) != expected:
                raise RuntimeError("official result count differs from requested task count")
            categories[split] = {"episodes": len(rows), "task_success": sum(r["task_success"] for r in rows) / len(rows)}
        write_json(args.output / "summary.json", {"status": "complete", "protocol": "stock_feedback_v2",
                   "smoke_policy": args.smoke_policy, "api_calls": 0 if args.smoke_policy else calls,
                   "all_categories": set(args.eval_sets) == set(SETS[args.env]),
                   "categories": categories, "category_mean_success": sum(r["task_success"] for r in categories.values()) / len(categories)})
    except BaseException as exc:
        write_json(args.output / "failure.json", {"status": "incomplete", "error_type": type(exc).__name__,
                   "http_status": getattr(exc, "status_code", None),
                   "provider_error": getattr(exc, "provider_error", None) or api_error_details(exc)})
        if isinstance(exc, FatalAPIError):
            print(str(exc), file=sys.stderr)
            with (args.output / "terminal.log").open("a") as log:
                log.write(str(exc) + "\n")
        else:
            print(f"Run incomplete ({type(exc).__name__}); see terminal.log and audit/", file=sys.stderr)
        return 1
    finally:
        os.chdir(old_cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
