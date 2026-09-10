import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluate_closed_loop_feedback import (
    BENCH, AuditedClient, FatalAPIError, Tee, adapted_function, allocate_output,
    arguments, feedback_evaluate, feedback_planner, official_config,
)


def stock_method(file, class_name, method, extra=None):
    """Load the real checked-in method without importing simulator dependencies."""
    path = BENCH / file
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    namespace = {"json": json, **(extra or {})}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method]


def cli(*extra):
    return arguments(["--provider", "openai", "--model", "gpt-5.4-mini",
                      "--output", "running/test/stock_feedback_test_unused", *extra])


def test_official_defaults_are_loaded_not_reinvented():
    import yaml
    for env in ("eb-hab", "eb-nav"):
        args = cli("--env", env, "--track", "full")
        stock = yaml.safe_load((BENCH / f"embodiedbench/configs/{env}.yaml").read_text())
        config = official_config(args)
        for key in stock.keys() - {"model_name", "eval_sets", "exp_name"}:
            assert config[key] == stock[key]
        assert args.frames == 1
    assert cli().base_url == "https://api.openlux.ai/v1"
    assert official_config(cli())["env_feedback"] is False
    assert official_config(cli("--frames", "3"))["multistep"] == 3


class Progress:
    def __init__(self, **kwargs):
        pass

    def update(self):
        pass


class FakeEnv:
    number_of_episodes = 1
    _current_episode_num = 0
    _max_episode_steps = 30
    _max_invalid_actions = 10
    _cur_invalid_actions = 0
    language_skill_set = ["nav", "pick"]
    episode_language_instruction = "Move the object"
    _episode_start_time = 0

    def reset(self):
        self._current_episode_num += 1
        self._current_step = 0
        self.episode_log = []

    def save_image(self, obs):
        return "image.png"

    def step(self, action, reasoning, *args):
        self._current_step += 1
        info = {"last_action_success": int(self._current_step != 1), "env_feedback": "SECRET",
                "task_success": int(self._current_step == 2), "task_progress": 0.5,
                "subgoal_reward": 0, "env_step": self._current_step, "episode_elapsed_seconds": 1,
                "action_id": action}
        self.episode_log.append(dict(info))
        return None, 0, self._current_step == 2, info

    def save_episode_log(self):
        self.saved = True


@pytest.mark.parametrize("env_name,class_name", [("eb-hab", "EB_HabitatEvaluator"), ("eb-nav", "EB_NavigationEvaluator")])
def test_only_hidden_failure_replan_changes_in_official_loop(env_name, class_name):
    name = "eb_habitat_evaluator" if env_name == "eb-hab" else "eb_navigation_evaluator"
    original = stock_method(f"embodiedbench/evaluator/{name}.py", class_name, "evaluate", {
        "tqdm": Progress, "logger": SimpleNamespace(info=lambda *a: None, debug=lambda *a: None),
        "np": SimpleNamespace(mean=lambda x: sum(x) / len(x)),
        "time": SimpleNamespace(time=lambda: 1),
    })
    strict, edits = feedback_evaluate(original, env_name)
    assert len(edits) == 1
    for evaluate, expected_calls in ((original, 2), (strict, 1)):
        env = FakeEnv()
        class Planner:
            output_json_error = 0
            planner_steps = 0

            def reset(self):
                pass

            def act(self, *args):
                self.planner_steps += 1
                return [0, 1], '{"executable_plan":[{"action_id":0},{"action_id":1}]}'

            def update_info(self, info):
                pass
        planner = Planner()
        rows = []
        evaluate(SimpleNamespace(env=env, planner=planner, save_episode_metric=rows.append))
        assert planner.planner_steps == expected_calls
        assert rows[0]["task_success"] == 1
        assert rows[0]["num_steps"] == 2
        assert env.episode_log[0]["last_action_success"] == 0
        if env_name == "eb-hab":
            assert rows[0]["num_invalid_actions"] == 1 and env.saved


@pytest.mark.parametrize("env_name", ["eb-hab", "eb-nav"])
def test_official_prompt_history_is_sanitized_without_losing_actions(env_name):
    nav = env_name == "eb-nav"
    file = "nav_planner.py" if nav else "vlm_planner.py"
    cls = "EBNavigationPlanner" if nav else "VLMPlanner"
    process = stock_method(f"embodiedbench/planner/{file}", cls, "process_prompt")
    class Stock:
        process_prompt = process
        def __init__(self, **kwargs):
            self.use_feedback = kwargs.get("use_feedback", True)
            self.episode_act_feedback = []
            self.n_shot = 0
            self.chat_history = False
            self.language_only = False
            self.actions = ["nav", "pick"]
            self.available_action_str = "nav,pick"
            self.system_prompt = "{} {} {}"
            self.following_prompt = "stock following prompt"

        def update_info(self, info):
            self.episode_act_feedback.append([info["action_id"], info["env_feedback"]])
    planner = feedback_planner(Stock, env_name)()
    info = {"action_id": 1, "env_feedback": "SECRET", "last_action_success": 0}
    planner.update_info(info)
    prompt = planner.process_prompt("instruction", planner.episode_act_feedback)
    assert "SECRET" not in prompt and "env feedback:" not in prompt
    assert "pick" in prompt
    assert info["env_feedback"] == "SECRET"


def test_source_drift_fails_closed():
    with pytest.raises(RuntimeError, match="stock source changed"):
        adapted_function(allocate_output, [("missing pattern", "replacement", 1)])


def test_api_wrapper_preserves_request_options_and_response(tmp_path):
    seen = []
    response = SimpleNamespace(model_dump=lambda: {"model": "gpt-5.4-mini"})
    def create(**kwargs):
        seen.append(kwargs)
        return response
    client = AuditedClient(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), tmp_path)
    kwargs = {"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": "prompt"}],
              "response_format": {"type": "json_schema", "json_schema": {"name": "stock"}}, "max_completion_tokens": 4096}
    assert client.create(**kwargs) is response
    assert seen == [kwargs]
    assert json.loads((tmp_path / "api_requests.jsonl").read_text())["request"] == kwargs


def test_fatal_credentials_do_not_enter_official_infinite_retry(tmp_path):
    class AuthError(Exception):
        status_code = 401
    def fail(**kwargs):
        raise AuthError("sensitive details")
    client = AuditedClient(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail))), tmp_path)
    with pytest.raises(FatalAPIError):
        client.create(model="m")
    assert "sensitive" not in (tmp_path / "api_errors.jsonl").read_text()


def test_output_collision_preserves_old_run(tmp_path):
    requested = tmp_path / "pilot"
    stamp = "202609081630"
    assert allocate_output(requested, stamp, create=True) == requested
    marker = requested / "old_result.json"
    marker.write_text("old result")
    preview = allocate_output(requested, stamp, create=False)
    assert preview.name == f"pilot_{stamp}" and not preview.exists()
    assert allocate_output(requested, stamp, create=True) == preview
    assert allocate_output(requested, stamp, create=True).name == f"pilot_{stamp}_001"
    assert marker.read_text() == "old result"


def test_retained_tee_can_write_and_flush_after_log_closes():
    console, log = io.StringIO(), io.StringIO()
    tee = Tee(console, log)
    tee.write("during run\n")
    log.close()
    assert tee.write("atexit reset") == len("atexit reset")
    tee.flush()
    assert console.getvalue() == "during run\natexit reset"
    console.close()
    tee.write("late cleanup")
    tee.flush()


def test_bad_request_records_redacted_provider_reason_and_status(tmp_path, monkeypatch):
    key = "gateway-private-value-123"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    class BadRequest(Exception):
        status_code = 400
        body = {"error": {"message": f"Invalid image; credential {key}", "code": "invalid_image"}}
    def fail(**kwargs):
        raise BadRequest()
    client = AuditedClient(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail))), tmp_path)
    with pytest.raises(FatalAPIError, match="Invalid image") as captured:
        client.create(model="gpt-5.4-mini")
    assert captured.value.status_code == 400
    raw = (tmp_path / "api_errors.jsonl").read_text()
    assert key not in raw and key not in str(captured.value)
    assert json.loads(raw)["provider_error"]["code"] == "invalid_image"


def test_stock_schema_closes_nested_objects_without_mutating_official_definition(tmp_path):
    from copy import deepcopy
    import runpy
    from scripts.evaluate_closed_loop_feedback import BENCH, SCHEMA_COMPATIBILITY
    schema = runpy.run_path(str(BENCH / "embodiedbench/planner/planner_config/generation_guide.py"))["vlm_generation_guide"]
    original = deepcopy(schema)
    seen = []
    def create(**kwargs):
        seen.append(kwargs)
        actual = kwargs['response_format']['json_schema']['schema']
        assert actual['additionalProperties'] is False
        assert actual['properties']['executable_plan']['items']['additionalProperties'] is False
        return SimpleNamespace(model_dump=lambda: {})
    client = AuditedClient(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), tmp_path)
    kwargs = {'model': 'gpt-5.4-mini', 'messages': ['unchanged prompt'], 'max_completion_tokens': 4096,
              'response_format': {'type': 'json_schema', 'json_schema': {'name': 'embodied_planning', 'schema': schema}}}
    client.create(**kwargs)
    assert schema == original
    actual = seen[0]['response_format']['json_schema']['schema']
    del actual['additionalProperties']
    del actual['properties']['executable_plan']['items']['additionalProperties']
    assert seen[0] == kwargs  # Only these two schema constraints changed.
    audit = json.loads((tmp_path / 'api_requests.jsonl').read_text())
    assert audit['schema_compatibility'] == SCHEMA_COMPATIBILITY
    assert audit['request']['response_format']['json_schema']['schema']['additionalProperties'] is False


def test_pydantic_schema_remains_owned_by_sdk():
    from scripts.evaluate_closed_loop_feedback import compatible_request
    class ActionPlan:
        pass
    kwargs = {'response_format': ActionPlan, 'messages': ['prompt']}
    assert compatible_request(kwargs) is kwargs


@pytest.mark.parametrize('model', ['gemini-3-flash-preview', 'gemini-3.5-flash'])
def test_gemini_inline_schema_retains_sdk_parsing_and_validation(tmp_path, model):
    import httpx
    import runpy
    from openai import OpenAI
    original = runpy.run_path(str(BENCH / 'embodiedbench/planner/planner_utils.py'))['ActionPlan']
    payload = {'visual_state_description': 'view', 'reasoning_and_reflection': 'reason',
               'language_plan': 'plan', 'executable_plan': [{'action_id': 0, 'action_name': 'act'}]}
    seen = []
    def handle(request):
        body = json.loads(request.content)
        seen.append(body)
        wire = json.dumps(body['response_format'])
        assert '$defs' not in wire and '$ref' not in wire
        return httpx.Response(200, json={'id': 'mock', 'object': 'chat.completion', 'created': 0,
            'model': 'gemini-3-flash-preview', 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': json.dumps(payload)}}]})
    with OpenAI(api_key='mock-only', http_client=httpx.Client(transport=httpx.MockTransport(handle))) as sdk:
        client = AuditedClient(sdk, tmp_path)
        response = client.parse(model=model, messages=[{'role': 'user', 'content': 'task'}],
                                response_format=original, temperature=0, max_tokens=4096)
    assert isinstance(response.choices[0].message.parsed, original)
    assert response.choices[0].message.parsed.model_dump() == payload
    assert '$defs' in original.model_json_schema()  # Official class unchanged.
    assert seen[0]['max_tokens'] == 4096 and seen[0]['temperature'] == 0


def test_google_response_schema_error_is_permanent():
    from scripts.evaluate_closed_loop_feedback import permanent_schema_error
    assert permanent_schema_error({'message': 'Unknown name "$defs" at generation_config.response_schema'})


@pytest.mark.parametrize('env', ['eb-hab', 'eb-nav'])
def test_qwen_endpoint_uses_same_official_feedback_settings(env):
    args = arguments(['--provider', 'qwen', '--model', 'Qwen/Qwen3-VL-8B-Instruct',
        '--env', env, '--base-url', 'http://192.168.1.185:8000/v1', '--eval-sets', 'all',
        '--episodes', '0', '--output', 'running/test/qwen_config_unused'])
    gpt = cli('--env', env, '--eval-sets', 'all', '--episodes', '0')
    qwen_config, gpt_config = official_config(args), official_config(gpt)
    qwen_config.pop('model_name'); gpt_config.pop('model_name')
    assert qwen_config == gpt_config
    assert args.api_key_env == 'QWEN_API_KEY'
    assert args.max_tokens is None and args.temperature is None


def test_qwen_requires_explicit_endpoint():
    with pytest.raises(SystemExit):
        arguments(['--provider', 'qwen', '--model', 'Qwen/Qwen3-VL-8B-Instruct',
                   '--output', 'running/test/qwen_config_unused'])


def test_resume_existing_dispatches_without_allocating_another_output(tmp_path, monkeypatch):
    import scripts.evaluate_closed_loop_feedback as driver
    import scripts.resume_closed_source_wo_feedback as controller
    args = cli('--resume-existing', '--dry-run')
    args.output = tmp_path
    saved = {**vars(args), 'protocol': 'stock_feedback_v2', 'official_config': official_config(args)}
    (tmp_path / 'config.json').write_text(json.dumps(saved, default=str))
    monkeypatch.setattr(driver, 'arguments', lambda _: args)
    seen = []
    monkeypatch.setattr(controller, 'resume', lambda root, options: seen.append((root, options.dry_run)))
    monkeypatch.setattr(driver, 'allocate_output', lambda *a, **kw: pytest.fail('must reuse the registered output'))
    assert driver.main([]) == 0
    assert seen == [(tmp_path, True)]


def test_native_result_writer_cannot_overwrite_an_existing_result(tmp_path):
    import os
    for env_name, filename, cls in (
        ('eb-hab', 'eb_habitat_evaluator.py', 'EB_HabitatEvaluator'),
        ('eb-nav', 'eb_navigation_evaluator.py', 'EB_NavigationEvaluator'),
    ):
        original = stock_method(f'embodiedbench/evaluator/{filename}', cls, 'save_episode_metric', {'os': os})
        method = adapted_function(original, [("'w', encoding='utf-8'", "'x', encoding='utf-8'", 1)])
        folder = tmp_path / env_name
        env = SimpleNamespace(log_path=str(folder), _current_episode_num=1, selected_indexes=[4])
        evaluator = SimpleNamespace(env=env)
        method(evaluator, {'task_success': 0})
        saved = next((folder / 'results').glob('episode_*_final_res.json'))
        assert saved.name == ('episode_1_final_res.json' if env_name == 'eb-hab' else 'episode_5_final_res.json')
        with pytest.raises(FileExistsError):
            method(evaluator, {'task_success': 1})
        assert json.loads(saved.read_text())['task_success'] == 0
