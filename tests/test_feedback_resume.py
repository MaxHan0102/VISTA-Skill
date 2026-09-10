import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluate_closed_loop_feedback import AuditedClient, FatalAPIError, arguments, official_config
from scripts.resume_closed_source_wo_feedback import checkpoint, next_gap, scan, worker_command


def config():
    args = arguments(['--provider', 'openai', '--model', 'gpt-5.4-mini', '--eval-sets', 'base',
                      '--episodes', '3', '--output', 'running/test/resume_test_unused'])
    return {**vars(args), 'official_config': official_config(args), 'protocol': 'stock_feedback_v2'}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, default=str))


def run(root, cfg, episodes):
    write(root / 'config.json', cfg)
    write(root / 'base/selection.json', {'dataset_sha256': 'hash'})
    path = root / 'audit/episodes.jsonl'
    path.parent.mkdir(exist_ok=True)
    path.write_text(''.join(json.dumps({'split': 'base', 'episode_number': i,
                                       'episode_id': str(i * 10)}) + '\n' for i in episodes))


def result(root, i, success=0):
    path = root / f'base/results/episode_{i}_final_res.json'
    write(path, {'task_success': success, 'num_steps': 2, 'instruction': 'task'})
    return path


def test_resume_counts_completed_failures_and_keeps_first_result(tmp_path):
    cfg = config()
    run(tmp_path, cfg, [1, 2])
    first = result(tmp_path, 1, success=0)
    broken = tmp_path / 'base/results/episode_2_final_res.json'
    broken.write_text('{')
    ranges, hashes = {'base': [1, 2, 3]}, {'base': 'hash'}
    records, identities, partial = scan(tmp_path, cfg, ranges, hashes)
    assert list(records) == ['base/1']
    assert next_gap(ranges, records) == ('base', 1, 2)
    assert partial == [str(broken)]
    child = tmp_path / '_resume/attempt_1'
    run(child, cfg, [1, 2, 3])
    result(child, 1, success=1)  # A later success cannot replace the first failure.
    result(child, 2, success=0)
    result(child, 3, success=1)
    records, identities, partial = scan(tmp_path, cfg, ranges, hashes, records)
    assert records['base/1']['path'] == str(first)
    assert next_gap(ranges, records) is None
    checkpoint(tmp_path, cfg, ranges, hashes, records, identities, partial)
    summary = json.loads((tmp_path / 'resume_summary.json').read_text())
    assert summary['completed'] == 3
    assert summary['category_mean_success'] == pytest.approx(1 / 3)
    assert broken.read_text() == '{'
    first.write_text('{}')
    with pytest.raises(ValueError, match='previously accepted result changed'):
        scan(tmp_path, cfg, ranges, hashes, records)


def test_changed_task_identity_cannot_merge(tmp_path):
    cfg = config()
    run(tmp_path, cfg, [1])
    result(tmp_path, 1)
    child = tmp_path / '_resume/attempt_1'
    run(child, cfg, [1])
    (child / 'audit/episodes.jsonl').write_text(json.dumps({'split': 'base', 'episode_number': 1, 'episode_id': 'wrong'}) + '\n')
    with pytest.raises(ValueError, match='episode identity changed'):
        scan(tmp_path, cfg, {'base': [1, 2, 3]}, {'base': 'hash'})


def test_protocol_and_dataset_drift_cannot_merge(tmp_path):
    cfg = config()
    run(tmp_path, cfg, [1])
    result(tmp_path, 1)
    with pytest.raises(ValueError, match='dataset drift'):
        scan(tmp_path, cfg, {'base': [1, 2, 3]}, {'base': 'changed'})
    child = tmp_path / '_resume/attempt_1'
    run(child, {**cfg, 'frames': 3}, [2])
    with pytest.raises(ValueError, match='different experiment settings'):
        scan(tmp_path, cfg, {'base': [1, 2, 3]}, {'base': 'hash'})


def test_gap_stops_before_already_completed_later_task():
    assert next_gap({'base': [1, 2, 3, 4, 5]}, {'base/1': {}, 'base/4': {}}) == ('base', 1, 2)


def test_resume_worker_preserves_budget_and_uses_same_model(tmp_path):
    cfg = config()
    cmd = worker_command(cfg, tmp_path, tmp_path / 'attempt', ('base', 1, 2), 3)
    assert cmd[cmd.index('--start-index') + 1] == '1'
    assert cmd[cmd.index('--model') + 1] == cfg['model']
    assert '--max-steps' not in cmd
    assert '--expected-episodes' in cmd


def test_transient_400_retries_exact_request_without_replanning(tmp_path, monkeypatch):
    import scripts.evaluate_closed_loop_feedback as driver
    calls = []
    sleeps = []
    class BadRequest(Exception):
        status_code = 400
        body = {'message': 'temporary upstream error'}
    response = SimpleNamespace(model_dump=lambda: {'usage': {}})
    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise BadRequest()
        return response
    monkeypatch.setattr(driver.time, 'sleep', sleeps.append)
    client = AuditedClient(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
                           tmp_path, api_retries=3, retry_delay=1)
    kwargs = {'model': 'gpt-5.4-mini', 'messages': ['same request'], 'max_completion_tokens': 4096}
    assert client.create(**kwargs) is response
    assert calls == [kwargs, kwargs, kwargs] and sleeps == [1, 2]
    assert client.calls == 3


def test_auth_failure_is_not_retried(tmp_path):
    class AuthError(Exception):
        status_code = 401
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        raise AuthError()
    client = AuditedClient(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
                           tmp_path, api_retries=3)
    with pytest.raises(FatalAPIError):
        client.create(model='gpt-5.4-mini')
    assert len(calls) == 1


def test_controller_keeps_partial_worker_progress_on_restart(tmp_path, monkeypatch):
    import scripts.resume_closed_source_wo_feedback as controller
    cfg = {**config(), 'smoke_policy': True}
    run(tmp_path, cfg, [1, 2])
    result(tmp_path, 1, success=0)
    monkeypatch.setattr(controller, 'datasets', lambda c: ({'base': [1, 2, 3]}, {'base': 'hash'}))
    monkeypatch.setattr(controller.time, 'sleep', lambda delay: None)
    starts = []
    def worker(cmd):
        start = int(cmd[cmd.index('--start-index') + 1])
        starts.append(start)
        output = Path(cmd[cmd.index('--output') + 1])
        artifact_root = Path(cmd[cmd.index('--artifact-root') + 1])
        assert artifact_root == tmp_path
        assert output.parent == tmp_path / '_sessions'
        child = {**cfg, 'start_index': start, 'episodes': 3 - start}
        run(output, child, list(range(start + 1, 4)))
        if start == 1:
            result(artifact_root, 2, success=0)
            (artifact_root / 'base/episode_3_step_0.json').write_text('interrupted')
            (artifact_root / 'base/results/episode_3_final_res.json').write_text('{')
            write(output / 'failure.json', {'http_status': 400})
            return 1
        assert not (artifact_root / 'base/episode_3_step_0.json').exists()
        assert not (artifact_root / 'base/results/episode_3_final_res.json').exists()
        result(artifact_root, 3, success=1)
        return 0
    monkeypatch.setattr(controller, 'launch', worker)
    args = SimpleNamespace(dry_run=False, api_retries=3, max_retries=3, retry_delay=0)
    controller.resume(tmp_path, args)
    assert starts == [1, 2]
    summary = json.loads((tmp_path / 'resume_summary.json').read_text())
    assert summary['status'] == 'complete' and summary['completed'] == 3
    assert summary['category_mean_success'] == pytest.approx(1 / 3)
    # Re-running a completed checkpoint launches no more workers.
    controller.resume(tmp_path, args)
    assert starts == [1, 2]
    assert not (tmp_path / '_resume').exists()
    assert next((tmp_path / '_interrupted').rglob('episode_3_step_0.json')).read_text() == 'interrupted'
    assert all(Path(v['path']).parents[2] == tmp_path for v in json.loads((tmp_path / 'resume_state.json').read_text())['records'].values())


def test_invalid_schema_stops_request_and_controller_retries(tmp_path, monkeypatch):
    import scripts.evaluate_closed_loop_feedback as driver
    from scripts.resume_closed_source_wo_feedback import retryable_failure
    details = {'code': 'invalid_json_schema', 'message': "Invalid schema: additionalProperties must be false"}
    class BadRequest(Exception):
        status_code = 400
        body = {'error': details}
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        raise BadRequest()
    monkeypatch.setattr(driver.time, 'sleep', lambda _: pytest.fail('permanent schema error must not retry'))
    audit = tmp_path / 'audit'
    audit.mkdir()
    client = AuditedClient(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), audit, api_retries=3)
    with pytest.raises(FatalAPIError) as caught:
        client.create(model='gpt-5.4-mini')
    assert caught.value.provider_error == details
    assert len(calls) == 1
    write(tmp_path / 'failure.json', {'http_status': 400})  # Also supports older failure files.
    assert retryable_failure(tmp_path) is False
    write(tmp_path / 'failure.json', {'http_status': 400, 'provider_error': details})
    (audit / 'api_errors.jsonl').unlink()
    assert retryable_failure(tmp_path) is False


def test_resume_records_schema_revision_without_replacing_old_results(tmp_path):
    cfg = config()
    run(tmp_path, cfg, [1])
    first = result(tmp_path, 1, success=0)
    ranges, hashes = {'base': [1, 2]}, {'base': 'hash'}
    records, _, _ = scan(tmp_path, cfg, ranges, hashes)
    records['base/1'].pop('schema_compatibility')  # A checkpoint written by the old controller.
    child = tmp_path / '_resume/attempt_1'
    run(child, {**cfg, 'schema_compatibility': 'closed_objects_v1'}, [1, 2])
    result(child, 1, success=1)
    result(child, 2, success=1)
    records, identities, partial = scan(tmp_path, cfg, ranges, hashes, records)
    checkpoint(tmp_path, cfg, ranges, hashes, records, identities, partial)
    summary = json.loads((tmp_path / 'resume_summary.json').read_text())
    assert records['base/1']['path'] == str(first)
    assert summary['category_mean_success'] == 0.5
    assert summary['schema_compatibility_counts'] == {'stock': 1, 'closed_objects_v1': 1}
    assert summary['mixed_schema_compatibility'] is True
