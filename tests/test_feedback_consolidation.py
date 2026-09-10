import json

from scripts.consolidate_closed_feedback import consolidate, episode_files
from scripts.resume_closed_source_wo_feedback import scan, sha
from test_feedback_resume import config, result, run


def test_consolidation_preserves_first_failure_and_archives_partial_files(tmp_path, monkeypatch):
    import scripts.consolidate_closed_feedback as module
    monkeypatch.setattr(module, 'REPO', tmp_path)
    root = tmp_path / 'running/test/merge'
    cfg = config()
    run(root, cfg, [1, 2])
    first = result(root, 1, 0)
    original_hash = sha(first)
    residue = root / 'base/episode_2_step_1.json'
    residue.write_text('partial log')
    child = root / '_resume/attempt_1'
    run(child, cfg, [1, 2])
    result(child, 1, 1)
    accepted = result(child, 2, 0)
    log = child / 'base/episode_2_step_2.json'
    log.write_text('complete log')
    image = child / 'base/images/episode_2/episode_2_step_0.png'
    image.parent.mkdir(parents=True)
    image.write_bytes(b'image')
    monkeypatch.setattr(module, 'datasets', lambda _: ({'base': [1, 2, 3]}, {'base': 'hash'}))
    records = consolidate(root)
    assert sha(first) == original_hash
    assert sha(root / 'base/results/episode_2_final_res.json') == sha(accepted)
    assert not residue.exists()
    assert next((root / '_interrupted').rglob(residue.name)).read_text() == 'partial log'
    assert (root / 'base/episode_2_step_2.json').read_text() == 'complete log'
    assert (root / 'base/images/episode_2/episode_2_step_0.png').read_bytes() == b'image'
    assert accepted.exists() and log.exists()
    scanned, _, _ = scan(root, cfg, {'base': [1, 2, 3]}, {'base': 'hash'}, records)
    assert scanned['base/2']['source_path'] == str(accepted)
    assert json.loads((root / 'base/results/summary.json').read_text())['task_success'] == 0
    assert consolidate(root) == scanned  # Idempotent; no duplicate trajectories.


def test_episode_file_selection_does_not_match_neighboring_ordinals(tmp_path):
    (tmp_path / 'base').mkdir()
    for name in ('episode_1_step_0.png', 'episode_10_step_0.png', 'episode_11_step_0.png'):
        (tmp_path / 'base' / name).touch()
    assert [p.name for p in episode_files(tmp_path, 'base', 1)] == ['episode_1_step_0.png']
