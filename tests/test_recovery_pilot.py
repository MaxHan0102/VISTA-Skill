from __future__ import annotations

import json
from pathlib import Path
import pickle

import pytest

from scripts.phase5_recovery_dataset import SOURCE, build_dataset
from scripts.phase5_recovery_pilot import local_metrics, preregister


def test_generated_roles_are_new_goals_with_disjoint_source_layouts():
    source = pickle.loads(SOURCE.read_bytes())
    original = pickle.dumps(source)
    generated, manifest = build_dataset(source)
    assert pickle.dumps(source) == original
    again, same = build_dataset(source)
    assert pickle.dumps(again) == pickle.dumps(generated) and same == manifest
    rows = manifest["tasks"]
    assert len(rows) == 36
    assert len({r["source_layout_sha256"] for r in rows}) == 36
    assert len({r["source_episode_id"] for r in rows}) == 36
    originals = {str(e["episode_id"]): e for e in source["all_eps"]}
    for row, episode in zip(rows, generated["all_eps"]):
        assert episode["goal_preds"] != originals[row["source_episode_id"]]["goal_preds"]
        assert episode["episode_id"] == row["episode_id"]
    for role in ("proxy", "finalist", "audit"):
        for endpoint in (0, 1):
            block = [r for r in rows if r["role"] == role and r["endpoint_index"] == endpoint]
            assert len(block) == 4
            assert sum(r["scope"] == "protected" for r in block) == 1


def test_preregistration_rejects_endpoint_and_code_drift(tmp_path, monkeypatch):
    import scripts.phase5_recovery_pilot as pilot
    root = tmp_path
    (root / "vista_skill").mkdir()
    source = root / "vista_skill/example.py"
    source.write_text("x = 1\n")
    (root / "scripts").mkdir()
    (root / "scripts/phase5_recovery_dataset.py").write_text("pass\n")
    driver = root / "scripts/phase5_recovery_pilot.py"
    driver.write_text("pass\n")
    (root / "config.json").write_text("{}")
    (root / "data.pickle").write_bytes(b"test")
    (root / "manifest.json").write_text(json.dumps({"dataset_sha256": pilot.sha(root / "data.pickle")}))
    protocol = {"output_dir": "run", "dataset": "data.pickle", "dataset_manifest": "manifest.json", "config": "config.json"}
    protocol_path = root / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    monkeypatch.setattr(pilot, "REPO", root)
    monkeypatch.setattr(pilot, "__file__", str(driver))
    preregister(protocol_path, protocol, ["a", "b"])
    preregister(protocol_path, protocol, ["a", "b"])
    with pytest.raises(ValueError, match="changed after preregistration"):
        preregister(protocol_path, protocol, ["b", "a"])
    source.write_text("x = 2\n")
    with pytest.raises(ValueError, match="changed after preregistration"):
        preregister(protocol_path, protocol, ["a", "b"])


def test_local_retry_metric_uses_preaction_evidence():
    from test_recovery import _chain
    events = _chain()
    events[2]["last_action_success"] = True
    records = [{"event_type": "transition", "payload": event} for event in events]
    metrics = local_metrics(records)
    assert metrics["retries_without_new_evidence"] == 1
    assert metrics["successful_retries_without_evidence"] == 1


def test_paired_analysis_recomputes_metrics_and_rejects_incomplete_data(tmp_path, monkeypatch):
    import scripts.phase5_recovery_pilot as pilot
    import scripts.phase5_variance_audit as variance
    real_root = pilot.REPO
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    monkeypatch.setattr(variance, "REPO", tmp_path)
    protocol = {"output_dir": "run", "config": str(real_root / "configs/vista_phase5_hab.json"),
                "rollout_seeds": [0, 1, 2]}
    rows = [{"episode_id": f"p57_proxy_{i:02d}", "role": "proxy", "endpoint_index": i % 2,
             "scope": "affected" if i < 6 else "protected", "subgroup": "same_scene"} for i in range(8)]
    for row in rows:
        for seed in protocol["rollout_seeds"]:
            for arm in ("parent", "candidate"):
                # Six affected tasks improve, two protected tasks stay unchanged.
                success = float(arm == "candidate" or row["scope"] == "protected")
                payload = {"episode_id": row["episode_id"], "task_success": success,
                           "task_progress": success, "environment_steps": 4, "invalid_actions": 0,
                           "planner_steps": 4, "planner_output_errors": 0, "trajectory": []}
                cost = {"calls": 4, "prompt_tokens": 40, "completion_tokens": 8}
                records = [{"event_type": "executor_usage", "payload": cost},
                           {"event_type": "method_usage", "payload": {"by_purpose": {}}},
                           {"event_type": "episode_result", "payload": payload}]
                path = tmp_path / "run/rollouts/proxy" / row["episode_id"] / f"s{seed}_{arm}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(json.dumps(r) + "\n" for r in records))
                pilot.write(path.with_suffix(".metadata.json"), {
                    "artifact_sha256": pilot.sha(path), "endpoint_index": row["endpoint_index"],
                    "executor_usage": cost, "method_usage": {}, "local_metrics": pilot.local_metrics(records),
                })
    manifest = {"tasks": rows}
    assert pilot.analyze_stage(protocol, manifest, "proxy") == "supportive"
    analysis_path = tmp_path / "run/proxy_analysis.json"
    previous = analysis_path.read_bytes()
    assert pilot.analyze_stage(protocol, manifest, "proxy") == "supportive"
    assert analysis_path.read_bytes() == previous
    path.unlink()
    with pytest.raises(FileNotFoundError):
        pilot.analyze_stage(protocol, manifest, "proxy")
