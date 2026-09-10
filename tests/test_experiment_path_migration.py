import json
from pathlib import Path
import tarfile

import pytest

from scripts.migrate_experiment_paths import PathRewriter, build_changes, canonical, run, sha


def moves():
    return [{"source": "running/legacy", "destination": "running/Phase1/legacy"}]


def test_path_boundaries_and_already_canonical_paths(tmp_path):
    rewrite = PathRewriter(tmp_path, moves())
    old = "running/legacy/events.jsonl"
    new = "running/Phase1/legacy/events.jsonl"
    assert rewrite(old) == new
    assert rewrite(str(tmp_path / old)) == str(tmp_path / new)
    assert rewrite("running/legacy_extra/events.jsonl") == "running/legacy_extra/events.jsonl"
    assert rewrite("running/Phase1/simulator/" + old) == "running/Phase1/simulator/" + old
    assert rewrite(new) == new


def test_hash_dependencies_and_signed_skill_envelope(tmp_path):
    event = b'{"image": "running/legacy/image.png", "task_success": 0}\n'
    envelope = {"schema_version": "2", "skill": {"version": 0},
                "protocol": {"source": "running/legacy/events.jsonl", "sha256": sha(event)}}
    envelope["artifact_sha256"] = canonical(envelope)
    artifact = json.dumps(envelope).encode()
    summary = json.dumps({"artifact": "running/legacy/skill.json", "file_sha256": sha(artifact),
                          "artifact_sha256": envelope["artifact_sha256"]}).encode()
    originals = {"events.jsonl": event, "skill.json": artifact, "summary.json": summary}
    changed, _, _ = build_changes(originals, PathRewriter(tmp_path, moves()))
    skill = json.loads(changed["skill.json"])
    result = json.loads(changed["summary.json"])
    assert skill["skill"] == envelope["skill"]
    assert skill["protocol"]["sha256"] == sha(changed["events.jsonl"])
    assert skill["artifact_sha256"] == canonical({k: skill[k] for k in ("schema_version", "skill", "protocol")})
    assert result["file_sha256"] == sha(changed["skill.json"])
    assert result["artifact_sha256"] == skill["artifact_sha256"]
    again, _, _ = build_changes(changed, PathRewriter(tmp_path, moves()))
    assert not again


def test_dynamic_phase_names_and_current_cli_defaults(tmp_path):
    mapping = [{"source": "running/phase2_audit_seed0.json",
                "destination": "running/Phase2/phase2_audit_seed0.json"},
               {"source": "running/vista_skill", "destination": "running/Phase1/vista_skill"}]
    rewrite = PathRewriter(tmp_path, mapping)
    assert rewrite('f"running/phase2_audit_seed{seed}.json"') == 'f"running/Phase2/phase2_audit_seed{seed}.json"'
    name = "vista_skill/integrations/embodiedbench/cli.py"
    changed, _, _ = build_changes({name: b'default="running/vista_skill/full"'}, rewrite)
    assert changed[name] == b'default="running/Phase5/vista_skill/full"'


def fixture_repo(tmp_path):
    destination = tmp_path / "running/Phase1/legacy"
    destination.mkdir(parents=True)
    (tmp_path / "running/legacy").symlink_to("Phase1/legacy")
    event = destination / "events.jsonl"
    event.write_text('{"image": "running/legacy/image.png", "task_success": 0}\n')
    (destination / "image.png").write_bytes(b"unchanged image")
    protected = tmp_path / "running/closed_feedback/events.jsonl"
    protected.parent.mkdir()
    protected.write_text('"running/legacy/image.png"')
    manifest = tmp_path / "moves.json"
    manifest.write_text(json.dumps(moves()))
    return manifest, event, protected


def test_apply_archives_originals_removes_alias_and_preserves_protected(tmp_path):
    manifest, event, protected = fixture_repo(tmp_path)
    before, excluded = event.read_bytes(), protected.read_bytes()
    output = tmp_path / "running/path_rewrite_backup"
    dry = run(tmp_path, manifest, output, False)
    assert dry["changed_files"] == 1
    assert not output.exists() and event.read_bytes() == before
    result = run(tmp_path, manifest, output, True)
    assert result["status"] == "verified"
    assert not (tmp_path / "running/legacy").is_symlink()
    assert "running/Phase1/legacy/image.png" in event.read_text()
    assert protected.read_bytes() == excluded
    with tarfile.open(output / "originals.tar.gz") as archive:
        assert archive.extractfile("running/Phase1/legacy/events.jsonl").read() == before
    assert run(tmp_path, manifest, output, False)["changed_files"] == 0


def test_refuses_alias_replaced_by_real_output_before_any_edit(tmp_path):
    manifest, event, _ = fixture_repo(tmp_path)
    alias = tmp_path / "running/legacy"
    alias.unlink()
    alias.mkdir()
    before = event.read_bytes()
    with pytest.raises(ValueError, match="expected alias"):
        run(tmp_path, manifest, tmp_path / "backup", True)
    assert event.read_bytes() == before


def test_refuses_protected_path_in_mapping(tmp_path):
    manifest, event, _ = fixture_repo(tmp_path)
    manifest.write_text(json.dumps([{"source": "running/closed_feedback",
                                     "destination": "running/Phase1/closed_feedback"}]))
    with pytest.raises(ValueError, match="Unsafe migration path"):
        run(tmp_path, manifest, tmp_path / "backup", True)


def test_refuses_preexisting_invalid_signed_artifact(tmp_path):
    artifact = {"schema_version": "2", "skill": {}, "protocol": {}, "artifact_sha256": "0" * 64}
    with pytest.raises(ValueError, match="digest mismatch"):
        build_changes({"skill.json": json.dumps(artifact).encode()}, PathRewriter(tmp_path, moves()))
