from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import json

import pytest

from vista_skill.integrations.embodiedbench import cli
from vista_skill.integrations.embodiedbench.cli import (
    _artifact_evaluation_manifest,
    _gate_rollout_seeds,
    _habitat_exp_name,
    _audit_evaluation_protocol,
    _official_episode_ids,
    _paired_selection_coordinates,
    parse_args,
)
from vista_skill.config import load_config
from vista_skill.protocol import TaskCoordinate, load_experiment_manifest
from vista_skill.skills import (
    initialize_shared_skill,
    load_skill_artifact_record,
    save_skill_artifact,
)


def test_cli_requires_explicit_workflow_and_exposes_control_modes() -> None:
    no_skill = parse_args(["evaluate", "--mode", "no_skill"])
    assert no_skill.command == "evaluate"
    assert no_skill.mode == "no_skill"
    full = parse_args(["experiment", "--method", "full"])
    assert full.method == "full"
    frozen = parse_args(["evaluate"])
    assert frozen.config == "configs/vista_p0.json"
    assert not frozen.diagnostic


def test_official_final_test_coordinates_are_read_only_and_hashed() -> None:
    episode_ids, dataset_hash = _official_episode_ids("base")
    assert len(episode_ids) > 0
    assert len(dataset_hash) == 64


def test_proxy_and_finalist_use_disjoint_task_pools() -> None:
    tasks = tuple(
        TaskCoordinate(str(index), str(index), "all", index)
        for index in range(20)
    )
    stages = _paired_selection_coordinates(
        tasks,
        (0, 1, 2),
        proxy_budget=10,
        finalist_budget=30,
    )
    assert len(stages["proxy"]) == 10
    assert len(stages["finalist"]) == 30
    assert not (
        {item.episode_id for item in stages["proxy"]}
        & {item.episode_id for item in stages["finalist"]}
    )
    assert {item.seed for item in stages["finalist"]} == {0, 1, 2}


def test_experiment_wiring_reaches_environment_without_evaluate_only_args(
    monkeypatch,
) -> None:
    args = parse_args(
        [
            "experiment",
            "--method",
            "rule_only",
            "--diagnostic",
            "--max-acquisition-episodes",
            "1",
        ]
    )

    def stop_at_environment(*args, **kwargs):
        raise RuntimeError("environment reached")

    monkeypatch.setattr(cli, "create_habitat_env", stop_at_environment)
    try:
        cli._run_experiment(args)
    except RuntimeError as error:
        assert str(error) == "environment reached"


@pytest.mark.parametrize(
    "metadata_key",
    ["config_sha256", "manifest_sha256", "rng_seed_policy"],
)
def test_controlled_evaluation_rejects_artifact_protocol_mismatch(
    tmp_path,
    metadata_key,
) -> None:
    config = load_config("configs/vista_p0.json")
    manifest = load_experiment_manifest("configs/eb_hab_train_validation_manifest.json")
    args = parse_args(["evaluate"])
    protocol = {
        "config_sha256": config.digest,
        "manifest_sha256": manifest.digest,
        "executor_model": args.model_name,
        "executor_model_type": args.model_type,
        "tensor_parallel": args.tp,
        "executor_temperature": 0.0,
        "max_completion_tokens": 1024,
        "n_shots": args.n_shots,
        "resolution": args.resolution,
        "frozen": True,
        "diagnostic": False,
        "acquisition_episode_budget": 60,
        "rng_seed_policy": "python+numpy+torch+habitat+openai_request",
    }
    protocol[metadata_key] = "0" * 64
    path = tmp_path / "frozen_skill.json"
    save_skill_artifact(
        path,
        replace(initialize_shared_skill(), frozen=True),
        protocol=protocol,
    )
    artifact = load_skill_artifact_record(path, require_frozen=True)

    with pytest.raises(ValueError, match=f"artifact {metadata_key} mismatch"):
        _audit_evaluation_protocol(args, config, manifest, artifact)


def test_diagnostic_evaluation_explicitly_allows_protocol_deviation(tmp_path) -> None:
    config = load_config("configs/vista_p0.json")
    manifest = load_experiment_manifest("configs/eb_hab_train_validation_manifest.json")
    args = parse_args(
        [
            "evaluate",
            "--diagnostic",
            "--model-name",
            "diagnostic-executor",
            "--n-shots",
            "1",
            "--resolution",
            "256",
        ]
    )
    path = tmp_path / "frozen_skill.json"
    save_skill_artifact(
        path,
        replace(initialize_shared_skill(), frozen=True),
        protocol={"config_sha256": "diagnostic"},
    )
    artifact = load_skill_artifact_record(path, require_frozen=True)

    _audit_evaluation_protocol(args, config, manifest, artifact)


def test_controlled_evaluation_rejects_diagnostic_artifact(tmp_path) -> None:
    config = load_config("configs/vista_p0.json")
    manifest = load_experiment_manifest("configs/eb_hab_train_validation_manifest.json")
    args = parse_args(["evaluate"])
    path = tmp_path / "diagnostic_skill.json"
    save_skill_artifact(
        path,
        replace(initialize_shared_skill(), frozen=True),
        protocol={
            "config_sha256": config.digest,
            "manifest_sha256": manifest.digest,
            "executor_model": args.model_name,
            "executor_model_type": args.model_type,
            "tensor_parallel": args.tp,
            "executor_temperature": 0.0,
            "max_completion_tokens": 1024,
            "n_shots": args.n_shots,
            "resolution": args.resolution,
            "frozen": True,
            "diagnostic": True,
            "acquisition_episode_budget": 1,
        },
    )
    artifact = load_skill_artifact_record(path, require_frozen=True)

    with pytest.raises(ValueError, match="artifact diagnostic mismatch"):
        _audit_evaluation_protocol(args, config, manifest, artifact)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("--model-name", "other-executor", "executor model differs"),
        ("--model-type", "local", "executor model type differs"),
        ("--tp", "2", "tensor parallel setting differs"),
        ("--n-shots", "1", "n-shot count differs"),
        ("--resolution", "256", "image resolution differs"),
    ],
)
def test_controlled_evaluation_rejects_runtime_protocol_drift(
    argument,
    value,
    message,
) -> None:
    config = load_config("configs/vista_p0.json")
    manifest = load_experiment_manifest("configs/eb_hab_train_validation_manifest.json")
    args = parse_args(["evaluate", argument, value])

    with pytest.raises(ValueError, match=message):
        _audit_evaluation_protocol(args, config, manifest, artifact=None)


def test_frozen_evaluation_audits_before_environment_creation(
    tmp_path,
    monkeypatch,
) -> None:
    config = load_config("configs/vista_p0.json")
    manifest = load_experiment_manifest("configs/eb_hab_train_validation_manifest.json")
    args = parse_args(
        [
            "evaluate",
            "--skill",
            str(tmp_path / "frozen_skill.json"),
        ]
    )
    save_skill_artifact(
        args.skill,
        replace(initialize_shared_skill(), frozen=True),
        protocol={
            "config_sha256": "0" * 64,
            "manifest_sha256": manifest.digest,
            "executor_model": args.model_name,
            "executor_model_type": args.model_type,
            "tensor_parallel": args.tp,
            "executor_temperature": 0.0,
            "max_completion_tokens": 1024,
            "n_shots": args.n_shots,
            "resolution": args.resolution,
            "frozen": True,
        },
    )
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "_load_verified_manifest", lambda path: manifest)

    def unexpected_environment(*args, **kwargs):
        raise AssertionError("environment must not be constructed before audit")

    monkeypatch.setattr(cli, "create_habitat_env", unexpected_environment)
    with pytest.raises(ValueError, match="artifact config_sha256 mismatch"):
        cli._run_frozen_evaluation(args)


def test_frozen_artifact_restores_and_checks_its_rotated_split(tmp_path) -> None:
    manifest = load_experiment_manifest("configs/eb_hab_train_validation_manifest.json")
    rotated = manifest.rotate_split(1)
    path = tmp_path / "frozen_skill.json"
    save_skill_artifact(
        path,
        replace(initialize_shared_skill(), frozen=True),
        protocol={
            "split_rotation_index": 1,
            "split_sha256": rotated.split.digest(),
        },
    )
    artifact = load_skill_artifact_record(path, require_frozen=True)
    restored = _artifact_evaluation_manifest(manifest, artifact)
    assert restored.split.audit == tuple(str(item) for item in range(20))
    assert not (
        set(restored.split.audit)
        & (set(restored.split.acquisition) | set(restored.split.selection))
    )

    bad_path = tmp_path / "bad_skill.json"
    save_skill_artifact(
        bad_path,
        replace(initialize_shared_skill(), frozen=True),
        protocol={"split_rotation_index": 1, "split_sha256": "0" * 64},
    )
    with pytest.raises(ValueError, match="split_sha256 mismatch"):
        _artifact_evaluation_manifest(
            manifest,
            load_skill_artifact_record(bad_path, require_frozen=True),
        )


def test_independent_run_seeds_have_disjoint_gate_rollout_seeds() -> None:
    assert not (set(_gate_rollout_seeds(0)) & set(_gate_rollout_seeds(1)))


def test_habitat_image_namespaces_distinguish_coordinates_and_skills() -> None:
    first = _habitat_exp_name("gate", "proxy", "episode-1", "s0", "skill-a")
    second = _habitat_exp_name("gate", "proxy", "episode-1", "s1", "skill-a")
    candidate = _habitat_exp_name("gate", "proxy", "episode-1", "s0", "skill-b")
    assert len({first, second, candidate}) == 3


def test_experiment_rejects_existing_output_before_environment(tmp_path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    args = parse_args(
        [
            "experiment",
            "--method",
            "rule_only",
            "--diagnostic",
            "--max-acquisition-episodes",
            "1",
            "--output-dir",
            str(output),
        ]
    )
    with pytest.raises(FileExistsError, match="already exists"):
        cli._run_experiment(args)


@pytest.mark.parametrize("existing_target", ["events", "summary"])
def test_evaluation_rejects_existing_outputs_before_environment(
    tmp_path,
    monkeypatch,
    existing_target,
) -> None:
    output = tmp_path / "events.jsonl"
    target = output if existing_target == "events" else output.with_suffix(".summary.json")
    target.write_text("existing\n", encoding="utf-8")
    args = parse_args(
        [
            "evaluate",
            "--mode",
            "no_skill",
            "--output",
            str(output),
        ]
    )

    def unexpected_environment(*args, **kwargs):
        raise AssertionError("environment must not be constructed for existing output")

    monkeypatch.setattr(cli, "create_habitat_env", unexpected_environment)
    with pytest.raises(FileExistsError, match="already exists"):
        cli._run_frozen_evaluation(args)


def test_evaluation_closes_environment_when_planner_construction_fails(
    tmp_path,
    monkeypatch,
) -> None:
    args = parse_args(
        [
            "evaluate",
            "--mode",
            "no_skill",
            "--diagnostic",
            "--max-episodes",
            "1",
            "--output",
            str(tmp_path / "events.jsonl"),
        ]
    )
    closed = []

    class FakeEnv:
        def close(self):
            closed.append(True)

    monkeypatch.setattr(cli, "create_habitat_env", lambda *args, **kwargs: FakeEnv())
    monkeypatch.setattr(cli, "seed_habitat_env", lambda env, seed: None)
    monkeypatch.setattr(
        cli,
        "_make_runner",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("planner failed")),
    )

    with pytest.raises(RuntimeError, match="planner failed"):
        cli._run_frozen_evaluation(args)
    assert closed == [True]


def test_experiment_runs_three_independent_rotated_acquisitions(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "experiment"
    args = parse_args(
        [
            "experiment",
            "--method",
            "rule_only",
            "--diagnostic",
            "--max-acquisition-episodes",
            "1",
            "--output-dir",
            str(output),
        ]
    )
    created = []
    seeded = []

    class FakeEnv:
        def close(self):
            pass

    def create_env(eval_set, *, episode_ids, **kwargs):
        created.append(tuple(episode_ids))
        return FakeEnv()

    class FakeRunner:
        def run_episode(self, *, expected_episode_id):
            return SimpleNamespace(episode_id=expected_episode_id)

    monkeypatch.setattr(cli, "create_habitat_env", create_env)
    monkeypatch.setattr(cli, "seed_habitat_env", lambda env, seed: seeded.append(seed))
    monkeypatch.setattr(cli, "_make_runner", lambda *args, **kwargs: FakeRunner())
    cli._run_experiment(args)

    assert created == [("0",), ("20",), ("40",)]
    assert seeded == [0, 1, 2]
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    assert manifest["run_count"] == 3
    assert [item["evolution_seed"] for item in manifest["runs"]] == [0, 1, 2]
    assert all((output / f"seed_{seed}" / "frozen_skill.json").exists() for seed in seeded)


# --- EB-Navigation evaluate-path selector ----------------------------------


def test_evaluate_env_defaults_to_eb_hab() -> None:
    assert parse_args(["evaluate"]).env == "eb-hab"


def test_nav_official_episode_ids() -> None:
    ids, dataset_hash = _official_episode_ids("base", env_name="eb-nav")
    assert ids[0] == "nav_0"
    assert ids == tuple(f"nav_{i}" for i in range(len(ids)))
    assert len(dataset_hash) == 64


def test_nav_official_episode_ids_rejects_hab_subset() -> None:
    with pytest.raises(ValueError):
        _official_episode_ids("spatial_relationship", env_name="eb-nav")


def test_experiment_rejects_nav_env(tmp_path) -> None:
    args = parse_args(
        ["experiment", "--env", "eb-nav", "--method", "full", "--method-model", "m"]
    )
    with pytest.raises(ValueError, match="EB-Habitat-only"):
        cli._run_experiment(args)


def test_nav_evaluate_skips_manifest_verification(monkeypatch, tmp_path) -> None:
    """Nav has no train manifest; evaluate must get past manifest verification to env creation."""

    def _boom(*a, **kw):
        raise RuntimeError("REACHED_ENV_CREATION")

    monkeypatch.setattr(cli, "create_nav_env", _boom)
    args = parse_args(
        [
            "evaluate", "--env", "eb-nav", "--eval-set", "base",
            "--stage", "official_test", "--mode", "no_skill",
            "--config", "configs/vista_nav.json", "--diagnostic",
            "--output", str(tmp_path / "events.jsonl"),
        ]
    )
    with pytest.raises(RuntimeError, match="REACHED_ENV_CREATION"):
        cli._run_frozen_evaluation(args)


def test_nav_evaluate_rejects_non_official_stage(tmp_path) -> None:
    args = parse_args(
        [
            "evaluate", "--env", "eb-nav", "--eval-set", "base",
            "--stage", "audit", "--mode", "no_skill",
            "--config", "configs/vista_nav.json", "--diagnostic",
            "--output", str(tmp_path / "events.jsonl"),
        ]
    )
    with pytest.raises(ValueError, match="official_test"):
        cli._run_frozen_evaluation(args)
