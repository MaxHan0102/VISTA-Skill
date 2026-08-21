from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from vista_skill.artifacts import JsonlArtifactWriter
from vista_skill.attribution import CreditAssigner
from vista_skill.baselines import (
    CommonGateProposalAdapter,
    EmbodiSkillFrontend,
    EmbodiSkillNativeUpdater,
    EpisodeSummary,
    UnconditionalReflectionFrontend,
)
from vista_skill.belief import BeliefLedger
from vista_skill.clustering import EventClusterer
from vista_skill.config import VistaConfig, load_config
from vista_skill.evaluation import (
    EpisodeCoordinate,
    PairedRolloutEvaluator,
    RolloutScore,
    composite_task_score,
)
from vista_skill.fault_injection import FaultType, inject_skill_fault
from vista_skill.evidence import EvidenceExtractor, _nav_feedback_strategy
from vista_skill.integrations.embodiedbench.environment import (
    create_habitat_env,
    create_nav_env,
    nav_goal_predicates,
    seed_habitat_env,
    seed_nav_env,
    seed_process_rngs,
)
from vista_skill.integrations.embodiedbench.planner import (
    configure_planner_inference_seed,
    make_skill_aware_planner,
)
from vista_skill.integrations.embodiedbench.runner import (
    EpisodeResult,
    HabitatRolloutRunner,
)
from vista_skill.lineage import LineageStore
from vista_skill.models import (
    JsonAttributionTeacher,
    JsonBoundedPatchGenerator,
    JsonGoalGrounder,
    JsonTrajectoryTeacher,
    JsonVisualEvidenceProvider,
    OpenAICompatibleJsonModel,
)
from vista_skill.pipeline import VistaSkillEngine
from vista_skill.protocol import ExperimentManifest, load_experiment_manifest
from vista_skill.schemas import SkillSpec
from vista_skill.skills import (
    empty_shared_skill,
    minimal_shared_skill,
    SkillArtifact,
    initialize_nav_skill,
    initialize_shared_skill,
    load_skill_artifact_record,
    render_skill,
    save_skill_artifact,
    save_content_addressed_skill,
    skill_digest,
)
from vista_skill.update_audit import (
    make_rotated_audit_plan,
    run_rotated_update_audit,
)
from vista_skill.workflow import (
    EvolutionWorkflow,
    TrajectoryEvolutionWorkflow,
    build_candidate_gate,
)


DEFAULT_MANIFEST = "configs/eb_hab_train_validation_manifest.json"

# Controlled trajectory-level baselines that share the acquisition/freeze/audit
# skeleton with the full method but evolve from whole-episode reflections.
_TRAJECTORY_METHODS = (
    "embodiskill_star_native",
    "embodiskill_star_common_gate",
    "vista_without_vtca",
)
_METHODS_REQUIRING_TEACHER = ("full", *_TRAJECTORY_METHODS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="External VISTA-Skill workflows for stock EB-Habitat."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    experiment = subparsers.add_parser(
        "experiment", help="Acquire evidence, evolve through the paired gate, and freeze."
    )
    _add_executor_args(experiment)
    experiment.add_argument(
        "--method",
        choices=("full", "rule_only", *_TRAJECTORY_METHODS),
        default="full",
    )
    experiment.add_argument("--method-model")
    experiment.add_argument("--method-base-url", default=os.environ.get("VISTA_METHOD_BASE_URL"))
    experiment.add_argument("--method-api-key", default=os.environ.get("VISTA_METHOD_API_KEY"))
    experiment.add_argument("--manifest", default=DEFAULT_MANIFEST)
    experiment.add_argument("--config", default="configs/vista_p0.json")
    experiment.add_argument("--output-dir", default="running/vista_skill/full")
    experiment.add_argument("--evolution-seeds", default="0,1,2")
    experiment.add_argument("--max-acquisition-episodes", type=int)
    experiment.add_argument(
        "--initial-skill",
        choices=("shared", "minimal", "empty"),
        default="shared",
        help="Initial Skill variant (diagnostic init-sensitivity regime): "
        "'shared' = S0; 'minimal'/'empty' = degraded starting points for "
        "testing whether evolution can grow or recover skills. Requires "
        "--diagnostic.",
    )
    experiment.add_argument(
        "--skill-fault",
        choices=(
            "termination",
            "procedure",
            "effect",
            "constraint",
            "activation",
            "effect_pick_inversion",
            "constraint_pick_multihold",
        ),
        help="Inject a structured fault into the initial shared Skill "
        "(fault-injection effectiveness diagnostic: recurrence becomes reachable, "
        "so evolution can be observed quickly). Requires --diagnostic.",
    )
    experiment.add_argument(
        "--diagnostic",
        action="store_true",
        help="Allow reduced episode counts; outputs are not controlled-protocol results.",
    )

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate a digest-checked frozen Skill with evolution disabled."
    )
    _add_executor_args(evaluate)
    evaluate.add_argument(
        "--mode",
        choices=("no_skill", "static_shared_skill", "frozen_skill"),
        default="frozen_skill",
    )
    evaluate.add_argument("--skill")
    evaluate.add_argument("--manifest", default=DEFAULT_MANIFEST)
    evaluate.add_argument("--config", default="configs/vista_p0.json")
    evaluate.add_argument(
        "--diagnostic",
        action="store_true",
        help="Allow protocol deviations while preserving artifact integrity checks.",
    )
    evaluate.add_argument(
        "--stage",
        choices=("acquisition", "selection", "audit", "official_test"),
        default="audit",
    )
    evaluate.add_argument("--output", default="running/vista_skill/frozen_audit/events.jsonl")
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--max-episodes", type=int)
    args = parser.parse_args(argv)
    if args.n_shots is None:
        # Match stock EmbodiedBench per-env defaults (eb-hab.yaml=10, eb-nav.yaml=3)
        # so VISTA rollouts share the published-EB in-context-example surface.
        args.n_shots = 3 if args.env == "eb-nav" else 10
    return args


def _add_executor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env", choices=("eb-hab", "eb-nav"), default="eb-hab")
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--model-type", default="remote")
    parser.add_argument("--executor-base-url", default=os.environ.get("remote_url"))
    parser.add_argument("--eval-set", default="train_validation")
    parser.add_argument(
        "--n-shots",
        type=int,
        default=None,
        help="In-context examples; defaults to 10 for eb-hab and 3 for eb-nav "
        "(matches stock EB per-env YAML so VISTA rollouts share the EB ICL surface).",
    )
    parser.add_argument("--resolution", type=int, default=500)
    parser.add_argument("--tp", type=int, default=1)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.executor_base_url:
        # Stock RemoteModel reads this at import time.
        os.environ["remote_url"] = args.executor_base_url
    if args.command == "experiment":
        _run_experiment(args)
    else:
        _run_frozen_evaluation(args)


def _run_experiment(args: argparse.Namespace) -> None:
    if args.env != "eb-hab":
        raise ValueError("evolution/experiment is EB-Habitat-only; EB-Nav supports 'evaluate'")
    config = load_config(args.config)
    configured_manifest = Path(str(config.raw["task_manifest"])).resolve()
    if Path(args.manifest).resolve() != configured_manifest:
        raise ValueError("controlled protocol manifest differs from config")
    manifest = _load_verified_manifest(args.manifest)
    _validate_controlled_executor(args, config)
    if args.eval_set != "train_validation":
        raise ValueError("the 60/20/20 evolution protocol requires train_validation")
    if args.method in _METHODS_REQUIRING_TEACHER and not args.method_model:
        raise ValueError(
            f"--method {args.method} requires --method-model; use rule_only only as an ablation"
        )
    if args.max_acquisition_episodes is not None and not args.diagnostic:
        raise ValueError("reduced acquisition requires --diagnostic")
    if args.skill_fault and not args.diagnostic:
        raise ValueError("--skill-fault is a diagnostic deviation and requires --diagnostic")
    if args.initial_skill != "shared" and not args.diagnostic:
        raise ValueError("--initial-skill is a diagnostic deviation and requires --diagnostic")
    if args.initial_skill != "shared" and args.skill_fault:
        raise ValueError("--initial-skill and --skill-fault are exclusive regime knobs")
    output_dir = Path(args.output_dir)
    _require_new_output(output_dir, "experiment output directory")
    experiment_id = uuid.uuid4().hex
    seeds = _parse_seeds(args.evolution_seeds)
    run_records = []
    for rotation_index, evolution_seed in enumerate(seeds):
        run_manifest = manifest.rotate_split(rotation_index)
        run_dir = output_dir / f"seed_{evolution_seed}"
        run_id = f"{experiment_id}_seed_{evolution_seed}"
        method_model = (
            _make_method_model(args, seed=evolution_seed)
            if args.method in _METHODS_REQUIRING_TEACHER
            else None
        )
        # The trajectory baselines evolve from whole-episode reflections, NOT
        # action-level VTCA -- that is full's mechanism. Give them a rule-only
        # engine (no per-action evidence/attribution teacher) so they pay only
        # their own trajectory-reflection cost (fair RQ4 + far faster). The
        # trajectory teacher is wired separately in _make_trajectory_workflow.
        engine_model = None if args.method in _TRAJECTORY_METHODS else method_model
        engine, goal_grounder = _make_engine(
            config,
            engine_model,
            skill_fault=args.skill_fault,
            initial_skill=args.initial_skill,
        )

        acquisition = run_manifest.coordinates_for("acquisition")
        if args.max_acquisition_episodes is not None:
            acquisition = acquisition[: args.max_acquisition_episodes]
        acquisition_ids = tuple(item.episode_id for item in acquisition)
        seed_process_rngs(evolution_seed)
        env = create_habitat_env(
            "train_validation",
            episode_ids=acquisition_ids,
            exp_name=_habitat_exp_name(run_id, "acquisition"),
            resolution=args.resolution,
        )
        try:
            seed_habitat_env(env, evolution_seed)
            gate_seeds = _gate_rollout_seeds(evolution_seed)
            protocol = {
                **_protocol_record(args, config, manifest),
                "evolution_seed": evolution_seed,
                "run_id": run_id,
                "split_rotation_index": rotation_index,
                "split_sha256": run_manifest.split.digest(),
                "gate_rollout_seeds": gate_seeds,
                "acquisition_episode_budget": len(acquisition),
            }
            if args.method == "full":
                paired = _make_paired_evaluator(
                    args, run_manifest, gate_seeds, run_dir, config, run_id=run_id
                )
                lineage = LineageStore(run_dir / "lineage.jsonl")
                workflow = EvolutionWorkflow(
                    engine,
                    generator=JsonBoundedPatchGenerator(method_model),
                    paired_evaluator=paired,
                    lineage=lineage,
                    config=config,
                    protocol=protocol,
                )
            elif args.method in _TRAJECTORY_METHODS:
                lineage = LineageStore(run_dir / "lineage.jsonl")
                workflow = _make_trajectory_workflow(
                    args,
                    engine=engine,
                    method_model=method_model,
                    run_manifest=run_manifest,
                    gate_seeds=gate_seeds,
                    run_dir=run_dir,
                    run_id=run_id,
                    lineage=lineage,
                    config=config,
                    protocol=protocol,
                )
            else:
                lineage = None
                workflow = None

            runner = _make_runner(
                args,
                env,
                engine.skill,
                run_dir / "acquisition.jsonl",
                engine=engine,
                goal_grounder=goal_grounder,
                expected_episode_ids=acquisition_ids,
                task_coordinates=acquisition,
                rollout_seed=evolution_seed,
            )
            acquisition_results = []
            ready_cluster_counts = []
            evolution_results = []
            for coordinate in acquisition:
                episode_result = runner.run_episode(
                    expected_episode_id=coordinate.episode_id
                )
                acquisition_results.append(episode_result)
                if workflow is not None:
                    workflow.consume_episode(
                        _episode_summary(episode_result, engine.skill)
                    )
                    ready_count, decisions = workflow.evolve_ready()
                    ready_cluster_counts.append(ready_count)
                    evolution_results.extend(decisions)
                    for item in decisions:
                        save_content_addressed_skill(
                            run_dir / "skill_objects", item.parent, protocol=protocol
                        )
                        if item.candidate is not None:
                            save_content_addressed_skill(
                                run_dir / "skill_objects", item.candidate, protocol=protocol
                            )
        finally:
            env.close()

        frozen = engine.freeze()
        save_skill_artifact(
            run_dir / "frozen_skill.json", frozen.skill, protocol=protocol
        )
        update_audit = None
        if workflow is not None:
            audit_plan = make_rotated_audit_plan(
                manifest,
                rotation_index=rotation_index,
                rollout_seeds=_audit_rollout_seeds(evolution_seed),
            )
            audit_evaluator = _make_audit_evaluator(
                args, run_manifest, audit_plan.coordinates, run_dir, run_id=run_id
            )
            update_audit = run_rotated_update_audit(
                lineage.accepted_snapshots,
                audit_evaluator,
                plan=audit_plan,
                output_path=run_dir / "update_audit.json",
            )
        record = {
            **protocol,
            "acquisition_episode_count": len(acquisition_results),
            "ready_clusters_by_episode": ready_cluster_counts,
            "evolution_decisions": [
                item.decision.accepted for item in evolution_results
            ],
            "frozen_skill_sha256": skill_digest(engine.skill),
            "method_usage": _usage_payload(method_model),
            "executor_usage": _executor_usage_payload(runner),
            "update_reliability": None
            if update_audit is None
            else dict(update_audit.reliability),
        }
        _write_json(run_dir / "run_manifest.json", record)
        run_records.append(record)

    _write_json(
        output_dir / "experiment_manifest.json",
        {
            **_protocol_record(args, config, manifest),
            "experiment_id": experiment_id,
            "run_count": len(run_records),
            "runs": run_records,
        },
    )


def _run_frozen_evaluation(args: argparse.Namespace) -> None:
    output = Path(args.output)
    summary_output = output.with_suffix(".summary.json")
    _require_new_output(output, "event artifact")
    _require_new_output(summary_output, "evaluation summary")
    config = load_config(args.config)
    run_id = f"evaluate_{uuid.uuid4().hex}"
    is_nav = args.env == "eb-nav"
    # EB-Nav ships no train_validation split, so there is no controlled manifest to verify.
    manifest = None if is_nav else _load_verified_manifest(args.manifest)
    artifact = None
    if args.mode == "frozen_skill":
        if not args.skill:
            raise ValueError("--mode frozen_skill requires --skill")
        artifact = load_skill_artifact_record(args.skill, require_frozen=True)
        skill = artifact.skill
    else:
        if args.skill:
            raise ValueError("--skill is only valid with --mode frozen_skill")
        base_skill = initialize_nav_skill() if is_nav else initialize_shared_skill()
        skill = replace(base_skill, frozen=True)
    _audit_evaluation_protocol(args, config, manifest, artifact)
    evaluation_manifest = (
        None if manifest is None else _artifact_evaluation_manifest(manifest, artifact)
    )
    if args.max_episodes is not None and not args.diagnostic:
        raise ValueError("reduced evaluation requires --diagnostic")
    if is_nav and args.stage != "official_test":
        raise ValueError("eb-nav only supports --stage official_test (no train_validation split)")
    if args.stage == "official_test":
        if args.eval_set == "train_validation":
            raise ValueError("official_test requires a stock test subset")
        # The stock test subsets' dataset episode_ids do not align with the id
        # field the manifest reads, so pin-by-id breaks (_select_ordered_episodes
        # rejects most ids). Load the full subset and iterate in stock order,
        # capped by --max-episodes (identical across methods => fair).
        episode_ids = None
        _, evaluation_dataset_hash = _official_episode_ids(
            args.eval_set, env_name=args.env
        )
        evaluation_manifest_hash = None
    else:
        if args.eval_set != "train_validation":
            raise ValueError("acquisition/selection/audit coordinates require train_validation")
        coordinates = evaluation_manifest.coordinates_for(args.stage)
        episode_ids = tuple(item.episode_id for item in coordinates)
        evaluation_dataset_hash = manifest.dataset_sha256
        evaluation_manifest_hash = manifest.digest
    if args.max_episodes is not None and episode_ids is not None:
        episode_ids = episode_ids[: args.max_episodes]
    seed_process_rngs(args.seed)
    env = _create_env(
        args,
        args.eval_set,
        episode_ids=episode_ids,
        exp_name=_habitat_exp_name(
            "evaluate",
            run_id,
            args.stage,
            args.eval_set,
            f"s{args.seed}",
            "no_skill" if args.mode == "no_skill" else skill_digest(skill)[:12],
        ),
        resolution=args.resolution,
    )
    try:
        _seed_env(args, env, args.seed)
        frozen_runtime = None
        if args.mode != "no_skill":
            frozen_runtime = _make_frozen_runtime(skill, env_name=args.env)
        runner = _make_runner(
            args,
            env,
            skill,
            output,
            engine=frozen_runtime,
            goal_grounder=None,
            expected_episode_ids=episode_ids,
            inject_skill=args.mode != "no_skill",
            rollout_seed=args.seed,
            goal_predicate_provider=nav_goal_predicates if is_nav else None,
        )
        # official_test loads the full subset (episode_ids=None) and is capped by
        # --max-episodes; train_validation stages pin a specific id list.
        max_ep = args.max_episodes if episode_ids is None else len(episode_ids)
        results = runner.run(max_episodes=max_ep)
    finally:
        env.close()
    _write_json(
        summary_output,
        {
            "env": args.env,
            "mode": "frozen_evaluation",
            "run_id": run_id,
            "stage": args.stage,
            "evaluation_mode": args.mode,
            "skill_sha256": None if args.mode == "no_skill" else skill_digest(skill),
            "artifact_sha256": None if artifact is None else artifact.artifact_sha256,
            "manifest_sha256": evaluation_manifest_hash,
            "dataset_sha256": evaluation_dataset_hash,
            "teacher_enabled": False,
            "attribution_enabled": False,
            "patching_enabled": False,
            "episodes": len(results),
            "diagnostic": args.diagnostic,
            "diagnostic_subset": args.max_episodes is not None,
            "rollout_seed": args.seed,
            "mean_task_success": _mean(item.task_success for item in results),
            "mean_task_progress": _mean(item.task_progress for item in results),
        },
    )


def _create_env(
    args: argparse.Namespace,
    eval_set: str,
    *,
    episode_ids: tuple[str, ...],
    exp_name: str,
    resolution: int,
):
    if args.env == "eb-nav":
        return create_nav_env(
            eval_set, episode_ids=episode_ids, exp_name=exp_name, resolution=resolution
        )
    return create_habitat_env(
        eval_set, episode_ids=episode_ids, exp_name=exp_name, resolution=resolution
    )


def _seed_env(args: argparse.Namespace, env, seed: int) -> None:
    if args.env == "eb-nav":
        seed_nav_env(env, seed)
    else:
        seed_habitat_env(env, seed)


def _make_frozen_runtime(
    skill: SkillSpec, *, env_name: str = "eb-hab"
) -> VistaSkillEngine:
    if env_name == "eb-nav":
        from vista_skill.action_schema import NavActionSchema

        return VistaSkillEngine(
            skill,
            action_schema=NavActionSchema(),
            evidence_extractor=EvidenceExtractor(
                feedback_strategy=_nav_feedback_strategy
            ),
            ledger=BeliefLedger(),
        )
    return VistaSkillEngine(
        skill,
        evidence_extractor=EvidenceExtractor(),
        ledger=BeliefLedger(),
    )


def _make_method_model(
    args: argparse.Namespace,
    *,
    seed: int,
) -> OpenAICompatibleJsonModel:
    return OpenAICompatibleJsonModel(
        args.method_model,
        base_url=args.method_base_url,
        api_key=args.method_api_key,
        temperature=0.0,
        # Evidence extraction can emit a long observations list; 1024 truncates
        # it mid-string -> json.loads fails. 4096 matches the executor cap and
        # leaves headroom for every teacher purpose (evidence/attribution/patch).
        max_tokens=4096,
        seed=seed,
    )


def _make_engine(
    config: VistaConfig,
    model: OpenAICompatibleJsonModel | None,
    *,
    skill_fault: str | None = None,
    initial_skill: str = "shared",
) -> tuple[VistaSkillEngine, JsonGoalGrounder | None]:
    kwargs = {
        "ledger": BeliefLedger(config.belief),
        "clusterer": EventClusterer(config.recurrence),
        "credit_assigner": CreditAssigner(config=config.attribution),
    }
    grounder = None
    if model is not None:
        kwargs.update(
            {
                "evidence_extractor": EvidenceExtractor(JsonVisualEvidenceProvider(model)),
                "credit_assigner": CreditAssigner(
                    JsonAttributionTeacher(model), config.attribution
                ),
            }
        )
        grounder = JsonGoalGrounder(model)
    if initial_skill == "minimal":
        # Degraded starting point (§4.2.3): no compiled rules and one-line
        # bodies, so the engine predicts from the fixed action schema only.
        skill = minimal_shared_skill()
    elif initial_skill == "empty":
        skill = empty_shared_skill()
    else:
        skill = initialize_shared_skill()
    if skill_fault is not None:
        # Diagnostic only (--diagnostic enforced upstream): a structured fault
        # makes recurrence reachable so evolution can be observed cheaply.
        skill = inject_skill_fault(skill, FaultType(skill_fault))
    return VistaSkillEngine(skill, **kwargs), grounder


def _make_trajectory_workflow(
    args: argparse.Namespace,
    *,
    engine: VistaSkillEngine,
    method_model: OpenAICompatibleJsonModel,
    run_manifest: ExperimentManifest,
    gate_seeds: tuple[int, ...],
    run_dir: Path,
    run_id: str,
    lineage: LineageStore,
    config: VistaConfig,
    protocol: Mapping[str, object],
) -> TrajectoryEvolutionWorkflow:
    """Build the episode-driven evolution driver for a controlled trajectory baseline.

    All three baselines share the same ``--method-model`` teacher (so teacher
    model, token budget, and call count stay matched with the full method). The
    common-gate variants route proposals through the identical VISTA
    ``CandidateGate``; the native variant uses EmbodiSkill body/appendix
    semantics without a paired gate.
    """
    teacher = JsonTrajectoryTeacher(method_model)
    if args.method == "vista_without_vtca":
        frontend = UnconditionalReflectionFrontend(teacher)
    else:
        frontend = EmbodiSkillFrontend(
            teacher, common_gate=(args.method == "embodiskill_star_common_gate")
        )
    if args.method == "embodiskill_star_native":
        updater: CommonGateProposalAdapter | EmbodiSkillNativeUpdater = (
            EmbodiSkillNativeUpdater(
                max_statements_per_field=config.patch.max_statements_per_field
            )
        )
    else:
        paired = _make_paired_evaluator(
            args, run_manifest, gate_seeds, run_dir, config, run_id=run_id
        )
        gate = build_candidate_gate(engine, paired, config)
        updater = CommonGateProposalAdapter(
            JsonBoundedPatchGenerator(method_model),
            gate,
            min_episodes=config.recurrence.min_independent_episodes,
        )
    return TrajectoryEvolutionWorkflow(
        engine,
        frontend=frontend,
        updater=updater,
        lineage=lineage,
        config=config,
        protocol=protocol,
    )


def _episode_summary(result: EpisodeResult, skill: SkillSpec) -> EpisodeSummary:
    """Bridge a rolled-out episode to the trajectory reflection contract."""
    return EpisodeSummary(
        episode_id=result.episode_id,
        instruction=result.instruction,
        success=bool(result.task_success),
        trajectory=result.trajectory,
        current_skill=render_skill(skill),
        failure_reason=result.failure_reason,
    )


def _make_runner(
    args: argparse.Namespace,
    env,
    skill: SkillSpec,
    output: Path,
    *,
    engine: VistaSkillEngine | None,
    goal_grounder: JsonGoalGrounder | None,
    expected_episode_ids: tuple[str, ...],
    inject_skill: bool = True,
    task_coordinates: Sequence = (),
    rollout_seed: int,
    goal_predicate_provider=None,
) -> HabitatRolloutRunner:
    _require_new_output(output, "event artifact")
    planner = _make_planner(
        args,
        env,
        skill,
        engine,
        inject_skill=inject_skill,
        rollout_seed=rollout_seed,
    )
    if goal_predicate_provider is None:
        goal_predicate_provider = (
            None
            if goal_grounder is None
            else lambda instruction, image, actions: goal_grounder.ground(
                instruction, image, actions
            )
        )
    return HabitatRolloutRunner(
        env,
        planner,
        engine,
        JsonlArtifactWriter(output),
        goal_predicate_provider=goal_predicate_provider,
        expected_episode_ids=expected_episode_ids,
        task_coordinates=task_coordinates,
    )


def _make_planner(
    args: argparse.Namespace,
    env,
    skill: SkillSpec,
    engine,
    *,
    inject_skill: bool = True,
    rollout_seed: int,
):
    from embodiedbench.planner import remote_model

    # temperature=0 and a 4096-token cap mirror stock EmbodiedBench's RemoteModel
    # defaults (embodiedbench/planner/remote_model.py), so VISTA rollouts share the
    # EB executor surface; temperature=0 also pins rollouts for reproducibility.
    remote_model.temperature = 0.0
    remote_model.max_completion_tokens = 4096

    if args.env == "eb-nav":
        from embodiedbench.evaluator.config.system_prompts import (
            eb_navigation_system_prompt,
        )
        from embodiedbench.evaluator.config.eb_navigation_example import (
            examples as nav_examples,
        )
        from embodiedbench.planner.nav_planner import EBNavigationPlanner

        planner_class = make_skill_aware_planner(EBNavigationPlanner)
        planner = planner_class(
            args.model_name,
            args.model_type,
            env.language_skill_set,
            eb_navigation_system_prompt,
            nav_examples,
            n_shot=args.n_shots,
            obs_key="head_rgb",
            chat_history=False,
            language_only=False,
            multistep=False,
            tp=args.tp,
        )
    else:
        from embodiedbench.evaluator.config.system_prompts import habitat_system_prompt
        from embodiedbench.planner.vlm_planner import VLMPlanner

        examples = json.loads(
            (
                Path(__file__).parents[3]
                / "EmbodiedBench/embodiedbench/evaluator/config/habitat_examples.json"
            ).read_text(encoding="utf-8")
        )
        planner_class = make_skill_aware_planner(VLMPlanner)
        planner = planner_class(
            args.model_name,
            args.model_type,
            env.language_skill_set,
            habitat_system_prompt,
            examples,
            n_shot=args.n_shots,
            obs_key="head_rgb",
            chat_history=False,
            language_only=False,
            use_feedback=True,
            multistep=0,
            tp=args.tp,
        )
    if args.model_type == "remote":
        configure_planner_inference_seed(planner, rollout_seed)
    if not inject_skill:
        return planner
    if engine is None:
        static_ledger = BeliefLedger()
        planner.configure_vista_prompt(lambda: skill, lambda: static_ledger)
    else:
        planner.configure_vista_prompt(
            lambda: engine.skill,
            lambda: engine.ledger,
            lambda: engine.emphasis_buffer.render(engine.current_step),
        )
    return planner


def _make_paired_evaluator(
    args: argparse.Namespace,
    manifest: ExperimentManifest,
    seeds: tuple[int, ...],
    output_dir: Path,
    config: VistaConfig,
    *,
    run_id: str,
) -> PairedRolloutEvaluator:
    selection = manifest.coordinates_for("selection")
    proxy_budget = config.gate.proxy_episode_budget
    finalist_budget = config.gate.finalist_episode_budget
    coordinates = _paired_selection_coordinates(
        selection,
        seeds,
        proxy_budget=proxy_budget,
        finalist_budget=finalist_budget,
    )
    cache: dict[tuple[str, str, int, str], RolloutScore] = {}

    def rollout(skill: SkillSpec, coordinate: EpisodeCoordinate, stage: str) -> RolloutScore:
        cache_key = (skill_digest(skill), coordinate.episode_id, coordinate.seed, stage)
        if cache_key in cache:
            return cache[cache_key]
        seed_process_rngs(coordinate.seed)
        env = create_habitat_env(
            "train_validation",
            episode_ids=(coordinate.episode_id,),
            exp_name=_habitat_exp_name(
                "gate",
                run_id,
                stage,
                coordinate.episode_id,
                f"s{coordinate.seed}",
                skill_digest(skill)[:12],
            ),
            resolution=args.resolution,
        )
        try:
            seed_habitat_env(env, coordinate.seed)
            artifact = (
                output_dir
                / "gate_rollouts"
                / stage
                / f"{coordinate.episode_id}_s{coordinate.seed}_{skill_digest(skill)[:10]}.jsonl"
            )
            frozen_skill = replace(skill, frozen=True)
            frozen_runtime = VistaSkillEngine(
                frozen_skill,
                evidence_extractor=EvidenceExtractor(),
                ledger=BeliefLedger(),
            )
            runner = _make_runner(
                args,
                env,
                frozen_skill,
                artifact,
                engine=frozen_runtime,
                goal_grounder=None,
                expected_episode_ids=(coordinate.episode_id,),
                task_coordinates=tuple(
                    item
                    for item in manifest.tasks
                    if item.episode_id == coordinate.episode_id
                ),
                rollout_seed=coordinate.seed,
            )
            result = runner.run_episode(expected_episode_id=coordinate.episode_id)
        finally:
            env.close()
        score = RolloutScore(
            score=composite_task_score(
                task_success=result.task_success,
                task_progress=result.task_progress,
                invalid_action_ratio=result.invalid_actions
                / max(1, result.environment_steps),
            ),
            success=bool(result.task_success),
        )
        cache[cache_key] = score
        return score

    return PairedRolloutEvaluator(coordinates, rollout)


def _make_audit_evaluator(
    args: argparse.Namespace,
    manifest: ExperimentManifest,
    coordinates: tuple[EpisodeCoordinate, ...],
    output_dir: Path,
    *,
    run_id: str,
) -> PairedRolloutEvaluator:
    indexed = {item.episode_id: item for item in manifest.tasks}
    cache: dict[tuple[str, str, int], RolloutScore] = {}

    def rollout(skill: SkillSpec, coordinate: EpisodeCoordinate, stage: str) -> RolloutScore:
        if stage != "audit":
            raise ValueError("update-audit evaluator only supports the audit stage")
        cache_key = (skill_digest(skill), coordinate.episode_id, coordinate.seed)
        if cache_key in cache:
            return cache[cache_key]
        seed_process_rngs(coordinate.seed)
        env = create_habitat_env(
            "train_validation",
            episode_ids=(coordinate.episode_id,),
            exp_name=_habitat_exp_name(
                "update_audit",
                run_id,
                coordinate.episode_id,
                f"s{coordinate.seed}",
                skill_digest(skill)[:12],
            ),
            resolution=args.resolution,
        )
        try:
            seed_habitat_env(env, coordinate.seed)
            artifact = (
                output_dir
                / "update_audit_rollouts"
                / coordinate.episode_id
                / f"s{coordinate.seed}_{skill_digest(skill)}.jsonl"
            )
            frozen_skill = replace(skill, frozen=True)
            runtime = VistaSkillEngine(
                frozen_skill,
                evidence_extractor=EvidenceExtractor(),
                ledger=BeliefLedger(),
            )
            runner = _make_runner(
                args,
                env,
                frozen_skill,
                artifact,
                engine=runtime,
                goal_grounder=None,
                expected_episode_ids=(coordinate.episode_id,),
                task_coordinates=(indexed[coordinate.episode_id],),
                rollout_seed=coordinate.seed,
            )
            result = runner.run_episode(expected_episode_id=coordinate.episode_id)
        finally:
            env.close()
        score = RolloutScore(
            score=composite_task_score(
                task_success=result.task_success,
                task_progress=result.task_progress,
                invalid_action_ratio=result.invalid_actions
                / max(1, result.environment_steps),
            ),
            success=bool(result.task_success),
        )
        cache[cache_key] = score
        return score

    return PairedRolloutEvaluator({"audit": coordinates}, rollout)


def _paired_selection_coordinates(
    selection,
    seeds: tuple[int, ...],
    *,
    proxy_budget: int,
    finalist_budget: int,
) -> dict[str, tuple[EpisodeCoordinate, ...]]:
    if not seeds:
        raise ValueError("paired selection requires at least one seed")
    if len(selection) < 2:
        raise ValueError("paired selection requires disjoint proxy/finalist task pools")
    proxy_task_count = min(proxy_budget, len(selection) // 2)
    proxy_tasks = selection[:proxy_task_count]
    finalist_tasks = selection[proxy_task_count:]
    proxy = tuple(
        EpisodeCoordinate(item.episode_id, seeds[0], item.subgroup)
        for item in proxy_tasks
    )[:proxy_budget]
    finalist = tuple(
        EpisodeCoordinate(item.episode_id, seed, item.subgroup)
        for item in finalist_tasks
        for seed in seeds
    )[:finalist_budget]
    if len(proxy) < proxy_budget or len(finalist) < finalist_budget:
        raise ValueError("selection tasks and seeds do not cover configured paired budgets")
    if {item.episode_id for item in proxy} & {item.episode_id for item in finalist}:
        raise RuntimeError("proxy and finalist task pools must be disjoint")
    return {"proxy": proxy, "finalist": finalist}


def _load_verified_manifest(path: str) -> ExperimentManifest:
    manifest = load_experiment_manifest(path)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("manifest_sha256") != manifest.digest:
        raise ValueError("experiment manifest digest mismatch")
    dataset = (
        Path(__file__).parents[3]
        / "EmbodiedBench/embodiedbench/envs/eb_habitat/datasets"
        / manifest.dataset
    )
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != manifest.dataset_sha256:
        raise ValueError("experiment dataset digest mismatch")
    return manifest


def _official_episode_ids(
    eval_set: str, *, env_name: str = "eb-hab"
) -> tuple[tuple[str, ...], str]:
    if env_name == "eb-nav":
        allowed = {
            "base",
            "common_sense",
            "complex_instruction",
            "visual_appearance",
            "long_horizon",
        }
        if eval_set not in allowed:
            raise ValueError(f"unsupported official EB-Nav test subset: {eval_set}")
        dataset = (
            Path(__file__).parents[3]
            / "EmbodiedBench/embodiedbench/envs/eb_navigation/datasets"
            / f"{eval_set}.json"
        )
        tasks = json.loads(dataset.read_text(encoding="utf-8"))["tasks"]
        return (
            tuple(f"nav_{index}" for index in range(len(tasks))),
            hashlib.sha256(dataset.read_bytes()).hexdigest(),
        )
    allowed = {
        "base",
        "common_sense",
        "complex_instruction",
        "spatial_relationship",
        "visual_appearance",
        "long_horizon",
    }
    if eval_set not in allowed:
        raise ValueError(f"unsupported official EB-Hab test subset: {eval_set}")
    dataset = (
        Path(__file__).parents[3]
        / "EmbodiedBench/embodiedbench/envs/eb_habitat/datasets"
        / f"{eval_set}.pickle"
    )
    from vista_skill.integrations.embodiedbench.manifest import build_manifest

    payload = build_manifest(dataset, split_sizes=None)
    return (
        tuple(str(item["episode_id"]) for item in payload["tasks"]),
        str(payload["dataset_sha256"]),
    )


def _protocol_record(
    args: argparse.Namespace,
    config: VistaConfig,
    manifest: ExperimentManifest,
) -> dict[str, object]:
    return {
        "protocol": config.raw["protocol"],
        "config_sha256": config.digest,
        "manifest_sha256": manifest.digest,
        "method": args.method,
        "skill_fault": getattr(args, "skill_fault", None),
        "initial_skill": getattr(args, "initial_skill", "shared"),
        "executor_model": args.model_name,
        "executor_model_type": args.model_type,
        "tensor_parallel": args.tp,
        "executor_temperature": float(config.raw["executor"]["temperature"]),
        "max_completion_tokens": int(
            config.raw["executor"]["max_completion_tokens"]
        ),
        "method_model": args.method_model,
        "resolution": args.resolution,
        "n_shots": args.n_shots,
        "frozen": True,
        "evolution_seeds": _parse_seeds(args.evolution_seeds),
        "diagnostic": args.diagnostic,
        "env": args.env,
        "rng_seed_policy": (
            "python+numpy+torch+ai2thor+openai_request"
            if args.env == "eb-nav"
            else "python+numpy+torch+habitat+openai_request"
        ),
    }


def _audit_evaluation_protocol(
    args: argparse.Namespace,
    config: VistaConfig,
    manifest: ExperimentManifest,
    artifact: SkillArtifact | None,
) -> None:
    """Fail closed when a controlled evaluation drifts from its frozen protocol."""

    if args.diagnostic:
        return
    if args.env == "eb-nav":
        # EB-Nav has no controlled evolution protocol; only guard artifact frozenness.
        if artifact is not None and not artifact.skill.frozen:
            raise ValueError("artifact skill is not frozen")
        return
    mismatches = []
    configured_manifest = Path(str(config.raw["task_manifest"])).resolve()
    if Path(args.manifest).resolve() != configured_manifest:
        mismatches.append("manifest path differs from config")
    executor = config.raw["executor"]
    expected_model = str(executor["model"])
    if not str(args.model_name).endswith(expected_model):
        mismatches.append("executor model differs from config")
    if args.n_shots != int(executor["n_shots"]):
        mismatches.append("n-shot count differs from config")
    if args.resolution != int(config.raw["environment"]["resolution"]):
        mismatches.append("image resolution differs from config")
    if args.model_type != str(executor["model_type"]):
        mismatches.append("executor model type differs from config")
    if args.tp != int(executor["tensor_parallel"]):
        mismatches.append("tensor parallel setting differs from config")
    if artifact is not None:
        expected_protocol = {
            "config_sha256": config.digest,
            "manifest_sha256": manifest.digest,
            "executor_model": args.model_name,
            "executor_model_type": args.model_type,
            "tensor_parallel": args.tp,
            "executor_temperature": float(executor["temperature"]),
            "max_completion_tokens": int(executor["max_completion_tokens"]),
            "n_shots": args.n_shots,
            "resolution": args.resolution,
            "frozen": True,
            "diagnostic": False,
            "acquisition_episode_budget": int(
                config.raw["environment"]["acquisition_tasks"]
            ),
            "rng_seed_policy": "python+numpy+torch+habitat+openai_request",
        }
        for key, expected in expected_protocol.items():
            if artifact.protocol.get(key) != expected:
                mismatches.append(f"artifact {key} mismatch")
        if not artifact.skill.frozen:
            mismatches.append("artifact skill is not frozen")
    if mismatches:
        raise ValueError(
            "controlled evaluation protocol mismatch: " + "; ".join(mismatches)
        )


def _artifact_evaluation_manifest(
    manifest: ExperimentManifest,
    artifact: SkillArtifact | None,
) -> ExperimentManifest:
    if artifact is None:
        return manifest
    raw_index = artifact.protocol.get("split_rotation_index", 0)
    if isinstance(raw_index, bool) or not isinstance(raw_index, int):
        raise ValueError("artifact split_rotation_index must be an integer")
    rotated = manifest.rotate_split(raw_index)
    expected_split_hash = artifact.protocol.get("split_sha256")
    if expected_split_hash is not None and expected_split_hash != rotated.split.digest():
        raise ValueError("artifact split_sha256 mismatch")
    return rotated


def _validate_controlled_executor(args: argparse.Namespace, config: VistaConfig) -> None:
    executor = config.raw["executor"]
    expected_model = str(executor["model"])
    if not str(args.model_name).endswith(expected_model):
        raise ValueError(
            f"controlled protocol requires executor {expected_model}, got {args.model_name}"
        )
    environment = config.raw["environment"]
    if args.n_shots != int(executor["n_shots"]):
        raise ValueError("controlled protocol n-shot count differs from config")
    if args.resolution != int(environment["resolution"]):
        raise ValueError("controlled protocol image resolution differs from config")
    if args.model_type != str(executor["model_type"]):
        raise ValueError("controlled protocol executor model type differs from config")
    if args.tp != int(executor["tensor_parallel"]):
        raise ValueError("controlled protocol tensor parallel setting differs from config")
    configured_seeds = tuple(int(item) for item in config.raw["evolution_seeds"])
    if _parse_seeds(args.evolution_seeds) != configured_seeds:
        raise ValueError("controlled protocol evolution seeds differ from config")


def _usage_payload(model: OpenAICompatibleJsonModel | None):
    if model is None:
        return None
    return {purpose: vars(counter) for purpose, counter in model.usage.items()}


def _executor_usage_payload(runner) -> dict[str, int] | None:
    """Acquisition-phase executor calls/tokens, captured by the seed wrapper.

    Returns ``None`` for non-remote executors (``local``/``custom``), which do
    not route through the OpenAI-compatible seed wrapper; those backends cannot
    be seed-controlled in a controlled run either (see docs/implementation.md).
    """
    planner = getattr(runner, "planner", None)
    usage = getattr(planner, "_vista_executor_usage", None)
    return None if usage is None else dict(usage)


def _parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("evolution seeds must be a non-empty unique list")
    return values


def _gate_rollout_seeds(evolution_seed: int) -> tuple[int, int, int]:
    """Derive three within-run rollout seeds without coupling independent runs."""
    base = int(evolution_seed) * 1000
    return base, base + 1, base + 2


def _audit_rollout_seeds(evolution_seed: int) -> tuple[int, int, int]:
    base = int(evolution_seed) * 1000 + 101
    return base, base + 1, base + 2


def _mean(values) -> float:
    items = tuple(values)
    return sum(items) / len(items) if items else 0.0


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_new_output(path: Path, label: str) -> None:
    if path.exists():
        raise FileExistsError(f"{label} already exists: {path}")


def _habitat_exp_name(*parts: object) -> str:
    values = [re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(item)) for item in parts]
    return "vista_skill/" + "/".join(values)


if __name__ == "__main__":
    main()
