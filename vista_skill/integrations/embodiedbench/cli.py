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
from vista_skill.action_schema import SkillOnlyActionSchema
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
from vista_skill.evidence import (
    EvidenceExtractor,
    EvidenceExtractorConfig,
    _nav_feedback_strategy,
)
from vista_skill.evidence_guard import (
    EvidenceGuardConfig,
    EvidenceReliabilityGuard,
    GuardMode,
)
from vista_skill.integrations.embodiedbench.environment import (
    create_habitat_env,
    create_nav_env,
    nav_goal_predicates,
    seed_habitat_env,
    seed_nav_env,
    seed_process_rngs,
)
from vista_skill.integrations.embodiedbench.planner import (
    ExecutorUsageTracker,
    configure_planner_inference_seed,
    make_skill_aware_planner,
)
from vista_skill.integrations.embodiedbench.runner import (
    EpisodeResult,
    HabitatRolloutRunner,
)
from vista_skill.integrations.embodiedbench.state_oracle import HabitatStateOracle
from vista_skill.integrations.embodiedbench.task_semantics import (
    load_habitat_task_semantic_tags,
)
from vista_skill.integrity import (
    artifact_contamination_reasons,
    load_evaluation_data_policy,
)
from vista_skill.lineage import LineageStore
from vista_skill.meta_skills import EvolutionMetaSkill, frozen_meta_skills
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
from vista_skill.schemas import SkillField, SkillSpec
from vista_skill.skills import (
    empty_shared_skill,
    interface_only_shared_skill,
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
from vista_skill.temporal import TemporalRuleMonitor
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


class _NoRolloutPairedEvaluator:
    """Diagnostic gate stop after static and cached-transition validation."""

    def evaluate(self, parent, candidate, *, stage: str, episode_budget: int):
        return ()


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
    experiment.add_argument("--config", default="configs/vista_phase5_hab.json")
    experiment.add_argument("--output-dir", default="running/Phase5/vista_skill/full")
    experiment.add_argument("--evolution-seeds", default="0,1,2")
    experiment.add_argument("--max-acquisition-episodes", type=int)
    experiment.add_argument(
        "--initial-skill",
        choices=("shared", "minimal", "empty", "interface"),
        default=None,
        help="Initial Skill variant (diagnostic init-sensitivity regime): "
        "'shared' = S0; 'minimal'/'empty' = degraded starting points for "
        "testing whether evolution can grow or recover skills. The config's "
        "declared variant is used when omitted; overrides require --diagnostic.",
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
    experiment.add_argument(
        "--transition-only-gate",
        action="store_true",
        help=(
            "Diagnostic only: materialize candidates and run static/cached-transition "
            "checks, but stop before paired rollout selection."
        ),
    )
    experiment.add_argument(
        "--candidate-field",
        choices=tuple(item.value for item in SkillField),
        help=(
            "Diagnostic only: materialize/evaluate ready candidates from one "
            "attributed field."
        ),
    )
    experiment.add_argument(
        "--skip-update-audit",
        action="store_true",
        help=(
            "Diagnostic only: stop after acquisition and candidate selection "
            "without the independent post-hoc update audit."
        ),
    )
    experiment.add_argument(
        "--state-oracle-labels",
        action="store_true",
        help=(
            "Write evaluation-only Habitat PDDL predicate labels next to transitions. "
            "Requires --diagnostic and is never exposed to the executor or method model."
        ),
    )
    experiment.add_argument(
        "--evidence-guard",
        choices=("none", "threshold", "strict", "authority_aware"),
        default=None,
        help=(
            "Evidence-fusion regime. Uses the config declaration when omitted; "
            "a command-line override requires --diagnostic."
        ),
    )
    experiment.add_argument("--guard-min-visual-confidence", type=float, default=0.75)
    experiment.add_argument("--guard-min-visual-coverage", type=float, default=0.50)
    experiment.add_argument(
        "--meta-skills",
        choices=("none", "frozen_v1"),
        default="none",
        help="Phase3C frozen Meta-Skill diagnostic arm; requires --diagnostic and --method full.",
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
    evaluate.add_argument("--config", default="configs/vista_phase5_hab.json")
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
    evaluate.add_argument("--output", default="running/Phase5/vista_skill/frozen_audit/events.jsonl")
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--max-episodes", type=int)
    evaluate.add_argument(
        "--meta-skills",
        choices=("none", "frozen_v1"),
        default="none",
        help="Add only the frozen observation-and-recovery Skill to the executor prompt.",
    )
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
    configured_initial = str(
        config.raw.get("initialization", {}).get("variant", "shared")
    )
    if args.initial_skill is None:
        args.initial_skill = configured_initial
    elif args.initial_skill != configured_initial and not args.diagnostic:
        raise ValueError("--initial-skill override differs from the controlled config")
    configured_guard = str(config.raw.get("evidence", {}).get("guard", "none"))
    if args.evidence_guard is None:
        args.evidence_guard = configured_guard
    elif args.evidence_guard != configured_guard and not args.diagnostic:
        raise ValueError("--evidence-guard override differs from the controlled config")
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
    if args.transition_only_gate and not args.diagnostic:
        raise ValueError("--transition-only-gate requires --diagnostic")
    if args.transition_only_gate and args.method != "full":
        raise ValueError("--transition-only-gate currently requires --method full")
    if args.candidate_field is not None and not args.diagnostic:
        raise ValueError("--candidate-field requires --diagnostic")
    if args.candidate_field is not None and args.method != "full":
        raise ValueError("--candidate-field currently requires --method full")
    if args.skip_update_audit and not args.diagnostic:
        raise ValueError("--skip-update-audit requires --diagnostic")
    if args.skill_fault and not args.diagnostic:
        raise ValueError("--skill-fault is a diagnostic deviation and requires --diagnostic")
    if args.initial_skill != "shared" and args.skill_fault:
        raise ValueError("--initial-skill and --skill-fault are exclusive regime knobs")
    if args.state_oracle_labels and not args.diagnostic:
        raise ValueError("--state-oracle-labels is evaluation-only and requires --diagnostic")
    if args.meta_skills != "none" and not args.diagnostic:
        raise ValueError("--meta-skills is a Phase3C deviation and requires --diagnostic")
    if args.meta_skills != "none" and args.method != "full":
        raise ValueError("--meta-skills currently requires --method full")
    for name in ("guard_min_visual_confidence", "guard_min_visual_coverage"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    output_dir = Path(args.output_dir)
    _require_new_output(output_dir, "experiment output directory")
    experiment_id = uuid.uuid4().hex
    seeds = _parse_seeds(args.evolution_seeds)
    run_records = []
    for rotation_index, evolution_seed in enumerate(seeds):
        run_manifest = manifest.rotate_split(rotation_index)
        run_dir = output_dir / f"seed_{evolution_seed}"
        run_id = f"{experiment_id}_seed_{evolution_seed}"
        executor_usage = ExecutorUsageTracker()
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
            evidence_guard=args.evidence_guard,
            guard_min_visual_confidence=args.guard_min_visual_confidence,
            guard_min_visual_coverage=args.guard_min_visual_coverage,
            attribution_meta_skill=(
                None
                if args.meta_skills == "none"
                else frozen_meta_skills().attribute_and_scope
            ),
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
                paired = (
                    _NoRolloutPairedEvaluator()
                    if args.transition_only_gate
                    else _make_paired_evaluator(
                        args,
                        run_manifest,
                        gate_seeds,
                        run_dir,
                        config,
                        run_id=run_id,
                        usage_tracker=executor_usage,
                    )
                )
                lineage = LineageStore(run_dir / "lineage.jsonl")
                workflow = EvolutionWorkflow(
                    engine,
                    generator=JsonBoundedPatchGenerator(
                        method_model,
                        None
                        if args.meta_skills == "none"
                        else frozen_meta_skills().patch_and_test,
                    ),
                    paired_evaluator=paired,
                    lineage=lineage,
                    config=config,
                    protocol=protocol,
                    allowed_fields=(
                        None
                        if args.candidate_field is None
                        else (SkillField(args.candidate_field),)
                    ),
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
                    usage_tracker=executor_usage,
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
                max_completion_tokens=int(
                    config.raw["executor"]["max_completion_tokens"]
                ),
                usage_tracker=executor_usage,
                usage_phase="acquisition",
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
        if (
            workflow is not None
            and not args.transition_only_gate
            and not args.skip_update_audit
        ):
            audit_plan = make_rotated_audit_plan(
                manifest,
                rotation_index=rotation_index,
                rollout_seeds=_audit_rollout_seeds(evolution_seed),
            )
            audit_evaluator = _make_audit_evaluator(
                args,
                run_manifest,
                audit_plan.coordinates,
                run_dir,
                config,
                run_id=run_id,
                usage_tracker=executor_usage,
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
            "evolution_dispositions": [
                item.decision.disposition for item in evolution_results
            ],
            "frozen_skill_sha256": skill_digest(engine.skill),
            "method_usage": _usage_payload(method_model),
            "executor_usage": executor_usage.payload(),
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
    if args.meta_skills != "none" and not args.diagnostic:
        raise ValueError("--meta-skills is a Phase3C deviation and requires --diagnostic")
    if args.meta_skills != "none" and args.mode == "no_skill":
        raise ValueError("--meta-skills requires a Skill-injected evaluation mode")
    run_id = f"evaluate_{uuid.uuid4().hex}"
    executor_usage = ExecutorUsageTracker()
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
    contamination_reasons = (
        ()
        if artifact is None
        else artifact_contamination_reasons(artifact.skill, artifact.protocol)
    )
    data_policy = load_evaluation_data_policy()
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
            max_completion_tokens=int(
                config.raw["executor"]["max_completion_tokens"]
            ),
            usage_tracker=executor_usage,
            usage_phase="frozen_evaluation",
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
            "meta_skills": args.meta_skills,
            "meta_skill_sha256": (
                None if args.meta_skills == "none" else frozen_meta_skills().sha256
            ),
            "episodes": len(results),
            "diagnostic": args.diagnostic,
            "diagnostic_subset": args.max_episodes is not None,
            "evaluation_data_policy": data_policy.policy_id,
            "evaluation_data_policy_sha256": data_policy.digest,
            "artifact_contamination_reasons": contamination_reasons,
            "claim_eligible": bool(
                args.stage == "official_test"
                and not args.diagnostic
                and args.max_episodes is None
                and not contamination_reasons
            ),
            "rollout_seed": args.seed,
            "mean_task_success": _mean(item.task_success for item in results),
            "mean_task_progress": _mean(item.task_progress for item in results),
            "executor_usage": executor_usage.payload(),
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
    evidence_guard: str = "none",
    guard_min_visual_confidence: float = 0.75,
    guard_min_visual_coverage: float = 0.50,
    attribution_meta_skill: EvolutionMetaSkill | None = None,
) -> tuple[VistaSkillEngine, JsonGoalGrounder | None]:
    kwargs = {
        "ledger": BeliefLedger(config.belief),
        "clusterer": EventClusterer(config.recurrence),
        "credit_assigner": CreditAssigner(config=config.attribution),
    }
    if initial_skill == "interface":
        kwargs["action_schema"] = SkillOnlyActionSchema()
    grounder = None
    if model is not None:
        guard = None
        include_feedback = True
        evidence_settings = config.raw.get("evidence", {})
        evidence_config = EvidenceExtractorConfig(
            visual_action_types=tuple(
                str(item)
                for item in evidence_settings.get("visual_action_types", ("place",))
            ),
            visual_on_unresolved_goals=bool(
                evidence_settings.get("visual_on_unresolved_goals", True)
            ),
            min_visual_confidence=guard_min_visual_confidence,
            min_visual_coverage=guard_min_visual_coverage,
        )
        if evidence_guard == "threshold":
            pass
        elif evidence_guard in {"strict", "authority_aware"}:
            # Guard v2 uses the same single feedback-conditioned VLM call as
            # the current method, then audits it against independently parsed
            # structured feedback and action-local reliability rules.
            include_feedback = True
            guard = EvidenceReliabilityGuard(
                EvidenceGuardConfig(
                    mode=(
                        GuardMode.STRICT
                        if evidence_guard == "strict"
                        else GuardMode.AUTHORITY_AWARE
                    ),
                    min_visual_confidence=guard_min_visual_confidence,
                    min_visual_coverage=guard_min_visual_coverage,
                )
            )
        kwargs.update(
            {
                "evidence_extractor": EvidenceExtractor(
                    JsonVisualEvidenceProvider(
                        model, include_feedback=include_feedback
                    ),
                    config=evidence_config,
                    guard=guard,
                ),
                "credit_assigner": CreditAssigner(
                    JsonAttributionTeacher(model, attribution_meta_skill),
                    config.attribution,
                ),
            }
        )
        grounder = JsonGoalGrounder(model)
    if initial_skill == "interface":
        skill = interface_only_shared_skill()
    elif initial_skill == "minimal":
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
    usage_tracker: ExecutorUsageTracker,
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
            args,
            run_manifest,
            gate_seeds,
            run_dir,
            config,
            run_id=run_id,
            usage_tracker=usage_tracker,
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
    expected_episode_ids: tuple[str, ...] | None,
    inject_skill: bool = True,
    task_coordinates: Sequence = (),
    rollout_seed: int,
    goal_predicate_provider=None,
    max_completion_tokens: int = 4096,
    usage_tracker: ExecutorUsageTracker | None = None,
    usage_phase: str = "executor",
) -> HabitatRolloutRunner:
    _require_new_output(output, "event artifact")
    planner = _make_planner(
        args,
        env,
        skill,
        engine,
        inject_skill=inject_skill,
        rollout_seed=rollout_seed,
        max_completion_tokens=max_completion_tokens,
        usage_tracker=usage_tracker,
        usage_phase=usage_phase,
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
        state_oracle=(
            HabitatStateOracle()
            if getattr(args, "state_oracle_labels", False)
            else None
        ),
    )


def _make_planner(
    args: argparse.Namespace,
    env,
    skill: SkillSpec,
    engine,
    *,
    inject_skill: bool = True,
    rollout_seed: int,
    max_completion_tokens: int = 4096,
    usage_tracker: ExecutorUsageTracker | None = None,
    usage_phase: str = "executor",
):
    from embodiedbench.planner import remote_model

    # Temperature 0 and the configured completion cap are part of the frozen
    # executor surface.  The release configs use stock EmbodiedBench's 4096-token
    # RemoteModel default; reading the cap from config keeps the recorded protocol
    # and the actual request from silently diverging.
    remote_model.temperature = 0.0
    remote_model.max_completion_tokens = max_completion_tokens

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
        configure_planner_inference_seed(
            planner,
            rollout_seed,
            usage_tracker=usage_tracker,
            usage_phase=usage_phase,
        )
    if not inject_skill:
        return planner
    observation_meta_skill = (
        ""
        if getattr(args, "meta_skills", "none") == "none"
        else frozen_meta_skills().observe_and_recover.instruction
    )
    temporal_monitor = TemporalRuleMonitor()
    temporal_monitor.start_episode(skill if engine is None else engine.skill)
    planner._vista_temporal_monitor = temporal_monitor
    if engine is None:
        static_ledger = BeliefLedger()
        planner.configure_vista_prompt(
            lambda: skill,
            lambda: static_ledger,
            observation_meta_skill_provider=lambda: observation_meta_skill,
            temporal_rule_provider=temporal_monitor.render,
        )
    else:
        planner.configure_vista_prompt(
            lambda: engine.skill,
            lambda: engine.ledger,
            lambda: engine.emphasis_buffer.render(engine.current_step),
            lambda: observation_meta_skill,
            temporal_monitor.render,
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
    usage_tracker: ExecutorUsageTracker,
) -> PairedRolloutEvaluator:
    selection = manifest.coordinates_for("selection")
    proxy_budget = config.gate.proxy_episode_budget
    finalist_budget = config.gate.finalist_episode_budget
    dataset = (
        Path(__file__).parents[3]
        / "EmbodiedBench/embodiedbench/envs/eb_habitat/datasets"
        / manifest.dataset
    )
    semantic_tags = load_habitat_task_semantic_tags(dataset)
    coordinates = _paired_selection_coordinates(
        selection,
        seeds,
        proxy_budget=proxy_budget,
        finalist_budget=finalist_budget,
        proxy_rollout_repeats=config.gate.proxy_rollout_repeats,
        semantic_tags=semantic_tags,
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
                max_completion_tokens=int(
                    config.raw["executor"]["max_completion_tokens"]
                ),
                usage_tracker=usage_tracker,
                usage_phase=f"gate_{stage}",
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
    config: VistaConfig,
    *,
    run_id: str,
    usage_tracker: ExecutorUsageTracker,
) -> PairedRolloutEvaluator:
    indexed = {item.episode_id: item for item in manifest.tasks}
    cache: dict[tuple[str, str, int], RolloutScore] = {}

    def rollout(skill: SkillSpec, coordinate: EpisodeCoordinate, stage: str) -> RolloutScore:
        if stage != "audit":
            raise ValueError("update-audit evaluator only supports the audit stage")
        cache_key = (skill_digest(skill), coordinate.episode_id, coordinate.seed)
        if cache_key in cache:
            return cache[cache_key]
        artifact = (
            output_dir
            / "update_audit_rollouts"
            / coordinate.episode_id
            / f"s{coordinate.seed}_{skill_digest(skill)}.jsonl"
        )
        completed = _load_completed_audit_rollout(
            artifact, expected_episode_id=coordinate.episode_id
        )
        if completed is not None:
            cache[cache_key] = completed
            return completed
        artifact = _next_audit_resume_artifact(artifact)
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
                max_completion_tokens=int(
                    config.raw["executor"]["max_completion_tokens"]
                ),
                usage_tracker=usage_tracker,
                usage_phase="update_audit",
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


def _load_completed_audit_rollout(
    artifact: Path, *, expected_episode_id: str
) -> RolloutScore | None:
    """Load a complete audit coordinate from disk after an interrupted run."""
    candidates = (artifact, *sorted(artifact.parent.glob(f"{artifact.stem}.resume*.jsonl")))
    for candidate in reversed(candidates):
        if not candidate.is_file():
            continue
        episode_result = None
        try:
            with candidate.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("event_type") == "episode_result":
                        episode_result = record.get("payload")
        except (OSError, json.JSONDecodeError):
            continue
        if episode_result is None:
            continue
        if str(episode_result.get("episode_id")) != str(expected_episode_id):
            raise ValueError(f"cached audit episode mismatch: {candidate}")
        environment_steps = int(episode_result["environment_steps"])
        invalid_actions = int(episode_result["invalid_actions"])
        return RolloutScore(
            score=composite_task_score(
                task_success=float(episode_result["task_success"]),
                task_progress=float(episode_result["task_progress"]),
                invalid_action_ratio=invalid_actions / max(1, environment_steps),
            ),
            success=bool(float(episode_result["task_success"])),
        )
    return None


def _next_audit_resume_artifact(artifact: Path) -> Path:
    if not artifact.exists():
        return artifact
    index = 1
    while True:
        candidate = artifact.with_name(f"{artifact.stem}.resume{index}.jsonl")
        if not candidate.exists():
            return candidate
        index += 1


def _paired_selection_coordinates(
    selection,
    seeds: tuple[int, ...],
    *,
    proxy_budget: int,
    finalist_budget: int,
    proxy_rollout_repeats: int = 1,
    semantic_tags: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, tuple[EpisodeCoordinate, ...]]:
    if not seeds:
        raise ValueError("paired selection requires at least one seed")
    if len(selection) < 2:
        raise ValueError("paired selection requires disjoint proxy/finalist task pools")
    if proxy_rollout_repeats < 1 or proxy_rollout_repeats > len(seeds):
        raise ValueError("proxy rollout repeats must be covered by registered seeds")
    if proxy_budget % proxy_rollout_repeats:
        raise ValueError("proxy budget must contain complete repeated-task blocks")
    proxy_task_count = min(
        proxy_budget // proxy_rollout_repeats,
        len(selection) // 2,
    )
    proxy_tasks = selection[:proxy_task_count]
    finalist_tasks = selection[proxy_task_count:]
    semantic_tags = semantic_tags or {}
    proxy = tuple(
        EpisodeCoordinate(
            item.episode_id,
            seed,
            item.subgroup,
            semantic_tags.get(item.episode_id, ()),
        )
        for item in proxy_tasks
        for seed in seeds[:proxy_rollout_repeats]
    )[:proxy_budget]
    finalist = tuple(
        EpisodeCoordinate(
            item.episode_id,
            seed,
            item.subgroup,
            semantic_tags.get(item.episode_id, ()),
        )
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
    data_policy = load_evaluation_data_policy()
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
        "transition_only_gate": bool(
            getattr(args, "transition_only_gate", False)
        ),
        "candidate_field": getattr(args, "candidate_field", None),
        "skip_update_audit": bool(
            getattr(args, "skip_update_audit", False)
        ),
        "evaluation_data_policy": data_policy.policy_id,
        "evaluation_data_policy_sha256": data_policy.digest,
        "state_oracle_labels": bool(
            getattr(args, "state_oracle_labels", False)
        ),
        "evidence_guard": getattr(args, "evidence_guard", "none"),
        "guard_min_visual_confidence": float(
            getattr(args, "guard_min_visual_confidence", 0.75)
        ),
        "guard_min_visual_coverage": float(
            getattr(args, "guard_min_visual_coverage", 0.50)
        ),
        "meta_skills": getattr(args, "meta_skills", "none"),
        "meta_skill_sha256": (
            None
            if getattr(args, "meta_skills", "none") == "none"
            else frozen_meta_skills().sha256
        ),
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
    manifest: ExperimentManifest | None,
    artifact: SkillArtifact | None,
) -> None:
    """Fail closed when a controlled evaluation drifts from its frozen protocol."""

    if args.diagnostic:
        return
    mismatches = []
    data_policy = load_evaluation_data_policy()
    configured_environment = str(config.raw["environment"]["name"])
    if args.env != configured_environment:
        mismatches.append("environment differs from config")
    if args.env == "eb-hab":
        if manifest is None:
            mismatches.append("EB-Habitat controlled evaluation requires a manifest")
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
            "executor_model": args.model_name,
            "executor_model_type": args.model_type,
            "tensor_parallel": args.tp,
            "executor_temperature": float(executor["temperature"]),
            "max_completion_tokens": int(executor["max_completion_tokens"]),
            "n_shots": args.n_shots,
            "resolution": args.resolution,
            "frozen": True,
            "diagnostic": False,
            "env": args.env,
            "evaluation_data_policy": data_policy.policy_id,
            "evaluation_data_policy_sha256": data_policy.digest,
            "rng_seed_policy": (
                "python+numpy+torch+ai2thor+openai_request"
                if args.env == "eb-nav"
                else "python+numpy+torch+habitat+openai_request"
            ),
        }
        if manifest is not None:
            expected_protocol["manifest_sha256"] = manifest.digest
        acquisition_budget = config.raw["environment"].get("acquisition_tasks")
        if acquisition_budget is not None:
            expected_protocol["acquisition_episode_budget"] = int(acquisition_budget)
        for key, expected in expected_protocol.items():
            if artifact.protocol.get(key) != expected:
                mismatches.append(f"artifact {key} mismatch")
        if not artifact.skill.frozen:
            mismatches.append("artifact skill is not frozen")
        contamination = artifact_contamination_reasons(
            artifact.skill, artifact.protocol
        )
        if contamination:
            mismatches.append(
                "artifact is quarantined: " + "; ".join(contamination)
            )
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
    configured_environment = str(config.raw["environment"]["name"])
    if args.env != configured_environment:
        raise ValueError(
            "controlled protocol environment differs from config"
        )
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
    requested_seeds = _parse_seeds(args.evolution_seeds)
    if args.diagnostic:
        if any(seed not in configured_seeds for seed in requested_seeds):
            raise ValueError(
                "diagnostic evolution seeds must be a subset of the controlled config"
            )
    elif requested_seeds != configured_seeds:
        raise ValueError("controlled protocol evolution seeds differ from config")


def _usage_payload(model: OpenAICompatibleJsonModel | None):
    if model is None:
        return None
    return {purpose: vars(counter) for purpose, counter in model.usage.items()}


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
