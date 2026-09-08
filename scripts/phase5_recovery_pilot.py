"""Bounded, resumable, two-endpoint natural recovery discovery diagnostic."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request
from types import SimpleNamespace

from vista_skill.action_schema import SkillOnlyActionSchema
from vista_skill.artifacts import JsonlArtifactWriter
from vista_skill.attribution import CreditAssigner
from vista_skill.config import load_config
from vista_skill.evidence import EvidenceExtractor, EvidenceExtractorConfig
from vista_skill.evidence_guard import EvidenceGuardConfig, EvidenceReliabilityGuard, GuardMode
from vista_skill.evolution import BoundedPatchApplier, CandidateGate, PairedEpisodeScore
from vista_skill.integrations.embodiedbench.cli import _make_runner
from vista_skill.integrations.embodiedbench.environment import (
    create_habitat_development_env, seed_habitat_env, seed_process_rngs,
)
from vista_skill.integrations.embodiedbench.planner import ExecutorUsageTracker
from vista_skill.models import JsonVisualEvidenceProvider, OpenAICompatibleJsonModel
from vista_skill.pipeline import VistaSkillEngine
from vista_skill.recovery import _near_evidence, propose_recovery
from vista_skill.schemas import dataclass_to_dict
from vista_skill.skills import interface_only_shared_skill, load_skill_artifact_record, save_skill_artifact, skill_digest
from scripts.phase5_variance_audit import _episode_payload, _record, analyze_records


REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dataclass_to_dict(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def completed_payload(path, episode_id):
    records = event_records(path)
    if sum(r["event_type"] == "episode_result" for r in records) != 1:
        raise ValueError(f"expected exactly one terminal episode result: {path}")
    return _episode_payload(path, episode_id=episode_id)


def stage_rows(protocol, manifest, stage):
    if stage == "mechanism":
        audit = read(REPO / protocol["output_dir"] / "discovery.json")
        ids = set(audit["independent_episodes"][:2])
        return [row for row in manifest["tasks"] if row["episode_id"] in ids]
    return [row for row in manifest["tasks"] if row["role"] == stage]


def stage_seeds(protocol, stage):
    return [protocol["acquisition_seed"]] if stage in {"acquisition", "mechanism"} else protocol["rollout_seeds"]


def local_metrics(records):
    pending = set()
    metrics = {"failed_target_triggers": 0, "retries_without_new_evidence": 0,
               "target_evidence_recoveries": 0, "successful_retries_without_evidence": 0,
               "guard_blocks": 0}
    for record in records:
        if record["event_type"] == "temporal_guard":
            metrics["guard_blocks"] += 1
        if record["event_type"] != "transition":
            continue
        event = record["payload"]
        action = event["action"]
        target = next(iter(action["arguments"]), "")
        # Count admission from pre-action information, not the action's outcome.
        if action["action_type"] == "pick" and target in pending:
            metrics["retries_without_new_evidence"] += 1
            metrics["successful_retries_without_evidence"] += int(event["last_action_success"] is True)
        for old in tuple(pending):
            if _near_evidence(event, old, "true"):
                pending.remove(old)
                metrics["target_evidence_recoveries"] += 1
        if action["action_type"] == "pick":
            if event["last_action_success"] is True:
                pending.discard(target)
            elif _near_evidence(event, target, "false"):
                metrics["failed_target_triggers"] += int(target not in pending)
                pending.add(target)
    return metrics


def preregister(protocol_path, protocol, base_urls):
    output = REPO / protocol["output_dir"]
    manifest = read(REPO / protocol["dataset_manifest"])
    if sha(REPO / protocol["dataset"]) != manifest["dataset_sha256"]:
        raise ValueError("generated dataset digest mismatch")
    source_files = sorted((REPO / "vista_skill").rglob("*.py")) + [
        Path(__file__).resolve(), REPO / "scripts/phase5_recovery_dataset.py",
        REPO / protocol["config"],
    ]
    payload = {
        "protocol": protocol, "protocol_sha256": sha(protocol_path),
        "dataset_manifest_sha256": sha(REPO / protocol["dataset_manifest"]),
        "source_sha256": {str(p.relative_to(REPO)): sha(p) for p in source_files},
        "executor_base_urls": list(base_urls),
        "diagnostic": True, "claim_eligible": False,
    }
    path = output / "preregistration.json"
    if path.exists():
        if read(path) != payload:
            raise ValueError("code, endpoints or protocol changed after preregistration")
    else:
        if list((output / "rollouts").glob("**/*.jsonl")):
            raise ValueError("cannot preregister after rollout artifacts exist")
        write(path, payload)
    return manifest


def preflight(protocol, base_urls):
    output = REPO / protocol["output_dir"]
    checks = []
    for index, url in enumerate(base_urls):
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(url.rstrip("/") + "/models", timeout=6) as response:
                models = json.load(response)
            if protocol["executor"]["model"] not in [m["id"] for m in models["data"]]:
                raise ValueError("registered executor model absent from endpoint")
            log_path = output / f"endpoint{index}_probe.log"
            with log_path.open("w") as log:
                result = subprocess.run([
                    sys.executable, str(REPO / "scripts/probe_vllm_endpoint.py"),
                    "--base-url", url, "--model", protocol["executor"]["model"],
                ], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            if result.returncode:
                raise RuntimeError(f"serving contract probe failed: {log_path}")
            checks.append({"endpoint": url, "passed": True, "models": models})
        except Exception as error:
            checks.append({"endpoint": url, "passed": False,
                           "error": f"{type(error).__name__}: {error}"})
    write(output / "endpoint_preflight.json", {"checks": checks})
    if not all(check["passed"] for check in checks):
        write(output / "run_status.json", {"status": "blocked_endpoint_preflight",
            "model_rollouts_started": False, "claim_eligible": False})
        raise RuntimeError("endpoint preflight failed; no model rollouts started")


def run_worker(protocol, manifest, stage, endpoint_index, base_urls):
    output = REPO / protocol["output_dir"]
    config = load_config(REPO / protocol["config"])
    executor = protocol["executor"]
    endpoint = base_urls[endpoint_index]
    os.environ["remote_url"] = endpoint
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.chdir(REPO / "EmbodiedBench")
    args = SimpleNamespace(env="eb-hab", model_name=executor["model"], model_type="remote",
                           n_shots=executor["n_shots"], resolution=executor["resolution"],
                           tp=executor["tensor_parallel"], meta_skills="none", state_oracle_labels=False)
    rows = [row for row in stage_rows(protocol, manifest, stage) if row["endpoint_index"] == endpoint_index]
    seeds = stage_seeds(protocol, stage)
    for row in rows:
        for seed in seeds:
            arms = ["parent"] if stage == "acquisition" else ["parent", "candidate"]
            if (int(row["episode_id"].rsplit("_", 1)[1]) + seed) % 2:
                arms.reverse()
            for arm in arms:
                artifact = output / "rollouts" / stage / row["episode_id"] / f"s{seed}_{arm}.jsonl"
                metadata_path = artifact.with_suffix(".metadata.json")
                if artifact.exists():
                    # Resume only complete coordinates; never silently rerun a failure.
                    completed_payload(artifact, row["episode_id"])
                    if not metadata_path.exists():
                        raise ValueError(f"missing completed coordinate metadata: {artifact}")
                    continue
                skill = interface_only_shared_skill() if arm == "parent" else load_skill_artifact_record(output / "candidate_skill.json").skill
                seed_process_rngs(seed)
                usage = ExecutorUsageTracker()
                model = OpenAICompatibleJsonModel(executor["model"], base_url=endpoint, api_key="EMPTY",
                    seed=seed, max_tokens=protocol["evidence"]["visual_max_completion_tokens"])
                extractor = EvidenceExtractor(
                    JsonVisualEvidenceProvider(model),
                    EvidenceExtractorConfig(visual_action_types=(), visual_on_unresolved_goals=False,
                        visual_on_failed_preconditions=True,
                        max_visual_calls_per_episode=protocol["evidence"]["max_visual_calls_per_episode"]),
                    guard=EvidenceReliabilityGuard(EvidenceGuardConfig(mode=GuardMode.AUTHORITY_AWARE)),
                )
                engine = VistaSkillEngine(replace(skill, frozen=stage != "acquisition"),
                    action_schema=SkillOnlyActionSchema(), evidence_extractor=extractor,
                    credit_assigner=CreditAssigner(config=config.attribution))
                started = time.monotonic()
                env = create_habitat_development_env(REPO / protocol["dataset"], episode_ids=(row["episode_id"],),
                    resolution=executor["resolution"],
                    exp_name=f"{protocol['protocol_id']}/{stage}/{row['episode_id']}/s{seed}_{arm}")
                try:
                    seed_habitat_env(env, seed)
                    runner = _make_runner(args, env, skill, artifact, engine=engine, goal_grounder=None,
                        expected_episode_ids=(row["episode_id"],), task_coordinates=(),
                        rollout_seed=seed, max_completion_tokens=executor["max_completion_tokens"],
                        usage_tracker=usage, usage_phase=stage)
                    result = runner.run_episode(expected_episode_id=row["episode_id"])
                    runner.writer.append("method_usage", {
                        "episode_id": row["episode_id"],
                        "by_purpose": {key: asdict(value) for key, value in model.usage.items()},
                    })
                finally:
                    env.close()
                write(metadata_path, {
                    "endpoint_index": endpoint_index, "endpoint": endpoint,
                    "skill_sha256": skill_digest(skill),
                    "elapsed_seconds": time.monotonic() - started,
                    "executor_usage": usage.payload(),
                    "method_usage": {key: asdict(value) for key, value in model.usage.items()},
                    "local_metrics": local_metrics(event_records(artifact)),
                    "artifact_sha256": sha(artifact),
                })
                print(f"COMPLETE {stage} {row['episode_id']} seed={seed} arm={arm} success={result.task_success}", flush=True)


def discover(protocol, manifest):
    output = REPO / protocol["output_dir"]
    events = []
    for row in manifest["tasks"]:
        if row["role"] == "acquisition":
            path = output / "rollouts/acquisition" / row["episode_id"] / f"s{protocol['acquisition_seed']}_parent.jsonl"
            completed_payload(path, row["episode_id"])
            events.extend(r["payload"] for r in event_records(path) if r["event_type"] == "transition")
    parent = interface_only_shared_skill()
    patch, audit = propose_recovery(parent, events, minimum_episodes=protocol["minimum_independent_chains"])
    write(output / "discovery.json", audit)
    save_skill_artifact(output / "parent_skill.json", parent, protocol={"diagnostic": True})
    if patch is None:
        return False
    candidate = BoundedPatchApplier().apply(parent, patch)
    write(output / "patch.json", patch)
    save_skill_artifact(output / "candidate_skill.json", candidate, protocol={
        "diagnostic": True, "claim_eligible": False, "promoted": False,
        "preregistration_sha256": sha(output / "preregistration.json"),
    })
    return True


def analyze_stage(protocol, manifest, stage):
    output = REPO / protocol["output_dir"]
    rows = stage_rows(protocol, manifest, stage)
    records, metadata = [], []
    for row in rows:
        for seed in stage_seeds(protocol, stage):
            for arm in ("parent", "candidate"):
                path = output / "rollouts" / stage / row["episode_id"] / f"s{seed}_{arm}.jsonl"
                payload = completed_payload(path, row["episode_id"])
                records.append(_record(payload, episode_id=row["episode_id"], seed=seed, arm=arm, path=path))
                meta = read(path.with_suffix(".metadata.json"))
                if meta["artifact_sha256"] != sha(path):
                    raise ValueError("rollout changed after completion")
                if meta["endpoint_index"] != row["endpoint_index"]:
                    raise ValueError("paired coordinate endpoint differs from assignment")
                disk_events = event_records(path)
                if meta["local_metrics"] != local_metrics(disk_events):
                    raise ValueError("local metrics do not reproduce from the trajectory")
                executor_usage = [r["payload"] for r in disk_events if r["event_type"] == "executor_usage"]
                method_usage = [r["payload"]["by_purpose"] for r in disk_events if r["event_type"] == "method_usage"]
                if len(executor_usage) != 1 or method_usage != [meta["method_usage"]]:
                    raise ValueError("missing or inconsistent episode usage records")
                for key in ("calls", "prompt_tokens", "completion_tokens"):
                    if executor_usage[0][key] != meta["executor_usage"][key]:
                        raise ValueError("executor cost does not match episode record")
                metadata.append({**meta, "episode_id": row["episode_id"], "seed": seed, "arm": arm})
    if stage == "mechanism":
        metrics = {}
        for arm in ("parent", "candidate"):
            selected = [r for r in records if r["arm"] == arm]
            metrics[arm] = {
                "successes": sum(r["task_success"] for r in selected),
                "progress_sum": sum(r["task_progress"] for r in selected),
                "retries_without_new_evidence": sum(m["local_metrics"]["retries_without_new_evidence"] for m in metadata if m["arm"] == arm),
                "guard_blocks": sum(r["temporal_guard_blocks"] for r in selected),
            }
        parent, candidate = metrics["parent"], metrics["candidate"]
        go = (candidate["retries_without_new_evidence"] < parent["retries_without_new_evidence"]
              and candidate["successes"] >= parent["successes"]
              and candidate["progress_sum"] >= parent["progress_sum"])
        write(output / "mechanism_records.json", {"records": records})
        write(output / "mechanism_analysis.json", {
            "verdict": "supportive" if go else "no_go", "metrics": metrics,
            "coordinate_metadata": metadata, "outcome_selected_acquisition_diagnostic": True,
            "claim_eligible": False,
        })
        return "supportive" if go else "no_go"
    indexed = {(r["episode_id"], r["seed"], r["arm"]): r for r in records}
    scores = []
    for row in rows:
        for seed in protocol["rollout_seeds"]:
            parent, candidate = (indexed[(row["episode_id"], seed, arm)] for arm in ("parent", "candidate"))
            scores.append(PairedEpisodeScore(
                episode_id=row["episode_id"], seed=seed, parent_score=parent["score"],
                candidate_score=candidate["score"], subgroup=row["subgroup"],
                parent_success=bool(parent["task_success"]), candidate_success=bool(candidate["task_success"]),
                semantic_tags=("action:pick",) if row["scope"] == "affected" else ("action:nav",),
            ))
    config = load_config(REPO / protocol["config"])
    gate = CandidateGate(object(), object(), object(), config.gate)
    result = gate._paired_stage("paired_proxy" if stage == "proxy" else "paired_finalist",
                               scores, 0.0, len(scores), ("action:pick",))
    # Independent success protection uses the same task-first resampling.
    success_scores = [replace(s, parent_score=float(s.parent_success), candidate_score=float(s.candidate_success)) for s in scores]
    success_config = replace(config.gate, semantic_affected_enabled=False)
    success_gate = CandidateGate(object(), object(), object(), success_config)
    success = success_gate._paired_stage("paired_finalist", success_scores, -0.05, len(scores), ())
    variance = analyze_records(records, episode_ids=[r["episode_id"] for r in rows],
        seeds=protocol["rollout_seeds"], variance_share_threshold=0.2, minimum_delta_divergent_tasks=2)
    contradictory = (result.metrics["affected_mean_delta"] <= 0
        or result.metrics["protected_mean_delta"] < -0.05
        or result.metrics["worst_subgroup_delta"] < -0.05)
    verdict = "supportive" if result.passed and success.metrics["lcb"] >= -0.05 else (
        "contradictory" if contradictory else "inconclusive")
    per_endpoint = {}
    for endpoint in (0, 1):
        ids = {r["episode_id"] for r in rows if r["endpoint_index"] == endpoint}
        per_endpoint[str(endpoint)] = {
            "independent_tasks": len(ids),
            "mean_delta": statistics.fmean(s.candidate_score - s.parent_score for s in scores if s.episode_id in ids),
        }
    summary = {"verdict": verdict, "gate": dataclass_to_dict(result),
               "success_noninferiority_lcb": success.metrics["lcb"], "variance": variance,
               "endpoint_strata": per_endpoint, "coordinate_metadata": metadata,
               "claim_eligible": False}
    write(output / f"{stage}_records.json", {"records": records})
    write(output / f"{stage}_analysis.json", summary)
    return verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(REPO / "configs/phase5_p57_recovery_pilot.json"))
    parser.add_argument("--base-urls", nargs=2, required=True)
    parser.add_argument("--worker", type=int, choices=(0, 1))
    parser.add_argument("--stage", choices=("acquisition", "mechanism", "proxy", "finalist", "audit"))
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    path = Path(args.protocol).resolve()
    protocol = read(path)
    if args.worker is None and not args.analyze_only:
        preflight(protocol, args.base_urls)
    if args.analyze_only and not (REPO / protocol["output_dir"] / "preregistration.json").exists():
        raise FileNotFoundError("analyze-only requires an existing preregistration")
    manifest = preregister(path, protocol, args.base_urls)
    if args.analyze_only:
        if args.stage not in {"mechanism", "proxy", "finalist", "audit"}:
            parser.error("analyze-only requires a paired stage")
        print(analyze_stage(protocol, manifest, args.stage))
        return
    if args.worker is not None:
        if args.stage is None:
            parser.error("worker requires stage")
        run_worker(protocol, manifest, args.stage, args.worker, args.base_urls)
        return
    output = REPO / protocol["output_dir"]
    decisions = {}
    for stage in ("acquisition", "mechanism", "proxy", "finalist", "audit"):
        processes, logs = [], []
        for index in (0, 1):
            log = (output / f"{stage}_worker{index}.log").open("a")
            logs.append(log)
            processes.append(subprocess.Popen([
                sys.executable, str(Path(__file__).resolve()), "--protocol", str(path),
                "--base-urls", *args.base_urls, "--worker", str(index), "--stage", stage,
            ], cwd=REPO, stdout=log, stderr=subprocess.STDOUT))
        codes = [process.wait() for process in processes]
        for log in logs:
            log.close()
        if any(codes):
            write(output / "run_status.json", {"status": "execution_error", "stage": stage, "exit_codes": codes})
            raise RuntimeError(f"{stage} worker failed: {codes}; retain artifacts and inspect logs")
        if stage == "acquisition":
            if not discover(protocol, manifest):
                decisions[stage] = "no_recurrent_candidate"
                break
        else:
            verdict = analyze_stage(protocol, manifest, stage)
            decisions[stage] = verdict
            if verdict in {"contradictory", "no_go"} or (
                stage == "finalist" and (verdict != "supportive" or decisions["proxy"] != "supportive")
            ):
                break
    positive = all(decisions.get(stage) == "supportive" for stage in ("proxy", "finalist", "audit"))
    selected = output / ("candidate_skill.json" if positive else "parent_skill.json")
    skill = replace(load_skill_artifact_record(selected).skill, frozen=True)
    save_skill_artifact(output / "frozen_skill.json", skill, protocol={
        "diagnostic": True, "claim_eligible": False, "diagnostic_promotion": positive,
        "preregistration_sha256": sha(output / "preregistration.json"),
    })
    write(output / "run_status.json", {"status": "completed", "decisions": decisions,
        "credible_positive_diagnostic": positive, "claim_eligible": False})
    costs = {}
    for stage in ("acquisition", "mechanism", "proxy", "finalist", "audit"):
        files = list((output / "rollouts" / stage).glob("**/*.metadata.json"))
        if not files:
            continue
        records = [read(file) for file in files]
        costs[stage] = {
            "episodes": len(records),
            "executor": {key: sum(r["executor_usage"][key] for r in records)
                         for key in ("calls", "prompt_tokens", "completion_tokens")},
            "method": {key: sum(v[key] for r in records for v in r["method_usage"].values())
                       for key in ("calls", "prompt_tokens", "completion_tokens")},
            "summed_coordinate_seconds": sum(r["elapsed_seconds"] for r in records),
        }
    write(output / "cost_summary.json", {
        "by_stage": costs, "includes_acquisition_and_every_completed_paired_stage": True,
        "remote_gpu_hours": None, "remote_gpu_hours_reason": "requires serving-host telemetry",
    })
    print(json.dumps(decisions), flush=True)


if __name__ == "__main__":
    main()
