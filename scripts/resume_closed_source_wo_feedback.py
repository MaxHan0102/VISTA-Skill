"""Resume the six stock-feedback-v2 runs at completed-episode boundaries.

Workers fill missing native episode files in the original run directory.
Per-process metadata/API logs live in _sessions/; incomplete files are archived.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.evaluate_closed_loop_feedback import BENCH, SETS, allocate_output, permanent_schema_error

RUNNER = REPO / "scripts/evaluate_closed_loop_feedback.py"
CONDA = Path("/root/miniconda3/bin/conda")
MODELS = (("openai", "gpt-5.4-mini"), ("gemini", "gemini-3-flash-preview"), ("gemini", "gemini-3.5-flash"))
# Transport retries/logging may change; experimental settings must not.
MATCH_FIELDS = ("protocol", "env", "provider", "model", "track", "seed", "frames", "n_shots",
                "resolution", "max_steps", "max_tokens", "temperature", "base_url", "smoke_policy")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def command_prefix(env):
    name = "max_embench" if env == "eb-hab" else "max_embench_nav"
    return [str(CONDA), "run", "--no-capture-output", "-n", name, "python", str(RUNNER)]


def native_hashes(env):
    habitat = env == "eb-hab"
    files = [f"embodiedbench/configs/{env}.yaml", "embodiedbench/planner/remote_model.py",
             "embodiedbench/planner/vlm_planner.py" if habitat else "embodiedbench/planner/nav_planner.py",
             "embodiedbench/envs/eb_habitat/EBHabEnv.py" if habitat else "embodiedbench/envs/eb_navigation/EBNavEnv.py",
             "embodiedbench/evaluator/eb_habitat_evaluator.py" if habitat else "embodiedbench/evaluator/eb_navigation_evaluator.py"]
    return {str((BENCH / p).relative_to(REPO)): sha(BENCH / p) for p in files}


def datasets(config):
    ranges, hashes = {}, {}
    for split in config["eval_sets"]:
        if config["env"] == "eb-hab":
            path = BENCH / f"embodiedbench/envs/eb_habitat/datasets/{split}.pickle"
            # Only the trusted, installed benchmark dataset is decoded.
            with path.open("rb") as stream:
                count = len(pickle.load(stream)["all_eps"])
        else:
            path = BENCH / f"embodiedbench/envs/eb_navigation/datasets/{split}.json"
            count = len(read(path)["tasks"])
        start = config["start_index"]
        end = start + config["episodes"] if config["episodes"] else count
        if not 0 <= start < end <= count:
            raise ValueError(f"invalid task range for {split}")
        ranges[split] = list(range(start + 1, end + 1))
        hashes[split] = sha(path)
    return ranges, hashes


def check_config(config):
    if config.get("protocol") != "stock_feedback_v2" or config.get("track") not in {"rgb_only", "full"}:
        raise ValueError("only stock_feedback_v2 runs can be resumed here")
    actual = native_hashes(config["env"])
    for path, expected in config.get("source_hashes", {}).items():
        if actual.get(path) != expected:
            raise ValueError(f"official source drift: {path}")


def compatible(root_config, child):
    if any(root_config.get(k) != child.get(k) for k in MATCH_FIELDS):
        raise ValueError("resume attempt has different experiment settings")
    left, right = dict(root_config["official_config"]), dict(child["official_config"])
    for key in ("eval_sets", "start_epi_index", "exp_name"):
        left.pop(key, None)
        right.pop(key, None)
    if left != right:
        raise ValueError("official configuration drift")
    check_config(child)


def read_jsonl(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise ValueError(f"corrupt audit log: {path}")
            # A process killed during the final write can leave a partial line.
    return rows


def scan(root, config, ranges, hashes, pinned=None):
    records = dict(pinned or {})
    identities, partial = {}, []
    provenance = read(root / "episode_provenance.json") if (root / "episode_provenance.json").exists() else {}
    # New workers write results directly into root, but keep their configuration
    # and API-call numbering in a separate session. Recover provenance even if a
    # process died between publishing a result and updating the checkpoint.
    session_origins = {}
    for session in sorted((root / "_sessions").glob("run_*")):
        if not (session / "config.json").exists():
            continue
        child = read(session / "config.json")
        compatible(config, child)
        for split in child["eval_sets"]:
            selection = session / split / "selection.json"
            if selection.exists() and read(selection)["dataset_sha256"] != hashes[split]:
                raise ValueError(f"dataset drift: {split}")
        for ep in read_jsonl(session / "audit/episodes.jsonl"):
            split, eid = ep["split"], str(ep["episode_id"])
            ordinal = int(eid[4:]) + 1 if config["env"] == "eb-nav" else ep["episode_number"]
            if ordinal not in ranges.get(split, []):
                raise ValueError("session task outside the registered range")
            previous = identities.setdefault(split, {}).setdefault(str(ordinal), eid)
            if previous != eid:
                raise ValueError(f"episode identity changed at {split}/{ordinal}")
            path = root / split / "results" / f"episode_{ordinal}_final_res.json"
            if path.is_file():
                session_origins[f"{split}/{ordinal}"] = {"path": str(path), "sha256": sha(path),
                    "episode_id": eid, "source_run": str(session),
                    "schema_compatibility": child.get("schema_compatibility", "stock")}
    provenance = {**session_origins, **provenance}
    for key, item in records.items():
        if not Path(item["path"]).is_file() or sha(item["path"]) != item["sha256"]:
            raise ValueError(f"previously accepted result changed: {key}")
    runs = [root] + sorted((root / "_resume").glob("attempt_*"))
    for run in runs:
        if not (run / "config.json").exists():
            continue
        child = read(run / "config.json")
        compatible(config, child)
        episode_map = {}
        for ep in read_jsonl(run / "audit/episodes.jsonl"):
            split = ep["split"]
            eid = str(ep["episode_id"])
            legacy = re.search(r"episode_id='([^']+)'", eid)
            if legacy:
                eid = legacy.group(1)
            ordinal = int(eid[4:]) + 1 if config["env"] == "eb-nav" else ep["episode_number"]
            if ordinal not in ranges.get(split, []):
                raise ValueError("episode audit is outside the registered task range")
            previous = identities.setdefault(split, {}).setdefault(str(ordinal), eid)
            if previous != eid:
                raise ValueError(f"episode identity changed at {split}/{ordinal}")
            episode_map[(split, ordinal)] = eid
        if run == root:
            for key, item in provenance.items():
                split, ordinal = key.split("/")
                ordinal = int(ordinal)
                if ordinal not in ranges.get(split, []):
                    raise ValueError(f"out-of-range provenance: {key}")
                eid = item["episode_id"]
                previous = identities.setdefault(split, {}).setdefault(str(ordinal), eid)
                if previous != eid:
                    raise ValueError(f"episode identity changed at {key}")
                episode_map[(split, ordinal)] = eid
        for split, allowed in ranges.items():
            selection = run / split / "selection.json"
            if selection.exists() and read(selection)["dataset_sha256"] != hashes[split]:
                raise ValueError(f"dataset drift: {split}")
            for path in sorted((run / split / "results").glob("episode_*_final_res.json")):
                ordinal = int(path.name.split("_")[1])
                if ordinal not in allowed:
                    raise ValueError(f"out-of-range result {path}")
                key = f"{split}/{ordinal}"
                try:
                    row = read(path)
                    if not isinstance(row, dict) or not isinstance(row.get("task_success"), (bool, int, float)) or not 0 <= row["task_success"] <= 1:
                        raise ValueError("invalid score")
                    if not isinstance(row.get("num_steps"), (int, float)) or row["num_steps"] < 0:
                        raise ValueError("missing step count")
                    if not isinstance(row.get("instruction"), str):
                        raise ValueError("missing instruction")
                except (ValueError, TypeError):
                    partial.append(str(path))
                    continue
                if (split, ordinal) not in episode_map:
                    raise ValueError(f"result lacks episode identity audit: {path}")
                if key not in records:
                    # First complete record wins, regardless of success/failure.
                    records[key] = {"path": str(path), "sha256": sha(path), "episode_id": episode_map[(split, ordinal)]}
                if records[key]["path"] == str(path):
                    records[key] = {**records[key], "schema_compatibility": child.get("schema_compatibility", "stock")}
                    if run == root and key in provenance:
                        origin = provenance[key]
                        if sha(path) != origin["sha256"]:
                            raise ValueError(f"consolidated result changed: {key}")
                        records[key].update(origin)
                        records[key]["path"] = str(path)
    return records, identities, partial


def next_gap(ranges, records):
    for split, ordinals in ranges.items():
        missing = [i for i in ordinals if f"{split}/{i}" not in records]
        if missing:
            first = last = missing[0]
            for i in missing[1:]:
                if i != last + 1:
                    break
                last = i
            return split, first - 1, last - first + 1
    return None


def checkpoint(root, config, ranges, hashes, records, identities, partial):
    categories = {}
    for split, expected in ranges.items():
        selected = [records[f"{split}/{i}"] for i in expected if f"{split}/{i}" in records]
        rows = [read(item["path"]) for item in selected]
        categories[split] = {"completed": len(rows), "expected": len(expected),
                             "task_success": sum(r["task_success"] for r in rows) / len(rows) if rows else None}
        if rows:
            folder = root / split
            folder.mkdir(parents=True, exist_ok=True)
            selection = folder / "selection.json"
            if not selection.exists():
                write(selection, {"start_index": expected[0] - 1, "episodes": len(expected),
                      "seed": config["seed"], "dataset_sha256": hashes[split], "order": "stock dataset/iterator"})
            if not (folder / "config.txt").exists():
                (folder / "config.txt").write_text(str(config["official_config"]))
    complete = next_gap(ranges, records) is None
    schema_counts = {}
    for item in records.values():
        revision = item.get("schema_compatibility", "stock")
        schema_counts[revision] = schema_counts.get(revision, 0) + 1
    write(root / "resume_state.json", {"version": 1, "protocol": config["protocol"],
          "dataset_hashes": hashes, "records": records, "identities": identities,
          "partial_results": partial})
    write(root / "resume_summary.json", {"status": "complete" if complete else "incomplete",
          "protocol": config["protocol"], "smoke_policy": config.get("smoke_policy", False),
          "categories": categories, "completed": len(records), "expected": sum(map(len, ranges.values())),
          "category_mean_success": sum(v["task_success"] for v in categories.values()) / len(categories) if complete else None,
          "result_index": str(root / "resume_state.json"),
          "schema_compatibility_counts": schema_counts,
          "mixed_schema_compatibility": len(schema_counts) > 1,
          "cost_note": "All original/attempt API audits, including abandoned partial episodes, contribute to cost."})
    write(root / "resume_expected_episodes.json", identities)
    write(root / "episode_provenance.json", {k: v for k, v in records.items()
          if Path(v["path"]).parents[2] == root})


def refresh_native_summaries(root, config, records):
    """Use the stock arithmetic mean, reading only canonical final results."""
    for split in config["eval_sets"]:
        rows = [read(item["path"]) for key, item in records.items() if key.startswith(split + "/")]
        if not rows:
            continue
        values = {}
        for row in rows:
            for key, value in row.items():
                if isinstance(value, str):
                    continue
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                values.setdefault(key, []).append(value)
        folder = root / split / "results"
        folder.mkdir(parents=True, exist_ok=True)
        name = "summary.json" if config["env"] == "eb-hab" else "summary_all.json"
        write(folder / name, {k: sum(v) / len(v) for k, v in values.items()})


def worker_command(config, root, output, gap, api_retries):
    split, start, count = gap
    cmd = command_prefix(config["env"]) + ["--provider", config["provider"], "--model", config["model"],
          "--track", config["track"], "--env", config["env"], "--eval-sets", split,
          "--start-index", str(start), "--episodes", str(count), "--seed", str(config["seed"]),
          "--frames", str(config["frames"]), "--base-url", config["base_url"],
          "--api-key-env", config["api_key_env"], "--output", str(output),
          "--artifact-root", str(root),
          "--api-retries", str(api_retries), "--expected-episodes", str(root / "resume_expected_episodes.json")]
    for field in ("n_shots", "resolution", "max_steps", "max_tokens", "temperature", "log_level"):
        if config.get(field) is not None:
            cmd += ["--" + field.replace("_", "-"), str(config[field])]
    if config.get("smoke_policy"):
        cmd.append("--smoke-policy")
    return cmd


def launch(cmd):
    process = subprocess.Popen(cmd, cwd=BENCH, start_new_session=True)
    try:
        return process.wait()
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
        raise


def retryable_failure(attempt):
    failure = read(attempt / "failure.json") if (attempt / "failure.json").exists() else {}
    status = failure.get("http_status")
    errors = read_jsonl(attempt / "audit/api_errors.jsonl")
    details = failure.get("provider_error") or (errors[-1].get("provider_error", {}) if errors else {})
    if permanent_schema_error(details):
        return False
    if status in {400, 408, 409, 429} or (isinstance(status, int) and status >= 500):
        return True
    return status is None and bool(errors) and errors[-1].get("error_type") in {"APIConnectionError", "APITimeoutError"}


def archive_incomplete(root, gap, records, session_name):
    from scripts.consolidate_closed_feedback import episode_files
    split, start, count = gap
    moves = []
    for ordinal in range(start + 1, start + count + 1):
        if f"{split}/{ordinal}" in records:
            raise ValueError("refusing to restart a completed task")
        for source in episode_files(root, split, ordinal):
            destination = root / "_interrupted" / session_name / source.relative_to(root)
            if destination.exists():
                raise ValueError(f"interrupted archive collision: {destination}")
            moves.append({"source": str(source), "destination": str(destination), "sha256": sha(source)})
    if moves:
        folder = root / "_interrupted" / session_name
        folder.mkdir(parents=True, exist_ok=True)
        write(folder / "manifest.json", {"status": "planned", "files": moves})
        for item in moves:
            source, destination = Path(item["source"]), Path(item["destination"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
        write(folder / "manifest.json", {"status": "complete", "files": moves})


def resume(root, args):
    config = read(root / "config.json")
    check_config(config)
    ranges, hashes = datasets(config)
    state = read(root / "resume_state.json") if (root / "resume_state.json").exists() else {}
    if state and state["dataset_hashes"] != hashes:
        raise ValueError("dataset changed since checkpoint")
    records, identities, partial = scan(root, config, ranges, hashes, state.get("records"))
    print(f"{root.name}: {len(records)}/{sum(map(len, ranges.values()))} completed; next={next_gap(ranges, records)}", flush=True)
    if args.dry_run:
        return
    if any(Path(item["path"]).parents[2] != root for item in records.values()):
        from scripts.consolidate_closed_feedback import consolidate
        consolidate(root)
    with (root / ".resume.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another resume controller owns this run")
        # Re-read after locking, in case another controller just updated the checkpoint.
        state = read(root / "resume_state.json") if (root / "resume_state.json").exists() else {}
        records, identities, partial = scan(root, config, ranges, hashes, state.get("records"))
        checkpoint(root, config, ranges, hashes, records, identities, partial)
        refresh_native_summaries(root, config, records)
        stalled = 0
        while (gap := next_gap(ranges, records)) is not None:
            if not config.get("smoke_policy") and config["provider"] != "qwen" and not os.environ.get(config["api_key_env"]):
                raise RuntimeError(f"missing exported {config['api_key_env']}")
            if config["env"] == "eb-nav" and not os.environ.get("DISPLAY"):
                raise RuntimeError("missing DISPLAY for Navigation")
            stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d%H%M")
            attempt = allocate_output(root / "_sessions" / f"run_{stamp}", stamp, create=False)
            archive_incomplete(root, gap, records, attempt.name)
            cmd = worker_command(config, root, attempt, gap, args.api_retries)
            print(f"Resume {config['model']} / {gap[0]} at task {gap[1]+1}, count {gap[2]}; {attempt}", flush=True)
            before = len(records)
            try:
                code = launch(cmd)
            finally:
                records, identities, partial = scan(root, config, ranges, hashes, records)
                checkpoint(root, config, ranges, hashes, records, identities, partial)
                refresh_native_summaries(root, config, records)
            if code == 0:
                if len(records) == before:
                    raise RuntimeError("worker reported success without new complete results")
                stalled = 0
                continue
            if not retryable_failure(attempt):
                raise RuntimeError(f"non-retryable worker failure; inspect {attempt}")
            stalled = 0 if len(records) > before else stalled
            stalled += 1
            if stalled > args.max_retries:
                raise RuntimeError("episode restart budget exhausted; completed results checkpointed")
            print(f"Worker interrupted; preserving {len(records)} results. Restart {stalled}/{args.max_retries} in {args.retry_delay}s", flush=True)
            time.sleep(args.retry_delay)
        print(f"Completed: {root / 'resume_summary.json'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="resume only this run; default continues all six canonical full runs")
    parser.add_argument("--dry-run", action="store_true", help="show checkpoints without API calls or writes")
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=3, help="bounded restarts at an unfinished task")
    parser.add_argument("--retry-delay", type=float, default=30)
    args = parser.parse_args(argv)
    if min(args.api_retries, args.max_retries, args.retry_delay) < 0:
        parser.error("retry settings cannot be negative")
    roots = [(args.run_dir.resolve(), None, None)] if args.run_dir else [
        (REPO / f"running/closed_feedback/{model}_{env}_v2_full", env, (provider, model))
        for env in SETS for provider, model in MODELS]
    try:
        for root, env, spec in roots:
            if REPO / "running" not in root.parents:
                raise ValueError("run directory must be under repository running/")
            if not (root / "config.json").exists():
                if args.run_dir or root.exists():
                    raise ValueError(f"missing config.json: {root}")
                if args.dry_run:
                    print(f"{root.name}: not started; will run all {len(SETS[env])} categories")
                    continue
                provider, model = spec
                preview = subprocess.check_output(command_prefix(env) + ["--provider", provider, "--model", model,
                    "--env", env, "--track", "rgb_only", "--eval-sets", "all", "--episodes", "0",
                    "--output", str(root), "--dry-run"], cwd=BENCH, text=True)
                config = json.loads(preview)
                config["dry_run"] = False
                config["source_hashes"] = native_hashes(env)
                config["resume_managed"] = True
                root.mkdir(parents=True, exist_ok=False)
                write(root / "config.json", config)
            resume(root, args)
    except KeyboardInterrupt:
        print("Stopped by user; completed tasks are checkpointed. Run this command again to continue.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Resume stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
