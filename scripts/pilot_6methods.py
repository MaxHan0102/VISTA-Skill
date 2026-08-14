"""6-method controlled pilot at matched (fair) settings, against a live vLLM server.

Fairness controls baked in -- every method gets the SAME:
  executor (Qwen3-VL-8B via one vLLM endpoint), teacher (same endpoint, for the
  teacher-requiring methods), initial Skill, config (configs/vista_pilot.json:
  reduced gate budgets 2/3 so rollouts are bounded but IDENTICAL across methods),
  acquisition episode count, evolution seeds (0,1,2), n_shots/resolution.

Pipeline:
  1) run `experiment` for the 4 evolution methods -> frozen skill + update_audit
     (RQ3 reliability) + method_usage/execuator_usage (RQ4 cost) per method.
  2) run `evaluate --stage official_test --eval-set base` for all 6 methods on the
     SAME small test set / seed -> mean task success (RQ1).
  3) print a comparison table.

Assumes vLLM is up on :8001 and Habitat renders on GPU 0 (CUDA_VISIBLE_DEVICES=0
forces the CLI's renderer off the vLLM GPU).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/root/max/VISTA-Skill")
EMBENCH_PY = "/root/miniconda3/envs/embench/bin/python"
BASE_URL = "http://127.0.0.1:8001/v1"
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CONFIG = "configs/vista_pilot.json"
MANIFEST = "configs/eb_hab_pilot_manifest.json"
ACQ = 15
EVAL_SET = "base"
EVAL_N = 20
EVAL_SEED = 0
OUT = REPO / "running" / "pilot"

EVOL_METHODS = [
    "full",
    "embodiskill_star_native",
    "embodiskill_star_common_gate",
    "vista_without_vtca",
]


def _env():
    return {
        **os.environ,
        "PYTHONPATH": str(REPO),
        "OPENAI_API_KEY": "EMPTY",
        "CUDA_VISIBLE_DEVICES": "0",  # Habitat renders on GPU0; vLLM is on GPU1
    }


def _cli(args, log):
    with open(log, "w") as f:
        subprocess.run(
            [EMBENCH_PY, "-m", "vista_skill.integrations.embodiedbench.cli", *args],
            cwd=str(REPO), env=_env(), stdout=f, stderr=subprocess.STDOUT, check=True,
        )


def run_experiments():
    for m in EVOL_METHODS:
        out = OUT / m
        if (out / "experiment_manifest.json").exists():
            print(f"[skip] experiment {m} (already done)", flush=True)
            continue
        if out.exists():
            import shutil
            shutil.rmtree(out)  # partial dir from a killed run -> start clean
        args = [
            "experiment", "--method", m, "--config", CONFIG, "--manifest", MANIFEST,
            "--diagnostic", "--max-acquisition-episodes", str(ACQ), "--evolution-seeds", "0",
            "--executor-base-url", BASE_URL, "--output-dir", str(out),
            "--method-model", MODEL, "--method-base-url", BASE_URL,
        ]
        print(f"[run] experiment {m} ...", flush=True)
        try:
            _cli(args, f"/tmp/pilot_exp_{m}.log")
        except subprocess.CalledProcessError as exc:
            print(f"[FAIL] experiment {m} (exit {exc.returncode}); see /tmp/pilot_exp_{m}.log -- continuing", flush=True)
            continue
        print(f"[done] experiment {m}", flush=True)


def _teacher_tokens(manifest):
    tot_p = tot_c = calls = 0
    for r in manifest.get("runs", []):
        for _purpose, u in (r.get("method_usage") or {}).items():
            calls += u.get("calls", 0)
            tot_p += u.get("prompt_tokens", 0)
            tot_c += u.get("completion_tokens", 0)
    return calls, tot_p, tot_c


def _exec_tokens(manifest):
    p = c = calls = 0
    for r in manifest.get("runs", []):
        u = r.get("executor_usage") or {}
        calls += u.get("calls", 0)
        p += u.get("prompt_tokens", 0)
        c += u.get("completion_tokens", 0)
    return calls, p, c


def _accepted(manifest):
    return [bool(d) for r in manifest.get("runs", []) for d in r.get("evolution_decisions", [])]


def _reliability(manifest):
    rels = [r.get("update_reliability") for r in manifest.get("runs", [])
            if r.get("update_reliability")]
    if not rels:
        return None
    keys = set().union(*[r.keys() for r in rels])
    return {k: sum(r.get(k, 0) for r in rels) / len(rels) for k in keys}


def run_evals():
    rows = []
    specs = [("no_skill", "no_skill", None),
             ("static_shared_skill", "static_shared_skill", None)]
    for m in EVOL_METHODS:
        skill = OUT / m / f"seed_{EVAL_SEED}" / "frozen_skill.json"
        specs.append((m, "frozen_skill", str(skill) if skill.exists() else None))
    for label, mode, skill in specs:
        output = OUT / "eval" / f"{label}.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        if (output.with_suffix(".summary.json")).exists():
            (output.with_suffix(".summary.json")).unlink()
        args = [
            "evaluate", "--mode", mode, "--config", CONFIG, "--manifest", MANIFEST,
            "--diagnostic", "--stage", "official_test", "--eval-set", EVAL_SET,
            "--max-episodes", str(EVAL_N), "--seed", str(EVAL_SEED),
            "--executor-base-url", BASE_URL, "--output", str(output),
        ]
        if skill:
            args += ["--skill", skill]
        print(f"[run] evaluate {label} ({mode}) ...", flush=True)
        # Run eval without check=True: a non-zero exit from habitat teardown
        # AFTER the summary is written must not discard a valid eval result.
        with open(f"/tmp/pilot_eval_{label}.log", "w") as f:
            subprocess.run(
                [EMBENCH_PY, "-m", "vista_skill.integrations.embodiedbench.cli", *args],
                cwd=str(REPO), env=_env(), stdout=f, stderr=subprocess.STDOUT,
            )
        try:
            summ = json.loads(output.with_suffix(".summary.json").read_text())
            rows.append((label, summ.get("mean_task_success"), summ.get("mean_task_progress")))
            print(f"[done] evaluate {label}: success={summ.get('mean_task_success'):.3f}", flush=True)
        except FileNotFoundError:
            print(f"[FAIL] evaluate {label}: no summary written", flush=True)
            rows.append((label, None, None))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    run_experiments()

    print("\n\n========== PILOT RESULTS ==========", flush=True)
    print("\n--- RQ1 task performance (eval on 'base', %d ep, seed %d) ---" % (EVAL_N, EVAL_SEED), flush=True)
    eval_rows = run_evals()
    for label, succ, prog in eval_rows:
        s = f"{succ:.3f}" if succ is not None else "  n/a"
        p = f"{prog:.3f}" if prog is not None else "  n/a"
        print(f"  {label:32s} success={s}  progress={p}", flush=True)

    print("\n--- RQ3 update reliability + RQ4 cost (experiment, 3 seeds x %d acq ep) ---" % ACQ, flush=True)
    print(f"  {'method':32s} {'accepted':>9} {'harmful':>8} {'benef':>7} {'teacher_tok':>12} {'exec_tok':>10}", flush=True)
    for m in EVOL_METHODS:
        mf = OUT / m / "experiment_manifest.json"
        if not mf.exists():
            print(f"  {m:32s} (no manifest)", flush=True)
            continue
        manifest = json.loads(mf.read_text())
        acc = _accepted(manifest)
        rel = _reliability(manifest) or {}
        tc, tp, tcom = _teacher_tokens(manifest)
        _, ep, ec = _exec_tokens(manifest)
        n_acc = sum(acc)
        print(f"  {m:32s} {n_acc:>9} {rel.get('harmful_updates', 0):>8.1f} "
              f"{rel.get('beneficial_updates', 0):>7.1f} {tp + tcom:>12d} {ep + ec:>10d}", flush=True)
    print("\nfairness: all methods share executor/teacher endpoint, config, seeds, "
          "acquisition count, n_shots, resolution; gate budgets identical (pilot 2/3).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
