"""Fault-repair effectiveness validation (fast path to testing the core claim).

E5/E6 froze every method at S0 because genuine skill faults never recurred at
pilot scale. This driver makes recurrence reachable by INJECTING a structured
fault into the initial shared Skill (docs/experiment_log.md open item 2), then
asks the only question that matters: can VISTA evolution detect, attribute,
and REPAIR the fault, and does performance recover?

Arms (executor/teacher/eval protocol identical across all):
  A  corrupted_frozen   -- faulty Skill, frozen, no evolution (lower bound)
  B  full               -- faulty Skill + full VISTA evolution (the method)
  C  vista_without_vtca -- faulty Skill + trajectory-reflection baseline
                          (strongest "reflection without VTCA" contrast)

References (same eval protocol, E6): S0 = 0.450, no_skill = 0.550.

Effectiveness readout: drop = S0 - A; recovery = B - A (want ~= drop), and
C - A (want ~0 if repair needs action-level credit assignment). Lineage/
gate decisions show whether the repair was accepted and nothing harmful was.

All runs are --diagnostic (protocol deviation: injected initial Skill, reduced
acquisition). Honours VISTA_METHOD_BASE_URL / VISTA_EMBENCH_PY like the pilot.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path("/root/max/VISTA-Skill")
EMBENCH_PY = os.environ.get("VISTA_EMBENCH_PY", "/root/miniconda3/envs/max_embench/bin/python")
BASE_URL = os.environ.get("VISTA_METHOD_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
FAULT = os.environ.get("VISTA_FAULT", "termination")
ACQ = int(os.environ.get("VISTA_FAULT_ACQ", "20"))
OUT = Path(os.environ.get("VISTA_FAULT_OUT", str(REPO / "running" / "fault_repair")))
CONFIG = os.environ.get("VISTA_FAULT_CONFIG", "configs/vista_fault_repair.json")
MANIFEST = os.environ.get(
    "VISTA_FAULT_MANIFEST",
    "configs/eb_hab_pilot_manifest.json"
    if CONFIG.endswith("vista_fault_repair.json")
    or CONFIG.endswith("vista_fault_repair_mind1.json")
    or CONFIG.endswith("vista_fault_repair_mind1_p10.json")
    else "configs/eb_hab_train_validation_manifest.json",
)
EVAL_SET = "base"
EVAL_N = 20
EVAL_SEED = 0

sys.path.insert(0, str(REPO))
from vista_skill.fault_injection import FaultType, inject_skill_fault  # noqa: E402
from vista_skill.skills import initialize_shared_skill, save_skill_artifact  # noqa: E402


def _env():
    return {
        **os.environ,
        "PYTHONPATH": str(REPO),
        "OPENAI_API_KEY": "EMPTY",
        "CUDA_VISIBLE_DEVICES": "0",
    }


def _cli(args, log):
    with open(log, "w") as f:
        subprocess.run(
            [EMBENCH_PY, "-m", "vista_skill.integrations.embodiedbench.cli", *args],
            cwd=str(REPO), env=_env(), stdout=f, stderr=subprocess.STDOUT,
        )


def build_arm_a_artifact() -> Path:
    path = OUT / "arm_a_faulty_skill.json"
    if path.exists():
        return path
    faulty = replace(
        inject_skill_fault(initialize_shared_skill(), FaultType(FAULT)),
        frozen=True,
    )
    save_skill_artifact(
        path,
        faulty,
        protocol={
            "purpose": "fault_repair_arm_a",
            "fault": FAULT,
            "diagnostic": True,
            "note": "injected-fault lower bound; evaluate with --diagnostic",
        },
    )
    print(f"[built] arm A faulty skill artifact ({FAULT}): {path}", flush=True)
    return path


def run_experiment(method: str) -> None:
    out = OUT / method
    if (out / "experiment_manifest.json").exists():
        print(f"[skip] experiment {method} (already done)", flush=True)
        return
    if out.exists():
        import shutil

        shutil.rmtree(out)
    args = [
        "experiment", "--method", method, "--config", CONFIG, "--manifest", MANIFEST,
        "--diagnostic", "--skill-fault", FAULT,
        "--max-acquisition-episodes", str(ACQ), "--evolution-seeds", "0",
        "--executor-base-url", BASE_URL, "--output-dir", str(out),
        "--method-model", MODEL, "--method-base-url", BASE_URL,
    ]
    print(f"[run] experiment {method} (fault={FAULT}, acq={ACQ}) ...", flush=True)
    _cli(args, f"/tmp/fault_repair_exp_{method}.log")
    if not (out / "experiment_manifest.json").exists():
        print(f"[FAIL] experiment {method}; see /tmp/fault_repair_exp_{method}.log", flush=True)


def run_eval(label: str, skill: str | None) -> None:
    output = OUT / "eval" / f"{label}.jsonl"
    if output.with_suffix(".summary.json").exists():
        print(f"[skip] evaluate {label} (already done)", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "evaluate", "--mode", "frozen_skill", "--config", CONFIG,
        "--manifest", MANIFEST, "--diagnostic", "--stage", "official_test",
        "--eval-set", EVAL_SET, "--max-episodes", str(EVAL_N),
        "--seed", str(EVAL_SEED), "--executor-base-url", BASE_URL,
        "--output", str(output),
    ]
    if skill:
        args += ["--skill", skill]
    print(f"[run] evaluate {label} ...", flush=True)
    _cli(args, f"/tmp/fault_repair_eval_{label}.log")
    if not output.with_suffix(".summary.json").exists():
        print(f"[FAIL] evaluate {label}; see /tmp/fault_repair_eval_{label}.log", flush=True)


def _lineage_summary(method: str) -> str:
    path = OUT / method / "seed_0" / "lineage.jsonl"
    if not path.exists():
        return "n/a"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    accepted = sum(1 for r in records if r.get("accepted"))
    return f"{accepted}/{len(records)} accepted"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    arm_a = build_arm_a_artifact()
    run_experiment("full")
    run_experiment("vista_without_vtca")
    run_eval("arm_a_corrupted_frozen", str(arm_a))
    run_eval("arm_b_full_evolved", str(OUT / "full" / "seed_0" / "frozen_skill.json"))
    run_eval(
        "arm_c_wo_vtca_evolved",
        str(OUT / "vista_without_vtca" / "seed_0" / "frozen_skill.json"),
    )

    print("\n=== fault-repair effectiveness ({} fault, eval {} {}eps seed {}) ===".format(
        FAULT, EVAL_SET, EVAL_N, EVAL_SEED), flush=True)
    rows = []
    for label in ("arm_a_corrupted_frozen", "arm_b_full_evolved", "arm_c_wo_vtca_evolved"):
        summary = OUT / "eval" / f"{label}.summary.json"
        if summary.exists():
            d = json.loads(summary.read_text())
            rows.append((label, d["mean_task_success"], d["mean_task_progress"]))
        else:
            rows.append((label, None, None))
    print(f"  {'reference S0 (E6)':34s} success=0.450  progress=0.533")
    print(f"  {'reference no_skill (E6)':34s} success=0.550  progress=0.633")
    for label, success, progress in rows:
        s = f"{success:.3f}" if success is not None else " n/a"
        p = f"{progress:.3f}" if progress is not None else " n/a"
        print(f"  {label:34s} success={s}  progress={p}")
    print(f"\n  lineage: full {_lineage_summary('full')} · "
          f"vista_without_vtca {_lineage_summary('vista_without_vtca')}")
    a = rows[0][1]
    b = rows[1][1]
    if a is not None and b is not None:
        print(f"  fault drop (S0 - A) = {0.450 - a:+.3f} · recovery (B - A) = {b - a:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
