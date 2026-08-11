# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

VISTA-Skill is a research project (CVPR 2027 target) studying **visual transition credit assignment for reliable skill evolution** in embodied agents under partial observability. It is being developed **on top of** EmbodiedBench, an ICML 2025 benchmark for multi-modal embodied agents. The repo has two disjoint parts:

- `context4agent/` — the *paper*: LaTeX (`latex/`, main + `sec/`), reference PDFs (`PDF/`), and **authoritative design docs in Chinese** (`markdown/`). The newest design doc is `20260806-VISTA-Skill最新方案-证据解耦视觉转移信用分配.md`. These docs state the method, baselines, and current project decisions.
- `EmbodiedBench/` — the *runnable benchmark*. It was a git submodule and is now inlined directly (see commit `59b7023`). All actual code lives here.

**Important:** the VISTA-Skill method itself is not yet implemented in code — it exists only as design docs + LaTeX. The code you can run today is stock EmbodiedBench. The natural integration point for the method is the planner layer (`EmbodiedBench/embodiedbench/planner/`) and the evaluator's per-step loop, where skill-predicted vs. evidence-extracted transitions and attribution would be wired in. Primary target environment is **EB-Habitat**; main baseline is EmbodiSkill; core model is Qwen3-VL-8B-Instruct on RTX 4090.

## Running evaluations (EmbodiedBench)

There are **three separate conda environments**, one per simulator family — you must activate the right one for the environment you target:

| env flag | conda env | simulator |
|----------|-----------|-----------|
| `eb-alf`, `eb-hab` | `embench` | AI2-THOR / Habitat-Sim |
| `eb-nav` | `embench_nav` | AI2-THOR |
| `eb-man` | `embench_man` | CoppeliaSim + PyRep |

First-time setup: `cd EmbodiedBench && bash install.sh` (creates all three envs, pulls datasets/sim assets, installs habitat-lab, CoppeliaSim, PyRep). Requires `git lfs`. The conda env definitions live in `EmbodiedBench/conda_envs/`.

AI2-THOR and Coppelia need a headless X server. Start it once in a separate tmux pane (default `X_DISPLAY=:1`):
```bash
python -m embodiedbench.envs.eb_alfred.scripts.startx 1
```

Run an evaluation (run from inside `EmbodiedBench/`, with the matching env activated):
```bash
conda activate embench
python -m embodiedbench.main env=eb-hab model_name=gpt-4o-mini exp_name=baseline
```
CLI args use **Hydra `key=value`` syntax and override the per-env YAML. Useful flags: `down_sample_ratio=0.1` (fast debug run on 10% of data), `eval_sets=[<subset>]` (single capability subset), `language_only=True` (text-only), `chat_history=True`, `n_shots=N`, `multiview`/`multistep`/`visual_icl`, `resolution`, `log_level=DEBUG`.

There is **no test suite**. "Does it work?" is answered by the per-environment smoke-test entry points in the README, e.g. `python -m embodiedbench.envs.eb_habitat.EBHabEnv`. Results and rendered frames are written under `EmbodiedBench/running/`.

### Model types
`model_type=` selects how the model is called:
- `remote` (default) — OpenAI-style API; set `OPENAI_API_KEY` / `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` / `DASHSCOPE_API_KEY`.
- `local` — offline via LMDeploy (separate `lmdeploy` conda env from `conda_envs/lmdeploy.yaml`); set `tp` for tensor parallelism.
- `custom` — a self-hosted Flask server (`EmbodiedBench/server.py`, currently `gemma-3-12b-it`); set `export server_url="IP:port/process"`.

## Architecture (EmbodiedBench)

The execution pipeline is **Hydra config → Evaluator → Environment + Planner loop**:

1. `embodiedbench/main.py` — Hydra entrypoint. Loads `configs/config.yaml` (schema of all CLI-overridable fields), then merges it with the per-env base (`configs/eb-{alf,hab,nav,man}.yaml`) and your CLI overrides, picks the evaluator class dynamically via `get_evaluator()`, and runs `evaluator.evaluate_main()`. Note it creates a `data` symlink for habitat on import.
2. `embodiedbench/evaluator/` — four `EB_*Evaluator` classes. Each owns an env instance and a `VLMPlanner`. `evaluate_main()` iterates episodes; `evaluate()` runs the per-episode loop: reset → for each step, `planner.act(obs, instruction)` returns an action, the env steps, `planner.update_info()` ingests feedback. `evaluator/config/system_prompts.py` and `*_examples.json` hold the prompts and in-context examples; `evaluator/config/visual_icl_examples/` holds annotated images for visual ICL.
3. `embodiedbench/envs/` — four Gym-style environments, each with an `EB*Env.py`: `eb_alfred` (AI2-THOR), `eb_habitat` (Habitat-Sim + `habitat-lab/`), `eb_navigation` (AI2-THOR), `eb_manipulation` (CoppeliaSim via `amsolver/`). Large datasets are NOT in the repo — EB-ALFRED, EB-Habitat assets, and EB-Manipulation data are cloned from HuggingFace during install.
4. `embodiedbench/planner/` — agent/action-selection layer, the most relevant surface for VISTA-Skill work:
   - `vlm_planner.py` (`VLMPlanner`) — base agent: builds prompts from system prompt + n-shot examples + available actions, calls the model, parses JSON → action, optionally keeps chat history.
   - `manip_planner.py`, `nav_planners.py` — env-specific subclasses.
   - `remote_model.py` / `custom_model.py` — model backends behind `model_type`.
   - `planner_config/generation_guide.py` — JSON output schema the model is told to produce; `planner_utils.py` has `fix_json` for repair. If you change the expected action/output schema, update it in both the guide and the env's action set.

## Conventions and gotchas

- Always run from the `EmbodiedBench/` directory (module paths and the `data` symlink assume it as CWD). Activate the conda env matching your `env=` first.
- The four env names appear as `eb-alf` / `eb-hab` / `eb-nav` / `eb-man` in CLI flags, YAML filenames, and evaluator dispatch tables in `main.py` — keep these consistent when adding envs.
- Do not enable several image-heavy flags (`visual_icl`, `multiview`, `multistep`, `chat_history`) at once — the README warns of conflicts and excessive image input.
- The `truncate` flag only affects EB-Navigation and only when `chat_history=True`.
- `context4agent/` design docs are in Chinese and are the source of truth for research direction, scope cuts, and decisions (e.g. why graph–skill co-evolution was descoped, why Skill-Pro PPO gates were dropped for a bounded-patch + paired-gate design). Read them before making method-level changes.
