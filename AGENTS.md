# Repository Guidelines

## Research Goal & Source of Truth

VISTA-Skill is a CVPR 2027 research project on visual transition credit assignment for reliable Skill evolution under partial observability. **Implementation status changes constantly — do not rely on this file for it; check the code and docs yourself.** The authoritative design lives in the Chinese docs under `context4agent/markdown/`; for what is currently implemented/verified, read `docs/implementation.md` and the root `README.md` and inspect `vista_skill/` (the method integrates with EmbodiedBench via non-invasive adapters, not by editing `EmbodiedBench/planner/`). Changes should advance paper claims and controlled experiments, not unrelated benchmark development. Use EB-Habitat as the primary environment, `EmbodiSkill*` and `EmbodiSkill*+Common Gate` as controlled baselines, and frozen Qwen3-VL-8B-Instruct on RTX 4090 as executor. Match executor, teacher, initial Skill, and evolution budget across comparisons; report performance, update reliability, subgroup regression, and teacher cost.

## Project Structure & Module Organization

`EmbodiedBench/` contains the stock benchmark's runnable Python code: Hydra configs in `embodiedbench/configs/`, agent logic in `planner/`, episode loops in `evaluator/`, and simulator adapters in `envs/`. `vista_skill/` holds the VISTA-Skill method itself (core VTCA, skill evolution, controlled baselines, experiment protocol, and EmbodiedBench adapters under `vista_skill/integrations/`); `configs/`, `tests/`, `scripts/`, and `docs/` hold its protocol configs, test suite, tooling, and status notes. Outputs belong under `running/`. `context4agent/` holds the paper (`latex/`), figures, references, and design notes. `EmbodiedBench/habitat-lab/` is an inlined upstream dependency; avoid unrelated edits there.

## Build, Test, and Development Commands

Run benchmark commands from `EmbodiedBench/`.

- `bash install.sh`: create simulator-specific Conda environments and install assets.
- `pip install -e .`: install the package in the active environment.
- `python -m embodiedbench.main env=eb-hab model_name=gpt-4o-mini exp_name=smoke down_sample_ratio=0.1`: run a short Hydra evaluation.
- `python -m embodiedbench.envs.eb_habitat.EBHabEnv`: smoke-test Habitat; equivalent entry points cover the other environments.
- `cd context4agent/latex && latexmk -pdf main.tex`: build the paper with the CVPR LaTeX dependencies.
- VISTA-Skill method commands (experiment/evaluate CLI, vLLM serving, method smoke): these live in `vista_skill/` and `scripts/` and evolve over time — consult `docs/implementation.md` and the root `README.md` for current usage rather than assuming flags.

The conda environments configured on this machine are `max_embench` (`eb-alf`/`eb-hab`), `max_embench_nav` (`eb-nav`), and `max_vllm` (serves Qwen3-VL-8B-Instruct via vLLM); `eb-man`/CoppeliaSim is not yet configured. Activate the matching env before running (stock `install.sh` creates `embench`/`embench_nav`/`embench_man`). Start the documented headless X server when required.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/modules, and `PascalCase` classes, following nearby code. Keep `eb-alf`, `eb-hab`, `eb-nav`, and `eb-man` consistent across configs and dispatch code. No repository-wide formatter is configured; avoid broad reformatting. Keep LaTeX sections in `sec/` with descriptive labels.

## Testing Guidelines

There is a VISTA-Skill test suite under `tests/` (run with `pytest`) with no fixed coverage threshold, plus per-environment EmbodiedBench smoke tests. The exact suite contents evolve — consult `tests/` and `docs/implementation.md` for what currently exists. Run the affected environment smoke test and a small evaluation when touching simulator paths, recording the environment, model, flags, seed, metrics, and result path. `habitat-lab/test/` covers only that dependency; use targeted `pytest habitat-lab/test/test_<area>.py` runs when modifying it. Name new tests `test_<behavior>.py`.

## Commit & Pull Request Guidelines

Use short imperative commit summaries and keep commits focused. PRs should identify the research question or engineering scope, link the governing design section or issue, list validation commands, and include metrics or visual evidence. Never commit API keys, downloaded datasets, simulator assets, or generated `running/` outputs.
