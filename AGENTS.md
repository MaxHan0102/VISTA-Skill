# Repository Guidelines

## Research Goal & Source of Truth

VISTA-Skill is a CVPR 2027 research project on visual transition credit assignment for reliable Skill evolution under partial observability. The method exists in design documents and LaTeX; runnable code is stock EmbodiedBench. Changes should advance paper claims and controlled experiments, not unrelated benchmark development. Use EB-Habitat as the primary environment, `EmbodiSkill*` and `EmbodiSkill*+Common Gate` as controlled baselines, and frozen Qwen3-VL-8B-Instruct on RTX 4090 as executor. Match executor, teacher, initial Skill, and evolution budget across comparisons; report performance, update reliability, subgroup regression, and teacher cost. Integrate through `embodiedbench/planner/` and evaluator step loops.

## Project Structure & Module Organization

`EmbodiedBench/` contains runnable Python code: Hydra configs live in `embodiedbench/configs/`, agent logic in `planner/`, episode loops in `evaluator/`, and simulator adapters in `envs/`. Outputs belong under `running/`. `context4agent/` holds the paper (`latex/`), figures, references, and design notes. `EmbodiedBench/habitat-lab/` is an inlined upstream dependency; avoid unrelated edits there.

## Build, Test, and Development Commands

Run benchmark commands from `EmbodiedBench/`.

- `bash install.sh`: create simulator-specific Conda environments and install assets.
- `pip install -e .`: install the package in the active environment.
- `python -m embodiedbench.main env=eb-hab model_name=gpt-4o-mini exp_name=smoke down_sample_ratio=0.1`: run a short Hydra evaluation.
- `python -m embodiedbench.envs.eb_habitat.EBHabEnv`: smoke-test Habitat; equivalent entry points cover the other environments.
- `cd context4agent/latex && latexmk -pdf main.tex`: build the paper with the CVPR LaTeX dependencies.

Use `embench` for `eb-alf`/`eb-hab`, `embench_nav` for `eb-nav`, and `embench_man` for `eb-man`. Start the documented headless X server when required.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/modules, and `PascalCase` classes, following nearby code. Keep `eb-alf`, `eb-hab`, `eb-nav`, and `eb-man` consistent across configs and dispatch code. No repository-wide formatter is configured; avoid broad reformatting. Keep LaTeX sections in `sec/` with descriptive labels.

## Testing Guidelines

There is no dedicated VISTA-Skill suite or coverage threshold. Run the affected environment smoke test and a small evaluation, recording the environment, model, flags, seed, metrics, and result path. `habitat-lab/test/` covers only that dependency; use targeted `pytest habitat-lab/test/test_<area>.py` runs when modifying it. Name new tests `test_<behavior>.py`.

## Commit & Pull Request Guidelines

Use short imperative commit summaries and keep commits focused. PRs should identify the research question or engineering scope, link the governing design section or issue, list validation commands, and include metrics or visual evidence. Never commit API keys, downloaded datasets, simulator assets, or generated `running/` outputs.
