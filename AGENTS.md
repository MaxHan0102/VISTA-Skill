# Repository Guidelines

## Research Goal & Source of Truth

VISTA-Skill is a CVPR 2027 research project on visual transition credit assignment for reliable Skill evolution under partial observability. **Implementation status changes constantly — do not rely on this file for it; check the code and docs yourself.** The authoritative design lives in the Chinese docs under `context4agent/markdown/`; for what is currently implemented/verified, read `docs/implementation.md` and the root `README.md` and inspect `vista_skill/` (the method integrates with EmbodiedBench via non-invasive adapters, not by editing `EmbodiedBench/planner/`). Changes should advance paper claims and controlled experiments, not unrelated benchmark development. Use EB-Habitat as the primary environment, `EmbodiSkill*` and `EmbodiSkill*+Common Gate` as controlled baselines, and frozen Qwen3-VL-8B-Instruct on RTX 4090 as executor. Match executor, teacher, initial Skill, and evolution budget across comparisons; report performance, update reliability, subgroup regression, and teacher cost.

## Server-Side LaTeX Is Reference-Only

Everything under `context4agent/latex/` is a stale server-side reference copy.
Do not edit, format, generate into, build from, or otherwise modify any file in
that directory. The paper is maintained in a separate local LaTeX project that
is not part of this workspace. Findings that may affect the paper belong in
`docs/` as implementation or experiment notes. Do not modify this server-side
LaTeX tree even when a task concerns paper claims; those edits belong in the
separate local project. This prohibition remains in force unless the user first
explicitly revokes this repository policy.

## Project Structure & Module Organization

`EmbodiedBench/` contains the stock benchmark's runnable Python code: Hydra configs in `embodiedbench/configs/`, agent logic in `planner/`, episode loops in `evaluator/`, and simulator adapters in `envs/`. `vista_skill/` holds the VISTA-Skill method itself (core VTCA, skill evolution, controlled baselines, experiment protocol, and EmbodiedBench adapters under `vista_skill/integrations/`); `configs/`, `tests/`, `scripts/`, and `docs/` hold its protocol configs, test suite, tooling, and status notes. Outputs belong under `running/`. `context4agent/` holds figures, references, design notes, and a stale reference-only paper copy under `latex/`. `EmbodiedBench/habitat-lab/` is an inlined upstream dependency; avoid unrelated edits there.

## Experiment Output Placement and Version Control

Use repository-root-relative paths below, regardless of the shell's working
directory. When launching from `EmbodiedBench/`, pass `../running/...` or an
absolute path for VISTA-owned output. Choose the category before launching;
do not create loose experiment directories directly under `running/`.

| Category | Canonical location |
|---|---|
| VISTA method experiments, including phase-specific diagnostics | `running/PhaseN/<experiment_id>/` (Phase1--Phase5; current method work is Phase5) |
| General smoke tests, mock/fixed-policy checks, resume fixtures | `running/test/<test_id>/` |
| Closed-model feedback experiments | `running/closed_feedback/<run_id>/` |
| Migration manifests, hash maps and local original backups | `running/migrations/<migration_id>/` |
| Native EmbodiedBench raw evaluator output | `EmbodiedBench/running/<environment>/<experiment_id>/` |
| Existing native navigation baseline console/analysis records | `running/embodiedbench_native_nav/<experiment_id>/` |

Keep run metadata, trajectories, metrics, API audit records and image references
under the owning experiment directory when the adapter supports it. Stock
simulator image output may require an adapter/output-path override; do not
change benchmark scoring or planner behavior to reorganize files. Record any
unavoidable separate raw-image location. Phase-owned historical simulator
images live under `running/PhaseN/simulator/`. See `docs/experiment_output_layout.md`.

Resume within the original run's `_resume/attempt_<id>/` directory. Resume
fixtures belong in `running/test/`, not alongside live experiments. Do not move,
rewrite or overwrite a directory another session is using; in particular keep
live `running/closed_feedback/` data in place. Keep the first complete result
and its provenance; do not silently replace failed trials with successful ones.

For a requested relocation, update stored paths and active script/documentation
references rather than leaving compatibility symlinks. Back up changed originals,
record old/new paths and hashes under `running/migrations/`, validate referenced
files, and preserve scores, actions and experiment conclusions. New storage
hashes do not constitute a new preregistration. Never rewrite the reference-only
`context4agent/latex/` tree.

Non-image experiment records in **both** `running/` and `EmbodiedBench/running/`
are eligible for Git: JSON/JSONL, configs, logs, summaries, Skill artifacts and
migration manifests. The previous blanket ban on generated `running/` outputs
no longer applies. Images (including extensionless `audit/input_images/` caches),
backup archives and runtime lock files stay ignored by `.gitignore`. Preserve
image paths/hashes in records even though the image bytes are stored locally.
Never commit credentials, downloaded datasets or simulator assets; check API
request/response logs for secrets before staging them. Enabling tracking does
not itself authorize committing or pushing outputs.

## Build, Test, and Development Commands

Run benchmark commands from `EmbodiedBench/`.

- `bash install.sh`: create simulator-specific Conda environments and install assets.
- `pip install -e .`: install the package in the active environment.
- `python -m embodiedbench.main env=eb-hab model_name=gpt-4o-mini exp_name=smoke down_sample_ratio=0.1`: run a short Hydra evaluation.
- `python -m embodiedbench.envs.eb_habitat.EBHabEnv`: smoke-test Habitat; equivalent entry points cover the other environments.
- VISTA-Skill method commands (experiment/evaluate CLI, vLLM serving, method smoke): these live in `vista_skill/` and `scripts/` and evolve over time — consult `docs/implementation.md` and the root `README.md` for current usage rather than assuming flags.

The conda environments configured on this machine are `max_embench` (`eb-alf`/`eb-hab`), `max_embench_nav` (`eb-nav`), and `max_vllm` (serves Qwen3-VL-8B-Instruct via vLLM); `eb-man`/CoppeliaSim is not yet configured. Activate the matching env before running (stock `install.sh` creates `embench`/`embench_nav`/`embench_man`). Start the documented headless X server when required.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/modules, and `PascalCase` classes, following nearby code. Keep `eb-alf`, `eb-hab`, `eb-nav`, and `eb-man` consistent across configs and dispatch code. No repository-wide formatter is configured; avoid broad reformatting. Do not run formatters over the reference-only `context4agent/latex/` tree.

## Testing Guidelines

There is a VISTA-Skill test suite under `tests/` (run with `pytest`) with no fixed coverage threshold, plus per-environment EmbodiedBench smoke tests. The exact suite contents evolve — consult `tests/` and `docs/implementation.md` for what currently exists. Run the affected environment smoke test and a small evaluation when touching simulator paths, recording the environment, model, flags, seed, metrics, and result path. `habitat-lab/test/` covers only that dependency; use targeted `pytest habitat-lab/test/test_<area>.py` runs when modifying it. Name new tests `test_<behavior>.py`.

## Commit & Pull Request Guidelines

Use short imperative commit summaries and keep commits focused. PRs should identify the research question or engineering scope, link the governing design section or issue, list validation commands, and include metrics or visual evidence. Non-image experiment records may be committed under the output policy above; never commit API keys, downloaded datasets, simulator assets, ignored images, backup archives or runtime locks.
