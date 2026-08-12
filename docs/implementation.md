# VISTA-Skill P0 Implementation

## Source of truth

The implementation follows the 2026-08-06 execution design. That version
narrows the primary system to three routes (`belief_refresh`, `skill_update`,
and `abstain`), a sparse typed predicate ledger, and a fixed action schema.
The broader four-memory formulation in the current LaTeX is represented by a
reserved `action_model_update` enum, but persistent action-model evolution is
disabled in P0.

## Runtime data flow

```text
frozen planner + active Skill + local evidence ledger
                         |
                    primitive action
             +-----------+-----------+
             |                       |
 fixed schema + Skill          pre/post RGB + public feedback
 expected transition           isolated evidence transition
             |                       |
             +------ typed mismatch--+
                         |
              rule-first target routing
         belief refresh / skill update / abstain
                         |
        recurrent independent skill-field evidence
                         |
      exact-target patch -> staged paired candidate gate
```

`vista_skill.pipeline.VistaSkillEngine` owns the method loop but not a
simulator. `HabitatRolloutRunner` wraps stock `EBHabEnv` and `VLMPlanner` from
outside EmbodiedBench, creating one event for every primitive action in a
multi-action plan.

The expected branch is explicitly materialized by `engine.prepare(...)` before
`env.step`. Only after the post-action image and public feedback exist does the
runner call `engine.process_prepared(...)`. This order is part of the
information-isolation contract, not merely an implementation convention.

## Enforced invariants

- `EvidenceRequest` has no Skill, expected transition, mismatch, attribution,
  patch, selection, or audit fields. The visual provider can only serialize
  this request.
- The belief ledger accepts only `PredicateEvidence`; predicted deltas and
  teacher decisions cannot mutate belief.
- Missing coverage and `unknown` produce unsupported/uncovered mismatches, not
  contradictions.
- Predicate normalization preserves instance digits such as `apple_1`.
- Every predicted change cites a fixed action-rule ID or a Skill version/rule
  and field.
- The five-field Skill includes compiled field prediction rules and a typed
  termination policy. A bounded patch updates textual and compiled views in
  one version so execution and attribution cannot silently diverge.
- S0 compiles observable procedure/effect/constraint obligations for the fixed
  nav/pick/place/open/close schema. Activation remains a trajectory-level
  diagnostic field because applicability is not a primitive state effect.
- Skill updates need independent episodes and unique evidence IDs.
- A patch touches exactly one attributed field. Exact-target operations fail
  closed when the old statement is absent.
- The candidate gate checks cached transition repair, paired bootstrap lower
  confidence bounds, and worst protected-subgroup regression. Repeated rollout
  seeds are averaged within task before bootstrap resampling.
- Acquisition checks recurrent clusters after every episode. An accepted
  version is promoted immediately, so later episodes execute and collect
  evidence against the new Skill rather than batching every event under v0.
- Runner attribution context uses only the manifest/episode instruction ID,
  public instruction, action arguments, and episode-local ledger. Necessary
  precondition checks make execution-lapse routing reachable without Habitat
  goal predicates or task-progress oracles.
- `freeze()` disables expected transitions, mismatch, attribution, clustering,
  and evolution. Frozen execution can still update the episode-local compact
  ledger from rule-based public feedback, without a teacher call.
- Habitat task progress, success, subgoal reward, and dataset goal predicates
  are logged as evaluation labels only. They never enter the evidence request.

## Package boundaries

- `schemas.py`: strict domain DTOs and enum invariants.
- `belief.py`: sparse three-valued ledger and provenance-aware merging.
- `action_schema.py`: deterministic EB-Hab style primitive effects and compiled
  Skill prediction rules.
- `evidence.py` / `models.py`: feedback-first evidence with an optional isolated
  pre/post visual provider.
- `mismatch.py` / `attribution.py` / `clustering.py`: typed differences,
  hierarchical credit assignment, and recurrence.
- `evolution.py` / `lineage.py`: bounded patches, paired gate, and append-only
  accepted/rejected provenance.
- `baselines.py`: controlled EmbodiSkill trajectory routing and no-VTCA
  frontends that can share the same update backend.
- `integrations/embodiedbench/`: optional stock environment/planner adapters.

## Experiment protocol

`configs/vista_p0.json` records thresholds, budgets, and the three evolution
seeds. `configs/eb_hab_train_validation_manifest.json` pins all 100 episode
coordinates and the dataset SHA-256. A release-grade run must additionally
persist exact model revisions, model server settings, prompt/skill/schema
hashes, teacher usage, candidate-evaluation episodes, wall time, and GPU hours.

The public 100-task `train_validation` dataset is divided at episode/task
coordinate level into 60 acquisition, 20 selection, and 20 frozen audit tasks.
Each configured evolution seed creates a fresh engine, ledger, clusterer,
lineage, model-usage record, and frozen artifact under `seed_<n>/`. Roles rotate
by one stable 20-task block per run: offsets 0, 20, and 40 for the three P0
runs. Within each rotated selection role, proxy and finalist use disjoint task
pools; the finalist pool uses three derived rollout seeds. Parent/candidate
runs use the same task, rollout seed, order, temperature, and executor
checkpoint.

After acquisition, evolution, and freeze have finished, every statically valid
proposal is evaluated on that run's rotated held-out audit role. Digest-checked
parent/candidate snapshots record whether the gate promoted the proposal. The
post-hoc report computes task-first overall and subgroup deltas,
beneficial-update precision, harmful-update rate, and missed-beneficial-update
rate. Audit results never feed back into selection or patching.

Skill artifact schema v2 hashes the complete `schema_version + skill +
protocol` envelope. Controlled evaluation restores and validates the artifact's
split rotation and checks config/manifest hashes, frozen state, executor name
and type, tensor parallel setting, n-shots, resolution, temperature, and token
budget before constructing Habitat. Controlled evaluation also rejects
diagnostic or reduced-acquisition artifacts. `--diagnostic` permits runtime protocol
deviations but never bypasses artifact-integrity hashing. Existing output paths
are rejected, and Habitat image namespaces include unique run, task, rollout
seed, and Skill identifiers. Python, NumPy, Torch, Habitat, and compatible
executor requests receive that rollout seed before construction/inference;
the model server must support the OpenAI `seed` field. Transition records
include pre/post image hashes.

The standalone CLI is run with both repositories importable. Full acquisition,
evolution, paired selection, lineage, promotion, and freeze use:

```bash
PYTHONPATH=EmbodiedBench:. python -m vista_skill.integrations.embodiedbench.cli \
  experiment --method full \
  --model-name Qwen/Qwen3-VL-8B-Instruct \
  --executor-base-url http://127.0.0.1:8000/v1 \
  --method-model Qwen/Qwen3-VL-8B-Instruct \
  --method-base-url http://127.0.0.1:8000/v1
```

Frozen audit never constructs an attribution or patch teacher:

```bash
PYTHONPATH=EmbodiedBench:. python -m vista_skill.integrations.embodiedbench.cli \
  evaluate --mode frozen_skill \
  --skill running/vista_skill/full/seed_0/frozen_skill.json --stage audit
```

`experiment` also writes `seed_<n>/update_audit.json`, proposal snapshots under
`seed_<n>/update_proposals/`, per-seed lineage/usage, and a top-level
`experiment_manifest.json`. A fresh `--output-dir` is required for every
invocation.

`evaluate --stage official_test --eval-set <subset>` supports each of the six
stock EB-Hab test subsets. `configs/methods.json` records implementation status:
the Full VISTA workflow and frozen No Skill/Static controls have CLI wiring;
the three trajectory-level baselines currently expose tested frontend/updater
library ports, with their CLI experiment orchestration still marked pending.

The core tests exercise model and simulator ports with deterministic fakes. A
real EB-Hab rollout additionally requires simulator assets, a working headless
EGL context, and model endpoints; these are not bundled with this package.

The latest local simulator initialization attempt (2026-08-12, `embench`, one
manifest episode at 64 px) loaded `train_validation.pickle` and reached
Habitat-Sim construction, then failed before reset because the host windowless
EGL context could not map CUDA device 0. Therefore no real simulator rollout or
Qwen endpoint smoke is claimed by this implementation snapshot.
