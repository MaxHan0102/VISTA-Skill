# VISTA-Skill P0 Implementation

For a short, plain-language Chinese summary of the full experimental history
and current conclusions, see [experiment_log_plain_zh.md](experiment_log_plain_zh.md).
The post-hoc human Target-Skill diagnostic and its frozen validation are recorded
separately in [experiment_log_phase4.md](experiment_log_phase4.md).

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

If an `experiment` process is interrupted after acquisition while
`update_audit.json` is still absent, resume only the independent audit with:

```bash
PYTHONPATH=EmbodiedBench:. python scripts/resume_phase2_update_audit.py \
  --run-dir running/<campaign>/seed_0 \
  --config configs/<matching-config>.json \
  --manifest configs/eb_hab_train_validation_manifest.json \
  --executor-base-url http://127.0.0.1:8000/v1
```

The command verifies the config/manifest/split digests recorded in lineage and
loads only rollout JSONL files containing a complete `episode_result` as disk
cache. An interrupted JSONL is retained; its replacement is written as a new
`.resumeN.jsonl` artifact. The command refuses to run once `update_audit.json`
exists and never reruns acquisition or proposal generation.

`evaluate --stage official_test --eval-set <subset>` supports each of the six
stock EB-Hab test subsets. `configs/methods.json` records implementation status:
the Full VISTA workflow, the frozen No Skill/Static controls, **and the three
trajectory-level controlled baselines** (`embodiskill_star_native`,
`embodiskill_star_common_gate`, `vista_without_vtca`) all have full CLI
experiment wiring. The trajectory baselines share the acquisition → evolve →
freeze → audit skeleton with the full method but evolve from whole-episode
reflections: a `JsonTrajectoryTeacher` (same `--method-model` backend, so
teacher model/calls/tokens stay matched) feeds an `EmbodiSkillFrontend` or
`UnconditionalReflectionFrontend`; proposals are routed through
`EmbodiSkillNativeUpdater` (native body/appendix semantics) or the identical
VISTA `CandidateGate` via `CommonGateProposalAdapter`
(`vista_skill/workflow.py::TrajectoryEvolutionWorkflow`). A proposal fires only
after `min_independent_episodes` distinct failed episodes support the same
attributed field — the trajectory analogue of action-level recurrence — so
candidate count and validation budget remain comparable across methods. Each
baseline run emits the same artifact set as the full method (acquisition.jsonl,
lineage.jsonl, update_proposals/, update_audit.json, gate_rollouts/ for the
common-gate variants, frozen_skill.json, run_manifest.json,
experiment_manifest.json).

`run_manifest.json` additionally records `executor_usage` (acquisition-phase
executor calls and prompt/completion tokens), captured by the seed wrapper in
`planner.py` without modifying EmbodiedBench; it is `null` for non-remote
(`local`/`custom`) executors, which cannot be seed-controlled in a controlled
run.

Supplementary diagnostics for RQ2 and the Phase-2 evidence Go/No-Go are provided
as library drivers: `vista_skill/fault_injection.py::run_fault_injection_evaluation`
(target/field Macro-F1, abstention quality, confusion matrix over injected
faults), `vista_skill/evidence_oracle.py` (`OracleEvidenceProvider`,
`NoisyEvidenceProvider`, `evaluate_calibration`, `selective_risk_curve`,
`compare_providers`), and `vista_skill/skills.py::minimal_shared_skill` /
`empty_shared_skill` for the §4.2.3 initialization-sensitivity comparison.

The core tests exercise model and simulator ports with deterministic fakes. A
real EB-Hab rollout additionally requires simulator assets, a working headless
EGL context, and model endpoints; these are not bundled with this package.

The earlier 2026-08-12 EGL failure is superseded on this host.  As of
2026-08-25, real EB-Hab rollouts and frozen Qwen3-VL-8B endpoint probes have
completed for Phase3A.  The evaluation-only Habitat state oracle now clears
Habitat's predicate truth cache before every observation; this is required
because action precondition checks otherwise leave pre-action values cached for
post-action queries.  A 10-episode non-interference run matched an oracle-off
reference exactly over 102 transitions and mapped all 624 post queries.

Phase3A then collected 40 natural and 20 fixed-script stress episodes and
froze a 300-transition, scene-disjoint dev/selection/audit dataset at
`running/phase3a/phase3a_dataset_v3_cachefix_20260825.json`.  Two 300-call
Qwen caches cover feedback-conditioned and images-only evidence.  The final
Guard v2 adds outcome-aware rejection, failed-action temporal persistence,
action-local visual relevance, conservative negative spatial relations, and
successful-place grounding without reading Skill predictions or expected
deltas.  It uses the same one feedback-conditioned VLM call as the current
method, so live `--evidence-guard strict|authority_aware` does not add a second
visual call.

The frozen audit result is
`running/phase3a/phase3a_guard_v2_frozen_audit_v4_20260825.json`.  It is a
pre-registered **No-Go**: false contradictions fell from 0.00738 to zero and
Skill-update recall stayed at 1.0, but the paired 95% CI included zero and
coverage fell by 14.85 percentage points versus the matched threshold arm.
Consequently the conditional live pilot and Phase3A scale-up were not run.
The complete chronology, including the oracle-cache failure and the reporting-
only cost amendment, is in `docs/experiment_log_phase3.md`.

Phase3B then tested feedback dependence and strict no-feedback temporal RGB with
the same frozen Qwen3-VL-8B executor.  The 30-task four-arm executor pilot found
that three-frame history recovered the no-feedback mean progress loss
(`0.5913 -> 0.6347`, versus feedback/current `0.6389`) but failed the frozen
adjacent-repetition burden gate (`0.0495 > 0.0359`).  A separately isolated
two-frame/three-frame evidence study removed raw feedback, success flags, and
the feedback-derived pre-ledger from both prompts.  On dev, temporal evidence
was precise when asserted (`0.9655`) but had only `0.0544` coverage,
`0.0157` contradiction recall, and no Skill-update recall; its predicate F1
fell below the strict pair arm (`0.0966` versus `0.1523`).  Phase3B is therefore
a pre-registered No-Go: selection metrics, fresh episodes 60--79, fresh audit,
and late-feedback fusion were not opened.  The immutable decision artifact is
`running/phase3b/phase3b_final_decision_20260825.json`; full causality and cost
records are in `docs/experiment_log_phase3.md`.

Phase3C tested whether three short, frozen, environment-neutral Meta-Skills
could improve safe Skill evolution without training.  The bundle
(`phase3c_frozen_v1`, 423 whitespace-token proxy, SHA256
`ec349ec8bbb0a6dced55267fc0d44c0259a4772f384b3adab765fa0faeff43cc`)
is wired behind diagnostic-only `--meta-skills frozen_v1`; default behavior is
unchanged.  Full tests passed (235), and the local frozen Qwen3-VL-8B endpoint
passed the 6/6 protocol probe.

The frozen offline gate is a **No-Go**.  Meta attribution reduced synthetic
target Macro-F1 from 0.5279 to 0.4321 and field Macro-F1 from 0.9387 to 0.2963;
natural Skill-update F1 fell from 0.5912 to 0.1538 (multihold) and from 0.5821
to 0.4286 (effect inversion).  Existing rule-first routing prevented any
regression but also received no benefit.  Both patch faults passed 10/10 and
metamorphic agreement was 0.9444, but the current compiled-rule correction was
already 10/10, so this is non-regression rather than incremental value.  The
matched core Meta arm used 1.4313x the teacher/patch tokens.  Per the
pre-registration, EB-Hab executor/live evolution and EB-Nav zero-shot branches
were cancelled and Phase3C v1 is not enabled in the main method.  The immutable
decision is `running/phase3c/offline/phase3c_offline_decision.json`; detailed
causality and limitations are in `docs/experiment_log_phase3.md`.

Phase4 P4-O1 added two frozen, five-field, post-hoc human Target Skills for
EB-Habitat and EB-Navigation.  They were synthesized from historical
`official_test/base` No-Skill/Static-Skill trajectories and therefore are
diagnostic references, not controlled evolution results.  On a disjoint
20-episode `official_test/common_sense` diagnostic, the Target macro-average
success was 0.425 versus 0.350 No Skill and 0.450 Static S0.  It substantially
changed the intended local mechanisms (lower aggregate invalid-action ratios;
NAV maximum open-loop length 1.0 in all 20 episodes) but did not establish a
primary-metric gain and increased NAV planner calls to 14.25/episode.  The
artifacts, paired uncertainty analysis, failure provenance, and implications
for concise/compiled Skill evolution are in `docs/experiment_log_phase4.md`.
