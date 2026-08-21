# VISTA-Skill Experiment Log

Long-term maintenance log for every experiment campaign on this project.
One entry per campaign: what was tested, exact settings, results,
interpretation, artifact location, and follow-ups. Append new entries at the
bottom; never edit old entries except to add a `Follow-up` note pointing to a
newer entry that supersedes its conclusions.

Conventions:
- "S0" = the spec-initialized shared Skill (`initialize_shared_skill()`); its
  content digest starts `54b60a62…`. Seeing `frozen=54b60a62` in a run means
  the method evolved nothing.
- Teacher token cost = sum over `method_usage` purposes; executor cost =
  `executor_usage` (planner calls). Both are recorded in each run's
  `run_manifest.json`.
- `running/` is **gitignored**: artifacts are machine-local. Numbers worth
  keeping must be copied into this file (that is its purpose).

Environment snapshot (shared by all entries below, unless noted):
- 2× RTX 4090 D 24 GB, driver 550.90.07 (= CUDA 12.4 ceiling).
- Serving (`max_vllm` env): vLLM 0.11.0, torch 2.8.0+cu128, transformers
  4.57.6 (NOT 5.x), flashinfer **uninstalled**, fp8, tp=1,
  `--max-model-len 16384 --max-num-batched-tokens 4096 --enforce-eager`,
  served name `Qwen/Qwen3-VL-8B-Instruct`, port 8001
  (`scripts/serve_qwen3vl_vllm.sh`). Habitat renders on the other GPU
  (`CUDA_VISIBLE_DEVICES`).
- Rollouts (`max_embench` env): python 3.9, habitat_sim 0.3.0, openai 1.60.
- Executor and teacher are the same local vLLM endpoint (fair by construction).
- Never set `HABITAT_SIM_LOG` to an invalid level (habitat_sim core-dumps).

---

## 2026-08-13 · E1 · Serving bring-up + interface contract validation

**Tested:** that a local vLLM endpoint can satisfy the exact request/response
contract of both model paths (executor `RemoteModel`, teacher
`OpenAICompatibleJsonModel`), with no simulator involved.

**Settings:** `scripts/probe_vllm_endpoint.py` (6 feature checks) +
`scripts/vista_real_model_smoke.py` (3 real method-layer calls).

**Results:** 6/6 and 3/3 pass — chat completions, `seed` determinism,
`response_format=json_schema strict` **with nullable unions** (xgrammar),
multimodal base64 images, usage tokens; real `complete_json`,
`JsonTrajectoryTeacher.reflect`, `JsonVisualEvidenceProvider.extract` all work
against the live server.

**Version findings (hard-won, keep):** driver 550 cannot run CUDA-13 wheels
(vLLM 0.27.x torch 2.13+cu130 fails); Qwen3-VL needs vLLM ≥ 0.11.0; vLLM 0.11
breaks on transformers ≥ 5.0 (`all_special_tokens_extended` removed); stale
flashinfer raises `TypeError` at import and vLLM's guard only catches
`ImportError`.

**Artifacts:** transient logs only; knowledge captured in
`scripts/serve_qwen3vl_vllm.sh` comments.

---

## 2026-08-13 · E2 · Real-rollout smoke ladder (Habitat + vLLM)

**Tested:** the full stack on the real simulator, layer by layer.

**Settings / results:**
1. `EBHabEnv` renders via EGL — reset + frame (170 KB) + step in 4.6 s.
   The long-standing "EGL/CUDA blocker" in older notes was a misdiagnosis.
2. `experiment --method rule_only --diagnostic --max-acquisition-episodes 1`
   (3 seeds): the executor **solved** "Move a ball to the sofa"
   (nav→pick→nav→place, task_success=1.0). Executor usage tracking works live.
3. `--method full` 1 episode: per-action evidence extraction (vLLM, pre+post
   images), rule-first attribution, clustering, per-purpose token billing —
   all verified on real data. Zero spurious `skill_update` on a solved episode.
4. Evolution-teacher path (attribution + bounded patch) vs live vLLM: patch
   generation + static validation pass.
5. Deterministic gate+audit smoke (`scripts/vista_gate_rollout_smoke.py`):
   real patch → static → transition → **paired-proxy Habitat rollouts** →
   lineage/snapshot → post-hoc audit with full reliability report. 4/4.

**Interpretation:** the pipeline is sound end-to-end on real sim + real model.
Every failure found was an integration bug, not a method flaw (see ledger).

**Artifacts:** `running/smoke_*` (local).

---

## 2026-08-13 · E3 · RQ2 attribution falsifiability (fault injection)

**Tested:** the paper's falsifiability hinge — action-level evidence-decoupled
attribution vs trajectory-level routing on a controlled diagnostic set.

**Settings:** `scripts/t3_attribution_falsifiability.py`; 24 balanced
fault cases (clean / procedure / effect / constraint / activation /
termination skill faults; stale-belief, unknown-as-false, identity belief
faults; 3 abstention motifs).

**Results:**

| assigner | target Macro-F1 | field Macro-F1 | abstention P/R |
|---|---|---|---|
| trajectory routing (always skill_update) | 0.196 | 0 | — |
| **VTCA rule-first** | **1.000** | **1.000** | 1.000 / 1.000 |
| VTCA + teacher | 1.000 (teacher never invoked — rules sufficed) | 1.000 | 1.000 |

**Interpretation:** on this diagnostic set VTCA separates skill/belief/abstain
faults essentially perfectly while trajectory routing cannot (+0.804 target
Macro-F1, ≫ the +10-point project gate). Caveat: the set is synthetic and
constructed from the same fault taxonomy the rules encode, so treat it as a
mechanism sanity bound, not an estimate of real-world accuracy. Real-episode
attribution quality still needs the Phase-2 evidence audit (200–300 labeled
events).

**Artifacts:** script committed; numbers above.

---

## 2026-08-13 · E4 · 6-method pilot #1 (ACQ=3, 3 seeds)

**Tested:** that all six controlled methods run end-to-end under matched
settings, and first cost numbers.

**Settings:** `scripts/pilot_6methods.py`; `configs/vista_pilot.json`
(gate proxy 2 / finalist 3, bootstrap 200); 90/5/5 pilot manifest; same
executor/teacher endpoint, initial Skill, seeds {0,1,2}, n_shots, resolution
for every method. Eval: `base`, 6 episodes, seed 0.

**Results:**
- RQ1: every method 0.667 (incl. no_skill) — no differentiation.
- RQ3: 0 updates everywhere (no recurrence reached at 3 acquisition episodes).
- RQ4 (teacher tokens, 3 seeds): full 135,370 · common-gate 5,281 ·
  native 1,348 · w/o-VTCA 1,463. Executor ~170 k for all (matched → fair).

**Interpretation:** harness + fairness verified; evolution cannot fire at this
scale. full's per-action VTCA costs ~25–100× the trajectory baselines' teacher
budget — that multiple is a stable, real characteristic (re-confirmed in E5).

---

## 2026-08-14 · E5 · Effectiveness run (ACQ=15, min_independent=1, 1 seed)

**Tested:** whether evolution actually changes Skills at a feasible scale and
what that does to performance/reliability/cost. `min_independent_episodes`
lowered 2→1 **specifically so evolution fires** (diagnostic deviation from the
controlled protocol — do not cite as a protocol result); single seed; gate
budgets 2/3.

**Results (acquisition, seed 0):**

| method | patches proposed | accepted | frozen skill | teacher tok | executor tok |
|---|---|---|---|---|---|
| full VISTA | 3 (all `termination/replace_exact`) | **0** | S0 | 237,349 | 299,130 |
| EmbodiSkill* native | 0 persistent (execution-lapse routing) | 0 | S0 | 11,129 | 286,609 |
| EmbodiSkill* + common gate | 0 persistent | 0 | S0 | 11,129 | 286,609 |
| VISTA w/o VTCA | 0 persistent | 0 | S0 | ~11 k | ~287 k |

Gate decisions for full's 3 patches: `static: PASS → transition_consistency:
FAIL` ("target was not repaired") each time — i.e. the teacher-proposed
termination patches did not actually repair the mismatch that motivated them,
and the gate rejected all three **before** any rollout spend.

**Results (eval, `base`, 20 episodes, seed 0):**

| method | success | progress |
|---|---|---|
| no_skill | 0.450 | 0.527 |
| static shared Skill (S0) | 0.500 | 0.608 |
| full VISTA | 0.500 | 0.608 |
| EmbodiSkill* native | 0.500 | 0.608 |
| EmbodiSkill* + common gate | 0.500 | 0.608 |
| VISTA w/o VTCA | (eval killed early — server reclaimed) | |

**Interpretation (honest):**
1. The only significant performance delta is **S0 vs no_skill: +5 pts**
   (0.450→0.500, 20 episodes) — the initial Skill itself helps the executor.
2. **No method evolved**: full proposed 3 patches, all correctly rejected by
   the transition replay; the trajectory baselines' teacher routed every
   failure as an execution lapse (appendix, not body). Hence all frozen
   Skills = S0 and all skilled methods tie at 0.500.
3. Genuine skill faults are **rare** at this scale with this executor+S0:
   15 episodes produced 3 `skill_update` attributions (all one field).
4. Reliability behaviour is as designed (0 harmful anywhere; gate demonstrably
   filters non-repairing patches 3/3), but with zero accepted updates the
   beneficial/harmful comparison between gate and no-gate is **untested**.
5. RQ4 multiple confirmed at larger scale: full ≈ **21×** the baselines'
   teacher tokens (237 k vs 11 k) for identical executor spend.

**What this means for the paper claim:** the *mechanisms* are validated
(attribution E3, gate filtering E5, cost accounting), but "evolution improves
performance" is **not yet demonstrated** — it requires enough acquisition for
real faults to recur (ACQ=60 controlled protocol), a harder task distribution
(e.g. `long_horizon`), or fault-injection-driven acquisition. The +5 pt
Skill-vs-no-Skill effect and the 21× teacher-cost multiple are the two numbers
worth carrying into the next campaign.

**Artifacts:** `running/pilot/<method>/` + `running/pilot/eval/*.summary.json`
(local, gitignored); `/tmp/pilot_6methods.log` (volatile).

---

## Bug / ops ledger (found during E1–E5; all fixed in code)

| # | symptom | root cause | fix |
|---|---|---|---|
| 1 | `episode order mismatch: expected 0, got 57` | habitat `Env` serves episodes from its build-time `_episode_iterator`; dataset is **not** episode_id-ordered | `_repoint_habitat_episode_iterator` in `environment.py` |
| 2 | patches silently rejected | `_patch_schema` left `action_type` a free string; model invented types | enum-constrained |
| 3 | `planner retry budget exhausted` | vLLM 400: 10-shot prompt (~4.2 k) + max_tokens 4096 > 8192 ctx | server `--max-model-len 16384` (must match config) |
| 4 | `JSONDecodeError` mid-evidence | teacher JSON exceeded completion budget (max_tokens arms race) | schema bounds (`maxItems` 16, `maxLength` 160) + degrade to no-evidence |
| 5 | runaway reflection (129 k chars) | unbounded `content`/`evidence_ids` | `maxLength` 500 / `maxItems` 16 + degrade to FAIL_EXECUTION |
| 6 | baselines costlier than full (RQ4 inverted) | trajectory methods ran full's action-level engine | `engine_model=None` for `_TRAJECTORY_METHODS` |
| 7 | official-test eval crash | subset episode ids don't match dataset (11/50 resolvable) | load full subset in stock order, cap via `--max-episodes` |
| 8 | `FileExistsError: update_audit.json` | orphaned CLI children kept writing after orchestrator kill | kill orchestrator **and** children; never interrupt mid-run |
| 9 | protocol mismatch | config said `max_completion_tokens` 1024, runtime 4096 | aligned config to 4096 (stock-EB surface) |
| 10 | runs died with the session | background tasks are session-tied | launch vLLM + pilots under `nohup … & disown` |

## Open items (next campaign)

1. **ACQ=60 controlled run** (`configs/vista_p0.json`, min_ind=2, gate 10/30,
   3 seeds) on a dedicated-GPU window — the real RQ1/RQ3 test.
2. Harder distribution for acquisition (e.g. `long_horizon`) or
   fault-injection-driven acquisition to make recurrence reachable.
3. Phase-2 evidence audit (200–300 labeled real events → predicate F1 / ECE /
   selective risk with `evidence_oracle.py` tooling).
4. Ablation switches not yet implemented (no-decoupling, no-abstain) — needed
   only for the T6 main table.
5. w/o-VTCA eval was cut short in E5; re-run when a GPU window exists.

---

## 2026-08-19 · E6 · Dual-server client deployment validation (E5 replica + first EB-NAV client run)

**Tested:** that the split layout — vLLM serving on two remote GPU servers
(192.168.1.185:8000 → EB-HAB client, 192.168.1.173:8000 → EB-NAV client),
single RTX 4090 D client box for both simulators — reproduces the E5 campaign
and delivers the first EB-NAV frozen-evaluation numbers. Both endpoints passed
probe 6/6 + VISTA-layer smoke 3/3 + gate-rollover smoke 4/4 before any run.

**Settings:** identical to E5 (pilot ACQ=15, seed 0, gate 2/3, min_ind=1,
eval `base` 20 eps seed 0) except serving stack: remote vLLM 0.27.1 /
torch 2.13 / transformers 5.15 (Stack A) instead of local vLLM 0.11, and both
simulators render on the client's one GPU. EB-NAV: `evaluate --env eb-nav
--stage official_test --eval-set base`, 60 eps, seed 0, modes
`no_skill` / `static_shared_skill`, config `configs/vista_nav.json`.

**Results (eval `base`, 20 eps):**

| method | success | progress |
|---|---|---|
| no_skill | 0.550 | 0.633 |
| static shared Skill (S0) | 0.450 | 0.533 |
| full VISTA | 0.450 | 0.533 |
| EmbodiSkill* native | 0.450 | 0.533 |
| EmbodiSkill* + common gate | 0.450 | 0.533 |
| VISTA w/o VTCA | 0.450 | 0.533 |

**Results (experiment):** full 3 proposals → 0 accepted (all rejected at
`transition_consistency`, "target was not repaired", exactly E5's pattern);
w/o-VTCA 5 proposals → 0 accepted (gate filtered all — first completed
w/o-VTCA experiment, closing E5 open item 5); native/common-gate 0 persistent
proposals (execution-lapse routing, as E5). All six frozen Skills = S0
(`54b60a62…`, v0). Teacher tokens: full 183,934 · w/o-VTCA 9,398 ·
native = common-gate 7,124; executor ≈ 295–307 k for all (matched).

**EB-NAV client (official_test/base, 60 eps, seed 0):** no_skill **0.633** vs
static_shared_skill **0.600**. The S0 skill slightly underperforms no-skill on
nav base (−3.3 pts ≈ 2 episodes) — direction opposite to EB-HAB E5's +5 pts at
20 eps; treat both as small-sample observations.

**Interpretation:** the dual-server deployment **behaves like the E5
single-server campaign** — same gate rejections, same S0 freeze, same
fairness/cost accounting; the no_skill-vs-S0 flip (0.550/0.450 here vs
0.450/0.500 in E5) is within 20-episode noise (binomial 95% CI ≈ ±0.22) and
plausibly stems from the vLLM 0.11→0.27 stack change. E5's headline conclusion
stands: mechanisms validated, "evolution improves performance" still awaits the
ACQ=60 controlled run. RQ4 multiple: full ≈ 20–26× the trajectory baselines'
teacher tokens.

**Deployment bugs found and fixed (all in adapters/tests, EmbodiedBench
untouched):**

| # | symptom | root cause | fix |
|---|---|---|---|
| 11 | nav client died `vulkaninfo failed to run` | `CUDA_VISIBLE_DEVICES` set → ai2thor auto-selects `gpu_device` → demands vulkaninfo (absent) for CUDA→Vulkan mapping | unset it for the nav client; CloudRendering + nvidia ICD suffices |
| 12 | nav client died `'Controller' has no attribute 'random_initilize'` | stock `EBNavEnv.seed()` typo, and ai2thor 5.0.0 removed the real API | `seed_nav_env` swallows the dead call; process-RNG seeding is the effective part |
| 13 | `full` died on episode 1 `invalid predicate name: ''` | model junk strings reach `PredicateKey.parse` in grounder/evidence paths | per-entry drop (degrade-not-abort) + `minLength`/`minItems` in schemas |
| 14 | `vista_without_vtca` died `common-gate proposals require independent episodes` | `proposal_cluster` hardcoded ≥2 vs pilot config `min_independent_episodes=1` | configurable floor (`min_episodes`, default 2) wired from config |

**Artifacts:** `running/pilot/` (6 methods + eval), `running/vista_skill/
nav_official/base/{no_skill,static_shared_skill}/` (local, gitignored).

**Follow-up:** open items 1–4 above remain; EB-NAV has 4 more subsets
(common_sense, complex_instruction, visual_appearance, long_horizon) to run
under the same protocol.

---

## 2026-08-20 · E7 · Fault-injection repair campaigns (fast effectiveness test; three fault classes)

**Tested:** the paper's core loop end-to-end — detect → attribute → repair →
gate-accept → performance recovery — by injecting a structured fault into the
initial Skill so recurrence is reachable at pilot scale (20 acq eps, seed 0,
`configs/vista_fault_repair.json`, min_ind=2, gate 2/3). Arms: A corrupted
frozen (lower bound) · B corrupted + full VISTA · C corrupted +
vista_without_vtca. New CLI flag `--skill-fault` (diagnostic-gated); driver
`scripts/fault_repair_effectiveness.py`.

**Results (eval `base`, 20 eps, seed 0, SOLO on the 185 server):**

| fault / arm | success | frozen skill | lineage |
|---|---|---|---|
| termination: A corrupted | 0.400 | faulty | — |
| termination: B full (v1, text-only patch) | 0.400 | faulty | 1 proposal → **field=termination (correct)**, rejected |
| termination: B full (v2, compiled patch) | 0.400 | faulty | 1 proposal with `termination_policy: all_goals_evidence` (correct compiled repair), rejected at transition_consistency |
| termination: C w/o-VTCA | 0.400 | faulty | 3 proposals → all field=**procedure (wrong)** |
| pick-inversion: A = B = C | 0.500 | faulty | B: 1 termination proposal; C: 3 procedure proposals |
| references | S0 0.450 · no_skill 0.550 | | |

**Verdict: the repair loop did NOT close on any fault class.** Three blockers,
each a real P0 design conservatism, were isolated:

1. **Synthetic-predicate faults** (`<field>_satisfied` from
   `inject_skill_fault`) can never route: their mismatches are always
   uncovered/unsupported and attribution.py abstains on any such mismatch
   (INSUFFICIENT_EVIDENCE). The E3 diagnostic set bypassed this by
   constructing covered contradictions directly — confirming E3's
   "mechanism sanity bound" caveat.
2. **Termination-policy faults** attribute correctly but are **replay-inert**:
   `_termination_change` derives `task_complete` from the policy enum, and the
   goal grounder's predicates (`pick_ball`, `place_ball_on_X`) are never
   evidence-confirmed, so ANY and ALL compile to identical expectations on
   cached events — no patch can shrink the mismatch set. (This also explains
   E5's three termination rejections.)
3. **Vocabulary-aligned faults** (new `effect_pick_inversion`: inverts S0's
   `holding` pick rule) ARE detected — 16 picks produced 6 covered holding
   contradictions — but event-level attribution masked every one: 3 vetoed by
   co-occurring `task_complete` unsupported expectations, 2 correctly judged
   execution lapses (pick while holding), 1 schema/skill dedupe ambiguity.
   20 episodes: 166 abstain / 23 belief_refresh / 9 skill_update (all
   termination, none effect).

**Positive sub-results worth carrying:**
- Attribution contrast replicated twice: VTCA located the correct field
  (termination ×2 campaigns) while trajectory reflection misattributed to
  procedure 6/6 times.
- After exposing the compiled view in the patch-generator prompt
  (`compiled_termination_policy`, field rules, compiled contract), the teacher
  emitted the correct compiled-level repair (policy enum + text) — the E5/E6
  text-only failure mode is fixed at the generation side.
- Gate safety held throughout: every non-repairing or ill-scoped patch was
  rejected (0 harmful, 0 false accepts across 8 proposals in 4 campaigns).

**Ops findings (ledger #15/16):**
- **#15** Concurrent load on the vLLM server breaks temp-0 determinism: the
  same faulty skill evaluated to 0.650 (concurrent) vs 0.400 (solo, twice).
  All controlled evaluations must run SOLO per server; the probe's
  seed-determinism check is only valid without concurrency.
- **#16** A queued `pgrep -f` waiter deadlocked because the tmux *server*
  process retains the first session's command line; and killing that pid kills
  the server (and every session). Match on the actual python cmdline, not the
  tmux invocation string.

**P1 work items implied (ordered):**
1. Goal-evidence alignment: ground goal predicates onto primitives the visual
   provider observes (unblocks termination repair verification + removes the
   task_complete unsupported pollution).
2. Per-mismatch (not per-event) attribution granularity, or at minimum
   uncovered-predicate isolation so one uncovered expectation cannot veto a
   covered contradiction (unblocks vocabulary-aligned faults).
3. Re-run this fault-repair campaign after 1–2 land; the harness
   (`scripts/fault_repair_effectiveness.py`, `--skill-fault`) is reusable as-is.

**Artifacts:** `running/fault_repair/` (termination, incl. `full_v1_textonly`
and `arm_b_v2_solo_rerun`), `running/fault_repair_pick/` (local, gitignored).

---

## 2026-08-20 · E8 · Repair-loop unblocking arc (termination fault, E8→E8g)

**Tested:** whether the E7 blockers could be removed so the repair loop runs
end-to-end on the termination fault (20 acq eps, seed 0, diagnostic config
`configs/vista_fault_repair_mind1.json` with `min_independent_episodes=1`
after E8d measured 1 true positive per 20 episodes — E5-precedent diagnostic
deviation; the controlled protocol is untouched).

Seven fixes, each isolated by a failing campaign iteration:

| iter | root cause found | fix |
|---|---|---|
| E8 | goal predicates were semantic labels (`pick_plate(robot_0)`) — satisfaction never resolves against primitive evidence keys | grounder constrained to holding/at/near/open vocabulary (P1-1) |
| E8 | recurrence key fragmented on (task-id hash, full AABB names): 16 correct attributions across 7 episodes → 0 proposals | semantic task_pattern + entity head nouns; policy-scope key for termination_conflict (P1-3) |
| E8b | goal keys copied instruction casing (`Sofa` vs `sofa`) → keys never matched; "put X on Y" grounded as intermediate `holding(X)` | normalize_entity on goal keys (P1-4) + move-task grounding rule |
| E8c | 3/4 cluster events were false positives: env says complete AND goals confirmed true in the same evidence (belief lag, unfixable by any policy) | route those to belief_refresh via goal-aware attribution context (P1-5) |
| E8e/E8f | teacher restated the implicated policy enum twice under shifted contexts | for "expects complete, evidence incomplete" conflicts the enum is DERIVED from evidence semantics (all_goals_evidence); the model only authors text (P1-7) |

**E8g outcome (all fixes in):** the loop ran end-to-end for the first time —
attribution ✓ (termination, 2 events), proposal with the derived
`all_goals_evidence` ✓, **transition_consistency passed for the first time
ever** ✓, **real paired rollouts executed for the first time** ✓
(`gate_rollouts/proxy/`), and the gate **rejected at paired_proxy** —
correctly: the candidate was never worse (proxy ep90 tie 1.0/1.0; ep91
progress 0.0→0.125 for the candidate) but the termination fault's behavioral
effect (~0.05 success, E7) is statistically unprovable at proxy budget 2.

| arm | success | lineage |
|---|---|---|
| A corrupted frozen | 0.400 | — |
| B full evolved | 0.400 (frozen = faulty, repairs rejected) | 0/2 accepted, both static+transition PASSED |
| C w/o-VTCA | 0.400 | 0/8 accepted, **7 procedure + 1 effect — 0/8 in the true field** |

Attribution contrast across all campaigns now stands at VTCA 4/4 correct
field vs trajectory reflection 0/14. Teacher 283k / executor 429k tokens for
the full arm (includes first-ever gate rollouts).

**Interpretation:** the pipeline is now fully exercisable — detection,
attribution, evidence-derived repair, compiled verification, and real paired
validation all run. What remains for an ACCEPTED update is a fault whose
behavioral effect is large enough for the paired gate to certify (see E9,
the `constraint_pick_multihold` campaign). The gate's rejection here is the
designed conservatism working, not a defect.

**Artifacts:** `running/fault_repair_e8*_termination*/` (each iteration kept:
`_fragmented`, `_casebug`, `_falsepos`, `_mind2`, `_echo`, `_echo2`, final
`e8g_termination`) — a complete root-cause ladder for the paper's
reliability-mechanism section.

**Post-script (P1-8/9/10, same day):** three further attribution/generation
fixes were isolated by the constraint-fault campaigns below —
(P1-8) a covered CONTRADICTION/MISSING_PROGRESS on a skill-sourced expectation
now outranks the execution-lapse veto (following a wrong rule is evidence
against the rule, not the executor); (P1-9) non-termination patches strip the
model's termination_policy echo; (P1-10) the refuted compiled rule's `after`
value is derived from the observed evidence. Together with P1-7 the patch's
entire compiled view is evidence-derived; the model authors text only.

---

## 2026-08-20 · E9 · Constraint-fault campaigns (E8h→E8p): the loop reaches real paired validation

**Tested:** the behaviorally-strong `constraint_pick_multihold` fault
(inverted `not_holding` pick rule + "gripper can hold multiple objects" text)
— designed so the repair's benefit is provable in paired rollouts, unlike the
termination fault (~0.05 behavioral effect, E8g). Diagnostic config chain:
`mind1` (min_ind=1) → `mind1_p10` → `fullsel_p10` (full manifest,
proxy=10/finalist=10→30, bootstrap=2000); driver gained
`VISTA_FAULT_CONFIG`/`VISTA_FAULT_MANIFEST` env overrides.

**Iteration ladder (each run kept under `running/fault_repair_e8*`):**

| run | blocker found | fix |
|---|---|---|
| E8h | 12/15 not_holding contradictions masked by the execution-lapse veto | P1-8 (constraint attributions 0→6) |
| E8i | model parrots termination_policy onto constraint patches; 6/6 fail static | P1-9 |
| E8j | model restates the refuted compiled rule; 7/7 pass static, fail transition | P1-10 |
| E8k | corrected rules included other fields → applier duplicate-ID crash | field-scope fix |
| E8l | **behavioral benefit directly observed**: candidate 02521ebd solves the two-object task (1.0/1.0) where the parent fails (0.0); proxy=2 cannot yield a positive LCB (mathematical power ceiling) | budget escalation |
| E8m/E8n | proxy=10 exceeds the pilot manifest's 5-task selection pool; driver MANIFEST hardcode | fullsel config + env overrides |
| E8o | **paired_proxy PASSED for the first time**: 10 tasks, mean_delta +0.195, LCB +0.0017, worst subgroup 0.0; finalist covered only 4/10 tasks (budget ÷ 3 seeds) → LCB 0.0 | finalist=30 |
| E8p | full-coverage finalist (10 tasks × 3 seeds): mean_delta ≈ +0.21, **LCB −0.0028** | — (structural, below) |

**Structural finding (gate design):** the multihold fault only affects
multi-object tasks (~1/3 of the pool); single-object task deltas are 0 by
construction, so the task-level bootstrap LCB over ALL tasks is dragged to ≈0
by the zero tail regardless of effect size. A subgroup-local repair cannot
satisfy "global LCB > 0" even though the gate's own subgroup machinery
(worst_subgroup_delta = 0.0, no regression anywhere) certifies it as safe and
the mean delta (+0.2, consistent across proxy and finalist) is real. The
acceptance criterion for subgroup-local improvements is a P1 design question:
candidate semantics = positive LCB within the affected subgroup +
non-regression elsewhere (which the existing subgroup check already proves).

**Status:** detection → attribution → evidence-derived repair → compiled
verification → real paired rollouts all work end-to-end; the first proposal
in history passed the paired proxy. Acceptance remains blocked by the global
LCB criterion's interaction with subgroup-local faults, not by any mechanism
failure. Baselines for contrast: trajectory reflection misattributed the
field in every campaign to date (0/14 cumulative before E9; the E8-series C
arms add more).

**Artifacts:** `running/fault_repair_e8{h..p}_*` (lapsemask → parityecho →
ruleecho → duprules → p2power → f4task → e8p), each stage's gate_rollouts
preserved.

**Follow-up (E8p final, 2026-08-21 04:00):** 7 proposals / 0 accepted (2
passed proxy, both rejected at the full-coverage finalist: mean_delta ≈ +0.21,
LCB −0.0028). Three-arm eval on `base`: A = B = C = 0.550 (all frozen at the
faulty skill; identical digests). Two coherent observations close the arc:
(1) the multihold fault is INVISIBLE on the base eval subset (0.550 ≥
no_skill's 0.550) while decisive on the gate's two-object selection task —
the damage, the repair's benefit, and the eval subset's power to see either
are all bound to the same multi-object task subgroup; (2) the C arm
misattributed the field 5/5 again (trajectory reflection cumulative
0/19 vs VTCA 4/4 correct-field campaigns). Full-arm cost: teacher 249k /
executor 425k tokens including the first-ever finalist rollouts.

---

## 2026-08-21 · E10 · Decision: build SkillFaultBench as a controlled layer on EmbodiedBench

**Context (why this decision, in full):** E5→E8p established that the repair
loop's every mechanism works — detection, attribution (VTCA 4/4 correct field
vs trajectory reflection 0/19), evidence-derived repair generation, compiled
verification, real paired rollouts, and a first-ever paired_proxy pass
(mean_delta +0.195, LCB +0.0017, worst subgroup 0.0) — yet zero updates were
ever accepted and no performance delta ever appeared in A/B/C evals. The
post-mortem attributes this to three quantified properties of the evaluation
substrate, not the method:

1. **Fault sparsity.** A frozen 8B executor on EB-HAB produces ~1
   skill-attributable fault per 20 acquisition episodes (E8d), below the
   recurrence gate at any feasible scale. Real faults never recur; injected
   faults had to be used instead (E7 onward).
2. **Subgroup invisibility.** Effects that are real and decisive at the
   task-subgroup level (candidate 1.0 vs parent 0.0 on two-object tasks,
   E8l) are invisible both to the gate's global task-level LCB (E8p: −0.0028
   against mean +0.21) and to the stock `base` eval subset (A=B=C=0.550).
   Damage, benefit, and measurement power are bound to the same subgroup the
   benchmark never stratifies.
3. **No credit-assignment ground truth.** EmbodiedBench labels task success
   only; nothing labels which skill field/rule is wrong, so attribution
   quality (the paper's core claim) is unmeasurable without our own fault
   injection.

**Survey (2026-08-21):** skill-evolution methods are crowded in 2025-26 —
SAGE (RL-validated skill libraries), SkillClaw (collective evolution),
Memento-Skills (RL skill rewrite), SkillMAS (co-evolution with failure
attribution), GRASP (gated regression-aware proposals), SKILLER (bounded
edits), SkillAudit (paired-trajectory accept/reject — a method sibling of
our gate). Evaluation is the recognized gap: SkillsBench measures static
skill augmentation; LifelongAgentBench measures task domains; PATH-Bench
measures experience utility; EmbodiSkill itself reports only task success on
EmbodiedBench. No existing benchmark provides fault ground truth,
recurrence-density control, subgroup-stratified pools, or update-reliability
metrics (beneficial/harmful precision, attribution accuracy). Every method
above needs exactly that.

**Decision process:** three options were considered. (a) Stay on stock
EmbodiedBench and only fix the gate's acceptance semantics — rejected as
insufficient: even a subgroup-aware gate cannot show what the base subsets
cannot measure, and fault sparsity still starves evolution. (b) Build a
standalone benchmark from scratch — rejected as wasteful: the executor
substrate, simulators, controlled protocol, and fairness machinery already
exist and are validated. (c) **Build a controlled fault layer on top of
EmbodiedBench environments (EB-HAB primary; EB-ALF and EB-NAV as generality
arms) — chosen.** Nine campaigns of infrastructure (inject_skill_fault fault
bank, the three-arm fault-repair harness, five-field skill schema with
compiled views, paired gate) are its seed, and each E8 iteration contributes
a design rule.

**SkillFaultBench design pillars:**

1. **Fault bank** with three admission criteria learned from E7–E9:
   vocabulary-aligned (predicates the evidence layer observes), replay-
   verifiable (repair provably shrinks the cached mismatch set), and
   behaviorally calibrated (subgroup effect size measured and tunable —
   faults must be neither behaviorally null like termination nor gate-
   invisible like un-stratified subgroup faults).
2. **Subgroup-stratified task pools** per fault: affected and protected
   subgroups constructed at designed ratios, with eval power computed
   against the fault's calibrated effect size (the E8p lesson as a spec).
3. **Reliability metric family:** detection rate, field-attribution accuracy
   (ground truth = the injected fault), repair-verification rate, acceptance
   correctness (beneficial/harmful update precision), subgroup-level
   drop/recovery, teacher cost per accepted update.

**Paper strategy:** main comparison table stays on stock EmbodiedBench
(matched executor/teacher/budget, EmbodiSkill* baselines); SkillFaultBench
carries the mechanism/falsifiability section (RQ2 attribution, RQ3
reliability). The E5→E8p negative results become the benchmark's motivation.

**Open items carried forward:** (1) gate acceptance semantics for
subgroup-local repairs (affected-subgroup LCB > 0 + global non-regression —
the existing subgroup check already proves the safety half); (2) executor-
regime study — a weaker executor should raise skill-fault density and skill-
text dependence (making evolution matter), at the risk of raising execution
lapses too; measure before committing (pilot: 20 episodes, attribution
distribution); (3) EB-ALF adapter (eb-alf runs in `max_embench`; not yet
wired into the VISTA runner); (4) EB-NAV fault layer (evaluate-only env —
faults enter through the frozen-skill artifact, evolution stays on EB-HAB).
