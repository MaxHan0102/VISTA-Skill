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
