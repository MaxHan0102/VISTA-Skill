# Phase 5 experiment log

This log starts after the 2026-09-02 requirement reset: VISTA-Skill must exceed
the published EmbodiSkill result on both EB-HAB and EB-NAV. Earlier phase logs
remain immutable historical records.

## E0 — repository, endpoint, and integrity audit — 2026-09-02

Status: engineering pass; no task-performance claim.

- Repository started clean at commit `e32d028`.
- Unit suite before changes: 238/238 passed in `max_embench`.
- Qwen endpoints `192.168.1.185:8000/v1` and
  `192.168.1.173:8001/v1` both passed the 6/6 serving probe: model identity,
  completion usage, seed determinism, strict JSON schema, trajectory-reflection
  schema, and multimodal image input. Both serve
  `Qwen/Qwen3-VL-8B-Instruct` with a 16384-token model context.
- The first endpoint additionally passed the 3/3 VISTA method-layer smoke:
  strict JSON with usage accounting, `JsonTrajectoryTeacher` reflection, and
  `JsonVisualEvidenceProvider` pre/post-image extraction. Smoke usage was 23/7,
  188/72, and 260/100 prompt/completion tokens respectively.
- The first EB-HAB smoke failed at EGL context creation only inside Codex's
  default isolated execution layer, where `/dev/nvidia*` and `/dev/dri` are not
  exposed. A host-level check on Pingan03 found the RTX 4090 D, driver
  575.64.03, and all required device nodes. Re-running through the host
  execution layer created the NVIDIA OpenGL 4.6 renderer and successfully
  reset/closed `train_validation` episode 0 at resolution 500. The installed
  `habitat_sim` reports `cuda_enabled=False`, but that flag did not prevent the
  proven EGL GPU-rendering path. This is an execution-isolation constraint, not
  a Pingan03 hardware, simulator-build, or model-server failure.
- The current full method used roughly 20--26x the teacher tokens of matched
  trajectory controls in prior pilots; 167/204 teacher calls in one multi-hold
  run were visual-evidence calls.
- The current gate had zero promotions. Offline effect-inversion audit found
  4/7 independently beneficial candidates; best affected-task delta was
  +0.0821 with bootstrap LCB +0.0051 and approximately zero protected change.
- The stale server-side paper reference contains a row combining a legacy
  exposed EB-HAB score with a later EB-NAV stock score under “Ours”. This is
  recorded as a correction required in the separate local paper project; the
  server-side LaTeX reference is not an experiment source of truth.
- `configs/vista_nav.json` recorded a 1024 completion cap while runtime always
  sent 4096. It is aligned to stock RemoteModel 4096, and runtime now consumes
  the config value.
- Controlled EB-NAV evaluation previously checked little beyond frozen state.
  It now validates the same executor surface and data-policy digest as EB-HAB.
- Historical Target-Skills/Skill v2 remain reproducible diagnostics but are
  quarantined from controlled claims.
- A digest-verified scan found 251 Skill artifact envelopes under `running/`;
  all 251 are pre-policy and therefore quarantined, with zero artifact eligible
  under the new boundary. This is intentional: the first eligible evolved Skill
  must be produced by a new Phase-5 run.

Artifacts: `configs/evaluation_data_policy_v1.json`,
`docs/evaluation_integrity.md`, and the code/tests referenced by the associated
commit. Next experiment must use only allowed development data.

## E1 — semantic affected/protected candidate gate — 2026-09-02

Status: implementation and unit-test pass; real simulator reset/close smoke
passed; mechanism experiment pending.

- Added outcome-independent EB-HAB task tags from symbolic `goal_preds` and
  `subgoals`. The adapter maps predicates to primitive action requirements and
  never reads success, progress, reward, trajectory, or model output.
- For action-local procedure/effect/constraint changes, candidate scope comes
  from the triggering clustered action. The gate requires positive bootstrap
  LCB on affected tasks and non-inferiority on all remaining protected tasks.
- Activation and termination changes remain on the global conservative gate;
  the method contains no injected-fault name, official subset, or episode-ID
  branch.
- Proxy/finalist task pools remain disjoint, and repeated seeds are averaged
  within task before bootstrap resampling.
- A deterministic test confirms that an affected benefit can pass even when
  unrelated tasks make the global LCB negative; a second test confirms that
  protected regression is rejected.
- Created new clean-boundary configs `vista_phase5_hab.json` and
  `vista_phase5_nav.json`; historical `vista_p0.json` remains available for old
  phase reproduction.
- Full unit suite passed after the change.

Repository-policy correction: an initial attempt to update the stale
server-side LaTeX reference was fully rolled back at the user's request. The
repository now declares `context4agent/latex/` reference-only; future paper
corrections are recorded in `docs/` and applied only in the separate local
LaTeX project. No server-side LaTeX file may be modified while this policy is
active.

The frozen effect-inversion experiment remains a bounded repair-regression
check. It is no longer the next primary experiment; the research-direction
decision in E2 supersedes that earlier ordering.

## E2 — discovery, repair, optimization, and dual performance/cost objective — 2026-09-02

Status: research decision frozen; implementation and natural-interaction
experiments pending.

The core research question is now:

> Can VISTA-Skill autonomously discover, validate, and accumulate executable
> rules from its own visual embodied interactions, at low cost, such that the
> evolved rules genuinely improve task success?

The goal remains analogous to EmbodiSkill's overall self-evolution setting but
must be stronger on both final task performance and evolution cost. VISTA-Skill
must support the full lifecycle rather than repair alone:

1. **Discovery:** add reusable rules that are absent from the current Skill,
   using recurring natural pre/action/post evidence from both successful and
   failed executions.
2. **Repair:** correct existing rules contradicted by reliable transition
   evidence.
3. **Optimization:** improve rules that are valid but lead to inefficient,
   brittle, or lower-success execution.
4. **Execution-lapse handling:** preserve a valid rule while making it more
   executable or salient when the executor fails to follow it.

Initialization decision:

- The primary arm will start from a structured `Empty/Interface-only S0` with
  no hand-authored task strategy, action-effect, constraint, or recovery rules.
  Public action names, argument formats, observation fields, and official
  in-context examples remain part of the benchmark interface rather than
  learned Skill content.
- The current hand-authored benchmark-aware `shared S0` becomes a warm-start
  initialization-sensitivity ablation, not the primary automatic-discovery
  condition.
- All controlled evolution methods must receive the same initial Skill within
  a comparison. `No Skill` remains a separate execution baseline.

Experimental-evidence decision:

- Manually injected incorrect Skills demonstrate localization and bounded
  repair only. They remain unit/mechanism and supplementary regression tests;
  they cannot support the main autonomous-evolution claim.
- Main evidence must come from clean, natural EmbodiedBench development
  interactions beginning from the frozen S0. The experiment must report Skill
  growth, held-out rule validity and reuse, executor adoption, task-success
  learning curves, protected-task regression, and calls/tokens/rollouts/wall
  time/GPU hours.
- The next primary EB-HAB experiment will therefore be a clean natural
  discovery pilot on `train_validation`, not a manually faulted-Skill run.

Required implementation changes identified by the audit:

- Recurrent `SUPPORTED_UNEXPECTED` transitions need a discovery route instead
  of being unconditionally absorbed as episode-local belief refreshes.
- The persistent update representation must distinguish discovery, repair,
  optimization, and execution-lapse handling while retaining evidence IDs and
  one-field bounded edits.
- Hand-authored environment transition knowledge must be separated from the
  public interface contract; rules claimed as discovered cannot already be
  encoded in the fixed action schema or initial Skill.
- Candidate validation will be staged as evidence reliability, transition
  consistency, executor adoption, and paired task utility. Proxy selection may
  use dense task progress, but finalist/audit acceptance must explicitly
  protect task success.
- Evidence-supported candidates with insufficient statistical power should
  remain pending for additional independent support rather than being treated
  as permanently invalid.

The primary condition continues to use frozen Qwen3-VL-8B-Instruct as both
executor and self-evolution model. GPT-5.2 remains a separately reported
contingency only if a pre-registered comparison demonstrates that Qwen's model
capability, rather than evidence, routing, adoption, or validation, is the
bottleneck. The desired final claim is simultaneous improvement over published
EmbodiSkill in official EB-HAB/EB-NAV task success and in measured evolution
cost.

Next: specify the interface-only S0 contract and implement the natural
discovery path before launching the clean Pingan03 EB-HAB pilot.

## E3 — interface-only S0 and natural effect Discovery smoke — 2026-09-02

Status: mechanism pass on real Pingan03 EB-HAB; candidate task-utility gate
pending. No official-test data used.

Implementation:

- interface_only_shared_skill() is now the Phase-5 EB-HAB primary S0. It has
  the five typed fields and persistent identity but zero statements, zero
  compiled prediction rules, and no active termination prediction.
- SkillOnlyActionSchema predicts only rules already accumulated in the Skill.
  Historical fixed primitive knowledge remains available to reproduce earlier
  repair experiments but cannot pre-explain Phase-5 discoveries.
- Persistent updates now carry an explicit discovery, repair, or optimization
  lifecycle label. Reliable action-bound SUPPORTED_UNEXPECTED evidence routes
  to effect/discovery; derived task_complete is excluded.
- Grounded predicates are generalized only through action arguments or the
  pre-action held object. Recurrence therefore joins different object
  instances while rejecting unbound episode-specific facts.
- A ready Discovery cluster produces a deterministic one-field append and
  compiled causal rule. It uses zero patch-teacher calls; Repair/Optimization
  retain the constrained-model path.

Live sequence:

1. The endpoint 192.168.1.185:8000/v1 again passed all 6/6 serving-contract
   checks.
2. Unguarded one-episode diagnostic
   running/phase5_discovery_smoke_20260902 completed a ball-to-sofa task in four
   actions with success 1.0 and progress 1.0. All four transitions reached
   Discovery. It also exposed a blocking evidence defect: after navigation,
   the VLM assigned near-one confidence to action-irrelevant claims that the
   ball had already moved and the gripper was free.
3. The same seed with the authority-aware guard,
   running/phase5_discovery_guarded_smoke_20260902, again succeeded in four
   actions. The guard retained only nav-to-near, pick-to-holding/not_holding,
   and place-to-at/holding/not_holding; the premature placement and gripper
   claims after navigation were downgraded. Method usage was one goal-grounding
   call (1938 prompt / 26 completion tokens) plus four visual-evidence calls
   (3116 / 1399). There were no attribution or patch calls.
4. A first two-episode recurrence attempt collected two successful natural
   tasks and 20 transitions, then aborted because Qwen returned truncated
   patch JSON. This directly motivated the deterministic zero-token Discovery
   patch.
5. The v2 two-episode recurrence diagnostic collected two successful tasks
   (ball-to-sofa: 4 steps, zero invalid; sponge-and-spoon-to-table: 24 steps,
   eight invalid). Four recurrent candidates were materialized:
   nav-to-near(arg0)=true, pick-to-holding(arg0)=true,
   pick-to-not_holding=false, and place-to-not_holding=true. All four passed
   static and cached-transition consistency checks without a patch-model call.
   They intentionally failed at the empty paired-proxy stage and were not
   promoted. The run was manually stopped when the generic post-run audit
   unexpectedly began simulator rollouts; those audit fragments are invalid
   and excluded. The diagnostic path now skips update audit by construction.
6. The completed v3 recurrence run is
   running/phase5_discovery_recurrence_v3_20260902. Both tasks succeeded
   (4 and 16 steps; 0 and 4 invalid actions), ready clusters progressed from 0
   to 4, and the same four candidates passed static and transition checks. The
   complete manifest records 20 visual-evidence calls and 30,117 visual
   prompt/completion tokens, making per-action visual extraction the next
   measured cost bottleneck.
7. Event-triggered evidence was then enabled in
   running/phase5_discovery_sparse_visual_smoke_20260902. On the same
   ball-to-sofa episode, task success, progress, four-step trajectory, and
   Discovery predicates were unchanged. Method usage fell from 6,479 to 1,964
   tokens (minus 69.7 percent) and elapsed time fell from about 29.5 to 6.6
   seconds. The only method-model call was goal grounding; transition evidence
   came from public structured feedback plus action-local ledger grounding.
   Visual extraction remains a configurable fallback for environments with
   incomplete feedback.

Interpretation:

- The former “updates never become candidates” bottleneck is resolved for
  naturally discovered action effects.
- Evidence reliability, not transition routing, was the first live bottleneck;
  authority-aware filtering is therefore part of the controlled Phase-5
  configuration.
- Unconditional per-action vision was a measured cost bottleneck. Prediction-
  blind event triggering removes it on feedback-complete EB-HAB transitions
  without removing the general visual fallback.
- This is not yet evidence of task improvement. Primitive-effect candidates
  are necessary plumbing, but the second task's eight invalid actions show
  that failure-conditioned procedure/constraint discovery and optimization are
  the next high-value target.

Next: mine repeated failed-action patterns into procedure/constraint
Discovery/Optimization candidates, add an executor-adoption check, and run one
bounded real paired gate before scaling acquisition.
