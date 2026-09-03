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

## E4 — natural constraint Discovery and first real paired gate — 2026-09-02

Status: natural precondition Discovery passed recurrence and cached-transition
validation; its first 10-task paired proxy was correctly rejected. No
official-test data was accessed and the frozen Skill remained interface-only
S0.

Implementation:

- A failed action with reliable deterministic environment feedback now remains
  eligible for causal Discovery instead of being automatically classified as
  stochastic. The observed failed state is attributed to the constraint field.
- Constraint Discovery generalizes across object instances. Two independent
  failed picks with `near(object)=false` produced the reusable rule: “Before
  picking, require evidence that the selected object is near; after a not-near
  failure, navigate or gather new evidence before retrying.”
- Compiled constraint markers are interpreted as failure states, so execution
  checks require the inverse state and do not incorrectly compile the failure
  as an action effect. Cached validation confirms that the candidate, but not
  its parent, explains the triggering failures.
- Discovery candidate identities are stable as additional support arrives.
  This fixes the observed duplicate-proposal bug, which had re-evaluated the
  same semantic rule whenever a cluster gained an evidence item.
- The Phase-5 budget now permits one candidate proposal per acquisition round
  and prioritizes constraint/procedure evidence over primitive effects. A
  diagnostic-only field filter isolates one candidate class without changing
  the controlled method.

Natural acquisition smoke:

- `running/phase5_constraint_discovery_smoke_20260902` used five clean
  train-validation interactions from interface-only S0. Four of five tasks
  succeeded. Across 41 transitions it produced 17 effect discoveries, three
  constraint discoveries, and 21 abstentions.
- The independent failures in episodes 1 and 3 formed the generalized
  `pick | near({arg0}) | false` constraint cluster. Static and cached-transition
  checks passed. Repeated no-change failures inside one episode remained
  abstentions rather than being counted as new independent evidence.
- The smoke exposed 16 duplicate candidate attempts from growing ready
  clusters; stable Discovery fingerprints plus the one-proposal budget resolve
  that cost leak.

First real paired proxy:

- `running/phase5_constraint_paired_pilot_20260902` acquired five episodes and
  evaluated the single constraint candidate against interface-only S0 on the
  ten registered proxy tasks (60--69). Acquisition and proxy completed before
  the generic post-hoc audit was manually stopped.
- Parent and candidate were identical on tasks 60, 61, 64, 65, 67, and 68.
  The candidate reduced task 62 from 11 steps / four invalid actions to six /
  one, and changed task 66 from failure to success. It did not solve task 63
  and changed task 69 from success to failure (progress 0.25, ten invalid
  actions).
- The task-first composite mean delta was `+0.00838`, but affected-task mean
  delta was only `+0.01198` and its bootstrap LCB was `-0.26973`; protected
  mean and LCB were both zero. The gate therefore rejected the rule as
  unproven. This is the desired reliability behavior: two positive cases and a
  slightly positive mean do not justify promotion in the presence of a severe
  affected-task regression and wide uncertainty.
- The candidate's effect on task 62/66/69 also shows that executor sampling at
  temperature zero is not sufficiently deterministic to attribute every
  trajectory difference to the prompt rule from a single pair. A repeat pilot
  and multi-seed finalist/audit remain necessary before any causal performance
  claim.
- The generic experiment path began a 20-task, three-seed parent/candidate
  post-hoc audit (120 episodes) after this bounded diagnostic. It was stopped
  after the first task because that cost was outside the pilot budget. Partial
  audit fragments are excluded. A new diagnostic-only
  `--skip-update-audit` flag records this omission explicitly and prevents the
  same hidden budget expansion; controlled runs still require the full audit.

Interpretation:

- VISTA-Skill now autonomously discovers both primitive effects and a reusable
  action precondition from its own clean interaction; the former
  “empty/interface Skill cannot grow” implementation bottleneck is resolved at
  mechanism level.
- Candidate rejection remains the correct result for this first rule: it has
  not demonstrated reliable task benefit. It must remain absent from frozen
  S0.
- The next bottleneck is no longer candidate materialization. It is stable
  executor adoption and statistically powered utility validation, followed by
  higher-level procedure/optimization discoveries whose prompt impact is more
  direct than a single generic precondition sentence.

Next: complete a second bounded paired replicate with the audit explicitly
disabled, compare per-task sign stability, and then either strengthen the
constraint's executable form or move to failure-conditioned procedure
Discovery before spending the full finalist/audit budget.

## E5 — exact paired replicate and complete cost accounting — 2026-09-03

Status: the natural constraint candidate and its rejection replicated exactly;
executor usage is now complete across acquisition and selection. Frozen Skill
remained interface-only S0 and no official-test data was accessed.

Runs:

- `running/phase5_constraint_paired_pilot_v2_20260902` repeated the bounded E4
  protocol: five acquisition episodes, one constraint candidate, ten proxy
  tasks / twenty parent-candidate rollouts, and explicit audit omission.
- `running/phase5_constraint_paired_pilot_v3_usage_20260902` repeated it once
  more after fixing run-scoped executor accounting. This third run exists to
  validate cost artifacts, not to expand the gate or tune on selection results.
- All 20 proxy rollouts in both repeats were identical to E4 in success,
  progress, environment steps, invalid actions, and full action trajectory.
  Task 62 again improved from 11 steps / four invalid actions to six / one;
  task 66 again changed from failure to success; task 69 again regressed from
  success to failure. The paired-proxy metrics were bit-for-bit unchanged:
  mean delta `+0.00838365`, affected mean `+0.01197664`, affected LCB
  `-0.26972789`, protected delta/LCB zero. The candidate was rejected in every
  run and the frozen artifact remained an empty-field S0.
- Natural acquisition is less rigid than seeded selection: E4 episode 1 used
  23 steps while v2/v3 used 20. Nevertheless, v2 and v3 acquisition matched,
  and all three runs formed the same generalized constraint candidate from the
  same independent failure pattern.

Cost audit from v3:

- Executor acquisition: 16 calls, 70,831 prompt tokens, 6,716 completion
  tokens.
- Executor paired proxy: 75 calls, 334,506 prompt tokens, 28,318 completion
  tokens.
- Full executor path: 91 calls, 405,337 prompt tokens, 35,034 completion tokens.
  The 25 episode-level usage records sum exactly to this manifest total.
- Method side: five goal-grounding calls, 9,716 prompt tokens, 160 completion
  tokens. There were no visual-evidence or patch-teacher calls. End-to-end this
  bounded diagnostic used 96 model calls and 450,247 prompt+completion tokens.
- Within proxy selection, parent S0 used 178,898 total executor tokens and the
  candidate used 183,926 (`+5,028`, about `+2.8%`). The gate itself consumed
  about 82% of executor tokens, so paired validation, not deterministic
  Discovery, is now the dominant cost.

Implementation consequence:

- The old manifest read usage only from the long-lived acquisition runner and
  omitted all short-lived gate/audit runners. A run-scoped tracker now flows
  through acquisition, proxy/finalist gate, and update audit; totals and a
  `by_phase` breakdown are written to the manifest, with per-episode deltas in
  every rollout JSONL. Frozen evaluation uses the same path.
- E4/v2 performance and reliability conclusions remain valid, but their old
  manifest cost fields are acquisition-only and must not be used for a cost
  comparison. v3 is the first complete-cost paired diagnostic.

Interpretation:

- Explicit request seeds make the gate behavior exactly reproducible on this
  endpoint, so the task-62/66 gains and task-69 regression are genuine effects
  of the candidate prompt under the registered gate coordinates, not a
  one-off sampling fluctuation.
- Exact reproducibility does not rescue the candidate: its affected-task lower
  confidence bound is still strongly negative. Spending finalist or audit
  budget on the same rule would not address the core weakness.
- The next method target is a more direct failure-conditioned procedure or
  optimization rule whose causal effect is larger and less subgroup-dependent.
  Gate cost should be controlled with sequential stopping or cheaper cached
  adoption checks, but reliability thresholds must remain unchanged.

Next: mine the repeated not-near failure trajectories for a bounded executable
procedure/optimization candidate, test executor adoption cheaply before paired
rollouts, and only launch another proxy if the new candidate changes the
targeted behavior in cached or minimal live checks.

## E6 — identifiable temporal rule and sequential safe-gate diagnostic — 2026-09-03

Status: implementation/mechanism pass, task-utility No-Go. The stronger
candidate was rejected and the frozen Skill remained interface-only S0. This
run reused the E4/E5 selection tasks 60--69 intentionally as a direct mechanism
comparison, so it is selection-contaminated and is not a new unbiased paper
result. No official-test data was accessed.

Implementation and validation:

- Commit `d138bef` implements the requested changes as runtime mechanisms, not
  only paper terminology. Every CreditAssigner `skill_update` now requires and
  persists a seven-condition `IdentifiabilityAudit`; failure of evidence,
  provenance, compliance, stochasticity, identity, unique-field, or action
  binding routes to abstention.
- Natural `pick | near({arg0})=false` Discovery now writes both a compiled
  precondition and a compiled temporal rule. Its episode-local monitor enforces
  `failed pick -> no same-target pick until successful navigation/evidence
  change`, injects active obligations into the prompt, checks admission before
  `env.step`, and logs guard blocks separately from environment invalid actions.
- The paired gate now evaluates registered prefixes and allocates alpha across
  finite Hoeffding-bound looks. Interim checks can only reject for futility or
  protected regression; they cannot promote early. A survivor still requires
  the unchanged full-budget task-first bootstrap gate. Repeated-seed finalist
  pools remain fixed-budget because incomplete task blocks are not independent.
- The complete unit suite passed 273/273, including temporal state-machine,
  runner admission, sequential stopping/full-budget fallback, artifact
  round-trip, and legacy v2 artifact compatibility tests. The renamed historical
  plan is now `docs/research_plan_phase4.md`.

Run and integrity:

- Successful run:
  `running/phase5_temporal_sequential_paired_e6_20260903_rerun1` with one
  evolution seed, five acquisition episodes, one constraint candidate, ten
  paired proxy tasks, and diagnostic audit omission. Config hash is
  `a5c39710014bc34e5694d34b9f1ba56c22e82444e8d256a33906992167ec8d94`.
- Two launch-only failures preceded it. The first lacked a method-client API-key
  placeholder and stopped before creating the simulator/output directory. The
  second proved EGL/RTX-4090 initialization but lacked the stock executor's
  `OPENAI_API_KEY`; it stopped before any task or model request and left no
  artifact. `rerun1` set both placeholders and exited normally. Neither launch
  is counted as an experiment attempt.
- Acquisition completed 5/5 episodes (4 successes), 38 primitive transitions,
  and the same generalized candidate from two independent episodes. It produced
  20 `skill_update` events (17 effect, three constraint); all 20 include a
  passing identifiability conjunction and complete provenance.
- All 20 proxy JSONL files contain a complete episode result. Both content-
  addressed parent/candidate artifacts and the rejected frozen envelope pass
  digest loading. The candidate (`4e90990d...`) contains one temporal rule; the
  final frozen S0 contains none because promotion was rejected.

Paired result:

| Task | Parent -> candidate outcome | Composite delta | Temporal observation |
|---|---|---:|---|
| 62 | success 11/4 -> success 6/1 steps/invalid | +0.01970 | no block needed |
| 63 | failure 21/10 -> failure 20/10 | -0.00238 | no block |
| 66 | failure 19/10 -> success 10/3 | +1.02263 | one repeated pick blocked; recovery succeeded |
| 69 | success 12/4 -> failure 21/10 | -1.01429 | no block: every failed pick followed a successful but unhelpful navigation |

The other six tasks were unchanged. Parent/candidate composite means were
`0.78301` and `0.78557`, for only `+0.002566`. The affected-task mean was
`+0.003666`, affected bootstrap LCB `-0.290476`, global LCB `-0.203095`,
protected mean/LCB zero, and worst subgroup delta `-0.002381`. The gate therefore
rejected at proxy. Compared with E5, the stronger executable form reduced mean
delta (`+0.008384 -> +0.002566`) and made the affected LCB more negative
(`-0.269728 -> -0.290476`); it did not repair the task-69 regression.

Sequential/cost result:

- The sequential gate performed four registered looks at 4/6/8/10 tasks. No
  confidence upper bound established early futility, so all ten pairs ran and
  `sequential_early_stop=0`. This validates the fail-closed fallback but provides
  no sequential rollout saving on this borderline candidate.
- Executor acquisition used 16 calls and 77,547 tokens; proxy used 71 calls and
  344,501 tokens; total executor use was 87 calls / 422,048 tokens. The method
  used five goal-grounding calls / 9,876 tokens and no visual-evidence or patch
  calls. End-to-end use was 92 calls / 431,924 tokens. This is 18,323 tokens
  below E5, but because both runs executed all 20 proxy rollouts, the reduction
  comes from changed trajectory/request lengths and cannot be attributed to
  sequential stopping. Proxy validation still consumed 81.6% of executor
  tokens.

Interpretation:

- The theoretical additions are now operational and auditable. VTCA
  identifiability is present on every live update, and the temporal monitor can
  alter a real trajectory (task 66) rather than merely decorate prose.
- They did not improve the registered performance/reliability conclusion. The
  task-66 gain is almost cancelled by task-69 harm, uncertainty remains large,
  and reliable promotion correctly stays at zero. This directly confirms that
  these tools improve rigor and can improve selected failure modes, but do not
  by themselves guarantee higher benchmark performance.
- The temporal fragment is too weak for task 69: “perform any successful nav”
  releases the obligation even if the new view does not increase evidence for
  the target. The next candidate should require observation-grounded search
  progress (new target evidence or a changed candidate location), not another
  generic retry prohibition. That rule must be acquired/evaluated on a fresh
  development rotation before any further paper-level utility claim.
