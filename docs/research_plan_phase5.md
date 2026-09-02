# Phase 5 research plan: exceed published EmbodiSkill on EB-HAB and EB-NAV

Frozen planning date: 2026-09-02. Target venue: CVPR 2027. EB-HAB and EB-NAV
are both primary benchmarks; transfer to a different benchmark begins only
after the EmbodiedBench claim is established.

## Non-negotiable success criterion

With the frozen Qwen3-VL-8B-Instruct executor and stock EmbodiedBench protocol,
VISTA-Skill must exceed the strongest published EmbodiSkill result in both
environments:

| environment | EmbodiSkill-GPT target | minimum strict improvement on 300 tasks |
|---|---:|---:|
| EB-HAB | 45.33% | at least 137/300 successes (45.67%) |
| EB-NAV | 50.33% | at least 152/300 successes (50.67%) |

The actual goal is a margin larger than one episode and a better
performance--cost frontier. Merely beating the local `EmbodiSkill*` analogue is
insufficient. Report all official subsets, task progress, invalid actions,
update reliability, subgroup regression, teacher calls/tokens, candidate
rollouts, wall time, and GPU hours.

## Current evidence and bottlenecks

1. VTCA localization works on controlled faults: action-level attribution hit
   the target field in 4/4 audited cases while trajectory reflection hit 0/26.
   Bounded exact-field patches were structurally valid in 10/10 cases.
2. End performance is not yet demonstrated. The current global bootstrap gate
   accepted no updates, even though the effect-inversion audit contained four
   independently beneficial candidates and the best affected-task lower bound
   was positive with negligible protected-task change.
3. Candidate selection is diluted by unrelated tasks. The gate groups by a
   hash-like subgroup and demands positive global LCB; it does not yet use an
   outcome-independent semantic affected/protected split.
4. Skill guidance is prompt-only. Compiled action rules inform expected
   transitions and attribution but do not yet constrain action admission,
   maintain an explicit executable goal agenda, or adapt planning horizon.
5. Full VISTA currently spends roughly 20--26 times the teacher tokens of the
   local trajectory controls because it calls a visual/attribution model at
   action granularity. This is incompatible with the desired cost claim.
6. EB-NAV has evaluation support but no evolution workflow or clean released
   development split. It is therefore a main-line implementation gap, not a
   future extension.
7. The local EB-HAB simulator is presently blocked at EGL/CUDA device mapping;
   both remote Qwen endpoints answer model probes. EB-HAB experiments resume
   only after the client exposes the simulator GPU correctly.
8. Historical official-test exposure invalidates the old Target-Skill and
   Skill-v2 runs for claims. The new prospective boundary is defined in
   `docs/evaluation_integrity.md`.

## EmbodiSkill source audit and comparison scope

The public source audited on 2026-09-02 is ALFWorld-only. The provided launch
script defaults to GPT-5.2 for Skill evolution, 10 epochs, 50 training tasks per
epoch, 30 action trials, 512 reflection output tokens, and 2048 revision output
tokens. The implementation invokes an episode reflection for every completed
episode, retries structured parsing up to three times, separately revises
manual sections and execution notes at epoch boundaries, and can add compaction
and refactor calls. It also includes rollback, repeated-action stuck handling,
success discovery/optimization, and skill-defect versus execution-lapse paths.

These are important engineering features, not incidental details. They make
the local rough `EmbodiSkill*` implementation unsuitable as the sole baseline
and support the hypothesis that EmbodiSkill has high evolution-token cost.
They do not reveal the unpublished EB-HAB/EB-NAV task generator, prompts, or
tricks, so the published scores remain the primary target and all source-based
cost comparisons must state their assumptions.

## Ordered campaign

### P5.0 — integrity and protocol repair

Quarantine contaminated artifacts, record any required paper-table correction
for the separate local LaTeX project, distinguish reported from reproduced
results in experiment metadata, enforce the data-policy digest, close EB-NAV
protocol validation, and align recorded/actual executor caps. The stale
server-side `context4agent/latex/` copy is reference-only and must not be edited.
Exit: full unit suite green and no historical artifact claim-eligible.

### P5.1 — make updates capable of changing behavior

Implement three benchmark-independent consumers of the same typed state:

1. semantic affected/protected candidate admission defined from task goals and
   candidate scope before rollout outcomes;
2. an executable goal agenda that orders unsatisfied predicates without
   encoding benchmark task names;
3. an action-admissibility/recovery guard compiled from action preconditions,
   the active Skill, and current evidence, with abstention/re-observation on
   uncertainty rather than oracle access.

Adaptive horizon is added only through generic goal count, unresolved
preconditions, and observed progress. No rule may branch on official subset,
episode ID, or hand-authored test trajectory.

Exit: on clean development tasks, at least one promoted update has positive
affected-task LCB and protected-task LCB above the frozen non-inferiority
margin; the executor measurably follows the changed behavior.

### P5.2 — selective self-evolution and cost control

The primary condition uses Qwen3-VL-8B-Instruct for both execution and its own
evolution; no stronger external teacher is present. Replace unconditional
action-level model calls with deterministic public-feedback parsing,
uncertainty/novelty triggers, batched evidence, cache reuse, and one model call
only when rule-based evidence cannot resolve the transition. Track tokens by
purpose.

Exit: retain attribution/update reliability within the frozen tolerance while
reducing evolution-model tokens by at least 80% from the current Full arm and
below the matched EmbodiSkill* trajectory control.

### P5.3 — EB-NAV becomes a complete evolution path

Construct and hash non-official development tasks, implement environment-neutral
experiment orchestration, and share VTCA/gate/agenda/guard code with EB-HAB.
Only action schemas, observations, public feedback parsers, and goal grounding
may be environment adapters.

Exit: acquisition, proposal, paired selection, audit, freeze, and cost logging
run in both environments with the same core method and no official-test tuning.

### P5.4 — teacher-capability contingency

Open GPT-5.2 only if a pre-registered clean-development comparison shows that
Qwen reaches correct attribution but systematically fails candidate generation
or revision, while GPT-5.2 repairs the same cached clusters and improves held-out
affected tasks without protected regression. Report this as a separate teacher
condition matching EmbodiSkill; it does not replace the Qwen self-evolution
main goal.

### P5.5 — freeze and official evaluation

Freeze code, prompts, schemas, Skills, configs, endpoint stack, seeds, and
analysis before any new official run. Run every subset in stock order and
publish failures as well as successes. Any method change after opening official
evaluation creates a new preregistered version and cannot use those outcomes
for tuning.

## Immediate critical path

1. Restore EB-HAB EGL access on the simulator client.
2. Implement and unit-test the outcome-independent semantic gate.
3. Add goal agenda and action-admissibility telemetry before enabling blocking.
4. Build clean EB-NAV development tasks and lift the EB-HAB-only experiment
   restriction.
5. Run small clean-development Qwen self-evolution experiments on the two
   endpoints; scale only arms that pass frozen promotion and cost gates.
