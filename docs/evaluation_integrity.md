# EmbodiedBench evaluation-integrity policy

Effective date: 2026-09-02. Machine-readable source of truth:
`configs/evaluation_data_policy_v1.json`.

## What can and cannot be restored

Some official EB-HAB and EB-NAV test results and trajectories were inspected in
earlier pilot work. That knowledge cannot be erased from the researchers or the
repository history. “Restoring pollution” therefore means a prospective claim
boundary, not pretending those exposures never happened:

- historical outputs remain available for provenance and debugging;
- Skills authored from official-test trajectories are marked and quarantined;
- historical official-test scores cannot select method content, prompts,
  thresholds, hyperparameters, or stopping decisions;
- controlled evaluation rejects contaminated or pre-policy Skill artifacts
  before constructing a simulator;
- new main-table values require a frozen method followed by a complete official
  subset run with diagnostic and episode-cap flags disabled.

The legacy human Target-Skills and offline Skill v2 results are post-hoc
diagnostics. They are not VISTA-Skill results and are ineligible for the main
paper. No file is deleted: deletion would hide provenance without undoing the
exposure.

## Allowed development data

- EB-HAB: the stock `train_validation` dataset, with acquisition, selection,
  and audit roles fixed before outcomes are read.
- EB-NAV: newly constructed non-official development tasks only. The released
  benchmark has no `train_validation` split, so official subset trajectories
  are not a substitute for development data.
- Simulator-derived labels may be used for diagnostic measurement on allowed
  development data, but must never enter executor/model inputs or final-test
  adaptation.

All six EB-HAB and all five EB-NAV official subsets are final evaluation data.
Aggregate scores already known from the benchmark or a paper may define a
target, but they may not drive patch-level or task-family tuning.

## Claim-eligible artifact contract

A controlled frozen Skill artifact must match all of the following:

- config and, for EB-HAB, task-manifest digests;
- executor name/type, tensor parallelism, temperature, completion cap,
  n-shots, and image resolution;
- environment-specific RNG policy and frozen state;
- current evaluation-data-policy ID and SHA-256;
- no oracle, post-hoc, official-test-derived, or source-contamination marker.

Evaluation summaries expose `claim_eligible` and contamination reasons. A run
is claim-eligible only when it evaluates a complete official subset after the
method is frozen. This flag establishes eligibility, not statistical validity;
the final report must still include every configured subset and seed.

## Fair comparison boundary

The EmbodiSkill paper values are the absolute performance target. Its public
repository currently exposes the ALFWorld implementation, not the EB-HAB or
EB-NAV code and training tasks behind the published results. Accordingly:

- future paper tables must explicitly distinguish reported EmbodiSkill results
  from locally executed controls;
- the local method is named `EmbodiSkill*` and treated as a controlled analogue,
  never an exact reproduction;
- performance must exceed the paper result, not merely `EmbodiSkill*`;
- attribution/update studies compare methods under matched initial Skill,
  acquisition experience, teacher condition, proposal budget, executor, and
  evaluation budget;
- any native teacher or budget difference is a separate arm.

## Enforcement implemented on 2026-09-02

- `vista_skill.integrity` loads and hashes the policy and classifies artifact
  contamination.
- both EB-HAB and EB-NAV now fail closed on runtime executor drift; EB-NAV no
  longer bypasses most protocol checks.
- executor completion limits are read from the frozen config. Both release
  configs use the stock EmbodiedBench RemoteModel cap of 4096; the earlier NAV
  config value of 1024 did not match the actual 4096 request and is retired.
- old artifacts lack the policy digest and cannot silently become controlled
  artifacts.
