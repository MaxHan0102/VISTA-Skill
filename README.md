# VISTA-Skill

VISTA-Skill implements evidence-decoupled visual transition credit assignment
for reliable evolution of procedural skills under partial observability. The
implementation lives in the standalone `vista_skill/` package. It reuses
EmbodiedBench through adapters and does not modify `EmbodiedBench/`.

The P0 method follows the latest design in
`context4agent/markdown/20260806-VISTA-Skill最新方案-证据解耦视觉转移信用分配.md`:

1. Compile a skill-predicted transition from the fixed action schema and the
   active five-field skill.
2. Build an evidence-supported transition from pre/post observations and
   public environment feedback in an isolated branch.
3. Attribute predicate-level mismatches to `belief_refresh`, `skill_update`,
   or `abstain`, then locate a skill field only for skill updates.
4. Require independent recurrence, a bounded field patch, and staged paired
   validation before a persistent skill version is promoted.

## Quick validation

```bash
python -m pytest
```

The core package uses only the Python standard library. Install the optional
OpenAI-compatible model client with `pip install -e '.[models]'`. The EB-Habitat runner
must be launched in an environment where EmbodiedBench and Habitat are already
installed:

```bash
python -m vista_skill.integrations.embodiedbench.cli --help
```

The CLI separates `experiment` from frozen `evaluate`. Full mode fails closed
without an explicit method-model backend; `rule_only` is labeled as a
diagnostic ablation. The checked-in task manifest fixes all 100 coordinates and
the dataset hash. Three independent evolution runs rotate the 60/20/20 roles,
evolve after each acquisition episode, freeze separately, and run post-hoc
paired update audits on their own held-out roles. Final evaluation rejects
non-frozen, tampered, or protocol-incompatible Skill artifacts.

See `docs/implementation.md` for architecture, invariants, experiment split
requirements, and integration details.
