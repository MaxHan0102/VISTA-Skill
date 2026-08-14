"""T3 -- RQ2 attribution falsifiability against a live vLLM endpoint.

The paper's RQ2 hinge: evidence-decoupled ACTION-level attribution must identify
the correct update target/field better than TRAJECTORY-level routing (which is the
EmbodiSkill*/VISTA-w/o-VTCA frontend: reflect on a failed trajectory and route the
failure -- effectively skill_update for any non-execution-lapse failure).

This script runs the fault-injection diagnostic three ways and compares:
  - VTCA rule-first attribution   (CreditAssigner, rule-only -- the method default)
  - VTCA teacher attribution      (CreditAssigner + JsonAttributionTeacher on vLLM)
  - trajectory-routing baseline   (predict skill_update for every case)

RQ2 Go signal: VTCA target_macro_f1 >> trajectory-routing target_macro_f1, and
field_macro_f1 > 0 (it locates the right skill field). No simulator needed.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

from vista_skill.action_schema import FixedActionSchema
from vista_skill.attribution import CreditAssigner
from vista_skill.config import load_config
from vista_skill.fault_injection import build_fault_cases, run_fault_injection_evaluation
from vista_skill.metrics import macro_f1
from vista_skill.models import JsonAttributionTeacher, OpenAICompatibleJsonModel
from vista_skill.skills import initialize_shared_skill

BASE_URL = os.environ.get("VISTA_METHOD_BASE_URL", "http://127.0.0.1:8001/v1")
MODEL = os.environ.get("VISTA_METHOD_MODEL", "Qwen/Qwen3-VL-8B-Instruct")


def _trajectory_baseline_macrof1(cases) -> float:
    """Trajectory routing: every failure -> skill_update (EmbodiSkill/w/o-VTCA)."""
    gold = [c.gold_target.value for c in cases]
    pred = ["skill_update"] * len(cases)
    return macro_f1(gold, pred)


def _row(name, report, billing=None):
    conf = report.confusion
    print(f"\n{name}")
    print(f"  target_macro_f1 = {report.target_macro_f1:.3f}   field_macro_f1 = {report.field_macro_f1:.3f}")
    print(f"  abstention_prec = {report.abstention_precision:.3f}   abstention_recall = {report.abstention_recall:.3f}")
    print(f"  confusion (gold->pred): { {f'{k[0]}->{k[1]}': v for k, v in conf.items()} }")
    if billing is not None:
        u = billing.get("vista_attribution")
        print(f"  teacher calls   = {u.calls if u else 0}")


def main() -> int:
    config = load_config("configs/vista_p0.json")
    schema = FixedActionSchema()
    skill = initialize_shared_skill()
    cases = build_fault_cases(skill, schema)
    print(f"fault cases: {len(cases)} | gold target distribution: {dict(Counter(c.gold_target.value for c in cases))}")

    # baseline (analytic)
    base_f1 = _trajectory_baseline_macrof1(cases)
    print(f"\n[trajectory-routing baseline] target_macro_f1 = {base_f1:.3f}  (predicts skill_update for every case)")

    # VTCA rule-first
    rule_report = run_fault_injection_evaluation(
        CreditAssigner(config=config.attribution), skill, cases, action_schema=schema,
    )
    _row("[VTCA rule-first attribution]", rule_report)

    # VTCA + teacher (real vLLM)
    model = OpenAICompatibleJsonModel(
        MODEL, base_url=BASE_URL, api_key="EMPTY", temperature=0.0, max_tokens=512, seed=0,
    )
    teacher_report = run_fault_injection_evaluation(
        CreditAssigner(JsonAttributionTeacher(model), config.attribution),
        skill, cases, action_schema=schema,
    )
    _row("[VTCA teacher attribution (vLLM)]", teacher_report, billing=model.usage)

    # RQ2 verdict
    rule_gain = rule_report.target_macro_f1 - base_f1
    teach_gain = teacher_report.target_macro_f1 - base_f1
    print("\n=== RQ2 signal ===")
    print(f"  rule-first  target Macro-F1 gain over trajectory routing: {rule_gain:+.3f}")
    print(f"  teacher     target Macro-F1 gain over trajectory routing: {teach_gain:+.3f}")
    print(f"  rule-first  field Macro-F1: {rule_report.field_macro_f1:.3f}  (>0 => locates the skill field)")
    ok = rule_report.target_macro_f1 > base_f1 + 0.1 and rule_report.field_macro_f1 > 0.5
    print(f"  VERDICT: {'VTCA attribution BEATS trajectory routing' if ok else 'weak -- investigate'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
