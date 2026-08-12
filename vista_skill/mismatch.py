from __future__ import annotations

import hashlib
from typing import Sequence

from vista_skill.schemas import (
    ExpectedChange,
    Mismatch,
    MismatchKind,
    PredicateEvidence,
    TruthValue,
)


def compare_transitions(
    expected: Sequence[ExpectedChange],
    evidence: Sequence[PredicateEvidence],
) -> tuple[Mismatch, ...]:
    evidence_by_key = {item.key: item for item in evidence}
    expected_by_key = {item.key: item for item in expected}
    mismatches: list[Mismatch] = []

    for prediction in expected:
        observed = evidence_by_key.get(prediction.key)
        if observed is None:
            mismatches.append(_mismatch(prediction, None, MismatchKind.UNCOVERED))
            continue
        if observed.after is TruthValue.UNKNOWN or observed.coverage <= 0.0:
            mismatches.append(
                _mismatch(prediction, observed, MismatchKind.EXPECTED_UNSUPPORTED)
            )
            continue
        if observed.after is prediction.after:
            continue
        if prediction.skill_field is not None and prediction.skill_field.value == "termination":
            kind = MismatchKind.TERMINATION_CONFLICT
        elif observed.before is observed.after and prediction.before is not prediction.after:
            kind = MismatchKind.MISSING_PROGRESS
        else:
            kind = MismatchKind.CONTRADICTION
        mismatches.append(_mismatch(prediction, observed, kind))

    for observed in evidence:
        if observed.key in expected_by_key or observed.after is TruthValue.UNKNOWN:
            continue
        if observed.before is not observed.after:
            mismatches.append(
                _mismatch(None, observed, MismatchKind.SUPPORTED_UNEXPECTED)
            )
    return tuple(mismatches)


def _mismatch(
    expected: ExpectedChange | None,
    evidence: PredicateEvidence | None,
    kind: MismatchKind,
) -> Mismatch:
    if expected is None and evidence is None:
        raise ValueError("mismatch requires an expected change or evidence")
    key = expected.key if expected is not None else evidence.key
    evidence_ids = () if evidence is None else (evidence.evidence_id,)
    raw_id = "|".join(
        (
            key.render(),
            kind.value,
            "" if expected is None else expected.source_id,
            *evidence_ids,
        )
    )
    return Mismatch(
        mismatch_id=hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:20],
        key=key,
        kind=kind,
        expected=expected,
        evidence=evidence,
        evidence_ids=evidence_ids,
    )
