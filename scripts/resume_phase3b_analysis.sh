#!/usr/bin/env bash
set -euo pipefail

cd /root/max/VISTA-Skill

PY=/root/miniconda3/envs/max_embench/bin/python
OUT=running/Phase3/phase3b

run_stage() {
    local stage="$1"
    shift
    "$PY" scripts/phase3a_run_stage.py \
        --stage "$stage" \
        --runtime-manifest "$OUT/$stage/runtime_manifest.json" \
        --console-log "$OUT/$stage/console.log" \
        --config-file configs/vista_fault_repair_fullsel_p10.json \
        --config-file configs/eb_hab_train_validation_manifest.json \
        -- "$@"
}

run_stage analyze_executor \
    env PYTHONPATH=. \
    "$PY" scripts/phase3b_analyze_executor_pilot.py \
    --summary "$OUT/executor_C0_current/summary.json" \
    --summary "$OUT/executor_C1_no_feedback/summary.json" \
    --summary "$OUT/executor_C2_temporal_no_feedback/summary.json" \
    --summary "$OUT/executor_C3_temporal_feedback/summary.json" \
    --output "$OUT/phase3b_executor_paired_analysis_20260825.json"

run_stage audit_temporal \
    env PYTHONPATH=. \
    "$PY" scripts/phase3b_audit_temporal_evidence.py \
    --dataset running/Phase3/phase3a/phase3a_dataset_v3_cachefix_20260825.json \
    --pair-strict "$OUT/T0_pair_strict.jsonl" \
    --temporal-strict "$OUT/T1_temporal_strict.jsonl" \
    --images-feedback running/Phase3/phase3a/cache_images_feedback_20260824.jsonl \
    --images-only running/Phase3/phase3a/cache_images_only_20260824.jsonl \
    --output "$OUT/phase3b_temporal_evidence_gated_analysis_20260825.json"
