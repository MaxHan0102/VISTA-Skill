#!/usr/bin/env bash
set -euo pipefail

cd /root/max/VISTA-Skill

PY=/root/miniconda3/envs/max_embench/bin/python
OUT=running/phase3b
DATASET=running/phase3a/phase3a_dataset_v3_cachefix_20260825.json
FB_CACHE=running/phase3a/cache_images_feedback_20260824.jsonl
IMAGE_CACHE=running/phase3a/cache_images_only_20260824.jsonl
EXECUTOR_ENDPOINT=http://192.168.1.185:8000/v1
EVIDENCE_ENDPOINT_A=http://192.168.1.185:8000/v1
EVIDENCE_ENDPOINT_B=http://192.168.1.173:8001/v1

mkdir -p "$OUT"

run_stage() {
    local stage="$1"
    shift
    "$PY" scripts/phase3a_run_stage.py \
        --stage "$stage" \
        --runtime-manifest "$OUT/$stage/runtime_manifest.json" \
        --console-log "$OUT/$stage/console.log" \
        --endpoint "$EXECUTOR_ENDPOINT" \
        --endpoint "$EVIDENCE_ENDPOINT_A" \
        --endpoint "$EVIDENCE_ENDPOINT_B" \
        --config-file configs/vista_fault_repair_fullsel_p10.json \
        --config-file configs/eb_hab_train_validation_manifest.json \
        -- "$@"
}

for arm in C0_current C1_no_feedback C2_temporal_no_feedback C3_temporal_feedback; do
    run_stage "executor_${arm}" \
        env PYTHONPATH=EmbodiedBench:. OPENAI_API_KEY=EMPTY \
        "$PY" scripts/phase3b_executor_feedback_pilot.py \
        --arm "$arm" \
        --episodes 30 \
        --seed 0 \
        --base-url "$EXECUTOR_ENDPOINT" \
        --output "$OUT/executor_${arm}/events.jsonl" \
        --summary "$OUT/executor_${arm}/summary.json"
done

run_stage evidence_T0_pair_strict \
    env PYTHONPATH=. OPENAI_API_KEY=EMPTY \
    "$PY" scripts/phase3b_extract_strict_temporal_evidence.py \
    --dataset "$DATASET" \
    --condition T0_pair_strict \
    --base-url "$EVIDENCE_ENDPOINT_A" \
    --base-url "$EVIDENCE_ENDPOINT_B" \
    --output "$OUT/T0_pair_strict.jsonl" \
    --summary "$OUT/T0_pair_strict.summary.json"

run_stage evidence_T1_temporal_strict \
    env PYTHONPATH=. OPENAI_API_KEY=EMPTY \
    "$PY" scripts/phase3b_extract_strict_temporal_evidence.py \
    --dataset "$DATASET" \
    --condition T1_temporal_strict \
    --base-url "$EVIDENCE_ENDPOINT_A" \
    --base-url "$EVIDENCE_ENDPOINT_B" \
    --output "$OUT/T1_temporal_strict.jsonl" \
    --summary "$OUT/T1_temporal_strict.summary.json"

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
    --dataset "$DATASET" \
    --pair-strict "$OUT/T0_pair_strict.jsonl" \
    --temporal-strict "$OUT/T1_temporal_strict.jsonl" \
    --images-feedback "$FB_CACHE" \
    --images-only "$IMAGE_CACHE" \
    --output "$OUT/phase3b_temporal_evidence_gated_analysis_20260825.json"
