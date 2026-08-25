#!/usr/bin/env bash
set -uo pipefail

cd /root/max/VISTA-Skill || exit 2
export PYTHONPATH=.
PY=/root/miniconda3/envs/max_embench/bin/python
BASE_URL=http://127.0.0.1:8000/v1
OUT=running/phase3c/offline
mkdir -p "$OUT"

run_one() {
  local output="$1"
  shift
  if [[ -s "$output" ]]; then
    echo "SKIP existing $output"
    return 0
  fi
  echo "START $output"
  "$@"
  local status=$?
  echo "END status=$status output=$output"
  return 0
}

run_one "$OUT/synthetic_meta.json" \
  "$PY" scripts/phase2_teacher_attribution.py \
  --base-url "$BASE_URL" --seeds 5 --meta-skills frozen_v1 \
  --output "$OUT/synthetic_meta.json"

for fault in constraint_pick_multihold effect_pick_inversion; do
  if [[ "$fault" == "constraint_pick_multihold" ]]; then
    events=running/phase2_multihold_provenance_fix/full/seed_0/acquisition.jsonl
  else
    events=running/phase2_effect_pick_inversion/full/seed_0/acquisition.jsonl
  fi
  for seed in 0 1 2 3 4; do
    run_one "$OUT/natural_${fault}_seed${seed}.json" \
      "$PY" scripts/phase2_natural_attribution_audit.py \
      --events "$events" --base-url "$BASE_URL" --fault "$fault" \
      --seed "$seed" --meta-skills frozen_v1 \
      --output "$OUT/natural_${fault}_seed${seed}.json"
  done
done

for fault in constraint_pick_multihold effect_pick_inversion; do
  run_one "$OUT/patch_${fault}_meta.json" \
    "$PY" scripts/phase2_patch_stability.py \
    --base-url "$BASE_URL" --fault "$fault" --trials 10 \
    --meta-skills frozen_v1 --output "$OUT/patch_${fault}_meta.json"
done

run_one "$OUT/metamorphic_meta.json" \
  "$PY" scripts/phase3c_metamorphic_attribution.py \
  --base-url "$BASE_URL" --seed 0 --output "$OUT/metamorphic_meta.json"

run_one "$OUT/phase3c_offline_decision.json" \
  "$PY" scripts/phase3c_finalize_offline.py \
  --root "$OUT" --output "$OUT/phase3c_offline_decision.json"

echo "PHASE3C_OFFLINE_COMPLETE"
