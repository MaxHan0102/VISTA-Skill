#!/usr/bin/env bash
# Serve Qwen3-VL-8B-Instruct locally with vLLM as an OpenAI-compatible API,
# matching the VISTA-Skill controlled-protocol executor contract
# (configs/vista_p0.json: model ends with "Qwen3-VL-8B-Instruct", precision=fp8,
#  temperature=0, accepts the OpenAI `seed` field, returns usage tokens).
#
# === Verified working recipe (vLLM 0.11.0 / torch 2.8 / transformers 4.57.x) ===
# Constraints discovered while bringing it up (see docs/memory):
#   - Use vLLM 0.11.0 (min version with Qwen3-VL arch) + torch 2.8
#     (cu128/cu129 both fine) + transformers 4.57.x (NOT 5.x: 5.x removed
#     all_special_tokens_extended that vLLM 0.11 reads).
#   - flashinfer MUST be uninstalled: its `array.array[int]` annotation raises
#     TypeError at import and vLLM's guard only catches ImportError.
#   - Driver 550 => CUDA 12.4 only (vLLM 0.27.x is CUDA-13 only and WILL NOT
#     run there); driver 575+ also works with this exact recipe.
#
# GPU layout is AUTO-DETECTED at launch:
#   - TP: 1 visible card -> tp=1 (8B fp8 ~= 9 GiB weights, fits one 24 GiB
#     card); >=2 cards -> tp=2. Larger tp is unverified and unnecessary at 8B;
#     override with TP=N if you really want it.
#   - Cards that already hold > BUSY_THRESHOLD_MIB (default 2048) MiB of other
#     processes -> treat the box as SHARED: util 0.6 + --enforce-eager +
#     --max-num-batched-tokens 4096 (the empirically safe shared-GPU recipe).
#     Otherwise use the dedicated-GPU recipe: util 0.9, no --enforce-eager,
#     vLLM-default batched tokens.
#   - IMPORTANT: --max-model-len is pinned to 16384 (the configs/vista_p0.json
#     value) on every layout. The 10-shot executor prompt (~4.2k tok) +
#     max_tokens (up to 4096) overflows 8192 and the planner's 3 retries on the
#     deterministic 400 exhaust, aborting the episode. Do NOT shrink it.
#   - Protocol note: controlled evaluation validates the tensor-parallel setting
#     recorded in run artifacts -- runs with different TP are not directly
#     comparable; keep TP matched across anything you report side by side.
#
# Env overrides: VLLM_PY (server python), TP, UTIL, BUSY_THRESHOLD_MIB.
# CUDA_VISIBLE_DEVICES, if set, restricts detection to the listed cards.
#
# Usage:  bash scripts/serve_qwen3vl_vllm.sh [PORT]
#          (docs/implementation.md CLI examples expect port 8000)
set -euo pipefail

PORT="${1:-8001}"
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
SERVED_NAME="Qwen/Qwen3-VL-8B-Instruct"   # must match configs/vista_p0.json + contain "Qwen3-VL"
PY="${VLLM_PY:-/root/miniconda3/envs/max_vllm/bin/python}"

export HF_HUB_OFFLINE=1              # weights already in HF cache; never hit network
export VLLM_NO_USAGE_STATS=1

# --- sanity checks ----------------------------------------------------------
[[ -x "$PY" ]] || { echo "ERROR: server python not found at $PY (set VLLM_PY=...)" >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi not available" >&2; exit 1; }

# The env's bin/ must be on PATH: vLLM's compile path shells out to `ninja`,
# which pip installs next to $PY but nothing puts on PATH without conda activate.
export PATH="$(dirname "$PY"):$PATH"

# HF_HUB_OFFLINE=1 forbids downloads, so the weights must already be cached.
HUB_DIR="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
if ! ls "$HUB_DIR"/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*/model.safetensors.index.json >/dev/null 2>&1; then
  echo "ERROR: Qwen3-VL-8B-Instruct weights not found under $HUB_DIR" >&2
  echo "       pre-download them, or unset HF_HUB_OFFLINE for the first run" >&2
  exit 1
fi

# --- GPU autodetection ------------------------------------------------------
# Respect an explicit CUDA_VISIBLE_DEVICES selection; otherwise use every card.
if [[ ${CUDA_VISIBLE_DEVICES+set} == set && -z "$CUDA_VISIBLE_DEVICES" ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES is set but empty (= no GPUs)" >&2; exit 1
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NGPU=$(( $(tr -s ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | grep -c .) ))
else
  NGPU="$(nvidia-smi --list-gpus | wc -l)"
fi
(( NGPU >= 1 )) || { echo "ERROR: no GPUs visible (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}')" >&2; exit 1; }

# Worst-case per-card memory (MiB): min total, max already used by others.
read -r MIN_TOTAL_MIB MAX_USED_MIB < <(nvidia-smi --query-gpu=memory.total,memory.used \
  --format=csv,noheader,nounits | awk -F', *' 'NR==1||$1+0<mint{mint=$1+0} NR==1||$2+0>maxu{maxu=$2+0} END{print mint+0, maxu+0}')
(( MIN_TOTAL_MIB > 0 )) || { echo "ERROR: could not read GPU memory from nvidia-smi" >&2; exit 1; }

BUSY_THRESHOLD_MIB="${BUSY_THRESHOLD_MIB:-2048}"
if (( MAX_USED_MIB > BUSY_THRESHOLD_MIB )); then SHARED=1; else SHARED=0; fi

# TP: derived from visible card count, capped at the verified range [1, 2].
if [[ -z "${TP:-}" ]]; then
  if (( NGPU >= 2 )); then TP=2; else TP=1; fi
fi
(( TP <= NGPU )) || { echo "ERROR: TP=$TP exceeds visible GPU count ($NGPU)" >&2; exit 1; }

# UTIL: shared -> leave room for the other residents:
#   util = (0.95*total - used) / total, clamped to [0.2, 0.6].
# Dedicated multi-GPU -> 0.9.
# Dedicated SINGLE-GPU -> 0.87, not higher: on this project the EB-Habitat
#   simulator shares the same card during experiments, and the measured
#   non-KV overhead of this model at tp=1 non-eager is ~18.6 GiB
#   (fp8 weights + CUDA graphs + profiling activations), so
#   KV ~= util*24.56 - 18.6 GiB. One 16384-token sequence needs 2.25 GiB KV:
#   util 0.8 -> KV 1.28 GiB (vLLM aborts), 0.87 -> ~2.65 GiB (fits, with
#   ~3 GiB left for habitat-sim), 0.9 -> ~3.4 GiB but only 0.4 GiB free.
if [[ -z "${UTIL:-}" ]]; then
  if (( SHARED )); then
    UTIL=$(awk -v t="$MIN_TOTAL_MIB" -v u="$MAX_USED_MIB" \
      'BEGIN{util=(0.95*t-u)/t; if(util>0.6)util=0.6; if(util<0.2)util=0.2; printf "%.2f", util}')
  elif (( NGPU == 1 )); then
    UTIL=0.87
  else
    UTIL=0.9
  fi
fi

# Rough capacity guard: fp8 weights ~= 9 GiB split across TP, plus >=3 GiB KV.
WEIGHTS_PER_CARD=$(( 9000 / TP ))
UTIL_PCT="$(awk -v u="$UTIL" 'BEGIN{printf "%d", u*100}')"
BUDGET_PER_CARD=$(( MIN_TOTAL_MIB * UTIL_PCT / 100 ))
if (( BUDGET_PER_CARD < WEIGHTS_PER_CARD + 3000 )); then
  echo "ERROR: budget ~${BUDGET_PER_CARD} MiB/card < weights ~${WEIGHTS_PER_CARD} MiB + 3 GiB KV at TP=$TP / util=$UTIL" >&2
  echo "       free the card(s), lower BUSY/TP constraints, or raise UTIL" >&2
  exit 1
fi

EXTRA=()
if (( SHARED )); then
  EXTRA+=(--enforce-eager --max-num-batched-tokens 4096)
fi

echo "vLLM launch plan: gpus=$NGPU tp=$TP shared=$SHARED (others hold ${MAX_USED_MIB} MiB, card total ${MIN_TOTAL_MIB} MiB) util=$UTIL eager=$([[ $SHARED == 1 ]] && echo yes || echo no) port=$PORT" >&2

exec "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP" \
  --max-model-len 16384 \
  --gpu-memory-utilization "$UTIL" \
  --trust-remote-code \
  --port "$PORT" \
  --quantization fp8 \
  ${EXTRA[@]+"${EXTRA[@]}"}
