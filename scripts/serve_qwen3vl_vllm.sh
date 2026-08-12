#!/usr/bin/env bash
# Serve Qwen3-VL-8B-Instruct locally with vLLM as an OpenAI-compatible API,
# matching the VISTA-Skill controlled-protocol executor config
# (configs/vista_p0.json: model=Qwen3-VL-8B-Instruct, precision=fp8, tp=1,
#  max_model_len=16384, temperature=0, supports OpenAI `seed` field).
#
# Usage:  bash scripts/serve_qwen3vl_vllm.sh [GPU_ID] [PORT] [QUANT]
#   GPU_ID  default 0   (CUDA_VISIBLE_DEVICES; tp=1 -> single GPU)
#   PORT    default 8000
#   QUANT   default fp8 (fallback: bf16 -- but bf16 8B needs more VRAM / tp=2)
#
# Weights are read from the HF cache (already present); no download.
set -euo pipefail

GPU_ID="${1:-0}"
PORT="${2:-8000}"
QUANT="${3:-fp8}"
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
# Served name must (a) be the exact string VISTA-Skill sends in the request
# `model` field, (b) contain substring "Qwen3-VL" (RemoteModel dispatch), and
# (c) end with "Qwen3-VL-8B-Instruct" (_validate_controlled_executor). The HF
# id satisfies all three and matches the CLI --model-name default, so no 404.
SERVED_NAME="Qwen/Qwen3-VL-8B-Instruct"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1              # weights already cached; never hit network
export VLLM_NO_USAGE_STATS=1

# --quantization fp8 = dynamic fp8 weight quantization (fits 8B in ~8GB VRAM).
# If fp8 fails to load, re-run with QUANT=bf16 and GPU tp=2 (edit --tensor-parallel-size).
QFLAG=()
if [[ "$QUANT" == "fp8" ]]; then
  QFLAG=(--quantization fp8)
fi

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --trust-remote-code \
  --port "$PORT" \
  --gpu-memory-utilization 0.5 \
  --dtype auto \
  "${QFLAG[@]}"
