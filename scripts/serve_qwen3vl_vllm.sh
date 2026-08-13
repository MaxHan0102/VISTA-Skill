#!/usr/bin/env bash
# Serve Qwen3-VL-8B-Instruct locally with vLLM as an OpenAI-compatible API,
# matching the VISTA-Skill controlled-protocol executor contract
# (configs/vista_p0.json: model ends with "Qwen3-VL-8B-Instruct", precision=fp8,
#  temperature=0, accepts the OpenAI `seed` field, returns usage tokens).
#
# === Verified working recipe on this box (driver 550.90.07 / CUDA 12.4) ===
# Constraints discovered while bringing it up (see docs/memory):
#   - Driver 550 => CUDA 12.4 only. vLLM 0.27.x is CUDA-13 only and WILL NOT run.
#   - Use vLLM 0.11.0 (min version with Qwen3-VL arch) + torch 2.8 cu128
#     (cu128 empirically runs on driver 550) + transformers 4.57.x (NOT 5.x:
#     5.x removed all_special_tokens_extended that vLLM 0.11 reads).
#   - flashinfer MUST be uninstalled: its `array.array[int]` annotation raises
#     TypeError at import and vLLM's guard only catches ImportError.
#   - GPUs are shared (~7 GiB resident per card), so use tp=2, util 0.6,
#     --enforce-eager, max_model_len 16384, --max-num-batched-tokens 4096.
#     IMPORTANT: max_model_len MUST be 16384 (the config value), NOT 8192 -- the
#     10-shot executor prompt (~4.2k tok) + max_tokens (up to 4096) overflows
#     8192 and the planner's 3 retries on the deterministic 400 exhaust, aborting
#     the episode. --max-num-batched-tokens 4096 shrinks the profile activation
#     so the larger context fits the shared-GPU KV budget. On a DEDICATED GPU
#     switch to TP=1 / util 0.9 and drop --enforce-eager for faster serving.
#
# Usage:  bash scripts/serve_qwen3vl_vllm.sh [PORT]
set -euo pipefail

PORT="${1:-8001}"
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
SERVED_NAME="Qwen/Qwen3-VL-8B-Instruct"   # must match configs/vista_p0.json + contain "Qwen3-VL"
PY="${VLLM_PY:-/root/miniconda3/envs/max_vllm/bin/python}"

export HF_HUB_OFFLINE=1              # weights already in HF cache; never hit network
export VLLM_NO_USAGE_STATS=1
# unset CUDA_VISIBLE_DEVICES so tp=2 sees both GPUs

exec "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.6 \
  --enforce-eager \
  --trust-remote-code \
  --port "$PORT" \
  --quantization fp8
