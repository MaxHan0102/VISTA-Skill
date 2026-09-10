#!/usr/bin/env bash
# One inference server per benchmark; pilot completion gates the full run.
set -euo pipefail
task_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
task_env=${1:?Usage: bash scripts/run_qwen_feedback_experiment.sh eb-hab|eb-nav [base-url]}
case "$task_env" in
  eb-hab) task_url=${2:-http://192.168.1.185:8000/v1} ;;
  eb-nav) task_url=${2:-http://192.168.1.173:8001/v1}; export DISPLAY=${DISPLAY:-:0} ;;
  *) echo "Unsupported environment: $task_env" >&2; exit 2 ;;
esac
# LAN inference must not pass through a public proxy.
export NO_PROXY="${NO_PROXY:-},192.168.1.185,192.168.1.173,127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
cd "$task_repo"
task_python=/root/miniconda3/envs/max_embench/bin/python
task_pilot="$task_repo/running/test/Qwen3-VL-8B-Instruct_${task_env}_feedback_pilot"
task_full="$task_repo/running/closed_feedback/Qwen3-VL-8B-Instruct_${task_env}_v2_full"
"$task_python" scripts/run_qwen_wo_feedback.py --env "$task_env" --base-url "$task_url" \
    --episodes 2 --output "$task_pilot"
exec "$task_python" scripts/run_qwen_wo_feedback.py --env "$task_env" --base-url "$task_url" \
    --output "$task_full"
