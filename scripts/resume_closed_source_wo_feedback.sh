#!/usr/bin/env bash
set -euo pipefail
# API keys and DISPLAY are inherited; each worker uses its matching conda env.
task_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec /root/miniconda3/envs/max_embench/bin/python \
    "$task_repo/scripts/resume_closed_source_wo_feedback.py" "$@"
