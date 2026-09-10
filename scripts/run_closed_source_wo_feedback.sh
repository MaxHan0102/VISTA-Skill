#!/usr/bin/env bash
# Reuse exported OpenLux credentials and DISPLAY; run the six v2 comparisons.
set -euo pipefail

task_dry_run=0
case "${1:-}" in
    "") ;;
    --dry-run) task_dry_run=1 ;;
    -h|--help)
        echo "Usage: bash scripts/run_closed_source_wo_feedback.sh [--dry-run]"
        echo "Runs three models on Habitat and Navigation, each pilot then full."
        echo "Requires exported OPENAI_API_KEY, GEMINI_API_KEY and DISPLAY."
        exit 0
        ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    echo "Expected at most --dry-run" >&2
    exit 2
fi

if (( ! task_dry_run )); then
    : "${OPENAI_API_KEY:?Export OPENAI_API_KEY before running this script}"
    : "${GEMINI_API_KEY:?Export GEMINI_API_KEY before running this script}"
    : "${DISPLAY:?Export DISPLAY for Navigation before running this script}"
fi

task_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# Conda shell functions are not inherited by a new bash process.
source /root/miniconda3/etc/profile.d/conda.sh
cd "$task_repo/EmbodiedBench"

for task_env in eb-hab eb-nav; do
    if [[ "$task_env" == eb-hab ]]; then
        conda activate max_embench
    else
        conda activate max_embench_nav
    fi

    for model_spec in \
        openai:gpt-5.4-mini \
        gemini:gemini-3-flash-preview \
        gemini:gemini-3.5-flash
    do
        task_provider="${model_spec%%:*}"
        task_model="${model_spec#*:}"
        for task_stage in pilot full; do
            if [[ "$task_stage" == pilot ]]; then
                task_sets=base
                task_episodes=2
            else
                task_sets=all
                task_episodes=0
            fi
            echo "Starting $task_env / $task_model / $task_stage (rgb_only, seed 0)"
            task_args=(
                --provider "$task_provider" --model "$task_model"
                --env "$task_env" --track rgb_only
                --eval-sets "$task_sets" --episodes "$task_episodes" --seed 0
                --output "$task_repo/running/closed_feedback/${task_model}_${task_env}_v2_${task_stage}"
                --resume-existing
            )
            if (( task_dry_run )); then
                task_args+=(--dry-run)
            fi
            if python "$task_repo/scripts/evaluate_closed_loop_feedback.py" "${task_args[@]}"; then
                echo "Finished $task_env / $task_model / $task_stage"
            else
                task_status=$?
                echo "Stopped: $task_env / $task_model / $task_stage failed (exit $task_status)." >&2
                exit "$task_status"
            fi
        done
    done
done
if (( task_dry_run )); then
    echo "All 12 configurations validated; no API calls or simulator runs."
else
    echo "All six model/environment groups completed (pilot and full)."
fi
