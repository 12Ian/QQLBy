#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo "Usage: bash run_trajectory_study.sh OUTPUT_DIR [cpu|cuda:0] [CONFIG.json] [--resume]" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
xuance_root="$(cd -- "$script_dir/.." && pwd)"
entry="$xuance_root/Trajectory Prediction/run.py"
device="${2:-cuda:0}"
config="${3:-$xuance_root/Trajectory Prediction/configs/paper_forecast.json}"
resume_args=()
if [[ $# -eq 4 ]]; then
  [[ "$4" == "--resume" ]] || { echo "Fourth argument must be --resume" >&2; exit 2; }
  resume_args=(--resume)
fi
python "$entry" check --device "$device" --config "$config" --backward
python "$entry" study --config "$config" --output "$1" --device "$device" \
  --data-seeds "${TRAJECTORY_DATA_SEEDS:-101,102,103}" --model-seeds "${TRAJECTORY_MODEL_SEEDS:-42,43,44}" \
  --tracks "${TRAJECTORY_STUDY_TRACKS:-180}" --steps "${TRAJECTORY_STUDY_STEPS:-240}" \
  --intention-epochs "${TRAJECTORY_INTENTION_EPOCHS:-70}" --figures "${resume_args[@]}"
