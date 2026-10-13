#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo "Usage: bash run_civilian_interaction.sh OUTPUT_DIR [cpu|cuda:0] [CONFIG.json] [--resume]" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
xuance_root="$(cd -- "$script_dir/.." && pwd)"
entry="$xuance_root/Trajectory Prediction/run.py"
output="$1"
device="${2:-cuda:0}"
config="${3:-$xuance_root/Trajectory Prediction/configs/paper_interaction.json}"
resume_args=()
if [[ $# -eq 4 ]]; then
  [[ "$4" == "--resume" ]] || { echo "Fourth argument must be --resume" >&2; exit 2; }
  resume_args=(--resume)
fi
python "$entry" check --config "$config" --device "$device" --backward
python "$entry" study --config "$config" --output "$output" --device "$device" \
  --data-mode civilian_interaction --scenes "${TRAJECTORY_STUDY_SCENES:-60}" --steps "${TRAJECTORY_STUDY_STEPS:-320}" \
  --data-seeds "${TRAJECTORY_DATA_SEEDS:-101,102,103}" --model-seeds "${TRAJECTORY_MODEL_SEEDS:-42,43,44}" \
  --models graph,lstm,gru,tcn,transformer,graph_no_intention,graph_no_context,graph_no_intention_context \
  --intention-epochs "${TRAJECTORY_INTENTION_EPOCHS:-70}" --figures "${resume_args[@]}"
