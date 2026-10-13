#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 5 ]]; then
  echo "Usage: bash train_trajectory_prediction.sh DATA.npz OUTPUT_DIR [cpu|cuda:0] [CONFIG.json] [--resume]" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
xuance_root="$(cd -- "$script_dir/.." && pwd)"
data_path="$1"
output_path="$2"
device="${3:-cuda:0}"
entry="$xuance_root/Trajectory Prediction/run.py"
config="${4:-$xuance_root/Trajectory Prediction/configs/server.json}"
resume_args=()
intention_epochs="${TRAJECTORY_INTENTION_EPOCHS:-70}"
if [[ $# -eq 5 ]]; then
  [[ "$5" == "--resume" ]] || { echo "Fifth argument must be --resume" >&2; exit 2; }
  resume_args=(--resume)
fi
python "$entry" check --device "$device" --config "$config" --data "$data_path" --backward
python "$entry" pipeline --data "$data_path" --config "$config" --output "$output_path" --device "$device" --intention-epochs "$intention_epochs" "${resume_args[@]}"
