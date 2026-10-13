#!/usr/bin/env bash
set -euo pipefail
# Military intercept-engagement reproduction (paper Sections III-IV).
#
#   bash bash/run_engagement.sh OUTPUT_DIR [cpu|cuda:0] [intention|overload|forecast|ablation] [--resume]
#
#   intention  IV-A nine-class semantic intention   (dt=0.5 s, O=20, stride 5)
#   overload   IV-A eleven-level overload intensity (dt=0.5 s, O=20, stride 5)
#   forecast   IV-B trajectory extrapolation        (dt=0.1 s, O=80, P=60)
#   ablation   military I/E four quadrants + baselines (IV-B protocol)
#
# The kinematics are always propagated on the dt/stride internal grid, so the
# same generator serves both protocols (decision D4).
#
# Every run writes a timestamped console log and a machine-readable
# run_manifest.json under OUTPUT_DIR so results survive a lost SSH session and
# later review has the exact commands, config and environment.
#
# AutoDL note: the instance disk is small; point OUTPUT_DIR at the data disk
# (/root/autodl-tmp/...) so datasets, checkpoints and logs do not fill /.
if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: bash run_engagement.sh OUTPUT_DIR [cpu|cuda:0] [intention|overload|forecast|ablation] [--resume]" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
xuance_root="$(cd -- "$script_dir/.." && pwd)"
configs="$xuance_root/Trajectory Prediction/configs"
entry="$xuance_root/Trajectory Prediction/run.py"
PYTHON="${PYTHON:-python}"   # set PYTHON to a specific interpreter if needed
output="$1"
device="${2:-cuda:0}"
protocol="${3:-forecast}"
case "$protocol" in
  # O/P/stride mirror the config so --target-windows can size the dataset.
  # initial_fov_coverage: IV-A trains on both visible and invisible starts,
  # IV-B starts with the defender inside the target FOV (paper Section IV).
  intention) config="$configs/engagement_intention.json"; sim_dt=0.5; stride=5; split=test
             obs=20; pred=0;  wstride=5; fov_coverage=0.5 ;;
  overload)  config="$configs/engagement_overload.json";  sim_dt=0.5; stride=5; split=test
             obs=20; pred=0;  wstride=5; fov_coverage=0.5 ;;
  forecast)  config="$configs/engagement_forecast.json";  sim_dt=0.1; stride=1; split=validation
             obs=80; pred=60; wstride=5; fov_coverage=1.0 ;;
  ablation)  config="$configs/engagement_forecast.json";  sim_dt=0.1; stride=1; split=validation
             obs=80; pred=60; wstride=5; fov_coverage=1.0 ;;
  # literal: the engineering forecast config with the two paper-literal knobs
  # (E1 decoder_zero_init=False, B4 teacher_temperature=1.0) - audit round 4.
  literal)   config="$configs/engagement_forecast_literal.json"; sim_dt=0.1; stride=1; split=validation
             obs=80; pred=60; wstride=5; fov_coverage=1.0 ;;
  *) echo "Protocol must be intention, overload, forecast, ablation or literal" >&2; exit 2 ;;
esac
resume_args=()
if [[ $# -eq 4 ]]; then
  [[ "$4" == "--resume" ]] || { echo "Fourth argument must be --resume" >&2; exit 2; }
  resume_args=(--resume)
fi
scenes="${TRAJECTORY_ENGAGEMENT_SCENES:-60}"
steps="${TRAJECTORY_ENGAGEMENT_STEPS:-400}"
# Optional epoch override, used for a fast end-to-end self-check before the
# full 200-epoch run (TRAJECTORY_EPOCHS=1 TRAJECTORY_TARGET_WINDOWS=300 ...).
epoch_args=()
[[ -n "${TRAJECTORY_EPOCHS:-}" ]] && epoch_args=(--epochs "$TRAJECTORY_EPOCHS")
# Paper IV-A/IV-B both report ~5x10^4 samples; the generator grows --scenes
# until the sliding-window count reaches the target (set 0 to keep --scenes).
target_windows="${TRAJECTORY_TARGET_WINDOWS:-50000}"
data="$output/data/engagement_${protocol}.npz"
rundir="$output/run/$protocol"
if [[ "$protocol" == "literal" ]]; then
  # The literal config trains on the same data as forecast (same protocol,
  # same seed), so reuse the forecast npz instead of regenerating it.
  data="$output/data/engagement_forecast.npz"
  rundir="$output/run/literal"
fi
mkdir -p "$output/data" "$output/logs"

# ---- unified logging: console + timestamped file --------------------------------
stamp="$(date +%Y%m%d_%H%M%S)"
log="$output/logs/${protocol}_${stamp}.log"
exec > >(tee -a "$log") 2>&1
echo "run started: $(date -Is)"
echo "command:     bash bash/run_engagement.sh $*"
echo "python:      $($PYTHON --version 2>&1)  ($PYTHON)"
echo "device arg:  $device"
echo "output root: $output"

# ---- round-2 calibration switches ----------------------------------------------
# TRAJECTORY_EVASION_RANGE "lo,hi"  per-segment evasion overload as fractions of
#                                 aT-max (paper IV-B "sampled within predefined
#                                 bounds"; default 0.5,1.0 = round 1).
# TRAJECTORY_CENSORED_ENDS 1       widen the two extreme overload bins so the
#                                 +/-10 g levels are reachable (eleven-level
#                                 task; 9 g saturation empties (9.5,10.5]).
# TRAJECTORY_PRIMITIVE_DURATION "lo,hi"  maneuver-primitive segment duration
#                                 bounds in seconds (paper leaves the bounds
#                                 open; default 2.0,6.0 = round 1).
# TRAJECTORY_EVASION_SEGMENT_S seconds  evasion segment length before the
#                                 nominal overload resamples (default 4).
evasion_range="${TRAJECTORY_EVASION_RANGE:-0.5,1.0}"
censored_ends="${TRAJECTORY_CENSORED_ENDS:-0}"
primitive_duration="${TRAJECTORY_PRIMITIVE_DURATION:-2.0,6.0}"
evasion_segment_s="${TRAJECTORY_EVASION_SEGMENT_S:-4.0}"
overload_args=(--evasion-overload-range "$evasion_range"
               --primitive-duration-range "$primitive_duration"
               --evasion-segment-seconds "$evasion_segment_s")
[[ "$censored_ends" == "1" ]] && overload_args+=(--censored-overload-ends)
echo "evasion overload range: $evasion_range   censored overload ends: $censored_ends"
echo "primitive duration:     $primitive_duration   evasion segment: ${evasion_segment_s}s"

# ---- machine-readable run manifest (appends one entry per protocol run) --------
manifest="$output/run_manifest.json"
"$PYTHON" - "$manifest" "$protocol" "$log" "$config" <<'PY' || true
import datetime, json, sys
from pathlib import Path
path, protocol, log, config = sys.argv[1:5]
state = {}
file = Path(path)
if file.exists():
    try:
        state = json.loads(file.read_text(encoding="utf-8"))
    except Exception:
        state = {}
state.setdefault("runs", {})[protocol] = {
    "started": datetime.datetime.now().isoformat(timespec="seconds"),
    "log": log,
    "config": config,
}
file.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

"$PYTHON" "$entry" check --config "$config" --device "$device" --backward
if [[ ! -f "$data" ]]; then
  # Table IV sampling; the trigger is r <= (2/3) r0 with 9 g / 15 g saturation.
  "$PYTHON" "$entry" simulate-engagement --output "$data" --scenes "$scenes" --steps "$steps" \
    --dt "$sim_dt" --sample-stride "$stride" --seed "${TRAJECTORY_ENGAGEMENT_SEED:-42}" \
    --aT-max-g 9 --aM-max-g 15 --fov-deg 60 \
    --initial-fov-coverage "$fov_coverage" \
    --target-windows "$target_windows" --observation-steps "$obs" \
    --prediction-steps "$pred" --window-stride "$wstride" \
    "${overload_args[@]}"
fi
if [[ "$protocol" == "ablation" ]]; then
  # I axis: intention on/off. E axis: engagement edges + candidate cropper +
  # E/R/T teachers on/off (decision D1/D5).
  "$PYTHON" "$entry" study --config "$config" --output "$output/ablation" --device "$device" \
    --data-mode engagement --scenes "$scenes" --steps "$steps" \
    --data-seeds "${TRAJECTORY_DATA_SEEDS:-101,102,103}" --model-seeds "${TRAJECTORY_MODEL_SEEDS:-42,43,44}" \
    --models graph,graph_no_intention,graph_no_engagement,graph_no_intention_engagement \
    --intention-epochs "${TRAJECTORY_INTENTION_EPOCHS:-70}" "${resume_args[@]}"
else
  # One sub-directory per protocol so bash/run_paper_targets.sh can find the
  # checkpoints and metrics of every experiment under a single run root.
  "$PYTHON" "$entry" train --data "$data" --config "$config" --output "$rundir" --device "$device" \
    "${epoch_args[@]}" "${resume_args[@]}"
  "$PYTHON" "$entry" evaluate --data "$data" --checkpoint "$rundir/best.pt" --split "$split" \
    --device "$device" --output "$rundir/${split}_metrics.json"
  echo "Checkpoint: $rundir/best.pt"
  echo "Metrics:    $rundir/${split}_metrics.json"
  echo "Next:       bash bash/run_paper_targets.sh $output $device"
fi
# ---- finalize the manifest entry after a successful run ------------------------
"$PYTHON" - "$manifest" "$protocol" "$log" <<'PY' || true
import datetime, json, sys
from pathlib import Path
path, protocol, log = sys.argv[1:4]
file = Path(path)
state = {}
if file.exists():
    try:
        state = json.loads(file.read_text(encoding="utf-8"))
    except Exception:
        state = {}
entry = state.get("runs", {}).get(protocol, {})
entry["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
entry["status"] = "ok"
file.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
echo "run finished: $(date -Is); log saved to $log"
