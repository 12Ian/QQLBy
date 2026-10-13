#!/usr/bin/env bash
set -euo pipefail
# Reproduce the four headline numbers of the paper from trained checkpoints.
#
#   bash bash/run_paper_targets.sh RUN_ROOT [cpu|cuda:0]
#
# Expects run_engagement.sh to have produced, under RUN_ROOT:
#   data/engagement_intention.npz   run/intention/best.pt    (IV-A nine-class)
#   data/engagement_overload.npz    run/overload/best.pt     (IV-A eleven-level)
#   data/engagement_forecast.npz    run/forecast/best.pt     (IV-B extrapolation)
#
# Anchors: 88% nine-class / 94% eleven-level / 102 m ADE-FDE /
#          150-202-81-173 m on the Fig. 14 composite scenarios.
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash bash/run_paper_targets.sh RUN_ROOT [cpu|cuda:0]" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
xuance_root="$(cd -- "$script_dir/.." && pwd)"
module="$xuance_root/Trajectory Prediction"
entry="$module/run.py"
scripts="$module/scripts"
PYTHON="${PYTHON:-python}"
root="$1"
device="${2:-cuda:0}"
trials="${TRAJECTORY_MC_TRIALS:-1000}"
# Round-2 calibration: which checkpoint to score (best = 0.4*ADE+0.6*FDE /
# validation-CE selection; last = final epoch, paper protocol "70 epochs" is
# agnostic about selection).  Set TRAJECTORY_CHECKPOINT=last to re-score.
ckpt="${TRAJECTORY_CHECKPOINT:-best}"
# Round-4 literal alignment: which run subdirectory holds the forecast
# checkpoint (run_engagement.sh literal writes run/literal).  Defaults to
# "forecast" so existing callers are untouched; the literal variant scores
# the same data/engagement_forecast.npz but a differently-trained model.
fc_dir="${TRAJECTORY_FORECAST_DIR:-forecast}"
export PYTHONPATH="$module/src:$xuance_root"

# Artifacts go to paper_targets/ for best and paper_targets_<ckpt>/ otherwise,
# so re-scoring with last.pt never clobbers or mixes with the best.pt report.
# A non-default forecast directory (literal) gets its own prefix for the same
# reason: the literal report must not overwrite the engineering-baseline one.
pt_dir="$root/paper_targets"
[[ "$ckpt" != "best" ]] && pt_dir="$root/paper_targets_${ckpt}"
[[ "$fc_dir" != "forecast" ]] && pt_dir="$root/paper_targets_${fc_dir}"
[[ "$fc_dir" != "forecast" && "$ckpt" != "best" ]] && pt_dir="$root/paper_targets_${fc_dir}_${ckpt}"
mkdir -p "$pt_dir" "$root/logs"
stamp="$(date +%Y%m%d_%H%M%S)"
log="$root/logs/paper_targets_${stamp}.log"
exec > >(tee -a "$log") 2>&1
echo "paper-targets run started: $(date -Is)"
echo "checkpoint:  $ckpt.pt (TRAJECTORY_CHECKPOINT)"

# ---- IV-A: 1000 independent Monte-Carlo trials per label --------------------
# The paper reports accuracy with the predictor re-seeded for every trial, so a
# single deterministic pass is not a substitute.
if [[ -f "$root/run/intention/$ckpt.pt" ]]; then
  echo "== IV-A nine-class Monte-Carlo ($trials trials/label, $ckpt.pt)"
  "$PYTHON" "$scripts/monte_carlo_intention.py" --data "$root/data/engagement_intention.npz" \
    --checkpoint "$root/run/intention/$ckpt.pt" --trials "$trials" \
    --device "$device" --output "$pt_dir/intention_mc.json"
else
  echo "== skip IV-A nine-class: $root/run/intention/$ckpt.pt missing"
fi
if [[ -f "$root/run/overload/$ckpt.pt" ]]; then
  echo "== IV-A eleven-level Monte-Carlo ($trials trials/label, $ckpt.pt)"
  "$PYTHON" "$scripts/monte_carlo_intention.py" --data "$root/data/engagement_overload.npz" \
    --checkpoint "$root/run/overload/$ckpt.pt" --trials "$trials" \
    --device "$device" --output "$pt_dir/overload_mc.json"
else
  echo "== skip IV-A eleven-level: $root/run/overload/$ckpt.pt missing"
fi

# ---- Fig. 14: the four composite maneuver scenarios (Table VI) --------------
if [[ -f "$root/run/$fc_dir/$ckpt.pt" ]]; then
  echo "== Fig. 14 composite scenarios ($ckpt.pt, run/$fc_dir)"
  "$PYTHON" "$scripts/reproduce_figure14.py" --checkpoint "$root/run/$fc_dir/$ckpt.pt" \
    --device "$device" --output "$pt_dir/figure14"
else
  echo "== skip Fig. 14: $root/run/$fc_dir/$ckpt.pt missing"
fi

# ---- IV-B held-out metrics + anchor comparison ------------------------------
forecast_metrics=""
if [[ "$ckpt" != "best" && -f "$root/run/$fc_dir/$ckpt.pt" ]]; then
  # Non-best checkpoints have no stored metrics; evaluate them here so the
  # report never mixes sources (e.g. last.pt scored against best.pt metrics).
  split="validation"
  [[ -f "$root/run/$fc_dir/test_metrics.json" ]] && split="test"
  "$PYTHON" "$entry" evaluate --data "$root/data/engagement_forecast.npz" \
    --checkpoint "$root/run/$fc_dir/$ckpt.pt" --split "$split" \
    --device "$device" --output "$pt_dir/forecast_${split}_metrics.json"
  forecast_metrics="$pt_dir/forecast_${split}_metrics.json"
elif [[ -f "$root/run/$fc_dir/validation_metrics.json" ]]; then
  forecast_metrics="$root/run/$fc_dir/validation_metrics.json"
elif [[ -f "$root/run/$fc_dir/test_metrics.json" ]]; then
  forecast_metrics="$root/run/$fc_dir/test_metrics.json"
fi
figure14=""
[[ -f "$pt_dir/figure14/figure14_summary.json" ]] && \
  figure14="$pt_dir/figure14/figure14_summary.json"

echo "== anchor comparison"
targets=(--output "$pt_dir/paper_targets_report.json")
[[ -f "$pt_dir/intention_mc.json" ]] && \
  targets+=(--intention "$pt_dir/intention_mc.json")
[[ -f "$pt_dir/overload_mc.json" ]] && \
  targets+=(--overload "$pt_dir/overload_mc.json")
[[ -n "$forecast_metrics" ]] && targets+=(--forecast "$forecast_metrics")
[[ -n "$figure14" ]] && targets+=(--figure14 "$figure14")
if [[ ${#targets[@]} -gt 2 ]]; then
  "$PYTHON" "$scripts/check_paper_targets.py" "${targets[@]}"
  echo "Report: $pt_dir/paper_targets_report.json"
else
  echo "No trained artefacts yet; rerun this script after run_engagement.sh."
fi
echo "paper-targets run finished: $(date -Is); log saved to $log"
