#!/bin/bash
# Detour criterion, scoped to what can finish before the handover.
#
# The unpruned criterion needs ~110 hours per 5M-step run; three seeds had reached 2/20 epochs
# after 23 hours and were stopped. Pruning on the lower bound -- straight-line distance never
# exceeds the detour distance, so a pursuer already losing the straight-line race cannot win the
# detour one -- cuts the exact computations from 76.3% of probe points to 6.0%, but 30 hours per
# seed is still ten times the euclidean cost and does not fit before tomorrow.
#
# Two things that do fit, and answer different questions:
#
#   (a) Evaluation-only. Take the euclidean-trained models and score them under the detour
#       criterion. The criterion changes what counts as an open escape direction, so this asks
#       whether the existing policies hold up when the escape set is measured more honestly --
#       no retraining, minutes not hours.
#
#   (b) A short 1M-step training run. Enough to see whether the training curve tracks or
#       diverges from the euclidean baseline at equal step count, which is the question a full
#       run would answer. Reported as a partial result, not as a converged one.
exec 9>/tmp/school_r22.lock
flock -n 9 || { echo "[R22] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/r22_summary.txt
RUN=xuance-master/examples/run_experiment.py
EV3=xuance-master/examples/evaluate_3d.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
CM="--target-min-speed 8 --evader-center-pull 1.0 --surround-spawn --spawn-radius 200 --cbf-safety --cbf-margin 2"
echo "[R22] ===== START $(date) =====" >> $SUM
rm -rf models/maddpg/apollonius_3d_dtr_s* logs/maddpg/apollonius_3d_dtr_s* 2>/dev/null

# (a) existing euclidean-trained models, re-scored under each criterion
for n in urb2_N3_v9_cx cx2_full_s4 cx2_full_s5; do
  FM=$(mp $n); [ -z "$FM" ] && continue
  for crit in euclidean visibility detour; do
    OUT=p3_results/crit_${n}_${crit}.json
    [ -f "$OUT" ] || timeout 3600 python $EV3 --policy model --model-path "$FM" --num-agents 3 \
      --building-mode complex --target-max-speed 9 $CM --criterion $crit \
      --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
    echo "[R22] RESCORE $n $crit $(grep -oE '"(success_rate|collision_rate)": [0-9.]+' $OUT 2>/dev/null | tr '\n' ' ')" >> $SUM
  done
done

# (b) short training at equal step count, detour against euclidean
job () {
  set -- $1
  local n=$1 sd=$2 crit=$3
  local t0=$(date +%s)
  if [ -z "$(mp $n)" ]; then
    echo "[R22] TRAIN $n crit=$crit $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed $sd --num-agents 3 --building-mode complex \
      --target-max-speed 9 $CM --criterion $crit --closure-weight 16 --pincer-weight 12 \
      --separation-weight 4 --separation-distance 20 --w-pos 0.6 --w-gap 0.8 \
      --steps 1000000 --parallels 8 --eval-interval 100000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
    echo "[R22] $n exit=$? elapsed=$(( ($(date +%s)-t0)/60 ))min" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[R22] $n NO_MODEL" >> $SUM; return; }
  local OUT=p3_results/eval_${n}_k30.json
  [ -f "$OUT" ] || python $EV3 --policy model --model-path "$FM" --num-agents 3 \
      --building-mode complex --target-max-speed 9 $CM --criterion $crit \
      --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  echo "[R22] $n $(grep -oE '"(success_rate|collision_rate|timeout_rate)": [0-9.]+' $OUT 2>/dev/null | tr '\n' ' ')" >> $SUM
}
export -f job mp; export RUN EV3 SUM CM
printf '%s\n' "sh_dtr_s1 1 detour" "sh_dtr_s2 2 detour" "sh_euc_s1 1 euclidean" "sh_euc_s2 2 euclidean" \
  | xargs -P 4 -I{} bash -c 'job "{}"'
echo "[R22] ===== DONE $(date) =====" >> $SUM
