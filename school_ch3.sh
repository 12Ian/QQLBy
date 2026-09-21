#!/bin/bash
# Everything Chapter 3 still needs to stand on its own, under the wall-free setting.
#
# The headline numbers already exist and are verified: 100% at ratios 0.818 and 0.909 in sparse
# airspace with captures a median 108 m from any face, and 93.3% in built-up density with
# captures 11 m from buildings and half the exits closed by structure. What is missing is not
# more of that -- it is the supporting evidence a chapter needs around it: baselines to compare
# against, ablations showing which components carry the result, and enough seeds on the headline
# configurations to quote a confidence interval.
#
# Everything here runs at the settings the headline numbers use: centre-pulled evader, surround
# spawn at 200 m, barrier filter at margin 2.
exec 9>/tmp/school_ch3.lock
flock -n 9 || { echo "[CH3] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/ch3_summary.txt
RUN=xuance-master/examples/run_experiment.py
EV3=xuance-master/examples/evaluate_3d.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
COMMON="--target-min-speed 8 --evader-center-pull 1.0 --surround-spawn --spawn-radius 200"
echo "[CH3] ===== START $(date) =====" >> $SUM

# --- baselines: no training needed, these are scripted policies ---
for spec in "greedy 3 9 medium" "random 3 9 medium" "greedy 3 10 medium" "random 3 10 medium" \
            "greedy 3 9 complex" "random 3 9 complex"; do
  set -- $spec; pol=$1; nag=$2; v=$3; dens=$4
  OUT=p3_results/base_${pol}_N${nag}_v${v}_${dens}.json
  [ -f "$OUT" ] || python $EV3 --policy $pol --num-agents $nag --building-mode $dens \
    --target-max-speed $v $COMMON --cbf-safety --cbf-margin 2 --k 30 --seed 7 \
    --out "$OUT" > /dev/null 2>&1
  echo "[CH3] BASELINE $pol N=$nag v=$v $dens $(grep -oE '"(success_rate|collision_rate)": [0-9.]+' $OUT 2>/dev/null | tr '\n' ' ')" >> $SUM
done

# --- ablations and extra seeds, all at ratio 0.909 where the headline sits ---
job () {
  set -- $1
  local n=$1 nag=$2 v=$3 dens=$4 sd=$5 extra=$6
  if [ -z "$(mp $n)" ]; then
    echo "[CH3] TRAIN $n ($extra) $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed $sd --num-agents $nag --building-mode $dens \
      --target-max-speed $v $COMMON --closure-weight 16 --pincer-weight 12 \
      --separation-weight 4 --separation-distance 20 --w-pos 0.6 --w-gap 0.8 \
      --cbf-safety --cbf-margin 2 $extra \
      --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
    echo "[CH3] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[CH3] $n NO_MODEL" >> $SUM; return; }
  local OUT=p3_results/eval_${n}_k30.json
  [ -f "$OUT" ] || python $EV3 --policy model --model-path "$FM" --num-agents $nag \
      --building-mode $dens --target-max-speed $v $COMMON --cbf-safety --cbf-margin 2 \
      --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  echo "[CH3] $n $(grep -oE '"(success_rate|collision_rate|timeout_rate)": [0-9.]+' $OUT 2>/dev/null | tr '\n' ' ')" >> $SUM
}
export -f job mp; export RUN EV3 SUM COMMON
# ablations: which component carries the result
printf '%s\n' \
  "abl_noclosure 3 10 medium 1 --reward-disable=r_closure" \
  "abl_nopincer 3 10 medium 1 --reward-disable=r_pincer" \
  "abl_nocollapse 3 10 medium 1 --no-guide-collapse" \
  "abl_nogap 3 10 medium 1 --reward-disable=r_gap" \
  | xargs -P 4 -I{} bash -c 'job "{}"'
# extra seeds on the two headline configurations
printf '%s\n' \
  "hd_N3_v10_med_s6 3 10 medium 6 " \
  "hd_N3_v10_med_s7 3 10 medium 7 " \
  "hd_N3_v9_cx_s2 3 9 complex 2 " \
  "hd_N3_v9_cx_s3 3 9 complex 3 " \
  | xargs -P 4 -I{} bash -c 'job "{}"'
echo "[CH3] ===== DONE $(date) =====" >> $SUM
