#!/bin/bash
# Does the detour criterion actually help, or just cost time?
#
# The escape criterion previously had two modes, and both misprice an occluded pursuer. Euclidean
# ignores buildings entirely, letting a pursuer "block" a direction through a wall. Visibility
# voids its claim outright, so a pursuer one corner away counts for nothing and the direction
# goes unassigned. detour_distance prices the real cost: routes around footprint corners and
# feeds the resulting arrival time into the same comparison. On the real complex layout the mean
# escape solid angle lands at 4.09 against 1.74 euclidean and 5.63 visibility -- between the two,
# which is where a correct model belongs.
#
# A better-calibrated criterion is not automatically a better policy, and this one costs about
# 5x per evaluation of the field, so training may simply get less done in the same budget. Three
# seeds against the five euclidean seeds already measured (82-89%).
#
# Timing is logged after the first epoch so the run can be abandoned early if the slowdown makes
# 5M steps impractical rather than discovering that in ten hours.
exec 9>/tmp/school_r21.lock
flock -n 9 || { echo "[R21] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/r21_summary.txt
RUN=xuance-master/examples/run_experiment.py
EV3=xuance-master/examples/evaluate_3d.py
PCG=xuance-master/examples/probe_capture_geometry.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
CM="--target-min-speed 8 --evader-center-pull 1.0 --surround-spawn --spawn-radius 200 --cbf-safety --cbf-margin 2"
echo "[R21] ===== START $(date) =====" >> $SUM

job () {
  set -- $1
  local n=$1 sd=$2 crit=$3
  local t0=$(date +%s)
  if [ -z "$(mp $n)" ]; then
    echo "[R21] TRAIN $n seed=$sd criterion=$crit $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed $sd --num-agents 3 --building-mode complex \
      --target-max-speed 9 $CM --criterion $crit --closure-weight 16 --pincer-weight 12 \
      --separation-weight 4 --separation-distance 20 --w-pos 0.6 --w-gap 0.8 \
      --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1 &
    local pid=$!
    # report the pace once the first epoch lands, so a hopeless slowdown shows up early
    ( sleep 900
      e=$(grep -aoE "Epoch: [0-9]+/[0-9]+" p3_results/train_$n.log 2>/dev/null | tail -1)
      echo "[R21] PACE $n after 15min: ${e:-not started}" >> $SUM ) &
    wait $pid
    echo "[R21] $n exit=$? elapsed=$(( ($(date +%s)-t0)/60 ))min" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[R21] $n NO_MODEL" >> $SUM; return; }
  local OUT=p3_results/eval_${n}_k30.json
  [ -f "$OUT" ] || python $EV3 --policy model --model-path "$FM" --num-agents 3 \
      --building-mode complex --target-max-speed 9 $CM --criterion $crit \
      --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  echo "[R21] $n $(grep -oE '"(success_rate|collision_rate|timeout_rate)": [0-9.]+' $OUT 2>/dev/null | tr '\n' ' ')" >> $SUM
  local G=p3_results/capgeo_r21_${n}.json
  [ -f "$G" ] || python $PCG --model-path "$FM" --num-agents 3 --building-mode complex \
      --target-max-speed 9 $CM --criterion $crit --k 40 --seed 7 --out "$G" > /dev/null 2>&1
}
export -f job mp; export RUN EV3 PCG SUM CM
printf '%s\n' "dtr_s1 1 detour" "dtr_s2 2 detour" "dtr_s3 3 detour" \
  | xargs -P 3 -I{} bash -c 'job "{}"'
echo "[R21] ===== DONE $(date) =====" >> $SUM
