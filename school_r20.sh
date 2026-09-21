#!/bin/bash
# The decisive ablation: strip the Apollonius machinery entirely.
#
# Replicated at three seeds, every single-component ablation is now null -- guide collapse
# 83.3% (p=0.40), escape-angle term 90.0% (p=0.64), closure, pincer and the barrier filter all
# indistinguishable from the full 87.8%. Removing one component at a time can hide a real
# contribution when the remaining components substitute for it, so the question that matters is
# whether the geometry contributes anything at all.
#
# This removes the whole guidance stack at once -- guide-point following, escape-angle reward,
# closure and pincer bonuses -- leaving progress toward the target plus the safety terms. That
# is a learned pursuer with obstacle avoidance and no Apollonius reasoning. Greedy pursuit gets
# 20.0% here and fails on an 80-90% collision rate, so if this arm lands near 87.8% then the
# advantage over greedy is obstacle avoidance and the geometry is decorative; if it drops toward
# greedy, the guidance is doing the work after all.
#
# A reviewer runs this ablation whether or not we do.
exec 8>/tmp/school_r19.lock
flock 8
exec 8>&-
exec 9>/tmp/school_r20.lock
flock -n 9 || { echo "[R20] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/r20_summary.txt
RUN=xuance-master/examples/run_experiment.py
EV3=xuance-master/examples/evaluate_3d.py
PCG=xuance-master/examples/probe_capture_geometry.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
CM="--target-min-speed 8 --evader-center-pull 1.0 --surround-spawn --spawn-radius 200 --cbf-safety --cbf-margin 2"
echo "[R20] ===== START $(date) =====" >> $SUM

job () {
  set -- $1
  local n=$1 sd=$2 extra=$3
  if [ -z "$(mp $n)" ]; then
    echo "[R20] TRAIN $n seed=$sd $extra $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed $sd --num-agents 3 --building-mode complex \
      --target-max-speed 9 $CM --closure-weight 16 --pincer-weight 12 \
      --separation-weight 4 --separation-distance 20 --w-pos 0.6 --w-gap 0.8 $extra \
      --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
    echo "[R20] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[R20] $n NO_MODEL" >> $SUM; return; }
  local OUT=p3_results/eval_${n}_k30.json
  [ -f "$OUT" ] || python $EV3 --policy model --model-path "$FM" --num-agents 3 \
      --building-mode complex --target-max-speed 9 $CM --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  echo "[R20] $n $(grep -oE '"(success_rate|collision_rate)": [0-9.]+' $OUT 2>/dev/null | tr '\n' ' ')" >> $SUM
  # if the geometry is inert, the herding signature should vanish too
  local G=p3_results/capgeo_r20_${n}.json
  [ -f "$G" ] || python $PCG --model-path "$FM" --num-agents 3 --building-mode complex \
      --target-max-speed 9 $CM --k 40 --seed 7 --out "$G" > /dev/null 2>&1
  echo "[R20] $n GEO $(grep -oE '"dirs_blocked_by_structure": \{"median": [0-9.]+' $G 2>/dev/null | grep -oE '[0-9.]+$') bld=$(grep -oE '"building_dist": \{"min": [0-9.]+, "median": [0-9.]+' $G 2>/dev/null | grep -oE '[0-9.]+$')" >> $SUM
}
export -f job mp; export RUN EV3 PCG SUM CM
printf '%s\n' \
  "nogeo_s1 1 --reward-disable=r_pos,r_gap,r_closure,r_pincer" \
  "nogeo_s2 2 --reward-disable=r_pos,r_gap,r_closure,r_pincer" \
  "nogeo_s3 3 --reward-disable=r_pos,r_gap,r_closure,r_pincer" \
  "cx2_full_s5 5 " \
  | xargs -P 4 -I{} bash -c 'job "{}"'
echo "[R20] ===== DONE $(date) =====" >> $SUM
