#!/bin/bash
# Does adding pursuers help against a faster evader, once crashes stop deciding the episode?
#
# Watching the vmax-12 clip, three pursuers visibly struggle: 163 steps to close, against 70 at
# vmax 11. The obvious response is to send more. Measured, more is worse -- 32.2% at N=3 falls
# to 6.7% at N=4 and 3.3% at N=5 -- but the failures are not misses. Collisions go 27.8% ->
# 63.3% -> 80.0%, and any single pursuer hitting a building fails the whole episode, so a larger
# team is mostly buying more chances to trip that rule.
#
# The barrier filter removes those crashes rather than rescoring them (102.9 hits/500 steps to 0
# offline; on a trained team at vmax 10, N=6 collisions fell 73.3% -> 10% and success rose
# 6.7% -> 36.7% with the filter in the training loop). So this is the honest test of the
# intuition: N = 3, 4, 5 against the faster evader, all with the filter in the loop and all with
# the surround spawn that a speed ratio above 1 requires.
exec 9>/tmp/school_r6.lock
flock -n 9 || { echo "[R6] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/r6_summary.txt
RUN=xuance-master/examples/run_experiment.py
EV3=xuance-master/examples/evaluate_3d.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
sr () { grep -oE '"(success_rate|collision_rate|timeout_rate)": [0-9.]+' "$1" 2>/dev/null | tr '\n' ' '; }
echo "[R6] ===== START $(date) =====" >> $SUM
find p3_results -name "train_ch4fix_*.log" -size +50M -delete 2>/dev/null
rm -rf logs/maddpg/apollonius_3d_r4* logs/maddpg/apollonius_3d_ch4fix_* 2>/dev/null
echo "[R6] free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)G" >> $SUM

job () {
  set -- $1
  local n=$1; local nag=$2
  if [ -z "$(mp $n)" ]; then
    echo "[R6] TRAIN $n N=$nag (vmax12 + CBF + surround) $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed 1 --num-agents $nag --building-mode medium \
      --target-max-speed 12 --target-min-speed 8 --closure-weight 16 --pincer-weight 12 \
      --separation-weight 4 --separation-distance 20 --w-pos 0.6 --w-gap 0.8 \
      --surround-spawn --spawn-radius 200 --cbf-safety \
      --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
    echo "[R6] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[R6] $n NO_MODEL" >> $SUM; return; }
  OUT=p3_results/eval_${n}_k30.json
  [ -f "$OUT" ] || python $EV3 --policy model --model-path "$FM" --num-agents $nag \
      --building-mode medium --target-max-speed 12 --target-min-speed 8 --surround-spawn \
      --cbf-safety --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  echo "[R6] $n N=$nag $(sr $OUT)" >> $SUM
}
export -f job mp sr; export RUN EV3 SUM
printf '%s\n' "v12cbf_N3 3" "v12cbf_N4 4" "v12cbf_N5 5" | xargs -P 3 -I{} bash -c 'job "{}"'
echo "[R6] ===== DONE $(date) =====" >> $SUM
