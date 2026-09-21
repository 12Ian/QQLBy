#!/bin/bash
# Warm-start the strict criterion from a policy that already encircles.
#
# Training strict from scratch has now returned 0.0 four times, and the in-training win rate is
# still 0.0 at 1.5M steps on the run with the repaired reward. The reason is visible in how the
# loose criterion ends an episode: it fires the moment the escape angle collapses, which is
# exactly when the closing phase would begin. A policy trained that way never observes the state
# it is now being asked to act in -- encircled at ~155 m with the guide points already collapsed
# to ~20 m. From scratch under the strict rule the terminal bonus is never reached either, so
# there is no signal to bootstrap from; it is a pure exploration problem, not a reward-shape one.
#
# r4surr_s1 encircles at 96.7%. Continuing it under the strict rule keeps that competence and
# puts the episodes where the closing behaviour has to be learned. This is the same warm-start
# that unlocked speed ratios above 1 in the single-target chapter.
exec 9>/tmp/school_r8.lock
flock -n 9 || { echo "[R8] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/r8_summary.txt
RUN=xuance-master/examples/run_experiment.py
EVM=xuance-master/examples/evaluate_mt.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
sr () { grep -oE '"(success_rate|per_target_capture_rate)": [0-9.]+' "$1" 2>/dev/null | tr '\n' ' '; }
cl () { grep -oE '"median": [0-9.]+' "$1" 2>/dev/null | head -1; }
BASE=$(mp r4surr_s1)
echo "[R8] ===== START $(date) base=$BASE =====" >> $SUM
[ -z "$BASE" ] && { echo "[R8] no base model" >> $SUM; exit 1; }

job () {
  set -- $1
  local n=$1; local sd=$2
  if [ -z "$(mp $n)" ]; then
    echo "[R8] CONT $n seed=$sd from loose-trained encircler $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed $sd --num-targets 2 --num-agents 6 \
      --env uav_pursuit_apollonius_multitarget_3d --building-mode medium \
      --closure-weight 16 --pincer-weight 12 --w-pos 0.6 --w-gap 0.8 \
      --target-max-speed 10 --target-min-speed 8 \
      --surround-spawn --spawn-radius 200 --strict-capture \
      --load-from "$BASE" --policy-only-load --start-noise 0.12 \
      --steps 3000000 --parallels 4 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
    echo "[R8] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[R8] $n NO_MODEL" >> $SUM; return; }
  OUT=p3_results/eval_${n}_k30.json
  [ -f "$OUT" ] || python $EVM --model-path "$FM" --num-agents 6 --num-targets 2 \
      --building-mode medium --target-max-speed 10 --surround-spawn --strict-capture \
      --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  echo "[R8] $n CONT-STRICT $(sr $OUT) closest_$(cl $OUT)" >> $SUM
}
export -f job mp sr cl; export RUN EVM SUM BASE
printf '%s\n' "r8cont_s1 1" "r8cont_s2 2" | xargs -P 2 -I{} bash -c 'job "{}"'
echo "[R8] ===== DONE $(date) =====" >> $SUM
