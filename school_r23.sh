#!/bin/bash
# Does the multi-target environment still work when it degenerates to the single-target one?
#
# Chapter 4 has returned 0.0 under the strict criterion through five different fixes, and I read
# that as the task being kinematically out of reach: pursuers starting ~220 m out with a 1 m/s
# net closing speed. Chapter 3 disproves it -- three pursuers, one target, ratio 0.909, spawn
# radius 200, and it scores 100%. Six pursuers against two targets is three per target: the same
# ratio, the same speeds, the same distance.
#
# So the failure is not the task. The test that separates a broken environment from a hard one is
# the degenerate case, and I never ran it: the multi-target environment with num_targets=1 and
# num_agents=3, every other parameter copied from the Chapter 3 configuration that reaches 100%.
# It should reproduce that number. If it does, the fault is in something only multi-target code
# touches -- allocation, the per-target escape field, the guide points. If it does not, the
# environment is broken even where it reduces to a problem already known to be solvable, and
# every earlier result was measuring that.
exec 9>/tmp/school_r23.lock
flock -n 9 || { echo "[R23] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/r23_summary.txt
RUN=xuance-master/examples/run_experiment.py
EVM=xuance-master/examples/evaluate_mt.py
EV3=xuance-master/examples/evaluate_3d.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
echo "[R23] ===== START $(date) =====" >> $SUM

job () {
  set -- $1
  local n=$1 ntg=$2 nag=$3 sd=$4
  if [ -z "$(mp $n)" ]; then
    echo "[R23] TRAIN $n targets=$ntg agents=$nag seed=$sd $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed $sd --num-targets $ntg --num-agents $nag \
      --env uav_pursuit_apollonius_multitarget_3d --building-mode medium \
      --target-max-speed 10 --target-min-speed 8 --evader-center-pull 1.0 \
      --closure-weight 16 --pincer-weight 12 --separation-weight 4 --separation-distance 20 \
      --w-pos 0.6 --w-gap 0.8 --surround-spawn --spawn-radius 200 --strict-capture \
      --cbf-safety --cbf-margin 2 \
      --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
    echo "[R23] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[R23] $n NO_MODEL" >> $SUM; return; }
  local OUT=p3_results/eval_${n}_k30.json
  [ -f "$OUT" ] || python $EVM --model-path "$FM" --num-agents $nag --num-targets $ntg \
      --building-mode medium --target-max-speed 10 --evader-center-pull 1.0 \
      --surround-spawn --spawn-radius 200 --strict-capture --cbf-safety --cbf-margin 2 \
      --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  echo "[R23] $n $(grep -oE '"(success_rate|per_target_capture_rate|collision_rate|timeout_rate)": [0-9.]+' $OUT 2>/dev/null | tr '\n' ' ')" >> $SUM
}
export -f job mp; export RUN EVM SUM
# the degenerate case first, then one step up, to locate where it breaks
printf '%s\n' "deg_M1_N3_s1 1 3 1" "deg_M1_N3_s2 1 3 2" "deg_M1_N6_s1 1 6 1" "deg_M2_N6_s1 2 6 1" \
  | xargs -P 4 -I{} bash -c 'job "{}"'
echo "[R23] ===== DONE $(date) =====" >> $SUM
