#!/bin/bash
# Arm D: Apollonius assignment computed ONCE at reset, then frozen.
#
# The three arms measured so far line up as complexity up, performance down:
#   static (zero re-assignment)   72.2%
#   distance, re-optimised        65.6%
#   Apollonius, re-optimised      56.7%
# The Apollonius criterion demonstrably bites (it changes 68.8% of assignment decisions) but
# re-optimising every step hurts, because the optimal assignment in this setting is nearly
# constant. So keep the part that should help -- a capture-ability-based INITIAL pairing, which
# is better founded than nearest-distance -- and drop the part that hurts. No code change is
# needed: with dynamic_alloc off the env assigns once in reset() and only backfills when a
# target dies, and reset() already routes through the Apollonius cost.
exec 9>/tmp/school_apofix.lock
flock -n 9 || { echo "[AF] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/apofix_summary.txt
RUN=xuance-master/examples/run_experiment.py
EMT=xuance-master/examples/evaluate_mt.py
REW="--building-mode medium --closure-weight 8 --w-pos 0.6 --w-gap 0.8 --target-max-speed 10 --target-min-speed 8"
MT="--env uav_pursuit_apollonius_multitarget_3d $REW"
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
guard () { local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 6 ] && return 1; return 0; }
job () {
  set -- $1
  local n=$1; local sd=$2
  guard || { echo "[AF] ABORT $n low disk" >> $SUM; return; }
  if [ -z "$(mp $n)" ]; then
    echo "[AF] TRAIN $n seed=$sd (apollonius-once, frozen) $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --algo maddpg --name "$n" --seed $sd --num-targets 2 --num-agents 6 \
      --apollonius-alloc --no-dynamic-alloc $MT --steps 5000000 --parallels 8 \
      --eval-interval 250000 --test-episode 20 > p3_results/train_$n.log 2>&1
    echo "[AF] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[AF] $n NO_MODEL" >> $SUM; return; }
  for K in 15 30; do
    [ -f p3_results/eval_${n}_k${K}.json ] || python $EMT --model-path "$FM" \
       --num-agents 6 --num-targets 2 --building-mode medium --target-max-speed 10 \
       --apollonius-alloc --no-dynamic-alloc --k $K --seed 7 \
       --out p3_results/eval_${n}_k${K}.json > /dev/null 2>&1
    echo "[AF] $n k$K $(grep -oE '"[a-z_]*(success|capture)[a-z_]*": [0-9.]+' p3_results/eval_${n}_k${K}.json 2>/dev/null | head -2 | tr '\n' ' ')" >> $SUM
  done
}
export -f job mp guard; export RUN EMT SUM MT
echo "[AF] ===== START $(date) =====" >> $SUM
printf '%s\n' "apofix_M2_s1 1" "apofix_M2_s2 2" "apofix_M2_s3 3" | xargs -P 3 -I{} bash -c 'job "{}"'
echo "[AF] ===== DONE $(date) =====" >> $SUM
