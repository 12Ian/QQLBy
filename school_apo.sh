#!/bin/bash
# Apollonius-based allocation, three seeds, on the school box. AutoDL cannot host this: its
# cgroup caps memory at 2 GB and every six-agent multi-target run was OOM-killed at startup.
# The criterion was verified to actually bite before committing GPU: across 750 real steps it
# assigned pursuers differently from the distance rule on 516 of them (68.8%).
exec 9>/tmp/school_apo.lock
flock -n 9 || { echo "[SAPO] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/school_apo_summary.txt
RUN=xuance-master/examples/run_experiment.py
EMT=xuance-master/examples/evaluate_mt.py
REW="--building-mode medium --closure-weight 8 --w-pos 0.6 --w-gap 0.8 --target-max-speed 10 --target-min-speed 8"
MT="--env uav_pursuit_apollonius_multitarget_3d $REW"
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
guard () { local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 8 ] && return 1; return 0; }
job () {
  set -- $1
  local n=$1; local sd=$2
  guard || { echo "[SAPO] ABORT $n low disk" >> $SUM; return; }
  if [ -z "$(mp $n)" ]; then
    echo "[SAPO] TRAIN $n seed=$sd $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --algo maddpg --name "$n" --seed $sd --num-targets 2 --num-agents 6 \
      --apollonius-alloc $MT --steps 5000000 --parallels 8 --eval-interval 250000 \
      --test-episode 20 > p3_results/train_$n.log 2>&1
    echo "[SAPO] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[SAPO] $n NO_MODEL" >> $SUM; return; }
  for K in 15 30; do
    [ -f p3_results/eval_${n}_k${K}.json ] || python $EMT --model-path "$FM" \
       --num-agents 6 --num-targets 2 --building-mode medium --target-max-speed 10 \
       --apollonius-alloc --k $K --seed 7 --out p3_results/eval_${n}_k${K}.json > /dev/null 2>&1
    echo "[SAPO] $n k$K $(grep -oE '"[a-z_]*(success|capture)[a-z_]*": [0-9.]+' p3_results/eval_${n}_k${K}.json 2>/dev/null | head -2 | tr '\n' ' ')" >> $SUM
  done
}
export -f job mp guard; export RUN EMT SUM MT
echo "[SAPO] ===== START $(date) =====" >> $SUM
printf '%s\n' "apo_M2_s1 1" "apo_M2_s2 2" "apo_M2_s3 3" | xargs -P 3 -I{} bash -c 'job "{}"'
echo "[SAPO] ===== DONE $(date) =====" >> $SUM
