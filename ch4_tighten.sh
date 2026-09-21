#!/bin/bash
# Chapter 4's central claim (dynamic allocation ~2x static) did not reproduce: 62.2% vs 73.3%
# over three seeds each, z=1.13, p~0.26 -- indistinguishable, not a win for either side. At
# k=15 each point carries a binomial SE near 12pp, which is too coarse to settle a headline
# claim, so re-evaluate every model at k=30 (pure evaluation, no retraining) and additionally
# measure assignment churn, which is the mechanism that would explain a dynamic-side deficit:
# re-optimising every step lets pursuers flip between targets instead of committing to one.
exec 9>/tmp/ch4_tighten.lock
flock -n 9 || { echo "[T] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
EMT=xuance-master/examples/evaluate_mt.py
SUM=p3_results/ch4_tighten_summary.txt
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
echo "[T] ===== START $(date) =====" >> $SUM
for n in ch4x_M2_s1 ch4x_M2_s2 ch4x_M2_s3; do
  FM=$(mp $n); [ -z "$FM" ] && continue
  [ -f p3_results/eval_${n}_k30.json ] || python $EMT --model-path "$FM" --num-agents 6 --num-targets 2 \
     --building-mode medium --target-max-speed 10 --k 30 --seed 7 \
     --out p3_results/eval_${n}_k30.json > /dev/null 2>&1
  echo "[T] $n k30 $(grep -oE '"[a-z_]*(success|capture)[a-z_]*": [0-9.]+' p3_results/eval_${n}_k30.json 2>/dev/null | head -2 | tr '\n' ' ')" >> $SUM
done
for n in ch4x_M2static_s1 ch4x_M2static_s2 ch4x_M2static_s3; do
  FM=$(mp $n); [ -z "$FM" ] && continue
  [ -f p3_results/eval_${n}_k30.json ] || python $EMT --model-path "$FM" --num-agents 6 --num-targets 2 \
     --building-mode medium --target-max-speed 10 --no-dynamic-alloc --k 30 --seed 7 \
     --out p3_results/eval_${n}_k30.json > /dev/null 2>&1
  echo "[T] $n k30 $(grep -oE '"[a-z_]*(success|capture)[a-z_]*": [0-9.]+' p3_results/eval_${n}_k30.json 2>/dev/null | head -2 | tr '\n' ' ')" >> $SUM
done
echo "[T] ===== DONE $(date) =====" >> $SUM
