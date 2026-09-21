#!/bin/bash
# Run the churn diagnostic on all six Chapter-4 M=2 models (three dynamic, three static).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/churn_summary.txt
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
: > $SUM
for n in ch4x_M2_s1 ch4x_M2_s2 ch4x_M2_s3; do
  FM=$(mp $n); [ -z "$FM" ] && continue
  echo -n "$n  " >> $SUM
  python xuance-master/examples/measure_alloc_churn.py --model-path "$FM" \
    --num-agents 6 --num-targets 2 --k 30 --seed 7 \
    --out p3_results/churn_${n}.json 2>/dev/null | tail -1 >> $SUM
done
for n in ch4x_M2static_s1 ch4x_M2static_s2 ch4x_M2static_s3; do
  FM=$(mp $n); [ -z "$FM" ] && continue
  echo -n "$n  " >> $SUM
  python xuance-master/examples/measure_alloc_churn.py --model-path "$FM" \
    --num-agents 6 --num-targets 2 --no-dynamic-alloc --k 30 --seed 7 \
    --out p3_results/churn_${n}.json 2>/dev/null | tail -1 >> $SUM
done
echo "CHURN_DONE" >> $SUM
