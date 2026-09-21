#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
E=xuance-master/examples/evaluate_3d.py
for pol in greedy random; do
  python $E --policy $pol --num-agents 3 --building-mode medium \
    --target-max-speed 10 --target-min-speed 8 --k 15 --seed 7 \
    --out p3_results/eval_base_${pol}_medium.json 2>/dev/null | grep RESULT
done
echo GR_OK
