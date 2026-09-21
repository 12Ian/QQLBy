#!/bin/bash
# Ch4 4.5 density generalization: eval existing ch4_M2 models across obstacle densities.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
EMT=xuance-master/examples/evaluate_mt.py
for name in ch4_M2_s1 ch4_M2_s2; do
  M=$(ls models/maddpg/apollonius_3d_${name}/*/final_train_model.pth | head -1)
  for D in empty open complex; do
    [ -f p3_results/eval_${name}_${D}.json ] && continue
    echo -n "$name @$D: "
    python $EMT --model-path "$M" --num-agents 6 --num-targets 2 --building-mode $D \
      --target-max-speed 10 --k 15 --seed 7 --out p3_results/eval_${name}_${D}.json 2>/dev/null | grep RESULT
  done
done
echo CH4DENS_DONE
