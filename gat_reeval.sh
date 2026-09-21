#!/bin/bash
# Re-eval the GAT-ablation models WITH the matching GAT flags (the campaign's eval omitted
# them -> arch mismatch -> silent fail). Models are fine; this is eval-only.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
E3=xuance-master/examples/evaluate_3d.py
COMMON="--policy model --num-agents 3 --building-mode medium --target-max-speed 10 --target-min-speed 8 --k 15 --seed 7"
reval () {  # $1=name  $2=flags
  M=$(ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth | head -1)
  echo -n "$1: "
  python $E3 --model-path "$M" $COMMON $2 --out p3_results/eval_$1_medium.json 2>/dev/null | grep RESULT
}
reval s3_gat_t_s1    "--use-graph-module"
reval s3_gat_t_s2    "--use-graph-module"
reval s3_gat_full_s1 "--use-obstacle-gat --use-graph-module"
reval s3_gat_full_s2 "--use-obstacle-gat --use-graph-module"
echo GAT_REEVAL_DONE
