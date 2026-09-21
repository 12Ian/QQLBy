#!/bin/bash
# P2 experiment campaign - Batch 1: 3.3 GAT ablation + 3.7 criterion ablation.
# Serial (single GPU). Waits for fix_dr to finish, then trains+evals each config
# at medium density. Results -> p2_results/eval_<name>.json. Run with nohup.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
mkdir -p p2_results
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
REWARD="--target-max-speed 6 --closure-weight 8 --w-pos 0.6 --w-gap 0.8"
STEPS="--steps 4000000 --parallels 16 --eval-interval 200000 --test-episode 20"
SUMMARY=p2_results/batch1_summary.txt

# wait for the current fix_dr run to finish (avoid GPU contention)
echo "[campaign] waiting for fix_dr to finish... $(date)"
while pgrep -f 'name fix_dr' >/dev/null 2>&1; do sleep 60; done
echo "[campaign] === P2 BATCH 1 START $(date) ===" | tee -a $SUMMARY

train () {  # $1=name  $2=gat/criterion flags
  echo "[campaign] TRAIN $1 ($2) $(date)" | tee -a $SUMMARY
  python $RUN --name "$1" --building-mode medium $REWARD $STEPS $2 > p2_results/train_$1.log 2>&1
}
evalf () {  # $1=name  $2=gat flags (must match training architecture)
  local FM
  FM=$(ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1)
  echo "[campaign] EVAL $1 (final model, medium) $(date)" | tee -a $SUMMARY
  echo -n "$1: " >> $SUMMARY
  python $EVAL --policy model --model-path "$FM" $2 --building-mode medium \
      --target-max-speed 6 --k 40 --out p2_results/eval_$1.json 2>/dev/null | grep RESULT | tee -a $SUMMARY
}

# ---- 3.3 GAT ablation (medium). gat_full also serves as 3.7's euclidean run. ----
train gat_none  "";                                            evalf gat_none  ""
train gat_t     "--use-graph-module";                          evalf gat_t     "--use-graph-module"
train gat_full  "--use-graph-module --use-obstacle-gat";       evalf gat_full  "--use-graph-module --use-obstacle-gat"

# ---- 3.7 criterion ablation (medium, full GAT). crit_euc == gat_full above. ----
train crit_vis  "--use-graph-module --use-obstacle-gat --criterion visibility"
evalf crit_vis  "--use-graph-module --use-obstacle-gat"

echo "[campaign] === P2 BATCH 1 DONE $(date) ===" | tee -a $SUMMARY
