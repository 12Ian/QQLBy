#!/bin/bash
# Strong-evader validation: full 5M training of the method against the STRONG evader at
# parity (vmax 11) and slightly-faster (vmax 12), @medium N=6, to get the real headline
# success rate (sv_val was only 2.75M / stuck). Idempotent + flock.
exec 9>/tmp/run_se.lock
flock -n 9 || { echo "[se] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
SUM=p3_results/strongeval.txt
echo "[se] === START $(date) ===" | tee -a $SUM

train_eval () {  # $1=name $2=target-max-speed
  local M; M=$(ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1)
  if [ -z "$M" ]; then
    echo "[se] TRAIN $1 (vmax=$2) $(date)" | tee -a $SUM
    python $RUN --name $1 --num-agents 6 --building-mode medium --closure-weight 8 --w-pos 0.6 --w-gap 0.8 \
      --target-max-speed $2 --target-min-speed 8 --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$1.log 2>&1
    echo "[se] $1 train_exit=$?" | tee -a $SUM
    M=$(ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1)
  fi
  [ -z "$M" ] && { echo "[se] $1 NO_MODEL" | tee -a $SUM; return; }
  echo -n "[se] $1 eval: " | tee -a $SUM
  python $EVAL --policy model --model-path "$M" --building-mode medium --num-agents 6 \
    --target-max-speed $2 --target-min-speed 8 --k 20 --seed 7 --out p3_results/eval_$1.json 2>/dev/null | grep RESULT | tee -a $SUM
}

train_eval sv_p9  9     # slightly slower (lambda ~= 0.82) - active-evasion sweet spot?
train_eval sv_p10 10    # near parity (lambda ~= 0.91)
train_eval sv_p11 11    # parity  (lambda ~= 1.0)
train_eval sv_p12 12    # slightly faster (lambda ~= 1.09)

echo "[se] === DONE $(date) ===" | tee -a $SUM
