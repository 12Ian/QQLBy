#!/bin/bash
# N=3 BEST-OF-BOTH. Failure-mode split (k=30) shows two independent modes: collision (10-30%)
# and timeout/false-capture (0-47%). The 90% champion is the only run with BOTH near zero.
# cw16 alone: s1 53% (coll .20 / to .27). pincer alone at N=4 cut timeout .47->.23.
# cw16+pincer was NEVER tried at N=3 (the earlier N=3 pincer run used cw8 - closure too low,
# 47%). This runs the combo on BOTH seeds -> an honest 2-seed mean for the candidate recipe.
# Own lock; runs concurrently with the other campaigns (20 cores, CPU-bound training).
exec 9>/tmp/run_n3combo.lock
flock -n 9 || { echo "[n3c] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
SUM=p3_results/n3combo_summary.txt
modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
echo "[n3c] === START $(date) ===" | tee -a $SUM

evalk () {
  local OUT=p3_results/eval_${1}_k$3.json
  [ -f "$OUT" ] && { echo "[n3c] SKIP eval $1"|tee -a $SUM; return; }
  local FM; FM=$(modelpath $1); [ -z "$FM" ] && { echo "[n3c] $1 NO_MODEL"|tee -a $SUM; return; }
  echo -n "[n3c] $1 k=$3: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --num-agents $2 --building-mode medium \
    --target-max-speed 12 --target-min-speed 8 --surround-spawn --k $3 --seed 7 \
    --out "$OUT" 2>/dev/null | grep RESULT | tee -a $SUM
}
train3 () {  # $1=name $2=seed $3=warm
  local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 12 ] && { echo "[n3c] ABORT low disk ${F}G"|tee -a $SUM; exit 1; }
  [ -n "$(modelpath $1)" ] && { echo "[n3c] SKIP train $1"|tee -a $SUM; return; }
  local WARM; WARM=$(modelpath $3); [ -z "$WARM" ] && { echo "[n3c] $1 NO_WARM($3)"|tee -a $SUM; return; }
  echo "[n3c] TRAIN $1 seed=$2 warm=$3 cw16+pin12 $(date)" | tee -a $SUM
  python $RUN --name "$1" --seed $2 --num-agents 3 --building-mode medium --surround-spawn \
    --target-max-speed 12 --target-min-speed 8 --closure-weight 16 --pincer-weight 12 \
    --w-pos 0.6 --w-gap 0.8 --load-from "$WARM" --policy-only-load --start-noise 0.12 --end-noise 0.01 \
    --steps 3000000 --parallels 8 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[n3c] $1 train_exit=$?" | tee -a $SUM
}

train3 cw16pin_N3_s1 1 fs_N3_s1; evalk cw16pin_N3_s1 3 30   # can it fix the WEAK seed (53%)?
train3 cw16pin_N3_s2 2 fs_N3_s2; evalk cw16pin_N3_s2 3 30   # can it beat the 90% champion?
echo "[n3c] === DONE $(date) ===" | tee -a $SUM
