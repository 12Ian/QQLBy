#!/bin/bash
# FIX THE REPORT'S OWN TABLE - N=4 (currently EXCLUDED as a death-spiral outlier).
# The whole Ch3 N-sweep (run_strong_n3.sh) was trained with --closure-weight 8, which this
# campaign proved is badly under-weighted (cw8->cw16 gained +26..+37pp on every seed at the
# harder vmax12). This re-runs N=4 in the REPORT'S EXACT protocol -- vmax10, medium, 5M steps,
# parallels 8, one-sided spawn, eval seed 7 -- changing ONLY the closure weight (8->16) plus
# the 2nd-nearest pincer term that fixed N=4's timeout mode at vmax12. Eval at k=15 (directly
# comparable to the report table) AND k=30 (honest tighter CI). 2 seeds.
exec 9>/tmp/run_v10_n4.lock
flock -n 9 || { echo "[v4] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
SUM=p3_results/v10_n4_summary.txt
modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
echo "[v4] === START $(date) ===" | tee -a $SUM

train_v10 () {  # $1=name $2=N $3=seed
  local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 12 ] && { echo "[v4] ABORT low disk ${F}G"|tee -a $SUM; exit 1; }
  [ -n "$(modelpath $1)" ] && { echo "[v4] SKIP train $1"|tee -a $SUM; return; }
  echo "[v4] TRAIN $1 N=$2 seed=$3 vmax10 cw16+pin12 5M $(date)" | tee -a $SUM
  python $RUN --name "$1" --seed $3 --num-agents $2 --building-mode medium \
    --target-max-speed 10 --target-min-speed 8 --closure-weight 16 --pincer-weight 12 \
    --w-pos 0.6 --w-gap 0.8 --steps 5000000 --parallels 8 --eval-interval 250000 \
    --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[v4] $1 train_exit=$?" | tee -a $SUM
}
eval_v10 () {  # $1=name $2=N $3=k
  local OUT=p3_results/eval_${1}_k$3.json
  [ -f "$OUT" ] && { echo "[v4] SKIP eval $1 k=$3"|tee -a $SUM; return; }
  local FM; FM=$(modelpath $1); [ -z "$FM" ] && { echo "[v4] $1 NO_MODEL"|tee -a $SUM; return; }
  echo -n "[v4] $1 k=$3: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --num-agents $2 --building-mode medium \
    --target-max-speed 10 --target-min-speed 8 --k $3 --seed 7 \
    --out "$OUT" 2>/dev/null | grep RESULT | tee -a $SUM
}

for s in 1 2; do
  train_v10 v10_N4_s$s 4 $s
  eval_v10 v10_N4_s$s 4 30
  eval_v10 v10_N4_s$s 4 15
done
echo "[v4] === DONE $(date) ===" | tee -a $SUM
