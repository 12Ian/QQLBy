#!/bin/bash
# EVADER-FASTER re-run (correct regime per client + plan premise): evader vmax 12 > pursuer
# vmax 11 (lambda = v_pursuer/v_evader = 11/12 ~= 0.92 < 1). This is the MEANINGFUL regime:
# a single slower pursuer can't catch a faster evader -> cooperative multi-UAV encirclement
# is required. Prior headline was evader-SLOWER (trivial). First & top priority: N-sweep at
# vmax 12 to find the optimal pursuer count & achievable success for a FASTER evader.
# Seed-outer loop -> 1-seed-per-N trend lands first. parallels 8, flock+idempotent+disk-guard.
exec 9>/tmp/run_faster.lock
flock -n 9 || { echo "[f] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
BASE="--building-mode medium --closure-weight 8 --w-pos 0.6 --w-gap 0.8 --target-max-speed 12 --target-min-speed 8"
SUM=p3_results/faster_summary.txt
echo "[f] === START $(date) ===" | tee -a $SUM
modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
train_if () {  # $1=name $2=args
  local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 10 ] && { echo "[f] ABORT low disk"|tee -a $SUM; exit 1; }
  if [ -n "$(modelpath $1)" ]; then echo "[f] SKIP $1"|tee -a $SUM; return; fi
  echo "[f] TRAIN $1 ($2) $(date)" | tee -a $SUM
  python $RUN --name "$1" $2 --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[f] $1 train_exit=$?" | tee -a $SUM
}
eval_if () {  # $1=name $2=N
  [ -f p3_results/eval_${1}_medium.json ] && { echo "[f] SKIP eval $1"|tee -a $SUM; return; }
  local FM; FM=$(modelpath $1); [ -z "$FM" ] && { echo "[f] $1 NO_MODEL"|tee -a $SUM; return; }
  echo -n "[f] $1 eval: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --num-agents $2 --building-mode medium --target-max-speed 12 --target-min-speed 8 --k 15 --seed 7 --out p3_results/eval_${1}_medium.json 2>/dev/null | grep RESULT | tee -a $SUM
}

# N-sweep vs a FASTER evader (skip N=4 = documented death-spiral). Seed-outer -> quick trend.
for s in 1 2; do
  for N in 3 5 6 8 10; do
    train_if f_N${N}_s$s "--seed $s --num-agents $N $BASE"
    eval_if f_N${N}_s$s $N
  done
done

echo "[f] === DONE $(date) ===" | tee -a $SUM
