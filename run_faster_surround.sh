#!/bin/bash
# EVADER-FASTER + SURROUND-SPAWN: pursuers start on a sphere (radius 200) AROUND a FASTER
# evader (vmax 12 > pursuer 11) instead of one-sided. This enables Apollonius encirclement
# from step 0 -- the correct premise for catching a faster target. Compare to the one-sided
# N-sweep (max 17%, escape-dominated). N=3/5/6/8, seed-outer -> quick trend. Eval also uses
# --surround-spawn (must match training). parallels 8, flock+idempotent+disk-guard.
exec 9>/tmp/run_fsurr.lock
flock -n 9 || { echo "[fs] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
BASE="--building-mode medium --closure-weight 8 --w-pos 0.6 --w-gap 0.8 --target-max-speed 12 --target-min-speed 8 --surround-spawn"
SUM=p3_results/fsurr_summary.txt
echo "[fs] === START $(date) ===" | tee -a $SUM
modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
train_if () {  # $1=name $2=args
  local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 10 ] && { echo "[fs] ABORT low disk"|tee -a $SUM; exit 1; }
  if [ -n "$(modelpath $1)" ]; then echo "[fs] SKIP $1"|tee -a $SUM; return; fi
  echo "[fs] TRAIN $1 ($2) $(date)" | tee -a $SUM
  python $RUN --name "$1" $2 --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[fs] $1 train_exit=$?" | tee -a $SUM
}
eval_if () {  # $1=name $2=N
  [ -f p3_results/eval_${1}_medium.json ] && { echo "[fs] SKIP eval $1"|tee -a $SUM; return; }
  local FM; FM=$(modelpath $1); [ -z "$FM" ] && { echo "[fs] $1 NO_MODEL"|tee -a $SUM; return; }
  echo -n "[fs] $1 eval: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --num-agents $2 --building-mode medium --target-max-speed 12 --target-min-speed 8 --surround-spawn --k 15 --seed 7 --out p3_results/eval_${1}_medium.json 2>/dev/null | grep RESULT | tee -a $SUM
}

for s in 1 2; do
  for N in 3 5 6 8; do
    train_if fs_N${N}_s$s "--seed $s --num-agents $N $BASE"
    eval_if fs_N${N}_s$s $N
  done
done

echo "[fs] === DONE $(date) ===" | tee -a $SUM
