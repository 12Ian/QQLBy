#!/bin/bash
# A: run N=4 directly with SAFE SPAWN (server env default spawn_safety_enabled=True) + surround
# + evader-faster (vmax12). Tests whether safe/valid spawn rescues the documented N=4 death-
# spiral, and completes the surround-faster N-sweep at N=4. Uses the SERVER's current merged
# env/scripts (my surround + SDD safe-spawn) -- deploys ONLY this campaign script.
exec 9>/tmp/run_n4safe.lock
flock -n 9 || { echo "[n4] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
BASE="--building-mode medium --closure-weight 8 --w-pos 0.6 --w-gap 0.8 --target-max-speed 12 --target-min-speed 8 --surround-spawn"
SUM=p3_results/n4safe_summary.txt
echo "[n4] === START $(date) ===" | tee -a $SUM
modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
train_if () {
  local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 10 ] && { echo "[n4] ABORT low disk"|tee -a $SUM; exit 1; }
  if [ -n "$(modelpath $1)" ]; then echo "[n4] SKIP $1"|tee -a $SUM; return; fi
  echo "[n4] TRAIN $1 ($2) $(date)" | tee -a $SUM
  python $RUN --name "$1" $2 --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[n4] $1 train_exit=$?" | tee -a $SUM
}
eval_if () {
  [ -f p3_results/eval_${1}_medium.json ] && { echo "[n4] SKIP eval $1"|tee -a $SUM; return; }
  local FM; FM=$(modelpath $1); [ -z "$FM" ] && { echo "[n4] $1 NO_MODEL"|tee -a $SUM; return; }
  echo -n "[n4] $1 eval: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --num-agents 4 --building-mode medium --target-max-speed 12 --target-min-speed 8 --surround-spawn --spawn-safety on --k 15 --seed 7 --out p3_results/eval_${1}_medium.json 2>/dev/null | grep RESULT | tee -a $SUM
}

for s in 1 2; do
  train_if fs_N4_s$s "--seed $s --num-agents 4 $BASE"
  eval_if fs_N4_s$s
done

echo "[n4] === DONE $(date) ===" | tee -a $SUM
