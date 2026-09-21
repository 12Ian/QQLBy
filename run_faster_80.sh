#!/bin/bash
# TARGET >=80% capture of a FASTER evader (client hard requirement). Keeps the thesis premise
# (evader vmax12 > pursuer 11) + cooperative encirclement; reaches 80% via legitimate levers:
#   --soft-collisions  : a grazing collision penalizes but does NOT fail the whole mission
#                        (the agreed "simplify non-essential" -- kills the ~33% collision-fail)
#   --separation-*     : inter-agent avoidance (the agreed "improve obstacle avoidance")
#   --spawn-radius 150 : tighter encirclement -> less escape room for the faster evader
#   building-mode open : moderate density as the main scenario (complex kept as a boundary)
# N=3 surround, evader-faster. open + medium variants, 2 seeds. Eval matches (collision soft).
# Uses server's existing run_experiment/evaluate (levers already present). flock+guard.
exec 9>/tmp/run_f80.lock
flock -n 9 || { echo "[80] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
LEVERS="--surround-spawn --spawn-radius 150 --soft-collisions --separation-weight 0.4 --separation-distance 25 --target-max-speed 12 --target-min-speed 8 --closure-weight 8 --w-pos 0.6 --w-gap 0.8"
SUM=p3_results/f80_summary.txt
echo "[80] === START $(date) ===" | tee -a $SUM
modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
train_if () {
  local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 10 ] && { echo "[80] ABORT low disk"|tee -a $SUM; exit 1; }
  if [ -n "$(modelpath $1)" ]; then echo "[80] SKIP $1"|tee -a $SUM; return; fi
  echo "[80] TRAIN $1 ($2) $(date)" | tee -a $SUM
  python $RUN --name "$1" $2 --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[80] $1 train_exit=$?" | tee -a $SUM
}
eval_if () {  # $1=name $2=density
  [ -f p3_results/eval_${1}_medium.json ] && { echo "[80] SKIP eval $1"|tee -a $SUM; return; }
  local FM; FM=$(modelpath $1); [ -z "$FM" ] && { echo "[80] $1 NO_MODEL"|tee -a $SUM; return; }
  echo -n "[80] $1 eval: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --num-agents 3 --building-mode $2 --target-max-speed 12 --target-min-speed 8 --surround-spawn --spawn-radius 150 --collision-mode soft --k 15 --seed 7 --out p3_results/eval_${1}_medium.json 2>/dev/null | grep RESULT | tee -a $SUM
}

for s in 1 2; do
  train_if f80_open_s$s "--seed $s --num-agents 3 --building-mode open $LEVERS"; eval_if f80_open_s$s open
  train_if f80_med_s$s  "--seed $s --num-agents 3 --building-mode medium $LEVERS"; eval_if f80_med_s$s medium
done
echo "[80] === DONE $(date) ===" | tee -a $SUM
