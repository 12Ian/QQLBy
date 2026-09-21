#!/bin/bash
# Toward >=80%: corrected levers. Prior f80 sweep FAILED (27%) because separation spread the
# pursuers (encirclement gaps -> faster evader escapes) and open density gave the fast evader
# room. Corrected: keep obstacles (medium, help corner), NO separation (stay tight), and
# soft-collisions (non-fatal) to UNLOCK large teams -- previously N>=6 collapsed by fatal
# collision; with soft-collisions, more pursuers can pack a tight seamless encirclement to
# seal a faster evader's escape directions. N-sweep 3/5/6/8, surround, evader-faster vmax12,
# radius 200 (proven), seed-outer. Eval collision-mode soft. flock+guard.
exec 9>/tmp/run_softcol.lock
flock -n 9 || { echo "[sc] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
LEVERS="--surround-spawn --soft-collisions --building-mode medium --target-max-speed 12 --target-min-speed 8 --closure-weight 8 --w-pos 0.6 --w-gap 0.8"
SUM=p3_results/softcol_summary.txt
echo "[sc] === START $(date) ===" | tee -a $SUM
modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
train_if () {
  local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 10 ] && { echo "[sc] ABORT low disk"|tee -a $SUM; exit 1; }
  if [ -n "$(modelpath $1)" ]; then echo "[sc] SKIP $1"|tee -a $SUM; return; fi
  echo "[sc] TRAIN $1 ($2) $(date)" | tee -a $SUM
  python $RUN --name "$1" $2 --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[sc] $1 train_exit=$?" | tee -a $SUM
}
eval_if () {  # $1=name $2=N
  [ -f p3_results/eval_${1}_medium.json ] && { echo "[sc] SKIP eval $1"|tee -a $SUM; return; }
  local FM; FM=$(modelpath $1); [ -z "$FM" ] && { echo "[sc] $1 NO_MODEL"|tee -a $SUM; return; }
  echo -n "[sc] $1 eval: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --num-agents $2 --building-mode medium --target-max-speed 12 --target-min-speed 8 --surround-spawn --collision-mode soft --k 15 --seed 7 --out p3_results/eval_${1}_medium.json 2>/dev/null | grep RESULT | tee -a $SUM
}

for s in 1 2; do
  for N in 3 5 6 8; do
    train_if sc_N${N}_s$s "--seed $s --num-agents $N $LEVERS"
    eval_if sc_N${N}_s$s $N
  done
done
echo "[sc] === DONE $(date) ===" | tee -a $SUM
