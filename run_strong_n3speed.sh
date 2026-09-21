#!/bin/bash
# N=3 flagship speed sweep: characterize success vs evader max-speed at the flagship team
# size, for a consistent headline (the earlier sv_/s_lam sweep was at N=6). vmax10 = the
# flagship s_N3 runs (already have). Adds vmax 8/9/11/12 at N=3, 1 seed each. flock+guard.
exec 9>/tmp/run_n3speed.lock
flock -n 9 || { echo "[s3v] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
BASE="--building-mode medium --closure-weight 8 --w-pos 0.6 --w-gap 0.8 --target-min-speed 8"
SUM=p3_results/strong_n3speed_summary.txt
echo "[s3v] === START $(date) ===" | tee -a $SUM
modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
for V in 8 9 11 12; do
  N=s3_lam_V${V}
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "${FREE:-0}" -lt 10 ]; then echo "[s3v] ABORT low disk ${FREE}G" | tee -a $SUM; exit 1; fi
  if [ -z "$(modelpath $N)" ]; then
    echo "[s3v] TRAIN $N (vmax=$V) $(date)" | tee -a $SUM
    python $RUN --name "$N" --seed 1 --num-agents 3 $BASE --target-max-speed $V \
      --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 > p3_results/train_$N.log 2>&1
    echo "[s3v] $N train_exit=$?" | tee -a $SUM
  fi
  FM=$(modelpath $N)
  if [ -n "$FM" ] && [ ! -f p3_results/eval_${N}_medium.json ]; then
    echo -n "[s3v] $N eval: " | tee -a $SUM
    python $EVAL --policy model --model-path "$FM" --seed 7 --building-mode medium \
      --num-agents 3 --target-max-speed $V --target-min-speed 8 --k 15 \
      --out p3_results/eval_${N}_medium.json 2>/dev/null | grep RESULT | tee -a $SUM
  fi
done
echo "[s3v] === DONE $(date) ===" | tee -a $SUM
