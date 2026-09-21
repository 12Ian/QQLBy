#!/bin/bash
# STRONG-EVADER flagship at N=3 (client-confirmed 2026-07-09). Data shows N=3 is the
# strongest & most stable vs the fast evader: @medium 73% (80/67), 0% escape, lowest
# variance -- consistent with the core finding "smaller teams beat larger ones vs a fast
# evader" (large teams collide). N=5 was weaker/variable (49%, 73/47/27); N=4 death-spiral.
# This run: 3rd N3 seed for the headline band + clean reward ablations at N3. Existing
# s_N3_s1/s2 density evals already done. parallels 8, flock+idempotent+disk-guard. Prefix s3_.
exec 9>/tmp/run_strong_n3.lock
flock -n 9 || { echo "[s3] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
BASE="--building-mode medium --closure-weight 8 --w-pos 0.6 --w-gap 0.8 --target-max-speed 10 --target-min-speed 8"
SUM=p3_results/strong_n3_summary.txt
echo "[s3] === START $(date) ===" | tee -a $SUM

modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
train_if () {  # $1=name $2=train-args
  local FREE; FREE=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "${FREE:-0}" -lt 10 ]; then echo "[s3] ABORT $1: low disk ${FREE}G free $(date)" | tee -a $SUM; exit 1; fi
  if [ -n "$(modelpath $1)" ]; then echo "[s3] SKIP train $1" | tee -a $SUM; return; fi
  echo "[s3] TRAIN $1 ($2) $(date)" | tee -a $SUM
  python $RUN --name "$1" $2 --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[s3] $1 train_exit=$?" | tee -a $SUM
}
eval_if () {  # $1=name $2=density $3=eval-args
  if [ -f p3_results/eval_${1}_${2}.json ]; then echo "[s3] SKIP eval $1@$2" | tee -a $SUM; return; fi
  local FM; FM=$(modelpath $1)
  if [ -z "$FM" ]; then echo "[s3] $1 @$2: NO_MODEL" | tee -a $SUM; return; fi
  echo -n "[s3] $1 @$2: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --seed 7 --building-mode $2 $3 --k 15 --out p3_results/eval_${1}_${2}.json 2>/dev/null | grep RESULT | tee -a $SUM
}

EV3="--num-agents 3 --target-max-speed 10 --target-min-speed 8"

# (1) 3rd N3 seed for the flagship headline band, eval across densities
train_if s_N3_s3 "--seed 3 --num-agents 3 $BASE"
for D in empty open medium complex; do eval_if s_N3_s3 $D "$EV3"; done

# (2) clean reward-term ablation at N3
for T in r_pos r_closure r_gap r_finish r_near r_safe; do
  for s in 1 2; do
    train_if s3_ab_${T}_s$s "--seed $s --num-agents 3 $BASE --reward-disable $T"
    eval_if s3_ab_${T}_s$s medium "$EV3"
  done
done

# (3) closure ablation at N3
for s in 1 2; do
  train_if s3_noclose_s$s "--seed $s --num-agents 3 $BASE --no-guide-collapse --closure-weight 0"
  eval_if s3_noclose_s$s medium "$EV3"
done

echo "[s3] === DONE $(date) ===" | tee -a $SUM
