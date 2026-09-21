#!/bin/bash
# STRONG-EVADER flagship at N=5. N=4 is a documented MADDPG training death-spiral outlier
# (client decision 2026-07-02: do NOT burn GPU on it). N=5 is the stable, sufficient-for-3D-
# encirclement flagship (strong-core: s_N5 = 73%/47% @medium). This run: (0) density
# generalization for existing stable N5+N3 models (evals only, fast); (1) 3rd N5 seed;
# (2) clean reward ablations at N5; (3) closure ablation at N5. parallels 8. flock+idempotent
# +disk-guard. Prefix "s5_" for new trains; reuses s_N5_*/s_N3_* models for density.
exec 9>/tmp/run_strong_n5.lock
flock -n 9 || { echo "[s5] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
BASE="--building-mode medium --closure-weight 8 --w-pos 0.6 --w-gap 0.8 --target-max-speed 10 --target-min-speed 8"
SUM=p3_results/strong_n5_summary.txt
echo "[s5] === START $(date) ===" | tee -a $SUM

modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
train_if () {  # $1=name $2=train-args
  local FREE; FREE=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "${FREE:-0}" -lt 10 ]; then echo "[s5] ABORT $1: low disk ${FREE}G free $(date)" | tee -a $SUM; exit 1; fi
  if [ -n "$(modelpath $1)" ]; then echo "[s5] SKIP train $1" | tee -a $SUM; return; fi
  echo "[s5] TRAIN $1 ($2) $(date)" | tee -a $SUM
  python $RUN --name "$1" $2 --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[s5] $1 train_exit=$?" | tee -a $SUM
}
eval_if () {  # $1=name $2=density $3=eval-args
  if [ -f p3_results/eval_${1}_${2}.json ]; then echo "[s5] SKIP eval $1@$2" | tee -a $SUM; return; fi
  local FM; FM=$(modelpath $1)
  if [ -z "$FM" ]; then echo "[s5] $1 @$2: NO_MODEL" | tee -a $SUM; return; fi
  echo -n "[s5] $1 @$2: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --seed 7 --building-mode $2 $3 --k 15 --out p3_results/eval_${1}_${2}.json 2>/dev/null | grep RESULT | tee -a $SUM
}

EV5="--num-agents 5 --target-max-speed 10 --target-min-speed 8"
EV3="--num-agents 3 --target-max-speed 10 --target-min-speed 8"

# (0) FAST: density generalization for existing stable models (reuse, evals only)
for name in s_N5_s1 s_N5_s2; do
  for D in empty open medium complex; do eval_if $name $D "$EV5"; done
done
for name in s_N3_s1 s_N3_s2; do
  for D in empty open medium complex; do eval_if $name $D "$EV3"; done
done

# (1) 3rd N5 seed to firm up the flagship, eval across densities
train_if s_N5_s3 "--seed 3 --num-agents 5 $BASE"
for D in empty open medium complex; do eval_if s_N5_s3 $D "$EV5"; done

# (2) clean reward-term ablation at N5
for T in r_pos r_closure r_gap r_finish r_near r_safe; do
  for s in 1 2; do
    train_if s5_ab_${T}_s$s "--seed $s --num-agents 5 $BASE --reward-disable $T"
    eval_if s5_ab_${T}_s$s medium "$EV5"
  done
done

# (3) closure ablation at N5
for s in 1 2; do
  train_if s5_noclose_s$s "--seed $s --num-agents 5 $BASE --no-guide-collapse --closure-weight 0"
  eval_if s5_noclose_s$s medium "$EV5"
done

echo "[s5] === DONE $(date) ===" | tee -a $SUM
