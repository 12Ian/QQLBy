#!/bin/bash
# Bring the Ch3-supplement sweeps to 3 seeds (consistency with the 3-seed core results).
# Adds seed 3 to: N-sweep (N=3/5/8; N=6 already has 3 via anchor, N=4 is a documented outlier),
# lambda-sweep (V=4/6/8/10), reward ablation (6 terms). Idempotent + flock-guarded, so safe to
# re-run and to @reboot-resume.
exec 9>/tmp/add_seed3.lock
flock -n 9 || { echo "[s3] another instance holds the lock; exiting"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
BASE="--closure-weight 8 --w-pos 0.6 --w-gap 0.8"
SPEED6="--target-max-speed 6"
ESEED="--seed 7"
SUM=p3_results/supp_summary.txt
echo "[s3] === START $(date) ===" | tee -a $SUM

modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
train_if () {
  if [ -n "$(modelpath $1)" ]; then echo "[s3] SKIP train $1" | tee -a $SUM; return; fi
  echo "[s3] TRAIN $1 ($2) $(date)" | tee -a $SUM
  python $RUN --name "$1" $2 --steps 5000000 --parallels 16 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[s3] $1 train_exit=$?" | tee -a $SUM
}
eval_if () {
  if [ -f p3_results/eval_${1}_${2}.json ]; then echo "[s3] SKIP eval $1@$2" | tee -a $SUM; return; fi
  local FM; FM=$(modelpath $1)
  if [ -z "$FM" ]; then echo "[s3] $1 @$2: NO_MODEL" | tee -a $SUM; return; fi
  echo -n "[s3] $1 @$2: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" $ESEED --building-mode $2 $3 --k 30 --out p3_results/eval_${1}_${2}.json 2>/dev/null | grep RESULT | tee -a $SUM
}

# N-sweep seed 3 (N=6 anchor already 3-seed; N=4 is a documented outlier)
for N in 3 5 8; do
  train_if nsweep_N${N}_s3 "--seed 3 --building-mode medium $BASE $SPEED6 --num-agents $N"
  for D in open medium; do eval_if nsweep_N${N}_s3 $D "$SPEED6 --num-agents $N"; done
done

# lambda-sweep seed 3 (constant-speed evader)
for V in 6 8 4 10; do
  train_if lsweep_V${V}_s3 "--seed 3 --building-mode medium $BASE --target-min-speed $V --target-max-speed $V"
  eval_if lsweep_V${V}_s3 medium "--target-min-speed $V --target-max-speed $V"
done

# reward-term ablation seed 3
for T in r_pos r_closure r_gap r_finish r_near r_safe; do
  train_if rewabl_${T}_s3 "--seed 3 --building-mode medium $BASE $SPEED6 --reward-disable $T"
  eval_if rewabl_${T}_s3 medium "$SPEED6"
done

echo "[s3] === DONE $(date) ===" | tee -a $SUM
