#!/bin/bash
# IDEMPOTENT resume of the Ch3 campaign after the 2026-07-01 02:24 server reboot killed it.
# Already done (kept): anchors, all Exp B (N-sweep), Exp C V6 x2 + V8_s1.
# Remaining: (1) N=4 REDO seeds 3,4,5  (2) Exp C V8_s2 + V4 + V10  (3) Exp A reward ablation x12.
# Skips any (name) whose final model / eval json already exists, so it is safe to re-run and to
# auto-resume on reboot. A flock guards against a double launch (manual nohup + @reboot cron).
exec 9>/tmp/resume_p3.lock
flock -n 9 || { echo "[resume] another instance holds the lock; exiting"; exit 0; }

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
mkdir -p p3_results
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
BASE="--closure-weight 8 --w-pos 0.6 --w-gap 0.8"
SPEED6="--target-max-speed 6"
ESEED="--seed 7"
SUM=p3_results/supp_summary.txt
echo "[resume] === START $(date) ===" | tee -a $SUM

modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }

train_if () {  # $1=name $2=args $3=steps
  if [ -n "$(modelpath $1)" ]; then echo "[resume] SKIP train $1 (model exists)" | tee -a $SUM; return; fi
  echo "[resume] TRAIN $1 ($2) $(date)" | tee -a $SUM
  python $RUN --name "$1" $2 --steps $3 --parallels 16 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[resume] $1 train_exit=$?" | tee -a $SUM
}
eval_if () {  # $1=name $2=density $3=args
  if [ -f p3_results/eval_${1}_${2}.json ]; then echo "[resume] SKIP eval $1@$2 (json exists)" | tee -a $SUM; return; fi
  local FM; FM=$(modelpath $1)
  if [ -z "$FM" ]; then echo "[resume] $1 @$2: NO_MODEL (train failed)" | tee -a $SUM; return; fi
  echo -n "[resume] $1 @$2: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" $ESEED --building-mode $2 $3 --k 30 --out p3_results/eval_${1}_${2}.json 2>/dev/null | grep RESULT | tee -a $SUM
}

# ---- (1) N=4 REDO (user priority): fresh seeds 3,4,5 (s1,s2 diverged = training instability) ----
for s in 3 4 5; do
  train_if nsweep_N4r_s$s "--seed $s --building-mode medium $BASE $SPEED6 --num-agents 4" 5000000
  for D in open medium; do eval_if nsweep_N4r_s$s $D "$SPEED6 --num-agents 4"; done
done

# ---- (2) Exp C remainder: constant-speed evader lambda sweep. V6 done; finish V8_s2, V4, V10. ----
for V in 8 4 10; do
  for s in 1 2; do
    train_if lsweep_V${V}_s$s "--seed $s --building-mode medium $BASE --target-min-speed $V --target-max-speed $V" 5000000
    eval_if lsweep_V${V}_s$s medium "--target-min-speed $V --target-max-speed $V"
  done
done

# ---- (3) Exp A: reward-term ablation @medium, 6 clean cells x 2 seeds ----
for T in r_pos r_closure r_gap r_finish r_near r_safe; do
  for s in 1 2; do
    train_if rewabl_${T}_s$s "--seed $s --building-mode medium $BASE $SPEED6 --reward-disable $T" 5000000
    eval_if rewabl_${T}_s$s medium "$SPEED6"
  done
done

echo "[resume] === DONE $(date) ===" | tee -a $SUM
