#!/bin/bash
# Chapter 4 multi-target campaign: (A) success vs number of targets M in {1,2,3} (N=6),
# (B) allocation ablation M=2 dynamic vs fixed. Same closure-reward method as Ch2-3.
# Waits for the baselines to free the GPU. Idempotent + flock-guarded. parallels 8
# (MADDPG footprint is safe on the shared 30GB box).
exec 9>/tmp/run_ch4.lock
flock -n 9 || { echo "[ch4] locked; exit"; exit 0; }
while pgrep -f run_baselines.sh >/dev/null 2>&1; do sleep 120; done

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_mt.py
ENV=uav_pursuit_apollonius_multitarget_3d
BASE="--env $ENV --num-agents 6 --building-mode medium --closure-weight 8 --target-max-speed 6"
SUM=p3_results/ch4_summary.txt
echo "[ch4] === START $(date) ===" | tee -a $SUM

modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
train_if () {  # $1=name $2=train-args
  if [ -n "$(modelpath $1)" ]; then echo "[ch4] SKIP train $1" | tee -a $SUM; return; fi
  echo "[ch4] TRAIN $1 ($2) $(date)" | tee -a $SUM
  python $RUN --name "$1" $2 --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[ch4] $1 train_exit=$?" | tee -a $SUM
}
eval_if () {  # $1=name $2=eval-args
  if [ -f p3_results/eval_${1}.json ]; then echo "[ch4] SKIP eval $1" | tee -a $SUM; return; fi
  local FM; FM=$(modelpath $1)
  if [ -z "$FM" ]; then echo "[ch4] $1: NO_MODEL" | tee -a $SUM; return; fi
  echo -n "[ch4] $1: " | tee -a $SUM
  python $EVAL --model-path "$FM" $2 --seed 7 --k 30 --out p3_results/eval_${1}.json 2>/dev/null | grep RESULT | tee -a $SUM
}

# (A) M-sweep
for M in 1 2 3; do
  for s in 1 2; do
    train_if ch4_M${M}_s$s "$BASE --num-targets $M --seed $s"
    eval_if ch4_M${M}_s$s "--num-agents 6 --num-targets $M --building-mode medium --target-max-speed 6"
  done
done

# (B) allocation ablation: M=2 fixed vs the dynamic M=2 above
for s in 1 2; do
  train_if ch4_M2fixed_s$s "$BASE --num-targets 2 --no-dynamic-alloc --seed $s"
  eval_if ch4_M2fixed_s$s "--num-agents 6 --num-targets 2 --building-mode medium --target-max-speed 6 --no-dynamic-alloc"
done

echo "[ch4] === DONE $(date) ===" | tee -a $SUM
