#!/bin/bash
# N=4 non-convergence DIAGNOSTIC. Ruled out: env geometry (greedy solves N=4), obs==state
# (MADDPG critic uses joint obs dim N*(obs+act)=180, not the 42-dim state). Leading hypothesis:
# guide-collapse pulls the 4 guide points to a shrinking centre -> agents converge & collide
# (learned N=4 collides 100% even @empty). Short 1.5M runs to localize the mechanism.
# Pauses the resume campaign + redundant N4r_s5 to give the GPU to clean diagnostics.
pkill -f resume_p3.sh 2>/dev/null
pkill -f "run_experiment.py --name nsweep_N4r" 2>/dev/null
sleep 6
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
BASE="--closure-weight 8 --w-pos 0.6 --w-gap 0.8 --target-max-speed 6 --num-agents 4"
SUM=p3_results/diag_n4.txt
echo "[diag] === START $(date) ===" | tee -a $SUM

run () {  # $1=name $2=train-density $3=extra-args $4=eval-density
  echo "[diag] TRAIN $1 (density=$2 $3) $(date)" | tee -a $SUM
  python $RUN --name "$1" --seed 1 --building-mode $2 $BASE $3 --steps 1500000 --parallels 16 --eval-interval 1500000 --test-episode 10 > p3_results/train_$1.log 2>&1
  echo "[diag] $1 train_exit=$?" | tee -a $SUM
  local FM; FM=$(ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1)
  if [ -z "$FM" ]; then echo "[diag] $1: NO_MODEL" | tee -a $SUM; return; fi
  echo -n "[diag] $1 eval@$4: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --seed 7 --building-mode $4 $BASE --k 20 2>/dev/null | grep -oE "success_rate[^,]*|collision_rate[^,]*" | tr "\n" " " | tee -a $SUM; echo "" | tee -a $SUM
}

# H1: guide-collapse death-spiral? Turn it OFF and see if N=4 converges @medium.
run diag_n4_nocollapse medium "--no-guide-collapse" medium
# H2: obstacle-dependent? Train & eval in the EMPTY scene (greedy solves it there).
run diag_n4_empty empty "" empty
# Control: normal N=4 short -> should stay flat/fail fast, confirming 1.5M distinguishes.
run diag_n4_control medium "" medium

echo "[diag] === DONE $(date) ===" | tee -a $SUM
