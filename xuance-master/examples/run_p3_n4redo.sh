#!/bin/bash
# Re-run the N=4 point of Exp B with fresh seeds. The s1/s2 N=4 runs DIVERGED
# (best training score ~ -1 vs ~580 for N=3/5/8), so N=4=0% is a training-stability
# outlier, NOT a capacity boundary (N=3 captures at 90%). Try seeds 3 and 4; if these
# also collapse, N=4 instability is systematic under these hyperparameters.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
BASE="--closure-weight 8 --w-pos 0.6 --w-gap 0.8 --target-max-speed 6"
SUM=p3_results/supp_summary.txt
echo "[p3] === N4-REDO START $(date) ===" | tee -a $SUM

train () { python $RUN --name "$1" $2 --steps 5000000 --parallels 16 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1; echo "[p3] $1 train_exit=$?" | tee -a $SUM; }
evald () {
  local FM; FM=$(ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1)
  if [ -z "$FM" ]; then echo "[p3] $1 @$2: NO_MODEL" | tee -a $SUM; return; fi
  echo -n "[p3] $1 @$2: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --seed 7 --building-mode $2 --target-max-speed 6 --num-agents 4 --k 30 --out p3_results/eval_${1}_${2}.json 2>/dev/null | grep RESULT | tee -a $SUM
}

for s in 3 4; do
  echo "[p3] TRAIN nsweep_N4r_s$s $(date)" | tee -a $SUM
  train nsweep_N4r_s$s "--seed $s --building-mode medium $BASE --num-agents 4"
  for D in open medium; do evald nsweep_N4r_s$s $D; done
done
echo "[p3] === N4-REDO DONE $(date) ===" | tee -a $SUM
