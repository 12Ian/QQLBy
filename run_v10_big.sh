#!/bin/bash
# FIX THE REPORT'S OWN TABLE - the sub-80% entries N=5 (49%), N=6 (31%), and re-run the
# N=3 flagship (80%) with the corrected closure weight. Same rationale as run_v10_n4.sh:
# the entire published N-sweep used --closure-weight 8, proven badly under-weighted.
# REPORT PROTOCOL: vmax10, medium, 5M steps, parallels 8, one-sided spawn, eval seed 7.
# Only the closure weight changes (8->16); N>=5 also gets the pincer term that cut the
# "encircled but never closes" timeout mode. Eval k=15 (comparable) + k=30 (honest).
exec 9>/tmp/run_v10_big.lock
flock -n 9 || { echo "[vB] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
SUM=p3_results/v10_big_summary.txt
modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
echo "[vB] === START $(date) ===" | tee -a $SUM

train_v10 () {  # $1=name $2=N $3=seed $4=pincer
  local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 12 ] && { echo "[vB] ABORT low disk ${F}G"|tee -a $SUM; exit 1; }
  [ -n "$(modelpath $1)" ] && { echo "[vB] SKIP train $1"|tee -a $SUM; return; }
  echo "[vB] TRAIN $1 N=$2 seed=$3 vmax10 cw16 pin=$4 5M $(date)" | tee -a $SUM
  python $RUN --name "$1" --seed $3 --num-agents $2 --building-mode medium \
    --target-max-speed 10 --target-min-speed 8 --closure-weight 16 --pincer-weight $4 \
    --w-pos 0.6 --w-gap 0.8 --steps 5000000 --parallels 8 --eval-interval 250000 \
    --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[vB] $1 train_exit=$?" | tee -a $SUM
}
eval_v10 () {  # $1=name $2=N $3=k
  local OUT=p3_results/eval_${1}_k$3.json
  [ -f "$OUT" ] && { echo "[vB] SKIP eval $1 k=$3"|tee -a $SUM; return; }
  local FM; FM=$(modelpath $1); [ -z "$FM" ] && { echo "[vB] $1 NO_MODEL"|tee -a $SUM; return; }
  echo -n "[vB] $1 k=$3: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --num-agents $2 --building-mode medium \
    --target-max-speed 10 --target-min-speed 8 --k $3 --seed 7 \
    --out "$OUT" 2>/dev/null | grep RESULT | tee -a $SUM
}

# N=3 flagship with corrected closure weight (report: 80%) - no pincer (cw16 alone won at N=3)
train_v10 v10_N3_s1 3 1 0;  eval_v10 v10_N3_s1 3 30; eval_v10 v10_N3_s1 3 15
# N=5 (report: 49%) and N=6 (report: 31%) - cw16 + pincer
train_v10 v10_N5_s1 5 1 12; eval_v10 v10_N5_s1 5 30; eval_v10 v10_N5_s1 5 15
train_v10 v10_N6_s1 6 1 12; eval_v10 v10_N6_s1 6 30; eval_v10 v10_N6_s1 6 15
# second N=3 seed for an honest band
train_v10 v10_N3_s2 3 2 0;  eval_v10 v10_N3_s2 3 30; eval_v10 v10_N3_s2 3 15
echo "[vB] === DONE $(date) ===" | tee -a $SUM
