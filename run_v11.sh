#!/bin/bash
# ROUND 2: kill the COLLISION mode that the closure fix exposed.
# Round 1 (cw16 [+pincer]) drove timeout/false-capture to ~0 everywhere (0-3%), proving the
# "encircled but never closes" root cause is fixed. Failures then shifted wholesale to
# COLLISION, which scales monotonically with team size (N3 30% -> N4 57% -> N6 77%) on an
# IDENTICAL obstacle set => these are teammate collisions from bunching at the capture point.
# This round adds ONLY the env's teammate separation barrier (weight 0 -> 4 at 20 m, the
# documented design point that leaves the ~24.5 m tetrahedral capture spacing untouched).
# Single variable changed vs the v10_* runs, so the comparison is clean.
# Usage: run_v11.sh <tag>   tag in {n4,n3,big}
TAG=${1:?tag required}
exec 9>/tmp/run_v11_$TAG.lock
flock -n 9 || { echo "[v11-$TAG] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
SUM=p3_results/v11_${TAG}_summary.txt
modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
echo "[v11-$TAG] === START $(date) ===" | tee -a $SUM

train_v11 () {  # $1=name $2=N $3=seed $4=pincer
  local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 12 ] && { echo "[v11-$TAG] ABORT low disk ${F}G"|tee -a $SUM; exit 1; }
  [ -n "$(modelpath $1)" ] && { echo "[v11-$TAG] SKIP train $1"|tee -a $SUM; return; }
  echo "[v11-$TAG] TRAIN $1 N=$2 seed=$3 vmax10 cw16 pin=$4 sep=4/20 5M $(date)" | tee -a $SUM
  python $RUN --name "$1" --seed $3 --num-agents $2 --building-mode medium \
    --target-max-speed 10 --target-min-speed 8 --closure-weight 16 --pincer-weight $4 \
    --separation-weight 4 --separation-distance 20 \
    --w-pos 0.6 --w-gap 0.8 --steps 5000000 --parallels 8 --eval-interval 250000 \
    --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[v11-$TAG] $1 train_exit=$?" | tee -a $SUM
}
eval_v11 () {  # $1=name $2=N $3=k
  local OUT=p3_results/eval_${1}_k$3.json
  [ -f "$OUT" ] && { echo "[v11-$TAG] SKIP eval $1 k=$3"|tee -a $SUM; return; }
  local FM; FM=$(modelpath $1); [ -z "$FM" ] && { echo "[v11-$TAG] $1 NO_MODEL"|tee -a $SUM; return; }
  echo -n "[v11-$TAG] $1 k=$3: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --num-agents $2 --building-mode medium \
    --target-max-speed 10 --target-min-speed 8 --k $3 --seed 7 \
    --out "$OUT" 2>/dev/null | grep RESULT | tee -a $SUM
}
job () { train_v11 "$1" "$2" "$3" "$4"; eval_v11 "$1" "$2" 30; eval_v11 "$1" "$2" 15; }

case "$TAG" in
  n4)  job v11_N4_s1 4 1 12; job v11_N4_s2 4 2 12 ;;   # the N=4 fix (report excluded it)
  n3)  job v11_N3_s1 3 1 0;  job v11_N3_s2 3 2 0  ;;   # flagship: no pincer, isolates separation
  big) job v11_N6_s1 6 1 12; job v11_N5_s1 5 1 12 ;;   # most collision-bound (77%/43%)
esac
echo "[v11-$TAG] === DONE $(date) ===" | tee -a $SUM
