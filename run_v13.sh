#!/bin/bash
# ROUND 4 - push N=4 to 80%. N=3 is DONE (83.3/83.3/86.7 -> mean 84.4%, target met).
# N=4 with sep 4/20 gives 63.3/83.3/66.7 (mean 71.1%) and collision is now the ONLY failure
# mode left (37%/17%/37%, timeout 0% on all three). The best seed is exactly the low-collision
# one (17% -> 83.3%), so cutting collisions further is the direct path to 80%.
# N=4's nominal tetrahedral capture spacing is ~24.5 m at a 15 m catch radius, so a 22 m
# barrier still does NOT oppose a valid four-agent enclosure -- there is real headroom above
# the 20 m/weight-4 setting. This runs sep 6/22 on all three seeds.
# Usage: run_v13.sh <tag>   tag in {a,b}
TAG=${1:?tag required}
exec 9>/tmp/run_v13_$TAG.lock
flock -n 9 || { echo "[v13-$TAG] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
SUM=p3_results/v13_${TAG}_summary.txt
modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
echo "[v13-$TAG] === START $(date) ===" | tee -a $SUM

job () {  # $1=name $2=seed
  local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 12 ] && { echo "[v13-$TAG] ABORT low disk ${F}G"|tee -a $SUM; exit 1; }
  if [ -z "$(modelpath $1)" ]; then
    echo "[v13-$TAG] TRAIN $1 N=4 seed=$2 cw16 pin12 sep=6/22 $(date)" | tee -a $SUM
    python $RUN --name "$1" --seed $2 --num-agents 4 --building-mode medium \
      --target-max-speed 10 --target-min-speed 8 --closure-weight 16 --pincer-weight 12 \
      --separation-weight 6 --separation-distance 22 \
      --w-pos 0.6 --w-gap 0.8 --steps 5000000 --parallels 8 --eval-interval 250000 \
      --test-episode 20 > p3_results/train_$1.log 2>&1
    echo "[v13-$TAG] $1 train_exit=$?" | tee -a $SUM
  else echo "[v13-$TAG] SKIP train $1" | tee -a $SUM; fi
  for k in 30 15; do
    local OUT=p3_results/eval_${1}_k$k.json
    [ -f "$OUT" ] && continue
    local FM; FM=$(modelpath $1); [ -z "$FM" ] && { echo "[v13-$TAG] $1 NO_MODEL"|tee -a $SUM; return; }
    echo -n "[v13-$TAG] $1 k=$k: " | tee -a $SUM
    python $EVAL --policy model --model-path "$FM" --num-agents 4 --building-mode medium \
      --target-max-speed 10 --target-min-speed 8 --k $k --seed 7 \
      --out "$OUT" 2>/dev/null | grep RESULT | tee -a $SUM
  done
}

case "$TAG" in
  a) job v13_N4_s1 1; job v13_N4_s2 2 ;;
  b) job v13_N4_s3 3 ;;
esac
echo "[v13-$TAG] === DONE $(date) ===" | tee -a $SUM
