#!/bin/bash
# ROUND 3. Round 2 (cw16 [+pincer] + separation 4/20) hit the targets at small N:
#   N=3: 83.3% / 83.3% (mean 83.3%, both seeds >= 80, timeout 0)   <- deliverable met
#   N=4: 63.3% / 83.3% (mean 73.3%, was EXCLUDED as a death-spiral) <- deliverable met
# but the SAME separation weight hurt large teams (N=6: 3%, timeout 27%) -- a 20 m barrier
# among 6 agents opposes the tight encirclement they need. Two jobs:
#   s3   : 3rd seed for N=3 and N=4 on the WINNING recipe -> firmer 3-seed means for the report
#   big2 : gentler barrier (weight 2, 15 m) for N=5/N=6, which are still collision-bound
# Usage: run_v12.sh <tag>   tag in {s3,big2}
TAG=${1:?tag required}
exec 9>/tmp/run_v12_$TAG.lock
flock -n 9 || { echo "[v12-$TAG] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
SUM=p3_results/v12_${TAG}_summary.txt
modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
echo "[v12-$TAG] === START $(date) ===" | tee -a $SUM

train_v12 () {  # $1=name $2=N $3=seed $4=pincer $5=sepw $6=sepd
  local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 12 ] && { echo "[v12-$TAG] ABORT low disk ${F}G"|tee -a $SUM; exit 1; }
  [ -n "$(modelpath $1)" ] && { echo "[v12-$TAG] SKIP train $1"|tee -a $SUM; return; }
  echo "[v12-$TAG] TRAIN $1 N=$2 seed=$3 pin=$4 sep=$5/$6 $(date)" | tee -a $SUM
  python $RUN --name "$1" --seed $3 --num-agents $2 --building-mode medium \
    --target-max-speed 10 --target-min-speed 8 --closure-weight 16 --pincer-weight $4 \
    --separation-weight $5 --separation-distance $6 \
    --w-pos 0.6 --w-gap 0.8 --steps 5000000 --parallels 8 --eval-interval 250000 \
    --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[v12-$TAG] $1 train_exit=$?" | tee -a $SUM
}
eval_v12 () {
  local OUT=p3_results/eval_${1}_k$3.json
  [ -f "$OUT" ] && { echo "[v12-$TAG] SKIP eval $1 k=$3"|tee -a $SUM; return; }
  local FM; FM=$(modelpath $1); [ -z "$FM" ] && { echo "[v12-$TAG] $1 NO_MODEL"|tee -a $SUM; return; }
  echo -n "[v12-$TAG] $1 k=$3: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --num-agents $2 --building-mode medium \
    --target-max-speed 10 --target-min-speed 8 --k $3 --seed 7 \
    --out "$OUT" 2>/dev/null | grep RESULT | tee -a $SUM
}
job () { train_v12 "$1" "$2" "$3" "$4" "$5" "$6"; eval_v12 "$1" "$2" 30; eval_v12 "$1" "$2" 15; }

case "$TAG" in
  s3)   job v11_N3_s3 3 3 0  4 20; job v11_N4_s3 4 3 12 4 20 ;;   # 3rd seeds, winning recipe
  big2) job v12_N6_s1 6 1 12 2 15; job v12_N5_s1 5 1 12 2 15 ;;   # gentler barrier for big teams
esac
echo "[v12-$TAG] === DONE $(date) ===" | tee -a $SUM
