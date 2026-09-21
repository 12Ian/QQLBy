#!/bin/bash
# N=4 PUSH, relaunch (the first attempt died on a lock race and never ran).
# Evidence (k=30): N=4 cw16 s1=37% s2=53% (mean 45%); N=4 pincer(cw8+pin12) s2=73% @k=15.
# cw32 HURT at N=3 (43% vs 53%), so "more closure weight" is exhausted -> the 2nd-nearest
# pincer term is the real N=4 lever. Priority order:
#   1. fair k=30 number for the 73% pincer model
#   2. REPLICATE the pincer recipe on seed 1 -> gives an honest 2-seed mean (the deliverable)
#   3. cw16+pin12 combo -> can it beat plain pincer?
# Own lock only; runs CONCURRENTLY with the N=3 campaign (training is CPU-bound, GPU at 3%).
exec 9>/tmp/run_n4push2.lock
flock -n 9 || { echo "[n4p2] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
SUM=p3_results/n4push2_summary.txt
modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
echo "[n4p2] === START $(date) ===" | tee -a $SUM

evalk () {  # $1=name $2=N $3=k
  local OUT=p3_results/eval_${1}_k$3.json
  [ -f "$OUT" ] && { echo "[n4p2] SKIP eval $1 k=$3"|tee -a $SUM; return; }
  local FM; FM=$(modelpath $1); [ -z "$FM" ] && { echo "[n4p2] $1 NO_MODEL"|tee -a $SUM; return; }
  echo -n "[n4p2] $1 k=$3: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --num-agents $2 --building-mode medium \
    --target-max-speed 12 --target-min-speed 8 --surround-spawn --k $3 --seed 7 \
    --out "$OUT" 2>/dev/null | grep RESULT | tee -a $SUM
}
train4 () {  # $1=name $2=seed $3=warm $4=closure $5=pincer
  local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 12 ] && { echo "[n4p2] ABORT low disk ${F}G"|tee -a $SUM; exit 1; }
  [ -n "$(modelpath $1)" ] && { echo "[n4p2] SKIP train $1"|tee -a $SUM; return; }
  local WARM; WARM=$(modelpath $3); [ -z "$WARM" ] && { echo "[n4p2] $1 NO_WARM($3)"|tee -a $SUM; return; }
  echo "[n4p2] TRAIN $1 seed=$2 warm=$3 cw=$4 pin=$5 $(date)" | tee -a $SUM
  python $RUN --name "$1" --seed $2 --num-agents 4 --building-mode medium --surround-spawn \
    --target-max-speed 12 --target-min-speed 8 --closure-weight $4 --pincer-weight $5 \
    --w-pos 0.6 --w-gap 0.8 --load-from "$WARM" --policy-only-load --start-noise 0.12 --end-noise 0.01 \
    --steps 3000000 --parallels 8 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[n4p2] $1 train_exit=$?" | tee -a $SUM
}

evalk pincer_N4_s2 4 30                                  # fair number for the 73% model
train4 pincer_N4_s1 1 fs_N4_s1 8 12;  evalk pincer_N4_s1 4 30   # REPLICATION -> 2-seed mean
train4 cw16pin_N4_s2 2 fs_N4_s2 16 12; evalk cw16pin_N4_s2 4 30 # combo
echo "[n4p2] === DONE $(date) ===" | tee -a $SUM
