#!/bin/bash
# N=4 PUSH TO 80%. Evidence so far (k=30 unless noted): N=3 cw16 = 90% (target met);
# N=4 cw16 = 53%; N=4 pincer(cw8+pin12) = 73% @k=15. So at N=4 the 2nd-nearest "pincer"
# closure term is the stronger lever, not closure-doubling -> COMBINE them (cw16 + pin12),
# 2 seeds. Also re-evaluates the existing pincer model at k=30 for a fair comparison.
# Waits for the confirm campaign by BLOCKING on its lock file (no pgrep -> no self-match).
exec 8>/tmp/run_confirm.lock
flock 8                      # blocks until the confirm campaign releases the lock
exec 9>/tmp/run_n4push.lock
flock -n 9 || { echo "[n4p] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
SUM=p3_results/n4push_summary.txt
modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
echo "[n4p] === START $(date) ===" | tee -a $SUM

evalk () {  # $1=name $2=N $3=k
  local OUT=p3_results/eval_${1}_k$3.json
  [ -f "$OUT" ] && { echo "[n4p] SKIP eval $1 k=$3"|tee -a $SUM; return; }
  local FM; FM=$(modelpath $1); [ -z "$FM" ] && { echo "[n4p] $1 NO_MODEL"|tee -a $SUM; return; }
  echo -n "[n4p] $1 k=$3: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --num-agents $2 --building-mode medium \
    --target-max-speed 12 --target-min-speed 8 --surround-spawn --k $3 --seed 7 \
    --out "$OUT" 2>/dev/null | grep RESULT | tee -a $SUM
}

train_combo () {  # $1=name $2=N $3=seed $4=warm
  local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 12 ] && { echo "[n4p] ABORT low disk ${F}G"|tee -a $SUM; exit 1; }
  if [ -n "$(modelpath $1)" ]; then echo "[n4p] SKIP train $1"|tee -a $SUM; return; fi
  local WARM; WARM=$(modelpath $4)
  [ -z "$WARM" ] && { echo "[n4p] $1 NO_WARM($4)"|tee -a $SUM; return; }
  echo "[n4p] TRAIN $1 N=$2 seed=$3 warm=$4 cw16+pin12 $(date)" | tee -a $SUM
  python $RUN --name "$1" --seed $3 --num-agents $2 --building-mode medium --surround-spawn \
    --target-max-speed 12 --target-min-speed 8 --closure-weight 16 --pincer-weight 12 \
    --w-pos 0.6 --w-gap 0.8 --load-from "$WARM" --policy-only-load --start-noise 0.12 --end-noise 0.01 \
    --steps 3000000 --parallels 8 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[n4p] $1 train_exit=$?" | tee -a $SUM
}

evalk pincer_N4_s2 4 30                                   # fair k=30 number for the 73% model
train_combo cw16pin_N4_s2 4 2 fs_N4_s2; evalk cw16pin_N4_s2 4 30; evalk cw16pin_N4_s2 4 15
train_combo cw16pin_N4_s1 4 1 fs_N4_s1; evalk cw16pin_N4_s1 4 30
evalk pincer_N3_s2 3 30                                   # completeness: pincer at N=3, k=30
echo "[n4p] === DONE $(date) ===" | tee -a $SUM
