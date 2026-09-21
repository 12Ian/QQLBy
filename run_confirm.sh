#!/bin/bash
# CONFIRM + EXTEND the breakthrough. probe_close_N3_s2 (warm-start + closure-weight 16)
# hit 100% vs the FASTER evader (vmax12) where fs_N3_s2 was 53% -> the sub-80% ceiling was
# an under-weighted CLOSURE incentive, not a capability limit. Now: (a) tighten the CI on
# that model with k=30, (b) replicate on a 2nd seed, (c) apply the SAME recipe to N=4
# (baseline 33%; pincer got 73%), (d) N=4 second seed. Evals at k=30 (tight CI) AND k=15
# (directly comparable to the existing fs_/sc_ tables). flock + disk guard.
exec 9>/tmp/run_confirm.lock
flock -n 9 || { echo "[cf] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
SUM=p3_results/confirm_summary.txt
modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
echo "[cf] === START $(date) ===" | tee -a $SUM

evalk () {  # $1=name $2=N $3=k
  local OUT=p3_results/eval_${1}_k$3.json
  [ -f "$OUT" ] && { echo "[cf] SKIP eval $1 k=$3"|tee -a $SUM; return; }
  local FM; FM=$(modelpath $1); [ -z "$FM" ] && { echo "[cf] $1 NO_MODEL"|tee -a $SUM; return; }
  echo -n "[cf] $1 k=$3: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --num-agents $2 --building-mode medium \
    --target-max-speed 12 --target-min-speed 8 --surround-spawn --k $3 --seed 7 \
    --out "$OUT" 2>/dev/null | grep RESULT | tee -a $SUM
}

train_cw16 () {  # $1=name $2=N $3=seed $4=warm_run
  local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 12 ] && { echo "[cf] ABORT low disk ${F}G"|tee -a $SUM; exit 1; }
  if [ -n "$(modelpath $1)" ]; then echo "[cf] SKIP train $1"|tee -a $SUM; return; fi
  local WARM; WARM=$(modelpath $4)
  [ -z "$WARM" ] && { echo "[cf] $1 NO_WARM($4)"|tee -a $SUM; return; }
  echo "[cf] TRAIN $1 N=$2 seed=$3 warm=$4 $(date)" | tee -a $SUM
  python $RUN --name "$1" --seed $3 --num-agents $2 --building-mode medium --surround-spawn \
    --target-max-speed 12 --target-min-speed 8 --closure-weight 16 --w-pos 0.6 --w-gap 0.8 \
    --load-from "$WARM" --policy-only-load --start-noise 0.12 --end-noise 0.01 \
    --steps 3000000 --parallels 8 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[cf] $1 train_exit=$?" | tee -a $SUM
}

# (a) tighten CI on the 100% model (no retrain)
evalk probe_close_N3_s2 3 30
# (c) winning recipe on N=4 (the headline N=4 fix)
train_cw16 cw16_N4_s2 4 2 fs_N4_s2; evalk cw16_N4_s2 4 30; evalk cw16_N4_s2 4 15
# (b) N=3 replication on a different seed
train_cw16 cw16_N3_s1 3 1 fs_N3_s1; evalk cw16_N3_s1 3 30; evalk cw16_N3_s1 3 15
# (d) N=4 second seed
train_cw16 cw16_N4_s1 4 1 fs_N4_s1; evalk cw16_N4_s1 4 30; evalk cw16_N4_s1 4 15
echo "[cf] === DONE $(date) ===" | tee -a $SUM
