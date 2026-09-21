#!/bin/bash
# SETTLE N=3 >=80% honestly. cw16 lifts every seed (+26..+37pp) but the ABSOLUTE level tracks
# the warm-start base (s2 53->90, s1 27->53) => mean 72%, short of 80%. Two questions, two arms:
#  A) cw32_N3_s1  : is closure still under-weighted? Push cw 16->32 on the WEAK seed (s1).
#                   If it lifts 53% substantially, closure weight is still the binding lever.
#  B) scr_N3_s3   : FROM SCRATCH 5M at cw16 (no warm-start) -> a clean, protocol-matched
#                   3rd seed, free of the "lucky base" confound, directly comparable to the
#                   original 5M baselines. This is what the report headline should rest on.
# Waits for the N=4 push by BLOCKING on its lock (no pgrep self-match).
exec 8>/tmp/run_n4push.lock
flock 8
exec 9>/tmp/run_n3settle.lock
flock -n 9 || { echo "[n3s] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
SUM=p3_results/n3settle_summary.txt
modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
echo "[n3s] === START $(date) ===" | tee -a $SUM

evalk () {  # $1=name $2=N $3=k
  local OUT=p3_results/eval_${1}_k$3.json
  [ -f "$OUT" ] && { echo "[n3s] SKIP eval $1 k=$3"|tee -a $SUM; return; }
  local FM; FM=$(modelpath $1); [ -z "$FM" ] && { echo "[n3s] $1 NO_MODEL"|tee -a $SUM; return; }
  echo -n "[n3s] $1 k=$3: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --num-agents $2 --building-mode medium \
    --target-max-speed 12 --target-min-speed 8 --surround-spawn --k $3 --seed 7 \
    --out "$OUT" 2>/dev/null | grep RESULT | tee -a $SUM
}
guard () { local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 12 ] && { echo "[n3s] ABORT low disk ${F}G"|tee -a $SUM; exit 1; }; return 0; }

# A) is closure STILL under-weighted? cw32 on the weak seed, warm-started like before.
if [ -z "$(modelpath cw32_N3_s1)" ]; then
  guard; WARM=$(modelpath fs_N3_s1)
  echo "[n3s] TRAIN cw32_N3_s1 cw=32 warm=fs_N3_s1 $(date)" | tee -a $SUM
  python $RUN --name cw32_N3_s1 --seed 1 --num-agents 3 --building-mode medium --surround-spawn \
    --target-max-speed 12 --target-min-speed 8 --closure-weight 32 --w-pos 0.6 --w-gap 0.8 \
    --load-from "$WARM" --policy-only-load --start-noise 0.12 --end-noise 0.01 \
    --steps 3000000 --parallels 8 --eval-interval 250000 --test-episode 20 > p3_results/train_cw32_N3_s1.log 2>&1
  echo "[n3s] cw32_N3_s1 train_exit=$?" | tee -a $SUM
fi
evalk cw32_N3_s1 3 30

# B) clean from-scratch 5M seed at cw16 (no warm-start confound) - the honest headline seed.
if [ -z "$(modelpath scr_N3_s3)" ]; then
  guard
  echo "[n3s] TRAIN scr_N3_s3 from-scratch 5M cw16 $(date)" | tee -a $SUM
  python $RUN --name scr_N3_s3 --seed 3 --num-agents 3 --building-mode medium --surround-spawn \
    --target-max-speed 12 --target-min-speed 8 --closure-weight 16 --w-pos 0.6 --w-gap 0.8 \
    --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 > p3_results/train_scr_N3_s3.log 2>&1
  echo "[n3s] scr_N3_s3 train_exit=$?" | tee -a $SUM
fi
evalk scr_N3_s3 3 30
echo "[n3s] === DONE $(date) ===" | tee -a $SUM
