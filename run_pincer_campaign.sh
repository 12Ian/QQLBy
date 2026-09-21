#!/bin/bash
# ROOT-CAUSE FIX for the vmax12 (faster-evader) closure ceiling: a strictly-faster reactive
# evader flees the single nearest pursuer, so lone closure stalls (~53% N3 / 33% N4 despite
# 15/15 Apollonius encirclement). --pincer-weight adds a 2nd-nearest closure reward so two
# pursuers close together and the evader flees one into the other. Warm-start the best
# existing encirclers (fs_N3_s2 53%, fs_N4_s2 33%) so encirclement skill is retained and only
# the final closure is re-shaped. Waits for the probe (single-GPU handoff). flock + disk guard.
exec 9>/tmp/pincer_campaign.lock
flock -n 9 || { echo "[pin] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
SUM=p3_results/pincer_summary.txt
modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }

# Clean single-GPU handoff: wait until the probe wrapper (train+eval) exits.
while pgrep -f "run_probe_close.sh" >/dev/null 2>&1; do sleep 60; done
echo "[pin] === START $(date) (probe finished) ===" | tee -a $SUM

train_pincer () {  # $1=name $2=N $3=warm_run
  local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 12 ] && { echo "[pin] ABORT low disk ${F}G"|tee -a $SUM; exit 1; }
  if [ -n "$(modelpath $1)" ]; then echo "[pin] SKIP $1"|tee -a $SUM; return; fi
  local WARM; WARM=$(modelpath $3)
  [ -z "$WARM" ] && { echo "[pin] $1 NO_WARM($3)"|tee -a $SUM; return; }
  echo "[pin] TRAIN $1 N=$2 warm=$3 $(date)" | tee -a $SUM
  python $RUN --name "$1" --seed 2 --num-agents $2 --building-mode medium --surround-spawn \
    --target-max-speed 12 --target-min-speed 8 --closure-weight 8 --pincer-weight 12 \
    --w-pos 0.6 --w-gap 0.8 --load-from "$WARM" --policy-only-load --start-noise 0.12 --end-noise 0.01 \
    --steps 3000000 --parallels 8 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[pin] $1 train_exit=$?" | tee -a $SUM
}
eval_pincer () {  # $1=name $2=N
  [ -f p3_results/eval_${1}.json ] && { echo "[pin] SKIP eval $1"|tee -a $SUM; return; }
  local FM; FM=$(modelpath $1); [ -z "$FM" ] && { echo "[pin] $1 NO_MODEL"|tee -a $SUM; return; }
  echo -n "[pin] $1 eval: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --num-agents $2 --building-mode medium \
    --target-max-speed 12 --target-min-speed 8 --surround-spawn --k 15 --seed 7 \
    --out p3_results/eval_${1}.json 2>/dev/null | grep RESULT | tee -a $SUM
}

train_pincer pincer_N3_s2 3 fs_N3_s2; eval_pincer pincer_N3_s2 3
train_pincer pincer_N4_s2 4 fs_N4_s2; eval_pincer pincer_N4_s2 4
echo "[pin] === DONE $(date) ===" | tee -a $SUM
