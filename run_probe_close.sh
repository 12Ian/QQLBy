#!/bin/bash
# PROBE (no code change): does aggressive closure-focused CONTINUED training on the best
# existing vmax12 encircler (fs_N3_s2, 53%) break the closure ceiling? Warm-start its policy,
# double the closure weight (8->16), low restart noise, +3M steps. Same surround-spawn/env as
# fs_N3 so the number is directly comparable to its 53%. Control signal for the reward fix.
exec 9>/tmp/probe_close.lock
flock -n 9 || { echo "[probe] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
SUM=p3_results/probe_close_summary.txt
modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 12 ] && { echo "[probe] ABORT low disk ${F}G" | tee -a $SUM; exit 1; }
BASE=$(modelpath fs_N3_s2)
[ -z "$BASE" ] && { echo "[probe] NO base model fs_N3_s2" | tee -a $SUM; exit 1; }
echo "[probe] === START $(date) warm=$BASE ===" | tee -a $SUM
LEVERS="--num-agents 3 --building-mode medium --surround-spawn --target-max-speed 12 --target-min-speed 8 --closure-weight 16 --w-pos 0.6 --w-gap 0.8"
python $RUN --name probe_close_N3_s2 --seed 2 $LEVERS \
  --load-from "$BASE" --policy-only-load --start-noise 0.12 --end-noise 0.01 \
  --steps 3000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
  > p3_results/train_probe_close_N3_s2.log 2>&1
echo "[probe] train_exit=$?" | tee -a $SUM
FM=$(modelpath probe_close_N3_s2)
if [ -n "$FM" ]; then
  echo -n "[probe] eval: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --num-agents 3 --building-mode medium \
    --target-max-speed 12 --target-min-speed 8 --surround-spawn --k 15 --seed 7 \
    --out p3_results/eval_probe_close_N3_s2.json 2>/dev/null | grep RESULT | tee -a $SUM
else
  echo "[probe] NO_MODEL after train" | tee -a $SUM
fi
echo "[probe] === DONE $(date) ===" | tee -a $SUM
