#!/bin/bash
# Chapter 4 on the school box, replicated VERBATIM from run_ch4_all.sh -- the script the
# published table actually came from. Three earlier attempts silently used a sibling script's
# parameters (evader vmax 6 not 10, default reward weights instead of --w-pos 0.6 --w-gap 0.8,
# six pursuers for M=3 instead of nine), so those numbers were never comparable. Device is left
# unset so it defaults to cuda:0 exactly as the original runs did.
exec 9>/tmp/school_ch4.lock
flock -n 9 || { echo "[SC] already running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EMT=xuance-master/examples/evaluate_mt.py
SUM=p3_results/school_ch4_summary.txt
REW="--building-mode medium --closure-weight 8 --w-pos 0.6 --w-gap 0.8 --target-max-speed 10 --target-min-speed 8"
MT="--env uav_pursuit_apollonius_multitarget_3d $REW"
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
guard () { local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 10 ] && return 1; return 0; }

job () {   # "name M N [extra]"
  set -- $1
  local name=$1; local M=$2; local N=$3; local extra=${4:-}
  guard || { echo "[SC] ABORT $name low disk" >> $SUM; return; }
  if [ -z "$(mp $name)" ]; then
    echo "[SC] TRAIN $name M=$M N=$N $extra $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --algo maddpg --name "$name" \
      --seed ${name##*_s} --num-targets $M --num-agents $N $extra $MT \
      --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$name.log 2>&1
    echo "[SC] $name exit=$? $(date +%m-%d\ %H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $name); [ -z "$FM" ] && { echo "[SC] $name NO_MODEL" >> $SUM; return; }
  [ -f p3_results/eval_${name}.json ] || python $EMT --model-path "$FM" \
      --num-agents $N --num-targets $M --building-mode medium --target-max-speed 10 \
      $extra --k 15 --seed 7 --out p3_results/eval_${name}.json > /dev/null 2>&1
  echo "[SC] $name $(grep -oE '"[a-z_]*(success|capture)[a-z_]*": [0-9.]+' p3_results/eval_${name}.json 2>/dev/null | head -2 | tr '\n' ' ')" >> $SUM
}
export -f job mp guard; export RUN EMT SUM MT

echo "[SC] ===== START $(date) =====" >> $SUM
printf '%s\n' \
  "ch4x_M2_s1 2 6" "ch4x_M2_s2 2 6" \
  "ch4x_M3_s1 3 9" \
  "ch4x_M2static_s1 2 6 --no-dynamic-alloc" \
  | xargs -P 4 -I{} bash -c 'job "{}"'
echo "[SC] ===== DONE $(date) =====" >> $SUM
