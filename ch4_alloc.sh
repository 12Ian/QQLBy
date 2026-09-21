#!/bin/bash
# Chapter 4's central claim is that learned dynamic allocation roughly doubles static
# assignment (published: 77% vs 40%). The verbatim re-run inverted it -- static 86.7% beat
# dynamic 63.3% -- and since the configuration is now confirmed correct (M=3 reproduced 93% as
# 86.7%), a config error can no longer explain it. One static seed cannot settle a claim this
# central, so run three seeds of each arm before concluding anything.
exec 9>/tmp/ch4_alloc.lock
flock -n 9 || { echo "[AL] already running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EMT=xuance-master/examples/evaluate_mt.py
SUM=p3_results/ch4_alloc_summary.txt
REW="--building-mode medium --closure-weight 8 --w-pos 0.6 --w-gap 0.8 --target-max-speed 10 --target-min-speed 8"
MT="--env uav_pursuit_apollonius_multitarget_3d $REW"
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }

job () {   # "name seed [extra]"
  set -- $1
  local name=$1; local sd=$2; local extra=${3:-}
  local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 10 ] && { echo "[AL] ABORT $name low disk" >> $SUM; return; }
  if [ -z "$(mp $name)" ]; then
    echo "[AL] TRAIN $name seed=$sd ${extra:-dynamic} $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --algo maddpg --name "$name" --seed $sd --num-targets 2 --num-agents 6 \
      $extra $MT --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$name.log 2>&1
    echo "[AL] $name exit=$? $(date +%m-%d\ %H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $name); [ -z "$FM" ] && { echo "[AL] $name NO_MODEL" >> $SUM; return; }
  [ -f p3_results/eval_${name}.json ] || python $EMT --model-path "$FM" \
      --num-agents 6 --num-targets 2 --building-mode medium --target-max-speed 10 \
      $extra --k 15 --seed 7 --out p3_results/eval_${name}.json > /dev/null 2>&1
  echo "[AL] $name $(grep -oE '"[a-z_]*(success|capture)[a-z_]*": [0-9.]+' p3_results/eval_${name}.json 2>/dev/null | head -2 | tr '\n' ' ')" >> $SUM
}
export -f job mp; export RUN EMT SUM MT

echo "[AL] ===== START $(date) =====" >> $SUM
printf '%s\n' \
  "ch4x_M2_s3 3" \
  "ch4x_M2static_s2 2 --no-dynamic-alloc" \
  "ch4x_M2static_s3 3 --no-dynamic-alloc" \
  | xargs -P 3 -I{} bash -c 'job "{}"'
echo "[AL] ===== DONE $(date) =====" >> $SUM
