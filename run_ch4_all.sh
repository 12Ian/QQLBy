#!/bin/bash
# Ch4 multi-target + learning baselines vs the STRONG evader (vmax10, N-matched to flagship).
# All interfaces smoke-validated 2026-07-10 (train+eval end-to-end OK for mappo & multitarget).
# Order: (A) baselines [mappo/masac/matd3 x2 seeds, single-target N=3] -> (B) Ch4 multitarget
# [M2/N6 x2, M3/N9, M2-static ablation] -> (C) N=3 speed-sweep tail. parallels 8, flock,
# idempotent, disk-guard. Compare baselines to the MADDPG N=3 flagship (80% @medium).
exec 9>/tmp/run_ch4all.lock
flock -n 9 || { echo "[cb] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
E3=xuance-master/examples/evaluate_3d.py
EMT=xuance-master/examples/evaluate_mt.py
REW="--building-mode medium --closure-weight 8 --w-pos 0.6 --w-gap 0.8 --target-max-speed 10 --target-min-speed 8"
SUM=p3_results/ch4base_summary.txt
echo "[cb] === START $(date) ===" | tee -a $SUM

diskguard () { local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 10 ] && { echo "[cb] ABORT low disk ${F}G" | tee -a $SUM; exit 1; }; return 0; }
mpath () { ls models/${2:-maddpg}/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }

train_if () {  # $1=name $2=algo $3=train-args
  diskguard
  if [ -n "$(mpath $1 $2)" ]; then echo "[cb] SKIP train $1" | tee -a $SUM; return; fi
  echo "[cb] TRAIN $1 [algo=$2] $(date)" | tee -a $SUM
  python $RUN --algo $2 --name "$1" $3 --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[cb] $1 train_exit=$?" | tee -a $SUM
}
eval3_if () {  # $1=name $2=algo
  [ -f p3_results/eval_${1}_medium.json ] && { echo "[cb] SKIP eval $1"|tee -a $SUM; return; }
  local FM; FM=$(mpath $1 $2); [ -z "$FM" ] && { echo "[cb] $1: NO_MODEL"|tee -a $SUM; return; }
  echo -n "[cb] $1 eval: " | tee -a $SUM
  python $E3 --algo $2 --policy model --model-path "$FM" --num-agents 3 --building-mode medium \
    --target-max-speed 10 --target-min-speed 8 --k 15 --seed 7 --out p3_results/eval_${1}_medium.json 2>/dev/null | grep RESULT | tee -a $SUM
}
evalmt_if () {  # $1=name $2=M $3=N $4=extra(eval)
  [ -f p3_results/eval_${1}.json ] && { echo "[cb] SKIP eval $1"|tee -a $SUM; return; }
  local FM; FM=$(mpath $1 maddpg); [ -z "$FM" ] && { echo "[cb] $1: NO_MODEL"|tee -a $SUM; return; }
  echo -n "[cb] $1 mt-eval: " | tee -a $SUM
  python $EMT --model-path "$FM" --num-agents $3 --num-targets $2 --building-mode medium \
    --target-max-speed 10 $4 --k 15 --seed 7 --out p3_results/eval_${1}.json 2>/dev/null | grep RESULT | tee -a $SUM
}

# ---------- (A) Learning baselines: single-target, N=3, strong evader ----------
for algo in mappo masac matd3; do
  for s in 1 2; do
    train_if base_${algo}_s$s $algo "--seed $s --num-agents 3 $REW"
    eval3_if base_${algo}_s$s $algo
  done
done

# ---------- (B) Ch4 multi-target vs strong evader (dynamic allocation) ----------
MT="--env uav_pursuit_apollonius_multitarget_3d $REW"
train_if ch4_M2_s1 maddpg "--seed 1 --num-targets 2 --num-agents 6 $MT"; evalmt_if ch4_M2_s1 2 6 ""
train_if ch4_M2_s2 maddpg "--seed 2 --num-targets 2 --num-agents 6 $MT"; evalmt_if ch4_M2_s2 2 6 ""
train_if ch4_M3_s1 maddpg "--seed 1 --num-targets 3 --num-agents 9 $MT"; evalmt_if ch4_M3_s1 3 9 ""
# allocation ablation: static (no dynamic alloc) vs dynamic above
train_if ch4_M2_static_s1 maddpg "--seed 1 --num-targets 2 --num-agents 6 --no-dynamic-alloc $MT"
evalmt_if ch4_M2_static_s1 2 6 "--no-dynamic-alloc"

# ---------- (C) N=3 speed-sweep tail (lowest priority; V10=flagship already) ----------
for V in 8 9 11 12; do
  train_if s3_lam_V${V} maddpg "--seed 1 --num-agents 3 --building-mode medium --closure-weight 8 --w-pos 0.6 --w-gap 0.8 --target-max-speed $V --target-min-speed 8"
  [ -f p3_results/eval_s3_lam_V${V}_medium.json ] || { FM=$(mpath s3_lam_V${V} maddpg); [ -n "$FM" ] && { echo -n "[cb] s3_lam_V${V} eval: "|tee -a $SUM; python $E3 --policy model --model-path "$FM" --num-agents 3 --building-mode medium --target-max-speed $V --target-min-speed 8 --k 15 --seed 7 --out p3_results/eval_s3_lam_V${V}_medium.json 2>/dev/null | grep RESULT|tee -a $SUM; }; }
done

echo "[cb] === DONE $(date) ===" | tee -a $SUM
