#!/bin/bash
# Baseline re-run after the OOM failures. Root cause: MAPPO/MASAC at parallels 8 exceeded
# the shared box's 30G RAM (xxy co-resident) -> OOM (exit 137); MATD3 is broken in this
# XuanCe build (MATD3_Policy missing parameters_actor) -> dropped. Fix: parallels 4 + 3M
# steps. WAITS for run_ch4_all to finish (serial GPU), then trains MAPPO x2 + MASAC x1
# (MASAC may still OOM on its replay buffer; attempt & note). N=3, strong evader, same
# reward as the MADDPG flagship. Eval vs flagship 80%. flock + disk-guard + idempotent.
exec 9>/tmp/run_basefix.lock
flock -n 9 || { echo "[bf] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
E3=xuance-master/examples/evaluate_3d.py
REW="--num-agents 3 --building-mode medium --closure-weight 8 --w-pos 0.6 --w-gap 0.8 --target-max-speed 10 --target-min-speed 8"
SUM=p3_results/basefix_summary.txt
echo "[bf] === START $(date) ===" | tee -a $SUM

# --- wait for the Ch4/baseline campaign to finish (serial GPU) ---
echo "[bf] waiting for run_ch4_all to finish..." | tee -a $SUM
while pgrep -f "run_ch4_al[l]" >/dev/null; do sleep 300; done
echo "[bf] run_ch4_all done, starting baseline re-run $(date)" | tee -a $SUM

mpath () { ls models/$2/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
train_if () {  # $1=name $2=algo
  local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 10 ] && { echo "[bf] ABORT low disk"|tee -a $SUM; exit 1; }
  if [ -n "$(mpath $1 $2)" ]; then echo "[bf] SKIP $1"|tee -a $SUM; return; fi
  echo "[bf] TRAIN $1 [algo=$2] parallels=4 steps=3M $(date)" | tee -a $SUM
  python $RUN --algo $2 --name "$1" --seed ${1##*_s} $REW --steps 3000000 --parallels 4 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[bf] $1 train_exit=$?" | tee -a $SUM
}
eval_if () {  # $1=name $2=algo
  [ -f p3_results/eval_${1}_medium.json ] && { echo "[bf] SKIP eval $1"|tee -a $SUM; return; }
  local FM; FM=$(mpath $1 $2); [ -z "$FM" ] && { echo "[bf] $1: NO_MODEL"|tee -a $SUM; return; }
  echo -n "[bf] $1 eval: " | tee -a $SUM
  python $E3 --algo $2 --policy model --model-path "$FM" --num-agents 3 --building-mode medium \
    --target-max-speed 10 --target-min-speed 8 --k 15 --seed 7 --out p3_results/eval_${1}_medium.json 2>/dev/null | grep RESULT | tee -a $SUM
}

for s in 1 2; do train_if base_mappo_s$s mappo; eval_if base_mappo_s$s mappo; done
train_if base_masac_s1 masac; eval_if base_masac_s1 masac   # attempt; may OOM on replay buffer

echo "[bf] === DONE $(date) ===" | tee -a $SUM
