#!/bin/bash
# Plan gaps 3.3 (GAT ablation) + 4.4 (Ch4 multi-target reward ablation), vs strong evader.
# GAT: flagship s_N3 = no-GAT baseline (reuse, 80%); add GAT-T (--use-graph-module) and
# full GAT (+--use-obstacle-gat), 2 seeds each, N=3. Ch4 reward ablation: M=2/N=6, disable
# each reward term, 1 seed. parallels 8, flock+idempotent+disk-guard. Interfaces smoke-OK.
exec 9>/tmp/run_gatch4.lock
flock -n 9 || { echo "[gc] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
E3=xuance-master/examples/evaluate_3d.py
EMT=xuance-master/examples/evaluate_mt.py
REW="--building-mode medium --closure-weight 8 --w-pos 0.6 --w-gap 0.8 --target-max-speed 10 --target-min-speed 8"
SUM=p3_results/gatch4_summary.txt
echo "[gc] === START $(date) ===" | tee -a $SUM
mpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
train_if () {  # $1=name $2=args
  local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${F:-0}" -lt 10 ] && { echo "[gc] ABORT low disk"|tee -a $SUM; exit 1; }
  if [ -n "$(mpath $1)" ]; then echo "[gc] SKIP $1"|tee -a $SUM; return; fi
  echo "[gc] TRAIN $1 ($2) $(date)" | tee -a $SUM
  python $RUN --name "$1" $2 --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[gc] $1 train_exit=$?" | tee -a $SUM
}
eval3_if () {  # $1=name
  [ -f p3_results/eval_${1}_medium.json ] && { echo "[gc] SKIP eval $1"|tee -a $SUM; return; }
  local FM; FM=$(mpath $1); [ -z "$FM" ] && { echo "[gc] $1 NO_MODEL"|tee -a $SUM; return; }
  echo -n "[gc] $1 eval: " | tee -a $SUM
  python $E3 --policy model --model-path "$FM" --num-agents 3 --building-mode medium --target-max-speed 10 --target-min-speed 8 --k 15 --seed 7 --out p3_results/eval_${1}_medium.json 2>/dev/null | grep RESULT | tee -a $SUM
}
evalmt_if () {  # $1=name
  [ -f p3_results/eval_${1}.json ] && { echo "[gc] SKIP eval $1"|tee -a $SUM; return; }
  local FM; FM=$(mpath $1); [ -z "$FM" ] && { echo "[gc] $1 NO_MODEL"|tee -a $SUM; return; }
  echo -n "[gc] $1 mt-eval: " | tee -a $SUM
  python $EMT --model-path "$FM" --num-agents 6 --num-targets 2 --building-mode medium --target-max-speed 10 --k 15 --seed 7 --out p3_results/eval_${1}.json 2>/dev/null | grep RESULT | tee -a $SUM
}

# (3.3) GAT ablation at N=3 (no-GAT = flagship s_N3, reuse)
for s in 1 2; do
  train_if s3_gat_t_s$s "--seed $s --num-agents 3 $REW --use-graph-module"; eval3_if s3_gat_t_s$s
  train_if s3_gat_full_s$s "--seed $s --num-agents 3 $REW --use-obstacle-gat --use-graph-module"; eval3_if s3_gat_full_s$s
done

# (4.4) Ch4 M=2 reward-term ablation, 1 seed
for T in r_pos r_closure r_gap r_finish r_near r_safe; do
  train_if ch4abl_${T}_s1 "--seed 1 --env uav_pursuit_apollonius_multitarget_3d --num-targets 2 --num-agents 6 $REW --reward-disable $T"
  evalmt_if ch4abl_${T}_s1
done

echo "[gc] === DONE $(date) ===" | tee -a $SUM
