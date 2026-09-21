#!/bin/bash
# STRONG-EVADER headline at N=4 (tetrahedron = minimal 3D encirclement). The N=6 core run
# sits in a high-variance transition zone (see N-sweep: N3=73% N5=60% N6=31% N8=10%, where
# N8 fails 57% by COLLISION not escape). N=4 is the principled, stable operating point.
# Re-runs the N=6-contaminated pieces (headline/density/ablation) at N=4. N-sweep + speed
# sweep stay as-is (already clean). parallels 8 (proven-stable). Prefix "s4_". flock+idempotent.
exec 9>/tmp/run_strong_n4.lock
flock -n 9 || { echo "[s4] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
BASE="--building-mode medium --closure-weight 8 --w-pos 0.6 --w-gap 0.8 --target-max-speed 10 --target-min-speed 8"
SUM=p3_results/strong_n4_summary.txt
echo "[s4] === START $(date) ===" | tee -a $SUM

modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
train_if () {  # $1=name $2=train-args
  local FREE; FREE=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "${FREE:-0}" -lt 10 ]; then echo "[s4] ABORT $1: low disk ${FREE}G free (<10G) $(date)" | tee -a $SUM; exit 1; fi
  if [ -n "$(modelpath $1)" ]; then echo "[s4] SKIP train $1" | tee -a $SUM; return; fi
  echo "[s4] TRAIN $1 ($2) $(date)" | tee -a $SUM
  python $RUN --name "$1" $2 --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[s4] $1 train_exit=$?" | tee -a $SUM
}
eval_if () {  # $1=name $2=density $3=eval-args
  if [ -f p3_results/eval_${1}_${2}.json ]; then echo "[s4] SKIP eval $1@$2" | tee -a $SUM; return; fi
  local FM; FM=$(modelpath $1)
  if [ -z "$FM" ]; then echo "[s4] $1 @$2: NO_MODEL" | tee -a $SUM; return; fi
  echo -n "[s4] $1 @$2: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --seed 7 --building-mode $2 $3 --k 15 --out p3_results/eval_${1}_${2}.json 2>/dev/null | grep RESULT | tee -a $SUM
}

EV="--num-agents 4 --target-max-speed 10 --target-min-speed 8"

# (1) headline full method at N=4, 3 seeds, eval across densities (density generalization)
for s in 1 2 3; do
  train_if s4_full_s$s "--seed $s --num-agents 4 $BASE"
  for D in empty open medium complex; do eval_if s4_full_s$s $D "$EV"; done
done

# (2) closure ablation at N=4 (guide-collapse + closure off)
for s in 1 2; do
  train_if s4_noclose_s$s "--seed $s --num-agents 4 $BASE --no-guide-collapse --closure-weight 0"
  eval_if s4_noclose_s$s medium "$EV"
done

# (3) reward-term ablation at N=4 (clean, uncontaminated)
for T in r_pos r_closure r_gap r_finish r_near r_safe; do
  for s in 1 2; do
    train_if s4_ab_${T}_s$s "--seed $s --num-agents 4 $BASE --reward-disable $T"
    eval_if s4_ab_${T}_s$s medium "$EV"
  done
done

echo "[s4] === DONE $(date) ===" | tee -a $SUM
