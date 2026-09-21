#!/bin/bash
# STRONG-EVADER full re-run (Ch3 equivalent). Evader now genuinely evades: wide sensing
# (250m), strong repulsion (coeff 40), no artificial centering. Headline speed vmax 10
# (lambda ~= 0.9, 65% success). All runs prefixed "s_". Idempotent + flock. parallels 16
# Slow strong-evader evals use k=15. IMPORTANT: parallels 8 (NOT 16) - against the harder
# strong evader, parallels 16 destabilises MADDPG training (s_full@16 got 7% vs sv_p10@8 = 80%,
# same seed/config). parallels 8 is the proven-stable setting.
exec 9>/tmp/run_strong.lock
flock -n 9 || { echo "[s] locked"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
BASE="--building-mode medium --closure-weight 8 --w-pos 0.6 --w-gap 0.8 --target-max-speed 10 --target-min-speed 8"
SUM=p3_results/strong_summary.txt
echo "[s] === START $(date) ===" | tee -a $SUM

modelpath () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
train_if () {  # $1=name $2=train-args
  if [ -n "$(modelpath $1)" ]; then echo "[s] SKIP train $1" | tee -a $SUM; return; fi
  echo "[s] TRAIN $1 ($2) $(date)" | tee -a $SUM
  python $RUN --name "$1" $2 --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[s] $1 train_exit=$?" | tee -a $SUM
}
eval_if () {  # $1=name $2=density $3=eval-args
  if [ -f p3_results/eval_${1}_${2}.json ]; then echo "[s] SKIP eval $1@$2" | tee -a $SUM; return; fi
  local FM; FM=$(modelpath $1)
  if [ -z "$FM" ]; then echo "[s] $1 @$2: NO_MODEL" | tee -a $SUM; return; fi
  echo -n "[s] $1 @$2: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --seed 7 --building-mode $2 $3 --k 15 --out p3_results/eval_${1}_${2}.json 2>/dev/null | grep RESULT | tee -a $SUM
}

EV="--num-agents 6 --target-max-speed 10 --target-min-speed 8"

# (1) headline full method, 3 seeds, N=6, eval across densities (density generalization)
for s in 1 2 3; do
  train_if s_full_s$s "--seed $s --num-agents 6 $BASE"
  for D in empty open medium complex; do eval_if s_full_s$s $D "$EV"; done
done

# (2) N-sweep (N=6 = full above)
for N in 3 5 8; do
  for s in 1 2; do
    train_if s_N${N}_s$s "--seed $s --num-agents $N $BASE"
    eval_if s_N${N}_s$s medium "--num-agents $N --target-max-speed 10 --target-min-speed 8"
  done
done

# (3) closure ablation (guide-collapse + closure off)
for s in 1 2; do
  train_if s_noclose_s$s "--seed $s --num-agents 6 $BASE --no-guide-collapse --closure-weight 0"
  eval_if s_noclose_s$s medium "$EV"
done

# (4) reward-term ablation
for T in r_pos r_closure r_gap r_finish r_near r_safe; do
  for s in 1 2; do
    train_if s_ab_${T}_s$s "--seed $s --num-agents 6 $BASE --reward-disable $T"
    eval_if s_ab_${T}_s$s medium "$EV"
  done
done

# (5) evader-speed sweep 2nd seed (seed 1 = sv_p8..12 already run)
for V in 8 9 10 11 12; do
  train_if s_lam_V${V}_s2 "--seed 2 --num-agents 6 --building-mode medium --closure-weight 8 --w-pos 0.6 --w-gap 0.8 --target-max-speed $V --target-min-speed 8"
  eval_if s_lam_V${V}_s2 medium "--num-agents 6 --target-max-speed $V --target-min-speed 8"
done

echo "[s] === DONE $(date) ===" | tee -a $SUM
