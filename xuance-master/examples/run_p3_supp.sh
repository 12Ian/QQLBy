#!/bin/bash
# P3 SUPPLEMENT campaign (Chapter 3) -- reviewed & hardened (3-lens adversarial review).
#   (B) success vs N   (C) success vs speed-ratio lambda (constant-speed evader)   (A) reward-term ablation
# Base = P2 full method (reward fix + closure, NO GAT), @medium, 5M steps, 2 seeds each.
# Hardening applied: guard empty model path (no silent fallback-to-default); seed eval RNG
# (--seed 7) for paired/reproducible episode draws; re-evaluate P2 anchors seeded; constant-speed
# evader for a clean lambda axis; sweep-outer/seed-inner so an early stop yields complete points;
# log train exit codes. r_finish/r_closure are now cleanly separable (env gate split).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
mkdir -p p3_results
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
BASE="--closure-weight 8 --w-pos 0.6 --w-gap 0.8"
SPEED6="--target-max-speed 6"
ESEED="--seed 7"
SUM=p3_results/supp_summary.txt
echo "[p3] === START $(date) ===" | tee -a $SUM

train () {  # $1=name  $2=extra-train-args  $3=steps
  echo "[p3] TRAIN $1 ($2) $(date)" | tee -a $SUM
  python $RUN --name "$1" $2 --steps $3 --parallels 16 --eval-interval 250000 --test-episode 20 > p3_results/train_$1.log 2>&1
  echo "[p3] $1 train_exit=$?" | tee -a $SUM
}
evald () {  # $1=name  $2=density  $3=extra-eval-args
  local FM; FM=$(ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1)
  if [ -z "$FM" ]; then echo "[p3] $1 @$2: NO_MODEL (train failed)" | tee -a $SUM; return; fi
  echo -n "[p3] $1 @$2: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" $ESEED --building-mode $2 $3 --k 30 --out p3_results/eval_${1}_${2}.json 2>/dev/null | grep RESULT | tee -a $SUM
}
evalp2 () {  # re-evaluate a P2 anchor model under the SEEDED eval. $1=p2name $2=density $3=extra
  local FM; FM=$(ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1)
  if [ -z "$FM" ]; then echo "[p3] anchor $1 @$2: NO_MODEL" | tee -a $SUM; return; fi
  echo -n "[p3] anchor $1 @$2: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" $ESEED --building-mode $2 $3 --k 30 --out p3_results/eval_anchor_${1}_${2}.json 2>/dev/null | grep RESULT | tee -a $SUM
}

# ---- Anchors: re-evaluate P2 full (N=6, lambda-cap 6) under the SEEDED eval so the N=6 point
#      of Exp B and the baseline of Exp A are PAIRED (same episode draws) with the new points. ----
for s in 1 2 3; do
  evalp2 full_s$s medium "$SPEED6"
  evalp2 full_s$s open "$SPEED6"
done

# ---- Exp B: success vs number of pursuers N (vary --num-agents; N=6 = full anchor above). ----
#      Probes the minimum team size for 3D solid-angle closure. Lead with the validated N=4.
for N in 4 5 3 8; do
  for s in 1 2; do
    train nsweep_N${N}_s$s "--seed $s --building-mode medium $BASE $SPEED6 --num-agents $N" 5000000
    for D in open medium; do evald nsweep_N${N}_s$s $D "$SPEED6 --num-agents $N"; done
  done
done

# ---- Exp C: success vs speed ratio lambda. CONSTANT-speed evader (min=max=V) so every point
#      has identical dynamics and lambda = V / pursuer-mean(~10) is well-defined. Pursuers are
#      9-11, so V=10 is hard-but-feasible (10% margin), not strict speed parity. ----
for V in 6 8 4 10; do
  for s in 1 2; do
    train lsweep_V${V}_s$s "--seed $s --building-mode medium $BASE --target-min-speed $V --target-max-speed $V" 5000000
    evald lsweep_V${V}_s$s medium "--target-min-speed $V --target-max-speed $V"
  done
done

# ---- Exp A: reward-term ablation @medium (base + disable one additive term). N=6, lambda~0.6.
#      Six clean cells: r_pos, r_closure, r_gap, r_finish, r_near, r_safe (closure/finish split). ----
for T in r_pos r_closure r_gap r_finish r_near r_safe; do
  for s in 1 2; do
    train rewabl_${T}_s$s "--seed $s --building-mode medium $BASE $SPEED6 --reward-disable $T" 5000000
    evald rewabl_${T}_s$s medium "$SPEED6"
  done
done

echo "[p3] === DONE $(date) ===" | tee -a $SUM
