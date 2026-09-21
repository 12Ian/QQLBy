#!/bin/bash
# P2 CORE campaign: prove the 3 revised innovation points with 3 seeds each.
#   (1) 3D Apollonius pursuit  (2) closure reward design  (3) domain randomization
# Method = reward fix + closure, NO GAT (ablation showed GAT no benefit). Serial GPU.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
mkdir -p p2_results
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
REWARD="--target-max-speed 6 --closure-weight 8 --w-pos 0.6 --w-gap 0.8"
SUM=p2_results/core_summary.txt
echo "[core] === START $(date) ===" | tee -a $SUM

train () {  # $1=name $2=extra-train-args $3=steps
  echo "[core] TRAIN $1 ($2) $(date)" | tee -a $SUM
  python $RUN --name "$1" $REWARD --steps $3 --parallels 16 --eval-interval 250000 --test-episode 20 $2 > p2_results/train_$1.log 2>&1
}
evald () {  # $1=name $2=density
  local FM; FM=$(ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1)
  echo -n "[core] $1 @$2: " | tee -a $SUM
  python $EVAL --policy model --model-path "$FM" --building-mode $2 --target-max-speed 6 --k 30 --out p2_results/eval_${1}_$2.json 2>/dev/null | grep RESULT | tee -a $SUM
}

# ---- Exp FULL: the method (medium, closure ON). Innovation 1+2 positive; also the
#      fixed-density baseline for the domain-randomization comparison. ----
for s in 1 2 3; do
  train full_s$s "--seed $s --building-mode medium" 5000000
  for D in empty open medium complex; do evald full_s$s $D; done
done

# ---- Exp NOCLOSE: closure-design ablation (medium, closure OFF). Innovation 2 negative. ----
for s in 1 2 3; do
  train noclose_s$s "--seed $s --building-mode medium --closure-weight 0 --no-guide-collapse" 5000000
  evald noclose_s$s medium
done

# ---- Exp DOMRAND: domain randomization. Innovation 3 (cross-density robustness). ----
for s in 1 2 3; do
  train domrand_s$s "--seed $s --randomize-density empty,open,medium" 6000000
  for D in empty open medium complex; do evald domrand_s$s $D; done
done

echo "[core] === DONE $(date) ===" | tee -a $SUM
