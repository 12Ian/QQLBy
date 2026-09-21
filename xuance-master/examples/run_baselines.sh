#!/bin/bash
# Learning-based BASELINE comparison: MAPPO (on-policy) + MASAC (SAC family) vs our MADDPG,
# on the SAME 3D env, SAME reward (closure design), N=6, @medium, matched [512,512] capacity,
# 3 seeds each. (MATD3 dropped: XuanCe MATD3_Policy missing parameters_actor. FACMAC/HASAC
# not in this XuanCe build.) Idempotent + flock-guarded. Waits for add_seed3 to free the GPU.
exec 9>/tmp/run_baselines.lock
flock -n 9 || { echo "[bl] locked; exit"; exit 0; }
# Server is now free (co-resident user gone); MAPPO/MASAC do NOT leak (60-min watch: stable
# ~5GB). Full scope, run concurrently with Ch4 (20 cores, plenty RAM).

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
BASE="--num-agents 6 --building-mode medium --closure-weight 8 --w-pos 0.6 --w-gap 0.8 --target-max-speed 6"
SUM=p3_results/baselines_summary.txt
echo "[bl] === START $(date) ===" | tee -a $SUM

modelpath () { ls models/$1/apollonius_3d_$2/*/final_train_model.pth 2>/dev/null | head -1; }  # $1=algo $2=name
train_if () {  # $1=algo $2=name $3=extra
  if [ -n "$(modelpath $1 $2)" ]; then echo "[bl] SKIP train $2" | tee -a $SUM; return; fi
  # Reduced (2M steps, parallels 2) + retry: MAPPO/MASAC leak memory on this shared box and
  # get OOM-killed (train_exit=137) mid-run. Smaller footprint + shorter horizon + up to 3
  # attempts give a fair shot at a completed baseline during a low-contention window.
  local attempt
  for attempt in 1 2; do
    [ -n "$(modelpath $1 $2)" ] && break
    echo "[bl] TRAIN $2 attempt $attempt ($3) $(date)" | tee -a $SUM
    python $RUN --algo $1 --name "$2" $3 --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 > p3_results/train_$2.log 2>&1
    echo "[bl] $2 attempt $attempt exit=$?" | tee -a $SUM
  done
}
eval_if () {  # $1=algo $2=name $3=density
  if [ -f p3_results/eval_${2}_${3}.json ]; then echo "[bl] SKIP eval $2@$3" | tee -a $SUM; return; fi
  local FM; FM=$(modelpath $1 $2)
  if [ -z "$FM" ]; then echo "[bl] $2 @$3: NO_MODEL" | tee -a $SUM; return; fi
  echo -n "[bl] $2 @$3: " | tee -a $SUM
  python $EVAL --algo $1 --policy model --model-path "$FM" --seed 7 --building-mode $3 --num-agents 6 --target-max-speed 6 --k 30 --out p3_results/eval_${2}_${3}.json 2>/dev/null | grep RESULT | tee -a $SUM
}

for A in mappo masac; do
  for s in 1 2; do
    train_if $A ${A}_s$s "--seed $s $BASE"
    for D in medium open; do eval_if $A ${A}_s$s $D; done
  done
done

echo "[bl] === DONE $(date) ===" | tee -a $SUM
