#!/bin/bash
# N=4 FIX attempt: CURRICULUM WARMUP. Diagnosis: N=4 learns fine @empty (95%) but direct @medium
# training collapses (death spiral: can't bootstrap obstacle avoidance; guide-collapse ruled out).
# Warm-start from the @empty policy and finetune @medium. If it converges, N=4's 0% is an
# optimization artifact, not a capability limit. Auto-resumes the main campaign afterwards.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
BASE="--closure-weight 8 --w-pos 0.6 --w-gap 0.8 --target-max-speed 6 --num-agents 4"
INIT=$(ls models/maddpg/apollonius_3d_diag_n4_empty/*/final_train_model.pth 2>/dev/null | head -1)
SUM=p3_results/n4_warmup.txt
echo "[warmup] === START $(date) init=$INIT ===" | tee -a $SUM

for s in 1 2; do
  echo "[warmup] TRAIN n4_warmup_s$s $(date)" | tee -a $SUM
  python $RUN --name n4_warmup_s$s --seed $s --building-mode medium $BASE --load-from "$INIT" \
    --steps 3000000 --parallels 16 --eval-interval 250000 --test-episode 20 > p3_results/train_n4_warmup_s$s.log 2>&1
  echo "[warmup] n4_warmup_s$s train_exit=$?" | tee -a $SUM
  FM=$(ls models/maddpg/apollonius_3d_n4_warmup_s$s/*/final_train_model.pth 2>/dev/null | head -1)
  if [ -z "$FM" ]; then echo "[warmup] n4_warmup_s$s: NO_MODEL" | tee -a $SUM; continue; fi
  for D in open medium; do
    echo -n "[warmup] n4_warmup_s$s @$D: " | tee -a $SUM
    python $EVAL --policy model --model-path "$FM" --seed 7 --building-mode $D --num-agents 4 --target-max-speed 6 --k 30 --out p3_results/eval_n4_warmup_s${s}_${D}.json 2>/dev/null | grep -oE "success_rate[^,]*|collision_rate[^,]*" | tr "\n" " " | tee -a $SUM; echo "" | tee -a $SUM
  done
done

echo "[warmup] === DONE $(date); relaunching main campaign ===" | tee -a $SUM
nohup bash xuance-master/examples/resume_p3.sh > p3_results/resume2.log 2>&1 &
