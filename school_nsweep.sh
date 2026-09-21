#!/bin/bash
# Unified N-sweep for N>=5, to remove the configuration confound.
#
# The published N curve is not apples-to-apples: N=3/4 ran cw16 + separation 4/20 while
# N=5/6/8 ran the layered recipe (close_k 4, obs_avoid 2, separation 2/15), which we separately
# measured to be worse. So "capability falls with N" cannot currently be separated from "N>=5
# was trained differently". This re-runs N=5/6/8 with the SAME recipe as N=4.
#
# It matters because the conditional metric P(capture | no collision) already shows N=3 -> 4
# rising (97.5% -> 98.4%), i.e. the raw drop there is collision accounting, not coordination.
# Whether that also holds for N>=5 is exactly what the confound currently hides.
#
# Waits on the Apollonius-arm lock, which that script holds while running, so this starts only
# after it finishes -- no contention on a box that is already at 20/20 load and 11 GB free.
exec 8>/tmp/school_apofix.lock
flock 8              # blocks until the D-arm campaign releases it
exec 8>&-
exec 9>/tmp/school_nsweep.lock
flock -n 9 || { echo "[NS] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/nsweep_unified_summary.txt
RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
diskfree () { df --output=avail -BG / | tail -1 | tr -dc 0-9; }

echo "[NS] ===== START $(date) free=$(diskfree)G =====" >> $SUM
# reclaim space: oversized training logs and tensorboard dirs of finished runs
find p3_results -name "train_*.log" -size +150M -delete 2>/dev/null
rm -rf logs/maddpg/apollonius_3d_apo_M2_s* logs/maddpg/apollonius_3d_lay_N* 2>/dev/null
echo "[NS] after cleanup free=$(diskfree)G" >> $SUM

job () {   # "name N seed"
  set -- $1
  local n=$1; local nag=$2; local sd=$3
  [ "$(diskfree)" -lt 6 ] && { echo "[NS] ABORT $n low disk" >> $SUM; return; }
  if [ -z "$(mp $n)" ]; then
    echo "[NS] TRAIN $n N=$nag seed=$sd (unified: cw16 pin12 sep4/20) $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed $sd --num-agents $nag --building-mode medium \
      --target-max-speed 10 --target-min-speed 8 --closure-weight 16 --pincer-weight 12 \
      --separation-weight 4 --separation-distance 20 --w-pos 0.6 --w-gap 0.8 \
      --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
    echo "[NS] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[NS] $n NO_MODEL" >> $SUM; return; }
  [ -f p3_results/eval_${n}_k30.json ] || python $EVAL --policy model --model-path "$FM" \
     --num-agents $nag --building-mode medium --target-max-speed 10 --target-min-speed 8 \
     --k 30 --seed 7 --out p3_results/eval_${n}_k30.json > /dev/null 2>&1
  echo "[NS] $n $(grep -oE '"(success_rate|collision_rate|timeout_rate)": [0-9.]+' p3_results/eval_${n}_k30.json 2>/dev/null | tr '\n' ' ')" >> $SUM
}
export -f job mp diskfree; export RUN EVAL SUM
printf '%s\n' "uni_N5_s1 5 1" "uni_N5_s2 5 2" "uni_N6_s1 6 1" \
              "uni_N6_s2 6 2" "uni_N8_s1 8 1" "uni_N8_s2 8 2" \
  | xargs -P 3 -I{} bash -c 'job "{}"'
echo "[NS] ===== DONE $(date) =====" >> $SUM
