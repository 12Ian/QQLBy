#!/bin/bash
# Find the hardest setting that still works once the walls stop helping.
#
# With the centre-pulled evader, ratio 1.00 trains to 0.0% at both N=3 and N=4, ~100% timeout --
# not crashes, they simply never close. And the sub-parity results turn out to have been
# wall-dependent too: at ratio 0.91 the old runs captured 97-100% of the time within 50 m of a
# face, median 10 m. So there is currently no measured setting where this method captures in
# open air, and jumping straight to the hardest one taught us nothing about whether the method
# works at all.
#
# Two knobs, varied one at a time so the answer is interpretable:
#   speed ratio -- 0.82 and 0.91, where each pursuer's Apollonius set is a bounded sphere and
#     the escape set is bounded without any help from the arena;
#   spawn radius -- 200 m is a long way to close at a small speed edge; 80 m puts the endgame
#     inside reach so we learn whether the closing behaviour is learnable at all.
#
# If the easy corner also returns 0.0%, the problem is the method, not the difficulty, and no
# amount of further compute is the answer.
exec 9>/tmp/school_r10.lock
flock -n 9 || { echo "[R10] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/r10_summary.txt
RUN=xuance-master/examples/run_experiment.py
EV3=xuance-master/examples/evaluate_3d.py
PCG=xuance-master/examples/probe_capture_geometry.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
sr () { grep -oE '"(success_rate|collision_rate|timeout_rate)": [0-9.]+' "$1" 2>/dev/null | tr '\n' ' '; }
echo "[R10] ===== START $(date) =====" >> $SUM
find p3_results -name "train_cp_*v12*.log" -delete 2>/dev/null
rm -rf logs/maddpg/apollonius_3d_cp_N*_v12_* models/maddpg/apollonius_3d_cp_N*_v12_* 2>/dev/null

job () {
  set -- $1
  local n=$1; local nag=$2; local v=$3; local rad=$4
  if [ -z "$(mp $n)" ]; then
    echo "[R10] TRAIN $n N=$nag vmax=$v radius=$rad $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed 1 --num-agents $nag --building-mode medium \
      --target-max-speed $v --target-min-speed 8 --evader-center-pull 1.0 \
      --closure-weight 16 --pincer-weight 12 --separation-weight 4 --separation-distance 20 \
      --w-pos 0.6 --w-gap 0.8 --surround-spawn --spawn-radius $rad --cbf-safety \
      --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
    echo "[R10] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[R10] $n NO_MODEL" >> $SUM; return; }
  OUT=p3_results/eval_${n}_k30.json
  [ -f "$OUT" ] || python $EV3 --policy model --model-path "$FM" --num-agents $nag \
      --building-mode medium --target-max-speed $v --target-min-speed 8 \
      --evader-center-pull 1.0 --surround-spawn --spawn-radius $rad --cbf-safety \
      --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  echo "[R10] $n N=$nag v=$v r=$rad $(sr $OUT)" >> $SUM
  G=p3_results/capgeo_r10_${n}.json
  [ -f "$G" ] || python $PCG --model-path "$FM" --num-agents $nag --building-mode medium \
      --target-max-speed $v --target-min-speed 8 --evader-center-pull 1.0 \
      --surround-spawn --cbf-safety --k 40 --seed 7 --out "$G" > /dev/null 2>&1
  echo "[R10] $n GEO boundary_median=$(grep -oE '"any_boundary_dist": \{"min": [0-9.]+, "median": [0-9.]+' $G 2>/dev/null | grep -oE '[0-9.]+$')" >> $SUM
}
export -f job mp sr; export RUN EV3 PCG SUM
# easiest corner first: clear speed edge AND a short closing distance
printf '%s\n' "ez_N3_v9_r80 3 9 80" "ez_N4_v9_r80 4 9 80" "ez_N3_v10_r80 3 10 80" "ez_N3_v9_r200 3 9 200" \
  | xargs -P 4 -I{} bash -c 'job "{}"'
echo "[R10] ===== DONE $(date) =====" >> $SUM
