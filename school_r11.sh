#!/bin/bash
# Locate the feasibility boundary, and test whether team size or closing distance can move it.
#
# With the wall crutch removed, ratio 0.82 and 0.91 both train to 100% with captures in open air
# (median 108 m from any face at spawn radius 200), while 1.00 trains to 0.0% at N=3 and N=4.
# The break sits exactly where the Apollonius set stops being a bounded sphere and becomes a
# half-space, so the interesting question is no longer "does it work" but "where does it stop,
# and what moves that point".
#
# Two candidate levers, varied against a common reference so the answer is readable:
#   team size  -- four half-spaces are the minimum that can bound a region in R^3, so if N is
#     the binding constraint, N=4/5 should survive at ratios where N=3 dies;
#   closing distance -- ratio 1.00 leaves no net closing speed over 200 m. Spawning at 80 m
#     asks only whether the endgame is learnable, separately from whether the approach is.
exec 9>/tmp/school_r11.lock
flock -n 9 || { echo "[R11] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/r11_summary.txt
RUN=xuance-master/examples/run_experiment.py
EV3=xuance-master/examples/evaluate_3d.py
PCG=xuance-master/examples/probe_capture_geometry.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
sr () { grep -oE '"(success_rate|collision_rate|timeout_rate)": [0-9.]+' "$1" 2>/dev/null | tr '\n' ' '; }
echo "[R11] ===== START $(date) =====" >> $SUM
find p3_results -name "train_cp_*.log" -size +40M -delete 2>/dev/null
rm -rf logs/maddpg/apollonius_3d_cp_* 2>/dev/null

job () {
  set -- $1
  local n=$1; local nag=$2; local v=$3; local rad=$4
  if [ -z "$(mp $n)" ]; then
    echo "[R11] TRAIN $n N=$nag vmax=$v r=$rad lambda=$(python -c "print(f'{$v/11:.3f}')") $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed 1 --num-agents $nag --building-mode medium \
      --target-max-speed $v --target-min-speed 8 --evader-center-pull 1.0 \
      --closure-weight 16 --pincer-weight 12 --separation-weight 4 --separation-distance 20 \
      --w-pos 0.6 --w-gap 0.8 --surround-spawn --spawn-radius $rad --cbf-safety \
      --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
    echo "[R11] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[R11] $n NO_MODEL" >> $SUM; return; }
  OUT=p3_results/eval_${n}_k30.json
  [ -f "$OUT" ] || python $EV3 --policy model --model-path "$FM" --num-agents $nag \
      --building-mode medium --target-max-speed $v --target-min-speed 8 \
      --evader-center-pull 1.0 --surround-spawn --spawn-radius $rad --cbf-safety \
      --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  echo "[R11] $n N=$nag v=$v r=$rad $(sr $OUT)" >> $SUM
  G=p3_results/capgeo_r11_${n}.json
  [ -f "$G" ] || python $PCG --model-path "$FM" --num-agents $nag --building-mode medium \
      --target-max-speed $v --target-min-speed 8 --evader-center-pull 1.0 \
      --surround-spawn --spawn-radius $rad --cbf-safety --k 40 --seed 7 \
      --out "$G" > /dev/null 2>&1
}
export -f job mp sr; export RUN EV3 PCG SUM
# where exactly does N=3 break, and does a fourth/fifth pursuer hold on past it
printf '%s\n' "bd_N3_v105_r200 3 10.5 200" "bd_N4_v105_r200 4 10.5 200" \
              "bd_N4_v11_r80 4 11 80" "bd_N5_v11_r80 5 11 80" \
  | xargs -P 4 -I{} bash -c 'job "{}"'
printf '%s\n' "bd_N3_v10_r200_s2 3 10 200" "bd_N3_v10_r200_s3 3 10 200" "bd_N4_v11_r200 4 11 200" \
  | xargs -P 3 -I{} bash -c 'job "{}"'
echo "[R11] ===== DONE $(date) =====" >> $SUM
