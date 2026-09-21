#!/bin/bash
# Capture in built-up ground instead of against the arena wall.
#
# The escape criterion already treats a direction terminating on a building as closed --
# geometry_open is false when the free ray length is short -- so herding the evader into
# structure is, mathematically, exactly "blocking the exits". It has not shown up in the results
# because medium density is 16 buildings over 2.6% of the floor: too sparse for the team to use.
# Complex density is 26 buildings over 33.3%, which is street-like, and its earlier 13.3% score
# was a collision problem (86.7% crash rate), not a pursuit one. The barrier filter now holds
# collisions near zero, so the scenario that previously destroyed the team is the one that
# should let it work with the buildings rather than around them.
#
# Judged on where captures land: near structure (buildings within 150 m, escape directions
# closed by structure rather than by pursuers) and away from the arena boundary.
exec 8>/tmp/school_r12.lock
flock 8
exec 8>&-
exec 9>/tmp/school_r13.lock
flock -n 9 || { echo "[R13] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/r13_summary.txt
RUN=xuance-master/examples/run_experiment.py
EV3=xuance-master/examples/evaluate_3d.py
PCG=xuance-master/examples/probe_capture_geometry.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
echo "[R13] ===== START $(date) =====" >> $SUM
find p3_results -name "train_bd_*.log" -size +40M -delete 2>/dev/null
rm -rf logs/maddpg/apollonius_3d_ez_* logs/maddpg/apollonius_3d_bd_* 2>/dev/null

job () {
  set -- $1
  local n=$1 nag=$2 v=$3 dens=$4 sd=$5
  if [ -z "$(mp $n)" ]; then
    echo "[R13] TRAIN $n N=$nag v=$v density=$dens seed=$sd $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed $sd --num-agents $nag --building-mode $dens \
      --target-max-speed $v --target-min-speed 8 --evader-center-pull 1.0 \
      --closure-weight 16 --pincer-weight 12 --separation-weight 4 --separation-distance 20 \
      --w-pos 0.6 --w-gap 0.8 --surround-spawn --spawn-radius 200 --cbf-safety \
      --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
    echo "[R13] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[R13] $n NO_MODEL" >> $SUM; return; }
  local OUT=p3_results/eval_${n}_k30.json
  [ -f "$OUT" ] || python $EV3 --policy model --model-path "$FM" --num-agents $nag \
      --building-mode $dens --target-max-speed $v --target-min-speed 8 \
      --evader-center-pull 1.0 --surround-spawn --spawn-radius 200 --cbf-safety \
      --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  local G=p3_results/capgeo_r13_${n}.json
  [ -f "$G" ] || python $PCG --model-path "$FM" --num-agents $nag --building-mode $dens \
      --target-max-speed $v --target-min-speed 8 --evader-center-pull 1.0 \
      --surround-spawn --spawn-radius 200 --cbf-safety --k 40 --seed 7 --out "$G" > /dev/null 2>&1
  echo "[R13] $n $(grep -oE '"(success_rate|collision_rate|timeout_rate)": [0-9.]+' $OUT 2>/dev/null | tr '\n' ' ')" >> $SUM
}
export -f job mp; export RUN EV3 PCG SUM
printf '%s\n' "urb_N3_v10_cx 3 10 complex 1" "urb_N4_v10_cx 4 10 complex 1" \
              "urb_N3_v9_cx 3 9 complex 1" "urb_N4_v11_cx 4 11 complex 1" \
  | xargs -P 4 -I{} bash -c 'job "{}"'
echo "[R13] ===== DONE $(date) =====" >> $SUM
