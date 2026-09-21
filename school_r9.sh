#!/bin/bash
# Retrain against an evader that no longer hugs the walls.
#
# Every capture measured under the old evader landed within 50 m of a side wall (median 14-16 m),
# with the nearest building ~100 m off and taking no part. That is not a detail: in three
# dimensions the intersection of three half-spaces is unbounded, so three pursuers cannot enclose
# an escape set on their own and the arena boundary was supplying the missing faces. Real
# airspace has no such walls, so those captures measured the arena, not the method. The evader
# now carries the standard APF attractive term toward the centre of the airspace, normalised
# per axis (the box is 1000 m wide and 300 m tall; scaling all three by the horizontal half-span
# left the vertical pull ~3x too weak and merely moved captures onto the floor).
#
# N = 3 against N = 4 at parity is the experiment the geometry actually predicts: four
# half-spaces are the minimum that can bound a region in R^3, so if the account is right, the
# fourth pursuer should matter far more here than it did when the walls were helping. Both arms
# carry the barrier filter so that collision differences cannot masquerade as a team-size effect.
#
# W is substituted by the caller from the calibration sweep.
exec 9>/tmp/school_r9.lock
flock -n 9 || { echo "[R9] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
W=1.0
SUM=p3_results/r9_summary.txt
RUN=xuance-master/examples/run_experiment.py
EV3=xuance-master/examples/evaluate_3d.py
PCG=xuance-master/examples/probe_capture_geometry.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
sr () { grep -oE '"(success_rate|collision_rate|timeout_rate)": [0-9.]+' "$1" 2>/dev/null | tr '\n' ' '; }
echo "[R9] ===== START $(date) centre_pull=$W =====" >> $SUM
find p3_results -name "train_r7*.log" -size +50M -delete 2>/dev/null
rm -rf logs/maddpg/apollonius_3d_r7* logs/maddpg/apollonius_3d_r8* 2>/dev/null

job () {
  set -- $1
  local n=$1; local nag=$2; local v=$3; local sd=$4
  if [ -z "$(mp $n)" ]; then
    echo "[R9] TRAIN $n N=$nag vmax=$v seed=$sd pull=$W $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed $sd --num-agents $nag --building-mode medium \
      --target-max-speed $v --target-min-speed 8 --evader-center-pull $W \
      --closure-weight 16 --pincer-weight 12 --separation-weight 4 --separation-distance 20 \
      --w-pos 0.6 --w-gap 0.8 --surround-spawn --spawn-radius 200 --cbf-safety \
      --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
    echo "[R9] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[R9] $n NO_MODEL" >> $SUM; return; }
  OUT=p3_results/eval_${n}_k30.json
  [ -f "$OUT" ] || python $EV3 --policy model --model-path "$FM" --num-agents $nag \
      --building-mode medium --target-max-speed $v --target-min-speed 8 \
      --evader-center-pull $W --surround-spawn --cbf-safety --k 30 --seed 7 \
      --out "$OUT" > /dev/null 2>&1
  echo "[R9] $n N=$nag v=$v $(sr $OUT)" >> $SUM
  # where the captures land now: the whole point of the change
  G=p3_results/capgeo_r9_${n}.json
  [ -f "$G" ] || python $PCG --model-path "$FM" --num-agents $nag --building-mode medium \
      --target-max-speed $v --target-min-speed 8 --evader-center-pull $W \
      --surround-spawn --cbf-safety --k 40 --seed 7 --out "$G" > /dev/null 2>&1
  echo "[R9] $n GEO $(grep -oE '"(median)": [0-9.]+' $G 2>/dev/null | head -2 | tr '\n' ' ')$(grep -oE '"frac_any_boundary_within_50m": [0-9.]+' $G 2>/dev/null)" >> $SUM
}
export -f job mp sr; export RUN EV3 PCG SUM W
# parity first: it is the headline and the cleanest test of the fourth pursuer
printf '%s\n' "cp_N3_v11_s1 3 11 1" "cp_N3_v11_s2 3 11 2" "cp_N4_v11_s1 4 11 1" "cp_N4_v11_s2 4 11 2" \
  | xargs -P 4 -I{} bash -c 'job "{}"'
printf '%s\n' "cp_N3_v12_s1 3 12 1" "cp_N4_v12_s1 4 12 1" "cp_N5_v12_s1 5 12 1" \
  | xargs -P 3 -I{} bash -c 'job "{}"'
echo "[R9] ===== DONE $(date) =====" >> $SUM
