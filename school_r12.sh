#!/bin/bash
# Firm up the boundary: capture geometry for every trained arm, plus extra seeds where the
# current numbers rest on one run.
#
# The picture so far, all with the centre-pulled evader so no arm can lean on a wall:
#   ratio 0.818, r=200 -> 100%, captures a median 108 m from any face (2% inside 50 m)
#   ratio 0.909, r=200 -> 100% on two seeds
#   ratio 0.955, r=200 -> 30% at N=3, 0% at N=4
#   ratio 1.000, r=200 -> 0% at N=4
#   ratio 1.000, r=80  -> 33% at N=4, 43% at N=5
# So the break sits near 0.95, closing distance dominates at parity (0% at 200 m vs 33-43% at
# 80 m), and at 80 m a fifth pursuer helps. The N=4 zero at 0.955 contradicts that trend and is
# a single seed -- N=4 has a known instability in this codebase, so it needs replication before
# it means anything.
exec 9>/tmp/school_r12.lock
flock -n 9 || { echo "[R12] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/r12_summary.txt
RUN=xuance-master/examples/run_experiment.py
EV3=xuance-master/examples/evaluate_3d.py
PCG=xuance-master/examples/probe_capture_geometry.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
echo "[R12] ===== START $(date) =====" >> $SUM

# geometry for every arm already trained
geo () {
  set -- $1
  local n=$1 nag=$2 v=$3 r=$4
  local FM; FM=$(mp $n); [ -z "$FM" ] && return
  local G=p3_results/capgeo_r12_${n}.json
  [ -f "$G" ] || python $PCG --model-path "$FM" --num-agents $nag --building-mode medium \
      --target-max-speed $v --target-min-speed 8 --evader-center-pull 1.0 \
      --surround-spawn --spawn-radius $r --cbf-safety --k 40 --seed 7 --out "$G" > /dev/null 2>&1
  echo "[R12] GEO $n $(grep -oE '"frac_any_boundary_within_50m": [0-9.]+' $G 2>/dev/null)" >> $SUM
}
export -f geo mp; export PCG SUM
printf '%s\n' "bd_N3_v10_r200_s2 3 10 200" "bd_N3_v10_r200_s3 3 10 200" "bd_N3_v105_r200 3 10.5 200" \
              "bd_N4_v11_r80 4 11 80" "bd_N5_v11_r80 5 11 80" "bd_N4_v11_r200 4 11 200" \
              "ez_N4_v9_r80 4 9 80" \
  | xargs -P 4 -I{} bash -c 'geo "{}"'

# replication where a single run is carrying a claim
job () {
  set -- $1
  local n=$1 nag=$2 v=$3 r=$4 sd=$5
  if [ -z "$(mp $n)" ]; then
    echo "[R12] TRAIN $n N=$nag v=$v r=$r seed=$sd $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed $sd --num-agents $nag --building-mode medium \
      --target-max-speed $v --target-min-speed 8 --evader-center-pull 1.0 \
      --closure-weight 16 --pincer-weight 12 --separation-weight 4 --separation-distance 20 \
      --w-pos 0.6 --w-gap 0.8 --surround-spawn --spawn-radius $r --cbf-safety \
      --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[R12] $n NO_MODEL" >> $SUM; return; }
  local OUT=p3_results/eval_${n}_k30.json
  [ -f "$OUT" ] || python $EV3 --policy model --model-path "$FM" --num-agents $nag \
      --building-mode medium --target-max-speed $v --target-min-speed 8 \
      --evader-center-pull 1.0 --surround-spawn --spawn-radius $r --cbf-safety \
      --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  echo "[R12] $n N=$nag v=$v r=$r $(grep -oE '"(success_rate|timeout_rate)": [0-9.]+' $OUT 2>/dev/null | tr '\n' ' ')" >> $SUM
}
export -f job; export RUN EV3
printf '%s\n' "bd_N3_v105_r200_s2 3 10.5 200 2" "bd_N4_v105_r200_s2 4 10.5 200 2" \
              "bd_N4_v11_r80_s2 4 11 80 2" "bd_N5_v11_r80_s2 5 11 80 2" \
  | xargs -P 4 -I{} bash -c 'job "{}"'
echo "[R12] ===== DONE $(date) =====" >> $SUM
