#!/bin/bash
# Three gaps left in the Chapter 3 story.
#
# 1. The transition zone. At ratio 0.955 with three pursuers the two seeds gave 30.0% and 83.3%
#    -- too wide to quote a number, and this is exactly the ratio where the report claims the
#    feasibility boundary sits. Two more seeds decide whether that is one bad run or genuine
#    bimodality.
#
# 2. The N=4 anomaly. At the same ratio, four pursuers returned 0.0% on both seeds while three
#    returned 30-83%, failing by timeout (96.7%) rather than collision. Reproducible, unexplained,
#    and it contradicts the N=5 > N=4 ordering seen at parity. Two more seeds establish whether
#    it is real before anyone tries to explain it.
#
# 3. The built-up models were TRAINED at the default margin of 6, which the sweep later showed
#    costs this scenario up to 40 points; only their evaluation used margin 2. Training with the
#    margin that actually works should beat 93.3%, and until that is run the reported number is
#    a model fighting its own safety filter.
exec 9>/tmp/school_r14.lock
flock -n 9 || { echo "[R14] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/r14_summary.txt
RUN=xuance-master/examples/run_experiment.py
EV3=xuance-master/examples/evaluate_3d.py
PCG=xuance-master/examples/probe_capture_geometry.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
echo "[R14] ===== START $(date) =====" >> $SUM
find p3_results -name "train_bd_*.log" -size +40M -delete 2>/dev/null
find p3_results -name "train_ez_*.log" -size +40M -delete 2>/dev/null
rm -rf logs/maddpg/apollonius_3d_urb_* logs/maddpg/apollonius_3d_bd_* 2>/dev/null

job () {
  set -- $1
  local n=$1 nag=$2 v=$3 dens=$4 rad=$5 mg=$6 sd=$7
  if [ -z "$(mp $n)" ]; then
    echo "[R14] TRAIN $n N=$nag v=$v $dens r=$rad margin=$mg seed=$sd $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed $sd --num-agents $nag --building-mode $dens \
      --target-max-speed $v --target-min-speed 8 --evader-center-pull 1.0 \
      --closure-weight 16 --pincer-weight 12 --separation-weight 4 --separation-distance 20 \
      --w-pos 0.6 --w-gap 0.8 --surround-spawn --spawn-radius $rad \
      --cbf-safety --cbf-margin $mg \
      --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
    echo "[R14] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[R14] $n NO_MODEL" >> $SUM; return; }
  local OUT=p3_results/eval_${n}_k30.json
  [ -f "$OUT" ] || python $EV3 --policy model --model-path "$FM" --num-agents $nag \
      --building-mode $dens --target-max-speed $v --target-min-speed 8 --evader-center-pull 1.0 \
      --surround-spawn --spawn-radius $rad --cbf-safety --cbf-margin $mg \
      --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  echo "[R14] $n $(grep -oE '"(success_rate|collision_rate|timeout_rate)": [0-9.]+' $OUT 2>/dev/null | tr '\n' ' ')" >> $SUM
  local G=p3_results/capgeo_r14_${n}.json
  [ -f "$G" ] || python $PCG --model-path "$FM" --num-agents $nag --building-mode $dens \
      --target-max-speed $v --target-min-speed 8 --evader-center-pull 1.0 \
      --surround-spawn --spawn-radius $rad --cbf-safety --cbf-margin $mg \
      --k 40 --seed 7 --out "$G" > /dev/null 2>&1
}
export -f job mp; export RUN EV3 PCG SUM
# transition zone and the N=4 anomaly, two fresh seeds each
printf '%s\n' "tz_N3_v105_s3 3 10.5 medium 200 2 3" "tz_N3_v105_s4 3 10.5 medium 200 2 4" \
              "tz_N4_v105_s3 4 10.5 medium 200 2 3" "tz_N4_v105_s4 4 10.5 medium 200 2 4" \
  | xargs -P 4 -I{} bash -c 'job "{}"'
# built-up, retrained at the margin that works
printf '%s\n' "urb2_N3_v9_cx 3 9 complex 200 2 1" "urb2_N3_v10_cx 3 10 complex 200 2 1" \
              "urb2_N4_v11_cx 4 11 complex 200 2 1" "urb2_N4_v10_cx 4 10 complex 200 2 1" \
  | xargs -P 4 -I{} bash -c 'job "{}"'
echo "[R14] ===== DONE $(date) =====" >> $SUM
