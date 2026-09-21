#!/bin/bash
# Move the ablations to the scenario where the components can actually matter.
#
# Run in sparse airspace they said nothing: removing the closure term, the guide collapse, or the
# escape-angle term each left the score at 100%, and dropping the pincer term cost 3.3 points.
# The baselines explain why -- greedy pursuit also scores 100% and 96.7% there, so nothing in the
# reward is load-bearing because the scenario does not demand anything.
#
# Built-up density is where the gap opens: learned 87.8% over three seeds against greedy 20.0%
# (z = 7.07), with greedy failing on an 80% collision rate. That is the setting whose ablations
# carry information, and where the herding behaviour that closes half the exits with structure
# actually happens.
#
# Also adds the sparse baseline at ratio 0.955, since the one regime where sparse airspace might
# still separate the methods is near the feasibility boundary.
exec 9>/tmp/school_ch3b.lock
flock -n 9 || { echo "[CH3B] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/ch3b_summary.txt
RUN=xuance-master/examples/run_experiment.py
EV3=xuance-master/examples/evaluate_3d.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
COMMON="--target-min-speed 8 --evader-center-pull 1.0 --surround-spawn --spawn-radius 200"
echo "[CH3B] ===== START $(date) =====" >> $SUM

# does greedy separate from learned anywhere in sparse airspace, or only in built-up?
for spec in "greedy 3 10.5 medium" "random 3 10.5 medium" "greedy 3 10 complex" "greedy 4 11 complex"; do
  set -- $spec; pol=$1; nag=$2; v=$3; dens=$4
  OUT=p3_results/base_${pol}_N${nag}_v${v}_${dens}.json
  [ -f "$OUT" ] || python $EV3 --policy $pol --num-agents $nag --building-mode $dens \
    --target-max-speed $v $COMMON --cbf-safety --cbf-margin 2 --k 30 --seed 7 \
    --out "$OUT" > /dev/null 2>&1
  echo "[CH3B] BASELINE $pol N=$nag v=$v $dens $(grep -oE '"(success_rate|collision_rate)": [0-9.]+' $OUT 2>/dev/null | tr '\n' ' ')" >> $SUM
done

job () {
  set -- $1
  local n=$1 nag=$2 v=$3 dens=$4 sd=$5 extra=$6
  if [ -z "$(mp $n)" ]; then
    echo "[CH3B] TRAIN $n ($extra) $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed $sd --num-agents $nag --building-mode $dens \
      --target-max-speed $v $COMMON --closure-weight 16 --pincer-weight 12 \
      --separation-weight 4 --separation-distance 20 --w-pos 0.6 --w-gap 0.8 \
      --cbf-safety --cbf-margin 2 $extra \
      --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
    echo "[CH3B] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[CH3B] $n NO_MODEL" >> $SUM; return; }
  local OUT=p3_results/eval_${n}_k30.json
  [ -f "$OUT" ] || python $EV3 --policy model --model-path "$FM" --num-agents $nag \
      --building-mode $dens --target-max-speed $v $COMMON --cbf-safety --cbf-margin 2 \
      --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  echo "[CH3B] $n $(grep -oE '"(success_rate|collision_rate|timeout_rate)": [0-9.]+' $OUT 2>/dev/null | tr '\n' ' ')" >> $SUM
}
export -f job mp; export RUN EV3 SUM COMMON
printf '%s\n' \
  "cxabl_noclosure 3 9 complex 1 --reward-disable=r_closure" \
  "cxabl_nopincer 3 9 complex 1 --reward-disable=r_pincer" \
  "cxabl_nocollapse 3 9 complex 1 --no-guide-collapse" \
  "cxabl_nogap 3 9 complex 1 --reward-disable=r_gap" \
  | xargs -P 4 -I{} bash -c 'job "{}"'
# the safety filter is the one component greedy visibly lacks; quantify what it is worth here
printf '%s\n' "cxabl_nocbf 3 9 complex 1 " | xargs -P 1 -I{} bash -c 'job "{}"'
echo "[CH3B] ===== DONE $(date) =====" >> $SUM
