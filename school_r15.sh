#!/bin/bash
# Why does a fourth pursuer hurt at ratio 0.955?
#
# The separation barrier is ruled out arithmetically: at the 19.5 m guide radius a tetrahedral
# arrangement puts neighbours 29.3 m apart, and the barrier only acts below 20 m, so it never
# engages for any N tested (N=6 is the tightest at 24.6 m). And the geometry runs the other way
# anyway -- below ratio 1 each pursuer contributes a bounded Apollonius ball, and the evader's
# safe region is their intersection, which four balls can only shrink relative to three. Four
# pursuers cannot be geometrically worse, so the 63.3% vs 7.8% gap (p < 1e-5, 4 and 3 seeds) is
# an optimisation failure rather than a capability limit.
#
# The sweep also has a hole that makes the gap hard to read: at r=200, N=4 was only ever run at
# ratios 0.955 and 1.000. Without knowing whether N=4 works at 0.909 there, we cannot tell a
# ratio that is too hard from a team size that never trains.
#
# Two arms:
#   fill the hole  -- N=4 at 0.909 and 0.818, r=200
#   warm start     -- N=4 at 0.955 initialised from the 0.909 policy. If a working starting point
#                     rescues it, the failure is exploration; if it still dies, something about
#                     N=4 at this ratio is structural and worth a different look.
exec 9>/tmp/school_r15.lock
flock -n 9 || { echo "[R15] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/r15_summary.txt
RUN=xuance-master/examples/run_experiment.py
EV3=xuance-master/examples/evaluate_3d.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
echo "[R15] ===== START $(date) =====" >> $SUM
find p3_results -name "train_tz_*.log" -size +40M -delete 2>/dev/null
rm -rf logs/maddpg/apollonius_3d_tz_* logs/maddpg/apollonius_3d_urb2_* 2>/dev/null

train () {   # name agents vmax seed [warmstart-model]
  set -- $1
  local n=$1 nag=$2 v=$3 sd=$4 base=$5
  local EX=""
  if [ -n "$base" ]; then
    local BM; BM=$(mp $base)
    [ -z "$BM" ] && { echo "[R15] $n base $base missing" >> $SUM; return; }
    EX="--load-from $BM --policy-only-load --start-noise 0.12"
  fi
  if [ -z "$(mp $n)" ]; then
    echo "[R15] TRAIN $n N=$nag v=$v seed=$sd ${base:+warm from $base} $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed $sd --num-agents $nag --building-mode medium \
      --target-max-speed $v --target-min-speed 8 --evader-center-pull 1.0 \
      --closure-weight 16 --pincer-weight 12 --separation-weight 4 --separation-distance 20 \
      --w-pos 0.6 --w-gap 0.8 --surround-spawn --spawn-radius 200 --cbf-safety --cbf-margin 2 \
      $EX --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
    echo "[R15] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[R15] $n NO_MODEL" >> $SUM; return; }
  local OUT=p3_results/eval_${n}_k30.json
  [ -f "$OUT" ] || python $EV3 --policy model --model-path "$FM" --num-agents $nag \
      --building-mode medium --target-max-speed $v --target-min-speed 8 --evader-center-pull 1.0 \
      --surround-spawn --spawn-radius 200 --cbf-safety --cbf-margin 2 \
      --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  echo "[R15] $n $(grep -oE '"(success_rate|collision_rate|timeout_rate)": [0-9.]+' $OUT 2>/dev/null | tr '\n' ' ')" >> $SUM
}
export -f train mp; export RUN EV3 SUM
# fill the hole first; the warm start depends on one of these existing
printf '%s\n' "n4gap_v10_s1 4 10 1" "n4gap_v10_s2 4 10 2" "n4gap_v9_s1 4 9 1" "n3gap_v105_s5 3 10.5 5" \
  | xargs -P 4 -I{} bash -c 'train "{}"'
printf '%s\n' "n4warm_v105_s1 4 10.5 1 n4gap_v10_s1" "n4warm_v105_s2 4 10.5 2 n4gap_v10_s1" \
  | xargs -P 2 -I{} bash -c 'train "{}"'
echo "[R15] ===== DONE $(date) =====" >> $SUM
