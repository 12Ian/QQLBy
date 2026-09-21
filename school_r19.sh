#!/bin/bash
# Two things stand between the current state and a deliverable.
#
# (1) Every ablation rests on a single seed. The load-bearing claim -- that the guide collapse is
#     worth 23.3 points and the escape-angle term 10 -- is the one a reviewer will press hardest,
#     and one run cannot carry it. Two more seeds each for the two components that mattered, and
#     for one that did not, so the null results are equally supported.
#
# (2) Chapter 4 is unusable. The phantom-target defect is now fixed: _field_and_guides set the
#     per-target speed but left self.target_position on whatever the inherited single-target
#     field last held, so the formation radius was computed against a target 442-965 m away.
#     Whether that alone revives it is doubtful -- the guide radius still collapsed to ~20 m at
#     zero escape angle regardless -- so this run is as much a test of the fix as a rescue
#     attempt, and it carries every other correction: surround spawn, strict capture, the three
#     restored reward terms, centre-pulled evader, filter margin 2.
exec 9>/tmp/school_r19.lock
flock -n 9 || { echo "[R19] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/r19_summary.txt
RUN=xuance-master/examples/run_experiment.py
EV3=xuance-master/examples/evaluate_3d.py
EVM=xuance-master/examples/evaluate_mt.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
CM="--target-min-speed 8 --evader-center-pull 1.0 --surround-spawn --spawn-radius 200 --cbf-safety --cbf-margin 2"
echo "[R19] ===== START $(date) =====" >> $SUM

# --- Chapter 4 smoke test first: does the fix change behaviour at all? ---
timeout 1800 python $RUN --name ch4fix2_smoke --seed 1 --num-targets 2 --num-agents 6 \
  --env uav_pursuit_apollonius_multitarget_3d --building-mode medium \
  --target-max-speed 10 --target-min-speed 8 --evader-center-pull 1.0 \
  --closure-weight 16 --pincer-weight 12 --w-pos 0.6 --w-gap 0.8 \
  --surround-spawn --spawn-radius 200 --strict-capture --cbf-safety --cbf-margin 2 \
  --steps 40000 --parallels 4 --eval-interval 20000 --test-episode 4 \
  > p3_results/train_ch4fix2_smoke.log 2>&1
echo "[R19] ch4 smoke exit=$? $(date +%H:%M)" >> $SUM
rm -rf models/maddpg/apollonius_3d_ch4fix2_smoke logs/maddpg/apollonius_3d_ch4fix2_smoke 2>/dev/null

job () {
  set -- $1
  local n=$1 nag=$2 v=$3 dens=$4 sd=$5 extra=$6
  if [ -z "$(mp $n)" ]; then
    echo "[R19] TRAIN $n seed=$sd $extra $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed $sd --num-agents $nag --building-mode $dens \
      --target-max-speed $v $CM --closure-weight 16 --pincer-weight 12 \
      --separation-weight 4 --separation-distance 20 --w-pos 0.6 --w-gap 0.8 $extra \
      --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
    echo "[R19] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[R19] $n NO_MODEL" >> $SUM; return; }
  local OUT=p3_results/eval_${n}_k30.json
  [ -f "$OUT" ] || python $EV3 --policy model --model-path "$FM" --num-agents $nag \
      --building-mode $dens --target-max-speed $v $CM --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  echo "[R19] $n $(grep -oE '"(success_rate|collision_rate)": [0-9.]+' $OUT 2>/dev/null | tr '\n' ' ')" >> $SUM
}
export -f job mp; export RUN EV3 SUM CM
# ablation replication: the two that mattered, plus one that did not
printf '%s\n' \
  "cx2_nocollapse_s2 3 9 complex 2 --no-guide-collapse" \
  "cx2_nocollapse_s3 3 9 complex 3 --no-guide-collapse" \
  "cx2_nogap_s2 3 9 complex 2 --reward-disable=r_gap" \
  "cx2_nogap_s3 3 9 complex 3 --reward-disable=r_gap" \
  | xargs -P 4 -I{} bash -c 'job "{}"'
printf '%s\n' \
  "cx2_noclosure_s2 3 9 complex 2 --reward-disable=r_closure" \
  "cx2_full_s4 3 9 complex 4 " \
  "cx2_v10_s4 3 10 complex 4 " \
  "cx2_v10_s5 3 10 complex 5 " \
  | xargs -P 4 -I{} bash -c 'job "{}"'
echo "[R19] ===== DONE $(date) =====" >> $SUM
