#!/bin/bash
# Round 3, three jobs.
#
# (1) Diagnose the strict-capture zeros. Chapter 4 scored 0.0 per-target capture under Chapter
#     3's rule -- not one target physically reached in 60 instances. Chapter 3 had the same
#     failure (encircle, never close) until its closure weight went 8 -> 16; Chapter 4 still
#     trains at 8. The probe reports closest approach, which separates "parks on the
#     encirclement shell" from a scoring fault.
#
# (2) Finish the CBF sweep. The queue guard misfired: this script chained on the N-sweep's lock
#     file, but the N-sweep was itself still blocked waiting on the Apollonius arm and had not
#     opened that file yet, so the lock was free and the CBF stage ran early -- N=5/6/8 were
#     skipped as "no model yet". Those models exist now.
#
# (3) Retrain Chapter 4 at closure weight 16 with the strict rule active, which is the fix
#     Chapter 3 already validated.
exec 9>/tmp/school_r3.lock
flock -n 9 || { echo "[R3] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/r3_summary.txt
RUN=xuance-master/examples/run_experiment.py
EV3=xuance-master/examples/evaluate_3d.py
EVM=xuance-master/examples/evaluate_mt.py
PRB=xuance-master/examples/probe_closure_gap.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
sr () { grep -oE '"(success_rate|collision_rate|timeout_rate|per_target_capture_rate)": [0-9.]+' "$1" 2>/dev/null | tr '\n' ' '; }
echo "[R3] ===== START $(date) =====" >> $SUM

# (1) why does Chapter 4 never physically reach a target
for n in apo_M2_s1 apofix_M2_s1; do
  FM=$(mp $n); [ -z "$FM" ] && continue
  EX="--apollonius-alloc"; case $n in apofix_*) EX="$EX --no-dynamic-alloc";; esac
  OUT=p3_results/gap_${n}.json
  [ -f "$OUT" ] || python $PRB --model-path "$FM" $EX --strict-capture --k 20 --seed 7 \
      --out "$OUT" > /dev/null 2>&1
  echo "[R3] GAP $n $(grep -oE '"(median|min|mean)": [0-9.]+' $OUT 2>/dev/null | tr '\n' ' ') within1x=$(grep -oE '"frac_within_catch_radius": [0-9.]+' $OUT 2>/dev/null)" >> $SUM
done

# (2) CBF on the larger teams, now that the unified models exist
for spec in "uni_N5_s1 5" "uni_N6_s1 6" "uni_N8_s1 8"; do
  set -- $spec; n=$1; nag=$2
  FM=$(mp $n); [ -z "$FM" ] && { echo "[R3] $n still missing" >> $SUM; continue; }
  for MODE in off on; do
    OUT=p3_results/cbf_${MODE}_${n}_k30.json
    [ -f "$OUT" ] && { echo "[R3] $n cbf=$MODE (cached) $(sr $OUT)" >> $SUM; continue; }
    EX=""; [ "$MODE" = on ] && EX="--cbf-safety"
    python $EV3 --policy model --model-path "$FM" --num-agents $nag --building-mode medium \
      --target-max-speed 10 --target-min-speed 8 $EX --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
    echo "[R3] $n N=$nag cbf=$MODE $(sr $OUT)" >> $SUM
  done
done

# (3) Chapter 4 retrained the way Chapter 3 was fixed: closure weight 16, strict rule on
job () {
  set -- $1
  local n=$1; local sd=$2
  if [ -z "$(mp $n)" ]; then
    echo "[R3] TRAIN $n seed=$sd (cw16 + strict) $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed $sd --num-targets 2 --num-agents 6 \
      --env uav_pursuit_apollonius_multitarget_3d --building-mode medium \
      --closure-weight 16 --w-pos 0.6 --w-gap 0.8 --target-max-speed 10 --target-min-speed 8 \
      --strict-capture --apollonius-alloc --no-dynamic-alloc \
      --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
    echo "[R3] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && return
  OUT=p3_results/eval_${n}_k30.json
  [ -f "$OUT" ] || python $EVM --model-path "$FM" --num-agents 6 --num-targets 2 \
      --building-mode medium --target-max-speed 10 --apollonius-alloc --no-dynamic-alloc \
      --strict-capture --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  echo "[R3] $n STRICT-TRAINED $(sr $OUT)" >> $SUM
}
export -f job mp sr; export RUN EVM SUM
printf '%s\n' "ch4fix_s1 1" "ch4fix_s2 2" "ch4fix_s3 3" | xargs -P 3 -I{} bash -c 'job "{}"'
echo "[R3] ===== DONE $(date) =====" >> $SUM
