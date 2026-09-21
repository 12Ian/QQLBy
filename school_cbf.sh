#!/bin/bash
# Two things the client asked for, queued behind the unified N-sweep.
#
# (a) CBF safety filter. Relaxing the crash rule would raise the score while leaving the
#     crashes in place; a barrier filter removes them instead. Offline on a random policy it
#     took 102.9 hits/500 steps to 0 across 8 seeds while touching only 4% of actions. These
#     runs measure what it does to a trained pursuit team, where the useful comparison is not
#     just collision rate but whether success survives the intervention.
#
#     Evaluated first on EXISTING models (no retraining): the filter is a wrapper on the
#     action, so it applies to an already-trained policy. Retraining with the filter in the
#     loop follows, since a policy that knows it will be corrected can learn to rely on it.
#
# (b) Chapter 4 re-scored under Chapter 3's capture rule. Ch4 counted a collapsed escape solid
#     angle as a capture; Ch3 requires physical reach. Re-evaluation only -- no retraining --
#     so the number moves for exactly one reason.
exec 8>/tmp/school_nsweep.lock
flock 8
exec 8>&-
exec 9>/tmp/school_cbf.lock
flock -n 9 || { echo "[CBF] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/cbf_summary.txt
RUN=xuance-master/examples/run_experiment.py
EV3=xuance-master/examples/evaluate_3d.py
EVM=xuance-master/examples/evaluate_mt.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
sr () { grep -oE '"(success_rate|collision_rate|timeout_rate)": [0-9.]+' "$1" 2>/dev/null | tr '\n' ' '; }
echo "[CBF] ===== START $(date) =====" >> $SUM

# --- (a1) CBF applied to already-trained policies, N=3..8, medium, vmax10 ---
for spec in "v10_N3_s1 3" "v10_N4_s1 4" "uni_N5_s1 5" "uni_N6_s1 6" "uni_N8_s1 8"; do
  set -- $spec; n=$1; nag=$2
  FM=$(mp $n); [ -z "$FM" ] && { echo "[CBF] $n no model yet" >> $SUM; continue; }
  for MODE in off on; do
    OUT=p3_results/cbf_${MODE}_${n}_k30.json
    [ -f "$OUT" ] && continue
    EXTRA=""; [ "$MODE" = on ] && EXTRA="--cbf-safety"
    python $EV3 --policy model --model-path "$FM" --num-agents $nag --building-mode medium \
      --target-max-speed 10 --target-min-speed 8 $EXTRA --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
    echo "[CBF] $n N=$nag cbf=$MODE $(sr $OUT)" >> $SUM
  done
done

# --- (a2) train WITH the filter in the loop, so the policy can exploit it (N=4 and N=6) ---
for spec in "cbf_N4_s1 4 1" "cbf_N6_s1 6 1"; do
  set -- $spec; n=$1; nag=$2; sd=$3
  if [ -z "$(mp $n)" ]; then
    echo "[CBF] TRAIN $n N=$nag (cbf in loop) $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed $sd --num-agents $nag --building-mode medium \
      --target-max-speed 10 --target-min-speed 8 --closure-weight 16 --pincer-weight 12 \
      --separation-weight 4 --separation-distance 20 --w-pos 0.6 --w-gap 0.8 \
      --cbf-safety --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
    echo "[CBF] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  FM=$(mp $n); [ -z "$FM" ] && continue
  OUT=p3_results/cbf_trained_${n}_k30.json
  [ -f "$OUT" ] || python $EV3 --policy model --model-path "$FM" --num-agents $nag \
      --building-mode medium --target-max-speed 10 --target-min-speed 8 --cbf-safety \
      --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  echo "[CBF] $n TRAINED-WITH-CBF $(sr $OUT)" >> $SUM
done

# --- (b) Chapter 4 re-scored under Chapter 3's rule (evaluation only) ---
for n in ch4o_M2_s1 ch4o_M2_s2 ch4o_M2static_s1 apo_M2_s1 apo_M2_s2 apo_M2_s3; do
  FM=$(mp $n); [ -z "$FM" ] && continue
  EXTRA=""; case $n in apo_*) EXTRA="--apollonius-alloc";; esac
  case $n in *static*) EXTRA="$EXTRA --no-dynamic-alloc";; esac
  OUT=p3_results/strict_${n}_k30.json
  [ -f "$OUT" ] || python $EVM --model-path "$FM" --num-agents 6 --num-targets 2 \
      --building-mode medium --target-max-speed 10 $EXTRA --strict-capture \
      --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  echo "[CBF] STRICT $n $(sr $OUT)" >> $SUM
done
echo "[CBF] ===== DONE $(date) =====" >> $SUM
