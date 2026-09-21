#!/bin/bash
# Re-run the two built-up N=4 runs that died on launch.
#
# They failed with worker-thread exceptions within a minute of starting, at the moment two
# batches overlapped and eight trainings were live on a 20-core box; the two N=3 runs launched
# alongside them finished normally. Running them alone separates resource contention from a real
# fault -- if they die again with nothing else on the machine, it is a bug worth chasing.
# Two at a time, not four.
exec 9>/tmp/school_r16.lock
flock -n 9 || { echo "[R16] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/r16_summary.txt
RUN=xuance-master/examples/run_experiment.py
EV3=xuance-master/examples/evaluate_3d.py
PCG=xuance-master/examples/probe_capture_geometry.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
echo "[R16] ===== START $(date) =====" >> $SUM
rm -rf models/maddpg/apollonius_3d_urb2_N4_* logs/maddpg/apollonius_3d_urb2_N4_* 2>/dev/null

job () {
  set -- $1
  local n=$1 nag=$2 v=$3
  echo "[R16] TRAIN $n N=$nag v=$v $(date +%m-%d\ %H:%M)" >> $SUM
  python $RUN --name "$n" --seed 1 --num-agents $nag --building-mode complex \
    --target-max-speed $v --target-min-speed 8 --evader-center-pull 1.0 \
    --closure-weight 16 --pincer-weight 12 --separation-weight 4 --separation-distance 20 \
    --w-pos 0.6 --w-gap 0.8 --surround-spawn --spawn-radius 200 --cbf-safety --cbf-margin 2 \
    --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
    > p3_results/train_$n.log 2>&1
  echo "[R16] $n exit=$? $(date +%H:%M)" >> $SUM
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[R16] $n NO_MODEL" >> $SUM; return; }
  local OUT=p3_results/eval_${n}_k30.json
  python $EV3 --policy model --model-path "$FM" --num-agents $nag --building-mode complex \
    --target-max-speed $v --target-min-speed 8 --evader-center-pull 1.0 \
    --surround-spawn --spawn-radius 200 --cbf-safety --cbf-margin 2 \
    --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  echo "[R16] $n $(grep -oE '"(success_rate|collision_rate|timeout_rate)": [0-9.]+' $OUT 2>/dev/null | tr '\n' ' ')" >> $SUM
  python $PCG --model-path "$FM" --num-agents $nag --building-mode complex \
    --target-max-speed $v --target-min-speed 8 --evader-center-pull 1.0 \
    --surround-spawn --spawn-radius 200 --cbf-safety --cbf-margin 2 \
    --k 40 --seed 7 --out p3_results/capgeo_r16_${n}.json > /dev/null 2>&1
}
export -f job mp; export RUN EV3 PCG SUM
printf '%s\n' "urb2_N4_v10_cx 4 10" "urb2_N4_v11_cx 4 11" | xargs -P 2 -I{} bash -c 'job "{}"'
echo "[R16] ===== DONE $(date) =====" >> $SUM
