#!/bin/bash
# Nail down the asymmetry: training at ratio 0.955 helps three pursuers and harms four.
#
# Measured so far at that ratio, N=4: trained from scratch 7.8% (0/0/23.3), warm-started from a
# 90% policy 0.0% (0/0), and the untouched 0.909 policies transfer at 16.7% and 60.0% with no
# training at all. N=3 goes the other way -- 68.7% trained against 60.0% transferred. So the
# gradient at this ratio appears to destroy competence for N=4 while building it for N=3, which
# would mean the earlier "N=4 collapses" reading was an artefact of how it was trained rather
# than anything about four pursuers.
#
# Two seeds of zero-shot is not enough to say that. This trains four more N=4 policies at 0.909
# and transfers every one, and does the same for N=3, so both the transfer numbers and the
# trained numbers rest on comparable sample sizes.
exec 9>/tmp/school_r18.lock
flock -n 9 || { echo "[R18] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/r18_summary.txt
RUN=xuance-master/examples/run_experiment.py
EV3=xuance-master/examples/evaluate_3d.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
echo "[R18] ===== START $(date) =====" >> $SUM

job () {
  set -- $1
  local n=$1 nag=$2 sd=$3
  if [ -z "$(mp $n)" ]; then
    echo "[R18] TRAIN $n N=$nag seed=$sd at ratio 0.909 $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed $sd --num-agents $nag --building-mode medium \
      --target-max-speed 10 --target-min-speed 8 --evader-center-pull 1.0 \
      --closure-weight 16 --pincer-weight 12 --separation-weight 4 --separation-distance 20 \
      --w-pos 0.6 --w-gap 0.8 --surround-spawn --spawn-radius 200 --cbf-safety --cbf-margin 2 \
      --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
    echo "[R18] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[R18] $n NO_MODEL" >> $SUM; return; }
  # score at its training ratio, then transfer untouched to the harder one
  for v in 10 10.5; do
    local OUT=p3_results/eval_${n}_v${v}.json
    [ -f "$OUT" ] || python $EV3 --policy model --model-path "$FM" --num-agents $nag \
        --building-mode medium --target-max-speed $v --target-min-speed 8 \
        --evader-center-pull 1.0 --surround-spawn --spawn-radius 200 --cbf-safety --cbf-margin 2 \
        --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
    echo "[R18] $n N=$nag v=$v $(grep -oE '"(success_rate|timeout_rate)": [0-9.]+' $OUT 2>/dev/null | tr '\n' ' ')" >> $SUM
  done
}
export -f job mp; export RUN EV3 SUM
printf '%s\n' "tr_N4_s4 4 4" "tr_N4_s5 4 5" "tr_N3_s4 3 4" "tr_N3_s5 3 5" \
  | xargs -P 4 -I{} bash -c 'job "{}"'
echo "[R18] ===== DONE $(date) =====" >> $SUM
