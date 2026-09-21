#!/bin/bash
# Redo the warm-start test from a base that actually works.
#
# The first attempt initialised from n4gap_v10_s1, which scored 16.7% -- the failing half of the
# bimodal pair at ratio 0.909. Starting the harder ratio from a policy that had not itself
# learned to close tests nothing, and both seeds returned 0.0%. n4gap_v10_s2 reached 93.3% at the
# same ratio and is the correct base.
#
# The question stands: N=4 fails at 0.955 from scratch (0/0/23.3) where N=3 succeeds
# (30/83/57/83/90), and geometry says four bounded Apollonius balls intersect to no more than
# three, so N=4 cannot be geometrically worse. If a competent starting point rescues it, the
# failure is exploration. If it still dies from a 93.3% base, the cause is elsewhere and worth a
# different line of attack.
#
# Also runs N=5 at the same ratio: at parity N=5 beat N=4 on both seeds, so if the deficit is
# specific to four agents rather than monotone in team size, that shows up here.
exec 9>/tmp/school_r17.lock
flock -n 9 || { echo "[R17] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/r17_summary.txt
RUN=xuance-master/examples/run_experiment.py
EV3=xuance-master/examples/evaluate_3d.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
BASE=$(mp n4gap_v10_s2)
echo "[R17] ===== START $(date) base=$BASE =====" >> $SUM
[ -z "$BASE" ] && { echo "[R17] base missing" >> $SUM; exit 1; }
# confirm the base is the good one before spending three hours on it
python $EV3 --policy model --model-path "$BASE" --num-agents 4 --building-mode medium \
  --target-max-speed 10 --target-min-speed 8 --evader-center-pull 1.0 --surround-spawn \
  --spawn-radius 200 --cbf-safety --cbf-margin 2 --k 20 --seed 11 \
  --out p3_results/basecheck_n4gap_v10_s2.json > /dev/null 2>&1
echo "[R17] base recheck $(grep -oE '"success_rate": [0-9.]+' p3_results/basecheck_n4gap_v10_s2.json 2>/dev/null)" >> $SUM

job () {
  set -- $1
  local n=$1 nag=$2 v=$3 sd=$4 warm=$5
  local EX=""
  [ "$warm" = "warm" ] && EX="--load-from $BASE --policy-only-load --start-noise 0.12"
  if [ -z "$(mp $n)" ]; then
    echo "[R17] TRAIN $n N=$nag v=$v seed=$sd $warm $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed $sd --num-agents $nag --building-mode medium \
      --target-max-speed $v --target-min-speed 8 --evader-center-pull 1.0 \
      --closure-weight 16 --pincer-weight 12 --separation-weight 4 --separation-distance 20 \
      --w-pos 0.6 --w-gap 0.8 --surround-spawn --spawn-radius 200 --cbf-safety --cbf-margin 2 \
      $EX --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
    echo "[R17] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[R17] $n NO_MODEL" >> $SUM; return; }
  local OUT=p3_results/eval_${n}_k30.json
  [ -f "$OUT" ] || python $EV3 --policy model --model-path "$FM" --num-agents $nag \
      --building-mode medium --target-max-speed $v --target-min-speed 8 --evader-center-pull 1.0 \
      --surround-spawn --spawn-radius 200 --cbf-safety --cbf-margin 2 \
      --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  echo "[R17] $n $(grep -oE '"(success_rate|collision_rate|timeout_rate)": [0-9.]+' $OUT 2>/dev/null | tr '\n' ' ')" >> $SUM
}
export -f job mp; export RUN EV3 SUM BASE
printf '%s\n' "n4warm2_v105_s1 4 10.5 1 warm" "n4warm2_v105_s2 4 10.5 2 warm" \
              "n5_v105_s1 5 10.5 1 cold" "n4gap_v10_s3 4 10 3 cold" \
  | xargs -P 4 -I{} bash -c 'job "{}"'
echo "[R17] ===== DONE $(date) =====" >> $SUM
