#!/bin/bash
# Chapter 4 under Chapter 3's capture rule, with the reward terms Chapter 4 was missing.
#
# Every earlier attempt at the strict rule returned exactly 0.0: closure weight 8 -> 16, the
# strict flag itself, and the surround spawn. None of them touched the actual cause. Comparing
# the two reward functions line by line, Chapter 4 was missing three terms Chapter 3 has:
#
#   * guide-point following (w_pos / r_guide). The guide radius does collapse toward
#     1.3 * catch_radius as the escape angle closes, but nothing rewarded flying to the
#     collapsed guide, so the collapse moved the guide and not the team.
#   * a terminal approach ramp. Chapter 3 pays (3r - d) / 2r inside three catch radii;
#     Chapter 4 paid 0 until it paid 80, leaving the final approach with no gradient.
#   * the pincer term, which keeps a single diving pursuer from carrying the score.
#
# That is why the team held station ~155 m out: the geometry was right and the incentive to act
# on it was absent. All three are ported verbatim from the single-target environment.
#
# Three arms separate the criterion from the reward fix, so a null result stays interpretable.
exec 8>/tmp/school_r6.lock
flock 8
exec 8>&-
exec 9>/tmp/school_r7.lock
flock -n 9 || { echo "[R7] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/r7_summary.txt
RUN=xuance-master/examples/run_experiment.py
EVM=xuance-master/examples/evaluate_mt.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
sr () { grep -oE '"(success_rate|per_target_capture_rate)": [0-9.]+' "$1" 2>/dev/null | tr '\n' ' '; }
cl () { grep -oE '"median": [0-9.]+' "$1" 2>/dev/null | head -1; }
echo "[R7] ===== START $(date) =====" >> $SUM
rm -rf models/maddpg/apollonius_3d_smoke_ch4rw logs/maddpg/apollonius_3d_smoke_ch4rw 2>/dev/null
find p3_results -name "train_v12cbf_*.log" -size +80M -delete 2>/dev/null
rm -rf logs/maddpg/apollonius_3d_uni_N* logs/maddpg/apollonius_3d_ch4x_* 2>/dev/null
echo "[R7] free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)G" >> $SUM

job () {
  set -- $1
  local n=$1; local sd=$2; local arm=$3
  local EX="--surround-spawn --spawn-radius 200 --closure-weight 16 --pincer-weight 12"
  case $arm in
    strict) EX="$EX --strict-capture";;
    apo)    EX="$EX --strict-capture --apollonius-alloc --no-dynamic-alloc";;
  esac
  if [ -z "$(mp $n)" ]; then
    echo "[R7] TRAIN $n arm=$arm seed=$sd $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed $sd --num-targets 2 --num-agents 6 \
      --env uav_pursuit_apollonius_multitarget_3d --building-mode medium \
      --w-pos 0.6 --w-gap 0.8 --target-max-speed 10 --target-min-speed 8 $EX \
      --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
    echo "[R7] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[R7] $n NO_MODEL" >> $SUM; return; }
  local EE="--surround-spawn"
  case $arm in
    strict) EE="$EE --strict-capture";;
    apo)    EE="$EE --strict-capture --apollonius-alloc --no-dynamic-alloc";;
  esac
  OUT=p3_results/eval_${n}_k30.json
  [ -f "$OUT" ] || python $EVM --model-path "$FM" --num-agents 6 --num-targets 2 \
      --building-mode medium --target-max-speed 10 $EE --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  echo "[R7] $n arm=$arm $(sr $OUT) closest_$(cl $OUT)" >> $SUM
}
export -f job mp sr cl; export RUN EVM SUM
printf '%s\n' "r7str_s1 1 strict" "r7str_s2 2 strict" "r7str_s3 3 strict" "r7apo_s1 1 apo" \
  | xargs -P 4 -I{} bash -c 'job "{}"'
echo "[R7] ===== DONE $(date) =====" >> $SUM
