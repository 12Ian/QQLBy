#!/bin/bash
# Chapter 4, retrained with the team ringed around each target at spawn.
#
# Why the previous fix did nothing: raising the closure weight to 16 and switching on the
# strict rule still returned 0.0 on all three seeds, because the problem was never the reward
# or the scoring. reset() places the pursuers relative to the inherited single target and then
# teleports every target to a fresh uniform position, so the team starts ~220 m out and on one
# side. Pursuers cap at 11 m/s against an evader at 10, so the net closing rate is 1 m/s and
# 200 m is not closable inside 400 steps once the evader manoeuvres. Physical reach was
# unreachable before the policy made a single decision; the Apollonius trap was not, which is
# exactly the behaviour that got learned.
#
# Three arms, so the two changes can be told apart:
#   surr   -- surround spawn, original loose criterion
#   sstr   -- surround spawn + strict criterion (the Chapter 3 rule)
#   apo    -- surround spawn + strict + Apollonius allocation frozen at reset
exec 9>/tmp/school_r4.lock
flock -n 9 || { echo "[R4] running"; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine
SUM=p3_results/r4_summary.txt
RUN=xuance-master/examples/run_experiment.py
EVM=xuance-master/examples/evaluate_mt.py
mp () { ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1; }
sr () { grep -oE '"(success_rate|per_target_capture_rate)": [0-9.]+' "$1" 2>/dev/null | tr '\n' ' '; }
echo "[R4] ===== START $(date) =====" >> $SUM
find p3_results -name "train_ch4fix_*.log" -size +80M -delete 2>/dev/null
rm -rf logs/maddpg/apollonius_3d_ch4fix_* logs/maddpg/apollonius_3d_uni_N* 2>/dev/null
echo "[R4] free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)G" >> $SUM

job () {
  set -- $1
  local n=$1; local sd=$2; local arm=$3
  local EX="--surround-spawn --spawn-radius 200"
  case $arm in
    sstr) EX="$EX --strict-capture";;
    apo)  EX="$EX --strict-capture --apollonius-alloc --no-dynamic-alloc";;
  esac
  if [ -z "$(mp $n)" ]; then
    echo "[R4] TRAIN $n arm=$arm seed=$sd $(date +%m-%d\ %H:%M)" >> $SUM
    python $RUN --name "$n" --seed $sd --num-targets 2 --num-agents 6 \
      --env uav_pursuit_apollonius_multitarget_3d --building-mode medium \
      --closure-weight 16 --w-pos 0.6 --w-gap 0.8 --target-max-speed 10 --target-min-speed 8 \
      $EX --steps 5000000 --parallels 8 --eval-interval 250000 --test-episode 20 \
      > p3_results/train_$n.log 2>&1
    echo "[R4] $n exit=$? $(date +%H:%M)" >> $SUM
  fi
  local FM; FM=$(mp $n); [ -z "$FM" ] && { echo "[R4] $n NO_MODEL" >> $SUM; return; }
  local EE="--surround-spawn"
  case $arm in
    sstr) EE="$EE --strict-capture";;
    apo)  EE="$EE --strict-capture --apollonius-alloc --no-dynamic-alloc";;
  esac
  OUT=p3_results/eval_${n}_k30.json
  [ -f "$OUT" ] || python $EVM --model-path "$FM" --num-agents 6 --num-targets 2 \
      --building-mode medium --target-max-speed 10 $EE --k 30 --seed 7 --out "$OUT" > /dev/null 2>&1
  echo "[R4] $n arm=$arm $(sr $OUT) closest=$(grep -oE '"median": [0-9.]+' $OUT 2>/dev/null | head -1)" >> $SUM
}
export -f job mp sr; export RUN EVM SUM
printf '%s\n' "r4surr_s1 1 surr" "r4sstr_s1 1 sstr" "r4sstr_s2 2 sstr" "r4apo_s1 1 apo" \
  | xargs -P 4 -I{} bash -c 'job "{}"'
echo "[R4] ===== DONE $(date) =====" >> $SUM
