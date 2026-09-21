#!/bin/bash
# Stable N=4 curriculum: bootstrap coordination in empty space, introduce four
# buildings, then fine-tune in the medium (16-building) benchmark.  The model
# architecture is unchanged at every stage, so checkpoints are load-compatible.
set -u

exec 9>/tmp/run_n4_stable.lock
flock -n 9 || { echo "[n4-stable] another run is active"; exit 0; }

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine

RUN=xuance-master/examples/run_experiment.py
EVAL=xuance-master/examples/evaluate_3d.py
OUT=p3_results/n4_stable
mkdir -p "$OUT"

# 20 m is below the nominal tetrahedral spacing at capture (~24.5 m), so the
# barrier discourages bunching without opposing a valid four-agent enclosure.
BASE="--num-agents 4 --target-max-speed 6 --closure-weight 8 --w-pos 0.6 --w-gap 0.8 --separation-weight 4 --separation-distance 20 --parallels 8"

model_path () {
  ls models/maddpg/apollonius_3d_$1/*/final_train_model.pth 2>/dev/null | head -1
}

train_stage () { # name density steps seed [checkpoint] [soft|hard]
  local name=$1 density=$2 steps=$3 seed=$4 init=${5:-} collision_mode=${6:-hard}
  local existing
  existing=$(model_path "$name")
  if [ -n "$existing" ]; then
    echo "$existing"
    return 0
  fi

  local load_args=()
  if [ -n "$init" ]; then
    # New stage, new replay distribution: retain policy/targets but use a fresh
    # optimizer and low exploration noise instead of restoring stale moments
    # and restarting at the destructive default noise=1.0.
    load_args=(--load-from "$init" --policy-only-load --start-noise 0.10 --end-noise 0.01)
  else
    load_args=(--start-noise 1.0 --end-noise 0.01)
  fi
  local collision_args=()
  if [ "$collision_mode" = "soft" ]; then
    collision_args=(--soft-collisions)
  fi
  echo "[n4-stable] train $name density=$density steps=$steps seed=$seed" >&2
  python "$RUN" --name "$name" --seed "$seed" --building-mode "$density" \
    $BASE --steps "$steps" --eval-interval 250000 --test-episode 20 \
    "${load_args[@]}" "${collision_args[@]}" > "$OUT/train_${name}.log" 2>&1

  existing=$(model_path "$name")
  if [ -z "$existing" ]; then
    echo "[n4-stable] missing checkpoint after $name" >&2
    return 1
  fi
  echo "$existing"
}

for seed in 1 2; do
  empty=$(train_stage "n4fix_empty_s${seed}" empty 1000000 "$seed") || exit 1
  open=$(train_stage "n4fix_open_soft_s${seed}" open 1000000 "$seed" "$empty" soft) || exit 1
  medium_soft=$(train_stage "n4fix_medium_soft_s${seed}" medium 2000000 "$seed" "$open" soft) || exit 1
  medium=$(train_stage "n4fix_medium_hard_s${seed}" medium 2000000 "$seed" "$medium_soft" hard) || exit 1

  python "$EVAL" --policy model --model-path "$medium" --seed 7 \
    --building-mode medium --num-agents 4 --target-max-speed 6 --k 30 \
    --out "$OUT/eval_n4fix_medium_s${seed}.json" \
    > "$OUT/eval_n4fix_medium_s${seed}.log" 2>&1
  grep RESULT "$OUT/eval_n4fix_medium_s${seed}.log" || true
done

echo "[n4-stable] completed"
