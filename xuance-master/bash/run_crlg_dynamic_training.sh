#!/usr/bin/env bash
# 用途：先做 CRL-G 日志字段检查，再运行 300 万步训练和 100 回合独立评估。
# 适用任务：CRL-G 三维制导方法迁移；必须传入唯一的实验名称。
# 启动方式：bash bash/run_crlg_dynamic_training.sh <实验名称>
# 输出位置：训练结果统一写入项目根目录 outputs/。
set -Eeuo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python=/home/ryy/miniconda3/envs/rl_env/bin/python
run_name="${1:?pass a unique run name}"
run_dir="$repo/outputs/runs/$run_name"
smoke_name="${run_name}_record_check"
mkdir -p "$run_dir"
cd "$repo"
export PYTHONUNBUFFERED=1

on_exit() {
    code=$?
    printf '%s\n' "$code" > "$run_dir/exit_code"
    if (( code != 0 )); then printf 'failed\n' > "$run_dir/stage.txt"; fi
}
trap on_exit EXIT

git rev-parse HEAD > "$run_dir/git_head.txt"
git branch --show-current > "$run_dir/git_branch.txt"
git status --short > "$run_dir/git_status.txt"
sha256sum examples/crlg_train.py examples/crlg_evaluate.py \
    xuance/configs/maddpg/crlg_3d.yaml \
    xuance/environment/multi_agent_env/crlg_3d/*.py \
    xuance/torch/agents/core/off_policy_marl.py > "$run_dir/code_sha256.txt"

model_path() {
    "$python" - "$repo/outputs/models/crlg_3d/crlg_3d_$1" <<'PY'
from pathlib import Path
import sys
models = list(Path(sys.argv[1]).glob('seed_*/final_train_model.pth'))
if len(models) != 1:
    raise SystemExit(f'expected one checkpoint, found {len(models)} in {sys.argv[1]}')
print(models[0])
PY
}

printf 'record_check_training\n' > "$run_dir/stage.txt"
"$python" examples/crlg_train.py --device cuda:0 --steps 80 --parallels 1 \
    --seed 1 --run-name "$smoke_name" --eval-interval 40 --eval-episodes 2 \
    --eval-output "$run_dir/record_check_periodic_success.jsonl" \
    > "$run_dir/record_check_train.log" 2>&1
smoke_model="$(model_path "$smoke_name")"
printf '%s\n' "$smoke_model" > "$run_dir/record_check_model.txt"

printf 'record_check_evaluation\n' > "$run_dir/stage.txt"
"$python" examples/crlg_evaluate.py --device cuda:0 --model "$smoke_model" \
    --episodes 1 --team-size 3 --target-mode CA --seed 2026 \
    --save-trajectories --output "$run_dir/record_check_evaluation.json" \
    > "$run_dir/record_check_evaluation.log" 2>&1

"$python" - "$repo/outputs/logs/crlg_3d/crlg_3d_$smoke_name" \
    "$run_dir/record_check_evaluation.json" "$run_dir/record_check_report.json" \
    "$run_dir/record_check_periodic_success.jsonl" <<'PY'
import json
import math
import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_root, evaluation_path, report_path, periodic_path = map(Path, sys.argv[1:])
events = list(log_root.rglob('events.out.tfevents*'))
if not events:
    raise SystemExit('no TensorBoard event files')
scalars = {}
for event in events:
    acc = EventAccumulator(str(event))
    acc.Reload()
    for tag in acc.Tags()['scalars']:
        scalars.setdefault(tag, []).extend(acc.Scalars(tag))
tags = set(scalars)
required = [
    'Train-Results/Episode-Steps', 'Train-Results/Episode-Rewards',
    *[f'Train-SubRewards-agent_0/{name}' for name in (
        'coverage_raw', 'prediction_raw', 'energy_raw', 'coverage',
        'prediction', 'energy', 'terminal_raw', 'terminal', 'boundary')],
    *[f'Train-CRLG/{name}' for name in (
        'initial_encounter_time_s', 'final_encounter_time_s',
        'encounter_time_change_s', 'terminal_min_miss_m',
        'closest_approach_m', 'final_coverage', 'active_agents',
        'invalid_episode', 'max_steps', 'terminal_phase_locked')],
    'Eval-CRLG/success_rate_5m_all',
    'Eval-CRLG/invalid_episode_rate',
]
missing = [prefix for prefix in required if not any(
    tag == prefix or tag.startswith(prefix + '/') for tag in tags)]
if missing:
    raise SystemExit(f'missing TensorBoard scalars: {missing}')
for prefix in required:
    tag = next(tag for tag in tags if tag == prefix or tag.startswith(prefix + '/'))
    if not scalars[tag] or not all(math.isfinite(item.value) for item in scalars[tag]):
        raise SystemExit(f'empty/nonfinite TensorBoard scalar: {tag}')

payload = json.loads(evaluation_path.read_text(encoding='utf-8'))
episode = payload['episodes'][0]
required_episode = (
    'initial_encounter_time_s', 'encounter_time_s', 'termination_reason',
    'episode_steps', 'png_steps', 'terminal_phase_locked',
    'terminal_min_miss_m', 'closest_approach_m', 'reward_components',
    'command_energy', 'coverage_mean', 'trajectory')
missing_episode = [key for key in required_episode if key not in episode]
if missing_episode:
    raise SystemExit(f'missing evaluation fields: {missing_episode}')
required_step = ('estimated_encounter_time_s', 'guidance_phase',
                 'coverage_probability', 'predicted_projected_miss_m',
                 'actual_miss_m', 'actions_2d')
if not episode['trajectory'] or any(key not in step for step in episode['trajectory']
                                    for key in required_step):
    raise SystemExit('missing trajectory fields')
if 'max_steps' not in payload['summary']['termination_reasons']:
    raise SystemExit('missing max_steps summary field')
periodic = [json.loads(line) for line in periodic_path.read_text(encoding='utf-8').splitlines()]
if [item['step'] for item in periodic] != [40, 80]:
    raise SystemExit(f'unexpected periodic evaluation steps: {periodic}')
for item in periodic:
    if not 0 <= item['success_rate_5m_all'] <= 1 or item['episodes'] != 2:
        raise SystemExit(f'invalid periodic success result: {item}')
report = {'passed': True, 'event_file_count': len(events),
          'log_root': str(log_root),
          'checked_tensorboard_prefixes': required,
          'checked_episode_fields': required_episode,
          'checked_trajectory_fields': required_step,
          'smoke_episode_steps': episode['episode_steps'],
          'smoke_termination_reason': episode['termination_reason'],
          'periodic_evaluation_steps': [item['step'] for item in periodic]}
report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
PY

printf 'full_training\n' > "$run_dir/stage.txt"
"$python" examples/crlg_train.py --device cuda:0 --steps 3000000 --parallels 16 \
    --seed 1 --run-name "$run_name" --eval-interval 100000 --eval-episodes 20 \
    --eval-output "$run_dir/periodic_success.jsonl" \
    > "$run_dir/full_train.log" 2>&1
full_model="$(model_path "$run_name")"
printf '%s\n' "$full_model" > "$run_dir/full_model.txt"

printf 'full_evaluation\n' > "$run_dir/stage.txt"
"$python" examples/crlg_evaluate.py --device cuda:0 --model "$full_model" \
    --episodes 100 --seed 2026 --save-trajectories \
    --output "$run_dir/full_evaluation.json" \
    > "$run_dir/full_evaluation.log" 2>&1
printf 'completed\n' > "$run_dir/stage.txt"
