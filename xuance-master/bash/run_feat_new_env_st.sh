#!/usr/bin/env bash
# 用途：后台启动旧版 Apollonius 三维单目标环境训练。
# 默认场景：4 架追捕机、空地图、500 万步、16 个并行环境。
# 启动方式：bash bash/run_feat_new_env_st.sh
# 输出位置：训练结果统一写入项目根目录 outputs/。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
cd "$ROOT"

if [[ "${RUN_IN_BACKGROUND:-1}" != "0" && "${UAV_BACKGROUND_CHILD:-0}" != "1" ]]; then
  EXP_BASE="${EXP_BASE:-feat_new_env_st}"
  STAMP="$(date +%Y%m%d_%H%M%S)"
  EXP_NAME="${EXP_NAME:-${EXP_BASE}_${STAMP}}"
  ALGO_HINT="${ALGO:-maddpg}"
  ENV_ID_HINT="apollonius_3d_${EXP_NAME}"
  RUN_DIR_HINT="$ROOT/outputs/runs/$EXP_NAME"
  LAUNCH_LOG="$RUN_DIR_HINT/launcher.log"
  mkdir -p "$RUN_DIR_HINT"

  nohup env UAV_BACKGROUND_CHILD=1 EXP_NAME="$EXP_NAME" "$SCRIPT_PATH" "$@" > "$LAUNCH_LOG" 2>&1 &
  PID=$!

  echo "[launcher] background single-target training started"
  echo "[launcher] pid=$PID"
  echo "[launcher] exp_name=$EXP_NAME"
  echo "[launcher] launcher_log=$LAUNCH_LOG"
  echo "[launcher] console_log=$RUN_DIR_HINT/console.log"
  echo "[launcher] runtime_file=$RUN_DIR_HINT/runtime.txt"
  echo "[launcher] result_dir=$ROOT/outputs/results/$ALGO_HINT/$ENV_ID_HINT"
  echo "[launcher] check_process=ps -p $PID -o pid,etime,cmd"
  echo "[launcher] follow_log=tail -f $RUN_DIR_HINT/console.log"
  exit 0
fi

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV:-rl_env}"

EXP_BASE="${EXP_BASE:-feat_new_env_st}"
STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-${EXP_BASE}_${STAMP}}"
ALGO="${ALGO:-maddpg}"
ENV_NAME="uav_pursuit_coverage_3d"
NUM_AGENTS="${NUM_AGENTS:-4}"
STEPS="${STEPS:-5000000}"
PARALLELS="${PARALLELS:-16}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250000}"
TEST_EPISODE="${TEST_EPISODE:-20}"
BUILDING_MODE="${BUILDING_MODE:-empty}"
SEED="${SEED:-1}"

if [[ -z "${DEVICE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    DEVICE="cuda:0"
  else
    DEVICE="cpu"
  fi
fi

ENV_ID="coverage_3d_${EXP_NAME}"
RUN_DIR="$ROOT/outputs/runs/$EXP_NAME"
CONSOLE_LOG="$RUN_DIR/console.log"
RUNTIME_FILE="$RUN_DIR/runtime.txt"
RESULT_DIR="$ROOT/outputs/results/$ALGO/$ENV_ID"
TB_LOG_DIR="$ROOT/outputs/logs/$ALGO/$ENV_ID"
MODEL_DIR="$ROOT/outputs/models/$ALGO/$ENV_ID"
mkdir -p "$RUN_DIR"

exec > >(tee -a "$CONSOLE_LOG") 2>&1

START_ISO="$(date --iso-8601=seconds)"
START_EPOCH="$(date +%s)"

echo "[run] start_time=$START_ISO"
echo "[run] root=$ROOT"
echo "[run] exp_name=$EXP_NAME"
echo "[run] console_log=$CONSOLE_LOG"
echo "[run] benchmark_results=$RESULT_DIR"
echo "[run] tensorboard_logs=$TB_LOG_DIR"
echo "[run] models=$MODEL_DIR"
echo "[run] runtime_file=$RUNTIME_FILE"
echo "[run] params: algo=$ALGO env=$ENV_NAME num_agents=$NUM_AGENTS steps=$STEPS parallels=$PARALLELS eval_interval=$EVAL_INTERVAL test_episode=$TEST_EPISODE building_mode=$BUILDING_MODE device=$DEVICE seed=$SEED"

CMD=(python -u examples/run_experiment.py
  --name "$EXP_NAME"
  --algo "$ALGO"
  --env "$ENV_NAME"
  --num-agents "$NUM_AGENTS"
  --building-mode "$BUILDING_MODE"
  --surround-spawn
  --steps "$STEPS"
  --parallels "$PARALLELS"
  --eval-interval "$EVAL_INTERVAL"
  --test-episode "$TEST_EPISODE"
  --seed "$SEED"
  --device "$DEVICE")

echo "[run] command: ${CMD[*]}"
set +e
"${CMD[@]}"
STATUS=$?
set -e

END_ISO="$(date --iso-8601=seconds)"
END_EPOCH="$(date +%s)"
ELAPSED=$((END_EPOCH - START_EPOCH))
{
  echo "start_time=$START_ISO"
  echo "end_time=$END_ISO"
  echo "elapsed_seconds=$ELAPSED"
  echo "exit_code=$STATUS"
  echo "console_log=$CONSOLE_LOG"
  echo "benchmark_results=$RESULT_DIR"
  echo "test_scores=$RESULT_DIR/test_scores.csv"
  echo "learning_curve=$RESULT_DIR/learning_curve.csv"
  echo "metadata=$RESULT_DIR/meta_data.json"
  echo "tensorboard_logs=$TB_LOG_DIR"
  echo "models=$MODEL_DIR"
} > "$RUNTIME_FILE"

echo "[run] end_time=$END_ISO elapsed_seconds=$ELAPSED exit_code=$STATUS"
echo "[run] test_scores=$RESULT_DIR/test_scores.csv"
echo "[run] learning_curve=$RESULT_DIR/learning_curve.csv"
echo "[run] metadata=$RESULT_DIR/meta_data.json"
exit "$STATUS"
