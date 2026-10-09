#!/usr/bin/env bash
# 用途：后台启动 Coverage 三维中密度可变建筑场景训练。
# 默认场景：4 追 1、最近 5 栋建筑特征、固定 medium 地图、300 万步、16 个并行环境。
# 启动方式：bash bash/run_medium_3m_background.sh
# 输出位置：训练结果统一写入项目根目录 outputs/。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
cd "$ROOT"

EXP_BASE="${EXP_BASE:-medium_4v1_k5_3m}"
STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-${EXP_BASE}_${STAMP}}"
ALGO_HINT="${ALGO:-maddpg}"
ENV_ID_HINT="coverage_3d_${EXP_NAME}"
RUN_DIR_HINT="$ROOT/outputs/runs/$EXP_NAME"
LAUNCH_LOG="$RUN_DIR_HINT/launcher.log"

if [[ "${RUN_IN_BACKGROUND:-1}" != "0" && "${UAV_BACKGROUND_CHILD:-0}" != "1" ]]; then
  mkdir -p "$RUN_DIR_HINT"
  nohup env UAV_BACKGROUND_CHILD=1 EXP_NAME="$EXP_NAME" "$SCRIPT_PATH" "$@" > "$LAUNCH_LOG" 2>&1 &
  PID=$!

  echo "[launcher] background training started"
  echo "[launcher] pid=$PID"
  echo "[launcher] exp_name=$EXP_NAME"
  echo "[launcher] launcher_log=$LAUNCH_LOG"
  echo "[launcher] console_log=$RUN_DIR_HINT/console.log"
  echo "[launcher] runtime_file=$RUN_DIR_HINT/runtime.txt"
  echo "[launcher] preflight_notes=$RUN_DIR_HINT/preflight_notes.md"
  echo "[launcher] result_dir=$ROOT/outputs/results/$ALGO_HINT/$ENV_ID_HINT"
  echo "[launcher] model_dir=$ROOT/outputs/models/$ALGO_HINT/$ENV_ID_HINT"
  echo "[launcher] check_process=ps -p $PID -o pid,etime,cmd"
  echo "[launcher] follow_log=tail -f $RUN_DIR_HINT/console.log"
  exit 0
fi

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV:-rl_env}"

ALGO="${ALGO:-maddpg}"
ENV_NAME="uav_pursuit_coverage_3d"
NUM_AGENTS="${NUM_AGENTS:-4}"
STEPS="${STEPS:-3000000}"
PARALLELS="${PARALLELS:-16}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
TEST_EPISODE="${TEST_EPISODE:-20}"
BUILDING_MODE="${BUILDING_MODE:-medium}"
OBSTACLE_GAT_K="${OBSTACLE_GAT_K:-5}"
TARGET_MIN_SPEED="${TARGET_MIN_SPEED:-4}"
TARGET_MAX_SPEED="${TARGET_MAX_SPEED:-12}"
SEED="${SEED:-1}"
START_NOISE="${START_NOISE:-}"
END_NOISE="${END_NOISE:-}"
LOAD_FROM="${LOAD_FROM:-}"
POLICY_ONLY_LOAD="${POLICY_ONLY_LOAD:-0}"
if [[ "$POLICY_ONLY_LOAD" == "1" && -z "$LOAD_FROM" ]]; then
  echo "[run] POLICY_ONLY_LOAD=1 需要同时指定 LOAD_FROM" >&2
  exit 2
fi
if [[ -n "$LOAD_FROM" && ! -f "$LOAD_FROM" ]]; then
  echo "[run] 检查点不存在: $LOAD_FROM" >&2
  exit 2
fi

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
PREFLIGHT_FILE="$RUN_DIR/preflight_notes.md"
RESULT_DIR="$ROOT/outputs/results/$ALGO/$ENV_ID"
TB_LOG_DIR="$ROOT/outputs/logs/$ALGO/$ENV_ID"
MODEL_DIR="$ROOT/outputs/models/$ALGO/$ENV_ID"
mkdir -p "$RUN_DIR"

exec > >(tee -a "$CONSOLE_LOG") 2>&1

START_ISO="$(date --iso-8601=seconds)"
START_EPOCH="$(date +%s)"
GIT_BRANCH="$(git branch --show-current 2>/dev/null || true)"
GIT_HEAD="$(git rev-parse --short HEAD 2>/dev/null || true)"
GIT_STATUS="$(git status --short 2>/dev/null || true)"

cat > "$PREFLIGHT_FILE" <<EOF
# Coverage 4v1 K5 3M Preflight Notes

start_time: $START_ISO
root: $ROOT
branch: $GIT_BRANCH
head: $GIT_HEAD
experiment: $EXP_NAME

## Training Setup

- algorithm: $ALGO
- environment: $ENV_NAME
- building_mode: $BUILDING_MODE
- num_pursuers: $NUM_AGENTS
- evader: 1
- steps: $STEPS
- parallels: $PARALLELS
- eval_interval: $EVAL_INTERVAL
- test_episode: $TEST_EPISODE
- nearest_buildings: $OBSTACLE_GAT_K
- target_speed_range: [$TARGET_MIN_SPEED, $TARGET_MAX_SPEED]
- seed: $SEED
- device: $DEVICE
- start_noise_override: ${START_NOISE:-algorithm_config}
- end_noise_override: ${END_NOISE:-algorithm_config}
- load_from: ${LOAD_FROM:-none}
- policy_only_load: $POLICY_ONLY_LOAD

## RL Engineering Assessment

This run cannot mathematically guarantee convergence. MADDPG in this environment is
off-policy, multi-agent, continuous-control training with non-stationarity from the
other pursuers, sparse terminal capture signals, dense shaped rewards, obstacle
avoidance penalties, and a moving evader. These factors make strict convergence
guarantees unrealistic.

The current setup is still reasonable as a training experiment because the policy
receives coverage preparation and strict coverage terms, relative target information,
building geometry and boundary features, and periodic evaluation. The main risks are
local optima in route preparation, over-avoidance near buildings, reward scale
imbalance after building-collision penalties, and limited generalization from a fixed map.

Practical convergence should be judged by evaluation curves rather than by the final
training return alone:

- success_rate should rise and stabilize.
- collision_rate, especially building collision, should fall.
- mean episode return should improve without becoming dominated by collision penalty.
- capture/min-distance diagnostics should improve in evaluation episodes.

## Output Locations

- console_log: $CONSOLE_LOG
- runtime_file: $RUNTIME_FILE
- result_dir: $RESULT_DIR
- test_scores: $RESULT_DIR/test_scores.csv
- learning_curve: $RESULT_DIR/learning_curve.csv
- tensorboard_logs: $TB_LOG_DIR
- models: $MODEL_DIR

## Git Status At Launch

\`\`\`
$GIT_STATUS
\`\`\`
EOF

echo "[run] start_time=$START_ISO"
echo "[run] root=$ROOT"
echo "[run] branch=$GIT_BRANCH head=$GIT_HEAD"
echo "[run] exp_name=$EXP_NAME"
echo "[run] console_log=$CONSOLE_LOG"
echo "[run] preflight_notes=$PREFLIGHT_FILE"
echo "[run] benchmark_results=$RESULT_DIR"
echo "[run] tensorboard_logs=$TB_LOG_DIR"
echo "[run] models=$MODEL_DIR"
echo "[run] runtime_file=$RUNTIME_FILE"
echo "[run] params: algo=$ALGO env=$ENV_NAME num_agents=$NUM_AGENTS steps=$STEPS parallels=$PARALLELS eval_interval=$EVAL_INTERVAL test_episode=$TEST_EPISODE building_mode=$BUILDING_MODE obstacle_gat_k=$OBSTACLE_GAT_K target_min_speed=$TARGET_MIN_SPEED target_max_speed=$TARGET_MAX_SPEED device=$DEVICE seed=$SEED"

CMD=(python -u examples/run_experiment.py
  --name "$EXP_NAME"
  --algo "$ALGO"
  --env "$ENV_NAME"
  --num-agents "$NUM_AGENTS"
  --surround-spawn
  --use-obstacle-gat
  --obstacle-gat-k "$OBSTACLE_GAT_K"
  --steps "$STEPS"
  --parallels "$PARALLELS"
  --eval-interval "$EVAL_INTERVAL"
  --test-episode "$TEST_EPISODE"
  --target-min-speed "$TARGET_MIN_SPEED"
  --target-max-speed "$TARGET_MAX_SPEED"
  --seed "$SEED"
  --device "$DEVICE")

# 固定建筑级别；当前配置关闭课程。
if [[ -n "$BUILDING_MODE" ]]; then
  CMD+=(--building-mode "$BUILDING_MODE")
fi
if [[ -n "$START_NOISE" ]]; then
  CMD+=(--start-noise "$START_NOISE")
fi
if [[ -n "$END_NOISE" ]]; then
  CMD+=(--end-noise "$END_NOISE")
fi
if [[ -n "$LOAD_FROM" ]]; then
  CMD+=(--load-from "$LOAD_FROM")
  if [[ "$POLICY_ONLY_LOAD" == "1" ]]; then
    CMD+=(--policy-only-load)
  fi
fi

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
  echo "preflight_notes=$PREFLIGHT_FILE"
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
