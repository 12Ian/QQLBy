#!/usr/bin/env bash
# 用途：后台训练固定 easy 场景，4 栋稀疏建筑、低速目标，不启用课程。
# 启动方式：bash bash/run_easy_background.sh
# 默认 300 万环境步、16 并行；可通过 STEPS、PARALLELS、EXP_NAME 等环境变量覆盖。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXP_BASE="${EXP_BASE:-easy_4v1_k5_3m}"
export BUILDING_MODE="${BUILDING_MODE:-open}"
export TARGET_MIN_SPEED="${TARGET_MIN_SPEED:-3}"
export TARGET_MAX_SPEED="${TARGET_MAX_SPEED:-5}"
exec bash "$SCRIPT_DIR/run_medium_3m_background.sh" "$@"
