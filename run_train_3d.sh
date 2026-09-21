#!/bin/bash
# Full 3D pursuit training (MADDPG, 10M steps, curriculum empty->open->medium).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd ~/UAV_Mine

nohup bash -c '
    echo "=================================================="
    echo "🚀 3D训练开始时间: $(date "+%Y-%m-%d %H:%M:%S")"
    echo "=================================================="
    START_TIME=$(date +%s)
    python xuance-master/examples/train_3d.py 2>&1 | tr "\r" "\n" | grep --line-buffered -v -E "it/s|%\|"
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo "=================================================="
    echo "✅ 3D训练结束时间: $(date "+%Y-%m-%d %H:%M:%S")"
    echo "⏱️  总耗时: $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
    echo "=================================================="
' > train3d_log.txt 2>&1 < /dev/null &

echo "🚀 3D full training started in background (pid $!)."
echo "👀 tail -f ~/UAV_Mine/train3d_log.txt"
