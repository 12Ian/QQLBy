#!/bin/bash

# 1. 初始化 Conda
source ~/miniconda3/etc/profile.d/conda.sh

# 2. 激活你的强化学习专属环境
conda activate rl_env

# 3. 确保进入属于你自己的【新目录】（修正了这里的 Bug！）
cd ~/UAV_Mine

# 4. 终极过滤魔法 + 运行时间记录
nohup bash -c '
    echo "=================================================="
    echo "🚀 训练开始时间: $(date "+%Y-%m-%d %H:%M:%S")"
    echo "=================================================="
    
    START_TIME=$(date +%s)
    
    # 运行代码并过滤进度条
    python xuance-master/examples/train.py 2>&1 | tr "\r" "\n" | grep --line-buffered -v -E "it/s|%\|"
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo "=================================================="
    echo "✅ 训练结束时间: $(date "+%Y-%m-%d %H:%M:%S")"
    echo "⏱️  总执行耗时: $((ELAPSED / 3600))小时 $(((ELAPSED / 60) % 60))分钟 $((ELAPSED % 60))秒"
    echo "=================================================="
' > train_log.txt 2>&1 &

# 5. 打印提示信息
echo "🚀 训练任务已在后台成功启动！"
echo "📄 进度条已被拦截，train_log.txt 现在非常干净，并记录了时间。"
echo "👀 随时输入 tail -f train_log.txt 查看实时进度。"