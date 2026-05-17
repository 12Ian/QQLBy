import os
import argparse
import subprocess

# 把你的日志基础路径配置在这里
BASE_LOG_DIR = r"/home/ryy/UAV_Mine/logs/maddpg/apollonius_obs_5"  # 修改为你的实际路径

def get_latest_log_dir(base_dir):
    """获取指定目录下最新的文件夹"""
    # 获取该目录下所有文件夹的完整路径
    dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not dirs:
        return None
    
    # 按照文件夹的修改时间进行排序，排在最后的就是最新的
    dirs.sort(key=os.path.getmtime)
    return dirs[-1]

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="一键启动 TensorBoard")
    parser.add_argument("-n", "--name", type=str, default=None, 
                        help="指定的日志文件夹名称 (例如: seed_1_2026_0308_204246)")
    
    args = parser.parse_args()

    # 判断是用户指定了名称，还是自动寻找最新
    if args.name:
        target_dir = os.path.join(BASE_LOG_DIR, args.name)
        if not os.path.exists(target_dir):
            print(f"❌ 找不到指定的目录: {target_dir}")
            return
    else:
        target_dir = get_latest_log_dir(BASE_LOG_DIR)
        if not target_dir:
            print(f"❌ 在 {BASE_LOG_DIR} 中没有找到任何日志。")
            return

    print(f"🚀 准备启动 TensorBoard...")
    print(f"📂 正在读取日志: {target_dir}")
    print("-" * 50)
    
    # 启动 TensorBoard
    # 注意：运行此脚本的终端必须已经激活了 xuance_marl 环境
    try:
        subprocess.run(["tensorboard", "--logdir", target_dir])
    except KeyboardInterrupt:
        print("\n⏹️ 已手动关闭 TensorBoard。")

if __name__ == "__main__":
    main()