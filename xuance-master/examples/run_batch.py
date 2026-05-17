import subprocess
import time

def run_experiments():
    # ================= 1. 手动定义你要跑的所有命令 =================
    
    # 定义真实的绝对路径
    train_script_path = r"/home/ryy/UAV_Project/xuance-master/examples/train.py"

    commands_to_run = [
        # [
        #     "python", train_script_path,  # <--- 在这里替换成绝对路径变量
        #     "--algo", "maddpg",
        #     "--env", "uav_pursuit_apollonius_obs",
        #     "--env-id", "apollonius_obs",
        #     "--device", "cuda:0",
        # ],
        # [
        #     "python", train_script_path,  # <--- 这里也替换
        #     "--algo", "maddpg",
        #     "--env", "uav_pursuit_apollonius_obs_4",
        #     "--env-id", "apollonius_obs_4",
        #     "--device", "cuda:0",
        # ],
        [
            "python", train_script_path,  # <--- 这里也替换
            "--algo", "maddpg",
            "--env", "uav_pursuit_apollonius_obs_5",
            "--env-id", "apollonius_obs_5",
            "--device", "cuda:0",
        ],
    ]

    total_exps = len(commands_to_run)
    print(f"🚀 即将开始串行跑批，共计 {total_exps} 个自定义任务！\n")

    # ================= 2. 依次按顺序执行 =================
    for i, cmd in enumerate(commands_to_run, start=1):
        cmd_string = " ".join(cmd)
        print(f"{'='*60}")
        print(f"▶️ 正在执行任务 [{i}/{total_exps}]:")
        print(f"💻 命令: {cmd_string}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # check=True 表示如果这条命令报错，会抛出异常被下面截获
            subprocess.run(cmd, check=True)
            
            elapsed = (time.time() - start_time) / 60
            print(f"\n✅ 任务 {i} 顺利完成! (耗时: {elapsed:.2f} 分钟)\n")
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 任务 {i} 运行失败崩溃！")
            print(f"⚠️ 错误信息: {e}\n")

    print("🎉 所有列出的实验指令已全部跑完！")

if __name__ == "__main__":
    run_experiments()