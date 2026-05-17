import imageio
from copy import deepcopy
import numpy as np
from xuance.engine import get_runner
from xuance.environment import make_envs

if __name__ == '__main__':
    # 1. 获取 runner
    runner = get_runner(algo="maddpg", env="uav_pursuit_apollonius_obs_5", env_id="apollonius_obs_5", config_path=r"/home/ryy/UAV_Project/xuance-master/xuance/configs/maddpg/uav_pursuit_apollonius_obs_5.yaml")

    # 2. 加载你刚刚跑出来的 best_model
    runner.agent.load_model(r"/home/ryy/UAV_Project/models/maddpg/apollonius_obs_5/seed_1_2026_0321_113154/final_train_model.pth")

    # 3. 创建测试专用的单线程环境
    config_test = deepcopy(runner.config)
    config_test.vectorize = "DummyVecMultiAgentEnv" 
    config_test.parallels = 1
    config_test.render = True
    config_test.render_mode = "rgb_array" 
    test_envs = make_envs(config_test)

    # 4. 初始化环境
    obs, info = test_envs.reset()
    frames = []

    print("🚀 开始运行并录制画面！（由于在服务器运行，将直接保存为视频文件）")

    # ==================== 第一阶段：运行与录制 ====================
    for step in range(10000): 
        action_dict = runner.agent.action(obs_dict=obs, test_mode=True)
        actions_execute = action_dict['actions']
        
        next_obs, rewards, terminated, truncated, info = test_envs.step(actions_execute)

        # 抓取当前帧并存入列表
        img_rgb = test_envs.render("rgb_array")[0]
        frames.append(img_rgb) 
            
        obs = next_obs

        if all(terminated[0].values()) or truncated[0]:
            print(f"✅ 在第 {step} 步环境到达终点。")
            break

    test_envs.close()

    # ==================== 第二阶段：将画面保存为视频 ====================
    if len(frames) > 0:
        video_path = "uav_replay.mp4"
        print(f"🎬 正在将录制画面压制为视频：{video_path} ... (可能需要几秒钟)")
        
        # 使用 imageio 保存为 mp4 视频，fps为帧率
        imageio.mimsave(video_path, frames, fps=30)
        
        print(f"✅ 视频保存成功！请在左侧文件目录中找到 {video_path}，下载到你自己的电脑上观看。")
    else:
        print("⚠️ 没有录制到任何画面。")