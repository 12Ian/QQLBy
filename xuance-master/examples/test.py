import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 定义您要绘制的算法列表（请确保对应的结果文件夹存在）
algorithms = ["MADDPG", "IDDPG", "MAPPO", "QMIX"]  # 请根据实际需要补全算法名称
env_id = "3v1_city"

# 定义存放所有实验结果的根目录（根据您的 config.json，通常在 results/ 目录下）
base_dir = "./results"

all_data = []

# 2. 遍历算法，读取数据并进行格式转换
for algo in algorithms:
    # 拼接对应算法的 test_scores.csv 路径
    file_path = os.path.join(base_dir, algo.lower(), env_id, "test_scores.csv")

    if os.path.exists(file_path):
        # 读取数据
        df = pd.read_csv(file_path)

        # 【关键步骤】将宽表（多列 episode）转换为长表（单列数值）
        # test_scores.csv 包含列：step, return_episode_0, return_episode_1...
        # melt 函数将其展平，方便 seaborn 自动计算阴影
        df_melted = df.melt(
            id_vars=["step"], var_name="episode", value_name="Average Return"
        )

        # 添加一列标识当前算法名称
        df_melted["algorithm"] = algo

        all_data.append(df_melted)
    else:
        print(f"找不到数据文件: {file_path}")

# 3. 开始绘图
if all_data:
    # 将所有算法的数据合并到一个大表格中
    final_df = pd.concat(all_data, ignore_index=True)

    # 设置 Seaborn 的绘图风格（完全对应您图片中的背景和网格）
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(12, 7))

    # 使用 seaborn 的 lineplot 绘制折线图
    # errorbar='sd' 表示阴影区域绘制标准差 (Standard Deviation)
    # 如果想绘制 95% 置信区间，可以改为 errorbar=('ci', 95)
    sns.lineplot(
        data=final_df,
        x="step",
        y="Average Return",
        hue="algorithm",  # 按算法名称区分颜色
        errorbar="sd",  # 绘制标准差阴影
        linewidth=2.5,  # 加粗折线
    )

    # 4. 美化图表元素
    plt.title(env_id, fontsize=18, pad=15)
    plt.xlabel("Step", fontsize=16)
    plt.ylabel("Average Return", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # 将图例放置在右下角
    plt.legend(title="algorithm", title_fontsize=16, fontsize=14, loc="lower right")

    # 紧凑布局并保存高清图片
    plt.tight_layout()
    plt.savefig(f"{env_id}_learning_curve.png", dpi=300)
    plt.show()
else:
    print("没有找到任何有效的数据文件，无法绘图。")
