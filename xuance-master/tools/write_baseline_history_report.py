"""Build the Chinese change/effect log and local video gallery from saved evidence."""
import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def link(path, label):
    return f"[{label}](<{path.resolve().as_posix()}>)"


def percent(value):
    return f"{float(value):.0%}"


def plot_progress(experiments, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        import subprocess
        font_path = subprocess.check_output(["fc-match", "Noto Serif CJK SC", "--format=%{file}"], text=True).strip()
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=130)
    for axis, experiment in zip(axes, experiments[-2:]):
        rows = experiment["current_evaluations"]
        steps = [int(row["step"]) / 1e6 for row in rows]
        axis.plot(steps, [float(row["safe_success_rate"]) * 100 for row in rows], "o-",
                  color="#16875c", markersize=3, linewidth=1.5, label="当前课程安全捕获")
        axis.plot(steps, [float(row["collision_rate"]) * 100 for row in rows], "o-",
                  color="#ca5757", markersize=3, linewidth=1.5, label="当前课程碰撞")
        previous_level = int(rows[0]["level"])
        start = steps[0]
        for i in range(1, len(rows) + 1):
            level = int(rows[i]["level"]) if i < len(rows) else None
            if level != previous_level:
                finish = steps[i - 1]
                axis.axvspan(start, finish, color=["#c9e0ff", "#d4ead5", "#ead5fb", "#ffe3b8", "#ffd4d8"][previous_level], alpha=.22)
                axis.text((start + finish) / 2, 108, f"第 {previous_level + 1} 级", ha="center", fontsize=10)
                if i < len(rows):
                    axis.axvline(finish, color="#8796a6", linestyle=":", linewidth=.8)
                    start = finish
                    previous_level = level
        final_rows = experiment["final_task_evaluations"]
        axis.plot([int(row["step"]) / 1e6 for row in final_rows],
                  [float(row["safe_success_rate"]) * 100 for row in final_rows], "s--",
                  color="#5373b9", markersize=3, linewidth=1.1, label="固定第五级安全捕获")
        axis.axhline(80, color="#16875c", linewidth=.7, linestyle=":", alpha=.5)
        axis.axhline(10, color="#ca5757", linewidth=.7, linestyle=":", alpha=.5)
        axis.set(title=f"实验 {experiment['experiment']}：按该轮新增步数统计", xlabel="新增环境转移 / 百万", ylabel="比例 / %", ylim=(0, 115))
        axis.set_yticks([0, 10, 20, 40, 60, 80, 100])
        axis.grid(alpha=.2)
        axis.legend(loc="center right", fontsize=8, framealpha=.9)
    fig.suptitle("课程进展与安全性：背景表示实际训练等级；蓝色虚线始终为原第五级任务", fontsize=12)
    fig.text(.5, .025, "每点 100 个无探索完整回合。当前课程难度随等级变化，跨等级折点不能解读为同任务性能退步。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .94))
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--videos", default="outputs/videos/baseline_levels_1_4_20261013_v2")
    parser.add_argument("--extra-videos", action="append", default=[])
    parser.add_argument("--replace-videos", action="append", default=[])
    args = parser.parse_args()
    root = args.root.resolve()
    videos = root / args.videos
    index = read_json(videos / "index.json")
    video_dirs = {row["display_level"]: videos / f"level_{row['display_level']}" for row in index}
    for extra in args.extra_videos:
        extra_root = root / extra
        for row in read_json(extra_root / "index.json"):
            assert row["display_level"] not in video_dirs
            video_dirs[row["display_level"]] = extra_root / f"level_{row['display_level']}"
            index.append(row)
    for replacement in args.replace_videos:
        replacement_root = root / replacement
        for row in read_json(replacement_root / "index.json"):
            level = row["display_level"]
            assert level in video_dirs
            index = [old for old in index if old["display_level"] != level] + [row]
            video_dirs[level] = replacement_root / f"level_{level}"
    index.sort(key=lambda row: row["display_level"])
    snapshot006 = read_json(root / "docs/baseline_exp006_status_2026-10-13.json")
    snapshot007_path = root / "docs/baseline_exp007_status_2026-10-13.json"
    has007 = snapshot007_path.exists()
    snapshot = read_json(snapshot007_path) if has007 else snapshot006
    active_experiment = "007" if has007 else "006"
    acceptance_path = root / "outputs/acceptance/exp007_stage5_candidate_v1/audit.json"
    acceptance = read_json(acceptance_path) if acceptance_path.exists() else None
    final_acceptance_path = root / "outputs/acceptance/exp007_final_acceptance_v1/audit.json"
    final_acceptance = read_json(final_acceptance_path) if final_acceptance_path.exists() else None
    snapshot_time = datetime.fromisoformat(snapshot["time"]).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    experiments = []
    for number in range(1, 8 if has007 else 7):
        name = f"{number:03}"
        method = "maddpg" if number < 5 else "matd3"
        directory = root / f"outputs/results/{method}/baseline_exp{name}"
        with (directory / "curriculum_eval.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        current = [row for row in rows if row["phase"] == "current"]
        final_task = [row for row in rows if int(row["level"]) == 4]
        experiments.append(dict(experiment=name, config=read_json(directory / "config.json"),
                                latest_current=current[-1], latest_final_task=final_task[-1],
                                highest_training_level=max(int(row["level"]) for row in current),
                                current_evaluations=current, final_task_evaluations=final_task))
    plot_path = root / "docs/baseline_change_history_progress_2026-10-13.png"
    plot_progress(experiments, plot_path)
    latest = experiments[-1]["latest_current"]
    completed = snapshot["process"]["status"] == "completed" and snapshot["process"].get("exit_code") == 0
    completed_steps = int(latest["step"]) if completed else snapshot["training"]["step"]
    fourth_video = next(row for row in index if row["display_level"] == 4)
    reached_fifth = snapshot006["training"]["level"] == 4
    if reached_fifth:
        experiments[-1]["highest_training_level"] = 4
    texts = [
        "初始稀疏捕获 + 距离/安全稠密奖励，MADDPG", "仅降低 Critic 学习率", 
        "距离势函数塑形、全队失败惩罚、奖励缩放；调整探索、batch 与前四级课程",
        "MADDPG 中 Actor 每两次 Critic 更新一次", "改用 MATD3：双 Critic、目标动作平滑、延迟 Actor 与目标网络更新",
        "完整策略权重接续第三级；制动风险代价、更低学习率/探索、缩小回放池",
        "接续已实际达到的第五级；制动距离越线后风险代价继续增长"
    ]
    lines = ["# Baseline：从第一级到当前的修改、效果与成功视频", "",
             f"训练状态快照：**{snapshot_time}**。当前实验 {active_experiment} 的训练进程实际存活：{snapshot['live']}。",
             f"{'本轮已完成' if completed else '当前'}新环境转移 {completed_steps:,}，实际训练第 {snapshot['training']['level'] + 1} 级；"
             f"最近整批评估在 {int(latest['step']):,} 步，安全捕获 {percent(latest['safe_success_rate'])}、"
             f"碰撞 {percent(latest['collision_rate'])}。", "",
             "## 1. 先区分课程等级与实验编号", "",
             f"第 1–5 级表示任务难度；001–{active_experiment} 表示训练方案版本。一个方案可以训练多个等级。"
             "同一方案进入下一级时主要改变课程参数，奖励和网络不必重写。", "",
             "评估均关闭探索噪声，使用完整最多 400 个决策步的任务。安全捕获要求整个回合所有追捕机均没有建筑、"
             "边界或队友碰撞。碰撞率与安全捕获率未必相加为 100%，因为还有无碰撞超时。", "",
             "### 初始 baseline 已落实的环境改造", "",
             "- 固定保存五张均匀城市地图，分别 20/40/60/80/100 栋；1000×1000 m，建筑逐级增加。",
             "- 建筑配比普通 50%、高层 40%、地标 10%；边长分别 20/40、40/60、60/80 m，楼顶高度 10–60、60–150、150–350 m。",
             "- 建筑固定，出生位置和方向按环境种子随机；出生安全检查，逃逸机使用带回中/避障的人工势场。",
             "- 四架追捕机，三维加速度动作；控制周期 1 s，物理子步 0.1 s，响应滞后及 jerk 限制。允许绕行或飞越建筑。",
             "- 奖励仅保留捕获、距离与安全；关闭覆盖奖励、轨迹预测和图模块，采用共享 512×512 MLP。",
             "- 修复环境独立随机数、评估种子和并行回合计数，记录奖励分项、真实 Actor/Critic 更新数、配置与源码快照。", "",
             "## 2. 每轮改了什么，实际效果是什么", "",
             "下表是各实验最后一次当前课程评估；不同课程难度不能直接作横向等难度比较。" +
             ("006 已正常结束，完成 3000000 个新转移。" if snapshot006["process"]["status"] == "completed"
              else "006 仍在训练，属于阶段性结果。") +
             ("007 正在第五级接续训练，其结果以本次快照为准。" if has007 and not completed else ""), "",
             "| 实验 | 主要修改 | 最高实际训练级 | 最后当前级：安全 / 碰撞 | 最后第五级测试：安全 / 碰撞 |",
             "|---|---|---:|---|---|"]
    for experiment, description in zip(experiments, texts):
        cur, final = experiment["latest_current"], experiment["latest_final_task"]
        lines.append(f"| {experiment['experiment']} | {description} | {experiment['highest_training_level'] + 1} | "
                     f"{percent(cur['safe_success_rate'])} / {percent(cur['collision_rate'])} | "
                     f"{percent(final['safe_success_rate'])} / {percent(final['collision_rate'])}（{int(final['step']):,} 步） |")
    lines += ["", "### 001 → 002：降低 Critic 学习率", "",
              "Critic 学习率从 0.001 降到 0.0003，Actor 0.0001、batch 1024、512×512 网络及其余条件保持一致；"
              "各训练 300000 个转移。第一级末次安全捕获 13%→26%，碰撞 87%→53%，仍未能升级。"
              "这是一组相同训练种子的对照，尚无跨训练种子结论。", "",
              "### 003：让接近目标与安全失败更明确", "",
              "加入距离势函数塑形 `gamma*Phi(s_next)-Phi(s)`，`Phi=-distance/10`；真实捕获/失败终止时下一势函数为 0，"
              "时间截断仍保留势函数并 bootstrap。奖励整体乘 0.1；训练首次任意机碰撞，全队失败惩罚 -300 并结束回合。"
              "评估继续使用原完整回合规则。", "",
              "同时把 Actor 学习率改为 0.00005、batch 改为 512、探索噪声 0.5→0.05（前 300000 转移衰减），"
              "降低前四级的速度、出生距离和捕获难度。最终第五级任务保持不变。", "",
              "效果：300000 步首次进入第二级；第二级最好安全 72%、碰撞 28%，末次退步到 31%/59%。"
              "说明能学会最初接近，但第二级学习仍不稳定。该轮同时改了多项，不能把收益归因于单项。", "",
              "### 004：仅延迟 Actor 更新", "",
              "Actor 每两次 Critic 更新一次；仍为单 Critic MADDPG。350000 步进入第二级，但末次安全 7%、碰撞 87%。"
              "单独延迟 Actor 没有解决问题。实际更新次数写入日志，没有更新时不填写虚假的零 Actor loss。", "",
              "### 005：采用 MATD3", "",
              "双独立 Critic，TD 目标取较小 Q；目标动作加入标准差 0.2、截断 ±0.5 的平滑噪声；"
              "Actor 每两次更新，并用 Q1 梯度；目标网络只在 Actor 更新时软更新，tau=0.001。"
              "修复 MATD3 初始化并补齐目标动作平滑，使用原生策略推理。", "",
              "沿用 003 的奖励、课程和学习率，从 002 同一 Actor 权重开始，Critic/优化器重新初始化。"
              "400000 步进入第二级、650000 步进入第三级；升级前模型第一级 100%/0%、第二级 98%/1%。"
              "1500000 步的第三级为 76%/22%，仍未通过升级；第五级末次仅 3%/97%。", "",
              "### 006：在已有能力上学习提前制动避障", "",
              "从 005 最后的第三级模型加载 Actor、双 Critic 及全部目标网络，共 36 个参数张量；"
              "优化器和回放池重新建立。前两级资格来自 005 的真实升级记录，006 从第三级接续，后续升级仍需原门槛。", "",
              "建筑/边界代价采用有效间隙：`max(0, clearance - 1.5*v_closing - v_closing²/(2*max_accel))`。"
              "权重 3→5，让高速接近障碍时更早承担代价。这是奖励指标，没有改写动作或阻止真实碰撞。", "",
              "| 参数 | 005 | 006 |", "|---|---:|---:|",
              "| 回放池 | 1000000 | 200000 |", "| 探索起点 / 终点 | 0.5 / 0.05 | 0.1 / 0.03 |",
              "| Actor 学习率 | 0.00005 | 0.00003 |", "| Critic 学习率 | 0.0003 | 0.00015 |",
              "| 学习率结束倍率 | 0.5 | 1.0 |", "| 本轮训练预算 | 1500000 | 3000000 个新转移 |", "",
              "效果：第三级在 300000/350000 新步数的评估为 87%/5%、100%/0%，实际进入第四级。" +
              ("第四级在 2350000/2400000 步的连续评估为 93%/7%、91%/9%，已按原门槛实际进入第五级。"
               if reached_fifth else "第四级安全捕获从初期 22%逐步提高到当前表中的结果，等待连续两次满足升级门槛。") +
              "本轮也同时修改多项，尚不能作单项因果归因。", "",
              "行为测试已随方案扩展到 38 项通过；006 的短 GPU 集成验证确认 496 次 Critic、248 次 Actor 更新、"
              "36 个有限权重张量及正确初始化。短验证仅证明训练链路可运行，性能结论来自整批任务评估。", "",
              "### 006 的第四级碰撞诊断", "",
              "冻结 1550000 步模型（该模型整批第四级安全 83%、碰撞 17%），另用种子 21002 做 64 回合首次碰撞诊断："
              "59 次安全捕获、5 次碰撞、0 次超时。它采用训练的首次碰撞终止规则，不能代替完整任务的新种子验收。", "",
              "5 次碰撞均涉及地标建筑 14 或 67，碰撞楼均是最近建筑观测中的第 1 栋。碰撞前剩余间隙约 1.1–7.4 m，"
              "同方向制动估计约 4.6–22.4 m；3 次仍朝建筑发出正加速度指令。两栋楼顶均为 350 m，"
              "达到飞行高度上限，不能靠升高飞越。该样本提示提前绕行/制动仍不足，不支持“障碍未进入观测”的解释。", "",
              "在这些状态，006 的有效间隙已截到 0，建筑风险代价进入饱和区。" +
              ("006 最后第五级为 84%/16%，最佳为 88%/12%，碰撞仍超门槛。007 因此接续其最佳第五级模型，"
               "仅将有效间隙改为保留负值，使越线后的代价继续增长。42 项行为测试及短 GPU 集成通过，"
               "启动评估复现父模型 88%/12%；后续固定种子最佳达到 94%/6%，但后期仍有波动。"
               "冻结最佳模型的新初始条件复核见第 6 节，未通过验收。"
               if has007 else "可以测试让制动距离越线后的代价继续增长；这只是由代码和碰撞样本提出的下一轮假设，尚无训练效果证明。"), "",
              f"{link(root / 'outputs/results/matd3/baseline_exp006/diagnostics/stage3_snapshot_20261013_0420/audit.json', '诊断核验')} · "
              f"{link(root / 'outputs/results/matd3/baseline_exp006/diagnostics/stage3_snapshot_20261013_0420/collisions_seed21002.json', '64 回合原始诊断')}", "",
              "### 课程学习曲线", "",
              f"![当前课程与原第五级任务评估]({plot_path.as_posix()})", "",
              "绿色/红色是当时正在训练的课程，蓝色虚线始终测原第五级任务。背景标记实际训练等级；"
              "因此跨等级的下降包含任务变难，不能单独认定网络能力退步。", "",
              "## 3. 每一级如何变难，何时进入下一级", "",
              "| 等级 | 建筑 | 逃逸机巡航 / 受威胁速度 m/s | 保持半径 m | 出生半径 m | 垂直展开 m |",
              "|---|---:|---|---:|---:|---:|",
              "| 1 | 20 | 1 / 3 | 50 | 100 | 30 |", "| 2 | 40 | 2 / 5 | 35 | 125 | 40 |",
              "| 3 | 60 | 3 / 7 | 25 | 150 | 50 |", "| 4 | 80 | 3 / 8 | 20 | 175 | 60 |",
              "| 5 | 100 | 3 / 9 | 15 | 200 | 70 |", "",
              f"此表对应 003–{active_experiment}。最初 001/002 使用更难的前几级：逃逸机巡航 3 m/s，最高 5/6/7/8/9 m/s，"
              "保持半径 30/25/20/18/15 m，出生半径均 200 m。因此 001/002 与后续第一级不属于等难度比较。", "",
              "每级至少 100000 个转移；每 50000 步评估 100 个无探索回合；**连续两次安全捕获 ≥80% 且碰撞 ≤10%**"
              "才在下次重置时升级。不会按时间强制跳级。训练末尾最终任务测试也不等同于实际已经在第五级训练。", "",
              "## 4. 各级成功视频", "",
              "红色为逃逸机；蓝/绿/紫/橙为四架追捕机。左侧三维局部轨迹，右上全地图，右下真实三维距离。"
              "视频使用 0.1 s 真实物理子步采样、2×回放；没有改变 1 s 动作决策周期，没有使用动作修正或人工控制。", "",
              "各级视频均在种子 23001 的首个回合成功。所有列出的成功视频都要求全回合所有追捕机零碰撞；"
              "每段动作在未加记录功能的原环境重放，"
              "逐步状态、奖励、终止和捕获结果一致。这里展示成功案例，不能用这些视频估算整体成功率。", "",
              "| 等级 | 对应模型与训练步数 | 模型整批评估：安全 / 碰撞 | 样例捕获时间 | 视频 |",
              "|---|---|---|---:|---|"]
    for row in index:
        level = row["display_level"]
        evaluation = row["checkpoint_eval"]
        movie = video_dirs[level] / "success.mp4"
        lines.append(f"| {level} | {row['experiment']}，{int(evaluation['step']):,} 步 | "
                     f"{percent(evaluation['safe_success_rate'])} / {percent(evaluation['collision_rate'])} | "
                     f"{row['capture_time']:.2f} s | {link(movie, '播放 MP4')} |")
    lines += ["", "视频模型已经单独冻结。早期第四级视频（006 的 1100000 步模型，76%/24%）仍保留在原目录；"
              f"上表当前第四级视频对应 {int(fourth_video['checkpoint_eval']['step']):,} 步模型。"
              "每个版本使用其自身的评估指标，不能把最新指标贴到旧模型视频上。", "",
              "第 1/2 级样例由 P1 连续保持捕获，第 3/4 级样例由 P4 连续保持捕获；其余样例的执行机和捕获类型见各自视频。"
              "当前成功判据允许单机完成，所以画面中其余机器可能尚未靠近目标；"
              "它们不是“四架同时形成几何包围”的证明。覆盖和严格合围属于后续研究问题，当前 baseline 沿用已确定判据。", "",
              f"{link(videos / 'video_gallery.html', '打开各级视频汇总页')} · "
              f"{link(videos / 'all_levels_video_index.json', '各级模型、种子与轨迹审计索引')} · "
              f"{link(videos / 'all_levels_video_audit.json', '各级视频与轨迹哈希核验')}", "",
              "### 第五级", "",
              (f"已在 2400000 步通过第四级门槛并实际进入第五级。最近第五级整批评估（{int(latest['step']):,} 步）："
               f"实验 {active_experiment} 最近整批安全捕获 {percent(latest['safe_success_rate'])}、碰撞 {percent(latest['collision_rate'])}。"
               if reached_fifth else "当前尚未实际进入第五级，不把跨级测试的成功样例当成“第五级已学会”。") +
              ("第五级成功视频已列入上表；每段注明其冻结模型的整批性能，单段成功不代表第五级达标。"
               if 5 in video_dirs else "第五级成功视频会用该级实际训练模型补充。") +
              ("最终300万步模型已完成新种子复核，结果见第7节。" if final_acceptance else
               "第五级目标仍需安全捕获 ≥80%、碰撞 ≤10%；这次种子复核结果见下表。"
               if acceptance else "第五级达标仍需安全捕获 ≥80%、碰撞 ≤10%，并用预留新种子 20001/30001 复核完整任务。"), "",
              "## 5. 后续记录方式", "",
              "每次新方案保存修改原因、参数前后值、起始权重、配置/源码哈希、评估曲线及实际升级步数；"
              "每级成功视频保存冻结模型哈希、课程参数、种子、筛选尝试记录、完整轨迹与无碰撞核验。"
              "报告正文和实时训练状态注明各自快照时间，保留旧视频与旧实验归档。", "",
              "训练采用同一训练种子；常规评估使用固定比较种子，新初始条件复核另外记录。"
              "这些结果不能声称跨训练种子的普遍效果或 loss 已收敛。", "",
              "修改只在服务器独立工作树 `feat-coverage-reaward`；未提交或推送。实验记录放在 docs，不写入仅保留仓库约定的 handoff。", ""]
    if acceptance:
        lines += ["## 6. 95万步保存模型的首次新种子复核（历史失败结果）", "",
                  f"冻结实验 007 的 {int(acceptance['checkpoint_eval']['step']):,} 步最佳模型；"
                  "两次都使用同一模型哈希、100 个无探索完整回合、原 100 栋固定城市和最多 400 步任务，碰撞不提前终止。", "",
                  "| 新种子 | 回合数 | 安全捕获 | 碰撞 | 通过 |", "|---|---:|---:|---:|---|"]
        for row in acceptance["reports"]:
            metrics = row["metrics"]
            lines.append(f"| {row['seed']} | {metrics['episodes']} | {percent(metrics['safe_success_rate'])} | "
                         f"{percent(metrics['collision_rate'])} | {'是' if row['passed'] else '否'} |")
        lines += ["", f"两批合计 {acceptance['episodes']} 回合：安全捕获 {acceptance['aggregate_safe_success_rate']:.1%}、"
                  f"碰撞 {acceptance['aggregate_collision_rate']:.1%}。" +
                  ("本次冻结模型复核通过。" if acceptance["passed"] else
                   "种子 20001 的碰撞 16% 超门槛，合计碰撞 10.5% 也超过 10%；本次验收未通过。"), "",
                  "20001/30001 已在此处使用；后续若依据它们调参或选模型，应作为验证种子，"
                  "不再称为未使用的新种子。" +
                  ("40001/50001的最终模型验收见第7节。" if final_acceptance else "下一次最终复核预留 40001/50001。"), "",
                  link(acceptance_path, "查看冻结模型与完整任务核验"), ""]
    if final_acceptance:
        lines += ["## 7. 300万步最终模型的新种子验收", "",
                  "查看新种子结果前已冻结最终模型；40001/50001未用于训练、课程升级或该模型选择。"
                  "每组100回合，沿用原完整400步任务、100栋城市、逃逸机3–9 m/s、15米保持半径。", "",
                  "| 新种子 | 回合数 | 安全捕获 | 碰撞 | 达到门槛 |", "|---|---:|---:|---:|---|"]
        for item in final_acceptance["reports"]:
            m = item["metrics"]
            lines.append(f"| {item['seed']} | {m['episodes']} | {percent(m['safe_success_rate'])} | "
                         f"{percent(m['collision_rate'])} | {'是' if item['passed'] else '否'} |")
        lines += ["", f"合计200回合：安全捕获{final_acceptance['aggregate_safe_success_rate']:.0%}、"
                  f"碰撞{final_acceptance['aggregate_collision_rate']:.0%}。"
                  + ("两批均达到原门槛，第五级本轮验收通过。" if final_acceptance["passed"] else "未通过全部门槛。"), "",
                  "最终模型仍存在早期课程退化：同一模型第1–5级固定比较安全率69%/76%/93%/85%/94%，"
                  "碰撞30%/24%/7%/15%/6%。历史各级优秀模型的成功视频用于展示当时能力；"
                  "当前第五级视频已更新为同一300万步最终模型，未替换历史失败记录。", "",
                  link(final_acceptance_path, "最终模型完整任务验收核验"), ""]
    report = root / "docs/baseline_change_history_and_videos_2026-10-13.md"
    report.write_bytes("\n".join(lines).encode("utf-8"))
    summary = dict(snapshot_time=snapshot["time"], current_training=snapshot["training"],
                   latest_current_evaluation=latest, experiments=experiments, videos=index,
                   heldout_acceptance=acceptance, final_acceptance=final_acceptance)
    (report.with_suffix(".json")).write_bytes(json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"))
    (videos / "all_levels_video_index.json").write_bytes(json.dumps(index, ensure_ascii=False, indent=2).encode("utf-8"))
    cards = []
    for row in index:
        level = row["display_level"]
        evaluation = row["checkpoint_eval"]
        relative = Path(os.path.relpath(video_dirs[level], videos)).as_posix()
        method = "连续保持 2 s" if row["capture_type"] == "sustained" else "5 m 直接拦截"
        cards.append(f'''<article><h2>第 {level} 级 · {row['building_count']} 栋建筑</h2>
<p>实验 {row['experiment']} · {int(evaluation['step']):,} 步 · 整批安全 {percent(evaluation['safe_success_rate'])} / 碰撞 {percent(evaluation['collision_rate'])}</p>
<video controls preload="metadata" poster="{relative}/capture.png" src="{relative}/success.mp4"></video>
<p>种子 {row['env_seed']} · {row['capture_time']:.2f} s 捕获 · {method} · 全回合零碰撞 · 2×回放</p>
<a href="{relative}/success.mp4" download>下载视频</a> · <a href="{relative}/manifest.json">查看模型和轨迹证据</a></article>''')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Baseline 各级成功捕获视频</title><style>body{margin:0;background:#f3f6fa;color:#18334c;font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1350px;margin:auto;padding:30px}h1{font-size:28px}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:24px}article{background:white;padding:20px;border-radius:14px;box-shadow:0 2px 12px #17334b12}h2{margin:0;font-size:23px}p{font-size:14px;color:#52657a}video{width:100%;border-radius:8px;background:#f7f9fc}a{color:#2469c0}@media(max-width:500px){section{grid-template-columns:1fr}main{padding:15px}}</style>
<main><h1>Baseline 各级：模型真实成功捕获</h1>
<p>这里是成功案例展示。每个视频顶部标明模型整批评估结果；单段成功不能代表整体成功率。红色逃逸机，四种颜色追捕机。视频同时显示真实三维轨迹、全地图和三维距离。</p>
<p>保留原判据：单机 5 m 内直接拦截，或同一架在课程半径内连续保持 2 s。每段使用原生 MATD3 模型，0.1 s 物理采样，无探索、无动作修正，并经原环境重放核对。</p>
<section>''' + "\n".join(cards) + '''</section><p>本页模型与视频为固定快照；最新训练进度及第五级验收请看修改与效果报告。</p></main></html>'''
    (videos / "video_gallery.html").write_bytes(document.encode("utf-8"))
    print(json.dumps(dict(report=str(report), gallery=str(videos / "video_gallery.html"),
                          latest_current=latest, video_count=len(index)), ensure_ascii=False))


if __name__ == "__main__":
    main()
