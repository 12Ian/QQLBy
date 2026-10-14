"""Write a dated environment assessment from a verified, frozen monitoring snapshot."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/reviews/exp008_environment_audit_2026-10-13_2106/audit.json"
audit = json.loads(AUDIT.read_text(encoding="utf-8"))
archive = ROOT / "docs/exp008_environment_audit_2026-10-13_2106_v2.zip"
with zipfile.ZipFile(archive) as snapshot:
    for name, record in audit["files"].items():
        assert hashlib.sha256(snapshot.read(name)).hexdigest() == record["sha256"], name
    eval_bytes = snapshot.read("outputs/results/matd3/baseline_exp008/curriculum_eval.csv")
assert not audit["source_hash_mismatches"]
status = audit["status"]
results = ROOT / "outputs/results/matd3/baseline_exp008"
evaluations = list(csv.DictReader(eval_bytes.decode("utf-8").splitlines()))
followup_path = AUDIT.parent / "followup_2114.json"
followup = json.loads(followup_path.read_text(encoding="utf-8")) if followup_path.exists() else None
if followup:
    seen = {(r["step"], r["phase"], r["level"]) for r in evaluations}
    for row in followup["latest_current"] + [followup["latest_final_task"]]:
        if (row["step"], row["phase"], row["level"]) not in seen:
            evaluations.append(row)
    evaluations.sort(key=lambda r: int(r["step"]))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
colors = ["#2366b3", "#da7c22", "#329658", "#8b5dab", "#cf4455"]
for level in range(5):
    rows = [r for r in evaluations if r["phase"] == "current" and int(r["level"]) == level]
    if not rows:
        continue
    x = [int(r["step"]) / 1e6 for r in rows]
    for ax, key in zip(axes, ["safe_success_rate", "collision_rate"]):
        ax.plot(x, [float(r[key]) * 100 for r in rows], "o-", color=colors[level],
                markersize=4, label=f"第{level + 1}级")
for ax, title, threshold in zip(axes, ["正式评估：安全捕获率", "正式评估：全回合碰撞率"], [80, 10]):
    ax.axhline(threshold, color="#666666", linestyle="--", linewidth=1,
               label=f"升级门槛 {threshold}%")
    for step in [1.45, 2.05, 2.20]:
        ax.axvline(step, color="#cccccc", linewidth=.8)
    ax.set(xlabel="累计环境转移 / 百万", ylabel="百分比", ylim=(-3, 103), title=title)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc="best")
fig.suptitle("实验008：各级100回合无探索评估；同一固定城市与评估流10001")
image_path = ROOT / ("docs/baseline_exp008_curriculum_2026-10-13_2114.png" if followup else
                     "docs/baseline_exp008_curriculum_2026-10-13_2105.png")
fig.savefig(image_path, dpi=160)
plt.close(fig)

table = []
for row in audit["levels"]:
    ev = row["last_current_evaluation"]
    if ev:
        tail = f"{int(ev['step']):,} | {float(ev['success_rate']):.0%} | {float(ev['safe_success_rate']):.0%} | {float(ev['collision_rate']):.0%} | {float(ev['mean_episode_steps']):.2f}"
    else:
        tail = "尚未训练 | — | — | — | —"
    table.append(f"| {row['level']} | {row['completed_episodes']:,} | {row['completed_transitions']:,} | {tail} |")

def link(relative):
    return (ROOT / relative).as_posix()

text = f"""# 实验008：课程、动力学、环境信息与追捕能力评估

核查时间：2026-10-13 21:05（北京时间）。状态快照：{status['step']:,}次环境转移，正在第{status['level'] + 1}级，预算3,000,000。训练和监督进程均在运行。29个训练源文件的SHA256与启动时metadata一致。

本文件是进行中的观察。下表第1–3级为各级离开前的历史成绩，不代表当前第4级权重仍保持这些成绩；训练结束后需要用同一最终模型重新测第1–5级。每项正式评估100回合、8并行、无探索；全回合统计碰撞，机间接触本轮不计。

## 1. 最新训练证据

| 级别 | 已完成回合 | 完整回合转移 | 最新当前级评估步数 | 捕获率 | 安全捕获率 | 碰撞率 | 平均决策步数 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

- 第1级在145万转移时升级；第2级在205万时升级；第3级在220万时升级。
- 第4级225万、230万、235万转移的安全成功率为25%、33%、48%；碰撞率为75%、67%、52%。235万建筑碰撞52%、边界碰撞34%；同一回合可能两者都发生，不能相加当总碰撞率。
- 第5级225万的跨级诊断捕获15%、安全0%、碰撞100%。它没有进入第5级训练，这项不是第5级训练成绩。
- 实际Critic更新{status['updates']:,}次、Actor更新{status['actor_updates']:,}次，来自训练状态中的真实计数；较低频CSV窗口可能稍旧。不能依据loss正负或有限值声称策略收敛。
- 完整回合转移按回合所属课程统计。升级时16个并行环境可能仍有旧级回合，因此每级完整回合转移与两个升级时刻之差可能不完全相等。进行中回合尚未计入。
- 本轮由007最终模型初始化，重新建立回放池；不是随机权重训练。

![当前课程曲线]({image_path.as_posix()})

这张图仅使用phase=current的正式评估。每个课程单独连线；灰色虚线是升级门槛，竖线是实际升级时刻。固定评估流10001被反复用于课程选择，不是最终盲测。更换种子但仍用同一城市，只能检验该城市上的出生随机性，不能替代新地图泛化评估。

## 2. 课程学习设计

| 级别 | 建筑数 | 逃逸机巡航/受威胁期望速度 m/s | 连续保持捕获半径 m |
|---|---:|---:|---:|
| 1 | 20 | 1 / 3 | 50 |
| 2 | 40 | 2 / 5 | 35 |
| 3 | 60 | 3 / 7 | 25 |
| 4 | 80 | 3 / 8 | 20 |
| 5 | 100 | 3 / 9 | 15 |

每级至少100,000环境转移。每50,000转移评估100回合，连续两次安全成功≥80%且碰撞≤10%才升级。最低训练量之前的通过次数不累积。直接拦截半径固定5米，连续保持要求固定2秒。

五级难度总体递增，适合建立baseline。建筑、速度、捕获半径同时变化，因此升级后的下降无法单独归因于某一项。已有成绩支持这种课程可以学会前三级任务；还不能证明它优于直接训练，也不能证明第五级已经学会，需有对照和最终复评。

## 3. 动力学：带约束的三维平动模型

| 参数 | 追捕机 | 逃逸机 |
|---|---:|---:|
| 最大速度 | 11 m/s | 课程规定3/5/7/8/9 m/s |
| 最大加速度范数 | 3 m/s² | 1 m/s² |
| 最大jerk范数 | 6 m/s³ | 8 m/s³ |
| 加速度响应滞后 | 0.5 s | 0.5 s |
| 物理步长 | 0.1 s | 0.1 s |
| 控制重算周期（008） | Actor每1 s | APF每0.1 s |

追捕机Actor输出三维加速度命令，经过范数限制、加速度响应滞后和jerk限制，再积分更新位置、速度。实际执行加速度不能瞬间跳到命令值。建筑采用棱柱，追捕机使用半径0.5米的接触检测；子步轨迹检查建筑、边界和捕获。训练建筑/边界碰撞会终止；评估继续统计完整回合，避免把碰撞后的捕获算成安全成功。

这足以研究加速、制动、绕楼或飞越楼顶、近距离追踪等能力。模型没有完整四旋翼姿态、推力与重力耦合、倾角限制、风扰或传感器误差。要检验实机可执行性，需要低层飞控跟踪和更真实的动力学验证。加速度上限3与1给追捕机明显机动优势；第五级最大速度9也小于追捕机11。

### 已接受的下一轮设置

用户已采纳：物理0.1秒＋决策0.5秒。008继续保持原来1秒决策，下一轮单独比较，当前没有改动008配置或启动第二个训练。

- Actor由1 Hz变为2 Hz，每次动作对应5个物理子步；逃逸机APF仍为10 Hz。
- 最长400秒保持不变，对应episode_length从400改为800。
- 保持每秒相同折扣衰减时，gamma由0.99改为sqrt(0.99)≈0.9949874371，环境势函数和learner使用相同gamma。
- 建筑/边界稠密惩罚已经按实际elapsed计时；progress不再额外乘0.5，成功与失败仍按事件发放一次，保持捕获仍按2秒计算。
- 默认制动反应时间是decision_dt＋acceleration_lag，因此由1.5秒变为1秒；这项应在下一轮配置与报告中明确记录。
- 比较要同时报告环境转移数与累计仿真秒数。3百万个0.5秒转移对应的名义仿真时长只有原1秒设置的一半；若按相同名义仿真时长比较，则是6百万对3百万，训练更新、噪声衰减和回放池对应的时间尺度也要说明。碰撞/捕获提前结束的实际elapsed还需要累计真实记录，不能只用步数乘dt代替实测。

## 4. 环境信息：理想状态反馈

地图1000×1000米，飞行高度50–350米。使用固定的20/40/60/80/100栋嵌套城市；普通/高层/地标建筑比例50%/40%/10%，宽度和高度沿用本轮已确定的尺寸。追捕机从城市西侧四个固定位置出发，逃逸机在东侧规定的自由空间盒中随机出生，每架真实初距至少450米。出生安全修正仍启用。

每个Actor的观测共79维，另拼接4维机号，用共享MLP生成动作：

| 信息 | 维数 | 当前来源 |
|---|---:|---|
| 自身位置、速度、加速度、保持进度 | 11 | 仿真真值 |
| 另外3机的相对位置与速度 | 18 | 仿真真值，无通信延迟 |
| 六个空域边界距离 | 6 | 精确几何 |
| 目标相对位置、速率、速度、加速度 | 10 | 任何距离均提供真值 |
| 课程编号、楼数、目标速度上限、捕获半径 | 4 | 已知任务参数 |
| 最近5栋楼的相对边界和有效标志 | 30 | 从全图几何选择，无遮挡/感知范围限制 |

所以600米外的追捕机也知道目标位置、速度和加速度。逃逸机的60米感知限制仅限制它自己的反应，不限制追捕机的目标观测。追捕机Actor运行时每次用更新后的观测重新决策，属于状态反馈策略；追捕机并没有使用逃逸机的APF控制器。

固定地图和单侧出生适合先学会任务，但包含绝对位置，可能同时学到特定地形与路线。已有实验不能分离通用追捕规律与城市记忆。最终盲测之外还需单独使用未训练城市、不同方位出生和目标运动模式验证。[泛化评估原始研究](https://arxiv.org/abs/1810.12282)也区分了训练分布内与分布外评估。

## 5. 逃逸机确实有APF，但威胁反应出现较晚

逃逸机每0.1秒重算控制。在追捕机进入60米以内时，追捕机排斥项生效并将期望速度切换到课程最大值；此前主要保持巡航。APF还包含建筑/边界排斥和接近空域边缘时的回中偏置。建筑接触后移动会被几何裁剪到可行空间，所以目标不会自由穿过建筑；这也意味着环境有接触后的人工纠正。

本快照中，各级最后100个完整训练回合的逃逸机统计如下。这里混合成功与失败，是训练采样统计，不能替代正式评估：

| 级别 | 平均速度 m/s | 最近追捕机在60米内的决策采样点比例 |
|---|---:|---:|
| 1 | 1.03 | 6.9% |
| 2 | 2.13 | 10.6% |
| 3 | 3.08 | 12.0% |
| 4 | 3.07 | 7.1% |

这不是连续时间感知占比，指标只在决策步末采样。结合代码可以判断：当前逃逸机大部分过程保持较低巡航速度，加速度1 m/s²又使加速较慢。它有逃逸反应，但还没有充分测试持续主动逃逸的对手。下一轮如需加强对手，建议把200米感知作为单独因素进行比较；它尚未实施，不能与0.5秒决策实验的结果混为同一个原因。

## 6. 是否学到了真实的追捕策略

**已经有证据支持：网络在固定城市、理想状态观测和当前APF对手下，能够学到安全接近并完成单机拦截/保持捕获的策略。** 前三级通过课程，第4级正在改善；第1级已有真实无碰撞视频，并经过原环境逐动作重放核验。

**目前仍缺少对四机协同包围的证据。** 成功判据是任意一架进入5米直接拦截，或任意同一架在课程捕获半径内保持2秒。没有要求其余三架形成包围、覆盖逃逸方向或共同压缩空间。progress主要鼓励各自接近，团队成功奖励允许单机完成任务。MATD3的集中式Critic和共享Actor提供协作学习能力，但任务判据决定是否必须协作；多智能体算法本身不能代替协同成功定义。[MADDPG原始论文](https://arxiv.org/abs/1706.02275)

例如：第1–3级末100回合的四机平均最终距离分别约196、172、187米；第1级视频由uav_3在50米内连续保持成功，其余机器仍更远。这符合当前已认可的baseline判据，说明当前成绩可以由单机接近达成。它不是“代码没有学习”的证据，也不是“形成几何包围”的证据。

如果“真实”是指实机环境中的追捕，当前还有感知、通信、动力学和未知环境差异，尚无实机迁移验证。如果“真实”是指基于反馈自主行动，当前确实是学习到的策略在闭环运行，但对新地图和强逃逸对手的有效性还需试验。

## 7. 后续顺序

1. 完成008原预算，对最终同一权重逐级复评，检查升级后的旧级能力保留情况。使用尚未用于008的60001种子；这仍是固定地图测试。保留本轮所有失败历史。
2. 对已训练到的级别制作真实无碰撞成功视频，保存模型哈希与重放验证；第1级复用现有视频，视频是筛选案例。
3. 下一轮先落实0.1秒物理＋0.5秒决策并对齐时间尺度，评估变化。
4. 单独加强逃逸感知/运动，再测试新出生位置、新城市和传感器扰动。
5. 若研究目标转为四机协同围捕，再共同确定包围/堵截成功定义和相应指标；当前单机baseline先完整保留用于对照。

## 证据与查看位置

- [进度与修改汇总]({link('docs/baseline_exp008_progress_2026-10-13.md')})
- [本次冻结观察记录]({AUDIT.as_posix()})
- [正式课程评估CSV]({link('outputs/results/matd3/baseline_exp008/curriculum_eval.csv')})
- [第1级已核验视频]({link('outputs/reviews/exp008_stage1_video_v1/level_1/success.mp4')})
- [环境观测与捕获代码]({link('xuance/environment/multi_agent_env/uav_pursuit_coverage_3d.py')})
- [动力学积分代码]({link('xuance/environment/multi_agent_env/uav_kinematics_3d.py')})
- [课程配置]({link('experiments/baseline_exp008.yaml')})

TensorBoard：[本机6006](http://127.0.0.1:6006/)。服务器日志根目录：`/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/logs/matd3/baseline_exp008/seed_1_2026_1013_183201/`。

本次传输包：`exp008_environment_audit_2026-10-13_2106_v2.zip`，SHA256 `ba7dd73e78cbedbc9c47ec5c7586a2885a4e25e72c9ccebc697ea17f382c3013`，已校验传输哈希、ZIP完整性和8项原始文件哈希。训练文件逐个复制，状态和CSV可能有少量时刻差异，未暂停训练。
"""
if followup:
    latest = followup["latest_current"][-1]
    s = followup["status"]
    extra = f"""## 21:14追加核查

服务器状态为{s['step']:,}转移，仍在第{s['level'] + 1}级，实际Critic更新{s['updates']:,}次、Actor更新{s['actor_updates']:,}次。第4级240万、245万、250万正式评估的安全成功率分别为49%、73%、62%，碰撞率分别为51%、27%、38%；尚未达升级门槛，表现有波动。

最新第4级250万评估：捕获率90%、安全成功率62%、建筑碰撞38%、边界碰撞17%、总碰撞38%，平均130.91决策步。250万第五级仍是跨级诊断，捕获50%、安全2%、碰撞98%，未进入第五级训练。

正文完整回合计数使用21:05冻结CSV；曲线已补入截至21:14的正式评估。追加记录见[followup_2114.json]({followup_path.as_posix()})。用户明确要求自动监控保持暂停、按需查看；服务器训练继续，之后按需核查并完成已授权的后续评估和交付。

"""
    text = text.replace("## 1. 最新训练证据", extra + "## 1. 21:05冻结训练证据")
output = ROOT / "docs/baseline_exp008_environment_assessment_2026-10-13.md"
output.write_text(text, encoding="utf-8")
print(json.dumps({"document": str(output), "curve": str(image_path), "step": status["step"]}, ensure_ascii=False))
