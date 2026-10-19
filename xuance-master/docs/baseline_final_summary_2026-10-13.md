# Baseline 本轮训练结束总结

核验时间（北京时间）：2026-10-13T16:29:04+08:00。

本轮实验 007 已正常结束，完成 3,000,000 个新增环境转移。同一冻结最终模型通过两组新种子的第五级验收。

新种子 40001/50001 各 100 回合，合计安全捕获 94%、碰撞 6%；原门槛为安全 ≥80%、碰撞 ≤10%。最终任务仍为 100 栋建筑、逃逸机 3–9 m/s、保持捕获半径 15 m、最多 400 步。

最终模型的早期课程能力有所退化：第1/2级安全捕获为69%/76%；第4级碰撞15%也超过安全门槛。第五级通过验收不表示该模型在所有等级均达标。逐级复评与历史升级模型分别列出。

## A. 每轮实际执行步数

步数为全部并行环境合计的新增决策转移，非单个环境步数，也非梯度更新次数。

| 实验 | 算法 | 新增环境转移 | 已完成训练回合 | 安全训练捕获 | Critic 更新 | Actor 更新 | 最终课程 |
|---|---|---:|---:|---:|---:|---:|---:|
| exp001 | maddpg | 300,000 | 810 | 34 | 18,686 | 18,686 | 1 |
| exp002 | maddpg | 300,000 | 953 | 166 | 18,686 | 18,686 | 1 |
| exp003 | maddpg | 1,500,000 | 23,941 | 11,432 | 93,494 | 93,494 | 2 |
| exp004 | maddpg | 1,500,000 | 29,487 | 10,140 | 93,494 | 46,747 | 2 |
| exp005 | matd3 | 1,500,000 | 34,242 | 18,662 | 93,494 | 46,747 | 3 |
| exp006 | matd3 | 3,000,000 | 124,449 | 99,075 | 187,244 | 93,622 | 5 |
| exp007 | matd3 | 3,000,000 | 123,590 | 108,908 | 187,244 | 93,622 | 5 |

以上各轮包含对照和未成功的尝试，不能把总和作为某个最终模型的累计训练步数。接续模型的来源、保存时刻和课程资格见下方修改记录及模型来源文件。

## B. 每级实际训练步数

完整回合的步数由原始 `episode_steps` 按课程累加。全局预算减去已完成回合转移的差值，是结束时未完成回合的步数；全部进程在最后一次升级后均已超过一个最大回合长度，因此差值可归入最终课程。此处已将差值单列并核对各级之和等于每轮总步数。

升级按重置生效，部分环境会继续完成旧级别回合，因此准确步数与升级请求区间会略有差异。

| 实验 | 课程 | 完整回合转移 | 未完成回合转移 | 实际合计 | 完成回合数 |
|---|---:|---:|---:|---:|---:|
| exp001 | 1 | 296,771 | 3,229 | 300,000 | 810 |
| exp002 | 1 | 297,168 | 2,832 | 300,000 | 953 |
| exp003 | 1 | 300,976 | 0 | 300,976 | 9,703 |
| exp003 | 2 | 1,195,737 | 3,287 | 1,199,024 | 14,238 |
| exp004 | 1 | 351,043 | 0 | 351,043 | 10,711 |
| exp004 | 2 | 1,146,314 | 2,643 | 1,148,957 | 18,776 |
| exp005 | 1 | 402,055 | 0 | 402,055 | 15,874 |
| exp005 | 2 | 249,508 | 0 | 249,508 | 6,492 |
| exp005 | 3 | 847,071 | 1,366 | 848,437 | 11,876 |
| exp006 | 3 | 351,669 | 0 | 351,669 | 7,198 |
| exp006 | 4 | 2,048,517 | 0 | 2,048,517 | 92,146 |
| exp006 | 5 | 599,517 | 297 | 599,814 | 25,105 |
| exp007 | 5 | 2,999,768 | 232 | 3,000,000 | 123,590 |

## C. 当前冻结模型在每一级的效果

重新评估本轮 300 万步最终模型，以及按固定第五级测试选择的最佳保存模型。每个模型每级 100 回合；比较种子 10001，无探索、最多 400 步，碰撞不提前终止。这与历史各级升级模型的成绩是两组不同结果，不混用。

| 模型 | 课程 | 建筑数 | 捕获率 | 安全捕获率 | 任意机碰撞率 | 平均决策步数 |
|---|---:|---:|---:|---:|---:|---:|
| 300 万步最终模型 | 1 | 20 | 89% | 69% | 30% | 94.65 |
| 300 万步最终模型 | 2 | 40 | 95% | 76% | 24% | 55.13 |
| 300 万步最终模型 | 3 | 60 | 100% | 93% | 7% | 23.56 |
| 300 万步最终模型 | 4 | 80 | 98% | 85% | 15% | 33.24 |
| 300 万步最终模型 | 5 | 100 | 100% | 94% | 6% | 27.55 |
| 最佳第五级保存模型 | 1 | 20 | 72% | 66% | 34% | 133.02 |
| 最佳第五级保存模型 | 2 | 40 | 96% | 70% | 30% | 54.18 |
| 最佳第五级保存模型 | 3 | 60 | 100% | 88% | 12% | 23.23 |
| 最佳第五级保存模型 | 4 | 80 | 100% | 88% | 12% | 25.06 |
| 最佳第五级保存模型 | 5 | 100 | 100% | 94% | 6% | 26.04 |

[两份冻结模型的来源与哈希](</home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/reviews/exp007_all_levels_v1/provenance.json>)

### 最终模型的验证与全新种子验收

模型在查看 40001/50001 结果前已冻结；这两组种子未用于训练、升级门槛或选取该模型。20001/30001 已用于此前95万步模型的失败验收，此次仅作为验证种子。

| 用途 | 种子 | 回合数 | 捕获率 | 安全捕获率 | 任意机碰撞率 | 平均决策步数 | 达到门槛 |
|---|---:|---:|---:|---:|---:|---:|---|
| 已使用的验证种子 | 20001 | 100 | 97% | 93% | 7% | 38.42 | 是 |
| 已使用的验证种子 | 30001 | 100 | 100% | 96% | 4% | 25.83 | 是 |
| 全新验收种子 | 40001 | 100 | 98% | 93% | 7% | 33.21 | 是 |
| 全新验收种子 | 50001 | 100 | 100% | 95% | 5% | 30.30 | 是 |

最终模型 SHA256：`6c1f41773b47238f898d0f4243b092b8506d2f1ce3bba70e47cac3df0a7d5a32`。

[全新种子完整验收核验](</home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/acceptance/exp007_final_acceptance_v1/audit.json>)

## D. 代码、参数、奖励、课程和视频修改记录


训练状态快照：**2026-10-13 16:12:16 +0800**。当前实验 007 的训练进程实际存活：False。
本轮已完成新环境转移 3,000,000，实际训练第 5 级；最近整批评估在 3,000,000 步，安全捕获 94%、碰撞 6%。

## 1. 先区分课程等级与实验编号

第 1–5 级表示任务难度；001–007 表示训练方案版本。一个方案可以训练多个等级。同一方案进入下一级时主要改变课程参数，奖励和网络不必重写。

评估均关闭探索噪声，使用完整最多 400 个决策步的任务。安全捕获要求整个回合所有追捕机均没有建筑、边界或队友碰撞。碰撞率与安全捕获率未必相加为 100%，因为还有无碰撞超时。

### 初始 baseline 已落实的环境改造

- 固定保存五张均匀城市地图，分别 20/40/60/80/100 栋；1000×1000 m，建筑逐级增加。
- 建筑配比普通 50%、高层 40%、地标 10%；边长分别 20/40、40/60、60/80 m，楼顶高度 10–60、60–150、150–350 m。
- 建筑固定，出生位置和方向按环境种子随机；出生安全检查，逃逸机使用带回中/避障的人工势场。
- 四架追捕机，三维加速度动作；控制周期 1 s，物理子步 0.1 s，响应滞后及 jerk 限制。允许绕行或飞越建筑。
- 奖励仅保留捕获、距离与安全；关闭覆盖奖励、轨迹预测和图模块，采用共享 512×512 MLP。
- 修复环境独立随机数、评估种子和并行回合计数，记录奖励分项、真实 Actor/Critic 更新数、配置与源码快照。

## 2. 每轮改了什么，实际效果是什么

下表是各实验最后一次当前课程评估；不同课程难度不能直接作横向等难度比较。006 已正常结束，完成 3000000 个新转移。

| 实验 | 主要修改 | 最高实际训练级 | 最后当前级：安全 / 碰撞 | 最后第五级测试：安全 / 碰撞 |
|---|---|---:|---|---|
| 001 | 初始稀疏捕获 + 距离/安全稠密奖励，MADDPG | 1 | 13% / 87% | 0% / 100%（300,000 步） |
| 002 | 仅降低 Critic 学习率 | 1 | 26% / 53% | 0% / 100%（300,000 步） |
| 003 | 距离势函数塑形、全队失败惩罚、奖励缩放；调整探索、batch 与前四级课程 | 2 | 31% / 59% | 0% / 100%（1,500,000 步） |
| 004 | MADDPG 中 Actor 每两次 Critic 更新一次 | 2 | 7% / 87% | 0% / 100%（1,500,000 步） |
| 005 | 改用 MATD3：双 Critic、目标动作平滑、延迟 Actor 与目标网络更新 | 3 | 76% / 22% | 3% / 97%（1,500,000 步） |
| 006 | 完整策略权重接续第三级；制动风险代价、更低学习率/探索、缩小回放池 | 5 | 84% / 16% | 84% / 16%（3,000,000 步） |
| 007 | 接续已实际达到的第五级；制动距离越线后风险代价继续增长 | 5 | 94% / 6% | 94% / 6%（3,000,000 步） |

### 001 → 002：降低 Critic 学习率

Critic 学习率从 0.001 降到 0.0003，Actor 0.0001、batch 1024、512×512 网络及其余条件保持一致；各训练 300000 个转移。第一级末次安全捕获 13%→26%，碰撞 87%→53%，仍未能升级。这是一组相同训练种子的对照，尚无跨训练种子结论。

### 003：让接近目标与安全失败更明确

加入距离势函数塑形 `gamma*Phi(s_next)-Phi(s)`，`Phi=-distance/10`；真实捕获/失败终止时下一势函数为 0，时间截断仍保留势函数并 bootstrap。奖励整体乘 0.1；训练首次任意机碰撞，全队失败惩罚 -300 并结束回合。评估继续使用原完整回合规则。

同时把 Actor 学习率改为 0.00005、batch 改为 512、探索噪声 0.5→0.05（前 300000 转移衰减），降低前四级的速度、出生距离和捕获难度。最终第五级任务保持不变。

效果：300000 步首次进入第二级；第二级最好安全 72%、碰撞 28%，末次退步到 31%/59%。说明能学会最初接近，但第二级学习仍不稳定。该轮同时改了多项，不能把收益归因于单项。

### 004：仅延迟 Actor 更新

Actor 每两次 Critic 更新一次；仍为单 Critic MADDPG。350000 步进入第二级，但末次安全 7%、碰撞 87%。单独延迟 Actor 没有解决问题。实际更新次数写入日志，没有更新时不填写虚假的零 Actor loss。

### 005：采用 MATD3

双独立 Critic，TD 目标取较小 Q；目标动作加入标准差 0.2、截断 ±0.5 的平滑噪声；Actor 每两次更新，并用 Q1 梯度；目标网络只在 Actor 更新时软更新，tau=0.001。修复 MATD3 初始化并补齐目标动作平滑，使用原生策略推理。

沿用 003 的奖励、课程和学习率，从 002 同一 Actor 权重开始，Critic/优化器重新初始化。400000 步进入第二级、650000 步进入第三级；升级前模型第一级 100%/0%、第二级 98%/1%。1500000 步的第三级为 76%/22%，仍未通过升级；第五级末次仅 3%/97%。

### 006：在已有能力上学习提前制动避障

从 005 最后的第三级模型加载 Actor、双 Critic 及全部目标网络，共 36 个参数张量；优化器和回放池重新建立。前两级资格来自 005 的真实升级记录，006 从第三级接续，后续升级仍需原门槛。

建筑/边界代价采用有效间隙：`max(0, clearance - 1.5*v_closing - v_closing²/(2*max_accel))`。权重 3→5，让高速接近障碍时更早承担代价。这是奖励指标，没有改写动作或阻止真实碰撞。

| 参数 | 005 | 006 |
|---|---:|---:|
| 回放池 | 1000000 | 200000 |
| 探索起点 / 终点 | 0.5 / 0.05 | 0.1 / 0.03 |
| Actor 学习率 | 0.00005 | 0.00003 |
| Critic 学习率 | 0.0003 | 0.00015 |
| 学习率结束倍率 | 0.5 | 1.0 |
| 本轮训练预算 | 1500000 | 3000000 个新转移 |

效果：第三级在 300000/350000 新步数的评估为 87%/5%、100%/0%，实际进入第四级。第四级在 2350000/2400000 步的连续评估为 93%/7%、91%/9%，已按原门槛实际进入第五级。本轮也同时修改多项，尚不能作单项因果归因。

行为测试已随方案扩展到 38 项通过；006 的短 GPU 集成验证确认 496 次 Critic、248 次 Actor 更新、36 个有限权重张量及正确初始化。短验证仅证明训练链路可运行，性能结论来自整批任务评估。

### 006 的第四级碰撞诊断

冻结 1550000 步模型（该模型整批第四级安全 83%、碰撞 17%），另用种子 21002 做 64 回合首次碰撞诊断：59 次安全捕获、5 次碰撞、0 次超时。它采用训练的首次碰撞终止规则，不能代替完整任务的新种子验收。

5 次碰撞均涉及地标建筑 14 或 67，碰撞楼均是最近建筑观测中的第 1 栋。碰撞前剩余间隙约 1.1–7.4 m，同方向制动估计约 4.6–22.4 m；3 次仍朝建筑发出正加速度指令。两栋楼顶均为 350 m，达到飞行高度上限，不能靠升高飞越。该样本提示提前绕行/制动仍不足，不支持“障碍未进入观测”的解释。

在这些状态，006 的有效间隙已截到 0，建筑风险代价进入饱和区。006 最后第五级为 84%/16%，最佳为 88%/12%，碰撞仍超门槛。007 因此接续其最佳第五级模型，仅将有效间隙改为保留负值，使越线后的代价继续增长。42 项行为测试及短 GPU 集成通过，启动评估复现父模型 88%/12%；后续固定种子最佳达到 94%/6%，但后期仍有波动。冻结最佳模型的新初始条件复核见第 6 节，未通过验收。

[诊断核验](</home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/results/matd3/baseline_exp006/diagnostics/stage3_snapshot_20261013_0420/audit.json>) · [64 回合原始诊断](</home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/results/matd3/baseline_exp006/diagnostics/stage3_snapshot_20261013_0420/collisions_seed21002.json>)

### 课程学习曲线

![当前课程与原第五级任务评估](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/docs/baseline_change_history_progress_2026-10-13.png)

绿色/红色是当时正在训练的课程，蓝色虚线始终测原第五级任务。背景标记实际训练等级；因此跨等级的下降包含任务变难，不能单独认定网络能力退步。

## 3. 每一级如何变难，何时进入下一级

| 等级 | 建筑 | 逃逸机巡航 / 受威胁速度 m/s | 保持半径 m | 出生半径 m | 垂直展开 m |
|---|---:|---|---:|---:|---:|
| 1 | 20 | 1 / 3 | 50 | 100 | 30 |
| 2 | 40 | 2 / 5 | 35 | 125 | 40 |
| 3 | 60 | 3 / 7 | 25 | 150 | 50 |
| 4 | 80 | 3 / 8 | 20 | 175 | 60 |
| 5 | 100 | 3 / 9 | 15 | 200 | 70 |

此表对应 003–007。最初 001/002 使用更难的前几级：逃逸机巡航 3 m/s，最高 5/6/7/8/9 m/s，保持半径 30/25/20/18/15 m，出生半径均 200 m。因此 001/002 与后续第一级不属于等难度比较。

每级至少 100000 个转移；每 50000 步评估 100 个无探索回合；**连续两次安全捕获 ≥80% 且碰撞 ≤10%**才在下次重置时升级。不会按时间强制跳级。训练末尾最终任务测试也不等同于实际已经在第五级训练。

## 4. 各级成功视频

红色为逃逸机；蓝/绿/紫/橙为四架追捕机。左侧三维局部轨迹，右上全地图，右下真实三维距离。视频使用 0.1 s 真实物理子步采样、2×回放；没有改变 1 s 动作决策周期，没有使用动作修正或人工控制。

各级视频均在种子 23001 的首个回合成功。所有列出的成功视频都要求全回合所有追捕机零碰撞；每段动作在未加记录功能的原环境重放，逐步状态、奖励、终止和捕获结果一致。这里展示成功案例，不能用这些视频估算整体成功率。

| 等级 | 对应模型与训练步数 | 模型整批评估：安全 / 碰撞 | 样例捕获时间 | 视频 |
|---|---|---|---:|---|
| 1 | 005，400,000 步 | 100% / 0% | 8.33 s | [播放 MP4](</home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/videos/baseline_levels_1_4_20261013_v2/level_1/success.mp4>) |
| 2 | 005，650,000 步 | 98% / 1% | 10.92 s | [播放 MP4](</home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/videos/baseline_levels_1_4_20261013_v2/level_2/success.mp4>) |
| 3 | 006，350,000 步 | 100% / 0% | 14.01 s | [播放 MP4](</home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/videos/baseline_levels_1_4_20261013_v2/level_3/success.mp4>) |
| 4 | 006，2,350,000 步 | 93% / 7% | 17.07 s | [播放 MP4](</home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/videos/baseline_level_4_graduation_exp006_20261013_v1/level_4/success.mp4>) |
| 5 | 007，3,000,000 步 | 94% / 6% | 19.34 s | [播放 MP4](</home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/videos/baseline_level_5_exp007_final_20261013_v1/level_5/success.mp4>) |

视频模型已经单独冻结。早期第四级视频（006 的 1100000 步模型，76%/24%）仍保留在原目录；上表当前第四级视频对应 2,350,000 步模型。每个版本使用其自身的评估指标，不能把最新指标贴到旧模型视频上。

第 1/2 级样例由 P1 连续保持捕获，第 3/4 级样例由 P4 连续保持捕获；其余样例的执行机和捕获类型见各自视频。当前成功判据允许单机完成，所以画面中其余机器可能尚未靠近目标；它们不是“四架同时形成几何包围”的证明。覆盖和严格合围属于后续研究问题，当前 baseline 沿用已确定判据。

[打开各级视频汇总页](</home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/videos/baseline_levels_1_4_20261013_v2/video_gallery.html>) · [各级模型、种子与轨迹审计索引](</home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/videos/baseline_levels_1_4_20261013_v2/all_levels_video_index.json>) · [各级视频与轨迹哈希核验](</home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/videos/baseline_levels_1_4_20261013_v2/all_levels_video_audit.json>)

### 第五级

已在 2400000 步通过第四级门槛并实际进入第五级。最近第五级整批评估（3,000,000 步）：实验 007 最近整批安全捕获 94%、碰撞 6%。第五级成功视频已列入上表；每段注明其冻结模型的整批性能，单段成功不代表第五级达标。最终300万步模型已完成新种子复核，结果见第7节。

## 5. 后续记录方式

每次新方案保存修改原因、参数前后值、起始权重、配置/源码哈希、评估曲线及实际升级步数；每级成功视频保存冻结模型哈希、课程参数、种子、筛选尝试记录、完整轨迹与无碰撞核验。报告正文和实时训练状态注明各自快照时间，保留旧视频与旧实验归档。

训练采用同一训练种子；常规评估使用固定比较种子，新初始条件复核另外记录。这些结果不能声称跨训练种子的普遍效果或 loss 已收敛。

修改只在服务器独立工作树 `feat-coverage-reaward`；未提交或推送。实验记录放在 docs，不写入仅保留仓库约定的 handoff。

## 6. 95万步保存模型的首次新种子复核（历史失败结果）

冻结实验 007 的 950,000 步最佳模型；两次都使用同一模型哈希、100 个无探索完整回合、原 100 栋固定城市和最多 400 步任务，碰撞不提前终止。

| 新种子 | 回合数 | 安全捕获 | 碰撞 | 通过 |
|---|---:|---:|---:|---|
| 20001 | 100 | 84% | 16% | 否 |
| 30001 | 100 | 95% | 5% | 是 |

两批合计 200 回合：安全捕获 89.5%、碰撞 10.5%。种子 20001 的碰撞 16% 超门槛，合计碰撞 10.5% 也超过 10%；本次验收未通过。

20001/30001 已在此处使用；后续若依据它们调参或选模型，应作为验证种子，不再称为未使用的新种子。40001/50001的最终模型验收见第7节。

[查看冻结模型与完整任务核验](</home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/acceptance/exp007_stage5_candidate_v1/audit.json>)

## 7. 300万步最终模型的新种子验收

查看新种子结果前已冻结最终模型；40001/50001未用于训练、课程升级或该模型选择。每组100回合，沿用原完整400步任务、100栋城市、逃逸机3–9 m/s、15米保持半径。

| 新种子 | 回合数 | 安全捕获 | 碰撞 | 达到门槛 |
|---|---:|---:|---:|---|
| 40001 | 100 | 93% | 7% | 是 |
| 50001 | 100 | 95% | 5% | 是 |

合计200回合：安全捕获94%、碰撞6%。两批均达到原门槛，第五级本轮验收通过。

最终模型仍存在早期课程退化：同一模型第1–5级固定比较安全率69%/76%/93%/85%/94%，碰撞30%/24%/7%/15%/6%。历史各级优秀模型的成功视频用于展示当时能力；当前第五级视频已更新为同一300万步最终模型，未替换历史失败记录。

[最终模型完整任务验收核验](</home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/acceptance/exp007_final_acceptance_v1/audit.json>)


### 当前奖励函数（实验007）

对追捕机 i，实际奖励为下列八项之和，所有项统一乘0.1。距离 d 为到逃逸机的三维距离，Δt 为本次动作实际执行时长（通常1秒，提前捕获时更短）。

```text
r_i = 0.1 * [ -d_i/1000 * Δt + (0.99*Φ_i(next)-Φ_i(now))
              -5*max(0,1-c_building_eff/50)^2 * Δt
              -5*max(0,1-c_boundary_eff/50)^2 * Δt
              -max(0,1-c_teammate/10)^2 * Δt
              + collision_penalty_i + 300*I(capture_without_terminal_failure)
              -300*I(training_collision_failure) ]
Φ_i = -d_i/10
c_eff = clearance - (1.5*v_closing + v_closing^2/(2*3))
```

制动公式中的接近速度取非负部分。007保留负有效间隙，因此制动不足越严重、惩罚越大。队友间隙是两机距离减去碰撞距离后截到非负值。首次建筑/边界/队友接触的原始扣分为80/100/80，持续接触另扣20/秒。真实终止时下一势函数为0；400步截断保留下一势函数并bootstrap。训练中首次任意追捕机碰撞触发全队失败；评估关闭这项提前终止，统计整个回合碰撞。

评估中的捕获奖励不检查此前是否曾碰撞；整回合零碰撞的安全捕获率由独立指标计算。

捕获沿用单机5米直接拦截，或同一架在课程半径内连续保持2秒；它尚未要求四架同时构成几何合围。

### 主要源码入口

| 内容 | 文件 |
|---|---|
| 环境、奖励、人工势场、课程与碰撞 | [环境源码](</home/ryy/UAV_Mine_baseline_coverage/xuance-master/xuance/environment/multi_agent_env/uav_pursuit_coverage_3d.py>) |
| 最终实验的完整参数 | [实验007配置](</home/ryy/UAV_Mine_baseline_coverage/xuance-master/experiments/baseline_exp007.yaml>) |
| 完整任务冻结评估 | [评估器](</home/ryy/UAV_Mine_baseline_coverage/xuance-master/examples/evaluate_baseline_checkpoint.py>) |
| 视频记录与原环境重放核验 | [视频工具](</home/ryy/UAV_Mine_baseline_coverage/xuance-master/tools/render_baseline_success_videos.py>) |
| 训练、奖励与源快照审计 | [日志审计](</home/ryy/UAV_Mine_baseline_coverage/xuance-master/tools/analyze_level5_baseline.py>) |

## E. 核验证据与文件改动

七轮源码快照哈希、完整训练回合奖励分项之和、更新日志有限性、实际结束状态和逐级步数已核验。日志有限不代表 loss 收敛；固定城市上的不同初始条件测试不代表对新城市或多个训练种子的结论。

### 当前 Git 文件状态

```text
 M xuance/engine/run_marl.py
 M xuance/environment/multi_agent_env/uav_pursuit_coverage_3d.py
 M xuance/torch/agents/core/off_policy_marl.py
 M xuance/torch/learners/multi_agent_rl/maddpg_learner.py
 M xuance/torch/learners/multi_agent_rl/matd3_learner.py
 M xuance/torch/policies/deterministic_marl.py
?? docs/baseline_change_history_and_videos_2026-10-13.json
?? docs/baseline_change_history_and_videos_2026-10-13.md
?? docs/baseline_change_history_progress_2026-10-13.png
?? docs/baseline_exp003_exp004_final_2026-10-13/
?? docs/baseline_exp005_final_2026-10-13/
?? docs/baseline_exp006_final_2026-10-13/
?? docs/baseline_exp006_final_exp007_start_metadata.json
?? docs/baseline_exp006_live_2026-10-13/
?? docs/baseline_exp006_status_2026-10-13.json
?? docs/baseline_exp007_candidate_review_2026-10-13_metadata.json
?? docs/baseline_exp007_status_2026-10-13.json
?? docs/baseline_final_summary_2026-10-13.json
?? docs/baseline_final_summary_2026-10-13.md
?? docs/baseline_goal_level5_2026-10-13.md
?? docs/baseline_grade4_video_archive_metadata.json
?? docs/baseline_level5_results_2026-10-13/
?? docs/baseline_local_file_hashes.json
?? docs/baseline_local_file_hashes_2026-10-13.json
?? docs/baseline_test_report_2026-10-12.json
?? docs/baseline_training_audit_2026-10-13.md
?? docs/baseline_training_results_2026-10-13/
?? docs/uav_pursuit_baseline.md
?? examples/evaluate_baseline_checkpoint.py
?? examples/run_baseline.py
?? experiments/
?? tests/
?? tools/
?? xuance/configs/maddpg/uav_pursuit_baseline_3d.yaml
?? xuance/engine/uav_baseline_logging.py
?? xuance/engine/uav_curriculum.py
?? xuance/environment/multi_agent_env/buildings/buildings_uniform_100.json
?? xuance/environment/multi_agent_env/buildings/buildings_uniform_20.json
?? xuance/environment/multi_agent_env/buildings/buildings_uniform_40.json
?? xuance/environment/multi_agent_env/buildings/buildings_uniform_60.json
?? xuance/environment/multi_agent_env/buildings/buildings_uniform_80.json
?? xuance/environment/multi_agent_env/buildings/uniform_map_manifest.json
```

### 已跟踪文件改动统计

```text
 xuance-master/xuance/engine/run_marl.py            | 167 +++++++
 .../multi_agent_env/uav_pursuit_coverage_3d.py     | 517 ++++++++++++++-------
 .../xuance/torch/agents/core/off_policy_marl.py    |  24 +-
 .../learners/multi_agent_rl/maddpg_learner.py      |  44 +-
 .../torch/learners/multi_agent_rl/matd3_learner.py |  42 +-
 .../xuance/torch/policies/deterministic_marl.py    |   4 +
 6 files changed, 607 insertions(+), 191 deletions(-)
```
