# 三维追捕 baseline

分支：`feat-coverage-reaward`。训练入口：`examples/run_baseline.py`；参数：`xuance/configs/maddpg/uav_pursuit_baseline_3d.yaml`。

## 1. 任务与终止条件

4 架追捕机，地图水平范围 1000 × 1000 m，允许飞行高度 50–350 m。动作是三个轴的加速度指令。决策间隔 1 s，物理积分间隔 0.1 s；每回合最多 400 个决策步。

沿用已经确认的成功条件，满足任意一条即可：

- 任意一架追捕机与逃逸机的欧氏距离不超过 5 m，直接拦截成功。
- 同一架追捕机连续 2 s 保持在当前课程的捕获半径内，持续捕获成功。离开范围后，这架飞机的计时归零。

成功时全队终止；超时为截断。建筑、边界和机间接触受罚，但当前配置下不直接终止回合。成功条件未增加多人包围或视线无遮挡要求。

## 2. 奖励

每架追捕机使用自己的距离和安全间距，成功奖励全队共享：

```
r_i = 300 × 本次成功
      - (d_i / 1000) × Δt
      - q(c_building, 30) × Δt
      - q(c_boundary, 30) × Δt
      - q(c_teammate, 10) × Δt
      + r_contact

q(c, L) = max(0, 1 - max(c, 0) / L)^2
```

`d_i` 是当前三维目标距离；`c_building` 是到最近建筑棱柱的三维最短距离减去飞机半径 0.5 m，包含楼顶距离；`c_boundary` 是到六个空域边界的最小允许间距；`c_teammate` 是最近队友中心距离减去接触阈值 1 m。以上距离单位均为 m，`Δt` 为本次实际推进的秒数，发生提前捕获时只累计已经推进的时间。

建筑接触事件扣 80，边界接触扣 100，机间接触扣 80；持续接触每秒再扣 20。每架飞机按接触种类计费，连续接触不会每个物理子步重复扣事件罚分；脱离后再次接触构成新事件。

例如远离障碍、边界和队友时，距离 200 m 的追捕机每秒获得 -0.2；移动到 100 m 时获得 -0.1。进入危险范围后额外受罚；距离建筑表面净间距 15 m 时，建筑项为 -0.25/s。成功项在真实成功终止时发放一次。

六项 TensorBoard 子奖励：`r_distance`、`r_obstacle`、`r_boundary`、`r_teammate`、`r_collision`、`r_terminal`。baseline 不计算覆盖、轨迹预测，也不使用准备度、闭合度、速度匹配、保持进度或额外时间/动作惩罚。继承配置中的 `w_cov` 等旧参数不会参与 baseline 奖励。

网络使用参数共享的 MADDPG：actor 和 critic 都为两层 512 单元 MLP，关闭 GAT 和图模块，保留每架飞机最近 5 栋建筑的几何观测。actor/critic 学习率分别为 1e-4/3e-4（编号实验 002 后调整），gamma=0.99，tau=0.001；正式配置使用 batch_size=1024，累计 1024 个环境转移后开始训练。

## 3. 五张固定地图与课程

| 显示等级 | 代码 level | 建筑数 | 逃逸机巡航/受威胁期望速度 m/s | 持续捕获半径 m |
|---|---:|---:|---:|---:|
| 1 | 0 | 20 | 3 / 5 | 30 |
| 2 | 1 | 40 | 3 / 6 | 25 |
| 3 | 2 | 60 | 3 / 7 | 20 |
| 4 | 3 | 80 | 3 / 8 | 18 |
| 5 | 4 | 100 | 3 / 9 | 15 |

追捕机最大速度固定为 11 m/s。逃逸机加速度上限固定为 1 m/s²；速度由动力学逐渐变化，表中的数值是期望值。

每张地图严格满足下列配额，长、宽分别按对应集合选取：

| 建筑类型 | 比例 | 长/宽 m | 高度 m |
|---|---:|---|---|
| 普通 | 50% | 20 或 40 | 10–60 |
| 高层 | 40% | 40 或 60 | 60–150 |
| 地标 | 10% | 60 或 80 | 150–350 |

建筑中心使用 10 × 10 个候选位置，间隔 100 m。每升级一次，在每行和每列各新增两栋；之前的建筑尺寸和位置保持一致。五级地面占用率分别为 3.68%、7.52%、11.60%、15.52%、19.20%，最小建筑平面净间距 20 m。50 m 以下的楼顶保留真实高度，追捕机可以直接从上方飞过。

地图文件提前保存到 `xuance/environment/multi_agent_env/buildings/buildings_uniform_*.json`。训练只加载这些文件，重置不生成新地图。`tools/generate_uniform_city_maps.py` 使用确定性规则复现文件，没有随机数；`uniform_map_manifest.json` 保存全部建筑的类别、尺寸和课程配额。

初始位置和朝向仍随种子采样，避免只记住一条轨迹。所有课程使用相同的出生规则：追捕机在目标周围约 200 m，垂直展开尺度 70 m；目标优先出生在地图中心附近。安全出生分配可能调整位置以避免建筑、边界和队友，因此实际初距不一定严格等于 200 m。

## 4. 升级规则

每个阶段至少训练 100,000 个累计环境转移；每 50,000 个环境转移进行一次无探索噪声评估，评估 100 回合。连续两次同时满足以下条件才升级：

- **安全成功率 ≥ 80%**：成功且整回合没有任何追捕机发生建筑、边界或机间接触。
- **回合碰撞率 ≤ 10%**：整回合任意追捕机发生过接触即计为碰撞回合。

一次未达标会清空连续通过次数。重复提交同一步数的评估不会累计两次通过。不按固定训练步数强制升级，不自动降级。升级请求在各训练环境下一次回合重置时生效，避免中途更换地图。

升级评估使用当前课程。每次评估同时测试固定第 5 级任务；`best_model.pth` 始终按第 5 级的安全成功率、较低碰撞率、平均回报排序选择，便于跨课程比较。各阶段另存 `best_stage_N.pth`。评估历史在 `curriculum_eval.csv`，课程转换和最终结果在 `baseline_summary.json`。

`curriculum_context_obs: true` 给 actor 和 critic 增加等级、建筑数、目标最大速度比和捕获半径四个归一化特征，帮助区分经验回放中的不同阶段。模型观测维度发生变化，旧覆盖模型的 checkpoint 不能直接作为该 baseline 的初始化。

## 5. 逃逸机与三维碰撞

逃逸机是人工势场控制器：在 `evader_sense` 范围内感知追捕机并转向逃离，同时避让建筑和空域边界。无威胁时保持当前航向；接近边缘时叠加回中偏置。baseline 开启 `evader_blend_center`，使回中偏置在存在排斥势场时也能作用于方向。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `evader_sense` | 60 m | 感知追捕机范围 |
| `evader_rep_coef` | 40 | 追捕机排斥权重 |
| `evader_obstacle_range` | 20 m | 建筑水平避让作用范围及楼顶附近的激活范围 |
| `evader_obstacle_coef` | 40 | 建筑排斥权重 |
| `evader_boundary_range` | 100 m | 边界排斥范围 |
| `evader_boundary_coef` | 50 | 边界排斥权重 |
| `evader_center_pull` | 1.5 | 回中偏置强度 |
| `evader_center_xy_range` | 150 m | 水平回中偏置激活范围 |
| `evader_center_z_range` | 75 m | 垂直回中偏置激活范围 |

扩大作用范围会更早反应；提高某一排斥权重会提高它相对于其他方向分量的影响。控制器将合成方向归一化，再按期望速度和加速度约束执行；因此权重不是速度倍数，也不能保证统一放大全部排斥权重会改变运动。

碰撞处理检查整个移动线段与半径扩展建筑棱柱的相交位置，停在首次接触前，留出 1 mm 余量。追捕机和逃逸机共用这个检查，允许绕行和安全飞越。它修复了下降穿过楼顶、一步跨过建筑以及恰好落在楼顶接触面的情况。发生接触仍记录碰撞指标和追捕机罚分。

## 6. 执行

服务器工作目录：`/home/ryy/UAV_Mine_baseline_coverage/xuance-master`。

```bash
cd /home/ryy/UAV_Mine_baseline_coverage/xuance-master
export PYTHONPATH="$PWD"

# 行为测试，包含实际多进程切图和短程 CPU 训练。
/home/ryy/miniconda3/envs/rl_env/bin/python -m pytest tests/test_uav_baseline.py -q

# 1024 步 GPU 集成测试；每回合 32 步、评估 2 回合、网络 64×64。
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/ryy/miniconda3/envs/rl_env/bin/python \
  examples/run_baseline.py --name baseline_smoke_01 --smoke --device cuda:0

# 正式训练，使用 512×512 网络、400 步回合和正式升级门槛。
/home/ryy/miniconda3/envs/rl_env/bin/python examples/run_baseline.py \
  --name baseline_run_01 --steps 10000000 --parallels 16 --device cuda:0
```

请为每次实验使用不同 `--name`。日志、模型、评估结果分别位于 `outputs/logs/maddpg/baseline_<name>`、`outputs/models/maddpg/baseline_<name>`、`outputs/results/maddpg/baseline_<name>`。

测试检查奖励数值、稀疏成功条件、楼顶与跨越碰撞、固定地图类别尺寸、势场参数、升级门槛、多进程切换、实际参数更新及有限的 loss/Q。runner 测试专门放宽升级门槛以执行真实升级路径，不使用这些门槛训练正式策略。1024 步 smoke 测试用于验证训练链路，不能证明追捕成功率或 loss 已收敛。

## 7. 本次测试结果（2026-10-12）

- `pytest tests/test_uav_baseline.py -q`：**27 passed，9.41 s**，包括实际 CPU 训练和课程升级。
- RTX 5080 上的 `smoke_20261012_coverage_v2`：**1024 个环境转移、496 次学习更新**，训练与评估流程退出码 0。
- actor 的 6 个参数张量、critic 的 6 个参数张量全部发生变化，模型参数均为有限值。
- 496 条 `loss_actor`、`loss_critic`、`predictQ` 记录全部有限；TensorBoard、评估 CSV、阶段最佳/最终任务最佳/最终训练模型均已保存。
- 正式升级门槛下 smoke 保持 level 0；该短程测试评估的成功率为 0，不能用于判断正式训练效果。
- `git diff --check` 通过；服务器原 checkout 仍在 `feat-traj-prediction`，保留原有 `examples/run_tensorboard.py` 改动。

具体指标和路径见 [测试报告](baseline_test_report_2026-10-12.json)。

## 8. 训练复查与编号对照（2026-10-13）

复查修复环境种子未生效、并行评估回合数超额和评估环境种子设置不完整的问题。当前行为测试为 **28 passed，10.33 s**。环境使用独立 RNG；每次评估重置 seed+10000 的种子流，默认 8 个并行环境严格统计 100 回合。

`--config experiments/baseline_expNNN.yaml` 可指定编号实验的冻结配置。运行开始即保存实际配置；每个训练区间刷新日志并保存 `latest_train_model.pth`；训练回合及每 1000 次更新的摘要分别在 `training_episodes.csv`、`training_updates.csv`，最新状态在 `training_status.json`。非有限 loss/Q 会触发错误。

编号实验与审查结果见 [训练复查](baseline_training_audit_2026-10-13.md)。`tools/analyze_numbered_baseline.py` 用于核验已完成实验的步数、评估回合数、奖励合计、实际配置差异、源代码一致性、TensorBoard 更新数量及数值，并输出比较 JSON、CSV 和曲线。

相同训练种子与 300000 步预算下，001（critic 学习率 0.001）的第一阶段最终捕获率/安全成功率/碰撞率为 21%/13%/87%，002（0.0003）为 38%/26%/53%。据此将默认 critic 学习率调整为 0.0003。002 第一阶段最佳安全成功率为 32%（250000 步），尚未达到课程升级门槛；两次实验的最后第 5 级安全成功率均为 0%，不能据此宣称五级课程已学会。详细对照见 [编号训练结果](baseline_training_results_2026-10-13/comparison.md)。
